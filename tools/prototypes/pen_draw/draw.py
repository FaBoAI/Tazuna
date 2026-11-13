#!/home/jetson/miniconda/envs/lerobot/bin/python
"""Draw a stroke file with the pen:  draw.py STROKES.json [--sim] [--z PLAN_Z] [--v MM_S] [--no-map] [--first N] [--last N]

STROKES.json: {"box_mm": [120, 120], "strokes": [[[u, v], ...], ...]}, millimetres, origin at the box centre, u to
the right and v up as the camera sees the sheet. The drawing is scaled into the drawing box (penlib.XC/YC/HALF),
turned with the sheet (ROT_DEG) and pre-distorted with the measured map (pen_state.json "map") so that the INK
lands where the drawing says. Each stroke: settled hover above its start, down, the stroke (split at sharp
corners), up. The arm starts from and returns to the side pose; torque stays ON.
"""
import argparse
import json
import math
import sys
import time

import penlib as P
from penlib import K, MotionAborted, SafetyError, np


def to_table(strokes, box_mm):
    s = 2.0 * P.HALF / float(max(box_mm))   # metres of table per millimetre of the drawing
    th = math.radians(P.ROT_DEG)
    u_hat = np.array([-math.sin(th), math.cos(th)])
    v_hat = np.array([-math.cos(th), -math.sin(th)])
    out = []
    for st in strokes:
        a = np.asarray(st, float)
        out.append(np.array([P.XC, P.YC]) + s * (a[:, :1] * u_hat + a[:, 1:2] * v_hat))
    return out


def inverse_map(pts, mp):
    """Commanded positions that make the ink land on pts (affine or quadratic map fitted by map_fit.py)."""
    if mp is None:
        return pts
    c0 = np.array(mp["centre"])
    coef = np.array(mp[mp["use"]]["coef"])
    quad = mp["use"] == "quad"

    def fwd(c):
        x, y = c[:, 0] - c0[0], c[:, 1] - c0[1]
        cols = [np.ones_like(x), x, y] + ([x * x, x * y, y * y] if quad else [])
        return np.stack(cols, axis=1) @ coef

    c = pts.copy()
    for _ in range(6):
        c = c + (pts - fwd(c))
    return c


def resample(p, step):
    """Polyline thinned to about `step` between points (corners kept)."""
    keep = [0]
    acc = 0.0
    for i in range(1, len(p) - 1):
        acc += float(np.linalg.norm(p[i] - p[i - 1]))
        a, b = p[i] - p[i - 1], p[i + 1] - p[i]
        turn = math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))))
        if acc >= step or turn > 25.0:
            keep.append(i)
            acc = 0.0
    keep.append(len(p) - 1)
    return p[keep]


def split_corners(p, deg=55.0):
    """Sub-polylines that end at corners sharper than deg (the arm stops there, so the corner stays sharp)."""
    parts, start = [], 0
    for i in range(1, len(p) - 1):
        a, b = p[i] - p[i - 1], p[i + 1] - p[i]
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        if na < 1e-9 or nb < 1e-9:
            continue
        if math.degrees(math.acos(max(-1.0, min(1.0, float(a @ b) / (na * nb))))) > deg:
            parts.append(p[start:i + 1])
            start = i
    parts.append(p[start:])
    return [q for q in parts if len(q) >= 2]


def pull_pieces(c, hyst=0.003, min_push=0.006):
    """Pieces of a commanded polyline, each to be drawn TOWARDS the robot (x falling) or sideways.

    The pen leans with its tip pointing away from the robot. Moved away from the robot the tip is pushed into the
    paper and chatters (2026-11-11: every line drawn that way came out dashed); moved towards the robot or sideways
    it trails and the line is whole. The polyline is cut where its x direction turns (with `hyst` of slack), and a
    piece that would travel more than `min_push` away from the robot is drawn from its other end. Neighbouring
    pieces overlap by one point so that the joins close."""
    x = c[:, 0]
    cuts, ext, direction = [0], 0, 0
    for i in range(1, len(c)):
        if direction >= 0 and x[i] > x[ext]:
            ext = i
        elif direction <= 0 and x[i] < x[ext]:
            ext = i
        if direction == 0:
            if abs(x[i] - x[0]) > hyst:
                direction = 1 if x[i] > x[0] else -1
                ext = i
        elif direction * (x[ext] - x[i]) > hyst:          # turned back by more than the slack: the extreme is a cut
            cuts.append(ext)
            direction, ext = -direction, i
    cuts.append(len(c) - 1)
    out = []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if b <= a:
            continue
        piece = c[max(0, a - 1):min(len(c), b + 2)]
        if piece[-1, 0] - piece[0, 0] > min_push:
            piece = piece[::-1]
        out.append(piece.copy())
    return out or [c]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("strokes")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--z", type=float, default=-0.0038, help="planned pen height while drawing (m)")
    ap.add_argument("--v", type=float, default=6.0, help="pen speed on the paper (mm/s)")
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--pull", action="store_true", help="cut the strokes so that the pen is never pushed away from the robot")
    ap.add_argument("--first", type=int, default=0)
    ap.add_argument("--last", type=int, default=None)
    ap.add_argument("--name", default="draw")
    a = ap.parse_args()
    with open(a.strokes) as fh:
        job = json.load(fh)
    st = P.load_state()
    mp = None if a.no_map else st.get("map")
    want = to_table(job["strokes"], job.get("box_mm", [120, 120]))
    cmds = [inverse_map(resample(w, 0.0012), mp) for w in want]
    if a.pull:
        cmds = [p for c in cmds for p in pull_pieces(c)]
    sel = list(range(len(cmds)))[a.first:a.last]
    total = sum(float(np.linalg.norm(np.diff(cmds[i], axis=0), axis=1).sum()) for i in sel)
    print("strokes %d (of %d), ink length %.0f mm, plan z %.1f mm, speed %.1f mm/s, map: %s" % (
        len(sel), len(cmds), total * 1000, a.z * 1000, a.v, "none" if mp is None else mp["use"]), flush=True)
    arm = P.make_arm(a.sim).open()
    t0 = time.time()
    done, log = 0, []
    try:
        if not arm.is_torqued():
            arm.torque_on()
        for i in sel:
            c = cmds[i]
            zs = a.z
            try:
                P.tip_path(arm, [(c[0][0], c[0][1], zs + P.LIFT)], 60.0, settle=True)       # settled hover above the start
                P.tip_path(arm, [(c[0][0], c[0][1], zs)], 12.0, settle=False)                # pen down
                time.sleep(0.0 if a.sim else 0.3)
                for part in split_corners(c):
                    P.tip_path(arm, [(p[0], p[1], zs) for p in part[1:]], a.v, settle=False, step=0.002)
                    time.sleep(0.0 if a.sim else 0.12)
                time.sleep(0.0 if a.sim else 0.15)
                P.tip_path(arm, [(c[-1][0], c[-1][1], zs + P.LIFT + 0.004)], 30.0, settle=0.2)   # pen up
            except (SafetyError, MotionAborted) as e:
                print(json.dumps({"stroke": i, "error": type(e).__name__, "message": str(e)[:300]}), flush=True)
                # get the pen off the paper before anything else
                q = arm.read_q()
                tip = K.fk(q, arm.tip_tool())[:3, 3]
                try:
                    P.tip_path(arm, [(tip[0], tip[1], max(tip[2], 0.0) + 0.02)], 30.0, settle=0.2)
                except (SafetyError, MotionAborted) as e2:
                    print(json.dumps({"lift_failed": str(e2)[:200]}), flush=True)
                    raise
                log.append({"stroke": i, "failed": str(e)[:200]})
                continue
            done += 1
            if done % 3 == 0 or i == sel[-1]:
                print("stroke %d/%d done (%.0f s)" % (done, len(sel), time.time() - t0), flush=True)
        P.go_aside(arm)
    finally:
        arm.close()
    res = {"ok": done == len(sel), "strokes_done": done, "strokes": len(sel), "failed": log, "seconds": round(time.time() - t0, 1)}
    if not a.sim:
        path, _ = P.snap_paper(a.name)
        res["photo"] = path
        P.save_state(last_draw={**res, "file": a.strokes, "z": a.z, "v": a.v})
    print(json.dumps(res))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
