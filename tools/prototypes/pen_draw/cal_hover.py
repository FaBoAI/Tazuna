#!/home/jetson/miniconda/envs/lerobot/bin/python
"""Refine the pen-tip point with the pen hovering just above the paper  (real arm unless --sim).

Near the table plane the camera model is at its best. The pen tip (as the current model has it) is held H above
the paper at a few places of the drawing box, photographed, found in the photo, and the tip point is refitted:
by default only its sideways part (y) and its length along the fingers (z); --full fits all three.
Nothing touches the paper. Writes config_pen.json; the arm ends at the side pose.
"""
import argparse
import json
import sys
import time

import penlib as P
from penlib import K, SafetyError, camera, cv2, np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--height", type=float, default=0.012)
    ap.add_argument("--full", action="store_true")
    a = ap.parse_args()
    cam = camera.CameraModel.load()
    arm = P.make_arm(a.sim).open()
    zt = float(arm.cfg["table_z"])
    p0 = np.array(arm.cfg["tip_offset"], float)
    s = P.HALF
    places = [("c", P.XC, P.YC), ("tl", P.XC - s, P.YC + s), ("tr", P.XC - s, P.YC - s), ("bl", P.XC + s, P.YC + s),
              ("br", P.XC + s, P.YC - s), ("ml", P.XC, P.YC + 0.085), ("mr", P.XC, P.YC - 0.085)]
    obs, log = [], []
    try:
        if not arm.is_torqued():
            arm.torque_on()
        for name, x, y in places:
            try:
                P.go_tip(arm, (x, y, zt + a.height + 0.02), P.PITCH, 0.0, 30.0, min_tip_z=0.02, settle=0.3)
                P.tip_path(arm, [(x, y, zt + a.height)], 20.0, settle=True)
            except SafetyError as e:
                print("skip %s: %s" % (name, str(e)[:150]), flush=True)
                continue
            time.sleep(0.0 if a.sim else 0.3)
            q = arm.read_q()
            tip = K.fk(q, arm.tip_tool())[:3, 3]
            anchor = cam.project([K.fk(q)[:3, 3]])[0]
            rec = {"name": name, "q_deg": np.degrees(q).round(3).tolist(), "model_tip": tip.round(4).tolist(),
                   "pred_uv": cam.project([tip])[0].round(1).tolist()}
            if not a.sim:
                path, img = P.snap_paper("hov_" + name)
                uv = P.find_pen_generic(img, anchor)
                rec.update(photo=path, uv=None if uv is None else [round(uv[0], 1), round(uv[1], 1)])
                if uv is not None:
                    obs.append((q, uv))
            print(json.dumps(rec), flush=True)
            log.append(rec)
            P.tip_path(arm, [(x, y, zt + a.height + 0.03)], 35.0, settle=0.1)     # straight up before the next move
        P.go_aside(arm)
    finally:
        arm.close()
    if a.sim:
        print("sim ok: %d places" % len(log))
        return 0
    if len(obs) < 3:
        print(json.dumps({"ok": False, "message": "pen tip found in only %d photos" % len(obs)}))
        return 1
    Ts = [K.fk(q) for q, _ in obs]
    uv = np.array([o[1] for o in obs], float)
    free = [0, 1, 2] if a.full else [1, 2]

    def res(v):
        p = p0.copy()
        p[free] = v
        return (cam.project([(T @ np.array([*p, 1.0]))[:3] for T in Ts]) - uv).ravel()

    v = p0[free].copy()
    for _ in range(40):
        r = res(v)
        J = np.zeros((len(r), len(free)))
        for k in range(len(free)):
            d = np.zeros(len(free))
            d[k] = 1e-5
            J[:, k] = (res(v + d) - r) / 1e-5
        step = np.linalg.lstsq(J, -r, rcond=None)[0]
        v = v + step
        if np.linalg.norm(step) < 1e-7:
            break
    p = p0.copy()
    p[free] = v
    e = np.hypot(*res(v).reshape(-1, 2).T)
    e0 = np.hypot(*res(p0[free]).reshape(-1, 2).T)
    out = {"ok": True, "tip_offset_before": p0.round(5).tolist(), "tip_offset": p.round(5).tolist(),
           "residual_px_before": e0.round(1).tolist(), "residual_px": e.round(1).tolist()}
    P.write_pen_config(p)
    P.save_state(tip_offset=p.round(5).tolist(), hover_fit=out, hover_obs=log)
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
