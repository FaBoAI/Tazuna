"""Tests of strokes.py (stroke planning) and simpaper.py (the simulated sheet, pen and camera).
No hardware: a simulated bus, the camera MODEL and the data of the pen evening only.
Run: python test_strokes.py [name ...]

What SPEC.md section 7 asks of the STROKES package, and where it is shown:
- pull invariant for random polylines and for the cat (push travel 275 mm -> at most 45 mm, about
  34 pieces, ink length within 3 %): test_pull_*
- corner splitting: test_split_corners_*, test_resample_*
- map inversion error under 0.05 mm: test_map_*
- layout stays inside sheet, reach and view: test_fit_box_*
- SimPaper renders a commanded square where the camera model says, within 1.5 px, and clips with
  the desk exposure: test_render_*, test_arm_square_*
- ink appears only where the true tip is at or below the paper: test_ink_only_*, test_hidden_*
"""
import ast
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["FD_CONFIG"] = os.path.join(HERE, "testdata", "pen_20261111", "config.json")
os.environ["FD_CAMERA_MODEL"] = os.path.join(HERE, "testdata", "pen_20261111", "camera_model.json")
# no tool file: these tests give the simulated arm its pen through cfg["tip_offset"] (tooling.py reads $FD_TOOL)
os.environ["FD_TOOL"] = os.path.join(HERE, "testdata", "pen_20261111", "no_such_tool_active.json")

import cv2  # noqa: E402
import numpy as np  # noqa: E402

# ---- nothing here may touch the arm or the camera (SPEC 0.1): fail loudly if anything tries
_TRIPPED = []


def _audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        p = os.fsdecode(args[0])
        if p.startswith("/dev/ttyACM") or p.startswith("/dev/video"):
            _TRIPPED.append((event, p))
            raise RuntimeError(f"a test opened {p}")
    elif event in ("subprocess.Popen", "os.system", "os.posix_spawn", "os.exec"):
        s = repr(args)
        if "v4l2-ctl" in s or "/dev/video" in s or "/dev/ttyACM" in s:
            _TRIPPED.append((event, s))
            raise RuntimeError(f"a test ran {s}")


sys.addaudithook(_audit)


def _no_capture(*a, **k):
    _TRIPPED.append(("cv2.VideoCapture", a))
    raise RuntimeError("a test opened the camera")


cv2.VideoCapture = _no_capture

import camera  # noqa: E402
import kinematics as K  # noqa: E402
import simpaper as SP  # noqa: E402
import strokes as S  # noqa: E402
from robot import ARM, Arm, MotionAborted, load_config  # noqa: E402
from sim_bus import SimBus  # noqa: E402

np.set_printoptions(precision=4, suppress=True)
DATA = os.path.join(HERE, "testdata", "pen_20261111")
with open(os.path.join(DATA, "manifest.json")) as _fh:
    MANIFEST = json.load(_fh)
CAM = camera.CameraModel.load()
TABLE_Z = MANIFEST["table_z"]
NIGHT = MANIFEST["drawing"]["layout"]                     # where the cat was drawn: XC, YC, HALF, ROT_DEG
NIGHT_MAP = MANIFEST["drawing"]["map_commanded_to_actual"]
PEN_TIP = [0.00153, -0.00809, 0.06211]                    # the marker's tip as calibrated that night
ASIDE_Q_DEG = [43.15, -20.31, 10.46, 63.28, -0.05]        # the side pose the photos were taken from
PITCH = 75.0
PLUS_X = S.constant_lean((1.0, 0.0))                      # the night's assumption
RADIAL = S.radial_lean()


class Skip(Exception):
    pass


def night_cat():
    return S.layout(S.load("cat"), (NIGHT["XC"], NIGHT["YC"]), NIGHT["HALF"], NIGHT["ROT_DEG"])


def dist_to_polyline(pts, poly):
    """Distance of every point to a polyline."""
    pts, poly = np.asarray(pts, float).reshape(-1, 2), np.asarray(poly, float).reshape(-1, 2)
    best = np.full(len(pts), np.inf)
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a
        t = np.clip(((pts - a) @ ab) / max(float(ab @ ab), 1e-300), 0.0, 1.0)
        best = np.minimum(best, np.linalg.norm(pts - (a + t[:, None] * ab), axis=1))
    return best


def random_polyline(rng, n=None):
    """A wandering line on the table: smooth stretches, sharp turns, loops, tiny wiggles."""
    n = int(rng.integers(2, 120)) if n is None else n
    p = [np.array([rng.uniform(0.15, 0.30), rng.uniform(-0.10, 0.10)])]
    heading = rng.uniform(0, 2 * math.pi)
    for _ in range(n - 1):
        kind = rng.random()
        heading += rng.normal(0, 0.25) if kind < 0.8 else rng.uniform(-math.pi, math.pi)
        step = rng.uniform(0.0002, 0.004) if kind < 0.95 else rng.uniform(0.004, 0.03)
        p.append(p[-1] + step * np.array([math.cos(heading), math.sin(heading)]))
    return np.array(p)


def swirl_lean(p):
    """A lean that turns with the place (no tool leans like this; the invariant must hold anyway)."""
    a = 40.0 * p[0] + 25.0 * p[1]
    return np.array([math.cos(a), math.sin(a)])


def pen_arm(q_deg=ASIDE_Q_DEG, tip=PEN_TIP):
    """A simulated arm that believes its tool point is the pen tip, torqued at the side pose."""
    cfg = load_config()
    cfg["tip_offset"] = list(tip)
    bus = SimBus(realtime=False)
    arm = Arm(bus=bus, cfg=cfg).open()
    raw = arm.q_to_raw_f(np.radians(q_deg))
    for j in ARM:
        bus.pos[j] = raw[j]
        bus.goal[j] = float(int(round(raw[j])))
        bus.sp[j] = raw[j]
    arm.torque_on()
    return arm, bus


def tip_path(arm, pts, v_mm_s=12.0, settle=False, step=0.002, pitch=PITCH, **kw):
    """Straight tip segments at about v_mm_s (as tools/pen_draw/penlib.tip_path did on the real arm)."""
    q0 = arm.read_q()
    p0 = K.fk(q0, arm.tip_tool())[:3, 3]
    wps = arm.plan_tip_path([(np.asarray(p, float), pitch, 0.0) for p in pts], q_start=q0, step_m=step)
    ln = float(sum(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float)) for a, b in zip([p0, *pts[:-1]], pts)))
    dq = float(np.degrees(np.abs(np.diff(np.array(wps), axis=0)).sum(axis=0).max()))
    speed = min(max(1.3 * dq * (v_mm_s / 1000.0) / max(ln, 1e-4), 2.0), 35.0)
    return arm.run_path(wps, speed, allow_contact=True, settle=settle, **kw)


def go_tip(arm, xyz, pitch=PITCH, speed=30.0):
    q = arm.read_q()
    return arm.run_path([q, arm.solve(np.asarray(xyz, float), pitch, 0.0, q_seed=q)], speed, settle=0.3)


def go_aside(arm):
    q = arm.read_q()
    return arm.run_path([q, np.radians(ASIDE_Q_DEG)], 30.0, settle=0.3)


def draw_polyline(arm, pts_xy, z, lift=0.02, v=12.0):
    """Hover, pen down, the line (stopping at every point), pen up."""
    go_tip(arm, (pts_xy[0][0], pts_xy[0][1], lift))
    tip_path(arm, [(pts_xy[0][0], pts_xy[0][1], z)], v)
    for p in pts_xy[1:]:
        tip_path(arm, [(p[0], p[1], z)], v)
    tip_path(arm, [(pts_xy[-1][0], pts_xy[-1][1], lift)], 30.0)


def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)


def line_offsets_px(img, paper, polyline_xy, skip_m=0.003, win=5):
    """How far the rendered line lies from where the camera model puts `polyline_xy` (on the paper's
    real surface): at points along it, the offset of the darkness centroid in a small window.
    Points within skip_m of a vertex are left out (two lines share the window there)."""
    dark = np.clip(float(np.median(gray(img)[paper._sheet_mask() > 0.999])) - gray(img), 0.0, None)
    pl = np.asarray(polyline_xy, float)
    offs = []
    for a, b in zip(pl[:-1], pl[1:]):
        ln = float(np.linalg.norm(b - a))
        for s in np.arange(skip_m, ln - skip_m + 1e-9, 0.001):
            p = a + (b - a) * (s / ln)
            u, v = paper._project(p[None, :])[0]
            iu, iv = int(round(u)), int(round(v))
            w = dark[iv - win:iv + win + 1, iu - win:iu + win + 1]
            if w.sum() < 30.0:
                offs.append(np.inf)                       # no ink where there should be some
                continue
            ys, xs = np.mgrid[iv - win:iv + win + 1, iu - win:iu + win + 1]
            offs.append(math.hypot(float((w * xs).sum() / w.sum()) - u, float((w * ys).sum() / w.sum()) - v))
    return np.array(offs)


def square(cx=0.24, cy=0.0, half=0.015):
    return np.array([[cx - half, cy - half], [cx - half, cy + half], [cx + half, cy + half], [cx + half, cy - half],
                     [cx - half, cy - half]])


def feed_line(paper, a, b, z_a, z_b=None, step=0.0001, lean=None):
    """Move the REAL tip straight from a to b (heights relative to the true surface)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = max(2, int(round(float(np.linalg.norm(b - a)) / step)) + 1)
    z_b = z_a if z_b is None else z_b
    for k in range(n):
        f = k / (n - 1)
        p = a + (b - a) * f
        paper.feed_tip((p[0], p[1], paper.surface_z(p[0], p[1]) + z_a + (z_b - z_a) * f), lean=lean)


# =========================================================================== strokes.py
def test_load_cat_by_name_and_path():
    cat = S.load("cat")
    assert cat["box_mm"] == [120.0, 120.0] and len(cat["strokes"]) == 18, (cat["box_mm"], len(cat["strokes"]))
    assert all(isinstance(s, np.ndarray) and s.ndim == 2 and s.shape[1] == 2 for s in cat["strokes"])
    assert "cat" in S.names()
    # the shipped drawing is the one of the evening, byte for byte
    with open(os.path.join(HERE, "drawings", "cat.json"), "rb") as a, open(os.path.join(DATA, "cat.json"), "rb") as b:
        assert a.read() == b.read()
    by_path = S.load(os.path.join(DATA, "cat.json"))
    assert all(np.array_equal(x, y) for x, y in zip(cat["strokes"], by_path["strokes"]))
    again = S.load({"box_mm": [120, 60], "strokes": [[[0, 0], [10, 5], [10, 5], [20, 0]]]})
    assert again["box_mm"] == [120.0, 60.0] and len(again["strokes"][0]) == 3      # the repeated point is dropped


def test_load_refuses_bad_drawings():
    bad = [{"strokes": [[[0, 0], [100, 0]]], "box_mm": [120, 120]},          # leaves the box
           {"strokes": [], "box_mm": [120, 120]},
           {"strokes": [[[0, 0], [float("nan"), 1]]]},
           {"strokes": [[[0, 0], [1, 1]]], "box_mm": [120, -1]},
           {"box_mm": [120, 120]},
           {"strokes": [[1, 2, 3]]}]
    for job in bad:
        try:
            S.load(job)
        except ValueError:
            continue
        raise AssertionError(f"accepted {job}")
    try:
        S.load("no_such_drawing")
    except ValueError as e:
        assert "cat" in str(e), e                      # the refusal names what exists
    else:
        raise AssertionError("an unknown name was accepted")


def test_layout_axes_turn_and_scale():
    d = {"box_mm": [120, 120], "strokes": [np.array([[0.0, 0.0], [60.0, 0.0]]), np.array([[0.0, 0.0], [0.0, 60.0]])]}
    right, up = S.layout(d, (0.25, 0.01), 0.05, 0.0)
    assert np.allclose(right, [[0.25, 0.01], [0.25, 0.06]])         # u (right in the photo) is +y
    assert np.allclose(up, [[0.25, 0.01], [0.20, 0.01]])            # v (up in the photo) is -x: towards the robot
    # in the photo: u to the right, v up
    uv = CAM.project([[0.25, 0.01, TABLE_Z], [0.25, 0.06, TABLE_Z], [0.20, 0.01, TABLE_Z]])
    assert uv[1, 0] - uv[0, 0] > 100 and abs(uv[1, 1] - uv[0, 1]) < 20, uv
    assert uv[0, 1] - uv[2, 1] > 60 and abs(uv[2, 0] - uv[0, 0]) < 20, uv
    # turned 6 degrees counter-clockwise seen from above, about the box centre
    right6, _ = S.layout(d, (0.25, 0.01), 0.05, 6.0)
    th = math.radians(6.0)
    assert np.allclose(right6[1], [0.25 - 0.05 * math.sin(th), 0.01 + 0.05 * math.cos(th)])
    # the longer side of a box that is not square becomes 2 * half
    wide = S.layout({"box_mm": [200, 100], "strokes": [np.array([[-100.0, -50.0], [100.0, 50.0]])]}, (0.25, 0.0), 0.05)
    assert np.allclose(wide[0], [[0.275, -0.05], [0.225, 0.05]]), wide
    # the same numbers as the script that drew the cat (tools/pen_draw/draw.py to_table)
    cat = S.load("cat")
    s = 2.0 * NIGHT["HALF"] / 120.0
    u_hat, v_hat = np.array([-math.sin(th), math.cos(th)]), np.array([-math.cos(th), -math.sin(th)])
    for mine, st in zip(night_cat(), cat["strokes"]):
        assert np.allclose(mine, np.array([NIGHT["XC"], NIGHT["YC"]]) + s * (st[:, :1] * u_hat + st[:, 1:2] * v_hat), atol=1e-12)
    corners = S.box_corners((0.25, 0.01), 0.05, 0.0)
    assert np.allclose(corners, [[0.20, -0.04], [0.20, 0.06], [0.30, 0.06], [0.30, -0.04]]), corners


def test_resample_spacing_and_corners():
    sq = square(half=0.02)
    r = S.resample(sq, 0.0012)
    d = np.linalg.norm(np.diff(r, axis=0), axis=1)
    assert d.max() <= 0.0012 + 1e-9 and d.min() > 0.0008, (d.min(), d.max())      # long segments get points
    for c in sq:
        assert np.min(np.linalg.norm(r - c, axis=1)) < 1e-12                         # the corners stay
    assert abs(S.length(r) - S.length(sq)) < 1e-9
    # a dense curve is thinned, stays on the curve and keeps its length
    t = np.linspace(0, 2 * math.pi, 2000)
    circle = np.stack([0.24 + 0.02 * np.cos(t), 0.02 * np.sin(t)], axis=1)
    rc = S.resample(circle, 0.0012)
    assert 100 <= len(rc) <= 110, len(rc)
    assert np.abs(np.linalg.norm(rc - [0.24, 0.0], axis=1) - 0.02).max() < 2e-5
    assert abs(S.length(rc) / S.length(circle) - 1.0) < 0.002
    assert np.allclose(rc[0], circle[0]) and np.allclose(rc[-1], circle[-1])
    # the cat keeps its length (the night's thinning lost none either: 877 mm laid out)
    want = night_cat()
    rs = [S.resample(w, 0.0012) for w in want]
    assert abs(sum(S.length(x) for x in rs) / sum(S.length(x) for x in want) - 1.0) < 0.003
    assert max(np.linalg.norm(np.diff(x, axis=0), axis=1).max() for x in rs) <= 0.0012 + 1e-9
    assert max(dist_to_polyline(x, w).max() for x, w in zip(rs, want)) < 1e-9       # every point ON the drawing
    assert len(S.resample(np.array([[0.2, 0.0]]), 0.001)) == 1


def test_split_corners():
    parts = S.split_corners(square())
    assert len(parts) == 4 and all(len(p) == 2 for p in parts), [len(p) for p in parts]
    assert all(np.array_equal(a[-1], b[0]) for a, b in zip(parts[:-1], parts[1:]))    # they share the corner
    parts = S.split_corners(S.resample(square(), 0.0012))
    assert len(parts) == 4 and abs(sum(S.length(p) for p in parts) - S.length(square())) < 1e-9
    t = np.linspace(0, 2 * math.pi, 200)
    circle = np.stack([0.24 + 0.02 * np.cos(t), 0.02 * np.sin(t)], axis=1)
    assert len(S.split_corners(circle)) == 1                                           # a curve is one motion
    bend = lambda deg: np.array([[0.2, 0.0], [0.22, 0.0], [0.22 + 0.02 * math.cos(math.radians(deg)), 0.02 * math.sin(math.radians(deg))]])
    assert len(S.split_corners(bend(40.0))) == 1 and len(S.split_corners(bend(70.0))) == 2
    assert len(S.split_corners(bend(40.0), deg=30.0)) == 2
    zig = np.array([[0.2, 0.0], [0.21, 0.01], [0.22, 0.0], [0.23, 0.01]])            # 90 degree zigzag
    assert [len(p) for p in S.split_corners(zig)] == [2, 2, 2]
    assert len(S.split_corners(np.array([[0.2, 0.0], [0.2, 0.0], [0.21, 0.0]]))) == 1  # a repeated point is no corner
    assert sum(len(S.split_corners(w)) for w in night_cat()) >= 18


def test_pull_invariant_random_polylines():
    rng = np.random.default_rng(20261112)
    worst, n_pieces, n_poly = -1.0, 0, 0
    for k in range(240):
        poly = S.resample(random_polyline(rng), 0.0012) if k % 3 else random_polyline(rng)   # raw ones too
        lean = (PLUS_X, RADIAL, swirl_lean, S.radial_lean(outward=False))[k % 4]
        hyst = (0.003, 0.003, 0.001, 0.0045, 0.0)[k % 5]
        min_push = (0.006, 0.006, 0.004, 0.006, 0.002, 0.010)[k % 6]
        overlap = (0.001, 0.001, 0.0, 0.002)[k % 4]
        merge = bool(k % 7 == 3)
        pieces = S.pull_pieces(poly, lean, hyst, min_push, overlap, merge=merge)
        n_poly, n_pieces = n_poly + 1, n_pieces + len(pieces)
        for pc in pieces:
            ex = S.push_excursion(pc, lean)
            worst = max(worst, ex - min_push)
            assert ex <= min_push + 1e-9, (k, ex, min_push, hyst, overlap, merge)        # THE invariant
            assert len(pc) >= 2
            assert dist_to_polyline(pc, poly).max() < 1e-9, k                            # pieces lie on the line
        # nothing of the line is lost, and little is drawn twice
        dense = np.concatenate([a + (b - a) * np.linspace(0, 1, 4)[:, None] for a, b in zip(poly[:-1], poly[1:])])
        gap = np.min([dist_to_polyline(dense, pc) for pc in pieces], axis=0)
        assert gap.max() < 1e-9, (k, gap.max())
        ink, cuts = sum(S.length(pc) for pc in pieces), len(pieces) - 1
        assert S.length(poly) - 1e-9 <= ink <= S.length(poly) + cuts * overlap + 1e-9, (k, ink, S.length(poly), cuts)
    assert n_pieces > n_poly                       # the test really cut something
    print(f"    {n_poly} polylines -> {n_pieces} pieces; closest to the limit: {worst * 1000:+.3f} mm (never above 0)")


def test_pull_directions_simple_cases():
    away = np.array([[0.20, 0.0], [0.26, 0.0]])                  # straight away from the robot: a push
    for lean in (PLUS_X, RADIAL):
        (pc,) = S.pull_pieces(S.resample(away, 0.001), lean)
        assert np.allclose(pc[0], away[1]) and np.allclose(pc[-1], away[0])          # drawn from the far end
        (pc,) = S.pull_pieces(S.resample(away[::-1], 0.001), lean)
        assert np.allclose(pc[0], away[1]) and np.allclose(pc[-1], away[0])          # a pull stays as it is
    side = np.array([[0.24, -0.03], [0.24, 0.03]])
    (pc,) = S.pull_pieces(S.resample(side, 0.001), PLUS_X)
    assert np.allclose(pc[0], side[0])                                               # sideways: untouched
    # with the radial lean a "sideways" line across the robot's front is pulled in, then pushed out
    wide = S.resample(np.array([[0.20, -0.12], [0.20, 0.12]]), 0.001)
    pcs = S.pull_pieces(wide, RADIAL)
    assert len(pcs) == 2 and all(S.push_excursion(p, RADIAL) <= 0.006 for p in pcs), len(pcs)
    assert all(abs(p[-1][1]) < 0.002 for p in pcs)                                   # both halves end in the middle
    # a V away and back: two pieces that both start at the far point and are pulled in, sharing 1 mm there
    v = S.resample(np.array([[0.20, -0.02], [0.25, 0.0], [0.20, 0.02]]), 0.001)
    a, b = S.pull_pieces(v, PLUS_X)
    assert a[-1][0] < 0.201 and b[-1][0] < 0.201 and a[0][0] > 0.249 and b[0][0] > 0.249
    assert abs((S.length(a) + S.length(b)) - (S.length(v) + 0.001)) < 1e-9
    # a tool that stands upright has no push side: nothing is cut
    assert len(S.pull_pieces(v, S.constant_lean((0.0, 0.0)))) == 1
    assert len(S.pull_pieces(np.array([[0.2, 0.0]]), PLUS_X)) == 1


def test_pull_cat_numbers():
    """SPEC 7: push travel 275 mm -> at most 45 mm, about 34 pieces, ink length within 3 %."""
    want = night_cat()
    total = sum(S.length(w) for w in want)
    assert abs(total * 1000 - 877.7) < 1.0, total
    before = sum(S.push_travel(w, PLUS_X) for w in want) * 1000
    assert abs(before - 275.0) < 2.0, before                    # as it was drawn on 2026-11-11
    out = {}
    for name, lean in (("+x", PLUS_X), ("radial", RADIAL)):
        pcs = [p for w in want for p in S.pull_pieces(S.resample(w, 0.0012), lean)]
        st = S.stats(pcs, lean)
        out[name] = st
        assert st["push_m"] * 1000 <= 45.0, st
        assert st["push_max_m"] <= 0.006 + 1e-9, st
        assert 30 <= st["count"] <= 38, st
        assert abs(st["ink_m"] / total - 1.0) < 0.03, (st["ink_m"], total)
        assert max(float(np.min([dist_to_polyline(p, w) for w in want], axis=0).max()) for p in pcs) < 1e-4
    assert out["+x"]["count"] == 34, out["+x"]
    before_r = sum(S.push_travel(w, RADIAL) for w in want) * 1000
    print(f"    cat: push {before:.1f} -> {out['+x']['push_m'] * 1000:.1f} mm (+x), {before_r:.1f} -> "
          f"{out['radial']['push_m'] * 1000:.1f} mm (radial); {out['+x']['count']} / {out['radial']['count']} pieces; "
          f"ink {total * 1000:.1f} -> {out['+x']['ink_m'] * 1000:.1f} mm")
    # merge=True: fewer pen lifts within the same invariant
    merged = [p for w in want for p in S.pull_pieces(S.resample(w, 0.0012), RADIAL, merge=True)]
    sm = S.stats(merged, RADIAL)
    assert sm["count"] < out["radial"]["count"] and sm["push_max_m"] <= 0.006 + 1e-9 and sm["push_m"] * 1000 <= 60.0, sm


def test_stats():
    a = np.array([[0.20, 0.0], [0.23, 0.0]])
    b = np.array([[0.23, 0.01], [0.20, 0.01]])
    st = S.stats([a, b], PLUS_X)
    assert st["count"] == 2 and st["points"] == 4
    assert abs(st["ink_m"] - 0.06) < 1e-12 and abs(st["penup_m"] - 0.01) < 1e-12
    assert abs(st["push_m"] - 0.03) < 1e-12 and abs(st["push_max_m"] - 0.03) < 1e-12
    st = S.stats([a, b])
    assert st["push_m"] is None and st["push_max_m"] is None
    assert S.stats([], PLUS_X)["count"] == 0 and S.stats([], PLUS_X)["push_max_m"] == 0.0
    wig = np.array([[0.2, 0.0], [0.202, 0.001], [0.2, 0.002], [0.202, 0.003]])
    assert abs(S.push_travel(wig, PLUS_X) - 0.004) < 1e-12 and abs(S.push_excursion(wig, PLUS_X) - 0.002) < 1e-12


def test_map_inversion_under_005_mm():
    rng = np.random.default_rng(5)
    pts = np.column_stack([rng.uniform(0.17, 0.33, 500), rng.uniform(-0.09, 0.09, 500)])
    shear10 = {"centre": [0.2575, 0.003], "kind": "affine", "coef": [[0.2575 + 0.004, 0.003 - 0.002], [1.0, 0.0], [-0.10, 1.0]]}
    tooling_form = {"centre": NIGHT_MAP["centre"], "kind": "quad", "coef": NIGHT_MAP["quad"]["coef"], "rms_mm": 0.94, "n": 9}
    worst = {}
    for name, mp in (("night affine", NIGHT_MAP), ("night quad", {**NIGHT_MAP, "use": "quad"}), ("shear 10 %", shear10),
                     ("tooling form", tooling_form)):
        cmd = S.invert_map(pts, mp)
        err = np.linalg.norm(S.apply_map(cmd, mp) - pts, axis=1).max() * 1000
        worst[name] = err
        assert err < 0.05, (name, err)
        assert np.linalg.norm(cmd - pts, axis=1).max() > 0.003                  # the map really moves things
    print("    inversion error (mm): " + ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))
    # the same answer as the night's fixed-point iteration (draw.py inverse_map), which drew the cat
    c0, coef = np.array(NIGHT_MAP["centre"]), np.array(NIGHT_MAP["affine"]["coef"])
    c = pts.copy()
    for _ in range(12):
        c = c + (pts - (np.column_stack([np.ones(len(c)), c - c0]) @ coef))
    assert np.abs(c - S.invert_map(pts, NIGHT_MAP)).max() < 1e-9
    # the grid marks of the night: commanded -> where the ink was found, within the fit's residual
    g = MANIFEST["grid_pitch75"]
    res = np.linalg.norm(S.apply_map(g["commanded_xy"], NIGHT_MAP) - np.array(g["ink_xy"]), axis=1) * 1000
    assert res.max() < 3.4 and math.sqrt(np.mean(res ** 2)) < 2.1, res
    assert np.array_equal(S.apply_map(pts, None), pts) and np.array_equal(S.invert_map(pts, None), pts)
    for bad in ({"centre": [0, 0], "kind": "affine", "coef": [[0, 0], [1, 0], [2, 0]]},       # singular
                {"centre": [0, 0], "kind": "affine", "coef": [[0, 0], [1, 0]]},               # wrong shape
                {"centre": [0, 0], "coef": [[0, 0], [1, 0], [0, 1]]}):                        # no kind
        try:
            S.invert_map(pts, bad)
        except ValueError:
            continue
        raise AssertionError(f"accepted {bad}")


def test_fit_map_reproduces_the_night():
    g = MANIFEST["grid_pitch75"]
    mp = S.fit_map(g["commanded_xy"], g["ink_xy"], centre=NIGHT_MAP["centre"])
    assert mp["kind"] == "affine" and mp["n"] == 9, mp                     # the choice of that night
    assert np.abs(np.array(mp["coef"]) - np.array(NIGHT_MAP["affine"]["coef"])).max() < 2e-3, mp["coef"]
    assert abs(mp["rms_mm"] - NIGHT_MAP["affine"]["rms_mm"]) < 0.1 and abs(mp["max_mm"] - NIGHT_MAP["affine"]["max_mm"]) < 0.15, mp
    assert abs(mp["fits"]["quad"]["rms_mm"] - NIGHT_MAP["quad"]["rms_mm"]) < 0.1, mp["fits"]
    # usable as it comes out
    pts = np.array(g["ink_xy"])
    assert np.linalg.norm(S.apply_map(S.invert_map(pts, mp), mp) - pts, axis=1).max() < 5e-5
    # an exact quadratic world with many marks is recognised; a noisy affine one is not over-fitted
    rng = np.random.default_rng(3)
    cmd = np.column_stack([rng.uniform(0.18, 0.32, 16), rng.uniform(-0.08, 0.08, 16)])
    quad_world = {**NIGHT_MAP, "use": "quad", "quad": {"coef": (np.array(NIGHT_MAP["quad"]["coef"]) * [[1], [1], [1], [8], [8], [8]]).tolist()}}
    mq = S.fit_map(cmd, S.apply_map(cmd, quad_world), centre=NIGHT_MAP["centre"])
    assert mq["kind"] == "quad" and mq["rms_mm"] < 0.01, mq
    ma = S.fit_map(cmd, S.apply_map(cmd, NIGHT_MAP) + rng.normal(0, 0.001, cmd.shape))
    assert ma["kind"] == "affine", ma
    assert S.fit_map(cmd[:5], S.apply_map(cmd[:5], NIGHT_MAP))["kind"] == "affine"
    for bad_cmd in (cmd[:2], np.column_stack([np.linspace(0.2, 0.3, 5), np.zeros(5)])):       # too few / on one line
        try:
            S.fit_map(bad_cmd, bad_cmd)
        except ValueError:
            continue
        raise AssertionError("fitted a map that the marks do not fix")


def _reach_and_view():
    arm, _ = pen_arm()
    tool, lim = arm.tip_tool(), arm.q_limits()
    calls = {"n": 0}

    def reach(x, y):
        calls["n"] += 1
        return all(K.ik((x, y, z), PITCH, 0.0, tool=tool, limits=lim)[1]["ok"] for z in (TABLE_Z - 0.004, TABLE_Z + 0.012))

    def view(x, y):
        u, v = CAM.project([[x, y, TABLE_Z]])[0]
        return 30 <= u <= 1890 and 30 <= v <= 1050

    return reach, view, calls


def test_fit_box_inside_sheet_reach_and_view():
    reach, view, calls = _reach_and_view()
    sheet = SP.sheet_polygon()
    t0 = time.time()
    r = S.fit_box(0.0525, sheet, reach, view, margin_m=0.01)
    dt = time.time() - t0
    assert r["ok"] and r["half_m"] <= 0.0525 + 1e-12, r
    assert abs(r["rot_deg"] - 6.0) < 0.01, r                                 # follows the sheet
    assert r["half_m"] >= 0.045, r                                           # the night's box (105 mm) nearly or fully fits
    cat = S.load("cat")
    for want_half in (0.0525, 0.03, 0.2):                                    # as wanted, small, far too big
        r = S.fit_box(want_half, sheet, reach, view, margin_m=0.01)
        assert r["ok"] and r["half_m"] <= want_half + 1e-12, r
        pts = np.concatenate(S.layout(cat, r["centre"], r["half_m"], r["rot_deg"]) + [np.array(r["corners"])])
        assert np.all(S._in_polygon(pts, sheet)) and S._edge_distance(pts, sheet).min() >= 0.01 - 1e-6, r
        assert all(reach(x, y) for x, y in pts[::3]) and all(view(x, y) for x, y in pts[::3]), r
        if want_half == 0.2:
            assert r["half_m"] < 0.2 and r["limited_by"] and r["reason"], r
        if want_half == 0.03:
            assert r["half_m"] == 0.03 and not r["limited_by"] and r["reason"] == "", r
            assert np.linalg.norm(np.array(r["centre"]) - sheet.mean(axis=0)) < 0.03, r     # in the middle when there is room
    print(f"    fit_box: {dt:.2f} s, {calls['n']} reach tests; too-big request -> half {r['half_m'] * 1000:.0f} mm, {r['reason']}")
    # prefer_xy moves the box when the size leaves a choice
    r2 = S.fit_box(0.03, sheet, reach, view, prefer_xy=(0.27, 0.05))
    assert r2["ok"] and np.linalg.norm(np.array(r2["centre"]) - [0.27, 0.05]) < 0.02, r2
    # a box that is not square
    r3 = S.fit_box(0.06, sheet, reach, view, box_mm=[120, 60])
    c3 = np.array(r3["corners"])
    assert r3["ok"] and np.all(S._in_polygon(c3, sheet)) and S._edge_distance(c3, sheet).min() >= 0.01 - 1e-6, r3
    assert abs(np.linalg.norm(c3[1] - c3[0]) / np.linalg.norm(c3[2] - c3[1]) - 2.0) < 1e-6, c3


def test_fit_box_limits_and_refusals():
    reach, view, _ = _reach_and_view()
    sheet = SP.sheet_polygon()
    # reach ends at x = 0.25: the box stays on this side, smaller if it must
    near = lambda x, y: x <= 0.25 and reach(x, y)
    r = S.fit_box(0.0525, sheet, near, view)
    assert r["ok"] and np.array(r["corners"])[:, 0].max() <= 0.25 + 1e-9, r
    # only the right half of the photo is in view
    right = lambda x, y: view(x, y) and CAM.project([[x, y, TABLE_Z]])[0][0] >= 960
    r = S.fit_box(0.0525, sheet, reach, right)
    assert r["ok"] and all(right(x, y) for x, y in r["corners"]), r
    # refusals say why
    r = S.fit_box(0.05, SP.sheet_polygon(size_m=(0.03, 0.03)), reach, view)
    assert not r["ok"] and r["centre"] is None and r["half_m"] == 0.0 and "sheet" in r["reason"], r
    r = S.fit_box(0.05, sheet, lambda x, y: False, view)
    assert not r["ok"] and "reach" in r["reason"] and r["limited_by"] == ["reach"], r
    r = S.fit_box(0.05, sheet, reach, lambda x, y: False)
    assert not r["ok"] and "view" in r["reason"], r
    # rot_deg given: not the sheet's; no sheet: the region in front of the robot, no turn
    r = S.fit_box(0.04, sheet, reach, view, rot_deg=0.0)
    assert r["ok"] and r["rot_deg"] == 0.0 and np.all(S._in_polygon(np.array(r["corners"]), sheet)), r
    disc = lambda x, y: math.hypot(x - 0.24, y) <= 0.05
    r = S.fit_box(0.05, None, disc, None)
    assert r["ok"] and r["rot_deg"] == 0.0 and all(disc(x, y) for x, y in r["corners"]), r
    assert 0.03 <= r["half_m"] <= 0.05 / math.sqrt(2) + 1e-9, r
    assert abs(S.sheet_rot_deg(SP.sheet_polygon(rot_deg=-12.0)) + 12.0) < 1e-6
    assert abs(S.sheet_rot_deg(SP.sheet_polygon(rot_deg=6.0)[[2, 3, 0, 1]]) - 6.0) < 1e-6      # whichever corner comes first
    try:
        S.fit_box(-1.0, sheet)
    except ValueError:
        pass
    else:
        raise AssertionError("a negative size was accepted")


def test_plan_pieces_pipeline():
    want = night_cat()
    plan = S.plan_pieces(want, RADIAL, NIGHT_MAP)
    assert plan["strokes"] == 18 and plan["stats"]["count"] == len(plan["pieces"]) == 34, plan["stats"]
    for pc in plan["pieces"]:
        assert np.linalg.norm(S.apply_map(pc["cmd"], NIGHT_MAP) - pc["ink"], axis=1).max() < 5e-5     # the ink lands on the plan
        assert S.push_excursion(pc["ink"], RADIAL) <= 0.006 + 1e-9
        assert np.array_equal(np.concatenate([pc["parts"][0]] + [p[1:] for p in pc["parts"][1:]]), pc["cmd"])
    assert [pc["stroke"] for pc in plan["pieces"]] == sorted(pc["stroke"] for pc in plan["pieces"])
    # what the night's script sent, for comparison: 862 mm commanded (manifest: ink_length_mm)
    cmd_len = sum(S.length(S.invert_map(S.resample(w, 0.0012), NIGHT_MAP)) for w in want) * 1000
    assert abs(cmd_len - MANIFEST["drawing"]["ink_length_mm"]) < 2.0, cmd_len
    # order=True: the same pieces, less travel in the air
    ordered = S.plan_pieces(want, RADIAL, NIGHT_MAP, order=True)
    assert sorted(id_ for id_ in map(lambda q: q["ink"].tobytes(), ordered["pieces"])) == sorted(q["ink"].tobytes() for q in plan["pieces"])
    assert ordered["stats"]["penup_m"] < plan["stats"]["penup_m"], (ordered["stats"]["penup_m"], plan["stats"]["penup_m"])
    assert ordered["stats"]["ink_m"] == plan["stats"]["ink_m"] and ordered["stats"]["push_max_m"] == plan["stats"]["push_max_m"]
    idx = S.order_pieces([q["ink"] for q in plan["pieces"]], start_xy=(0.215, -0.15))
    assert sorted(idx) == list(range(34)) and S.order_pieces([]) == []
    print(f"    travel in the air: {plan['stats']['penup_m'] * 1000:.0f} mm in drawing order, {ordered['stats']['penup_m'] * 1000:.0f} mm ordered")
    plain = S.plan_pieces(want)                                    # no lean, no map: the strokes as they are
    assert plain["stats"]["count"] == 18 and plain["stats"]["push_m"] is None
    assert all(np.array_equal(pc["cmd"], pc["ink"]) for pc in plain["pieces"])


def test_strokes_is_pure():
    """strokes.py must not import robot, camera or fd (SPEC, package notes): numpy and the standard library only."""
    with open(os.path.join(HERE, "strokes.py")) as fh:
        tree = ast.parse(fh.read())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            mods.add((node.module or "").split(".")[0])
    assert mods <= {"__future__", "json", "math", "os", "numpy"}, mods


# =========================================================================== simpaper.py
def test_render_levels_and_clipping():
    paper = SP.SimPaper(cam=CAM)
    for w in night_cat():
        paper.add_ink(w, old=False)
    on_sheet = paper._sheet_mask() > 0.999
    img = paper.render("paper")
    assert img.shape == (1080, 1920, 3) and img.dtype == np.uint8
    g = gray(img)
    assert 165 <= np.median(g[on_sheet]) <= 180, np.median(g[on_sheet])              # paper about 175
    assert float((img[on_sheet].max(axis=1) >= 250).mean()) == 0.0                     # nothing clipped
    assert g[on_sheet].min() < 60                                                      # ink is dark
    off = paper._sheet_mask() < 0.001
    assert np.median(g[off]) < 40 and float(off.mean()) > 0.4                          # dark desk around the sheet
    # the locked "desk" exposure: the paper is clipped to 255, the desk is well exposed
    desk = paper.render("desk")
    assert float((desk[on_sheet].min(axis=1) >= 250).mean()) > 0.97, float((desk[on_sheet].min(axis=1) >= 250).mean())
    assert np.median(desk[on_sheet]) == 255
    assert 90 <= np.median(gray(desk)[off]) <= 170, np.median(gray(desk)[off])
    # as in the real photos of the evening
    real_p = cv2.imread(os.path.join(DATA, "photos", "225720_cat1.jpg"))
    real_d = cv2.imread(os.path.join(DATA, "overexposed", "0637_penz_after_abort_raw.jpg"))
    assert abs(np.median(g[on_sheet]) - np.median(gray(real_p)[620:700, 500:620])) < 20
    assert float((real_d[620:700, 500:620].min(axis=2) >= 250).mean()) > 0.97
    assert abs(np.median(g[off]) - np.median(gray(real_p)[600:800, 1550:1800])) < 25
    try:
        paper.render("auto")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown exposure was rendered")
    # noise is optional, seeded, and of the size asked for
    noisy = paper.render("paper", noise=2.0)
    diff = noisy.astype(np.float32) - img.astype(np.float32)
    assert 1.5 < diff[on_sheet].std() < 2.5, diff[on_sheet].std()
    assert np.array_equal(paper.render("paper"), img)                                  # without noise: the same photo


def test_render_line_width_and_darkness_like_the_real_marker():
    """About 1 mm of marker: 3..5 px wide in the photo, core nearly black (photo 225720: 3-4 px, 5..30)."""
    paper = SP.SimPaper(cam=CAM)
    feed_line(paper, (0.24, -0.03), (0.24, 0.03), -0.002)          # across the camera's view: 2.6 px per mm
    feed_line(paper, (0.22, 0.05), (0.27, 0.05), -0.002, lean=None)  # along it: 3.5 px per mm
    paper.pen_up()
    g = gray(paper.render("paper"))
    level = float(np.median(g[paper._sheet_mask() > 0.999]))
    u, v = paper._project(np.array([[0.24, 0.0]]))[0]
    col = g[int(v) - 8:int(v) + 9, int(round(u))]
    width_h = float((level - col).sum() / (level - col.min()))      # equivalent width in px
    u, v = paper._project(np.array([[0.245, 0.05]]))[0]
    row = g[int(round(v)), int(u) - 8:int(u) + 9]
    width_v = float((level - row).sum() / (level - row.min()))
    assert 2.2 <= width_h <= 4.5 and 3.0 <= width_v <= 5.5, (width_h, width_v)
    assert col.min() < 60 and row.min() < 40, (col.min(), row.min())
    print(f"    marker line: {width_h:.1f} px wide (core {col.min():.0f}) across, {width_v:.1f} px (core {row.min():.0f}) along the view")


def test_render_square_where_the_camera_model_says():
    """SPEC 7: a commanded square is rendered where the camera model says, within 1.5 px."""
    worst = 0.0
    for kw in ({}, {"contact": SP.night_truths()["contact"]}, {"sheet": SP.sheet_polygon(rot_deg=0.0), "pen_width_m": 0.0004}):
        for sq in (square(), square(0.20, -0.06, 0.02), square(0.30, 0.07, 0.01)):
            paper = SP.SimPaper(cam=CAM, **kw)
            for a, b in zip(sq[:-1], sq[1:]):
                feed_line(paper, a, b, -0.0015)
            paper.pen_up()
            ink = np.concatenate([ln["xy"] for ln in paper.ink()])
            assert dist_to_polyline(ink, sq).max() < 1e-9 and abs(paper.ink_length() - S.length(sq)) < 6e-4
            for blur in (0.0, 0.6, 1.2):
                off = line_offsets_px(paper.render("paper", blur_px=blur), paper, sq)
                assert len(off) > 40 and np.all(np.isfinite(off)), (kw, off)
                worst = max(worst, float(off.max()))
                assert off.max() <= 1.5, (kw, sq[0], blur, off.max())
            # the corners themselves are where the model says, too
            g = gray(paper.render("paper"))
            for c in sq[:-1]:
                u, v = paper._project(c[None, :])[0]
                assert g[int(round(v)), int(round(u))] < np.median(g[paper._sheet_mask() > 0.999]) - 25, c
    print(f"    rendered line centre vs camera model: worst {worst:.2f} px")
    assert worst <= 1.0                       # in fact well inside the 1.5 px that are asked


def test_arm_square_through_the_monitor():
    """A square commanded to the simulated arm: the paper sees the MEASURED joint poses of every
    motion and the photo shows the square where the camera model puts the commanded one."""
    arm, bus = pen_arm()
    paper = SP.SimPaper(arm, CAM, push_skips=False)  # truths = the model (no hidden error); attaches itself
    assert paper.samples == 0 and "run_path" in arm.__dict__
    sq = square()
    draw_polyline(arm, sq, TABLE_Z - 0.001)
    assert paper.samples > 500 and paper._last_q is not None
    ink = paper.ink()
    assert len(ink) == 1, [len(ln["xy"]) for ln in ink]                              # one unbroken line
    assert 0.118 < paper.ink_length() < 0.132, paper.ink_length()               # 120 mm, plus the encoder steps' zigzag
    track = dist_to_polyline(ink[0]["xy"], sq) * 1000
    assert track.max() < 1.0, track.max()                                            # the sim arm tracks well
    assert 0.0005 < paper.max_depth_m < 0.0025, paper.max_depth_m
    # photo with the arm still above the square: the silhouette is in the way
    blocked = paper.render("paper")
    go_aside(arm)
    img = paper.render("paper")
    assert np.abs(blocked.astype(int) - img.astype(int)).max() > 80
    off_meas = line_offsets_px(img, paper, ink[0]["xy"][::8], skip_m=0.0, win=4)
    off_cmd = line_offsets_px(img, paper, sq)
    assert np.all(np.isfinite(off_cmd)) and len(off_cmd) > 60
    print(f"    arm square: ink vs commanded {track.max():.2f} mm max; photo vs commanded square {off_cmd.max():.2f} px max "
          f"(median {np.median(off_cmd):.2f}); {paper.samples} poses")
    assert off_cmd.max() <= 1.5, off_cmd.max()
    assert np.median(off_meas[np.isfinite(off_meas)]) < 1.0
    # the arm aside does not hide the square, and render(show_arm=False) is the same photo there
    u0, v0 = paper._project(sq.min(axis=0)[None, :])[0]
    u1, v1 = paper._project(sq.max(axis=0)[None, :])[0]
    box = (slice(int(min(v0, v1)) - 5, int(max(v0, v1)) + 5), slice(int(min(u0, u1)) - 5, int(max(u0, u1)) + 5))
    assert np.array_equal(img[box], paper.render("paper", show_arm=False)[box])
    paper.detach()
    assert "run_path" not in arm.__dict__ and "_simpaper" not in arm.__dict__


def test_attach_changes_no_motion_and_chains_monitors():
    """Watching must not change what the arm does (the default path of run_path stays what it is),
    and a caller's own monitor still runs and still stops the motion."""
    results = []
    for watch in (False, True):
        arm, bus = pen_arm()
        paper = SP.SimPaper(arm, CAM) if watch else None
        go_tip(arm, (0.24, 0.0, 0.03))
        r = tip_path(arm, [(0.24, 0.03, 0.01), (0.26, 0.03, 0.0)], 20.0)
        results.append((r["q_meas"].copy(), r["duration_s"], r["peak_err_deg"], dict(bus.pos), len(bus.log)))
        if watch:
            assert paper.samples > 100 and paper.ink_length() == 0.0             # seen, and no ink in the air
    a, b = results
    assert np.array_equal(a[0], b[0]) and a[1:] == b[1:], (a[1:3], b[1:3])
    # the caller's monitor is called with the same arguments and its True stops the motion
    arm, bus = pen_arm()
    paper = SP.SimPaper(arm, CAM)
    seen = []

    def inner(t, q_plan, q_meas):
        seen.append((t, q_plan.copy(), q_meas.copy()))
        return len(seen) >= 30

    try:
        arm.run_path([arm.read_q(), arm.solve(np.array([0.24, 0.0, 0.03]), PITCH, 0.0)], 30.0, monitor=inner)
    except MotionAborted as e:
        assert e.info.get("reason") == "monitor", e.info
    else:
        raise AssertionError("the caller's monitor did not stop the motion")
    assert len(seen) == 30 and paper.samples == 30
    assert np.allclose(paper._last_q, np.mean([q for _, _, q in seen[-5:]], axis=0))     # the real pose: 100 ms of readings
    # by hand, without attaching: monitor() is the callback Arm.run_path wants
    arm, bus = pen_arm()
    paper = SP.SimPaper(arm, CAM, attach=False)
    assert "run_path" not in arm.__dict__
    cb = paper.monitor()
    arm.run_path([arm.read_q(), arm.solve(np.array([0.24, 0.0, 0.03]), PITCH, 0.0)], 30.0, monitor=cb)
    n1, t1 = paper.samples, paper._time
    assert n1 > 50 and t1 > 0.5
    arm.run_path([arm.read_q(), arm.solve(np.array([0.24, 0.02, 0.03]), PITCH, 0.0)], 30.0, monitor=cb)   # the same callback again
    assert paper.samples > n1 and paper._time > t1 + 0.2                        # the paper's clock runs on


def test_ink_only_at_or_below_the_paper():
    """SPEC 7: ink appears only where the true tip is at or below the paper."""
    # the tip comes down along a line: +2 mm above the paper at the start, 2 mm below at the end
    paper = SP.SimPaper(cam=CAM)
    feed_line(paper, (0.24, -0.03), (0.24, 0.03), +0.002, -0.002)
    paper.pen_up()
    (ln,) = paper.ink()
    assert abs(ln["xy"][:, 1].min() - 0.0) < 2e-4 and abs(ln["xy"][:, 1].max() - 0.03) < 2e-4, ln["xy"][[0, -1]]
    assert np.all(ln["depth_m"] >= 0.0) and abs(paper.max_depth_m - 0.002) < 1e-6
    g = gray(paper.render("paper"))
    level = float(np.median(g[paper._sheet_mask() > 0.999]))
    for y, inked in ((-0.025, False), (-0.01, False), (-0.002, False), (0.004, True), (0.015, True), (0.028, True)):
        u, v = paper._project(np.array([[0.24, y]]))[0]
        patch = g[int(round(v)) - 3:int(round(v)) + 4, int(round(u)) - 3:int(round(u)) + 4]
        assert (patch.min() < level - 25) == inked, (y, patch.min(), level)
    # exactly at the paper counts as on it; a hair above does not
    paper = SP.SimPaper(cam=CAM)
    assert paper.feed_tip((0.24, 0.0, paper.surface_z(0.24, 0.0))) is True
    assert paper.feed_tip((0.24, 0.01, paper.surface_z(0.24, 0.01) + 1e-5)) is False
    assert len(paper.ink()) == 1 and len(paper.ink()[0]["xy"]) == 1                 # a dot
    # a tilted contact surface: a tip that keeps its height meets the paper only on the high side
    truth = SP.night_truths()
    paper = SP.SimPaper(cam=CAM, contact=truth["contact"])
    y0 = truth["contact"]["y0"]
    assert abs((paper.surface_z(0.25, y0 + 0.0525) - paper.surface_z(0.25, y0 - 0.0525)) - 0.0015) < 1e-9   # 1.5 mm across the box
    for k in range(601):
        y = y0 - 0.03 + 0.0001 * k
        paper.feed_tip((0.25, y, TABLE_Z))
    paper.pen_up()
    (ln,) = paper.ink()
    assert abs(ln["xy"][:, 1].min() - y0) < 2e-4 and ln["xy"][:, 1].max() > y0 + 0.0298, ln["xy"][[0, -1]]
    # other forms of the surface
    assert abs(SP.SimPaper(cam=CAM, contact=0.004).surface_z(0.2, 0.1) - 0.004) < 1e-12
    assert abs(SP.SimPaper(cam=CAM, contact=(0.001, 0.01, -0.02)).surface_z(0.2, 0.1) - (0.001 + 0.002 - 0.002)) < 1e-12
    assert abs(SP.SimPaper(cam=CAM, contact=lambda x, y: 0.5 * x).surface_z(0.2, 0.1) - 0.1) < 1e-12
    assert SP.SimPaper(cam=CAM).surface_z(0.2, 0.1) == TABLE_Z


def test_hidden_tool_offset_and_position_error():
    """The ink lands where the REAL tip is, not where the model believes it: a tool offset that is
    wrong by 15 mm sideways and 8 mm in length, plus a sheared error of 10 %."""
    truth = SP.night_truths()
    arm, bus = pen_arm()
    paper = SP.SimPaper(arm, CAM, tip_error=truth["tip_error"], pos_error=truth["pos_error"])
    assert np.allclose(paper.model_tip_offset, PEN_TIP) and np.allclose(paper.true_tip_offset - PEN_TIP, truth["tip_error"])
    c, m = np.array(truth["pos_error"]["centre"]), np.array(truth["pos_error"]["matrix"])

    def real_tip(q):                                   # from the truths, computed here and not by SimPaper
        r = K.fk(q, K.tool_offset(*(np.array(PEN_TIP) + truth["tip_error"])))[:3, 3]
        return np.array([*(r[:2] + m @ (r[:2] - c)), r[2]])

    # the model believes the tip is 6 mm ABOVE the paper; the real pen is longer and touches
    a, b = (0.24, -0.04), (0.24, 0.04)
    go_tip(arm, (a[0], a[1], 0.035))
    assert paper.ink() == [] and real_tip(arm.read_q())[2] > 0.02                     # the hover is a hover in truth too
    tip_path(arm, [(a[0], a[1], TABLE_Z + 0.006)], 15.0)
    q_start = arm.read_q()
    tip_path(arm, [(b[0], b[1], TABLE_Z + 0.006)], 15.0)
    q_end = arm.read_q()
    tip_path(arm, [(b[0], b[1], 0.035)], 30.0)
    ink = np.concatenate([ln["xy"] for ln in paper.ink()])
    assert 0.07 < paper.ink_length() < 0.095, paper.ink_length()
    for q, aim in ((q_start, a), (q_end, b)):
        model = K.fk(q, arm.tip_tool())[:3, 3]
        assert abs(model[2] - (TABLE_Z + 0.006)) < 0.0015 and np.linalg.norm(model[:2] - aim) < 0.0015, model   # the model's view: in the air, on the aim
        real = real_tip(q)
        assert real[2] < TABLE_Z - 0.0005, real                                       # the truth: in the paper
        assert np.linalg.norm(ink - real[:2], axis=1).min() < 6e-4, (real, aim)       # and that is where the ink is
        assert 0.012 < np.linalg.norm(real[:2] - aim) < 0.03, (real, aim)             # centimetres beside the aim
        assert np.allclose(paper.true_tip(q), real, atol=1e-9)
    # the sheared part alone: what pos_error adds is 10 % of the distance across the box, along x
    plain = SP.SimPaper(arm, CAM, tip_error=truth["tip_error"], attach=False)
    d = [paper.true_tip(q)[:2] - plain.true_tip(q)[:2] for q in (q_start, q_end)]
    assert abs(abs(d[0][0] - d[1][0]) - 0.10 * abs(plain.true_tip(q_start)[1] - plain.true_tip(q_end)[1])) < 1e-9, d
    assert abs(d[0][1]) < 1e-12 and 0.006 < abs(d[0][0] - d[1][0]) < 0.010, d
    # the photo shows the ink at the real place and nothing at the commanded one
    go_aside(arm)
    g = gray(paper.render("paper"))
    level = float(np.median(g[paper._sheet_mask() > 0.999]))
    u, v = paper._project(np.array([[0.24, 0.0]]))[0]
    assert g[int(v) - 4:int(v) + 5, int(u) - 4:int(u) + 5].min() > level - 10
    mid = ink[len(ink) // 2]
    u, v = paper._project(mid[None, :])[0]
    assert g[int(round(v)) - 2:int(round(v)) + 3, int(round(u)) - 2:int(round(u)) + 3].min() < level - 40
    assert paper.truth()["true_tip_offset"] == paper.true_tip_offset.tolist()
    print(f"    hidden tool error: aim {b} -> ink {np.round(real_tip(q_end)[:2], 4).tolist()} "
          f"({np.linalg.norm(real_tip(q_end)[:2] - b) * 1000:.1f} mm beside it)")


def test_height_wander_is_seeded_smooth_and_bounded():
    def run(seed, wander=1.5):
        p = SP.SimPaper(cam=CAM, wander_mm=wander, seed=seed)
        # 6 mm/s for 20 s, planned 0.5 mm into the paper; the wander is added to the tip height as feed() does
        for k in range(1000):
            t = k * 0.02
            y = -0.06 + 0.006 * t
            p.feed_tip((0.24, y, p.surface_z(0.24, y) - 0.0005 + p.wander(t)), t=t)
        p.pen_up()
        return p

    a, b, c = run(1), run(1), run(2)
    ia, ib, ic = (np.concatenate([ln["xy"] for ln in p.ink()]) for p in (a, b, c))
    assert np.array_equal(ia, ib)                                           # the same seed: the same world
    assert len(ia) != len(ic) or not np.allclose(ia, ic)                    # another seed: another one
    # the wander lifts the pen off the paper for parts of the line (2026-11-11: two of seven marks left nothing)
    fracs = [p.ink_length() / 0.12 for p in (a, c, run(3), run(4), run(5))]
    assert all(0.15 < f < 0.999 for f in fracs) and min(fracs) < 0.9, fracs
    assert abs(run(1, wander=0.0).ink_length() - 0.12) < 3e-4                # without it the line is whole
    ts = np.arange(0.0, 300.0, 0.02)
    w = np.array([a.wander(t) for t in ts])
    assert np.abs(w).max() <= 0.0015 + 1e-12 and np.abs(w).max() > 0.0009, np.abs(w).max()
    assert np.abs(np.diff(w)).max() < 5e-5                                   # under 0.05 mm per control cycle: smooth
    assert abs(w.mean()) < 2e-4
    # through the arm: feed() adds it to the real tip
    arm, _ = pen_arm()
    p = SP.SimPaper(arm, CAM, wander_mm=1.5, seed=1)
    q = arm.read_q()
    z = [p.true_tip(q, t)[2] for t in (0.0, 2.0, 4.0, 6.0)]
    assert np.ptp(z) > 0.0003 and np.allclose(np.array(z) - K.fk(q, arm.tip_tool())[2, 3], [p.wander(t) for t in (0.0, 2.0, 4.0, 6.0)])


def test_pushed_pen_skips_pulled_pen_does_not():
    lean = np.array([math.sin(math.radians(15.0)), 0.0])           # the pen of the night: tip 15 degrees ahead, towards +x
    out = {}
    for name, a, b in (("push", (0.20, 0.0), (0.28, 0.0)), ("pull", (0.28, 0.0), (0.20, 0.0)),
                       ("side", (0.24, -0.04), (0.24, 0.04)), ("diag_pull", (0.27, -0.03), (0.21, 0.03)),
                       ("diag_push", (0.21, -0.03), (0.27, 0.03))):
        p = SP.SimPaper(cam=CAM, seed=3, skips="tip-leading")    # (SPEC 9.1: a hidden truth the test sets; default none)
        feed_line(p, a, b, -0.002, lean=lean)
        p.pen_up()
        full = float(np.linalg.norm(np.array(b) - np.array(a)))
        out[name] = (p.ink_length() / full, len(p.ink()))
    assert 0.3 < out["push"][0] < 0.8 and out["push"][1] >= 8, out          # dashes
    assert 0.3 < out["diag_push"][0] < 0.85 and out["diag_push"][1] >= 6, out
    for name in ("pull", "side", "diag_pull"):
        assert out[name][0] > 0.995 and out[name][1] == 1, out              # whole lines
    # an upright pen, skipping switched off, or the default (no direction loses ink): whole
    for kw, ln in (({"skips": "tip-leading"}, np.array([0.01, 0.0])), ({"push_skips": False}, lean),
                   ({"skips": "tip-leading"}, None), ({}, lean)):
        p = SP.SimPaper(cam=CAM, seed=3, **kw)
        feed_line(p, (0.20, 0.0), (0.28, 0.0), -0.002, lean=ln)
        p.pen_up()
        assert len(p.ink()) == 1 and p.ink_length() > 0.0795, (kw, ln)
    # through the arm: at gripper pitch 75 the pen's lean points away from the robot's pan axis
    arm, _ = pen_arm()
    p = SP.SimPaper(arm, CAM)
    for xy in ((0.24, 0.0), (0.22, 0.08), (0.28, -0.07)):
        q = arm.solve(np.array([xy[0], xy[1], 0.0]), PITCH, 0.0)
        ln = p.lean_xy(q)
        size = float(np.linalg.norm(ln))
        assert 0.15 < size < 0.45, size                                       # sin of 9..27 degrees
        assert float(ln @ RADIAL(xy)) / size > 0.85, (xy, ln, RADIAL(xy))
    # ... so a stroke away from the robot comes out dashed and the same stroke towards it whole
    res = {}
    for name, pts in (("away", [(0.21, 0.0), (0.27, 0.0)]), ("towards", [(0.27, 0.0), (0.21, 0.0)])):
        arm, _ = pen_arm()
        p = SP.SimPaper(arm, CAM, seed=5, skips="tip-leading")
        draw_polyline(arm, pts, TABLE_Z - 0.0015)
        res[name] = (p.ink_length() / 0.06, len(p.ink()))
    assert res["away"][1] >= 5 and res["away"][0] < 0.85 and res["towards"][1] == 1 and res["towards"][0] > 0.97, res
    print(f"    pushed: {out['push'][0] * 100:.0f} % inked in {out['push'][1]} dashes; arm away {res['away']}, towards {res['towards']}")


def test_darkness_and_width_grow_with_depth():
    paper = SP.SimPaper(cam=CAM)
    depths = (0.0001, 0.0005, 0.001, 0.003)
    for i, d in enumerate(depths):
        feed_line(paper, (0.22 + 0.01 * i, -0.02), (0.22 + 0.01 * i, 0.02), -d)
        paper.pen_up()
    ink = paper.ink()
    assert len(ink) == 4
    dark = [float(ln["darkness"].mean()) for ln in ink]
    width = [float(ln["width_m"].mean()) for ln in ink]
    assert dark == sorted(dark) and dark[0] < 0.55 and dark[2] == 1.0, dark
    assert width == sorted(width) and width[0] < 0.0007 and abs(width[2] - 0.001) < 1e-9 and width[3] > 0.0011, width
    g = gray(paper.render("paper"))
    level = float(np.median(g[paper._sheet_mask() > 0.999]))
    core, area = [], []
    for i in range(4):
        u, v = paper._project(np.array([[0.22 + 0.01 * i, 0.0]]))[0]
        col = g[int(round(v)) - 6:int(round(v)) + 7, int(round(u))]
        core.append(float(col.min()))
        area.append(float((level - col).sum()))
    assert core == sorted(core, reverse=True) and core[0] > core[3] + 40, core       # the deeper, the darker
    assert area == sorted(area) and area[3] > 2.0 * area[0], area                    # ... and the wider
    # pressing harder on the spot darkens the dot
    p = SP.SimPaper(cam=CAM)
    for d in (0.0, 0.0002, 0.0008, 0.002):
        p.feed_tip((0.24, 0.0, TABLE_Z - d))
    assert len(p.ink()[0]["xy"]) == 1 and p.ink()[0]["darkness"][0] == 1.0 and abs(p.max_depth_m - 0.002) < 1e-9


def test_thin_ink_vanishes_with_the_desk_exposure():
    """The night's trap: with the locked exposure a thin line is not there, and nothing says so."""
    thin = SP.SimPaper(cam=CAM, pen_width_m=0.0004)
    bold = SP.SimPaper(cam=CAM)
    for p in (thin, bold):
        for i in range(5):
            feed_line(p, (0.21 + 0.012 * i, -0.03), (0.21 + 0.012 * i + 0.006, 0.03), -0.0004)
            p.pen_up()
    u0, v0 = thin._project(np.array([[0.20, -0.04]]))[0]
    u1, v1 = thin._project(np.array([[0.28, 0.04]]))[0]
    box = (slice(int(min(v0, v1)), int(max(v0, v1))), slice(int(min(u0, u1)), int(max(u0, u1))))
    tp, td = gray(thin.render("paper"))[box], thin.render("desk")[box]
    assert np.median(tp) - tp.min() > 25, (np.median(tp), tp.min())                 # paper exposure: the lines show
    assert int((tp < np.median(tp) - 15).sum()) > 300
    assert td.min() >= 250, td.min()                                                  # desk exposure: nothing at all
    with_noise = thin.render("desk", noise=1.5)[box]
    assert float((with_noise.min(axis=2) >= 250).mean()) > 0.99
    # a bold marker line pressed hard leaves a trace even then, but a much weaker one
    hard = SP.SimPaper(cam=CAM)
    feed_line(hard, (0.24, -0.03), (0.24, 0.03), -0.002)
    hard.pen_up()
    hp, hd = gray(hard.render("paper"))[box], gray(hard.render("desk"))[box]
    assert hp.min() + 20 < hd.min() < 200 and int((hd < 128).sum()) < int((hp < 128).sum()), (hp.min(), hd.min())
    print(f"    thin pen: paper exposure core {tp.min():.0f} (paper {np.median(tp):.0f}), desk exposure min {td.min()}; "
          f"bold line with desk exposure: core {hd.min():.0f}")


def test_old_ink_and_new_ink():
    old = [np.array([[0.20, 0.06], [0.20, 0.066]]), {"xy": [[0.21, 0.06], [0.21, 0.066]], "width_m": 0.0004, "darkness": 0.5}]
    paper = SP.SimPaper(cam=CAM, old_ink=old)
    before = paper.render("paper")
    assert len(paper.ink()) == 2 and all(ln["old"] for ln in paper.ink()) and paper.ink(new_only=True) == []
    assert paper.ink_length() == 0.0 and abs(paper.ink_length(new_only=False) - 0.012) < 1e-9
    feed_line(paper, (0.25, -0.02), (0.25, 0.02), -0.002)
    paper.pen_up()
    after = paper.render("paper")
    assert len(paper.ink()) == 3 and len(paper.ink(new_only=True)) == 1 and abs(paper.ink_length() - 0.04) < 2e-4
    g0, g1 = gray(before), gray(after)
    level = float(np.median(g0[paper._sheet_mask() > 0.999]))
    for xy, drop in (((0.20, 0.063), 60), ((0.21, 0.063), 8)):                       # both old marks, in both photos
        u, v = paper._project(np.array([xy]))[0]
        win = (slice(int(round(v)) - 3, int(round(v)) + 4), slice(int(round(u)) - 3, int(round(u)) + 4))
        assert g0[win].min() < level - drop and g1[win].min() < level - drop, (xy, g0[win].min(), level)
    changed = np.argwhere(np.abs(g1 - g0) > 10)
    uv = paper._project(np.array([[0.25, -0.02], [0.25, 0.02]]))
    assert len(changed) > 200
    assert changed[:, 1].min() >= uv[:, 0].min() - 6 and changed[:, 1].max() <= uv[:, 0].max() + 6     # only the new stroke
    assert changed[:, 0].min() >= uv[:, 1].min() - 6 and changed[:, 0].max() <= uv[:, 1].max() + 6
    # the truth a detector is compared with
    cov = paper.coverage([np.array([[0.25, -0.02], [0.25, 0.02]])])
    assert cov["drawn_fraction"] == 1.0 and cov["offset_mm"]["mean"] < 0.2, cov
    cov = paper.coverage([np.array([[0.252, -0.04], [0.252, 0.04]])])
    assert 0.45 < cov["drawn_fraction"] < 0.6 and 1.9 < cov["offset_mm"]["mean"] < 2.6, cov
    assert paper.coverage([np.array([[0.20, 0.06], [0.20, 0.066]])])["drawn_fraction"] == 0.0     # old ink is not "drawn"
    # the sheet as plain data: another process can go on with it
    twin = SP.SimPaper(cam=CAM).load_state(json.loads(json.dumps(paper.state())))
    assert np.array_equal(twin.render("paper"), after) and len(twin.ink(new_only=True)) == 1
    paper.clear()
    assert np.array_equal(paper.render("paper"), before) and paper.max_depth_m == 0.0
    paper.clear(old_too=True)
    assert paper.ink() == []


def test_cat_through_the_whole_pipeline_in_simulation():
    """layout -> pull pieces -> pre-distortion -> simulated arm -> SimPaper -> truth: with a sheared
    position error that the map knows, the ink lands on the drawing; without pull pieces the pushed
    lines come out dashed, as on 2026-11-11."""
    truth = SP.night_truths()
    aff = np.array([[NIGHT["XC"], NIGHT["YC"]], [1.0, 0.0], [-0.10, 1.0]])           # the world's shear as a map
    world_map = {"centre": [NIGHT["XC"], NIGHT["YC"]], "kind": "affine", "coef": aff.tolist()}
    want = night_cat()[2:9]                       # the face: seven strokes are enough to show it (and quick)
    total = sum(S.length(w) for w in want)
    res = {}
    for name, lean in (("as drawn", None), ("pull pieces", RADIAL)):
        arm, _ = pen_arm()
        paper = SP.SimPaper(arm, CAM, pos_error=truth["pos_error"], seed=11, skips="tip-leading")
        plan = S.plan_pieces(want, lean, world_map)
        for pc in plan["pieces"]:
            c = pc["cmd"]
            go_tip(arm, (c[0][0], c[0][1], 0.02))
            tip_path(arm, [(c[0][0], c[0][1], TABLE_Z - 0.0015)], 15.0)
            for part in pc["parts"]:
                tip_path(arm, [(p[0], p[1], TABLE_Z - 0.0015) for p in part[1:]], 15.0)
            tip_path(arm, [(c[-1][0], c[-1][1], 0.02)], 30.0)
        res[name] = {"cov": paper.coverage(want, tol_m=0.002), "tight": paper.coverage(want, tol_m=0.0006),
                     "pieces": plan["stats"]["count"], "lines": len(paper.ink()), "ink": paper.ink_length()}
    raw, pull = res["as drawn"], res["pull pieces"]
    print(f"    face of the cat ({total * 1000:.0f} mm): as drawn {raw['lines']} ink lines from {raw['pieces']} strokes, "
          f"{raw['tight']['drawn_fraction']:.2f} of the line inked; pull pieces {pull['lines']} lines from {pull['pieces']} pieces, "
          f"{pull['tight']['drawn_fraction']:.2f} inked, offset {pull['cov']['offset_mm']['mean']:.2f} mm mean / "
          f"{pull['cov']['offset_mm']['p95']:.2f} p95")
    assert pull["cov"]["drawn_fraction"] >= 0.99 and pull["cov"]["offset_mm"]["mean"] < 0.6, pull    # on the drawing
    # nearly whole lines: what is left are the few millimetres of push that the 3 mm of slack allow
    assert pull["lines"] < raw["lines"] - 5 and pull["tight"]["drawn_fraction"] >= 0.93, pull
    assert raw["lines"] >= raw["pieces"] + 5, raw                                    # pushed strokes: dashes
    assert raw["tight"]["drawn_fraction"] < pull["tight"]["drawn_fraction"] - 0.05, (raw["tight"], pull["tight"])
    assert raw["ink"] < 0.95 * pull["ink"], (raw["ink"], pull["ink"])


def test_render_speed():
    paper = SP.SimPaper(cam=CAM)
    for w in night_cat():
        for k in range(0, len(w) - 1, 40):                 # as many separate lines as a dashed drawing has
            paper.add_ink(S.resample(w[k:k + 41], 0.00025), old=False)
    t0 = time.perf_counter()
    paper.render("paper")
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    paper.render("desk")
    again = time.perf_counter() - t0
    print(f"    render of the cat ({sum(len(ln['xy']) for ln in paper.ink())} ink points): {first:.2f} s, next photo {again:.2f} s")
    if first > 2.0:                                        # six cores shared with other test runs
        t0 = time.perf_counter()
        paper.add_ink([[0.2, 0.0], [0.2, 0.001]])
        paper.render("paper")
        first = time.perf_counter() - t0
    assert first < 2.0 and again < 2.0, (first, again)


def test_no_hardware_was_touched():
    """Runs last (z): the audit hook saw no serial port, no video device, no v4l2-ctl."""
    assert _TRIPPED == [], _TRIPPED
    try:
        open("/dev/ttyACM0")
    except RuntimeError:
        _TRIPPED.clear()                                   # the guard works (and nothing was opened)
    except OSError:
        raise AssertionError("the audit hook is not armed")
    else:
        raise AssertionError("the audit hook is not armed")


test_zz_no_hardware_was_touched = test_no_hardware_was_touched
del test_no_hardware_was_touched


if __name__ == "__main__":
    import traceback
    only = sys.argv[1:]
    fails = skipped = 0
    n = 0
    t_all = time.time()
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn) and (not only or any(o in name for o in only)):
            n += 1
            t0 = time.time()
            try:
                fn()
                print(f"PASS {name} ({time.time() - t0:.1f} s)")
            except Skip as e:
                skipped += 1
                print(f"SKIP {name}: {e}")
            except Exception:
                fails += 1
                print("FAIL", name)
                traceback.print_exc()
    print(f"tests: {n}  passed: {n - fails - skipped}  skipped: {skipped}  failures: {fails}  ({time.time() - t_all:.0f} s)")
    sys.exit(1 if fails else 0)
