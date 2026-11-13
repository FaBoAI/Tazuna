"""Shared helpers for drawing with a pen taped to the SO-101 gripper (2026-11-11).

The pen tip is treated as the arm's tool point: config_pen.json is the live config.json with
"tip_offset" replaced by the pen tip's place in gripper_frame_link axes (x: along the jaw-opening
axis, the direction the pen points; y: sideways; z: along the fingers). With FD_CONFIG pointing at
that file, robot.Arm plans, checks (floor, limits, keep-outs) and moves the PEN TIP.

Nothing here knows about the standard `fd ready` / `fd park` poses: with a pen on the gripper those
poses are NOT safe (the pen points at the base in the folded pose). The scripts in this directory
start from, and return to, the pose the arm was found in.
"""
import json
import math
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PEN_DIR = os.path.dirname(os.path.abspath(__file__))
CFG_PATH = os.path.join(PEN_DIR, "config_pen.json")
STATE_PATH = os.path.join(PEN_DIR, "pen_state.json")
os.environ["FD_CONFIG"] = CFG_PATH  # robot.load_config() reads it; must be set before Arm() is built
sys.path.insert(0, HERE)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import camera  # noqa: E402
import kinematics as K  # noqa: E402
from robot import ARM, Arm, MotionAborted, SafetyError, load_config  # noqa: E402,F401

FD = os.path.join(HERE, "fd")


def load_state() -> dict:
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as fh:
            return json.load(fh)
    return {}


def save_state(**kw) -> dict:
    st = load_state()
    st.update(kw)
    with open(STATE_PATH, "w") as fh:
        json.dump(st, fh, indent=1)
    return st


def write_pen_config(tip_offset) -> dict:
    """config_pen.json = live config.json with the pen tip as the tool point."""
    with open(os.path.join(HERE, "config.json")) as fh:
        cfg = json.load(fh)
    cfg["tip_offset"] = [round(float(v), 5) for v in tip_offset]
    cfg["pen_note"] = ("pen-drawing config: tip_offset is the PEN TIP in gripper_frame_link axes; do not use it with "
                       "fd ready / fd park")
    with open(CFG_PATH, "w") as fh:
        json.dump(cfg, fh, indent=1)
    return cfg


def make_arm(sim: bool, start_raw: dict | None = None) -> Arm:
    """Real arm, or a simulated one that starts at start_raw (default: the pose saved in pen_state.json)."""
    if not sim:
        return Arm()
    from sim_bus import SimBus
    bus = SimBus(realtime=False)
    raw = start_raw or load_state().get("home_raw")
    if raw:
        for j, v in raw.items():
            bus.pos[j] = float(v)
            bus.goal[j] = float(v)
    return Arm(bus=bus, cfg=load_config())


def snap(name: str) -> str:
    """Photo with the live camera through `fd snap --no-robot` (the servo bus is not opened). Returns the raw path."""
    env = {k: v for k, v in os.environ.items() if k not in ("FD_CONFIG", "FD_CAMERA_MODEL")}
    p = subprocess.run([FD, "snap", "--name", name, "--plain", "--no-robot"], capture_output=True, text=True,
                       timeout=60, env=env)
    r = json.loads(p.stdout)
    if not r.get("ok"):
        raise RuntimeError("snap failed: %s" % r)
    return r["raw"]


def pen_T(q, tip_offset=None, cfg=None):
    """4x4 pose of the pen-tip tool frame."""
    off = tip_offset if tip_offset is not None else (cfg or load_config())["tip_offset"]
    return K.fk(np.asarray(q, float), K.tool_offset(*off))


def free_path_min_z(arm: Arm, q_from, q_to, n: int = 60):
    """Lowest pen-tip height along a straight joint-space move (what run_path does between two waypoints)."""
    tool = arm.tip_tool()
    zs = [K.fk(q_from + (q_to - q_from) * k / n, tool)[2, 3] for k in range(n + 1)]
    return float(min(zs))


def go_joints(arm: Arm, q_to, speed: float = 25.0, min_tip_z: float = 0.02, **kw):
    """Joint-space move with the pen tip kept above min_tip_z (checked before anything moves)."""
    q_now = arm.read_q()
    q_to = np.asarray(q_to, float)
    lo = free_path_min_z(arm, q_now, q_to)
    if lo < min_tip_z:
        raise SafetyError("free move would bring the pen tip down to z=%.4f (limit %.4f)" % (lo, min_tip_z))
    return arm.run_path([q_now, q_to], speed, **kw)


def go_tip(arm: Arm, xyz, pitch: float, roll: float = 0.0, speed: float = 25.0, min_tip_z: float = 0.02, **kw):
    """Joint-space move that ends with the pen tip at xyz (pitch / roll of the gripper as given)."""
    q_to = arm.solve(np.asarray(xyz, float), pitch, roll, q_seed=arm.read_q())
    return go_joints(arm, q_to, speed, min_tip_z, **kw)


def find_pen_tip(img, pred_uv, win: int = 110, thr: int = 185):
    """Pixel of the pen tip: the lowest point of the dark blob nearest to the predicted pixel, inside a window.
    Returns (u, v) or None."""
    h, w = img.shape[:2]
    x0, x1 = max(0, int(pred_uv[0]) - win), min(w, int(pred_uv[0]) + win)
    y0, y1 = max(0, int(pred_uv[1]) - win), min(h, int(pred_uv[1]) + win)
    mx = img[y0:y1, x0:x1].max(axis=2)
    dark = (mx < thr).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(dark, 8)
    best, bd = None, 1e9
    for k in range(1, n):
        if st[k, cv2.CC_STAT_AREA] < 60:
            continue
        ys, xs = np.where(lab == k)
        # the pen comes down from above: its blob must reach the upper part of the window and be slender at the bottom
        yb = ys.max()
        xb = xs[ys >= yb - 3]
        if np.ptp(xb) > 40:
            continue
        d = math.hypot(xb.mean() + x0 - pred_uv[0], yb + y0 - pred_uv[1])
        if d < bd:
            best, bd = (float(xb.mean() + x0), float(yb + y0 + 0.5)), d
    return best


# ---------------------------------------------------------------------------- drawing geometry (shared)
PITCH = 80.0          # gripper pitch while drawing (the pen lies along the fingers: 10 deg from vertical)
XC, YC = 0.223, 0.003   # centre of the drawing box on the table (base frame, metres)
HALF = 0.0525         # half of the drawing box side
ROT_DEG = 6.0         # the sheet is taped about 6 deg askew: the drawing is turned with it
X_TOP, X_BOT = XC - HALF - 0.007, XC + HALF + 0.006   # the two lines just outside the box used for height strokes
ASIDE = (0.215, -0.150, 0.075)   # pen tip place where the arm hides none of the drawing from the camera
LIFT = 0.008          # pen-up height above the paper


def tip_path(arm, pts, v_mm_s: float = 25.0, settle=0.3, step: float = 0.003, pitch: float = PITCH, contact: bool = True,
             freeze: bool = False):
    """Move the pen tip along straight segments through pts [(x, y, z), ...] at about v_mm_s.

    freeze=True keeps the arm's gravity correction at the value the motion starts with (no integral action while
    it moves). The harness integrates the joint error whenever a joint is planned slower than 5 deg/s, which at
    drawing speed winds up against friction and lets the pen wander by several millimetres; a stroke therefore
    starts from a pose that was settled (settle=True) and is then run frozen, without a settle at its end."""
    q0 = arm.read_q()
    tool = arm.tip_tool()
    p0 = K.fk(q0, tool)[:3, 3]
    wps = arm.plan_tip_path([(np.asarray(p, float), pitch, 0.0) for p in pts], q_start=q0, step_m=step)
    length = float(sum(np.linalg.norm(np.asarray(b, float) - np.asarray(a, float)) for a, b in zip([p0, *pts[:-1]], pts)))
    dq = float(np.degrees(np.abs(np.diff(np.array(wps), axis=0)).sum(axis=0).max()))
    speed = 1.3 * dq * (v_mm_s / 1000.0) / max(length, 1e-4)
    speed = min(max(speed, 2.0), 35.0)
    if not freeze:
        return arm.run_path(wps, speed, allow_contact=contact, settle=settle)
    ki = arm.cfg.get("ki_per_s", 2.0)
    arm.cfg["ki_per_s"] = 0.0
    try:
        return arm.run_path(wps, speed, allow_contact=contact, settle=False)
    finally:
        arm.cfg["ki_per_s"] = ki


def go_aside(arm, speed: float = 30.0):
    """Lift the pen (if it is low) and move the arm to the side pose (joint space), for a photo."""
    tool = arm.tip_tool()
    q = arm.read_q()
    p = K.fk(q, tool)[:3, 3]
    if p[2] < 0.03:
        pitch_now = K.pose_summary(q, tool)["pitch_down_deg"]
        for dp, dx in ((0.0, 0.0), (-6.0, -0.01), (-12.0, -0.02)):       # the measured pitch may be out of reach higher up
            try:
                tip_path(arm, [(p[0] + dx, p[1], 0.035)], 40.0, settle=0.1, pitch=pitch_now + dp)
                break
            except SafetyError:
                if dp <= -12.0:
                    raise
        q = arm.read_q()
    q_side = np.radians(load_state().get("aside_q_deg", [43.15, -20.31, 10.46, 63.28, -0.05]))
    lo = free_path_min_z(arm, q, q_side)
    if lo < 0.02:
        raise SafetyError("the way to the side pose would bring the pen tip down to z=%.3f" % lo)
    return arm.run_path([q, q_side], speed, settle=0.5)


def stretch_crop(img, cx, cy, half_w=170, half_h=100, scale=3):
    """Contrast-stretched, enlarged crop around a pixel (for a human look at faint ink). Returns (image, x0, y0)."""
    x0 = max(0, min(img.shape[1] - 2 * half_w, int(cx) - half_w))
    y0 = max(0, min(img.shape[0] - 2 * half_h, int(cy) - half_h))
    g = img[y0:y0 + 2 * half_h, x0:x0 + 2 * half_w].min(axis=2).astype(np.float32)
    lo, hi = np.percentile(g, 0.3), np.percentile(g, 99.5)
    t = np.clip((g - lo) * 255.0 / max(hi - lo, 1), 0, 255).astype(np.uint8)
    return cv2.cvtColor(cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), cv2.COLOR_GRAY2BGR), x0, y0


def ink_map(img):
    """Thin dark lines on paper as a positive image (black-hat filter): broad shadows give no response."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))).astype(np.int16)


def ink_on_segment(before, after, cam, a_xy, b_xy, z_plane: float, corridor_px: int = 60, thr: int = 13):
    """New ink (a thin dark line in `after` that is not in `before`) near the commanded stroke a -> b.
    Photos must be exposed for the paper (snap_paper). Returns {"found", "n", "s_lo", "s_hi", "ink_mm", "lat_mm",
    "end_xy"}: s = position along the commanded stroke as the camera sees it (0 = a, 1 = b), ink_mm = length of the
    inked part, lat_mm = sideways offset of the ink from the commanded line, end_xy = where the ink ends (table)."""
    new_ink = ink_map(after) - (0 if before is None else cv2.dilate(ink_map(before).astype(np.uint8), np.ones((5, 5), np.uint8)).astype(np.int16))
    a = np.array([*a_xy, z_plane], float)
    b = np.array([*b_xy, z_plane], float)
    ext = (b - a) * 0.6
    pa, pb = cam.project([a - ext, b + ext])
    mask = np.zeros(new_ink.shape, np.uint8)
    cv2.line(mask, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), 255, thickness=2 * corridor_px)
    ys, xs = np.where((new_ink > thr) & (mask > 0))
    if len(xs) < 12:
        return {"found": False, "n": int(len(xs))}
    if len(xs) > 1500:
        idx = np.linspace(0, len(xs) - 1, 1500).astype(int)
        xs, ys = xs[idx], ys[idx]
    pts = np.array([cam.to_plane(np.array([float(u), float(v)]), z_plane)[:2] for u, v in zip(xs, ys)])
    d = b[:2] - a[:2]
    ln = float(np.linalg.norm(d))
    d = d / ln
    perp = np.array([-d[1], d[0]])
    s = (pts - a[:2]) @ d / ln
    lat = (pts - a[:2]) @ perp
    med = float(np.median(lat))
    keep = np.abs(lat - med) < 0.004          # the main line (drops stray specks)
    s, lat, pts = s[keep], lat[keep], pts[keep]
    if len(s) < 10:
        return {"found": False, "n": int(len(s))}
    s_lo, s_hi = float(np.percentile(s, 1.5)), float(np.percentile(s, 98.5))
    end = a[:2] + d * s_hi * ln + perp * float(np.median(lat[s > s_hi - 0.15]))
    return {"found": True, "n": int(len(s)), "s_lo": round(s_lo, 3), "s_hi": round(s_hi, 3),
            "ink_mm": round((s_hi - s_lo) * ln * 1000.0, 2), "lat_mm": round(med * 1000.0, 2),
            "end_xy": [round(float(end[0]), 4), round(float(end[1]), 4)]}


# ---------------------------------------------------------------------------- photos exposed for white paper
PHOTO_DIR = os.path.join(HERE, "runs", "pen_draw")
PAPER_EXPOSURE = (("auto_exposure", 1), ("exposure_dynamic_framerate", 0), ("exposure_time_absolute", 110), ("gain", 0))


def snap_paper(name: str, exposure: int | None = None, n_avg: int = 4):
    """Photo with a short exposure, so that white paper is not clipped and thin ink lines show. The harness's own
    locked settings (tuned for the dark desk) are put back afterwards. Returns (path, image)."""
    os.makedirs(PHOTO_DIR, exist_ok=True)
    camera.lock_settings()  # focus / white balance as always
    pairs = [(k, (exposure if (k == "exposure_time_absolute" and exposure) else v)) for k, v in PAPER_EXPOSURE]
    camera._ctl(pairs)
    try:
        img = camera.grab(n_avg=n_avg, warm=10, lock=False)
    finally:
        camera.lock_settings()
    path = os.path.join(PHOTO_DIR, "%s_%s.jpg" % (time.strftime("%H%M%S"), name))
    cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return path, img


def find_pen_generic(img, anchor_uv, max_dist: int = 380, dark: int = 95, min_area: int = 350):
    """Lowest point (in the image) of the dark thing hanging below the gripper: the pen tip, for a pen seen against
    white paper in a short-exposure photo. anchor_uv: pixel of the gripper fingertip. Returns (u, v) or None."""
    mx = img.max(axis=2)
    n, lab, st, _ = cv2.connectedComponentsWithStats((mx < dark).astype(np.uint8), 8)
    best = None
    for k in range(1, n):
        x, y, w, h, area = st[k]
        if area < min_area or area > 90000:
            continue
        dx = max(x - anchor_uv[0], 0, anchor_uv[0] - (x + w))
        dy = max(y - anchor_uv[1], 0, anchor_uv[1] - (y + h))
        if math.hypot(dx, dy) > max_dist:
            continue
        ys, xs = np.where(lab == k)
        yb = int(ys.max())
        if yb >= img.shape[0] - 3:
            continue
        cand = (yb, float(xs[ys >= yb - 2].mean()), int(area))
        if best is None or cand[0] > best[0]:
            best = cand
    return None if best is None else (best[1], best[0] + 0.5)
