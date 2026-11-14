"""Tests for the perception side: scene.py, camera.py, vision.py, fastcam.py, blockfit.py, capfit.py.

    python tests/test_scene.py              all tests + the accuracy tables
    python tests/test_scene.py jaws runtime only the tests whose name contains one of the words
    python tests/test_scene.py --live       additionally open the real camera once (FastCamera), then
                                      put it back to automatic mode

No hardware is touched without --live (and even then only the camera). Saved frames and the hand
measured ground truth are in tests/testdata/ (frames.json). camera_model.json and config.json are only read.
The hardware lock (tools/motion/hwlock.py) is the USER's here, and --live honours it. Two tests drive
fastcam.py / camera.py with stand-ins in the place of the camera and of v4l2-ctl; with the user's lock set those
modules refuse before they reach the stand-ins, so for the duration of these two tests ONLY the lock is stood in
for (_lock_stood_in: "not locked"), and while it is, a real cv2.VideoCapture, a real subprocess.run and any open
of a camera or servo device fail the test instead of reaching the hardware.
tests/testdata/takeout_20261111/ holds frames of the day the cap was taken out of the box by hand-driven
commands (crowded box, turned box, gripper in front of it): the test_takeout_* tests compare only
presence, kinds, counts, sizes and relative positions on them, never absolute coordinates.
"""

import os as _os  # the harness modules live in tools/<feature>/ (see tools/tazuna_paths.py)
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "tools"))
import tazuna_paths  # noqa: E402,F401

import json
import math
import os
import sys
import time

import cv2
import numpy as np

import camera as C
import kinematics as K
import scene as S
import vision as V

HERE = os.path.dirname(os.path.abspath(__file__))

# The scenes and frames used here were measured with the calibration of 2026-11-10 (before the
# shoulder_lift zero was corrected on 2026-11-11). Pin that calibration: re-calibrating the real arm
# must not change these tests (and they must not depend on it).
os.environ["FD_CONFIG"] = os.path.join(HERE, "testdata", "calib_20261110", "config.json")
os.environ["FD_CAMERA_MODEL"] = os.path.join(HERE, "testdata", "calib_20261110", "camera_model.json")
TD = os.path.join(HERE, "testdata")
META = json.load(open(os.path.join(TD, "frames.json")))
FRAMES = {f["file"]: f for f in META["frames"]}
TRUTH = META["truth"]
CAM = C.CameraModel.load()
try:
    from robot import load_config
    CFG = load_config()
    if CFG.get("table_z") is None:
        raise KeyError("table_z")
except Exception as _e:                                    # robot.py belongs to another owner
    print(f"NOTE: robot.load_config() not usable ({type(_e).__name__}: {_e}); using table_z from testdata")
    CFG = {"table_z": TRUTH["table_z"]}
Z0 = float(CFG["table_z"])

# frames in which the arm stands away from the table (ready pose or parked) and what is loose on the table
ARM_AWAY = {
    "0004_ready_raw.jpg": [("box", "red")],
    "0030_overview_raw.jpg": [("box", "red")],
    "0113_locate_raw.jpg": [("box", "red")],
    "0132_final_overview_raw.jpg": [],
    "0134_cap_scene_raw.jpg": [("cylinder", "white")],
    "0146_cap_final_raw.jpg": [],
    "0148_cube_scene_raw.jpg": [("box", "blue"), ("cylinder", "ivory")],
    "0168_cube_final_raw.jpg": [("cylinder", "ivory")],
}
# contents of the black box (colour multiset) for every arm-away frame
CONTENTS = {
    "0004_ready_raw.jpg": [], "0030_overview_raw.jpg": [], "0113_locate_raw.jpg": [],
    "0132_final_overview_raw.jpg": ["red"], "0134_cap_scene_raw.jpg": ["red"], "0146_cap_final_raw.jpg": ["red", "white"],
    "0148_cube_scene_raw.jpg": ["red", "white"], "0168_cube_final_raw.jpg": ["blue", "red", "white"],
}
# frame with the gripper near an object -> arm-free frame of the same scene
JAW_REF = {"0028": "0030", "0118": "0113", "0120": "0113", "0122": "0113", "0124": "0113", "0126": "0113",
           "0128": "0113", "0130": "0113", "0140": "0134", "0142": "0134", "0144": "0134", "0158": "0148",
           "0160": "0148", "0162": "0148", "0164": "0148", "0166": "0148"}
REPORT = {"accuracy": [], "container": [], "jaws": [], "runtime": {}, "synthetic": {}, "brightness": []}

_img_cache, _scene_cache = {}, {}


def frame_file(prefix: str) -> str:
    return next(f for f in FRAMES if f.startswith(prefix))


def image(name: str, gain: float = 1.0) -> np.ndarray:
    key = (name, gain)
    if key not in _img_cache:
        img = cv2.imread(os.path.join(TD, name))
        assert img is not None, name
        if gain != 1.0:
            img = np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)
        _img_cache[key] = img
    return _img_cache[key]


def arm_state(name: str):
    f = FRAMES[name]
    q = np.radians(f["q_deg"])
    jaw = None if f["gripper_raw"] is None else -11.41 + (f["gripper_raw"] - 2050) * 360.0 / 4096.0
    return q, jaw


def perceive(name: str, gain: float = 1.0):
    key = (name, gain)
    if key not in _scene_cache:
        q, jaw = arm_state(name)
        _scene_cache[key] = S.perceive(image(name, gain), CAM, CFG, arm_q=q, jaw_q_deg=jaw)
    return _scene_cache[key]


def loose(sc):
    return sc.graspable()


def ang_diff_deg(a_deg: float, b_deg: float, period: float = 180.0) -> float:
    return (a_deg - b_deg + period / 2) % period - period / 2


# =========================================================================== synthetic scenes
TABLE_BGR = (75, 75, 140)
SYN_COLORS = {"green": (70, 200, 60), "yellow": (40, 220, 240), "blue": (230, 150, 40), "red": (40, 50, 250),
              "white": (250, 250, 250), "orange": (30, 140, 250), "purple": (200, 60, 150), "black": (45, 45, 45)}


def render(objects, cam=CAM, z0=Z0, ss=2, noise=2.0, seed=0, table=TABLE_BGR, blur=0.8, base=None):
    """Render shaded boxes / cylinders standing on the table through a camera model.
    objects: [{"kind", "x", "y", "yaw", "size", "bgr": colour of the lit top, "shade": (side a, side b)}]."""
    fc = S._FastCam(cam)
    w, h = C.W, C.H
    if base is None:
        img = np.zeros((h * ss, w * ss, 3), np.uint8)
        img[:] = table
    else:
        img = cv2.resize(base, (w * ss, h * ss), interpolation=cv2.INTER_NEAREST)

    def px(pts):
        return np.round(fc.project(pts) * ss * 8 + (ss - 1) * 4).astype(np.int32)

    for o in sorted(objects, key=lambda o: -math.hypot(o["x"] - fc.C[0], o["y"] - fc.C[1])):
        top = np.array(o["bgr"], float)
        sa, sb = o.get("shade", (0.62, 0.48))
        zb = o.get("z_bottom", z0)
        if o["kind"] == "box":
            c = S.box_corners(o["x"], o["y"], o["yaw"], *o["size"], zb)
            ctr = c.mean(axis=0)
            for idx, f in (((0, 1, 5, 4), sa), ((1, 2, 6, 5), sb), ((2, 3, 7, 6), sa), ((3, 0, 4, 7), sb), ((4, 5, 6, 7), 1.0)):
                p = c[list(idx)]
                n = np.cross(p[1] - p[0], p[2] - p[1])
                if np.dot(n, p.mean(axis=0) - ctr) < 0:
                    n = -n
                if np.dot(n, fc.C - p.mean(axis=0)) <= 0:
                    continue
                cv2.fillConvexPoly(img, px(p), tuple(float(v) for v in top * f), lineType=cv2.LINE_AA, shift=3)
        else:
            d, _, hh = o["size"]
            a = np.linspace(0, 2 * math.pi, 72, endpoint=False)
            ring = np.stack([o["x"] + d / 2 * np.cos(a), o["y"] + d / 2 * np.sin(a)], axis=1)
            lo = np.column_stack([ring, np.full(len(a), zb)])
            hi = np.column_stack([ring, np.full(len(a), zb + hh)])
            cv2.fillConvexPoly(img, cv2.convexHull(px(np.vstack([lo, hi]))), tuple(float(v) for v in top * sa),
                               lineType=cv2.LINE_AA, shift=3)
            cv2.fillConvexPoly(img, px(hi), tuple(float(v) for v in top), lineType=cv2.LINE_AA, shift=3)
    out = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    if blur:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    if noise:
        rng = np.random.default_rng(seed)
        out = np.clip(out.astype(np.float32) + rng.normal(0, noise, out.shape), 0, 255).astype(np.uint8)
    return out


def syn_obj(kind, color, x, y, size, yaw_deg=0.0):
    shade = (0.78, 0.66) if color in ("red", "white") else (0.62, 0.48)
    return {"kind": kind, "x": x, "y": y, "yaw": math.radians(yaw_deg), "size": size, "bgr": SYN_COLORS[color],
            "shade": shade, "color": color}


def look_at_camera(center, target, f=1400.0, k1=0.0):
    c, t = np.asarray(center, float), np.asarray(target, float)
    zc = (t - c) / np.linalg.norm(t - c)
    xc = np.cross(zc, [0.0, 0.0, 1.0])
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.vstack([xc, yc, zc])
    rvec, _ = cv2.Rodrigues(R)
    return C.CameraModel(f, C.W / 2.0, C.H / 2.0, rvec, -R @ c, k1)


# =========================================================================== perception on the saved frames
def test_objects_accuracy():
    """Every loose truth object is found with the right kind and colour and within the tolerances."""
    for key in ("red_block", "white_cap", "blue_block", "ivory_cylinder"):
        t = TRUTH[key]
        for name in t["frames"]:
            sc = perceive(name)
            cand = [o for o in loose(sc) if o.kind == t["kind"] and o.color == t["color"]]
            assert len(cand) == 1, f"{key} in {name}: {[(o.kind, o.color) for o in loose(sc)]} {sc.warnings}"
            o = cand[0]
            dc = math.hypot(o.center[0] - t["center"][0], o.center[1] - t["center"][1]) * 1000
            ds = [(a - b) * 1000 for a, b in zip(o.size, t["size"])]
            dyaw = ang_diff_deg(math.degrees(o.yaw), t["long_axis_yaw_deg"]) if "long_axis_yaw_deg" in t else 0.0
            REPORT["accuracy"].append((key, name[:4], o.kind, o.color, dc, ds[0], ds[1], ds[2], dyaw, o.confidence))
            assert dc <= 5.0, f"{key} {name}: centre off by {dc:.1f} mm"
            assert abs(ds[0]) <= 5.0 and abs(ds[1]) <= 5.0, f"{key} {name}: length/width off by {ds[:2]}"
            assert abs(ds[2]) <= 6.0, f"{key} {name}: height off by {ds[2]:.1f} mm"
            assert abs(dyaw) <= 8.0, f"{key} {name}: yaw off by {dyaw:.1f} deg"
            assert o.confidence >= 0.7, f"{key} {name}: confidence {o.confidence}"
            assert o.size[0] >= o.size[1], "length must be >= width"
            assert abs(o.z_bottom - Z0) < 1e-9 and o.in_container is None
            assert len(o.footprint) >= 4 and o.pixel_bbox[2] > 0


def test_boxes_are_normalised():
    """VM-2: length >= width and yaw is the LONG axis, in [0, pi); cylinders have yaw 0."""
    for name in ARM_AWAY:
        for o in perceive(name).objects:
            assert o.size[0] >= o.size[1] - 1e-12, (name, o.kind, o.size)
            assert 0.0 <= o.yaw < math.pi + 1e-9, (name, o.yaw)
            if o.kind == "cylinder":
                assert o.yaw == 0.0 and abs(o.size[0] - o.size[1]) < 1e-12
            if o.kind == "box" and o.in_container is None:
                # the footprint's long edge really points along yaw
                p = np.array(o.footprint)
                e = max((p[(i + 1) % 4] - p[i] for i in range(4)), key=lambda v: float(np.linalg.norm(v)))
                assert abs(ang_diff_deg(math.degrees(math.atan2(e[1], e[0])), math.degrees(o.yaw))) < 1.0


def test_container():
    """The black box: one container, measured rim height, centre / size / orientation, inner geometry."""
    t = TRUTH["black_box"]
    for name in ARM_AWAY:
        sc = perceive(name)
        cs = sc.containers()
        assert len(cs) == 1, f"{name}: {len(cs)} containers; {sc.warnings}"
        c = cs[0]
        assert c.color == "black" and c.kind == "container"
        dc = math.hypot(c.center[0] - t["center"][0], c.center[1] - t["center"][1]) * 1000
        rim = c.size[2] * 1000
        # The truth lists the outer size as (111, 100) with yaw 94 deg; 94 deg is the direction of the
        # 100 mm (far) edge in the reviewer's fit, so the LONG axis is at 94 - 90 = 4 deg.
        dyaw = ang_diff_deg(math.degrees(c.yaw), t["yaw_deg"] - 90.0)
        REPORT["container"].append((name[:4], dc, (c.size[0] - t["outer_size"][0]) * 1000,
                                    (c.size[1] - t["outer_size"][1]) * 1000, rim, dyaw, c.confidence))
        assert dc <= 8.0, f"{name}: container centre off by {dc:.1f} mm"
        assert 24.0 <= rim <= 36.0, f"{name}: rim height {rim:.1f} mm"
        assert any("measured" in n and "NOT measured" not in n for n in c.notes), f"{name}: rim not measured: {c.notes}"
        assert abs(c.size[0] - t["outer_size"][0]) * 1000 <= 6.0 and abs(c.size[1] - t["outer_size"][1]) * 1000 <= 6.0, c.size
        assert abs(dyaw) <= 8.0, f"{name}: container long axis off by {dyaw:.1f} deg"
        assert c.confidence >= 0.8
        inner = c.inner
        assert inner is not None and set(inner) >= {"center", "yaw", "size", "floor_z", "rim_z"}
        assert abs(inner["rim_z"] - (Z0 + c.size[2])) < 1e-9 and Z0 < inner["floor_z"] < inner["rim_z"]
        assert 0.015 <= inner["rim_z"] - inner["floor_z"] <= 0.035          # inner depth about 25 mm
        assert inner["size"][0] < c.size[0] and inner["size"][1] < c.size[1] and inner["size"][0] >= inner["size"][1]
        assert c in sc.containers() and c not in sc.graspable()


def test_rim_height_is_measured_not_assumed():
    """VM-0: the rim height follows the image. Changing the assumed default must not change it, and
    the assumed floor thickness enters one to one."""
    name = "0113_locate_raw.jpg"
    q, jaw = arm_state(name)
    h0 = perceive(name).containers()[0].size[2]
    cfg = dict(CFG, scene={"container_height_mm": 50.0})
    h1 = S.perceive(image(name), CAM, cfg, arm_q=q, jaw_q_deg=jaw).containers()[0].size[2]
    assert abs(h1 - h0) < 1e-6, (h0, h1)
    cfg = dict(CFG, scene={"container_floor_mm": 2.0})
    c2 = S.perceive(image(name), CAM, cfg, arm_q=q, jaw_q_deg=jaw).containers()[0]
    assert abs((h0 - c2.size[2]) - 0.003) < 0.0006, (h0, c2.size[2])
    assert abs((c2.inner["rim_z"] - c2.inner["floor_z"]) - (h0 - 0.005)) < 0.0006
    # a rim that cannot be measured is said so (no wall/floor step: the whole interior painted uniformly dark)
    img = image(name).copy()
    cv2.fillConvexPoly(img, np.array([[964, 712], [1298, 712], [1396, 984], [966, 984]], np.int32), (26, 23, 25))
    c3 = S.perceive(img, CAM, CFG, arm_q=q, jaw_q_deg=jaw).containers()
    assert len(c3) == 1 and c3[0].confidence <= 0.5 and any("NOT measured" in n for n in c3[0].notes), c3[0].notes
    assert abs(c3[0].size[2] - 0.030) < 1e-9


def test_contents_flagged_in_container():
    """Things inside the box: right colours, flagged in_container, never offered as graspable."""
    for name, colours in CONTENTS.items():
        sc = perceive(name)
        cid = sc.containers()[0].id
        inside = [o for o in sc.objects if o.in_container is not None]
        assert sorted(o.color for o in inside) == sorted(colours), f"{name}: {[o.color for o in inside]} != {colours}"
        for o in inside:
            assert o.in_container == cid and o not in sc.graspable()
            assert o.confidence <= 0.5 and any("inside container" in n for n in o.notes)
            assert abs(o.z_bottom - sc.containers()[0].inner["floor_z"]) < 1e-9
            # its (approximate) centre lies inside the opening
            inner = sc.containers()[0].inner
            d = np.array(o.center) - np.array(inner["center"])
            ca, sa = math.cos(inner["yaw"]), math.sin(inner["yaw"])
            assert abs(d[0] * ca + d[1] * sa) <= inner["size"][0] / 2 + 1e-6
            assert abs(-d[0] * sa + d[1] * ca) <= inner["size"][1] / 2 + 1e-6
            assert o.z_top <= inner["rim_z"] + 1e-6
        for colour in set(colours):
            if colour not in [k[1] for k in ARM_AWAY[name]]:
                try:
                    S.find_object(sc, {"color": colour})
                    raise AssertionError(f"{name}: a {colour} thing inside the box was offered as a target")
                except LookupError:
                    pass
            assert S.find_object(sc, {"color": colour, "in_container": True}).in_container == cid


def test_only_real_objects_reported():
    """Nothing is reported for the robot, cables, the glare patch, reflections or the box's outer wall."""
    for name, expected in ARM_AWAY.items():
        sc = perceive(name)
        got = sorted((o.kind, o.color) for o in loose(sc))
        assert got == sorted(expected), f"{name}: loose objects {got}, expected {sorted(expected)}; {sc.warnings}"
        assert not sc.warnings, f"{name}: unexpected warnings {sc.warnings}"
    truth_pos = [(TRUTH[k]["kind"], TRUTH[k]["color"], TRUTH[k]["center"]) for k in
                 ("red_block", "white_cap", "blue_block", "ivory_cylinder")]
    for name in FRAMES:                         # every frame, also with the arm over the table
        sc = perceive(name)
        for o in loose(sc):
            near = [t for t in truth_pos if t[0] == o.kind and t[1] == o.color
                    and math.hypot(o.center[0] - t[2][0], o.center[1] - t[2][1]) < 0.008]
            assert near, f"{name}: spurious object {o.kind} {o.color} at {np.round(o.center, 3)} {o.notes}"
        assert len(sc.containers()) <= 1


def test_arm_is_masked():
    """With the joint angles the arm is not an object; the mask covers the gripper in every frame."""
    fc = S._FastCam(CAM)
    for name in FRAMES:
        q, jaw = arm_state(name)
        m = S.arm_mask((C.H // 2, C.W // 2), CAM, q, jaw, scale=0.5)
        model = S.jaw_model(q, jaw if jaw is not None else 0.0)
        for part in ("fixed", "moving"):
            uv = fc.project(model[part]["pts"]) * 0.5
            inside = [(0 <= u < C.W // 2 and 0 <= v < C.H // 2) for u, v in uv]
            vals = [m[int(v), int(u)] for (u, v), ok in zip(uv, inside) if ok]
            assert all(vals), f"{name}: {part} jaw sticks out of the arm mask"
    # parked arm with unknown jaw angle: still nothing but the cap
    name = "0134_cap_scene_raw.jpg"
    q, _ = arm_state(name)
    sc = S.perceive(image(name), CAM, CFG, arm_q=q, jaw_q_deg=None)
    assert sorted((o.kind, o.color) for o in loose(sc)) == [("cylinder", "white")]


def test_container_seen_with_arm_waiting_above_it():
    """skills.py lets the arm wait straight above the container (fingertip 65 mm over the rim) and
    looks again from there: the container, its rim height and its contents must still come out."""
    name = "0168_cube_final_raw.jpg"
    ref = perceive(name)
    box = ref.containers()[0]
    tool = K.tool_offset(-0.001, 0.0, 0.0068)
    for dz, roll, jaw in ((0.065, 0.0, -3.0), (0.065, -65.0, -3.0), (0.05, 0.0, -3.0), (0.065, -65.0, 30.0)):
        q, info = K.ik((box.inner["center"][0], box.inner["center"][1], box.inner["rim_z"] + dz), 70.0,
                       math.radians(roll), tool=tool)
        assert info["ok"], info
        sc = S.perceive(image(name), CAM, CFG, arm_q=q, jaw_q_deg=jaw)
        cs = sc.containers()
        assert len(cs) == 1, (dz, roll, jaw, sc.warnings)
        d = math.hypot(cs[0].center[0] - box.center[0], cs[0].center[1] - box.center[1])
        assert d < 0.003 and abs(cs[0].size[2] - box.size[2]) < 0.002, (dz, roll, jaw, d, cs[0].size, cs[0].notes)
        assert sorted(o.color for o in sc.objects if o.in_container) == ["blue", "red", "white"], (dz, roll, jaw)
        assert [(o.kind, o.color) for o in loose(sc)] == [("cylinder", "ivory")]
        inside = S.objects_in_container(image(name), CAM, CFG, cs[0], arm_q=q, jaw_q_deg=jaw)
        assert sorted(b["color"] for b in inside) == ["blue", "red", "white"]
    # jaws wide open towards the camera reach in front of a corner of the box: either it is still
    # found correctly or it is refused with the reason, never reported at a wrong place
    q, info = K.ik((box.inner["center"][0], box.inner["center"][1], box.inner["rim_z"] + 0.065), 70.0, 0.0, tool=tool)
    sc = S.perceive(image(name), CAM, CFG, arm_q=q, jaw_q_deg=45.0)
    if sc.containers():
        c = sc.containers()[0]
        assert math.hypot(c.center[0] - box.center[0], c.center[1] - box.center[1]) < 0.004, c.center
    else:
        assert any("hidden by the robot arm" in w for w in sc.warnings), sc.warnings
    # the arm over a CORNER of the box: the box is not guessed, the warning says why
    corner = S.box_corners(box.center[0], box.center[1], box.yaw, box.size[0], box.size[1], 0.0, 0.0)[2]
    q, info = K.ik((corner[0], corner[1], box.inner["rim_z"] + 0.02), 70.0, 0.0, tool=tool)
    if info["ok"]:
        sc = S.perceive(image(name), CAM, CFG, arm_q=q, jaw_q_deg=30.0)
        assert not sc.containers() and any("hidden by the robot arm" in w for w in sc.warnings), sc.warnings


def test_brightness_robustness():
    """VM-1: a 5 / 10 % brighter (and darker) image moves no centre by more than 4 mm and changes no
    kind or colour (the thresholds are relative to the table brightness of the frame)."""
    worst = 0.0
    for name in ("0113_locate_raw.jpg", "0134_cap_scene_raw.jpg", "0148_cube_scene_raw.jpg", "0168_cube_final_raw.jpg"):
        base = perceive(name)
        ref = {(o.kind, o.color): o for o in base.objects if o.in_container is None}
        ref_in = sorted(o.color for o in base.objects if o.in_container is not None)
        for gain in (1.05, 1.10, 0.95, 0.90):
            sc = perceive(name, gain)
            got = {(o.kind, o.color): o for o in sc.objects if o.in_container is None}
            assert set(got) == set(ref), f"{name} x{gain}: {sorted(got)} != {sorted(ref)}"
            assert sorted(o.color for o in sc.objects if o.in_container is not None) == ref_in, f"{name} x{gain}"
            for k, o in ref.items():
                d = math.hypot(got[k].center[0] - o.center[0], got[k].center[1] - o.center[1]) * 1000
                dh = abs(got[k].size[2] - o.size[2]) * 1000
                worst = max(worst, d)
                REPORT["brightness"].append((name[:4], gain, k[1], k[0], d, dh))
                assert d <= 4.0, f"{name} x{gain}: {k} moved {d:.1f} mm"
                assert dh <= 3.0, f"{name} x{gain}: {k} height changed {dh:.1f} mm"
    REPORT["brightness_worst_mm"] = worst


def test_reflection_safety_net():
    """VM-1, second line of defence: if the mirror reflection under the red block does get into its
    mask (emulated by narrowing the table colour family), it is cut off at the dark contact line and
    the result stays the same; without a leak nothing is cut."""
    name = "0113_locate_raw.jpg"
    q, jaw = arm_state(name)
    ref = [o for o in loose(perceive(name)) if o.color == "red"][0]
    assert not any("mirror reflection" in n for n in ref.notes)
    for gain in (1.0, 1.1):
        sc = S.perceive(image(name, gain), CAM, dict(CFG, scene={"table_s_max": 170}), arm_q=q, jaw_q_deg=jaw)
        got = [o for o in loose(sc) if o.color == "red"]
        assert len(got) == 1 and got[0].kind == "box", [(o.kind, o.color) for o in loose(sc)]
        o = got[0]
        assert any("mirror reflection" in n for n in o.notes) and o.confidence <= 0.85, o.notes
        assert math.hypot(o.center[0] - ref.center[0], o.center[1] - ref.center[1]) < 0.0015, (gain, o.center, ref.center)
        assert abs(o.size[2] - ref.size[2]) < 0.0015, (gain, o.size, ref.size)
    for fname in ARM_AWAY:                       # the normal frames: never triggered
        for o in perceive(fname).objects:
            assert not any("mirror reflection" in n for n in o.notes), (fname, o.color, o.notes)


def test_runtime():
    """<= 0.35 s per frame on this machine (best of 3 runs per frame; other jobs share the CPU tonight)."""
    S.perceive(image("0113_locate_raw.jpg"), CAM, CFG, arm_q=arm_state("0113_locate_raw.jpg")[0])   # warm up
    best, all_runs = {}, []
    for name in FRAMES:
        q, jaw = arm_state(name)
        img = image(name)
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            S.perceive(img, CAM, CFG, arm_q=q, jaw_q_deg=jaw)
            ts.append(time.perf_counter() - t0)
        if min(ts) > 0.25:                       # disturbed by another job: look twice more
            for _ in range(4):
                t0 = time.perf_counter()
                S.perceive(img, CAM, CFG, arm_q=q, jaw_q_deg=jaw)
                ts.append(time.perf_counter() - t0)
        best[name] = min(ts)
        all_runs += ts
    REPORT["runtime"] = {"median_best_ms": 1000 * float(np.median(list(best.values()))),
                         "max_best_ms": 1000 * max(best.values()), "worst_frame": max(best, key=best.get)[:4],
                         "median_all_ms": 1000 * float(np.median(all_runs)), "max_all_ms": 1000 * max(all_runs)}
    assert max(best.values()) <= 0.35, REPORT["runtime"]
    sc = perceive("0148_cube_scene_raw.jpg")
    assert set(sc.timings) >= {"segment", "containers", "objects", "contents", "total"} and sc.stamp > 0


def test_refusals_are_explicit():
    """Bad input is refused with the reason; a frame without a visible table yields a warning, not objects."""
    sc = S.perceive(np.zeros((C.H, C.W, 3), np.uint8), CAM, CFG)
    assert not sc.objects and len(sc.warnings) == 1 and "table was not recognised" in sc.warnings[0]
    gray = np.full((C.H, C.W, 3), 128, np.uint8)
    sc = S.perceive(gray, CAM, CFG)
    assert not sc.objects and "table was not recognised" in sc.warnings[0]
    for bad, cfg, exc in ((np.zeros((720, 1280, 3), np.uint8), CFG, "camera model"), (None, CFG, "BGR"),
                          (image("0113_locate_raw.jpg"), {}, "table_z"),
                          (image("0113_locate_raw.jpg"), dict(CFG, scene={"no_such_key": 1}), "unknown keys")):
        try:
            S.perceive(bad, CAM, cfg)
            raise AssertionError(f"no error for {exc}")
        except ValueError as e:
            assert exc in str(e), e


def test_exposure_warnings():
    """A frame that is much brighter / darker than the locked exposure expects is flagged."""
    name = "0113_locate_raw.jpg"
    q, jaw = arm_state(name)
    for gain, word in ((1.35, "too bright"), (0.45, "dark")):
        img = np.clip(image(name).astype(np.float32) * gain, 0, 255).astype(np.uint8)
        sc = S.perceive(img, CAM, CFG, arm_q=q, jaw_q_deg=jaw)
        assert any(word in w for w in sc.warnings), (gain, sc.warnings)
    for gain in (0.8, 1.2):                       # the tested range stays quiet and correct
        sc = perceive(name, gain)
        assert not sc.warnings and [(o.kind, o.color) for o in loose(sc)] == [("box", "red")], (gain, sc.warnings)
    # bluer light (daylight under the locked white balance): still recognised, same place
    img = image(name).astype(np.float32)
    img[..., 0] *= 1.25
    img[..., 2] *= 0.95
    sc = S.perceive(np.clip(img, 0, 255).astype(np.uint8), CAM, CFG, arm_q=q, jaw_q_deg=jaw)
    ref = loose(perceive(name))[0]
    got = [o for o in loose(sc) if o.kind == "box"]
    assert len(got) == 1 and math.hypot(got[0].center[0] - ref.center[0], got[0].center[1] - ref.center[1]) < 0.004, \
        ([(o.kind, o.color, o.notes) for o in sc.objects], sc.warnings)
    assert len(sc.containers()) == 1


def test_annotate():
    name = "0148_cube_scene_raw.jpg"
    img = image(name)
    before = img.copy()
    sc = perceive(name)
    vis = S.annotate(img, sc, CAM)
    assert vis.shape == img.shape and vis.dtype == np.uint8 and np.array_equal(img, before)
    assert np.count_nonzero(np.any(vis != img, axis=2)) > 3000
    out = os.path.join(TD, "_annotated_0148.jpg")
    assert cv2.imwrite(out, vis, [cv2.IMWRITE_JPEG_QUALITY, 85])
    S.annotate(img, S.Scene(objects=[], table_z=Z0, warnings=["x"]), CAM)       # empty scene is fine


def test_find_object():
    sc = S.perceive(image("0148_cube_scene_raw.jpg"), CAM, CFG, arm_q=arm_state("0148_cube_scene_raw.jpg")[0])
    blue = S.find_object(sc, {"color": "blue", "kind": "box"})
    assert blue.color == "blue" and blue.kind == "box" and S.find_object(sc, {"color": "blue", "kind": "cube"}) is blue
    assert S.find_object(sc, {"id": blue.id}) is blue
    box = S.find_object(sc, {"color": "black", "kind": "container"})
    assert box.kind == "container" and S.find_object(sc, {"kind": "container"}) is box
    assert S.find_object(sc, {"kind": "cylinder"}).color == "ivory"
    assert S.find_object(sc, {"nearest_to": (0.275, -0.02)}).color == "ivory"
    assert S.find_object(sc, {"nearest_to": (0.28, -0.07)}) is blue
    assert S.find_object(sc, {"largest": True}) is blue
    assert S.find_object(sc, {"color": None, "kind": None, "id": blue.id}) is blue          # None values are ignored
    n_warn = len(sc.warnings)
    first = S.find_object(sc, {})                # two loose objects: deterministic choice, recorded
    assert first.in_container is None and first.kind != "container" and len(sc.warnings) == n_warn + 1
    assert "2 objects match" in sc.warnings[-1]
    for q, word in (({"color": "green"}, "nothing matches"), ({"kind": "sphere"}, "unknown kind"), ({"id": 99}, "no object with id"),
                    ({"color": "red"}, "nothing matches"), ({"colour": "red"}, "unknown query key"),
                    ({"id": blue.id, "color": "red"}, "is blue"), ({"id": blue.id, "kind": "cylinder"}, "is a box"),
                    ({"nearest_to": "here"}, "nearest_to"), ({"color": "black"}, "nothing matches")):
        try:
            S.find_object(sc, q)
            raise AssertionError(f"no LookupError for {q}")
        except LookupError as e:
            assert word in str(e), (q, str(e))
    assert "#" in str(_lookup_text(sc, {"color": "green"}))       # the message lists what is there
    red_in = S.find_object(sc, {"color": "red", "in_container": True})
    assert red_in.in_container == box.id


def _lookup_text(sc, q):
    try:
        S.find_object(sc, q)
    except LookupError as e:
        return str(e)
    return ""


def test_object_present():
    """Still there / gone / hidden by the arm."""
    red = S.find_object(perceive("0113_locate_raw.jpg"), {"color": "red", "kind": "box"})
    blue = S.find_object(perceive("0148_cube_scene_raw.jpg"), {"color": "blue"})
    ivory = S.find_object(perceive("0148_cube_scene_raw.jpg"), {"color": "ivory"})
    cap = S.find_object(perceive("0134_cap_scene_raw.jpg"), {"color": "white"})

    def present(obj, name):
        q, jaw = arm_state(name)
        return S.object_presence(image(name), CAM, CFG, obj, arm_q=q, jaw_q_deg=jaw)

    assert present(red, "0030_overview_raw.jpg")["present"] is True
    assert present(red, "0132_final_overview_raw.jpg")["present"] is False        # it is in the box now
    assert present(cap, "0146_cap_final_raw.jpg")["present"] is False
    assert present(blue, "0168_cube_final_raw.jpg")["present"] is False
    assert present(ivory, "0168_cube_final_raw.jpg")["present"] is True
    r = present(blue, "0160_cube_at_grasp_raw.jpg")                              # the gripper is around it
    assert r["present"] is None and "hidden" in r["reason"] and r["visible_fraction"] < 0.5
    q, _ = arm_state("0160_cube_at_grasp_raw.jpg")
    assert S.object_present(image("0160_cube_at_grasp_raw.jpg"), CAM, CFG, blue, arm_q=q) is True   # cannot tell -> not "gone"
    q, _ = arm_state("0168_cube_final_raw.jpg")
    assert S.object_present(image("0168_cube_final_raw.jpg"), CAM, CFG, blue, arm_q=q) is False
    assert S.object_present(image("0148_cube_scene_raw.jpg"), CAM, CFG, blue) is True
    # something inside the box
    sc = perceive("0148_cube_scene_raw.jpg")
    red_in = S.find_object(sc, {"color": "red", "in_container": True})
    assert S.object_present(image("0168_cube_final_raw.jpg"), CAM, CFG, red_in) is True
    assert S.object_present(image("0113_locate_raw.jpg"), CAM, CFG, red_in) is False


def test_objects_in_container():
    """What the verify step of a pick relies on: the count of the placed colour goes up."""
    def inside(name, box_from):
        box = perceive(box_from).containers()[0]
        return S.objects_in_container(image(name), CAM, CFG, box, arm_q=arm_state(name)[0])

    for name, colours in CONTENTS.items():
        got = inside(name, name)
        assert sorted(b["color"] for b in got) == sorted(colours), (name, [b["color"] for b in got])
        for b in got:
            assert set(b) >= {"color", "center", "pixel_bbox", "area_px", "kind", "size", "near_arm"} and b["area_px"] > 800
            assert b["near_arm"] is False                               # the arm is away in these frames
    # Real frames with the real gripper above / in the box. With the arm pose the gripper and what it
    # holds are masked, and whatever remains next to the mask is flagged near_arm (partly hidden, or
    # still held: frame 0166 shows the edge of the blue block that is still in the jaws). Without the
    # arm pose the white gripper and the held object come back as contents: callers must pass arm_q.
    box113 = perceive("0113_locate_raw.jpg").containers()[0]
    box148 = perceive("0148_cube_scene_raw.jpg").containers()[0]

    def seen(name, box, with_arm):
        q, jaw = arm_state(name)
        img = image(name)
        if with_arm:
            return S.objects_in_container(img, CAM, CFG, box, arm_q=q, jaw_q_deg=jaw)
        return S.objects_in_container(img, CAM, CFG, box)

    held = seen("0128_in_box_pre_release_raw.jpg", box113, True)        # red block still held inside the box
    assert held == [], [(b["color"], b["area_px"]) for b in held]
    assert "white" in [b["color"] for b in seen("0128_in_box_pre_release_raw.jpg", box113, False)]
    rel = seen("0130_released_raw.jpg", box113, True)                   # released, gripper still right above it
    assert [b["color"] for b in rel] == ["red"] and rel[0]["near_arm"] and any("robot arm" in n for n in rel[0]["notes"])
    pre = seen("0166_cube_pre_release_raw.jpg", box148, True)           # blue block still held above the box
    assert all(b["near_arm"] for b in pre if b["color"] == "blue"), [(b["color"], b["area_px"], b["near_arm"]) for b in pre]
    assert sum(b["area_px"] for b in pre if b["color"] == "blue") < 0.2 * 15000      # most of it is masked
    assert not any(b["near_arm"] for b in seen("0166_cube_pre_release_raw.jpg", box148, False))
    # the stand-by pose above the box (skills.py verifies from there): things lying on the floor stay clear
    box168 = perceive("0168_cube_final_raw.jpg").containers()[0]
    tool = K.tool_offset(-0.001, 0.0, 0.0068)
    for dz, roll, jaw in ((0.065, 0.0, 30.0), (0.065, -65.0, 30.0), (0.065, 90.0, -3.0), (0.04, 0.0, 30.0)):
        q, info = K.ik((box168.inner["center"][0], box168.inner["center"][1], box168.inner["rim_z"] + dz), 70.0,
                       math.radians(roll), tool=tool)
        assert info["ok"], info
        got = S.objects_in_container(image("0168_cube_final_raw.jpg"), CAM, CFG, box168, arm_q=q, jaw_q_deg=jaw)
        assert sorted(b["color"] for b in got) == ["blue", "red", "white"] and not any(b["near_arm"] for b in got), \
            (dz, roll, jaw, [(b["color"], b["near_arm"]) for b in got])
    for before, after, colour in (("0113_locate_raw.jpg", "0132_final_overview_raw.jpg", "red"),
                                  ("0134_cap_scene_raw.jpg", "0146_cap_final_raw.jpg", "white"),
                                  ("0148_cube_scene_raw.jpg", "0168_cube_final_raw.jpg", "blue")):
        n0 = sum(b["color"] == colour for b in inside(before, before))
        n1 = sum(b["color"] == colour for b in inside(after, before))       # same container object, later frame
        assert n1 == n0 + 1, (before, after, colour, n0, n1)
    try:
        S.objects_in_container(image("0113_locate_raw.jpg"), CAM, CFG, loose(perceive("0113_locate_raw.jpg"))[0])
        raise AssertionError("a loose object was accepted as a container")
    except ValueError as e:
        assert "container" in str(e)


def check_scene_wellformed(sc):
    """Every field has the documented type and range, and the whole thing serialises to JSON."""
    ids = [o.id for o in sc.objects]
    assert len(set(ids)) == len(ids)
    cont_ids = {o.id for o in sc.objects if o.kind == "container"}
    for o in sc.objects:
        assert o.kind in ("box", "cylinder", "container", "unknown"), o.kind
        assert o.color in S.COLOR_NAMES, o.color
        assert len(o.center) == 2 and all(isinstance(v, float) and math.isfinite(v) for v in o.center), o.center
        assert len(o.size) == 3 and all(isinstance(v, float) and math.isfinite(v) and v >= 0 for v in o.size), o.size
        assert o.size[0] >= o.size[1] - 1e-12 and isinstance(o.yaw, float) and math.isfinite(o.yaw)
        assert isinstance(o.z_bottom, float) and 0.0 <= o.confidence <= 1.0
        assert len(o.footprint) >= 3 and all(len(p) == 2 and all(math.isfinite(v) for v in p) for p in o.footprint)
        area = sum(o.footprint[i][0] * o.footprint[(i + 1) % len(o.footprint)][1]
                   - o.footprint[(i + 1) % len(o.footprint)][0] * o.footprint[i][1] for i in range(len(o.footprint)))
        assert area >= -1e-12, "footprint must be counter-clockwise"
        assert len(o.pixel_bbox) == 4 and all(isinstance(v, int) for v in o.pixel_bbox)
        assert isinstance(o.notes, list) and all(isinstance(n, str) for n in o.notes)
        assert o.in_container is None or o.in_container in cont_ids
        if o.kind == "container":
            assert isinstance(o.inner, dict) and o.in_container is None
        else:
            assert o.inner is None
    assert all(isinstance(w, str) for w in sc.warnings)
    json.dumps(sc.to_dict())


def test_scenes_are_wellformed_and_json():
    for name in FRAMES:
        check_scene_wellformed(perceive(name))
    name, ref = frame_file("0160"), frame_file("0148")
    q, jaw = arm_state(name)
    json.dumps(S.detect_jaws(image(name), image(ref), CAM, q, jaw, CFG))
    json.dumps(S.detect_jaws(image(frame_file("0118")), image(frame_file("0113")), CAM, *arm_state(frame_file("0118")), CFG))
    sc = perceive("0168_cube_final_raw.jpg")
    json.dumps(S.objects_in_container(image("0168_cube_final_raw.jpg"), CAM, CFG, sc.containers()[0]))
    json.dumps(S.object_presence(image("0168_cube_final_raw.jpg"), CAM, CFG, loose(sc)[0]))


def test_fuzz_no_crash():
    """Random clutter (odd sizes, shapes cut by borders, overlapping things, random arm poses): never an
    exception, always a well-formed scene."""
    rng = np.random.default_rng(7)
    names = list(SYN_COLORS)
    bases = [None, image("0132_final_overview_raw.jpg"), image("0113_locate_raw.jpg")]
    for i in range(24):
        objs = []
        for _ in range(int(rng.integers(1, 6))):
            kind = "box" if rng.random() < 0.6 else "cylinder"
            a, b, c = sorted(rng.uniform(0.004, 0.09, 3))
            size = (c, a, b) if kind == "box" else (c, c, a)
            objs.append(syn_obj(kind, names[int(rng.integers(len(names)))], float(rng.uniform(0.08, 0.36)),
                                float(rng.uniform(-0.28, 0.28)), size, float(rng.uniform(0, 180))))
        base = bases[i % 3]
        img = render(objs, base=base, seed=i, noise=0.0 if base is not None else 2.0)
        q = None if i % 4 == 0 else np.radians([rng.uniform(-40, 40), rng.uniform(-90, 40), rng.uniform(-40, 80),
                                                rng.uniform(20, 95), rng.uniform(-90, 90)])
        jaw = None if i % 3 == 0 else float(rng.uniform(-11, 60))
        sc = S.perceive(img, CAM, CFG, arm_q=q, jaw_q_deg=jaw)
        check_scene_wellformed(sc)
        S.annotate(img, sc, CAM)
        for o in sc.objects[:3]:
            S.object_presence(img, CAM, CFG, o, arm_q=q, jaw_q_deg=jaw)
        for c in sc.containers():
            S.objects_in_container(img, CAM, CFG, c, arm_q=q, jaw_q_deg=jaw)
        if q is not None:
            r = S.detect_jaws(img, bases[1], CAM, q, jaw if jaw is not None else 10.0, CFG)
            assert isinstance(r["ok"], bool) and (r["dxy_base"] is None or all(math.isfinite(v) for v in r["dxy_base"]))
    # pure noise and flat colours
    for img in (rng.integers(0, 255, (C.H, C.W, 3), dtype=np.uint8), np.full((C.H, C.W, 3), 255, np.uint8),
                np.tile(np.array(TABLE_BGR, np.uint8), (C.H, C.W, 1))):
        check_scene_wellformed(S.perceive(img, CAM, CFG))


# =========================================================================== synthetic generalisation
def test_synthetic_boxes_and_cylinders():
    """Other colours, sizes (15..60 mm), orientations and places: right kind / colour, tight accuracy."""
    cases = []
    sizes = ((0.040, 0.023, 0.018), (0.025, 0.025, 0.025), (0.060, 0.030, 0.020), (0.020, 0.015, 0.015), (0.050, 0.020, 0.040))
    places = ((0.20, -0.10), (0.25, 0.0), (0.28, -0.07), (0.18, 0.12))
    i = 0
    for si, size in enumerate(sizes):
        for yaw in (0, 20, 45, 70, 90, 110, 135, 160):
            cases.append(("box", places[i % 4], size, yaw))
            i += 1
    for d, hh in ((0.030, 0.015), (0.026, 0.022), (0.020, 0.040), (0.050, 0.020), (0.015, 0.015)):
        for pl in ((0.20, -0.10), (0.26, 0.02), (0.18, 0.12)):
            cases.append(("cylinder", pl, (d, d, hh), 0))
    names = list(SYN_COLORS)
    errs = []
    for i, (kind, (x, y), size, yaw) in enumerate(cases):
        colour = names[i % len(names)]
        img = render([syn_obj(kind, colour, x, y, size, yaw)], seed=i)
        sc = S.perceive(img, CAM, {"table_z": Z0})
        objs = loose(sc)
        assert len(objs) == 1 and not sc.containers(), f"case {i} {kind} {colour}: {[(o.kind, o.color) for o in sc.objects]} {sc.warnings}"
        o = objs[0]
        want = colour if colour in S.COLOR_NAMES else "other"
        assert o.kind == kind and o.color == want, f"case {i}: {o.kind} {o.color} for {kind} {colour} {o.notes}"
        if want == "other":
            assert any(n.startswith("colour: " + colour) for n in o.notes), o.notes
            assert S.find_object(sc, {"color": colour}) is o            # finer names are searchable
        dc = math.hypot(o.center[0] - x, o.center[1] - y) * 1000
        ds = [abs(a - b) * 1000 for a, b in zip(o.size, size)]
        dyaw = abs(ang_diff_deg(math.degrees(o.yaw), yaw)) if kind == "box" and size[0] - size[1] > 0.004 else 0.0
        errs.append((dc, max(ds[:2]), ds[2], dyaw))
        assert dc <= 2.0 and max(ds[:2]) <= 2.5 and ds[2] <= 2.5 and dyaw <= 3.0, \
            f"case {i} {kind} {colour} size {size} yaw {yaw} at ({x}, {y}): dc {dc:.2f} ds {ds} dyaw {dyaw:.1f}"
        assert o.confidence >= 0.7, (i, o.confidence, o.notes)
    e = np.array(errs)
    REPORT["synthetic"] = {"n": len(e), "centre_max": float(e[:, 0].max()), "centre_med": float(np.median(e[:, 0])),
                           "size_max": float(e[:, 1].max()), "height_max": float(e[:, 2].max()), "yaw_max": float(e[:, 3].max())}


def test_synthetic_several_objects_and_real_background():
    """Three new objects at once on the real table image (glare, cables, the real box all present)."""
    base = image("0132_final_overview_raw.jpg")           # table with the box, nothing loose
    objs = [syn_obj("box", "green", 0.22, -0.12, (0.035, 0.020, 0.020), 30.0),
            syn_obj("cylinder", "yellow", 0.20, -0.02, (0.028, 0.028, 0.030)),
            syn_obj("box", "purple", 0.29, -0.03, (0.030, 0.030, 0.015), 60.0)]
    img = render(objs, base=base, noise=0.0, blur=0.6)
    q, jaw = arm_state("0132_final_overview_raw.jpg")
    sc = S.perceive(img, CAM, CFG, arm_q=q, jaw_q_deg=jaw)
    got = loose(sc)
    assert len(got) == 3, [(o.kind, o.color, o.notes) for o in got]
    for t in objs:
        o = min(got, key=lambda o: math.hypot(o.center[0] - t["x"], o.center[1] - t["y"]))
        assert o.kind == t["kind"], (t["color"], o.kind, o.notes)
        assert math.hypot(o.center[0] - t["x"], o.center[1] - t["y"]) < 0.003, (t["color"], o.center)
        assert all(abs(a - b) < 0.003 for a, b in zip(o.size, t["size"])), (t["color"], o.size)
    assert len(sc.containers()) == 1 and [o.color for o in sc.objects if o.in_container] == ["red"]
    ids = [o.id for o in sc.objects]
    assert len(set(ids)) == len(ids) and all(sc.by_id(i).id == i for i in ids)


def test_synthetic_hard_cases_are_flagged_not_guessed():
    """What cannot be modelled is returned with low confidence and a note, never as a normal object."""
    # 1. a solid black block is a loose object, not a container
    img = render([syn_obj("box", "black", 0.18, 0.12, (0.05, 0.02, 0.04), 70.0)], seed=1)
    sc = S.perceive(img, CAM, {"table_z": Z0})
    assert not sc.containers() and len(loose(sc)) == 1 and loose(sc)[0].color == "black" and loose(sc)[0].kind == "box"
    # 2. a flat sheet (2 mm): reported, but flagged as too flat to grasp
    img = render([syn_obj("box", "green", 0.24, -0.05, (0.05, 0.03, 0.002), 20.0)], seed=2)
    sc = S.perceive(img, CAM, {"table_z": Z0})
    assert len(loose(sc)) == 1, [(o.kind, o.notes) for o in sc.objects]
    o = loose(sc)[0]
    assert o.confidence <= 0.45 and (o.kind == "unknown" or any("flat" in n for n in o.notes)), (o.kind, o.confidence, o.notes)
    # 3. a red block whose side faces look like the table's own reflections: only its top is seen.
    #    It must not come back as a confident object at a wrong place.
    dark_red = dict(syn_obj("box", "red", 0.24, -0.05, (0.04, 0.023, 0.018), 97.0), bgr=(50, 60, 240), shade=(0.62, 0.48))
    sc = S.perceive(render([dark_red], seed=3), CAM, {"table_z": Z0})
    for o in loose(sc):
        assert o.confidence <= 0.45 and o.notes, (o.kind, o.confidence, o.notes)
    # 4. two objects of different colour, one in front of the other (they overlap in the image): they are
    #    taken apart by colour; the front one is measured, the hidden one is flagged, nothing is merged
    two = [syn_obj("box", "blue", 0.230, -0.050, (0.030, 0.020, 0.020), 0.0),
           syn_obj("box", "yellow", 0.262, -0.052, (0.030, 0.020, 0.020), 0.0)]
    sc = S.perceive(render(two, seed=4), CAM, {"table_z": Z0})
    got = {o.color: o for o in loose(sc)}
    assert set(got) == {"blue", "yellow"}, [(o.kind, o.color, o.notes) for o in loose(sc)]
    front, back = got["yellow"], got["blue"]
    assert front.kind == "box" and math.hypot(front.center[0] - 0.262, front.center[1] + 0.052) < 0.003, (front.kind, front.center)
    assert all(abs(a - b) < 0.003 for a, b in zip(front.size, (0.030, 0.020, 0.020))) and front.confidence <= 0.7
    assert any("touches another object" in n for n in front.notes)
    assert back.kind == "unknown" and back.confidence <= 0.2 and any("hides its base" in n for n in back.notes), back.notes
    # 4a. the same two of ONE colour cannot be taken apart: one low-confidence unknown, not a box
    same = [dict(two[0], bgr=SYN_COLORS["green"]), dict(two[1], bgr=SYN_COLORS["green"])]
    sc = S.perceive(render(same, seed=4), CAM, {"table_z": Z0})
    for o in loose(sc):
        assert o.confidence <= 0.5 or o.kind == "unknown", (o.kind, o.color, o.size, o.confidence, o.notes)
    # 4b. an object standing right behind the box: its base is hidden by the far wall
    base = image("0132_final_overview_raw.jpg")
    img = render([syn_obj("box", "green", 0.175, 0.11, (0.03, 0.03, 0.05), 0.0)], base=base, noise=0.0, blur=0.6)
    q, jaw = arm_state("0132_final_overview_raw.jpg")
    sc = S.perceive(img, CAM, CFG, arm_q=q, jaw_q_deg=jaw)
    assert len(sc.containers()) == 1 and len(loose(sc)) == 1, (sc.warnings, [(o.kind, o.color) for o in sc.objects])
    o = loose(sc)[0]
    assert o.kind == "unknown" and o.confidence <= 0.2 and any("hidden behind a container" in n for n in o.notes), \
        (o.kind, o.confidence, o.notes)
    # 5. an object cut by the lower image border is not reported, and that is said
    img = render([syn_obj("cylinder", "blue", 0.315, 0.0, (0.05, 0.05, 0.02))], seed=5)
    sc = S.perceive(img, CAM, {"table_z": Z0})
    assert not loose(sc) and any("lower image border" in w for w in sc.warnings), sc.warnings
    # 6. something much larger than the gripper: either only a warning, or an object marked as too large
    for size, at in (((0.15, 0.11, 0.02), (0.19, -0.02)), ((0.24, 0.18, 0.03), (0.18, -0.01))):
        img = render([syn_obj("box", "green", at[0], at[1], size, 10.0)], seed=6)
        sc = S.perceive(img, CAM, {"table_z": Z0})
        if loose(sc):
            assert all(o.confidence <= 0.3 and any("larger than the gripper" in n for n in o.notes) for o in loose(sc)), \
                [(o.kind, o.size, o.confidence, o.notes) for o in loose(sc)]
        else:
            assert any("too big" in w or "lower image border" in w for w in sc.warnings), sc.warnings


def test_color_names():
    vt = 137.0
    def px(bgr, n=400):
        return np.tile(np.array(bgr, np.uint8), (n, 1))
    assert S.color_name(px((40, 50, 250)), vt)[0] == "red"
    assert S.color_name(px((230, 150, 40)), vt)[0] == "blue"
    assert S.color_name(px((70, 200, 60)), vt)[0] == "green"
    assert S.color_name(px((40, 220, 240)), vt)[0] == "yellow"
    assert S.color_name(px((255, 253, 255)), vt)[0] == "white"
    assert S.color_name(px((30, 30, 30)), vt)[0] == "black"
    assert S.color_name(np.vstack([px((255, 253, 255), 500), px((176, 220, 231), 400)]), vt)[0] == "ivory"
    assert S.color_name(np.vstack([px((255, 253, 255), 600), px((231, 221, 233), 300)]), vt)[0] == "white"
    name, detail = S.color_name(px((30, 140, 250)), vt)
    assert name == "other" and detail.startswith("orange")
    assert S.color_name(px((120, 120, 120)), vt) [1].startswith("gray")
    assert S.color_name(np.zeros((0, 3), np.uint8), vt)[0] == "other"
    # the name does not depend on the exposure
    for g in (0.8, 1.0, 1.2):
        both = np.vstack([px((255, 253, 255), 500), px((176, 220, 231), 400)]).astype(np.float32) * g
        assert S.color_name(np.clip(both, 0, 255).astype(np.uint8), vt * g)[0] == "ivory", g


# =========================================================================== 2026-11-11: taking things OUT of the box
# Frames of the afternoon in which the white cap was taken out of the black box with hand-driven low-level
# commands (tests/testdata/takeout_20261111/; joint angles and gripper position of every frame from the event log).
# Camera and kinematic calibration were off by 1-2 cm near the robot base on that day, so nothing here is
# compared with absolute coordinates: presence, kind, counts, sizes and positions RELATIVE to each other only.
TK = os.path.join(TD, "takeout_20261111")
TK_FRAMES = {f["file"]: f for f in json.load(open(os.path.join(TK, "frames.json")))["frames"]}
BOX_L, BOX_W = 0.113, 0.1005                    # outer size of the black box
CAP_D, CAP_H = 0.031, 0.0165                    # the white PET-bottle cap (open side up)
IVORY_D, IVORY_H, DISC_D = 0.026, 0.022, 0.035  # ivory cylinder, blue disc
TK_WITH_BOX = ("0008", "1101", "0030", "0034", "0040", "0042", "0046", "0048", "0056", "0072", "0084", "0088",
               "0090", "0094", "0108")
TK_ARM_AWAY = ("0008", "1101", "0030", "0048", "0088", "0090")          # arm parked or at the ready pose
REPORT.update(takeout=[], takeout_cap=[], takeout_runtime={})
_tk_cache = {}


def tk_file(prefix: str) -> str:
    return next(f for f in TK_FRAMES if f.startswith(prefix))


def tk_image(prefix: str) -> np.ndarray:
    key = ("img", prefix)
    if key not in _tk_cache:
        img = cv2.imread(os.path.join(TK, tk_file(prefix)))
        assert img is not None, prefix
        _tk_cache[key] = img
    return _tk_cache[key]


def tk_state(prefix: str):
    f = TK_FRAMES[tk_file(prefix)]
    return np.radians(f["q_deg"]), -11.41 + (f["gripper_raw"] - 2050) * 360.0 / 4096.0


def tk(prefix: str):
    """perceive() of a take-out frame with the arm masked by its joint angles, as fd.py calls it (cached)."""
    if prefix not in _tk_cache:
        q, jaw = tk_state(prefix)
        _tk_cache[prefix] = S.perceive(tk_image(prefix), CAM, CFG, arm_q=q, jaw_q_deg=jaw)
    return _tk_cache[prefix]


def tk_box(prefix: str):
    sc = tk(prefix)
    cs = sc.containers()
    assert len(cs) == 1, f"{prefix}: {len(cs)} containers; warnings: {sc.warnings}"
    return sc, cs[0]


def tk_inside(sc, color=None, kind=None, hollow=None) -> list:
    return [o for o in sc.objects if o.in_container is not None and (color is None or o.color == color)
            and (kind is None or o.kind == kind) and (hollow is None or o.hollow == hollow)]


def dist_mm(a, b) -> float:
    return 1000.0 * math.hypot(a.center[0] - b.center[0], a.center[1] - b.center[1])


def test_takeout_container_found():
    """The box is found in every view: floor mostly covered (five things in it), turned by up to 35 deg,
    the gripper above it or standing in it, and in the two crowded views of the morning. Its size is
    the box's size (113 x 100.5 mm; the camera model of that day stretches a turned box by a few mm)."""
    for k in TK_WITH_BOX:
        sc, c = tk_box(k)
        length, width, height = (1000.0 * v for v in c.size)
        REPORT["takeout"].append((tk_file(k)[:20], length, width, height, math.degrees(c.yaw), c.confidence,
                                  ", ".join(f"{o.color} {o.kind}{' (open)' if o.hollow else ''} {o.confidence:.2f}"
                                            for o in sc.objects if o.kind != "container")))
        assert c.kind == "container" and c.color == "black", (k, c.kind, c.color)
        assert abs(length - 1000 * BOX_L) <= 7.0 and abs(width - 1000 * BOX_W) <= 8.0, f"{k}: {length:.1f} x {width:.1f} mm"
        assert 25.0 <= height <= 31.5, f"{k}: rim {height:.1f} mm"
        inner = c.inner
        assert inner and Z0 < inner["floor_z"] < inner["rim_z"] and abs(inner["rim_z"] - (Z0 + c.size[2])) < 1e-9
        assert 0.020 <= inner["rim_z"] - inner["floor_z"] <= 0.028, (k, inner)
        assert c.confidence >= (0.7 if k in TK_ARM_AWAY else 0.5), f"{k}: confidence {c.confidence} {c.notes}"
        assert c not in sc.graspable() and len(c.footprint) == 4
        check_scene_wellformed(sc)


def test_takeout_container_any_rotation():
    """Turned about the vertical axis it is the same box: square to the table (0030), turned by about
    16 deg (0048) and by about 31 deg (0088; the back of the finger pushed it round), and in the morning
    by about 13 deg the other way (0008, against its usual place in the old frames)."""
    (_, c0), (_, c1), (_, c2), (_, c3) = tk_box("0030"), tk_box("0048"), tk_box("0088"), tk_box("0008")
    d1 = ang_diff_deg(math.degrees(c1.yaw), math.degrees(c0.yaw))
    d2 = ang_diff_deg(math.degrees(c2.yaw), math.degrees(c0.yaw))
    d3 = ang_diff_deg(math.degrees(c3.yaw), math.degrees(perceive("0113_locate_raw.jpg").containers()[0].yaw))
    assert -20.0 <= d1 <= -12.0 and -36.0 <= d2 <= -27.0 and 9.0 <= d3 <= 18.0, (d1, d2, d3)
    for c in (c1, c2):
        assert abs(c.size[0] - c0.size[0]) <= 0.003 and abs(c.size[1] - c0.size[1]) <= 0.005, (c.size, c0.size)
        assert abs(c.size[2] - c0.size[2]) <= 0.001
    for c in (c0, c1, c2, c3):                       # the footprint is the rectangle that centre, size and yaw describe
        p = np.array(c.footprint)
        e = max((p[(i + 1) % 4] - p[i] for i in range(4)), key=lambda v: float(np.linalg.norm(v)))
        assert abs(ang_diff_deg(math.degrees(math.atan2(e[1], e[0])), math.degrees(c.yaw))) < 1.0
        assert abs(float(np.linalg.norm(e)) - c.size[0]) < 1e-6 and np.allclose(p.mean(axis=0), c.center, atol=1e-6)
        ins = np.array(c.inner["center"])
        assert np.allclose(ins, c.center) and c.inner["size"][0] < c.size[0] and c.inner["size"][1] < c.size[1]


def test_takeout_container_with_gripper_in_front():
    """The gripper above or in the box: the same box comes out as with the arm away; a corner behind the
    gripper is rebuilt from its two rim edges (and said); a box divided by the gripper is put together;
    with fewer than four sides in view it is refused with the reason."""
    for a, b, tol_mm in (("0034", "0030", 2.0), ("0042", "0048", 2.0), ("0046", "0048", 3.0), ("0090", "0088", 2.0),
                         ("0108", "0088", 2.0)):
        (_, ca), (_, cb) = tk_box(a), tk_box(b)
        assert dist_mm(ca, cb) <= tol_mm, f"{a} vs {b}: centre differs by {dist_mm(ca, cb):.1f} mm"
        assert abs(ang_diff_deg(math.degrees(ca.yaw), math.degrees(cb.yaw))) <= 2.0, (a, b)
        assert all(abs(u - v) <= 0.003 for u, v in zip(ca.size, cb.size)), (a, ca.size, cb.size)
    for k in ("0046", "0040"):                       # the gripper hides the far corner
        _, c = tk_box(k)
        assert any("hidden by the robot arm" in n and "rebuilt" in n for n in c.notes), (k, c.notes)
        assert 0.5 <= c.confidence <= 0.7, (k, c.confidence)
    _, c = tk_box("0072")                            # the gripper stands in the box
    assert any("divides its image" in n for n in c.notes) and 0.5 <= c.confidence <= 0.7, c.notes
    sc = tk("0066")                                  # the gripper deep in the box: fewer than four sides of the rim
    assert not sc.containers() and any("hidden by the robot arm" in w and "not reported as a container" in w
                                       for w in sc.warnings), sc.warnings
    check_scene_wellformed(sc)
    _, c = tk_box("0084")                            # the cap held above the box: the box, and no cap in it
    assert not tk_inside(tk("0084"), "white") and not tk("0084").graspable()


def test_takeout_rim_height_from_outer_wall():
    """Where the foot of the near outer wall is in view the rim height is measured on it (rim edge to
    foot, nothing assumed) and the floor thickness follows from the inner wall; the same value in every
    view. Where it is not (the box at its usual place, near the lower image border), the inner wall plus
    the ASSUMED floor is used as before, and the notes say which."""
    hs, floors = [], []
    for k in ("0030", "0034", "0042", "0048", "0088", "0090"):
        _, c = tk_box(k)
        assert any("measured on the near outer wall" in n and "the floor is" in n for n in c.notes), (k, c.notes)
        assert not any("NOT measured" in n for n in c.notes)
        hs.append(c.size[2])
        floors.append(c.inner["floor_z"] - Z0)
        assert 0.0235 <= c.inner["rim_z"] - c.inner["floor_z"] <= 0.0275, (k, c.inner)       # the inner wall, measured
    assert max(hs) - min(hs) <= 0.001, hs
    assert 0.0255 <= float(np.mean(hs)) <= 0.0295, hs
    assert all(0.0005 <= f <= 0.0045 for f in floors), floors
    REPORT["takeout_rim"] = (1000 * float(np.mean(hs)), 1000 * (max(hs) - min(hs)), [round(1000 * f, 1) for f in floors])
    for k in ("0040", "0072"):                       # far inner wall behind the gripper: the outer wall alone
        _, c = tk_box(k)
        assert any("measured on the near outer wall" in n and "ASSUMED" in n for n in c.notes), (k, c.notes)
        assert abs(c.size[2] - float(np.mean(hs))) <= 0.001, (k, c.size)
    for k in ("1101", "0008"):                       # foot of the wall outside the image: as before
        _, c = tk_box(k)
        assert any("far inner wall" in n and "ASSUMED" in n for n in c.notes), (k, c.notes)
        assert not any("near outer wall" in n for n in c.notes) and abs(c.size[2] - 0.030) <= 0.0015, (k, c.size)
        assert abs((c.inner["floor_z"] - Z0) - 0.005) < 1e-9
    # A floor thickness that the user gives in the config is used in every view and counts as known;
    # the image is checked against it (here: against what the image itself says, and 3 mm more).
    seen = float(np.mean(floors))
    q, jaw = tk_state("0030")
    for given, agrees in ((seen, True), (seen + 0.003, False)):
        cfg = dict(CFG, scene=dict(CFG.get("scene") or {}, container_floor_mm=1000.0 * given))
        sc = S.perceive(tk_image("0030"), CAM, cfg, arm_q=q, jaw_q_deg=jaw)
        c = sc.containers()[0]
        ivory = [o for o in sc.objects if o.in_container == c.id and (o.kind, o.color) == ("cylinder", "ivory")][0]
        if agrees:
            assert any("measured on the near outer wall" in n for n in c.notes) and c.confidence >= 0.9, c.notes
            assert abs(c.size[2] - float(np.mean(hs))) <= 0.001 and ivory.confidence >= 0.7
        else:
            assert abs((c.inner["floor_z"] - Z0) - given) < 1e-9 and c.confidence <= 0.6, (c.inner, c.confidence)
            assert any("as given in scene.container_floor_mm" in n for n in c.notes), c.notes
            assert any("the camera calibration or that value is wrong" in n for n in c.notes), c.notes
            assert ivory.confidence <= 0.5 and any("floor" in n and "ASSUMED" in n for n in ivory.notes), ivory.notes
    cfg = dict(CFG, scene=dict(CFG.get("scene") or {}, container_floor_mm=5.0))
    name = "0146_cap_final_raw.jpg"                  # the foot of the wall is not in view: the given value stands
    q, jaw = arm_state(name)
    sc = S.perceive(image(name), CAM, cfg, arm_q=q, jaw_q_deg=jaw)
    c = sc.containers()[0]
    assert any("as given in scene.container_floor_mm" in n for n in c.notes) and not any("ASSUMED" in n for n in c.notes)
    assert abs(c.size[2] - perceive(name).containers()[0].size[2]) < 1e-9
    cap = [o for o in sc.objects if o.hollow]
    assert len(cap) == 1 and not any("floor" in n and "ASSUMED" in n for n in cap[0].notes), cap[0].notes


def test_takeout_contents_separated():
    """What lies in the box is reported thing by thing: the ivory cylinder and the white cap that touch
    each other in the image, the blue disc lying on the blue block, the red block; with sizes, and at
    the places they have relative to each other."""
    def u(o):
        return o.pixel_bbox[0] + 0.5 * o.pixel_bbox[2]

    def v(o):
        return o.pixel_bbox[1] + 0.5 * o.pixel_bbox[3]

    for k in ("0030", "0034", "0042", "0048"):
        sc, box = tk_box(k)
        ins = tk_inside(sc)
        assert all(o.in_container == box.id for o in ins) and not sc.graspable(), (k, [(o.kind, o.color) for o in sc.objects])
        ivory, cap, disc, red = (tk_inside(sc, "ivory", "cylinder"), tk_inside(sc, "white", "cylinder"),
                                 tk_inside(sc, "blue", "cylinder"), tk_inside(sc, "red"))
        assert len(ivory) == 1 and len(cap) == 1 and len(disc) == 1 and len(red) == 1, (k, [(o.kind, o.color) for o in ins])
        assert 5 <= len(ins) <= 6 and {o.color for o in ins} == {"ivory", "white", "blue", "red"}, (k, len(ins))
        ivory, cap, disc, red = ivory[0], cap[0], disc[0], red[0]
        assert abs(ivory.size[0] - IVORY_D) <= 0.002 and abs(ivory.size[2] - IVORY_H) <= 0.004, (k, ivory.size)
        assert ivory.confidence >= 0.7 and not ivory.hollow, (k, ivory.confidence, ivory.notes)
        assert abs(cap.size[0] - CAP_D) <= 0.002 and cap.hollow, (k, cap.size, cap.notes)
        assert abs(disc.size[0] - DISC_D) <= 0.003 and not disc.hollow, (k, disc.size)
        assert disc.confidence <= 0.5 and any("ASSUMED" in n for n in disc.notes), (k, disc.confidence, disc.notes)
        assert red.confidence <= 0.5 and any("ASSUMED" in n or "approximate" in n for n in red.notes)
        # in the image: cylinder, cap, red block from left to right; the disc in front of (below) the cap
        assert u(ivory) < u(cap) < u(red) and u(ivory) < u(disc) < u(red) and v(disc) > v(cap), k
        # on the floor: cap and cylinder are neighbours that do not overlap
        assert 0.5 * 1000 * (IVORY_D + CAP_D) - 3.0 <= dist_mm(ivory, cap) <= 45.0, (k, dist_mm(ivory, cap))
        ca, sa = math.cos(box.inner["yaw"]), math.sin(box.inner["yaw"])
        for o in ins:                                # every centre lies inside the opening
            d = np.array(o.center) - np.array(box.inner["center"])
            assert abs(d[0] * ca + d[1] * sa) <= box.inner["size"][0] / 2 + 1e-6, (k, o.color, o.kind)
            assert abs(-d[0] * sa + d[1] * ca) <= box.inner["size"][1] / 2 + 1e-6, (k, o.color, o.kind)
            assert abs(o.z_bottom - box.inner["floor_z"]) < 1e-9 and any("inside container" in n for n in o.notes)
    for a, b in (("0034", "0030"), ("0042", "0048")):        # the same scene past the gripper: the same things, same places
        for colour in ("ivory", "white"):
            oa, ob = tk_inside(tk(a), colour, "cylinder")[0], tk_inside(tk(b), colour, "cylinder")[0]
            assert dist_mm(oa, ob) <= 1.5 and all(abs(p - q) <= 0.001 for p, q in zip(oa.size, ob.size)), (a, b, colour)
    for k in ("0088", "0090", "0094", "0108"):               # after the take-out: the cap is on the table, the rest inside
        sc, box = tk_box(k)
        assert not tk_inside(sc, "white") and len(tk_inside(sc, "ivory", "cylinder")) == 1, k
        assert len(tk_inside(sc, "blue", "cylinder")) == 1 and len(tk_inside(sc, "red")) == 1, k
        assert abs(tk_inside(sc, "ivory", "cylinder")[0].size[0] - IVORY_D) <= 0.002
        assert abs(tk_inside(sc, "blue", "cylinder")[0].size[0] - DISC_D) <= 0.003
        loose = sc.graspable()
        assert [(o.kind, o.color) for o in loose] == [("cylinder", "white")] and loose[0].hollow, (k, loose)
        assert abs(loose[0].size[0] - CAP_D) <= 0.002 and dist_mm(loose[0], box) > 70.0, (k, loose[0].size)
    sc, box = tk_box("1101")                                 # the morning: disc on the ivory cylinder, cap on its side
    assert {o.color for o in tk_inside(sc)} >= {"blue", "red", "white"}, [(o.kind, o.color) for o in tk_inside(sc)]
    disc = tk_inside(sc, "blue", "cylinder")
    assert len(disc) == 1 and abs(disc[0].size[0] - DISC_D) <= 0.003 and not disc[0].hollow, [o.size for o in disc]
    sc, box = tk_box("0008")                                 # the morning: cap (open side up), blue and red block
    assert sorted(o.color for o in tk_inside(sc)) == ["blue", "red", "white"], [(o.kind, o.color) for o in tk_inside(sc)]


def test_takeout_cap_hollow_fields():
    """The cap is seen to be open at the top (hollow), in the box and on the table, with the optional
    SceneObject fields filled; solid things never are. Its confidence says how its height was found."""
    from fdtypes import SceneObject
    for k in ("0030", "0034", "0042", "0048", "0056", "0008"):
        sc, box = tk_box(k)
        caps = tk_inside(sc, hollow=True)
        assert len(caps) == 1 and (caps[0].kind, caps[0].color) == ("cylinder", "white"), (k, [(o.kind, o.color) for o in caps])
        cap, floor = caps[0], box.inner["floor_z"]
        REPORT["takeout_cap"].append((tk_file(k)[:20], "in the box", 1000 * cap.size[0], 1000 * cap.size[2],
                                      None if cap.wall is None else 1000 * cap.wall,
                                      None if cap.z_inner_bottom is None else 1000 * (cap.z_inner_bottom - floor),
                                      cap.confidence))
        assert abs(cap.size[0] - CAP_D) <= 0.002 and abs(cap.size[2] - CAP_H) <= 0.0035, (k, cap.size)
        assert abs(cap.z_bottom - floor) < 1e-9
        seen = [n for n in cap.notes if n.startswith("open at the top")]
        assert len(seen) == 1 and "mm down from the rim" in seen[0], cap.notes
        depth_mm = float(seen[0].split("is seen ")[1].split(" mm down")[0])
        assert 10.5 <= depth_mm <= 14.0, seen[0]      # the inner wall is seen 12 mm down from the rim, in every view
        if any("own base" in n for n in cap.notes):   # its height is measured: the level the inner wall is seen down to
            assert floor < cap.z_inner_bottom <= floor + 0.004, (k, cap.z_inner_bottom, floor)
            assert abs(1000 * (cap.z_top - cap.z_inner_bottom) - depth_mm) <= 0.51, (k, cap.z_top - cap.z_inner_bottom)
        else:                                         # its base is hidden: only the depth below the rim is known
            assert cap.z_inner_bottom is None and "the base is hidden" in seen[0], (k, cap.z_inner_bottom, seen)
        if cap.wall is not None:
            assert 0.0008 <= cap.wall <= 0.0045 and abs(cap.inner_radius - (cap.size[0] / 2 - cap.wall)) < 1e-9, (k, cap.wall)
        else:
            assert cap.inner_radius is None
        assert any(n.startswith("open at the top") for n in cap.notes), cap.notes
    for k in ("0030", "0034", "0042", "0048"):       # its base is behind the disc: height from the cavity, said so
        cap = tk_inside(tk(k), hollow=True)[0]
        assert 0.5 <= cap.confidence <= 0.7, (k, cap.confidence)
        assert any("ASSUMED" in n and "cavity" in n for n in cap.notes), (k, cap.notes)
    cap = tk_inside(tk("0056"), hollow=True)[0]       # its base in view: measured like the ivory cylinder
    assert cap.confidence >= 0.75 and abs(cap.size[2] - CAP_H) <= 0.002 and any("own base" in n for n in cap.notes), cap.notes
    cap = tk_inside(tk("0008"), hollow=True)[0]       # floor height not measured in that view: at most 0.5, said so
    assert cap.confidence == 0.5 and any("floor" in n and "ASSUMED" in n for n in cap.notes), (cap.confidence, cap.notes)
    ref = None
    for k in ("0088", "0090", "0094", "0108"):        # on the table
        sc = tk(k)
        cap = S.find_object(sc, {"hollow": True})
        assert cap.in_container is None and (cap.kind, cap.color) == ("cylinder", "white") and cap in sc.graspable()
        REPORT["takeout_cap"].append((tk_file(k)[:20], "on the table", 1000 * cap.size[0], 1000 * cap.size[2],
                                      None if cap.wall is None else 1000 * cap.wall, 1000 * (cap.z_inner_bottom - Z0),
                                      cap.confidence))
        assert abs(cap.size[0] - CAP_D) <= 0.002 and abs(cap.size[2] - CAP_H) <= 0.0035, (k, cap.size)
        assert Z0 < cap.z_inner_bottom <= Z0 + 0.004 and cap.confidence >= 0.7, (k, cap.z_inner_bottom, cap.confidence)
        assert 0.0105 <= cap.z_top - cap.z_inner_bottom <= 0.014, (k, cap.z_top - cap.z_inner_bottom)
        assert S.find_object(sc, {"color": "white", "kind": "cap"}) is cap
        ref = ref or cap
        assert dist_mm(cap, ref) <= 2.0, k
    sc = tk("0030")                                   # in the box it is not offered as a loose thing
    cap = S.find_object(sc, {"hollow": True, "in_container": True})
    assert cap.color == "white" and cap.in_container is not None
    # the solid ivory cylinder: never called open; its over-exposed top cannot be called closed either, and says so
    ivory = S.find_object(sc, {"in_container": True, "color": "ivory"})
    assert ivory.kind == "cylinder" and ivory.hollow is not True
    assert ivory.hollow is False or any("open or closed top: not determined" in n for n in ivory.notes), ivory.notes
    for want in (True, False):                        # the query keeps exactly the things that were judged that way
        try:
            got = S.find_object(sc, {"hollow": want, "in_container": True})
            assert got.hollow is want
        except LookupError:
            assert not any(o.hollow is want for o in tk_inside(sc))
    try:
        S.find_object(sc, {"hollow": True})
        raise AssertionError("a cap inside the box was offered as a loose object")
    except LookupError as e:
        assert "open top" in str(e)                   # the message lists what is there
    # the blue disc lies on the blue block; in 0094 its near edge catches the light, which looks like the bright
    # bottom of a 21 mm deep cavity in a thing that would be 11 mm high: not open, less certain, and it says so
    disc = tk_inside(tk("0094"), "blue", "cylinder")[0]
    assert disc.hollow is None and disc.confidence <= 0.5, (disc.hollow, disc.confidence)
    assert any("open or closed top: not determined" in n for n in disc.notes), disc.notes
    for k in TK_WITH_BOX:                             # nothing but the cap is called hollow
        for o in tk(k).objects:
            assert o.hollow in (None, False, True) and (o.hollow is None or isinstance(o.hollow, bool))
            assert not o.hollow or (o.kind, o.color) == ("cylinder", "white"), (k, o.kind, o.color, o.notes)
            if o.kind != "cylinder":
                assert o.hollow is None                # only round tops are looked into
            if not o.hollow:
                assert o.inner_radius is None and o.wall is None and o.z_inner_bottom is None
    for name in ARM_AWAY:                             # the old frames: solid cylinders and blocks on the table
        for o in perceive(name).graspable():
            assert not o.hollow, (name, o.kind, o.color)
    for name in ("0146_cap_final_raw.jpg", "0168_cube_final_raw.jpg"):     # ... and the cap standing in the box there
        caps = [o for o in perceive(name).objects if o.hollow]
        assert len(caps) == 1 and caps[0].in_container is not None and abs(caps[0].size[0] - CAP_D) <= 0.002
        assert caps[0].confidence <= 0.5 and any("floor" in n and "ASSUMED" in n for n in caps[0].notes)
    d = [o for o in tk("0030").to_dict()["objects"] if o.get("hollow")]
    assert len(d) == 1 and {"hollow", "inner_radius", "wall", "z_inner_bottom", "z_top"} <= set(d[0]) and json.dumps(d)
    plain = SceneObject(id=1, kind="box", color="red", center=(0.2, 0.0), yaw=0.0, size=(0.04, 0.02, 0.02), z_bottom=Z0,
                        footprint=[(0, 0), (1, 0), (1, 1), (0, 1)], pixel_bbox=(0, 0, 1, 1), confidence=1.0)
    assert plain.hollow is None and plain.inner_radius is None and plain.wall is None and plain.z_inner_bottom is None
    vis = S.annotate(tk_image("0030"), tk("0030"), CAM)
    assert vis.shape == tk_image("0030").shape and np.count_nonzero(np.any(vis != tk_image("0030"), axis=2)) > 3000


def test_takeout_low_confidence_says_why():
    """Whatever is not sure says why in its notes, so that a skill can refuse instead of guessing."""
    words = ("ASSUMED", "approximate", "hidden", "robot arm", "less certain", "NOT measured", "divides", "cut by")
    for k in TK_WITH_BOX:
        for o in tk(k).objects:
            limit = 0.9 if o.kind == "container" else 0.5
            if o.confidence < limit or (o.in_container is not None and o.confidence == 0.5):
                assert any(any(w in n for w in words) for n in o.notes), (k, o.kind, o.color, o.confidence, o.notes)
            if o.in_container is not None and o.kind != "cylinder":
                assert o.confidence <= 0.5, (k, o.kind, o.color, o.confidence)
    for k in ("0040", "0046", "0072"):                # next to the gripper nothing is trusted
        for o in tk_inside(tk(k)):
            assert o.confidence <= 0.5 and not o.hollow, (k, o.kind, o.color, o.confidence)


def test_takeout_runtime():
    """As fast as before on these crowded frames: <= 0.35 s (best of 3), like the other frames."""
    best = {}
    for k in sorted(TK_FRAMES):
        img = tk_image(k[:4])
        q, jaw = tk_state(k[:4])
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            S.perceive(img, CAM, CFG, arm_q=q, jaw_q_deg=jaw)
            ts.append(time.perf_counter() - t0)
        best[k] = min(ts)
    REPORT["takeout_runtime"] = {"median_ms": 1000 * float(np.median(list(best.values()))), "max_ms": 1000 * max(best.values()),
                                 "worst": max(best, key=best.get)[:16], "n": len(best)}
    assert max(best.values()) <= 0.35, REPORT["takeout_runtime"]


# --------------------------------------------------------------------------- the geometry behind it, on synthetic input
def _outline_of(mask):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2)
    return cnt, cv2.convexHull(cnt).reshape(-1, 2)


def test_outline_sides_rebuild_a_hidden_corner():
    """_outline_sides: the four sides of a convex outline from its own pixels; a corner that is cut off
    (by the arm, or by something bright) is the intersection of its two sides; three sides are refused."""
    quad = np.array([[60, 40], [250, 60], [270, 200], [40, 180]], np.int32)
    mask = np.zeros((240, 320), np.uint8)
    cv2.fillConvexPoly(mask, quad, 255)
    nothing = np.zeros_like(mask)

    def nearest(r, pt):
        d = np.linalg.norm(r["corners"] - np.array(pt, float), axis=1)
        return int(np.argmin(d)), float(d.min())

    r = S._outline_sides(*_outline_of(mask), nothing)
    assert r is not None and all(r["corner_seen"]) and not any(r["corner_blocked"]) and r["chord"] < 14.0, r
    assert all(nearest(r, p)[1] <= 1.5 for p in quad) and min(r["seen"]) >= 0.9 and max(r["hidden"]) == 0.0, r
    arm = np.zeros_like(mask)
    cv2.circle(arm, (250, 60), 45, 255, -1)
    cut = mask.copy()
    cut[arm > 0] = 0
    blocked = cv2.dilate(arm, np.ones((9, 9), np.uint8))
    r = S._outline_sides(*_outline_of(cut), blocked)          # hidden by the arm
    assert r is not None and r["chord"] >= 14.0 and sum(r["corner_seen"]) == 3, r
    i, d = nearest(r, (250, 60))
    assert d <= 2.0 and not r["corner_seen"][i] and r["corner_blocked"][i], (i, d, r)
    assert all(nearest(r, p)[1] <= 1.5 for p in quad[[0, 2, 3]])
    assert r["hidden"][i] > 0.1 and r["hidden"][i - 1] > 0.1 and r["seen"][i] < 0.9
    r = S._outline_sides(*_outline_of(cut), nothing)          # covered by something that is not the arm
    i, d = nearest(r, (250, 60))
    assert r is not None and d <= 2.0 and not r["corner_seen"][i] and not r["corner_blocked"][i], (i, d, r)
    side = np.zeros_like(mask)
    side[:75] = 255                                           # a whole side behind the arm: three sides are no box
    cut = mask.copy()
    cut[side > 0] = 0
    assert S._outline_sides(*_outline_of(cut), cv2.dilate(side, np.ones((9, 9), np.uint8))) is None
    assert S._outline_sides(np.zeros((10, 2)), np.zeros((3, 2)), nothing) is None


def test_round_top_circle_and_its_height():
    """_fit_top_circle: the far half of a rim seen through the camera model gives the circle back when it
    is put on the plane of the rim. On a plane 5 mm too low it is still a circle (so the outline alone
    cannot tell the height) but a larger one, 6-7 mm farther from the camera: the height matters, and
    centre, radius and height are only consistent together. Outline points that belong to a neighbour
    are dropped; a straight edge is no rim."""
    fc = S._FastCam(CAM)
    x, y, r, z = 0.22, 0.06, 0.0155, Z0 + 0.021
    a = np.linspace(0.0, 2.0 * math.pi, 600, endpoint=False)
    n = fc.C[:2] - np.array([x, y])
    n = n / np.linalg.norm(n)
    far = (np.cos(a) * n[0] + np.sin(a) * n[1]) <= 0.0
    ring = np.column_stack([x + r * np.cos(a), y + r * np.sin(a), np.full(len(a), z)])
    arc = np.unique(np.round(fc.project(ring[far])), axis=0)
    f = S._fit_top_circle(fc, arc, z)
    assert f is not None and math.hypot(f["x"] - x, f["y"] - y) < 0.0004 and abs(f["r"] - r) < 0.0004, f
    assert f["rms"] < 0.0002 and f["cover_deg"] > 160.0 and f["frac"] > 0.9, f
    g = S._fit_top_circle(fc, arc, z - 0.005)
    shift = float((np.array([g["x"], g["y"]]) - np.array([f["x"], f["y"]])) @ n)
    assert g is not None and g["rms"] < 0.0003 and -0.009 < shift < -0.004 and 0.00015 < g["r"] - f["r"] < 0.0006, (g, shift)
    # a fifth of the outline belongs to something that touches it (3 mm outside the rim): dropped, the same circle
    bump = far & (np.arange(len(a)) % 300 < 60)
    mixed = ring.copy()
    mixed[bump, :2] += 0.003 * np.column_stack([np.cos(a[bump]), np.sin(a[bump])])
    m = S._fit_top_circle(fc, np.unique(np.round(fc.project(mixed[far])), axis=0), z)
    assert m is not None and math.hypot(m["x"] - x, m["y"] - y) < 0.0006 and abs(m["r"] - r) < 0.0006, m
    assert 0.6 <= m["frac"] < 0.95 and int(np.count_nonzero(m["keep"])) == m["n"], m
    edge = np.column_stack([np.linspace(x - 0.02, x + 0.02, 80), np.full(80, y), np.full(80, z)])
    assert S._fit_top_circle(fc, np.unique(np.round(fc.project(edge)), axis=0), z) is None      # a straight edge
    assert S._fit_top_circle(fc, arc[:5], z) is None                                            # too few points


def _painted_cup(x, y, r, h, wall, depth, inside="cavity", top=253):
    """A white upright cylinder drawn through the camera model; inside its rim: "cavity" = a darker far
    inner wall down to `depth` and the bright inner bottom beyond, "flat" = nothing, "boss" = a dark band
    that does not begin at the rim (like the hub on the blue disc), "shadow" = a soft shadow across."""
    fc = S._FastCam(CAM)
    img = np.zeros((C.H, C.W, 3), np.uint8)
    img[:] = (38, 38, 40)
    a = np.linspace(0.0, 2.0 * math.pi, 160, endpoint=False)

    def poly(rr, z):
        ring = np.column_stack([x + rr * np.cos(a), y + rr * np.sin(a), np.full(len(a), z)])
        return np.round(fc.project(ring) * 8).astype(np.int32)

    def fill(target, pts, col):
        cv2.fillConvexPoly(target, cv2.convexHull(pts), col, lineType=cv2.LINE_AA, shift=3)

    zt = Z0 + h
    fill(img, np.vstack([poly(r, zt), poly(r, Z0)]), (228, 230, 232))        # body
    fill(img, poly(r, zt), (top, top, top))                                   # top
    opening, bottom, part = (np.zeros(img.shape[:2], np.uint8) for _ in range(3))
    fill(opening, poly(r - wall, zt), 255)
    fill(bottom, poly(r - wall, zt - depth), 255)
    if inside == "cavity":
        img[(opening > 0) & (bottom == 0)] = (top - 41, top - 39, top - 37)   # far inner wall
    elif inside == "boss":
        fill(part, poly(r - wall, zt - 0.4 * depth), 255)
        img[(opening > 0) & (part > 0) & (bottom == 0)] = (150, 152, 155)
    elif inside == "shadow":
        rows = np.nonzero(opening.any(axis=1))[0]
        ramp = np.interp(np.arange(img.shape[0]), [rows.min(), rows.max()], [top - 48.0, float(top)])
        sel = opening > 0
        img[sel] = np.repeat(ramp[:, None], img.shape[1], axis=1)[sel][:, None].astype(np.uint8)
    return cv2.GaussianBlur(img, (0, 0), 0.8), zt


def test_cavity_synthetic():
    """_cavity on painted tops: an inner wall with a step to the inner bottom is a cavity and its depth is
    read right; a flat top, a dark boss in the middle of a flat top and a soft shadow across it are not."""
    fc = S._FastCam(CAM)
    slivers = []
    for x, y, r, h, wall, depth in ((0.23, 0.02, 0.016, 0.020, 0.002, 0.014), (0.20, -0.08, 0.0155, 0.0165, 0.0018, 0.0125),
                                    (0.27, 0.09, 0.022, 0.030, 0.0025, 0.016)):
        img, zt = _painted_cup(x, y, r, h, wall, depth)
        cav = S._cavity(img, fc, x, y, r, zt)
        assert cav["hollow"] and cav["judged"] and abs(cav["depth"] - depth) <= 0.0015 and cav["contrast"] > 15.0, (r, depth, cav)
        assert cav["wall"] is not None and abs(cav["wall"] - wall) <= 0.0012, (wall, cav)
        for kind, word in (("flat", "no darker inner wall"), ("boss", ""), ("shadow", "")):
            img, zt = _painted_cup(x, y, r, h, wall, depth, kind)
            cav = S._cavity(img, fc, x, y, r, zt)
            assert not cav["hollow"] and cav["depth"] is None and cav["wall"] is None and word in cav["why"], (kind, cav)
            assert not cav["judged"], (kind, cav)       # over-exposed (flat) or not understood: it cannot be told
        # "closed" is said only of a top that is evenly lit, neither over-exposed nor dark, and shows nothing
        for top, judged in ((200, True), (253, False), (45, False)):
            img, zt = _painted_cup(x, y, r, h, wall, depth, "flat", top=top)
            cav = S._cavity(img, fc, x, y, r, zt)
            assert not cav["hollow"] and cav["judged"] is judged, (top, cav)
        img, zt = _painted_cup(x, y, r, h, wall, depth, top=200)                # a cavity in a top that is not over-exposed
        cav = S._cavity(img, fc, x, y, r, zt)
        assert cav["hollow"] and cav["judged"] and abs(cav["depth"] - depth) <= 0.0015, cav
        assert cav["bottom_frac"] >= 0.15, cav
        # a bright sliver at the near edge only (the edge of a flat top catching the light, or a cavity too deep
        # to be looked into): the inner bottom is not seen over a real area, so it is not called open
        dist = float(np.linalg.norm(fc.C[:2] - np.array([x, y])))
        t_max = 2.0 * (r - wall) * float(fc.C[2] - (Z0 + h)) / dist             # how far down the far inner wall is in view
        img, zt = _painted_cup(x, y, r, h, wall, 0.86 * t_max, top=200)
        cav = S._cavity(img, fc, x, y, r, zt)
        assert not cav["hollow"] and not cav["judged"] and cav["depth"] is None, cav
        assert "covers only" in cav["why"] or "no darker inner wall" in cav["why"], cav
        slivers.append("covers only" in cav["why"])
    assert any(slivers), slivers                      # the rule about the visible bottom area was the one that refused
    cav = S._cavity(np.zeros((C.H, C.W, 3), np.uint8), fc, 0.23, 0.02, 0.002, Z0 + 0.01)        # too small to judge
    assert not cav["hollow"] and not cav["judged"] and cav["why"]


def test_outer_wall_height_synthetic():
    """_outer_wall_height on boxes drawn through the camera model: the rim height above the table comes
    out to a millimetre for different heights and turns, with nothing assumed; with the foot of the wall
    outside the image it is not measured, and says so."""
    fc = S._FastCam(CAM)
    for x, y, yaw_deg, height in ((0.20, 0.05, 0.0, 0.030), (0.21, -0.04, 25.0, 0.024), (0.19, 0.08, -35.0, 0.045)):
        yaw = math.radians(yaw_deg)
        box = {"kind": "box", "x": x, "y": y, "yaw": yaw, "size": (0.11, 0.10, height), "bgr": (60, 60, 62), "shade": (2.0, 1.6)}
        img = render([box], noise=1.0)
        quad = fc.project(S.box_corners(x, y, yaw, 0.11, 0.10, height, Z0)[4:])
        res = S._outer_wall_height(img, fc, quad, float(TABLE_BGR[2]), Z0, 0.030, wide=True)
        assert res["measured"] and abs(res["H"] - height) <= 0.001 and res["n"] >= 6, (yaw_deg, height, res)
        assert res["spread_m"] <= 0.0024
    box = {"kind": "box", "x": 0.31, "y": 0.05, "yaw": 0.0, "size": (0.11, 0.10, 0.03), "bgr": (60, 60, 62), "shade": (2.0, 1.6)}
    quad = fc.project(S.box_corners(0.31, 0.05, 0.0, 0.11, 0.10, 0.03, Z0)[4:])
    res = S._outer_wall_height(render([box], noise=1.0), fc, quad, float(TABLE_BGR[2]), Z0, 0.030)
    assert not res["measured"] and "foot" in res["why"], res


# =========================================================================== jaws
def test_detect_jaws_saved_frames():
    """Per frame: what is seen, the implied offset, and that occluded / merged noses are NOT trusted."""
    res = {}
    for k, rk in JAW_REF.items():
        name, ref = frame_file(k), frame_file(rk)
        q, jaw = arm_state(name)
        t0 = time.perf_counter()
        r = S.detect_jaws(image(name), image(ref), CAM, q, jaw, CFG)
        r["ms"] = (time.perf_counter() - t0) * 1000
        res[k] = r
        REPORT["jaws"].append((k, FRAMES[name]["name"], jaw, r))
        assert set(r) >= {"ok", "fixed", "moving", "dxy_base", "confidence"}
        for part in ("fixed", "moving"):
            assert set(r[part]) >= {"uv", "pred_uv", "visible"} and r[part]["pred_uv"] is not None
            if r[part]["visible"]:
                assert r[part]["uv"] is not None and r[part]["quality"] >= 0.5
        if r["ok"]:
            assert r["dxy_base"] is not None and 0.0 < r["confidence"] <= 1.0
            assert math.hypot(*r["dxy_base"]) < 0.008, (k, r["dxy_base"])       # FK and camera agree to millimetres
        else:
            assert r["confidence"] < 0.5 and r["reason"], (k, r)
    # open gripper above / at the red block: the moving jaw is in front of the fixed one
    for k in ("0118", "0120", "0122"):
        r = res[k]
        assert r["ok"] and r["moving"]["visible"] and not r["fixed"]["visible"], (k, r["reason"])
        assert "hidden behind the moving jaw" in r["fixed"]["reason"]
        assert r["along_only_from_fixed"] and r["confidence"] >= 0.7
        o = S.jaw_model(*arm_state(frame_file(k)))["open_dir"]
        assert abs(r["dxy_base"][0] * o[0] + r["dxy_base"][1] * o[1]) < 1e-9       # nothing reported along the opening
    # white cap between the jaws: the moving nose merges with it and is refused; the fixed nose is used
    r = res["0140"]
    assert r["ok"] and r["fixed"]["visible"] and not r["moving"]["visible"] and "pale" in r["moving"]["reason"], r["moving"]
    # blue block: fixed nose seen; the moving nose is over the glare patch (0160) / pressed on the block (0162)
    for k in ("0160", "0162"):
        assert res[k]["ok"] and res[k]["fixed"]["visible"] and not res[k]["moving"]["visible"], (k, res[k]["reason"])
        assert res[k]["confidence"] >= 0.7
    d = np.array(res["0160"]["dxy_base"]) - np.array(res["0162"]["dxy_base"])      # same arm pose twice
    assert np.linalg.norm(d) < 0.0015, d
    # closed gripper: both fingers as one body (not "whichever nose is nearer the camera")
    for k in ("0028", "0158"):
        assert res[k]["located_as_one_body"] and res[k]["fixed"]["uv"] is not None
        assert res[k]["fixed"]["shift_px"] == res[k]["moving"]["shift_px"]
    # holding the white cap in the air / gripper over the cluttered left side: refused
    for k in ("0142", "0164"):
        assert not res[k]["ok"] and res[k]["dxy_base"] is None and res[k]["confidence"] == 0.0, (k, res[k]["reason"])
    assert max(r["ms"] for r in res.values()) < 150.0


def _paint_gripper(ref, q, jaw_deg, shift_xy=(0.0, 0.0), jaw_err_deg=0.0, moving_extra_xy=(0.0, 0.0)):
    """Reference image + both fingers painted white where a gripper displaced by shift_xy (base
    frame) and with a jaw angle error would appear."""
    fc = S._FastCam(CAM)
    model = S.jaw_model(q, jaw_deg + jaw_err_deg)
    img = ref.copy()
    parts = []
    for part in ("fixed", "moving"):
        pts = model[part]["pts"].copy()
        pts[:, 0] += shift_xy[0] + (moving_extra_xy[0] if part == "moving" else 0.0)
        pts[:, 1] += shift_xy[1] + (moving_extra_xy[1] if part == "moving" else 0.0)
        parts.append((float(fc.depth(pts).mean()), pts))
    for _, pts in sorted(parts, key=lambda p: -p[0]):                       # far finger first
        hull = cv2.convexHull(np.round(fc.project(pts) * 8).astype(np.int32))
        cv2.fillConvexPoly(img, hull, (246, 244, 246), lineType=cv2.LINE_AA, shift=3)
    return img


def _side_pose(x=0.20, y=0.045, z=0.012, opening_azimuth_deg=-90.0):
    """Top-down pose whose jaws open sideways in the image (both noses visible from the camera)."""
    tool = K.tool_offset(-0.001, 0.0, 0.0068)
    q = None
    pan = 0.0
    for _ in range(4):
        roll = opening_azimuth_deg + math.degrees(pan) + 2.789
        q, info = K.ik((x, y, z), 90.0, math.radians(roll), tool=tool)
        assert info["ok"], info
        pan = float(q[0])
    return q


def test_detect_jaws_synthetic_offsets():
    """Known displacements are recovered; a jaw-angle error is recognised as such; inconsistent
    noses are refused."""
    ref = image("0132_final_overview_raw.jpg")
    q = _side_pose()
    jaw = 30.0
    r0 = S.detect_jaws(_paint_gripper(ref, q, jaw), ref, CAM, q, jaw, CFG)
    assert r0["ok"] and r0["fixed"]["visible"] and r0["moving"]["visible"] and r0["agree"] is True, r0
    assert math.hypot(*r0["dxy_base"]) < 0.0008 and r0["confidence"] >= 0.9, r0
    for sx, sy in ((0.004, 0.0), (0.0, -0.005), (-0.003, 0.004), (0.006, 0.006)):
        r = S.detect_jaws(_paint_gripper(ref, q, jaw, (sx, sy)), ref, CAM, q, jaw, CFG)
        assert r["ok"] and r["agree"] is True, (sx, sy, r["reason"])
        err = math.hypot(r["dxy_base"][0] - sx, r["dxy_base"][1] - sy)
        REPORT.setdefault("jaw_synthetic_err_mm", []).append(err * 1000)
        assert err < 0.0008, f"shift ({sx}, {sy}) recovered as {r['dxy_base']} ({err * 1000:.2f} mm off)"
        assert r["confidence"] >= 0.9
    # the jaw is really 3 deg more open than its raw position says: the fixed nose still gives the offset
    r = S.detect_jaws(_paint_gripper(ref, q, jaw, (0.003, -0.002), jaw_err_deg=3.0), ref, CAM, q, jaw, CFG)
    assert r["ok"] and r["agree"] is True and abs(r["jaw_angle_error_deg"] - 3.0) < 0.8, r
    assert math.hypot(r["dxy_base"][0] - 0.003, r["dxy_base"][1] + 0.002) < 0.0008
    # the moving nose is displaced ACROSS the opening direction (no jaw angle can do that): refuse
    o = S.jaw_model(q, jaw)["open_dir"]
    across = (-o[1] * 0.006, o[0] * 0.006)
    r = S.detect_jaws(_paint_gripper(ref, q, jaw, moving_extra_xy=across), ref, CAM, q, jaw, CFG)
    assert r["ok"] is False and r["agree"] is False and r["confidence"] < 0.5 and "disagree" in r["reason"], r
    # the gripper is far from where the kinematics puts it: refused, never a wild number
    r = S.detect_jaws(_paint_gripper(ref, q, jaw, (0.03, 0.02)), ref, CAM, q, jaw, CFG)
    assert not r["ok"] and r["dxy_base"] is None, r
    # closed jaws (one body) with a displacement
    qc = _side_pose(z=0.02)
    r = S.detect_jaws(_paint_gripper(ref, qc, -10.5, (0.003, 0.003)), ref, CAM, qc, -10.5, CFG)
    assert r["ok"] and r["located_as_one_body"], r["reason"]
    assert math.hypot(r["dxy_base"][0] - 0.003, r["dxy_base"][1] - 0.003) < 0.001, r["dxy_base"]


def test_detect_jaws_occlusion_and_white_background():
    ref = image("0132_final_overview_raw.jpg")
    jaw = 30.0
    # jaws opening towards the camera: the moving jaw hides the fixed nose -> moving only, cross component only
    q = _side_pose(opening_azimuth_deg=0.0, y=0.0)
    img = _paint_gripper(ref, q, jaw, (0.0, 0.004))
    r = S.detect_jaws(img, ref, CAM, q, jaw, CFG)
    assert r["ok"] and not r["fixed"]["visible"] and r["moving"]["visible"] and r["along_only_from_fixed"], r["reason"]
    o = S.jaw_model(q, jaw)["open_dir"]
    assert abs(r["dxy_base"][0] * o[0] + r["dxy_base"][1] * o[1]) < 1e-9
    assert abs(r["dxy_base"][1] - 0.004) < 0.001 and 0.6 <= r["confidence"] <= 0.8, r
    # a white object already lies under the fixed nose (it is in the reference too): fixed nose refused
    q = _side_pose()
    model = S.jaw_model(q, jaw)
    u, v = S._FastCam(CAM).project(model["fixed"]["nose"][None])[0]
    ref2 = ref.copy()
    cv2.circle(ref2, (int(u), int(v) + 6), 34, (252, 250, 252), -1)
    r = S.detect_jaws(_paint_gripper(ref2, q, jaw), ref2, CAM, q, jaw, CFG)
    assert not r["fixed"]["visible"] and "pale" in r["fixed"]["reason"], r["fixed"]
    assert r["confidence"] <= 0.8 and (r["dxy_base"] is None or r["along_only_from_fixed"])
    # no gripper in the image at all
    r = S.detect_jaws(ref, ref, CAM, q, jaw, CFG)
    assert not r["ok"] and r["dxy_base"] is None and r["confidence"] == 0.0 and r["reason"]
    # bad input is refused with a reason, not an exception
    for args, word in (((ref, None, CAM, q, jaw, CFG), "reference"), ((ref, ref[:500], CAM, q, jaw, CFG), "reference"),
                       ((ref, ref, CAM, None, jaw, CFG), "arm_q"), ((ref, ref, CAM, q, None, CFG), "jaw_q_deg"),
                       ((ref, ref, CAM, q, jaw, {}), "table_z")):
        r = S.detect_jaws(*args)
        assert r["ok"] is False and word in r["reason"] and r["dxy_base"] is None, (word, r["reason"])


def test_jaw_model_matches_gripper_geometry():
    """The embedded finger outlines agree with gripper_model / the kinematics landmarks."""
    import gripper_model as GM
    assert abs(S._JAW_FIXED_MM[:, 2].max() - GM.FIXED_TIP_Z) < 0.05
    assert abs(S._JAW_MOVING_Q0_MM[:, 2].max() - 7.273) < 0.05
    assert S._JAW_FIXED_MM[:, 0].min() > -0.2 and S._JAW_MOVING_Q0_MM[:, 0].max() < -15.7      # pads at x = 0 / -15.8
    for qj in (-11.41, 0.0, 20.0, 45.0):
        mm = S._moving_jaw_mm(qj)
        assert abs(mm[:, 2].max() - GM.moving_tip_depth(qj)) < 0.35, (qj, mm[:, 2].max(), GM.moving_tip_depth(qj))
    q = np.radians([10.0, 20.0, 10.0, 60.0, 15.0])
    m = S.jaw_model(q, 25.0)
    T = K.fk(q)
    assert np.allclose(m["fixed"]["nose"], T[:3, 3] + T[:3, :3] @ (np.array(S.nose_points_gf(25.0)["fixed"]) / 1000.0))
    assert abs(np.linalg.norm(m["open_dir"]) - 1.0) < 1e-9 and m["fixed"]["dist"].min() < 1e-6
    jp = getattr(K, "jaw_points", None)
    if jp is not None:                                   # KIN's landmarks, when present, are the same points
        lm = jp(q, 25.0)
        assert np.linalg.norm(lm["fixed_nose"] - m["fixed"]["nose"]) < 0.0005
        assert np.linalg.norm(lm["moving_nose"] - m["moving"]["nose"]) < 0.0005


# =========================================================================== camera maths (synthetic)
def test_camera_roundtrips():
    rng = np.random.default_rng(0)
    for k1 in (0.0, 0.08, -0.1):
        cam = look_at_camera((0.75, 0.03, 0.42), (0.14, 0.0, 0.0), f=1400.0, k1=k1)
        fc = S._FastCam(cam)
        pts = np.column_stack([rng.uniform(0.1, 0.4, 60), rng.uniform(-0.2, 0.2, 60), rng.uniform(-0.03, 0.1, 60)])
        uv = cam.project(pts)
        assert np.abs(fc.project(pts) - uv).max() < 1e-6, k1
        for p, (u, v) in zip(pts[:20], uv[:20]):
            back = cam.to_plane((u, v), p[2])
            assert np.linalg.norm(back - p) < (1e-9 if k1 == 0 else 2e-5), (k1, np.linalg.norm(back - p))
            o, d = cam.ray((u, v))
            assert abs(np.linalg.norm(d) - 1.0) < 1e-12 and np.allclose(o, cam.center)
            assert np.linalg.norm(np.cross(p - o, d)) < (1e-9 if k1 == 0 else 2e-5)
        zs = pts[:, 2].copy()
        zs[:] = 0.02
        back = fc.to_plane(cam.project(np.column_stack([pts[:, :2], zs])), 0.02)
        assert np.abs(back - np.column_stack([pts[:, :2], zs])).max() < (1e-9 if k1 == 0 else 2e-5)
        assert np.all(fc.depth(pts) > 0) and np.allclose(fc.C, cam.center)
        assert np.linalg.norm(cam.to_table((900, 700), (0.01, 0.0, 0.0)) - cam.to_plane((900, 700), 0.01)) < 1e-12
        tilted = cam.to_table((900, 700), (0.0, 0.05, -0.02))
        assert abs(tilted[2] - (0.05 * tilted[0] - 0.02 * tilted[1])) < 1e-12
    assert np.all(np.isnan(S._FastCam(CAM).to_plane([[960.0, 10.0]], 5.0)))        # plane above the camera: no hit
    try:
        CAM.to_plane((960, 10), 5.0)
        raise AssertionError("no error for a plane behind the camera")
    except ValueError:
        pass
    d = CAM.to_dict()
    again = C.CameraModel(d["f"], d["cx"], d["cy"], d["rvec"], d["tvec"], d["k1"])
    assert np.allclose(again.project([[0.25, 0.0, 0.0]]), CAM.project([[0.25, 0.0, 0.0]]))


def _calib_points(cam, n, rng, noise=0.0):
    pts = np.column_stack([rng.uniform(0.12, 0.36, n), rng.uniform(-0.18, 0.18, n), rng.uniform(-0.02, 0.11, n)])
    uv = cam.project(pts) + rng.normal(0, noise, (n, 2)) if noise else cam.project(pts)
    return pts, uv


def test_fit_camera_pinned_and_bounds():
    """VM-4: a pinned focal length stays pinned; the search never leaves its range and says when it
    ends at a bound."""
    rng = np.random.default_rng(1)
    true = look_at_camera((0.75, 0.03, 0.42), (0.14, 0.0, 0.0), f=1400.0)
    pts, uv = _calib_points(true, 16, rng)
    m = C.fit_camera(pts, uv)
    assert abs(m.f - 1400.0) <= 1.0 and m.rms < 0.3 and np.linalg.norm(m.center - true.center) < 0.002
    assert m.f_at_bound is False and len(m.rms_profile) > 10 and C.check_model(m, pts) == []
    for f in (1400.0, 1430.0, 1250.0):
        m = C.fit_camera(pts, uv, f_range=(f, f))
        assert m.f == f, f"pinned f {f} came back as {m.f}"        # v1 returned 1410 for 1400
        assert m.f_at_bound is False and len(m.rms_profile) == 1
    m = C.fit_camera(pts, uv, f_range=(1500.0, 1650.0))
    assert 1500.0 <= m.f <= 1650.0 and m.f_at_bound is True and any("search range" in p for p in C.check_model(m, pts))
    m = C.fit_camera(pts, uv, f_range=(1395.0, 1403.0), f_step=10.0)             # refinement stays inside
    assert 1395.0 <= m.f <= 1403.0
    pts_n, uv_n = _calib_points(true, 20, rng, noise=2.0)
    m = C.fit_camera(pts_n, uv_n, f_range=(1400.0, 1400.0))
    assert m.f == 1400.0 and 1.0 < m.rms < 4.5
    res = C.residuals(m, pts_n, uv_n)
    vec = C.residual_vectors(m, pts_n, uv_n)
    assert np.allclose(np.linalg.norm(vec, axis=1), res) and abs(vec.mean()) < 1.5
    for bad in (((pts[:5], uv[:5]), {}), ((pts, uv[:10]), {}), ((pts, uv), {"f_range": (1500.0, 1400.0)}),
                ((pts, uv), {"f_step": 0.0})):
        try:
            C.fit_camera(*bad[0], **bad[1])
            raise AssertionError(f"no error for {bad[1]}")
        except ValueError:
            pass


def test_fit_camera_k1_rules():
    """VM-6: a distortion term is accepted only with enough points, a real improvement, a small and
    invertible value; the focal length stays where the search put it."""
    rng = np.random.default_rng(2)
    true = look_at_camera((0.75, 0.03, 0.42), (0.14, 0.0, 0.0), f=1400.0)
    pts, uv = _calib_points(true, 15, rng, noise=2.0)
    m = C.fit_camera(pts, uv, f_range=(1400.0, 1400.0), fit_k1=True)
    assert m.k1 == 0.0 and "at least 25" in m.k1_note
    accepted = 0
    for seed in range(6):                               # noise only: an extra parameter must not be taken
        rng = np.random.default_rng(10 + seed)
        pts, uv = _calib_points(true, 40, rng, noise=2.0)
        m = C.fit_camera(pts, uv, f_range=(1400.0, 1400.0), fit_k1=True)
        accepted += m.k1 != 0.0
        assert m.f == 1400.0 and abs(m.k1) <= 0.15
    assert accepted == 0, f"k1 accepted on {accepted}/6 distortion-free data sets"
    dist = look_at_camera((0.75, 0.03, 0.42), (0.14, 0.0, 0.0), f=1400.0, k1=0.10)
    rng = np.random.default_rng(3)
    pts, uv = _calib_points(dist, 60, rng, noise=0.3)
    m = C.fit_camera(pts, uv, f_range=(1400.0, 1400.0), fit_k1=True)
    assert abs(m.k1 - 0.10) < 0.03 and m.f == 1400.0 and "accepted" in m.k1_note, (m.k1, m.k1_note)
    assert C._roundtrip_error(m) < 0.5
    wild = C.CameraModel(1400.0, 960.0, 540.0, true.rvec, true.tvec, -0.45, 5.0, 30)
    assert any("cannot be inverted" in p for p in C.check_model(wild))
    below = look_at_camera((0.4, 0.0, -0.3), (0.2, 0.0, 0.0))
    below.rms = 5.0
    assert any("not above" in p for p in C.check_model(below, pts))


def test_leave_one_out():
    """The in-sample rms understates the predictive error: the helper reports the latter in mm."""
    rng = np.random.default_rng(4)
    true = look_at_camera((0.75, 0.03, 0.42), (0.14, 0.0, 0.0), f=1400.0)
    pts, uv = _calib_points(true, 12, rng, noise=2.0)
    m = C.fit_camera(pts, uv, f_range=(1400.0, 1400.0))
    loo = C.leave_one_out(pts, uv, f_range=(1400.0, 1400.0))
    assert len(loo["px"]) == 12 and loo["rms_px"] > m.rms and loo["max_mm"] >= loo["rms_mm"] > 0.3
    assert loo["f_min"] == loo["f_max"] == 1400.0
    exact = C.leave_one_out(*_calib_points(true, 10, rng), f_range=(1400.0, 1400.0))
    assert exact["max_mm"] < 0.05
    # the real calibration points: the stored model is reproduced and its predictive error is ~2x the rms
    path = os.path.join(tazuna_paths.CONFIG_DIR, "calib_points.json")
    if os.path.exists(path):
        cp = json.load(open(path))
        xyz, uv = np.array([p["xyz"] for p in cp]), np.array([p["uv"] for p in cp])
        m = C.fit_camera(xyz, uv, f_range=(CAM.f, CAM.f))
        loo = C.leave_one_out(xyz, uv, f_range=(CAM.f, CAM.f))
        REPORT["calib"] = {"n": len(cp), "rms_px": m.rms, "loo_rms_mm": loo["rms_mm"], "loo_max_mm": loo["max_mm"],
                           "same_as_stored_mm": float(np.linalg.norm(m.center - CAM.center) * 1000)}
        assert m.f == CAM.f and loo["rms_px"] > m.rms


# =========================================================================== vision.py fixes
def test_fit_box_roi_clipping():
    """VM-7: the model outline is clipped to the same window as the mask."""
    true = (0.262, 0.085, math.radians(4.0), 0.111, 0.100, 0.030)
    full = V.project_hull_mask(CAM, V.box_corners(*true, Z0), (C.H, C.W))
    init = [0.255, 0.08, 0.0, 0.10, 0.10, 0.03]
    steps = (0.01, 0.01, 0.1, 0.01, 0.01, 0.0)                      # height known and fixed
    comp = V.largest_component(full, 2000)
    assert comp["roi_cut"] is False
    p, s = V.fit_box(CAM, comp["mask"], Z0, init, steps=steps)
    assert math.hypot(p[0] - true[0], p[1] - true[1]) < 0.002 and s > 0.99 and p[5] == 0.03
    roi = (200, 740, 1800, 1080)                                    # cuts the far 40 rows of the box
    comp = V.largest_component(full, 2000, roi)
    assert comp["roi_cut"] is True
    p_old, s_old = V.fit_box(CAM, comp["mask"], Z0, init, steps=steps)            # as v1 did: no window
    p_new, s_new = V.fit_box(CAM, comp["mask"], Z0, init, steps=steps, roi=roi)
    e_old = math.hypot(p_old[0] - true[0], p_old[1] - true[1])
    e_new = math.hypot(p_new[0] - true[0], p_new[1] - true[1])
    assert e_old > 0.008 and e_new < 0.004 and s_new > 0.99, (e_old, e_new, s_new)
    assert V.largest_component(full, 2000, (200, 430, 1340, 1080))["roi_cut"] is True       # cut on the right
    assert V.largest_component(full, 2000, (200, 430, 1800, 1080))["roi_cut"] is False      # only the image bottom
    for bad in ((500, 500, 400, 600), (0, 0, 10, 10)):
        try:
            V.fit_box(CAM, full, Z0, init, steps=steps, roi=bad)
            raise AssertionError("no error for an empty window")
        except ValueError:
            pass
    # nelder_mead keeps a zero-step parameter exactly fixed
    x, f = V.nelder_mead(lambda v: (v[0] - 1.0) ** 2 + (v[1] - 2.0) ** 2, [0.0, 0.5], [0.3, 0.0])
    assert x[1] == 0.5 and abs(x[0] - 1.0) < 1e-2


def test_grid_labels_and_save_view():
    """VM-8: grid labels survive a crop; an impossible crop is a clear error."""
    img = image("0113_locate_raw.jpg")
    g = V.draw_grid(img, 120)
    assert g.shape == img.shape and not np.array_equal(g, img)
    line_only = img.copy()
    for x in range(0, C.W, 120):
        cv2.line(line_only, (x, 0), (x, C.H - 1), (255, 255, 0), 1)
    for y in range(0, C.H, 120):
        cv2.line(line_only, (0, y), (C.W - 1, y), (255, 255, 0), 1)
    for crop in ((660, 300, 1160, 720), (200, 380, 1800, 1080), (1300, 500, 1700, 900)):
        x0, y0, x1, y1 = crop
        n_label = np.count_nonzero(np.any(g[y0:y1, x0:x1] != line_only[y0:y1, x0:x1], axis=2))
        assert n_label > 400, f"no grid labels inside crop {crop} ({n_label} px)"
    sub = V.draw_grid(img[300:720, 660:1160], 120, offset=(660, 300))
    assert sub.shape == (420, 500, 3) and np.all(sub[:, 720 - 660] == (255, 255, 0)) is not None
    assert np.array_equal(sub[5, 60], (255, 255, 0)) or np.array_equal(sub[200, 60], (255, 255, 0))   # line at x = 720
    out = os.path.join(TD, "_grid_view.jpg")
    assert V.save_view(g, out, crop=(660, 300, 1160, 720), scale=0.5) == out and cv2.imread(out).shape[:2] == (210, 250)
    assert cv2.imread(V.save_view(g, out, crop=(1800, 1000, 2500, 1500))).shape[:2] == (80, 120)   # clipped, not empty
    for kwargs, word in (({"crop": (2000, 1200, 2300, 1400)}, "empty"), ({"crop": (500, 500, 400, 600)}, "empty"),
                         ({"scale": 0.0}, "scale"), ({"scale": float("nan")}, "scale")):
        try:
            V.save_view(g, out, **kwargs)
            raise AssertionError(f"no error for {kwargs}")
        except ValueError as e:
            assert word in str(e)
    try:
        V.draw_grid(img, 2)
        raise AssertionError("no error for a 2 px grid")
    except ValueError:
        pass
    os.remove(out)


def test_find_tip_fixes():
    """VM-5: two jaw blobs are one gripper (lowest point wins, `expect` only gates); a tip in front
    of pale static background is refused with the reason."""
    ref = np.zeros((C.H, C.W, 3), np.uint8)
    ref[:] = TABLE_BGR
    img = ref.copy()
    cv2.rectangle(img, (800, 500), (822, 700), (245, 245, 245), -1)       # jaw A, bottom at row 700
    cv2.rectangle(img, (850, 500), (872, 716), (245, 245, 245), -1)       # jaw B, 28 px away, 16 px lower
    for expect in (None, (811, 700), (861, 716)):
        det = V.find_tip(img, ref, roi=(0, 420, 1920, 1075), expect=expect)
        assert det is not None and abs(det["uv"][1] - 716) <= 1 and 850 <= det["uv"][0] <= 872, (expect, det and det["uv"])
    info = {}
    assert V.find_tip(img, ref, expect=(300, 300), max_dist=100, info=info) is None and "expected tip" in info["reason"]
    info = {}
    assert V.find_tip(ref, ref, info=info) is None and info["reason"]
    # the same gripper in front of a pale (glare-like) static patch: refused
    ref2 = ref.copy()
    cv2.rectangle(ref2, (760, 660), (920, 760), (165, 158, 185), -1)
    img2 = ref2.copy()
    cv2.rectangle(img2, (800, 500), (822, 700), (245, 245, 245), -1)
    cv2.rectangle(img2, (850, 500), (872, 716), (245, 245, 245), -1)
    info = {}
    assert V.find_tip(img2, ref2, roi=(0, 420, 1920, 1075), info=info) is None
    assert "static low-saturation background" in info["reason"], info
    det = V.find_tip(img2, ref2, roi=(0, 420, 1920, 1075), static_frac_max=1.0)        # the check can be relaxed explicitly
    assert det is not None and det["static_pale_fraction"] > 0.15
    # without a reference the old behaviour is kept (no static check possible)
    assert V.find_tip(img, None, roi=(0, 420, 1920, 1075)) is not None


def test_legacy_names_still_work():
    """Names fd.py uses must stay importable and behave."""
    for n in ("red_mask", "black_mask", "white_mask", "components", "largest_component", "polygon", "gripper_mask",
              "find_tip", "box_corners", "project_hull_mask", "iou", "nelder_mead", "fit_box", "draw_grid",
              "draw_points", "draw_box", "draw_axes", "save_view"):
        assert callable(getattr(V, n)), n
    for n in ("grab", "lock_settings", "restore_auto", "CameraModel", "fit_camera", "residuals"):
        assert callable(getattr(C, n)), n
    assert C.W == 1920 and C.H == 1080 and C.DEV.startswith("/dev/video") and C.CAM_MODEL_PATH.endswith("camera_model.json")
    for n in ("load", "save", "project", "ray", "to_plane"):
        assert callable(getattr(C.CameraModel, n)), n
    assert CAM.center.shape == (3,)
    img = image("0113_locate_raw.jpg")
    red = V.largest_component(V.red_mask(img), 1500, (200, 430, 1800, 1080))
    assert red is not None and abs(red["bbox"][0] - 595) < 12 and V.red_mask(img, v_min=165).sum() <= V.red_mask(img).sum()
    assert V.largest_component(V.black_mask(img), 20000, (200, 430, 1800, 1080)) is not None
    assert len(V.polygon(red["mask"])) >= 4 and V.white_mask(img).shape == img.shape[:2]
    vis = img.copy()
    V.draw_points(vis, {"a": (100, 100)})
    V.draw_axes(vis, CAM, (0.15, 0.0, Z0))
    V.draw_box(vis, CAM, V.box_corners(0.25, 0.0, 0.3, 0.04, 0.02, 0.02, Z0))
    assert not np.array_equal(vis, img)
    assert np.allclose(V.box_corners(0.25, 0.0, 0.3, 0.04, 0.02, 0.02, Z0), S.box_corners(0.25, 0.0, 0.3, 0.04, 0.02, 0.02, Z0))
    assert V.iou(red["mask"], red["mask"]) == 1.0


def test_blockfit_capfit_wrappers():
    import blockfit
    import capfit
    r = blockfit.measure_block(image("0113_locate_raw.jpg"), CAM, Z0, roi=(200, 430, 1800, 1080))
    t = TRUTH["red_block"]
    assert r is not None and math.hypot(r["center_xy"][0] - t["center"][0], r["center_xy"][1] - t["center"][1]) < 0.005
    assert r["length"] >= r["width"] and abs(r["height"] - t["size"][2]) < 0.006
    assert abs(ang_diff_deg(r["yaw_deg"], t["long_axis_yaw_deg"])) < 8 and len(r["top_corners"]) == 4
    assert blockfit.measure_block(image("0113_locate_raw.jpg"), CAM, Z0, color="blue") is None
    assert blockfit.measure_block(image("0113_locate_raw.jpg"), CAM, Z0, roi=(0, 0, 300, 300)) is None
    c = capfit.fit_cap(image("0134_cap_scene_raw.jpg"), CAM, Z0, (250, 600, 900, 1075))
    t = TRUTH["white_cap"]
    assert c is not None and math.hypot(c["center_xy"][0] - t["center"][0], c["center_xy"][1] - t["center"][1]) < 0.005
    assert abs(c["diameter"] - t["size"][0]) < 0.005 and abs(c["height"] - t["size"][2]) < 0.006
    assert c["mask"].shape == (C.H, C.W) and c["mask"].any()
    assert capfit.fit_cap(image("0113_locate_raw.jpg"), CAM, Z0) is None


# =========================================================================== fastcam (fake device)
_STOOD_IN = []          # not empty while a test has the hardware lock stood in for (_lock_stood_in)


def _no_device_while_stood_in(event, args):
    """Audit hook, armed only while the hardware lock is stood in for: no camera or servo device is opened and
    v4l2-ctl is not started. (Outside those two tests it does nothing: --live opens the real camera.)"""
    if not _STOOD_IN:
        return
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        p = os.fsdecode(args[0])
        if p.startswith(("/dev/video", "/dev/ttyACM", "/dev/ttyUSB")):
            raise RuntimeError(f"test_scene: hardware opened while the lock was stood in for: {p}")
    if event == "subprocess.Popen" and args and "v4l2" in str(args[0]) + str(args[1] if len(args) > 1 else ""):
        raise RuntimeError("test_scene: v4l2-ctl started while the lock was stood in for")


sys.addaudithook(_no_device_while_stood_in)


def _lock_stood_in(fn):
    """Decorator for a test whose camera and v4l2-ctl are stand-ins: for its duration, and for no longer, the
    hardware lock answers "not locked" (hwlock.info: what camera.py and fastcam.py ask before they touch the
    device). With the user's lock set, the code under test would otherwise refuse before it reaches the test's
    stand-ins (2026-11-13: 48 of 50 on the locked rig).

    While the lock is out of the way nothing real may be in reach: cv2.VideoCapture and subprocess.run are
    refusals unless the test has put its own stand-in there (the test's own save / restore then puts the refusal
    back, and this decorator the real ones), and the audit hook above refuses device files and v4l2-ctl. There
    is no suite-wide scratch lock on purpose: `--live` and every other test keep honouring the user's lock."""
    import functools

    import hwlock

    def no_capture(*a, **k):
        raise AssertionError(f"test_scene: cv2.VideoCapture{a!r} asked for the real camera while the lock was stood in for")

    def no_run(*a, **k):
        raise AssertionError(f"test_scene: subprocess.run{a!r} was called for real while the lock was stood in for")

    @functools.wraps(fn)
    def test(*a, **k):
        real = (hwlock.info, cv2.VideoCapture, C.subprocess.run)
        hwlock.info, cv2.VideoCapture, C.subprocess.run = (lambda: None), no_capture, no_run
        _STOOD_IN.append(fn.__name__)
        try:
            return fn(*a, **k)
        finally:
            _STOOD_IN.pop()
            hwlock.info, cv2.VideoCapture, C.subprocess.run = real

    return test


class _FakeCapture:
    """Stand-in for cv2.VideoCapture: 30 frames per second, frame k is filled with the value k % 250."""
    period = 1.0 / 30.0
    instances = []

    def __init__(self, *a, **k):
        self.t0 = time.time()
        self.n = 0
        self.opened = True
        self.stall_after = None
        self.props = {}
        _FakeCapture.instances.append(self)

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def isOpened(self):
        return self.opened

    def grab(self):
        if self.stall_after is not None and self.n >= self.stall_after:
            time.sleep(0.02)
            return False
        due = self.t0 + (self.n + 1) * self.period
        delay = due - time.time()
        if delay > 0:
            time.sleep(delay)
        self.n += 1
        return True

    def retrieve(self):
        return True, np.full((C.H, C.W, 3), self.n % 250, np.uint8)

    def release(self):
        self.opened = False


@_lock_stood_in
def test_fastcam_with_fake_device():
    import fastcam
    calls = {"lock": 0, "restore": 0}
    saved = (fastcam.cv2.VideoCapture, fastcam.C.lock_settings, fastcam.C.restore_auto, fastcam.C.settings_locked,
             fastcam.C.auto_modes_on)
    fastcam.cv2.VideoCapture = _FakeCapture
    fastcam.C.lock_settings = lambda: calls.__setitem__("lock", calls["lock"] + 1)
    fastcam.C.restore_auto = lambda: calls.__setitem__("restore", calls["restore"] + 1)
    fastcam.C.settings_locked = lambda: True            # "already locked": no extra settling time
    fastcam.C.auto_modes_on = lambda: []                # the lock took effect
    try:
        _FakeCapture.instances.clear()
        with fastcam.FastCamera(lock=True, restore=True, warm_frames=2) as cam:
            assert calls == {"lock": 1, "restore": 0}                     # locked exactly once, before streaming
            dev = _FakeCapture.instances[-1]
            assert dev.props[cv2.CAP_PROP_FRAME_WIDTH] == C.W and dev.props[cv2.CAP_PROP_FOURCC] == cv2.VideoWriter_fourcc(*"MJPG")
            a = cam.latest()
            assert a.shape == (C.H, C.W, 3) and int(a[0, 0, 0]) > 2            # warm-up frames were dropped
            t0 = time.perf_counter()
            for _ in range(5):
                cam.latest()
            assert (time.perf_counter() - t0) / 5 < 0.05                       # < 50 ms once warm
            time.sleep(0.15)
            t_mark = time.time()
            n_at_mark = dev.n
            fresh = cam.wait_fresh(t_mark)
            # exposed after the mark: at least latency_s later, i.e. 2+ frames after the one current at the mark
            assert int(fresh[0, 0, 0]) >= (n_at_mark + 2) % 250, (int(fresh[0, 0, 0]), n_at_mark)
            assert time.time() - t_mark < 0.5
            avg = cam.latest(n_avg=2)                                          # both frames are newer than the mark
            assert float(avg[0, 0, 0]) >= float(fresh[0, 0, 0]) and avg.dtype == np.uint8
            b = cam.latest()
            b[:] = 0                                                           # the caller owns its copy
            assert cam.latest()[0, 0, 0] != 0
            assert cam.age() < 0.2 and cam.stats["frames"] > 5 and 20 < cam.stats["fps"] < 40
            # the device stops delivering: never an old frame, always an error with the reason
            dev.stall_after = dev.n
            time.sleep(0.4)
            for fn in (lambda: cam.latest(max_age_s=0.2), lambda: cam.wait_fresh(time.time(), timeout_s=0.3)):
                try:
                    fn()
                    raise AssertionError("a stale frame was handed out")
                except RuntimeError as e:
                    assert "camera" in str(e)
        assert calls == {"lock": 1, "restore": 1} and not dev.opened               # restored and released
        # restore=False (as fd.py uses it): no restore; lock=False: no lock; double close is harmless
        cam = fastcam.FastCamera(lock=True, restore=False, warm_frames=0)
        assert cam.__enter__() is cam
        cam.close()
        cam.close()
        assert calls == {"lock": 2, "restore": 1}
        cam = fastcam.FastCamera(lock=False, restore=True, warm_frames=0)
        cam.close()
        assert calls == {"lock": 2, "restore": 1}
        try:
            cam.latest()
            raise AssertionError("latest() after close did not fail")
        except RuntimeError as e:
            assert "not running" in str(e)

        class _Dead(_FakeCapture):
            def isOpened(self):
                return False

        fastcam.cv2.VideoCapture = _Dead
        try:
            fastcam.FastCamera(lock=True, restore=True)
            raise AssertionError("an unopened device was accepted")
        except RuntimeError as e:
            assert "cannot open camera" in str(e)
        assert calls == {"lock": 3, "restore": 2}                                  # settings put back after the failure
        # settings had to be changed (camera was in automatic mode): the first settle_s seconds are dropped
        fastcam.cv2.VideoCapture = _FakeCapture
        fastcam.C.settings_locked = lambda: False
        t0 = time.time()
        cam = fastcam.FastCamera(lock=True, restore=False, warm_frames=0, settle_s=0.4)
        first = cam.latest()
        assert time.time() - t0 >= 0.4 and int(first[0, 0, 0]) >= 11, (time.time() - t0, int(first[0, 0, 0]))
        assert cam.stats["settings_changed"] is True
        cam.close()
        # the camera cannot be asked (None): carry on, but treat the settings as changed
        fastcam.C.settings_locked = lambda: None
        fastcam.C.auto_modes_on = lambda: None
        cam = fastcam.FastCamera(lock=True, restore=False, warm_frames=0, settle_s=0.2)
        assert cam.latest().shape == (C.H, C.W, 3) and cam.stats["settings_changed"] is True
        cam.close()
        # the lock did not take (a control could not be set): refused with the reason, device never opened,
        # and the settings are put back when restore=True
        fastcam.C.auto_modes_on = lambda: ["auto_exposure=3", "focus_automatic_continuous=1"]
        n_dev, n_restore = len(_FakeCapture.instances), calls["restore"]
        for restore in (True, False):
            try:
                fastcam.FastCamera(lock=True, restore=restore)
                raise AssertionError("a camera that is still in automatic mode was accepted")
            except RuntimeError as e:
                assert "could not be locked" in str(e) and "auto_exposure=3" in str(e), str(e)
        assert len(_FakeCapture.instances) == n_dev and calls["restore"] == n_restore + 1
        # lock=False never asks
        cam = fastcam.FastCamera(lock=False, restore=False, warm_frames=0)
        assert cam.stats["settings_changed"] is None
        cam.close()
    finally:
        (fastcam.cv2.VideoCapture, fastcam.C.lock_settings, fastcam.C.restore_auto, fastcam.C.settings_locked,
         fastcam.C.auto_modes_on) = saved


@_lock_stood_in
def test_camera_settings_readback():
    """camera.read_settings / settings_locked: the camera's own rounding is not a change, an automatic
    mode or another value is, and 'cannot ask' is reported as unknown (None), never as locked."""
    locked = dict(C.LOCKED)
    saved_read, saved_run = C.read_settings, C.subprocess.run
    try:
        C.read_settings = lambda: dict(locked)
        assert C.settings_locked() is True
        C.read_settings = lambda: dict(locked, gain=28, exposure_time_absolute=332)   # reported while streaming
        assert C.settings_locked() is True
        C.read_settings = lambda: dict(locked, gain=40)
        assert C.settings_locked() is False
        C.read_settings = lambda: dict(locked, focus_absolute=25)
        assert C.settings_locked() is False
        for k, v in C.AUTO:                                                           # any automatic mode still on
            C.read_settings = lambda k=k, v=v: dict(locked, **{k: v})
            assert C.settings_locked() is False, k
        C.read_settings = lambda: {k: v for k, v in locked.items() if k != "gain"}    # incomplete answer
        assert C.settings_locked() is False
        C.read_settings = lambda: {}
        assert C.settings_locked() is None and C.auto_modes_on() is None
        # auto_modes_on: only the mode switches count, by name and value
        C.read_settings = lambda: dict(locked, gain=40)
        assert C.auto_modes_on() == []
        C.read_settings = lambda: dict(locked, auto_exposure=3, white_balance_automatic=1)
        assert C.auto_modes_on() == ["white_balance_automatic=1", "auto_exposure=3"]
        C.read_settings = lambda: {k: v for k, v in locked.items() if k != "focus_automatic_continuous"}
        assert C.auto_modes_on() == ["focus_automatic_continuous=missing"]                # e.g. not the C920
        C.read_settings = saved_read

        class _Out:
            stdout = "white_balance_automatic: 0\ngain: 29\nfocus_absolute: 20\nnonsense\nexposure_time_absolute: x\n"

        C.subprocess.run = lambda *a, **k: _Out()
        assert C.read_settings() == {"white_balance_automatic": 0, "gain": 29, "focus_absolute": 20}
        assert C.settings_locked() is False                                           # most controls missing

        def no_tool(*a, **k):
            raise OSError("v4l2-ctl is not installed")

        C.subprocess.run = no_tool
        assert C.read_settings() == {} and C.settings_locked() is None
    finally:
        C.read_settings, C.subprocess.run = saved_read, saved_run


def test_the_lock_is_stood_in_for_only_inside_the_two_fake_camera_tests():
    """While a test has the hardware lock stood in for (_lock_stood_in), nothing real is in reach: OpenCV's
    capture, subprocess.run (v4l2-ctl), the device files. Afterwards, also after a test that failed, everything
    is what it was; only the two tests with a fake camera run that way; and the lock that counts everywhere else,
    --live included, is the user's."""
    import hwlock
    real_info, real_cap, real_run, env_lock = hwlock.info, cv2.VideoCapture, C.subprocess.run, os.environ.get("TAZUNA_HW_LOCK")
    lock_before = hwlock.info()
    seen = {}

    @_lock_stood_in
    def probe():
        seen["info"], seen["message"] = hwlock.info(), hwlock.message("The camera")
        for name, fn in (("capture", lambda: cv2.VideoCapture(C.DEV, cv2.CAP_V4L2)),
                         ("run", lambda: C.subprocess.run(["v4l2-ctl", "-d", C.DEV, "-C", "gain"], capture_output=True)),
                         ("lock_settings", C.lock_settings), ("restore_auto", C.restore_auto), ("read_settings", C.read_settings),
                         ("grab", C.grab)):
            try:
                fn()
                seen[name] = "reached the device"
            except AssertionError as e:
                seen[name] = str(e)
        for path in ("/dev/video_none_for_tests", "/dev/ttyACM_none_for_tests", "/dev/ttyUSB_none_for_tests"):
            try:
                open(path)
                seen[path] = "opened"
            except RuntimeError as e:
                seen[path] = str(e)
            except OSError as e:
                seen[path] = f"asked for ({e})"
        raise KeyError("a test that fails")

    try:
        probe()
        raise AssertionError("the probe did not run to its end")
    except KeyError:
        pass
    assert seen["info"] is None and seen["message"] is None, seen
    for name in ("capture", "run", "lock_settings", "restore_auto", "read_settings", "grab"):
        assert "for real while the lock was stood in for" in seen[name] or "asked for the real camera" in seen[name], (name, seen[name])
    for path in ("/dev/video_none_for_tests", "/dev/ttyACM_none_for_tests", "/dev/ttyUSB_none_for_tests"):
        assert seen[path] == f"test_scene: hardware opened while the lock was stood in for: {path}", seen[path]
    # afterwards everything is what it was, and this file never names a lock of its own
    assert hwlock.info is real_info and cv2.VideoCapture is real_cap and C.subprocess.run is real_run and _STOOD_IN == []
    assert hwlock.info() == lock_before and os.environ.get("TAZUNA_HW_LOCK") == env_lock
    with open(__file__, encoding="utf-8") as f:
        assert 'environ["TAZUNA_HW_LOCK"] =' not in f.read().replace("'", '"').split("def test_the_lock_is_stood_in_for")[0]
    wrapped = sorted(n for n, f in globals().items() if n.startswith("test_") and hasattr(f, "__wrapped__"))
    assert wrapped == ["test_camera_settings_readback", "test_fastcam_with_fake_device"], wrapped
    # outside those two tests the user's lock is honoured: when it is set, the camera code refuses and asks nothing
    # (not tried on a rig that is not locked: these calls would then go to the real camera)
    if lock_before is not None:
        for fn in (lambda: C._locked("The camera"), C.lock_settings, C.restore_auto, C.grab):
            try:
                fn()
                raise AssertionError("the user's hardware lock was not honoured")
            except hwlock.HardwareLocked:
                pass
        assert C.read_settings() == {} and C.settings_locked() is None


def live_camera_check():
    """Only with --live: the real C920 through FastCamera, then back to automatic mode."""
    import fastcam
    try:
        was_locked = C.settings_locked()
        t0 = time.perf_counter()
        with fastcam.FastCamera(lock=True, restore=True) as cam:
            print(f"LIVE camera: settings were {'already locked' if was_locked else 'not locked (automatic mode)'}; "
                  f"opened in {(time.perf_counter() - t0) * 1000:.0f} ms; locked now: {C.settings_locked()}")
            t0 = time.perf_counter()
            img = cam.latest()
            first = (time.perf_counter() - t0) * 1000
            time.sleep(0.5)
            ts = []
            for _ in range(10):
                t0 = time.perf_counter()
                cam.latest()
                ts.append((time.perf_counter() - t0) * 1000)
                time.sleep(0.03)
            t = time.time()
            cam.wait_fresh(t)
            fresh = (time.time() - t) * 1000
            print(f"LIVE camera: frame {img.shape}, mean brightness {img.mean():.1f}, first latest() {first:.1f} ms, "
                  f"latest() median {np.median(ts):.2f} ms, wait_fresh {fresh:.0f} ms, {cam.stats['fps']:.1f} fps")
            sc = S.perceive(cam.latest(n_avg=2), CAM, CFG)
            print("LIVE perceive:", [(o.kind, o.color) for o in sc.objects], sc.warnings)
    finally:
        C.restore_auto()
        vals = C.read_settings()
        auto = all(vals.get(k) == v for k, v in C.AUTO) if vals else None
        print("LIVE camera: restored to automatic mode (read back: "
              f"{'confirmed' if auto else 'NOT CONFIRMED ' + str(vals) if auto is False else 'unknown'})")


# =========================================================================== report
def print_report():
    if REPORT["accuracy"]:
        print("\nACCURACY against testdata truth (errors: estimate - truth)")
        print("object          frame kind     colour  centre mm  dL mm  dW mm  dH mm  dyaw deg  conf")
        for key, fr, kind, col, dc, dl, dw, dh, dy, conf in REPORT["accuracy"]:
            print(f"{key:15s} {fr}  {kind:8s} {col:6s}  {dc:8.1f}  {dl:+5.1f}  {dw:+5.1f}  {dh:+5.1f}  {dy:+8.1f}  {conf:4.2f}")
    if REPORT["container"]:
        print("\nCONTAINER (black box) per arm-away frame")
        print("frame  centre mm  dL mm  dW mm  rim mm  long-axis dyaw deg  conf")
        for fr, dc, dl, dw, rim, dy, conf in REPORT["container"]:
            print(f"{fr}   {dc:8.1f}  {dl:+5.1f}  {dw:+5.1f}  {rim:6.1f}  {dy:+18.1f}  {conf:4.2f}")
    if "brightness_worst_mm" in REPORT:
        b = REPORT["brightness"]
        print(f"\nBRIGHTNESS x1.05 / x1.10 / x0.95 / x0.90: largest centre shift {REPORT['brightness_worst_mm']:.2f} mm, "
              f"largest height change {max(r[5] for r in b):.2f} mm over {len(b)} object-gain pairs")
    if REPORT["runtime"]:
        r = REPORT["runtime"]
        print(f"\nRUNTIME perceive(), 24 frames: best-of-3 per frame median {r['median_best_ms']:.0f} ms, "
              f"slowest frame {r['max_best_ms']:.0f} ms ({r['worst_frame']}); all runs median {r['median_all_ms']:.0f} ms, "
              f"max {r['max_all_ms']:.0f} ms (limit 350 ms)")
    if REPORT.get("takeout"):
        print("\nTAKE-OUT frames of 2026-11-11 (tests/testdata/takeout_20261111): the box (113 x 100.5 mm outside) and what is seen")
        print("frame                 L mm   W mm  rim mm  yaw deg  conf  other objects (colour kind confidence)")
        for fr, length, width, height, yaw, conf, others in REPORT["takeout"]:
            print(f"{fr:20s} {length:5.1f}  {width:5.1f}  {height:6.1f}  {yaw:7.1f}  {conf:4.2f}  {others}")
    if "takeout_rim" in REPORT:
        mean, spread, floors = REPORT["takeout_rim"]
        print(f"   rim height measured on the near outer wall: mean {mean:.1f} mm, spread over the views {spread:.1f} mm; "
              f"floor thickness that follows from the inner wall: {floors} mm")
    if REPORT.get("takeout_cap"):
        print("\nTHE CAP (31 mm across, 16.5 mm high, open side up) as an open-top object")
        print("frame                 where          d mm   h mm  wall mm  inner wall seen down to, above its base mm  conf")
        for fr, where, d, h, wall, bottom, conf in REPORT["takeout_cap"]:
            print(f"{fr:20s}  {where:13s} {d:5.1f}  {h:5.1f}  {'   -' if wall is None else f'{wall:4.1f}'}     "
                  f"{'     -' if bottom is None else f'{bottom:6.1f}'}                                     {conf:4.2f}")
    if REPORT.get("takeout_runtime"):
        r = REPORT["takeout_runtime"]
        print(f"\nRUNTIME perceive(), {r['n']} take-out frames: best-of-3 per frame median {r['median_ms']:.0f} ms, "
              f"slowest frame {r['max_ms']:.0f} ms ({r['worst']}) (limit 350 ms)")
    if REPORT["synthetic"]:
        s = REPORT["synthetic"]
        print(f"\nSYNTHETIC {s['n']} single objects (8 colours, 15..60 mm, 8 yaws, 4 places): centre max {s['centre_max']:.2f} mm "
              f"(median {s['centre_med']:.2f}), length/width max {s['size_max']:.2f} mm, height max {s['height_max']:.2f} mm, "
              f"yaw max {s['yaw_max']:.2f} deg")
    if "calib" in REPORT:
        c = REPORT["calib"]
        print(f"\nCAMERA calibration data ({c['n']} points, f pinned): in-sample rms {c['rms_px']:.1f} px; leave-one-out "
              f"{c['loo_rms_mm']:.1f} mm rms, {c['loo_max_mm']:.1f} mm max; refit differs from camera_model.json by "
              f"{c['same_as_stored_mm']:.2f} mm")
    if REPORT["jaws"]:
        print("\nJAWS (detect_jaws): camera minus kinematics at nose height, base frame, mm")
        print("frame name            jaw deg  ok  conf  dxy used        fixed: seen dxy             moving: seen dxy            ms")
        for k, nm, jaw, r in REPORT["jaws"]:
            def cell(j):
                d = "       -      " if j["dxy_base"] is None else f"({j['dxy_base'][0] * 1000:+5.1f},{j['dxy_base'][1] * 1000:+5.1f})"
                return f"{'yes' if j['visible'] else 'no ':3s} {d}"
            d = "       -      " if r["dxy_base"] is None else f"({r['dxy_base'][0] * 1000:+5.1f},{r['dxy_base'][1] * 1000:+5.1f})"
            extra = "one body" if r.get("located_as_one_body") else ("cross only" if r.get("along_only_from_fixed") else "")
            print(f"{k}  {nm[:15]:15s} {jaw:6.1f}  {'yes' if r['ok'] else 'no ':3s} {r['confidence']:4.2f}  {d}  {cell(r['fixed'])}     "
                  f"{cell(r['moving'])}    {r['ms']:4.0f}  {extra}")
        print("   (no saved frame shows both noses separately and cleanly: 'agree' is exercised by the synthetic tests)")
        for k, nm, jaw, r in REPORT["jaws"]:
            why = "; ".join(f"{p}: {r[p]['reason']}" for p in ("fixed", "moving") if not r[p]["visible"] and r[p]["reason"])
            if why:
                print(f"   {k}: {why[:230]}")


if __name__ == "__main__":
    import traceback
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    live = "--live" in sys.argv
    fails, n = 0, 0
    t_all = time.time()
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn) and (not args or any(a in name for a in args)):
            n += 1
            t0 = time.time()
            try:
                fn()
                print(f"PASS {name} ({time.time() - t0:.1f} s)")
            except Exception:
                fails += 1
                print("FAIL", name)
                traceback.print_exc()
    print_report()
    if live:
        live_camera_check()
    print(f"\ntests: {n}  passed: {n - fails}  failures: {fails}  ({time.time() - t_all:.0f} s)")
    sys.exit(1 if fails else 0)
