"""Tests for the air calibration (tools/calib/aircal.py) and its command `fd calib-air` (tools/calib/aircal_cli.py).
Run: python tests/test_aircal.py [name-part ...]      (no hardware: simulated servo bus, synthetic and saved photos;
                                                       about 10 minutes, JH_QUICK=1 skips four of the saved-photo tests)

What they show:
- the tip-only solver finds a known tip offset (along the fingers, 9 cm beside them, across them) from the real
  kinematics and a known camera, with 3 px noise, wrong candidates in a third of the photos and photos without
  the tip, to 1 mm; it says so when the camera has moved by 3 cm or turned by 2 deg; too few photos are an error,
  not a result;
- the full solver finds camera and tip (6 mm, 2 .. 3 mm), from one pixel per photo (with wrong ones) and from
  candidate lists without a camera to start from;
- the candidate detector finds the tip in synthetic pictures of both brushes (dark rod with a small dark-orange
  point, also loaded with red; blue rod with a large gold head, with that head's own size limits), in every
  direction, and nothing in a picture that did not change;
- the first look keeps its bound, refuses a registered tip that is 9 cm short and passes a good one, on
  synthetic photos and on the two look photos of 2026-11-13; it never passes by colour alone: without a photo
  of the scene it is "not checked", and an arm that already stands at the look pose is turned aside about its
  base for that photo (the REAL tip of a tool 9 cm longer than registered keeps its height) and comes back;
- the planned poses keep the modelled tip above their minimum height and inside the tool's joint limits;
- the whole command on the simulated arm (photos drawn where a known camera would see the brush) finds the tip
  (1.5 mm), flies the low set with the tip the high set gave, and ends folded with the torque on; with a camera
  that has moved the full solve finds camera and tip; a wrong registration stops at the look and nothing else
  moves; a poor first set, and a tip far from its registration, fly no low set; a tool held by its jaw is held
  again before it is measured; --dry writes no goal; an interrupt, a stopped motion and a failing camera say
  where the arm is; --save writes the scratch tool file / camera model (with a backup) and refuses a bad fit
  without --force; --sim without --from / --dry and a locked rig refuse;
- what the review of 2026-11-13 found: --force only stores (with a tool 9 cm longer than registered, every
  combination of options that does not carry the operator's own --no-look stops at the look with the REAL tip
  2 cm up; no option flies a low set with an unmeasured tip); photos taken with the jaw elsewhere, or of
  another tool, are never stored, photos whose records name no tool only with --force; the capture writes
  where the jaw stood and when into every record and stops when a photo cannot be stored; with a painting
  set-up the arm folds where `fd paint rest` would and every hint names that command; a camera model that
  cannot be read is a refusal; --sim after the command is refused and no simulated tool file is written into
  the project folder; every command a `next` names is one the parsers accept;
- what the second review (2026-11-14) found: the fold at the end is planned with the tip the photos gave (a tool
  taped across the fingers, registered on its line of sight 93 mm from its tip as the refused look suggests: the
  REAL tip stays 2 cm up where it ended 17 mm under the table), a single set is solved before the fold, where no
  fold is safe with the solved tip the arm holds and the result is not ok, and every hint names the storing
  before anything that folds; the look does not pass on paint the arm hid in the photo before (synthetic, and
  the verifier's starts near the look pose on the evening's desk): the arm is turned aside first wherever the
  camera model sees it at the place of the look, and what lies where it stood cannot pass; a stored tip leads to
  `fd paint status` when a painting set-up is on record and never to the pen calibration; a tool file without
  a jaw position gets the one read during the photos; the refusal of photos from before a recorded jaw hold
  names the way back; every backticked `fd ...` is parsed exactly as it is printed (no placeholder is filled in);
- what the third review (2026-11-14) found: the arm is folded only with a tip that is confirmed (a fit of the run
  that passed the gates, or a tip on record as measured that the photos do not contradict), else it holds where
  the last photo was taken and the result is not ok (the verifier's two runs: 9 of 21 photos hidden, and the tool
  whose body lies beside the finger line); the look draws everything a tool of 30 cm can reach from the gripper,
  takes its first photo with the look pose turned aside, and takes no look over a background with paint near the
  place of the look (the 24 starts in which a look passed with sideways tools on the evening's desk: 0 pass, by
  the photos alone and with that rule); the hint of a folded arm names the storing first while no tip is on
  record as measured; no command is printed that is refused as typed (the two modules' own texts are read for
  that, and a `calib-air` command is put through the command's own option checks); an abbreviated option is refused;
- what the last verification (2026-11-14) left open: the base turns only with the tool lifted. The first look raises
  the gripper 10 cm straight up without turning it, turns its base there and comes straight down, so that whatever
  is taped on passes over the table 10 cm above where it hung (the REAL tip of tools longer than registered stays
  8 cm and more above the table while the base turns and over the rig's water dish, where it passed at 12 .. 47 mm);
  a lift that ends short does not let the base turn, and where no such way exists the look is refused before
  anything moves;
- the five captures of 2026-11-13 give the numbers of that day, and the morning's photos against the camera
  model from before the camera was moved give the warning (local photos: skipped when missing).

A pass here is a pass on the SIMULATED arm and on saved photos. The command has not run on the real arm.
Every test uses scratch files ($FD_TOOL, $FD_CAMERA_MODEL, $FD_PAINT_SETUP, $FD_RUNS, $TAZUNA_HW_LOCK, scratch
folders): the rig's tool file, camera model, painting set-up, backups/ and runs/ are never written (the last
test checks that).
"""

import os as _os  # the harness modules live in tools/<feature>/ (see tools/tazuna_paths.py)
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "tools"))
import tazuna_paths  # noqa: E402,F401

import argparse
import contextlib
import copy
import glob
import io
import itertools
import json
import math
import os
import re
import shlex
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
T_START = time.time()
os.environ["FD_CONFIG"] = os.path.join(HERE, "testdata", "pen_20261111", "config.json")
_TMP = tempfile.mkdtemp(prefix="tz_aircal_")
TOOL_PATH = os.path.join(_TMP, "tool_active.json")
MODEL_PATH = os.path.join(_TMP, "camera_model.json")
SETUP_PATH = os.path.join(_TMP, "paint_setup.json")    # no such file unless a test writes one: "no painting set-up"
LOCK = os.path.join(_TMP, "hardware.lock")
os.environ["FD_TOOL"] = TOOL_PATH
os.environ["FD_CAMERA_MODEL"] = MODEL_PATH
os.environ["FD_PAINT_SETUP"] = SETUP_PATH
os.environ["FD_PAINT_STATE"] = os.path.join(_TMP, "paint_state.json")
os.environ["FD_RUNS"] = os.path.join(_TMP, "runs")      # (fd's run log, should anything ever write one)
os.environ["TAZUNA_HW_LOCK"] = LOCK

# the camera model of 2026-11-13 (the morning's air calibration: what the photos of that day were taken with)
CAM_DICT = {"f": 1430.0, "cx": 960.0, "cy": 540.0, "k1": 0.0,
            "rvec": [1.612758669942364, 1.6178146130486317, -0.8615937575277688],
            "tvec": [-0.08753385134198806, -0.08456418778922864, 0.6391351033336194],
            "rms_px": 8.891271500942649, "n_points": 55, "image_size": [1920, 1080]}
with open(MODEL_PATH, "w") as _fh:
    json.dump(CAM_DICT, _fh)

RIG_FILES = [os.path.join(tazuna_paths.ROOT, n) for n in ("tool_active.json", "tool_active.sim.json", "paint_setup.json",
                                                          "paint_state.json", "paint_state.sim.json")] + \
            [os.path.join(tazuna_paths.CONFIG_DIR, n) for n in ("camera_model.json", "config.json")]


def _rig_files() -> dict:
    """The rig's own state files as they are now (path -> bytes, None = no file): these tests must leave them
    exactly as they found them."""
    out = {}
    for p in RIG_FILES:
        try:
            with open(p, "rb") as fh:
                out[p] = fh.read()
        except FileNotFoundError:
            out[p] = None
    return out


_RIG_FILES = _rig_files()  # before any test ran
_touched = []


def _no_hardware(event, args):
    """Audit hook: these tests must never open the servo bus or the camera."""
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        p = os.fsdecode(args[0])
        if p.startswith("/dev/ttyACM") or p.startswith("/dev/ttyUSB") or p.startswith("/dev/video"):
            _touched.append(p)
            raise RuntimeError(f"test touched hardware: open({p})")
    if event == "subprocess.Popen" and args and "v4l2" in str(args[0]) + str(args[1] if len(args) > 1 else ""):
        _touched.append("v4l2-ctl")
        raise RuntimeError(f"test touched the camera: {args[:2]}")


sys.addaudithook(_no_hardware)

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import aircal as AC  # noqa: E402
import aircal_cli as CLI  # noqa: E402
import camera as C  # noqa: E402
import fd  # noqa: E402
import hwlock  # noqa: E402
import kinematics as K  # noqa: E402
import tooling as TL  # noqa: E402
from robot import ARM, Arm, MotionAborted, SafetyError, load_config  # noqa: E402
from servo_bus import BusError  # noqa: E402
from sim_bus import SimBus  # noqa: E402

fd.log_event = lambda *a, **k: None  # no run log from tests
# What these tests stand on in other modules. Asserted here, once: while the modules were written side by side some
# tests asked `hasattr(tooling, ...)` and accepted a weaker behaviour (or were skipped) when a name was missing, so
# a renamed function would have left the suite green with a check gone (review of 2026-11-14, R23).
for _name in ("hold_jaw", "jaw_in_place", "jaw_tolerance", "set_measured_tip", "sim_tool_state", "JAW_SAME_TICKS", "JAW_END_TOL_TICKS"):
    assert hasattr(TL, _name), f"tooling.{_name} is gone: tests/test_aircal.py stands on it"
for _name in ("gripper_goal_raw", "on_jaw", "tip_measured"):
    assert _name in TL.Tool.__dataclass_fields__, f"tooling.Tool.{_name} is gone: tests/test_aircal.py stands on it"
assert callable(getattr(fd, "_split_own_words", None)), "fd._split_own_words is gone: the spelling test stands on it"
assert os.path.realpath(fd.RUNS) == os.path.realpath(os.environ["FD_RUNS"]), "fd.py does not honour $FD_RUNS"

# Every result the command returns in this suite is kept: the test near the end checks that each command a
# `next` names is one the parsers accept (`fd calib-air --dry` was printed for a day while only
# `fd calib-air -- --dry` worked).
_RESULTS = []
_cli_main = CLI.main


def _recording_main(argv=None, **kw):
    res = _cli_main(argv, **kw)
    _RESULTS.append(res)
    return res


CLI.main = _recording_main          # (fd.py calls aircal_cli.main by this name too)
np.set_printoptions(precision=4, suppress=True)
CFG = load_config()
TZ = float(CFG["table_z"])
TPR = 4096.0 / (2.0 * math.pi)
CAM = C.CameraModel.load()
RUNS = os.path.join(tazuna_paths.ROOT, "runs", "20261113_110358")       # the day's photos (local, not in git)
THIN_TIP = [0.0160, -0.0011, 0.0558]     # the thin round brush, jaw closed (measured that evening)
SIDE_TIP = [-0.0902, -0.0097, 0.0610]    # its first mount: 9 cm beside the finger line
LONG_TIP = [-0.1217, -0.0016, 0.1141]    # its second mount: 17 cm out, registered as 8 cm along the fingers
FLAT_TIP = [0.0085, 0.0007, 0.0611]      # the flat brush of the morning
SHORT_GUESS = [-0.001, 0.0, 0.0868]      # "0.08 m along the fingers": the guess that was 9 cm short


class Skip(Exception):
    pass


class RecBus(SimBus):
    """SimBus that remembers every pose the simulated arm passes through."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.trace = []

    def _advance(self, dt):
        super()._advance(dt)
        self.trace.append({j: self.pos[j] for j in ARM})


def thin_tool(tip=THIN_TIP, **over):
    kw = dict(name="brush_thin", tip_offset=[float(v) for v in tip], mount="other", pitch_deg=65.0,
              attached_at="2026-11-13T17:17:45", joint_limits_deg={"wrist_flex": [20.0, 97.0], "wrist_roll": [-5.0, 5.0]})
    kw.update(over)
    return TL.Tool(**kw)


def measured_tool(tip=THIN_TIP, **over):
    """The thin brush with its tip ON RECORD AS MEASURED (what `fd calib-air --save` writes): for the tests that are
    about what happens around a capture (where it folds, what the hints say), not about its fit. Since round 3 a
    capture folds the arm only with a confirmed tip: a registered tip that was never measured is none."""
    tool = thin_tool(tip, **over)
    CLI._store_tip(tool, tip, {"at": "2026-11-13T17:25:00", "saved_at": "2026-11-13T17:30:00", "by": "calib-air", "solve": "tip-only",
                               "rms_px": 6.6, "used": 34, "of": 36})
    assert CLI._tip_record(tool) is not None
    return tool


def flat_tool(tip=FLAT_TIP):
    return TL.Tool(name="brush", tip_offset=[float(v) for v in tip], mount="along_fingers", pitch_deg=75.0,
                   attached_at="2026-11-13T12:25:17", joint_limits_deg={"wrist_flex": [20.0, 103.0], "wrist_roll": [-5.0, 5.0]})


def q_of(pos, cfg=CFG):
    z, s = cfg["zero_ticks"], cfg["sign"]
    return np.array([s[j] * (pos[j] - z[j]) / TPR for j in ARM])


def sim_arm(tool):
    """A simulated arm carrying `tool`, limp in the tool's rest pose (as `fd --sim` starts)."""
    bus = RecBus(realtime=False)
    arm = Arm(bus=bus, cfg=load_config(), tool=tool).open()
    TL.sim_start(arm)
    bus.trace.clear()
    return arm, bus


@contextlib.contextmanager
def tool_file(tool):
    """The tool file of the test ($FD_TOOL) holds `tool` (None: no file) for the duration."""
    TL.clear()
    if tool is not None:
        TL.save(tool)
    try:
        yield
    finally:
        TL.clear()


@contextlib.contextmanager
def model_file(data: dict):
    """The camera model file of the test ($FD_CAMERA_MODEL) holds `data` for the duration."""
    with open(MODEL_PATH, "w") as fh:
        json.dump(data, fh)
    try:
        yield
    finally:
        with open(MODEL_PATH, "w") as fh:
            json.dump(CAM_DICT, fh)
        shutil.rmtree(os.path.join(_TMP, "backups"), ignore_errors=True)


def moved_camera(cam, shift=(0.0, 0.0, 0.0), turn_deg=(0.0, 0.0, 0.0)):
    """The camera somewhere else: its centre shifted (base frame, m), turned about its own axes (deg)."""
    r = cv2.Rodrigues(np.radians(np.asarray(turn_deg, dtype=float)))[0] @ cam.R
    c = cam.center + np.asarray(shift, dtype=float)
    return C.CameraModel(cam.f, cam.cx, cam.cy, cv2.Rodrigues(r)[0].ravel(), -r @ c, 0.0, cam.rms, cam.n)


_POSES = {}


def poses(family: str) -> np.ndarray:
    """The joint poses the command would fly with the thin brush ("tip": 35) / the flat one ("full": about 60):
    planned once on the simulated arm, with the real kinematics and the tools' joint limits."""
    if family not in _POSES:
        tool = thin_tool() if family == "tip" else flat_tool()
        arm, _bus = sim_arm(tool)
        qs = []
        for name in CLI.DEFAULT_SETS[family]:
            qs += [p["q"] for p in AC.plan_set(arm, tool, AC.POSE_SETS[family][name])]
        _POSES[family] = np.array(qs)
    return _POSES[family]


def synth(qs, tip, cam, rng, noise=3.0, wrong=1.0 / 3.0, hidden=0.12) -> list:
    """Candidate pixels per photo: the true tip as `cam` sees it (+ noise; missing in `hidden` of the photos and
    wherever it is outside the picture), and in `wrong` of the photos one to three pixels of something else."""
    rot, org = AC._gripper_frames(qs)
    uv = cam.project(org + rot @ np.asarray(tip, dtype=float))
    out = []
    for i in range(len(qs)):
        pts = []
        if 0.0 <= uv[i][0] < 1920.0 and 0.0 <= uv[i][1] < 1080.0 and rng.random() >= hidden:
            pts.append(uv[i] + rng.normal(0.0, noise, 2))
        if rng.random() < wrong:
            pts += [[rng.uniform(0.0, 1920.0), rng.uniform(0.0, 1080.0)] for _ in range(int(rng.integers(1, 4)))]
        out.append(np.asarray(pts, dtype=float).reshape(-1, 2))
    return out


# ----------------------------------------------------------------------------- synthetic photos
def _bgr(h, s, v):
    return tuple(int(x) for x in cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0])


ORANGE = _bgr(12, 210, 110)    # the thin brush's bristles under the "paper" exposure: dark orange
RED = _bgr(177, 220, 130)      # ... loaded with red paint
GOLD = _bgr(20, 170, 200)      # the flat brush's ferrule and bristles
BLUE = _bgr(108, 200, 150)     # its handle
NAVY = (70, 45, 30)            # the thin brush's handle: dark, not blue


def scene() -> np.ndarray:
    """The desk as the camera sees it without the arm: a sheet, a palette with red and orange paint, an orange
    and a red stroke on the sheet (bristle-coloured, elongated, and NOT moving)."""
    img = np.full((1080, 1920, 3), (52, 40, 66), np.uint8)
    cv2.fillConvexPoly(img, np.array([[330, 430], [1130, 425], [1560, 1079], [0, 1079], [0, 760]], np.int32), (205, 208, 206))
    cv2.rectangle(img, (1180, 330), (1680, 760), (226, 228, 225), -1)
    for (x, y), col in (((1270, 420), RED), ((1420, 430), ORANGE), ((1560, 600), _bgr(110, 200, 140))):
        cv2.ellipse(img, (x, y), (48, 30), 0, 0, 360, col, -1)
    cv2.line(img, (420, 980), (470, 880), ORANGE, 9, cv2.LINE_AA)
    cv2.line(img, (700, 1000), (722, 930), RED, 7, cv2.LINE_AA)
    return img


BASE = scene()


def _seg(img, p0, p1, w0, w1, colour):
    d = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    n = np.array([-d[1], d[0]]) / max(float(np.linalg.norm(d)), 1e-9)
    poly = np.array([p0 + n * w0 / 2.0, p1 + n * w1 / 2.0, p1 - n * w1 / 2.0, p0 - n * w0 / 2.0])
    cv2.fillConvexPoly(img, np.round(poly).astype(np.int32), colour, lineType=cv2.LINE_AA)


def draw_2d(img, tip, angle_deg: float, kind: str = "thin", bristles=None, scale: float = 1.0):
    """A brush painted straight into a picture: its tip at pixel `tip`, the handle running off in direction
    angle_deg (0 = to the right, 90 = down). thin: dark rod, pointed bristles 44 x 11 px; flat: blue rod, gold
    head 95 x 34 px. Returns the pixel a little way up the handle (where a gripper would be seen)."""
    tip = np.asarray(tip, dtype=float)
    d = np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))])
    if kind == "thin":
        _seg(img, tip + d * 420 * scale, tip + d * 44 * scale, 16 * scale, 16 * scale, NAVY)
        _seg(img, tip + d * 44 * scale, tip, 11 * scale, 2 * scale, bristles or ORANGE)
    else:
        _seg(img, tip + d * 700 * scale, tip + d * 95 * scale, 20 * scale, 20 * scale, BLUE)
        _seg(img, tip + d * 95 * scale, tip, 34 * scale, 34 * scale, bristles or GOLD)
    return tip + d * 300 * scale


def draw_tool(img, cam, q, tip, kind: str = "thin", bristles=None):
    """The brush where `cam` sees it with the arm at joint pose q: a straight tool that ends at `tip` (gripper
    frame axes) and runs back towards the fingers. thin: 6 mm dark handle, bristles 16 mm long, 4 mm wide, pointed;
    flat: 7 mm blue handle, gold head 34 mm long and 12 mm wide."""
    t = K.fk(np.asarray(q, dtype=float))
    rot, org = t[:3, :3], t[:3, 3]
    tip = np.asarray(tip, dtype=float)
    axis = tip - np.asarray(CFG["tip_offset"], dtype=float)
    axis /= np.linalg.norm(axis)

    def see(back):      # the point `back` metres up the tool from its tip: (pixel, pixels per metre there)
        p = org + rot @ (tip - axis * back)
        return cam.project([p])[0], cam.f / float((cam.R @ p + cam.tvec)[2])

    parts = ((0.170, 0.016, 0.006, 0.006, NAVY), (0.016, 0.0, 0.004, 0.0007, bristles or ORANGE)) if kind == "thin" else \
        ((0.300, 0.034, 0.007, 0.007, BLUE), (0.034, 0.0, 0.012, 0.012, bristles or GOLD))
    for b0, b1, w0, w1, col in parts:
        (p0, s0), (p1, s1) = see(b0), see(b1)
        _seg(img, p0, p1, w0 * s0, w1 * s1, col)
    return img


ARM_WHITE = (232, 234, 236)


def draw_arm(img, cam, q):
    """The arm itself where `cam` sees it at joint pose q: its links as white bars from the base to the fingertip
    (4 .. 6 cm wide, the gripper 3 cm), so that a photo HIDES what lies behind the arm, as a real one does (the
    drawing the verifier of 2026-11-14 used). Without it only the brush is in a synthetic photo."""
    finger = K.tool_offset(*CFG["tip_offset"])
    pts = [np.array([0.0, 0.0, 0.02])] + [f[:3, 3] for f in K.link_frames(q)] + [K.fk(q)[:3, 3], K.fk(q, finger)[:3, 3]]

    def see(p):
        return cam.project([p])[0], cam.f / max(float((cam.R @ np.asarray(p) + cam.tvec)[2]), 0.05)

    for (a, b), w in zip(zip(pts[:-1], pts[1:]), (0.06, 0.05, 0.045, 0.04, 0.045, 0.05, 0.03)):
        (pa, sa), (pb, sb) = see(a), see(b)
        _seg(img, pa, pb, w * sa, w * sb, ARM_WHITE)
        cv2.circle(img, tuple(int(round(v)) for v in pb), int(round(w * sb / 2.0)), ARM_WHITE, -1, cv2.LINE_AA)
    return img


def camera_for(arm, true_tip, cam=None, kind="thin", hand_every: int = 3, shots=None, fail_at=None, hide=None, base=None,
               links: bool = False):
    """A photo function for the simulated arm: the scene with the brush where the camera would see it, its tip
    REALLY at true_tip whatever the tool file says. In every hand_every-th photo something orange and elongated
    that does not belong to the arm lies somewhere else (a hand reaching in). shots collects the joint pose of
    every photo; fail_at: raise this exception at photo number fail_at[0]; hide(n) True: no brush in photo n;
    base: the desk without the arm (default: BASE); links: the arm itself is in the photo too (draw_arm)."""
    cam = CAM if cam is None else cam
    shots = [] if shots is None else shots
    rng = np.random.default_rng(7)
    desk = BASE if base is None else base

    def photo():
        n = len(shots)
        q = arm.read_q()
        shots.append(q)
        if fail_at is not None and n == fail_at[0]:
            raise fail_at[1]
        img = desk.copy()
        if links:
            draw_arm(img, cam, q)
        if not (hide is not None and hide(n)):
            draw_tool(img, cam, q, true_tip, kind)
        if hand_every and n % hand_every == 1:
            c = np.array([rng.uniform(150, 1750), rng.uniform(120, 900)])
            a = rng.uniform(0, math.pi)
            _seg(img, c, c + 70 * np.array([math.cos(a), math.sin(a)]), 16, 12, _bgr(9, 150, 170))
        return img

    photo.shots = shots
    return photo


def run_command(argv, tool, true_tip, prepare=None, **camera):
    """aircal_cli.main(argv) as the REAL command (no --sim), on a simulated arm that carries `tool` while the brush
    is really at true_tip; photos from camera_for; the capture folder is a scratch one. -> (result, arm, bus,
    photo function, capture root). The tool file of the test must hold `tool` (tool_file). prepare(arm, bus):
    called before the command, to leave the simulated arm as some earlier command would have."""
    arm, bus = sim_arm(TL.active())
    assert arm.tool is not None and arm.tool.name == tool.name
    if prepare is not None:
        prepare(arm, bus)
        bus.trace.clear()
    photo = camera_for(arm, true_tip, **camera)
    root = tempfile.mkdtemp(prefix="run_", dir=_TMP)
    saved = (CLI._make_arm, CLI._photo_fn, CLI._run_dir)

    def make(sim):
        assert not sim
        return arm

    CLI._make_arm, CLI._photo_fn, CLI._run_dir = make, (lambda: photo), (lambda: root)
    err = io.StringIO()
    try:
        with contextlib.redirect_stderr(err):
            res = CLI.main(list(argv))
    finally:
        CLI._make_arm, CLI._photo_fn, CLI._run_dir = saved
    json.dumps(res, default=fd._jsonable, allow_nan=False)       # what fd prints must be JSON, strictly
    return res, arm, bus, photo, root


_FOLDERS = {}


def capture_folder(kind: str = "good") -> str:
    """A capture folder as the command leaves it, with synthetic photos of the thin brush (tip really at THIN_TIP,
    registered 7 mm off). "good": the 35 poses of the tip-only sets; "few": 20 of them, the brush in 8 only."""
    if kind not in _FOLDERS:
        folder = os.path.join(_TMP, "capture_" + kind)
        os.makedirs(folder)
        qs = poses("tip") if kind == "good" else poses("tip")[:20]
        recs = []
        rng = np.random.default_rng(3)
        for i, q in enumerate(qs):
            img = BASE.copy()
            if kind == "good" or i % 5 in (0, 2):
                draw_tool(img, CAM, q, THIN_TIP)
            if i % 3 == 1:
                c = np.array([rng.uniform(150, 1750), rng.uniform(120, 900)])
                _seg(img, c, c + np.array([60.0, 25.0]), 16, 12, _bgr(9, 150, 170))
            path = os.path.join(folder, f"high_{i:02d}.jpg")
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            recs.append({"i": i, "q": [float(v) for v in q], "image": path})
        with open(os.path.join(folder, "records_0_high.json"), "w") as fh:
            json.dump({"tool": thin_tool([0.020, 0.0, 0.050]).to_dict(), "records": recs, "skipped": [], "stopped": None}, fh)
        _FOLDERS[kind] = folder
    return _FOLDERS[kind]


def fd_calib_air(*words, sim=False):
    """`fd [--sim] calib-air WORDS` through fd.main, exactly as the operator types it and as every `next` of the
    command spells it -> (result, exit code). There is no second spelling to fall back on: until the evening of
    2026-11-13 fd's own parser refused a first word that looks like an option ("unrecognized arguments: --dry",
    exit code 2, no JSON), this helper then tried `fd calib-air -- --dry`, and the tests passed while the
    operator's command did not work. A SystemExit of fd is a failure here."""
    argv = (["--sim"] if sim else []) + ["calib-air", *words]
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = fd.main(argv)
    except SystemExit as e:
        raise AssertionError(f"`fd {' '.join(argv)}` ended in fd's own parser (exit code {e.code}), without a "
                             f"result: {err.getvalue().strip()[-300:]}") from None
    return json.loads(out.getvalue().strip().splitlines()[-1]), rc


def epoch(stamp: str) -> float:
    """Seconds of a "YYYY-MM-DDTHH:MM:SS" the command wrote (local time)."""
    return time.mktime(time.strptime(stamp, "%Y-%m-%dT%H:%M:%S"))


def real_heights(arm, bus, tool, real_tip) -> list:
    """Height above the table of the tip the tool REALLY has (whatever its file says), at every pose the
    simulated arm passed through."""
    real = TL.Tool.from_dict(tool.to_dict())
    real.tip_offset = [float(v) for v in real_tip]
    return [TL.tip_height(arm, q_of(p), real) for p in bus.trace]


def at_rest_of(real_tip):
    """prepare(arm, bus) for run_command: the simulated arm lies folded as the tool it REALLY carries allows (its
    real tip 5 cm up), not as the registered one would (a simulated arm starts in the rest pose of the tool on
    file: with a tool 9 cm longer than registered that pose has the real tip inside the table, where no real
    arm can be)."""
    def prepare(arm, bus):
        assert TL.sim_place(arm, TL.rest_q(arm, thin_tool(real_tip)))
    return prepare


def held_at_look_pose(real_tip=None):
    """prepare(arm, bus) for run_command: the arm holds at the look pose with the torque on, where a refused look
    leaves it. It is brought there the way the command does it (from the folded pose, at the look's speed): a
    simulated arm that is only placed somewhere and switched on gives way by a degree in its first step, which
    an arm that already holds does not. real_tip: the fold it starts from is the one the real tool allows."""
    def prepare(arm, bus):
        if real_tip is not None:
            assert TL.sim_place(arm, TL.rest_q(arm, thin_tool(real_tip)))
        TL._power(arm, None)
        TL.free_move(arm, AC.look_pose(arm, arm.tool, CAM)["q"], AC.LOOK_SPEED_DEG_S, settle=0.5)
    return prepare


@contextlib.contextmanager
def short_sets():
    """The pose sets of the tip-only solve cut down to 8 high poses (and the usual low ones): for tests of what
    happens around a capture, not of its fit (8 photos never pass the gates, so the low set is not flown)."""
    saved = AC.POSE_SETS["tip"]
    AC.POSE_SETS["tip"] = dict(saved, high={"grid": ((0.17, 0.25), (-0.10, 0.16), (0.08, 0.13)), "min_height": 0.06})
    try:
        yield
    finally:
        AC.POSE_SETS["tip"] = saved


# the painting set-up of 2026-11-13 evening, as far as folding the arm away needs it (canvas, photo pose, what
# stands on the table): written to $FD_PAINT_SETUP by the tests that want "a painting set-up is on record"
PAINT_SETUP = {"photo_q_deg": [-89.2, -61.7, 36.2, 85.6, 0.0],
               "canvas": {"origin_xy": [0.146, -0.118], "size_mm": [224, 153], "pitch_deg": 65.0},
               "contact": {"z0": 0.00542, "x0": 0.2225, "y0": -0.019, "bx": 0.0362, "by": -0.0062}, "press_mm": 5.0,
               "wells": {n: {"xy": xy, "z_mm": 2.5} for n, xy in (("red", [0.0716, 0.2651]), ("blue", [0.1453, 0.2583]),
                                                                    ("yellow", [0.1987, 0.1882]), ("black", [0.2578, 0.1763]))},
               "water": {"rinse": {"xy": [0.346, 0.192], "z_mm": 6.0}},
               "blot": {"xy": [0.354, 0.012], "size": [0.03, 0.04], "z": 0.006}, "paints": {}}


@contextlib.contextmanager
def paint_setup(data=PAINT_SETUP):
    """The painting set-up file of the test ($FD_PAINT_SETUP) holds `data` (a dict, or text as it is: a damaged
    file) for the duration; afterwards there is none again."""
    with open(SETUP_PATH, "w", encoding="utf-8") as fh:
        if isinstance(data, str):
            fh.write(data)
        else:
            json.dump(data, fh)
    try:
        yield
    finally:
        os.remove(SETUP_PATH)


@contextlib.contextmanager
def any_background():
    """The clean-background rule of the look (aircal.look_background, round 3) switched off in memory: for the tests
    of what the look does WITHOUT it over a desk with paint near the place of the look (with it, such a look is
    refused after the photo before, and nothing else of the look is exercised)."""
    real = AC.look_background
    AC.look_background = lambda *a, **k: dict(real(*a, **k), clean=True)
    try:
        yield
    finally:
        AC.look_background = real


FOLD_KEEP = re.compile(r"`fd tool rest --pan=(-?\d+(?:\.\d)?) --keep-torque`")


def folds_where_it_points(text: str, arm_report=None) -> bool:
    """Does a hint name `fd tool rest --pan=A --keep-torque` as a command that works as typed, with A the base angle
    the result reports for the arm (when given)? (Until 2026-11-14 the hint read `--pan P`, which fd refuses.)"""
    m = FOLD_KEEP.search(text)
    if not m or "--pan P" in text:
        return False
    return arm_report is None or abs(float(m.group(1)) - float(arm_report["q_deg"][0])) <= 0.11


def raises(exc, fn, *a, **k):
    try:
        fn(*a, **k)
    except exc as e:
        return e
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not raise {exc.__name__}")


# ----------------------------------------------------------------------------- the detector
def test_background_in_bands_is_the_plain_median():
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 256, (37, 21, 3), dtype=np.uint8) for _ in range(6)]
    assert np.array_equal(AC.background(imgs, rows=10), np.median(np.stack(imgs), axis=0).astype(np.uint8))
    assert np.array_equal(AC.background(imgs[:5]), np.median(np.stack(imgs[:5]), axis=0).astype(np.uint8))


def test_candidates_find_the_thin_tip_in_every_direction_and_with_red_paint():
    for angle in (0.0, 35.0, 90.0, 140.0, 180.0, 225.0, 270.0, 315.0):
        for bristles in (ORANGE, RED):
            img = BASE.copy()
            tip = np.array([760.0, 610.0]) + 90.0 * np.array([math.cos(math.radians(angle + 60)), math.sin(math.radians(angle + 60))])
            near = draw_2d(img, tip, angle, "thin", bristles)
            cands = AC.candidates(img, BASE, near)
            assert len(cands) == 1, (angle, cands)
            c = cands[0]
            assert np.linalg.norm(np.subtract(c["tip"], tip)) < 4.0, (angle, c, tip)
            # the other end is where the bristles meet the handle, 44 px up the tool: the tip is the FAR one
            assert 30.0 < np.linalg.norm(np.subtract(c["tail"], tip)) < 50.0 and 16.0 <= c["len_px"] and c["width_px"] <= 22.0
            # the tip does not depend on a handle colour: the same answer for a tool with no handle in the picture
            bare = BASE.copy()
            _seg(bare, tip + 44.0 * (near - tip) / 300.0, tip, 11, 2, bristles)
            assert np.linalg.norm(np.subtract(AC.candidates(bare, BASE, near)[0]["tip"], tip)) < 4.0
            # seen from further away (bristles 26 x 7 px) it is still found; the flat brush's detector, which wants
            # a head of 250 px, sees nothing of it (that is why candidates exist)
            small = BASE.copy()
            near = draw_2d(small, tip, angle, "thin", bristles, scale=0.6)
            got = AC.candidates(small, BASE, near)
            assert len(got) == 1 and np.linalg.norm(np.subtract(got[0]["tip"], tip)) < 4.0, (angle, got)
            assert AC.detect_tip(small, BASE) is None, angle


def test_candidates_find_the_flat_head_with_its_own_sizes_and_detect_tip_still_does():
    for angle in (20.0, 110.0, 200.0, 250.0, 300.0):
        img = BASE.copy()
        tip = np.array([900.0, 600.0])
        near = draw_2d(img, tip, angle, "flat")
        # a head of 95 x 34 px is no thin brush: the evening's size limits leave it out ...
        assert AC.candidates(img, BASE, near) == [], angle
        # ... the limits are parameters: with the flat head's own, the far end of the head is the tip
        cands = AC.candidates(img, BASE, near, **AC.FLAT_HEAD)
        assert len(cands) == 1 and np.linalg.norm(np.subtract(cands[0]["tip"], tip)) < 6.0, (angle, cands)
        head = AC.detect_tip(img, BASE)
        assert head is not None and head["how"] == "handle" and np.linalg.norm(np.subtract(head["tip"], tip)) < 6.0, (angle, head)


def test_candidates_see_nothing_in_a_picture_that_did_not_change():
    near = (700.0, 400.0)
    assert AC.candidates(BASE, BASE, near) == [] and AC.candidates(BASE, BASE, near, **AC.FLAT_HEAD) == []
    assert AC.detect_tip(BASE, BASE) is None
    # by colour alone (no background) the strokes on the sheet ARE candidates: what the background is for
    assert len(AC.candidates(BASE, None, near)) >= 2
    # a thing that is too far from the gripper, too small, too large or not elongated is none
    img = BASE.copy()
    draw_2d(img, (300.0, 300.0), 0.0, "thin")
    assert AC.candidates(img, BASE, (1500.0, 900.0)) == [] and len(AC.candidates(img, BASE, None)) == 1
    img = BASE.copy()
    cv2.circle(img, (800, 600), 14, ORANGE, -1)
    cv2.line(img, (500, 520), (508, 524), ORANGE, 3)
    cv2.line(img, (300, 700), (700, 760), ORANGE, 24)
    assert AC.candidates(img, BASE, (700.0, 600.0)) == []


# ----------------------------------------------------------------------------- tip only
def test_tip_solver_finds_a_known_offset_among_wrong_candidates():
    """Tolerance 1 mm. With 3 px noise (per axis) the formal error is 0.11 .. 0.22 mm; wrong candidates and the
    25 px selection radius add nothing systematic (nine runs with other seeds were 0.07 .. 0.61 mm off). The tip
    9 cm beside the fingers is outside the picture in 16 of the 35 poses, which were planned for a tool along
    the fingers: it is found from the 15 .. 17 that show it."""
    qs = poses("tip")
    rng = np.random.default_rng(11)
    for tip, guess, at_least in ((THIN_TIP, None, 28), (SIDE_TIP, SHORT_GUESS, 12), ([0.060, 0.004, 0.012], [0.055, 0.0, 0.0068], 28)):
        cands = synth(qs, tip, CAM, rng)
        n_true = sum(1 for c in cands if len(c))
        fit = AC.solve_tip(qs, cands, CAM, tip_guess=guess)
        err = float(np.linalg.norm(np.asarray(fit["tip_offset"]) - tip)) * 1000.0
        assert err < 1.0, (tip, fit["tip_offset"], err)
        assert at_least <= fit["used"] <= fit["n"] <= n_true and fit["photos"] == len(qs), (tip, fit["used"])
        assert 2.0 < fit["rms_px"] < 6.0 and fit["max_px"] <= 25.0 and max(fit["sigma"]) < 0.0005, (fit["rms_px"], fit["sigma"])
        # every photo is either used or has a reason
        assert len(fit["unused"]) == len(qs) - fit["used"] and all(u["why"] for u in fit["unused"])
        assert sorted(fit["index"] + [u["i"] for u in fit["unused"]]) == list(range(len(qs)))
    # the across-the-fingers tip is outside the evening's box (z < 3 cm): found because the box follows the
    # registration there. For the tools of the day the box is the evening's.
    box = AC.vote_box([0.055, 0.0, 0.0068])
    assert box[2][0] < 0.0 and box[2][1] == 0.25 and box[0] == AC.VOTE_BOX[0] and box[1] == AC.VOTE_BOX[1], box
    for tip in (None, THIN_TIP, SIDE_TIP, FLAT_TIP, SHORT_GUESS, [0.0234, -0.0025, 0.0523], [-0.001, 0.0, 0.1268]):
        assert AC.vote_box(tip) == AC.VOTE_BOX, tip
    assert AC.vote_box(LONG_TIP)[0][0] < -0.17 and AC.vote_box([0.0, 0.10, 0.06])[1][1] > 0.149


def test_tip_solver_says_when_the_camera_has_moved():
    """The camera really stands 3 cm from where its model says (sideways, up, along its line of sight), or is
    turned by 2 deg: the solve still returns a tip (wrong by 1 .. 3 cm), and the verdict says the camera moved.
    With the camera where the model says there is no such warning."""
    qs = poses("tip")
    rng = np.random.default_rng(5)
    axes = {"sideways": CAM.R.T @ np.array([1.0, 0.0, 0.0]), "up": CAM.R.T @ np.array([0.0, -1.0, 0.0]),
            "along the line of sight": CAM.R.T @ np.array([0.0, 0.0, 1.0])}
    cases = [("not moved", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))] + [(k, tuple(0.03 * v), (0.0, 0.0, 0.0)) for k, v in axes.items()] \
        + [("turned 2 deg", (0.0, 0.0, 0.0), (0.0, 2.0, 0.0))]
    for name, shift, turn in cases:
        cands = synth(qs, THIN_TIP, moved_camera(CAM, shift, turn), rng)
        fit = AC.solve_tip(qs, cands, CAM)
        fit["check"] = AC.camera_check(qs, cands, CAM, fit["tip_offset"])
        v = CLI.judge("tip", fit, CAM)
        err = float(np.linalg.norm(np.asarray(fit["tip_offset"]) - THIN_TIP)) * 1000.0
        if name == "not moved":
            assert v["moved"] is None and not v["problems"] and err < 1.0, (name, v, err)
            assert fit["check"]["moved_mm"] < 8.0 and fit["check"]["d"] < CLI.CHECK_D, fit["check"]
        else:
            assert v["moved"] and any("camera seems to have moved" in p for p in v["problems"]), (name, v, fit["check"])
            assert fit["check"]["d"] >= CLI.CHECK_D, (name, fit["check"])
            if shift != (0.0, 0.0, 0.0):
                # the cross-check finds the move itself (3 cm), and the tip-only answer is off by centimetres
                assert 22.0 < fit["check"]["moved_mm"] < 38.0 and err > 10.0, (name, fit["check"], err)


def test_too_few_photos_are_an_error():
    qs = poses("tip")
    rng = np.random.default_rng(2)
    good = synth(qs, THIN_TIP, CAM, rng, wrong=0.0, hidden=0.0)
    e = raises(ValueError, AC.solve_tip, qs[:5], good[:5], CAM)
    assert "at least 6 photos" in str(e), e
    # enough photos, but the tip is in only four of them: no consensus
    few = [g if i in (0, 7, 14, 21) else np.zeros((0, 2)) for i, g in enumerate(good)]
    e = raises(ValueError, AC.solve_tip, qs, few, CAM)
    assert "at least 6" in str(e), e
    # candidates that belong to nothing: no offset explains six of them
    junk = [np.array([[rng.uniform(0, 1920), rng.uniform(0, 1080)]]) for _ in qs]
    e = raises(ValueError, AC.solve_tip, qs, junk, CAM)
    assert "agree on one tip offset" in str(e), e
    # the full solve: nine unknowns want eight photos
    uv = np.array([g[0] for g in good])
    e = raises(ValueError, AC.solve, qs[:7], uv[:7], THIN_TIP, CAM.f, CAM.cx, CAM.cy)
    assert "at least 8 poses" in str(e), e
    raises(ValueError, AC.solve, qs, junk, THIN_TIP, CAM.f, CAM.cx, CAM.cy)       # junk is no camera pose either
    raises(ValueError, AC.solve_tip, qs, good[:-1], CAM)       # poses and observations must match


# ----------------------------------------------------------------------------- camera pose + tip
def test_full_solver_finds_camera_and_tip_from_one_pixel_per_photo():
    """The morning's use: one detected pixel per photo, a few of them wrong. Tolerances: camera centre 6 mm, tip
    2 mm (formal errors with 3 px noise and 55 photos: 1 mm and 0.6 mm)."""
    qs = poses("full")
    rng = np.random.default_rng(4)
    true_cam = moved_camera(CAM, (0.04, -0.03, 0.02), (1.5, -2.0, 1.0))      # NOT where the rig's model says
    rot, org = AC._gripper_frames(qs)
    uv = true_cam.project(org + rot @ np.asarray(FLAT_TIP)) + rng.normal(0.0, 3.0, (len(qs), 2))
    bad = rng.choice(len(qs), 5, replace=False)
    uv[bad] += rng.uniform(40.0, 90.0, (5, 2))
    fit = AC.solve(qs, uv, SHORT_GUESS, CAM.f, CAM.cx, CAM.cy)          # start: a tip guess 2.7 cm off, no camera
    assert np.linalg.norm(np.asarray(fit["center"]) - true_cam.center) < 0.006, (fit["center"], true_cam.center)
    assert np.linalg.norm(np.asarray(fit["tip_offset"]) - FLAT_TIP) < 0.002, fit["tip_offset"]
    assert fit["used"] == len(qs) - 5 and fit["n"] == fit["photos"] == len(qs) and 2.5 < fit["rms_px"] < 6.0, fit["used"]
    assert sorted(u["i"] for u in fit["unused"]) == sorted(int(b) for b in bad)
    assert max(fit["sigma"][6:]) < 0.002 and max(fit["center_sigma"]) < 0.004 and fit["cond"] < 200.0
    # the same pixels, the rig's stale model offered as a second start: the same answer
    again = AC.solve(qs, uv, SHORT_GUESS, CAM.f, CAM.cx, CAM.cy, start=(CAM.rvec, CAM.tvec))
    assert np.allclose(again["tip_offset"], fit["tip_offset"], atol=2e-5) and np.allclose(again["center"], fit["center"], atol=5e-5)


def test_full_solver_finds_camera_and_tip_from_candidate_lists():
    """A tool the flat-brush detector does not know: several candidates per photo, most of them wrong, the true
    one missing in some. No camera to start from. Tolerances as above, the tip 3 mm (fewer photos)."""
    qs = poses("tip")
    rng = np.random.default_rng(9)
    true_cam = moved_camera(CAM, (-0.03, 0.04, 0.015), (-1.0, 2.5, 0.5))
    cands = synth(qs, THIN_TIP, true_cam, rng, wrong=0.6)
    fit = AC.solve(qs, cands, [0.020, 0.0, 0.050], CAM.f, CAM.cx, CAM.cy)
    assert np.linalg.norm(np.asarray(fit["center"]) - true_cam.center) < 0.006, (fit["center"], true_cam.center)
    assert np.linalg.norm(np.asarray(fit["tip_offset"]) - THIN_TIP) < 0.003, fit["tip_offset"]
    assert fit["used"] >= 25 and fit["rms_px"] < 6.0, (fit["used"], fit["rms_px"])
    # nothing but junk does not become a camera
    junk = [np.array([[rng.uniform(0, 1920), rng.uniform(0, 1080)] for _ in range(3)]) for _ in qs]
    raises(ValueError, AC.solve, qs, junk, THIN_TIP, CAM.f, CAM.cx, CAM.cy)


# ----------------------------------------------------------------------------- poses
def test_planned_poses_keep_the_modelled_tip_up_and_the_joints_inside_the_tools_limits():
    for family, tool in (("tip", thin_tool()), ("full", flat_tool()), ("tip", thin_tool(SIDE_TIP))):
        arm, bus = sim_arm(tool)
        lim = arm.q_limits()
        assert abs(math.degrees(lim["wrist_flex"][0]) - 20.0) < 1e-6 and abs(math.degrees(lim["wrist_roll"][1]) - 5.0) < 1e-6
        q = arm.read_q()
        for name in CLI.DEFAULT_SETS[family]:
            spec = AC.POSE_SETS[family][name]
            planned = AC.plan_set(arm, tool, spec)
            assert len(planned) >= (8 if tool.tip_offset[0] > -0.05 else 3), (family, name, len(planned))
            for p in planned:
                h = TL.tip_height(arm, p["q"], tool)
                assert h >= spec["min_height"] - 1e-6, (family, name, p["xyz"], h)
                assert abs(h - (p["xyz"][2] - TZ)) < 1e-4
                assert all(lim[j][0] - 1e-9 <= p["q"][i] <= lim[j][1] + 1e-9 for i, j in enumerate(ARM)), (name, np.degrees(p["q"]))
                TL.check_pose(arm, tool, p["q"], min_height=spec["min_height"] - 1e-6)
            # a low set flies at the tool's working pitch where the arm can hold it
            if AC.is_low(spec):
                assert sum(1 for p in planned if abs(p["pitch"] - tool.pitch_deg) < 0.2) >= len(planned) // 2
            chain = AC.plannable(arm, tool, planned, q)
            assert len(chain["poses"]) + len(chain["refused"]) == len(planned) and len(chain["poses"]) >= len(planned) - 2
            assert chain["lowest_m"] >= TL.MIN_FREE_HEIGHT + TL.PLAN_MARGIN - 1e-6, chain["lowest_m"]
            q = chain["q_end"]
        assert bus.io["writes"] == 0 and not bus.log, "planning wrote to the servo bus"
    # explicit places: heights as given, the order kept, an unreachable place left out
    arm, _bus = sim_arm(thin_tool())
    pts = [(0.20, 0.0, 0.05, 65.0), (0.60, 0.0, 0.05, 65.0), (0.24, 0.10, 0.04, 65.0), (0.22, 0.0, 0.005, 65.0)]
    got = AC.plan_points(arm, arm.tool, pts, min_height=0.03)
    assert [p["xyz"][:2] for p in got] == [[0.20, 0.0], [0.24, 0.10]], got
    # the low sets of the day: over the sheet lowest, higher beside it, highest where the dishes stood
    low = AC.low_points((0.16, 0.32), (-0.10, 0.14), (0.035, 0.05, 0.06), 65.0)
    assert low == [(0.16, -0.10, 0.035, 65.0), (0.16, 0.14, 0.05, 65.0), (0.32, 0.14, 0.06, 65.0), (0.32, -0.10, 0.035, 65.0)]


# ----------------------------------------------------------------------------- the first look
def test_look_pose_is_inside_its_bound_and_in_the_picture():
    finger = K.tool_offset(*CFG["tip_offset"])
    for tool in (thin_tool(), thin_tool(SHORT_GUESS), thin_tool([-0.001, 0.0, 0.1768]), flat_tool()):
        arm, bus = sim_arm(tool)
        pose = AC.look_pose(arm, tool, CAM)
        got = K.pose_summary(pose["q"], finger)
        assert got["xyz"][2] - TZ >= AC.LOOK_MIN_HEIGHT - 1e-3 and got["pitch_down_deg"] <= AC.LOOK_MAX_PITCH + 0.5, got
        lim = arm.q_limits()
        assert all(lim[j][0] <= pose["q"][i] <= lim[j][1] for i, j in enumerate(ARM))
        assert 0 <= pose["gripper_px"][0] < 1920 and 0 <= pose["gripper_px"][1] < 1080
        assert AC.LOOK_EDGE_PX <= pose["tip_px"][0] <= 1919 - AC.LOOK_EDGE_PX and AC.LOOK_EDGE_PX <= pose["tip_px"][1] <= 1079 - AC.LOOK_EDGE_PX
        assert pose["tip_height_m"] >= TL.MIN_FREE_HEIGHT + TL.PLAN_MARGIN
        # the bound the module text states: a tool along the fingers up to 20 cm, any direction up to 10 cm, stays 2 cm up
        t = K.fk(pose["q"])
        f0 = np.asarray(CFG["tip_offset"], dtype=float)
        assert (t[:3, 3] + t[:3, :3] @ (f0 + 0.20 * TL._FINGER_AXIS))[2] - TZ >= 0.02 - 1e-3
        for d in (TL._FIXED_JAW_AXIS, -TL._FIXED_JAW_AXIS, np.cross(TL._FINGER_AXIS, TL._FIXED_JAW_AXIS), -TL._FINGER_AXIS,
                  t[:3, :3].T @ np.array([0.0, 0.0, -1.0])):
            assert (t[:3, 3] + t[:3, :3] @ (f0 + 0.10 * d))[2] - TZ >= 0.02 - 1e-3, d
        # the way there from the rest pose does not bring a tool of unknown length lower than it is at either end
        plan = TL.plan_free_move(arm, tool, arm.read_q(), pose["q"])
        assert AC.unknown_tool_dip(arm, tool, CLI._waypoints(plan)) is None
        assert bus.io["writes"] == 0
    # the first candidate is where the evening's looks were taken; a longer registration stands further back
    arm, _ = sim_arm(thin_tool())
    assert AC.look_pose(arm, arm.tool, CAM)["fingertip"] == [0.28, 0.0, 0.12]
    arm, _ = sim_arm(thin_tool([-0.001, 0.0, 0.1768]))
    assert AC.look_pose(arm, arm.tool, CAM)["fingertip"][0] < 0.28
    # a way that dips is found: down to a low pose in the middle
    arm, _ = sim_arm(thin_tool())
    q0, q1 = arm.read_q(), AC.look_pose(arm, arm.tool, CAM)["q"]
    mid, _pitch = TL.work_q(arm, arm.tool, (0.22, 0.0), 0.03)
    dip = AC.unknown_tool_dip(arm, arm.tool, [q0, mid, q1])
    assert dip is not None and dip["lowest_m"] < 0.02 and dip["lowest_m"] < min(dip["ends_m"]) - 0.005, dip
    # no pose at all: a registration that cannot be brought into the picture
    arm, _ = sim_arm(thin_tool([0.0, 0.0, 0.29]))
    e = raises(SafetyError, AC.look_pose, arm, arm.tool, CAM)
    assert "no look pose" in str(e), e


def test_look_check_refuses_a_guess_9cm_short_and_passes_a_good_one():
    arm, _bus = sim_arm(thin_tool(SHORT_GUESS))
    q = AC.look_pose(arm, arm.tool, CAM)["q"]
    before = draw_tool(BASE.copy(), CAM, arm.read_q(), LONG_TIP)       # the arm folded, the brush somewhere else
    # the second mount of the evening: registered 8 cm along the fingers, really 17 cm out
    img = draw_tool(BASE.copy(), CAM, q, LONG_TIP)
    chk = AC.look_check(img, CAM, q, SHORT_GUESS, bg=before)
    assert chk["ok"] is False and chk["nearest_px"] > 200.0 and chk["nearest_mm"] > 50.0, chk
    # the suggestion lies on the line of sight of the candidate: registered there, the look passes
    assert AC.look_check(img, CAM, q, chk["suggest_tip"], bg=before)["nearest_px"] < 5.0
    assert AC.look_check(img, CAM, q, LONG_TIP, bg=before)["ok"] is True
    # a good registration (5 mm off) passes; 2.5 cm too long does not (60 px are about 14 mm here)
    img = draw_tool(BASE.copy(), CAM, q, THIN_TIP)
    good = AC.look_check(img, CAM, q, np.add(THIN_TIP, [0.003, 0.002, 0.003]), bg=before)
    assert good["ok"] is True and good["nearest_px"] < 40.0 and good["limit_px"] == AC.LOOK_PX, good
    far = AC.look_check(img, CAM, q, np.add(THIN_TIP, [0.0, 0.0, 0.025]), bg=before)
    assert far["ok"] is False and 60.0 < far["nearest_px"] < 200.0 and 15.0 < far["nearest_mm"] < 40.0, far
    # the strokes on the sheet are no tips: with the photo from before they are not even candidates ...
    assert all(np.hypot(c[0] - 445, c[1] - 930) > 80 for c in good["candidates"]), good["candidates"]
    # ... and a registration that would be seen right on the orange stroke is still refused
    on_stroke = AC.look_check(BASE, CAM, q, [0.0, 0.0, 0.06], bg=BASE)
    assert on_stroke["ok"] is False and on_stroke["nearest_px"] is None and on_stroke["suggest_tip"] is None
    # the flat brush's head is found by its own detector
    img = draw_tool(BASE.copy(), CAM, q, FLAT_TIP, "flat")
    assert AC.look_check(img, CAM, q, FLAT_TIP, bg=before)["ok"] is True
    # no tool in the picture at all
    none = AC.look_check(BASE, CAM, q, THIN_TIP, bg=before)
    assert none["ok"] is False


def test_look_check_on_the_look_photos_of_2026_11_13():
    """Two photos taken at the look pose that evening (local files). 16:46: the jaw closed, the tip measured three
    minutes later at (-5.1, -2.5, 60.4) mm. 16:43: the third taping, jaw open, while the tool file still said
    (-121.7, -1.6, 114.1): the registration nobody had looked at."""
    shots = {"0034_snap_raw.jpg": [-0.09, -43.43, 50.54, 23.29, -0.05], "0032_snap_raw.jpg": [-0.09, -43.43, 50.54, 23.29, 0.12]}
    paths = {n: os.path.join(RUNS, n) for n in list(shots) + ["0028_snap_raw.jpg"]}
    if not all(os.path.isfile(p) for p in paths.values()):
        raise Skip("the photos of 2026-11-13 are not on this computer")
    before = cv2.imread(paths["0028_snap_raw.jpg"])                     # 40 s earlier, the arm at its stand-by pose
    q = np.radians(shots["0034_snap_raw.jpg"])
    img = cv2.imread(paths["0034_snap_raw.jpg"])
    ok = AC.look_check(img, CAM, q, [-0.0051, -0.0025, 0.0604], bg=before)
    assert ok["ok"] is True and ok["checked"] is True and ok["nearest_px"] < 40.0, ok
    for bg in (before, None):
        long = AC.look_check(img, CAM, q, SHORT_GUESS, bg=bg)           # 2.6 cm too long
        assert long["ok"] is False and 100.0 < long["nearest_px"] < 160.0, long
    # without the photo from before, the same tip is seen just as near, and the look is still not passed: by
    # colour alone a stroke on the desk would be seen as near
    alone = AC.look_check(img, CAM, q, [-0.0051, -0.0025, 0.0604], bg=None)
    assert alone["ok"] is False and alone["checked"] is False and alone["nearest_px"] < 40.0, alone
    img = cv2.imread(paths["0032_snap_raw.jpg"])
    for bg in (before, None):
        bad = AC.look_check(img, CAM, np.radians(shots["0032_snap_raw.jpg"]), LONG_TIP, bg=bg)
        assert bad["ok"] is False and bad["nearest_px"] > 150.0, bad


# ----------------------------------------------------------------------------- the command
def test_options_that_cannot_be_are_refused_before_anything_is_opened():
    made = []
    saved = CLI._make_arm
    CLI._make_arm = lambda sim: made.append(sim)
    try:
        for argv, word in ((["--sim"], "--dry"), (["--tip-only", "--sets", "air,low"], "not allowed"), (["--sets", "air,sky"], "--sets"),
                           (["--sets", "air,air"], "--sets"), (["--dry", "--save"], "--dry"), (["--end", "park"], "--end"),
                           (["--sim", "--dry", "--from", _TMP], "--dry"), (["--nonsense"], "nonsense"),
                           # --force stores, and that is all it does: alone it is refused before anything is opened
                           (["--force"], "needs --save"), (["--no-look", "--force"], "needs --save"),
                           (["--sim", "--from", _TMP, "--force"], "needs --save"), (["--sim", "--dry", "--force"], "needs --save")):
            r = CLI.main(argv)
            assert r["ok"] is False and r["stage"] == "options" and word in r["reason"] and r["next"] and r["say"], (argv, r)
    finally:
        CLI._make_arm = saved
    assert not made and not _touched
    # without a camera model there is nothing a tip-only solve could keep
    os.rename(MODEL_PATH, MODEL_PATH + ".away")
    try:
        r = CLI.main(["--sim", "--dry", "--tip-only"])
        assert r["ok"] is False and r["stage"] == "options" and "--sets air,low" in r["next"], r
        with tool_file(thin_tool()):
            r = CLI.main(["--sim", "--dry"])        # ... and the default is the full solve
            assert r["ok"] and r["mode"] == "full" and [s["name"] for s in r["sets"]] == ["air", "low"], r
            assert r["look"]["registered_px"] is None
    finally:
        os.rename(MODEL_PATH + ".away", MODEL_PATH)
    # no tool, nothing to measure
    with tool_file(None):
        r = CLI.main(["--sim", "--dry"])
        assert r["ok"] is False and r["stage"] == "tool" and "fd tool attach" in r["next"], r


def test_an_abbreviated_option_is_refused():
    """Round 3, T17. argparse took `--no` for --no-look and `--sa` for --save. --no-look is the operator's statement
    that the registered tip was checked against a photo (the first set is then flown on that word alone), --save
    writes the tool file: each option counts only as the whole word. An abbreviation is refused before anything is
    opened."""
    made = []
    saved = CLI._make_arm
    CLI._make_arm = lambda sim: made.append(sim)
    folder = os.path.join(_TMP, "nowhere")
    try:
        with tool_file(thin_tool()):
            for argv in (["--no"], ["--no-l"], ["--no-loo"], ["--sa"], ["--sav", "--no-look"], ["--dr"], ["--sim", "--dr"], ["--tip"],
                         ["--save", "--forc"], ["--fro", folder], ["--en", "stay"], ["--set", "air,low"]):
                r = CLI.main(argv)
                assert r["ok"] is False and r["stage"] == "options" and "unrecognized arguments" in r["reason"], (argv, r)
                assert "`fd calib-air --help`" in r["next"] and "アームは動かしていません" in r["say"], r
            for word in ("--no", "--dr"):                       # ... and through fd, as the operator types it
                r, rc = fd_calib_air(word, sim=True)
                assert rc == 1 and r["stage"] == "options" and r["sim"] is True and "unrecognized arguments" in r["reason"], r
    finally:
        CLI._make_arm = saved
    assert not made, "an abbreviated option got as far as opening the arm"
    # the whole words are what they were
    a = CLI.build_parser().parse_args(["--no-look", "--save", "--force", "--end", "stay", "--from", folder])
    assert a.no_look and a.save and a.force and a.end == "stay" and a.from_dir == folder
    a = CLI.build_parser().parse_args(["--tip-only", "--dry"])
    assert a.tip_only and a.dry and not a.no_look and not a.save


def test_the_hardware_lock_makes_the_real_command_refuse():
    del _touched[:]
    made = []
    saved = CLI._make_arm

    def no_arm(sim):
        made.append(sim)
        raise AssertionError("the arm was asked for although the hardware is locked")

    CLI._make_arm = no_arm
    hwlock.lock("test_aircal")
    try:
        with tool_file(thin_tool()):
            for argv in ([], ["--dry"], ["--sets", "air,low"], ["--no-look", "--save"]):
                r = CLI.main(argv)
                assert r["ok"] is False and r["stage"] == "hardware" and "hardware is locked" in r["reason"], (argv, r)
                assert "fd hw unlock" in r["next"] and r["say"]
            assert not made and not _touched
            # through fd, as the operator types it
            r, rc = fd_calib_air("--dry")
            assert rc == 1 and r["stage"] == "hardware" and not _touched, r
    finally:
        CLI._make_arm = saved
        hwlock.unlock()
    # the simulated plan and the saved photos need no hardware: they work while it is locked
    hwlock.lock("test_aircal")
    try:
        with tool_file(thin_tool()):
            assert CLI.main(["--sim", "--dry"])["ok"] is True
    finally:
        hwlock.unlock()
    assert not os.path.exists(LOCK)


def test_dry_plans_and_writes_no_goal():
    with tool_file(thin_tool()):
        arm, bus = sim_arm(TL.active())
        saved = CLI._make_arm
        CLI._make_arm = lambda sim: arm
        try:
            r = CLI.main(["--sim", "--dry"])
        finally:
            CLI._make_arm = saved
        assert r["ok"] and r["stage"] == "planned" and r["mode"] == "tip-only", r
        assert bus.io["writes"] == 0 and not bus.log and not any(bus.torque.values()), "a dry run wrote to the servo bus"
        high, low = r["sets"]
        assert (high["name"], low["name"]) == ("high", "low") and high["poses"] >= 20 and low["poses"] == 12
        assert high["plannable"] == high["poses"] and low["plannable"] == low["poses"] and r["plannable"] == r["poses"] == high["poses"] + 12
        assert high["lowest_tip_mm"] >= 60.0 and 30.0 <= low["lowest_tip_mm"] <= 36.0 and r["lowest_tip_mm"] == low["lowest_tip_mm"]
        assert "planned again with the tip" in low["note"]
        # the heights are of the MODELLED tip, and the result says so while the tip is a guess
        assert r["tip_measured"] is False and "MODELLED" in r["heights"] and "NOT been measured" in r["heights"], r["heights"]
        assert r["look"]["fingertip"] == [0.28, 0.0, 0.12] and r["look"]["way_safe_for_unknown_length"] is True
        # (round 3: a folded arm is no place for the photo before: a tool of up to 30 cm could cover the place of the
        # look from there. It is taken at the look pose turned aside about the base, over a clean background)
        assert "at the look pose turned" in r["look"]["photo_before"] and "aside" in r["look"]["photo_before"], r["look"]
        # (round 3b, U2) ... and the plan says what that turn passes over, before the look has judged anything
        # (and since the evening of 2026-11-14: the base turns only with the tool lifted; what has to be clear is the
        # place where the arm comes down for that photo)
        over = r["look"]["turn_over"]
        assert "pan" in over and "lifted 10 cm straight up" in over and "at least 10 cm above the table" in over, over
        assert "clear of anything higher than about 2 cm" in over and "one longer than registered hangs lower" in over, over
        assert "lifted 10 cm straight up" in r["look"]["via"] and "straight down" in r["look"]["via"], r["look"]["via"]
        assert "blank sheet" in r["look"]["background"] and "120 px" in r["look"]["background"], r["look"]["background"]
        assert "holds where the last photo was taken" in r["end"]["note"] and "was never measured" in r["end"]["note"], r["end"]
        assert "`fd calib-air`" in r["next"] and "アームは動かしていません" in r["say"] and r["sim"] is True
        # no painting set-up: the fold afterwards is the tool's own rest pose, torque on
        assert r["end"]["torque"] == "on" and r["end"]["fold"]["by"] == "tool" and r["end"]["fold"]["pan_deg"] == TL.REST_PAN_DEG
        # the full solve's sets, and --no-look
        r = CLI.main(["--sim", "--dry", "--sets", "air,low", "--no-look"])
        assert r["ok"] and r["mode"] == "full" and [s["name"] for s in r["sets"]] == ["air", "low"], r
        assert r["sets"][0]["poses"] >= 28 and r["sets"][1]["poses"] >= 25 and r["look"] == {"skipped": "--no-look"}
    # a tool whose tip has been measured: the note says by what
    t = thin_tool()
    CLI._store_tip(t, THIN_TIP, {"at": "2026-11-13T17:25:00", "by": "calib-air", "rms_px": 6.6, "used": 34, "of": 36})
    with tool_file(t):
        r = CLI.main(["--sim", "--dry"])
        assert r["tip_measured"] is True and "as measured on 2026-11-13T17:25:00" in r["heights"], r["heights"]
    # a low set alone, with a tip that is only registered: the plan says that the run will not fly it
    with tool_file(thin_tool()):
        r = CLI.main(["--sim", "--dry", "--sets", "low", "--no-look"])
        assert r["ok"] is False and r["stage"] == "plan" and "has not been measured" in r["reason"], r
        assert r["sets"][0]["flown"] is False and r["sets"][0]["plannable"] >= 25 and r["plannable"] == 0
        assert "--sets air,low --dry" in r["next"]
    # through fd, in the spelling every `next` uses: the wiring, and the "sim" mark
    with tool_file(thin_tool()):
        r, rc = fd_calib_air("--dry", sim=True)
        assert rc == 0 and r["stage"] == "planned" and r["sim"] is True and r["plannable"] >= 30, r
        r, rc = fd_calib_air("--tip-only", "--dry", sim=True)
        assert rc == 0 and r["stage"] == "planned" and r["mode"] == "tip-only", r
        r, rc = fd_calib_air("--sets", "air,low", "--dry", sim=True)
        assert rc == 0 and r["stage"] == "planned" and r["mode"] == "full", r
        r, rc = fd_calib_air(sim=True)                      # no words at all: fd hands that through as it is
        assert rc == 1 and r["stage"] == "options" and r["sim"] is True, r
        r, rc = fd_calib_air("--", "--dry", sim=True)       # the spelling fd needed until 2026-11-13 still works
        assert rc == 0 and r["stage"] == "planned", r


def test_the_whole_command_on_the_simulated_arm_finds_the_tip():
    """Registered 7 mm off; the brush is drawn at its real tip by a known camera; a hand in every third photo.
    Tolerance 1.5 mm: the drawn tip is found to about a pixel, and the bristles' last half millimetre is thinner
    than a pixel in the far photos."""
    registered = thin_tool([0.020, 0.0, 0.050])
    with tool_file(registered):
        before = open(TOOL_PATH, "rb").read()
        res, arm, bus, photo, root = run_command([], registered, THIN_TIP)
        assert res["ok"] is True and res["stage"] == "done" and res["mode"] == "tip-only", res
        err = float(np.linalg.norm(np.asarray(res["tip_offset"]) - THIN_TIP)) * 1000.0
        assert err < 1.5, (res["tip_mm"], err)
        assert res["poses"] == res["photos"] >= 30 and res["used"] >= res["photos"] - 4 and res["rms_px"] < 5.0, res
        assert max(res["tip_sigma_mm"]) < 0.5 and abs(res["from_registered_mm"] - 7.0) < 1.5
        assert res["gates"] == {"passed": True, "problems": []} and "warning" not in res
        assert res["camera_model"]["check"]["moved_mm"] < 8.0
        # the look came first, passed, and both of its photos are there
        look = res["look"]
        assert look["ok"] is True and look["nearest_px"] < AC.LOOK_PX and os.path.isfile(look["photo"]) and os.path.isfile(look["photo_before"])
        assert np.allclose(photo.shots[1], AC.look_pose(arm, registered, CAM)["q"], atol=math.radians(1.0))
        # the high set with the registered tip, then the low set with the tip the high set gave
        high, low = res["sets"]
        assert (high["name"], low["name"]) == ("high", "low") and high["photos"] == high["plannable"] >= 20 and low["photos"] == 12
        assert high["planned_with_tip_mm"] == [20.0, 0.0, 50.0]
        assert np.linalg.norm(np.asarray(low["planned_with_tip_mm"]) / 1000.0 - THIN_TIP) < 0.0015, low["planned_with_tip_mm"]
        assert all(os.path.isfile(s["records"]) for s in res["sets"])
        assert len(glob.glob(os.path.join(res["dir"], "high_*.jpg"))) == high["photos"]
        # every record says when its photo was taken and where the jaw stood for it, as the servo said
        recs, taken_with, _names = CLI.read_records(res["dir"])
        jaw_raw = round(bus.pos["gripper"])
        assert len(recs) == res["photos"] and all(r["gripper_raw"] == jaw_raw and r["gripper_goal_raw"] == jaw_raw for r in recs), recs[0]
        assert all(T_START - 1.0 <= epoch(r["at"]) <= time.time() + 1.0 and r["at"][11:] == r["t"] for r in recs), recs[0]
        assert taken_with["name"] == "brush_thin" and taken_with["attached_at"] == registered.attached_at
        assert res["dir"].startswith(root) and os.path.basename(res["dir"]).startswith("aircal_")
        # the REAL tip never came lower than 2.5 cm (the low set flies 3.5 cm over the sheet by the model)
        real = registered.__class__.from_dict(registered.to_dict())
        real.tip_offset = list(THIN_TIP)
        heights = [TL.tip_height(arm, q_of(p), real) for p in bus.trace]
        assert min(heights) > 0.025, min(heights)
        # folded away, torque ON, and the result says so
        q_end = q_of(bus.pos)
        assert all(bus.torque[j] == 1 for j in ARM) and res["arm"]["torque"] == "on" and "rest pose" in res["arm"]["at"]
        assert TL.in_rest_pose(arm, real, q_end)[0], np.degrees(q_end)
        assert "トルク ON" in res["say"] and f"`fd calib-air --from {res['dir']} --save`" in res["next"]
        # (folded with the tip the photos gave. The tool file has no tip on record as measured and this run stored
        # none: the arm's hint names the storing before anything that folds; round 3, T4)
        assert res["arm"]["tip"] == "modelled with the tip solved in this run" and res["end"]["fold"]["by"] == "tool"
        nxt, stores = res["arm"]["next"], f"`fd calib-air --from {res['dir']} --save`"
        assert "FIRST " + stores in nxt and "on record as measured" in nxt and nxt.index(stores) < nxt.index("`fd tool rest --pan="), nxt
        assert folds_where_it_points(nxt, res["arm"]) and "先にこの測定結果を保存" in res["say"], res["say"]
        # nothing was stored, and the arm carries the registered tool again
        assert res["saved"] is False and open(TOOL_PATH, "rb").read() == before and arm.tool.tip_offset == [0.020, 0.0, 0.050]
        # the photos of that run, solved again without the arm, give the same tip; with --save it is stored
        again = CLI.main(["--sim", "--from", res["dir"]])
        assert again["ok"] and np.allclose(again["tip_offset"], res["tip_offset"], atol=1e-5) and again["used"] == res["used"], again
        assert again["taken"]["jaw"] == {"raw": [jaw_raw, jaw_raw], "squeezed": False, "photos": res["photos"]}, again["taken"]
        t0 = time.time()
        stored = CLI.main(["--sim", "--from", res["dir"], "--save"])
        assert stored["ok"] and stored["saved"] is True and np.allclose(TL.active().tip_offset, res["tip_offset"], atol=1e-5), stored
        # the record of the measurement: "at" is when the last photo was taken, "saved_at" when it was stored
        rec = CLI._tip_record(TL.active())
        assert rec == stored["tip_measured"] and rec["at"] == max(r["at"] for r in recs) and rec["photos"] == res["dir"], rec
        assert t0 - 1.0 <= epoch(rec["saved_at"]) <= time.time() + 1.0 and epoch(rec["at"]) <= epoch(rec["saved_at"])
        assert rec["jaw_raw"] == [jaw_raw, jaw_raw] and rec["jaw_squeezed"] is False


def test_save_in_the_run_and_end_stay():
    registered = thin_tool([0.018, 0.001, 0.058], contact={"x0": 0.2, "y0": 0.0, "a": 0.004, "bx": 0.0, "by": 0.0, "rms_mm": 0.3, "points": []},
                           calibrated={"contact": {"at": "2026-11-13T17:40:00"}})
    with tool_file(registered):
        t0 = time.time()
        res, arm, bus, photo, root = run_command(["--no-look", "--save", "--end", "stay"], registered, THIN_TIP, hand_every=0)
        t1 = time.time()
        assert res["ok"] is True and res["saved"] is True and res["look"] == {"skipped": "--no-look"}, res
        assert not os.path.exists(os.path.join(res["dir"], "look.jpg"))
        t = TL.active()
        assert np.linalg.norm(np.asarray(t.tip_offset) - THIN_TIP) < 0.0015 and t.name == "brush_thin" and t.pitch_deg == 65.0
        assert t.joint_limits_deg == registered.joint_limits_deg and t.attached_at == registered.attached_at
        rec = CLI._tip_record(t)
        assert rec["by"] == "calib-air" and rec["used"] == res["used"] and rec["of"] == res["photos"] and rec["rms_px"] == res["rms_px"]
        # when: the photos and the storing both lie inside this run (seconds, not dates: the suite may run over
        # midnight), the photos first
        assert t0 - 1.0 <= epoch(rec["at"]) <= epoch(rec["saved_at"]) <= t1 + 1.0 and rec["solve"] == "tip-only", rec
        assert rec["photos"] == res["dir"] and "forced" not in rec
        # the tip moved: what was measured with the old one is gone, as after `fd tool set --tip`
        assert t.contact is None and t.calibrated == {} and "contact" in res["dropped"] and res["tip_changed_mm"] > 1.0
        assert "measure it again" in res["next"] and "保存しました" in res["say"]
        # --end stay: where the last photo was taken, torque on
        assert np.allclose(q_of(bus.pos), photo.shots[-1], atol=math.radians(0.5)) and "stay" in res["arm"]["at"]
        assert all(bus.torque[j] == 1 for j in ARM)
        assert folds_where_it_points(res["arm"]["next"], res["arm"]) and "end" not in res


def test_the_full_solve_on_the_simulated_arm_finds_a_camera_that_has_moved():
    """The morning in simulation: the flat brush, registered 1 cm off, and a camera that stands 37 mm and 2 deg from
    where the model on file says. The look (which can only use that model) stops; with --no-look the air set is
    flown, solved, the low set planned with its tip, and camera and tip come out (6 mm, 2 mm) and are stored."""
    true_cam = moved_camera(CAM, (0.03, 0.02, -0.01), (1.0, -1.5, 0.5))
    registered = flat_tool([-0.001, 0.0, 0.0668])
    with tool_file(registered):
        res, arm, bus, photo, root = run_command(["--sets", "air,low"], registered, FLAT_TIP, kind="flat", cam=true_cam)
        assert res["ok"] is False and res["stage"] == "look" and "camera has moved" in res["reason"], res
        assert "--no-look" in res["next"] and "fd tool set --tip" in res["next"] and len(photo.shots) == 2
        model_before = open(MODEL_PATH, "rb").read()
        res, arm, bus, photo, root = run_command(["--sets", "air,low", "--no-look", "--save"], registered, FLAT_TIP, kind="flat", cam=true_cam)
        try:
            assert res["ok"] is True and res["mode"] == "full" and res["detector"] == "head" and res["saved"] is True, res
            assert np.linalg.norm(np.asarray(res["camera"]["center"]) - true_cam.center) < 0.006, (res["camera"], true_cam.center)
            assert np.linalg.norm(np.asarray(res["tip_offset"]) - FLAT_TIP) < 0.002, res["tip_mm"]
            assert 30.0 < res["camera"]["moved_mm"] < 45.0 and res["gates"]["passed"] and res["used"] >= 50, res
            air, low = res["sets"]
            assert (air["name"], low["name"]) == ("air", "low") and air["photos"] >= 28 and low["photos"] >= 25
            assert air["planned_with_tip_mm"] == [-1.0, 0.0, 66.8]
            assert np.linalg.norm(np.asarray(low["planned_with_tip_mm"]) / 1000.0 - FLAT_TIP) < 0.003, low
            assert low["lowest_tip_mm"] >= 29.0 and res["arm"]["torque"] == "on" and "rest pose" in res["arm"]["at"]
            # the camera model on file is the new one, the old one is kept, the tool has the measured tip
            new = json.load(open(MODEL_PATH))
            assert np.linalg.norm(np.asarray(new["center_base_m"]) - true_cam.center) < 0.006 and "fd calib-air" in new["source"]
            assert open(res["files"]["camera_model_backup"], "rb").read() == model_before
            assert os.path.dirname(res["files"]["camera_model_backup"]) == os.path.join(_TMP, "backups")
            t = TL.active()
            assert np.allclose(t.tip_offset, res["tip_offset"], atol=1e-5) and CLI._tip_record(t)["solve"] == "full" and t.pitch_deg == 75.0
        finally:
            with open(MODEL_PATH, "w") as fh:
                json.dump(CAM_DICT, fh)
            shutil.rmtree(os.path.join(_TMP, "backups"), ignore_errors=True)


def test_a_wrong_registration_stops_at_the_look_and_nothing_else_moves():
    """The second mount of the evening in simulation: 8 cm registered, the tip 17 cm out."""
    registered = thin_tool(SHORT_GUESS)
    with tool_file(registered):
        res, arm, bus, photo, root = run_command(["--save"], registered, LONG_TIP)
        assert res["ok"] is False and res["stage"] == "look" and res["saved"] is False, res
        assert "px" in res["reason"] and "mm" in res["reason"] and "not started" in res["reason"]
        assert res["look"]["nearest_px"] > AC.LOOK_PX and res["look"]["nearest_mm"] > 30.0
        assert "fd tool set --tip" in res["next"] and res["look"]["photo"] in res["next"]
        assert "--sets air,low --no-look" in res["next"]        # ... or the tool is where it is registered and the camera moved
        # the one word that goes on is the operator's own look; --force is named only to say that it does not
        assert "Only --no-look skips the look; --force does not" in res["next"]
        # (the photo before: at the look pose turned aside about the base, where no tool of up to 30 cm can cover
        # the place of the look; the background there is clean)
        assert res["look"]["checked"] is True and res["look"]["before"]["taken"].startswith("at the look pose turned aside")
        assert res["look"]["background"]["clean"] is True and res["look"]["background"]["paint_px"] == 0, res["look"]["background"]
        assert np.allclose(photo.shots[0][1:], AC.look_pose(arm, registered, CAM)["q"][1:], atol=math.radians(1.0))
        assert os.path.isfile(res["look"]["photo"]) and "確認" in res["say"]
        # two photos (before, look), the arm at the look pose with the torque on, no calibration pose flown
        assert len(photo.shots) == 2 and not glob.glob(os.path.join(res["dir"], "high_*"))
        q_look = AC.look_pose(arm, registered, CAM)["q"]
        assert np.allclose(q_of(bus.pos), q_look, atol=math.radians(1.0)) and "look pose" in res["arm"]["at"]
        assert res["arm"]["torque"] == "on" and all(bus.torque[j] == 1 for j in ARM)
        gf = [K.fk(q_of(p))[2, 3] - TZ for p in bus.trace]
        assert min(gf) > gf[0] - 0.005, "on the way to the look the gripper came lower than it was folded"
        assert np.allclose(TL.active().tip_offset, SHORT_GUESS)
        # the tip the look suggests, registered: the look passes (the capture then runs: not repeated here)
        s = res["look"]["suggest_tip"]
        img = draw_tool(BASE.copy(), CAM, q_look, LONG_TIP)
        assert AC.look_check(img, CAM, q_look, s, bg=BASE)["ok"] is True


def test_a_poor_first_set_flies_no_low_set_and_is_not_stored():
    """... and without the look (--no-look): the high set is flown with the wrong registration, most photos do not
    show the tip, and the command must neither go down to 3.5 cm with that, nor store it, nor FOLD the arm with it.

    In this simulation the tip is in 6 of the 20 photos and 5 of them agree: no fit at all. The other outcome, a
    tip from few photos that is reported with a warning and refused for storing (stage "save"), is what the real
    second mount gave (8 of 20: test_real_evening_first_and_second_mount) and what the "few" folder gives
    (test_a_bad_fit_is_not_stored_without_force). Until round 3 the arm was then folded with the registered tip
    (stage "solve", with a note); now nothing confirms that tip, and the arm holds where the last photo was taken."""
    registered = thin_tool(SHORT_GUESS)
    with tool_file(registered):
        before = open(TOOL_PATH, "rb").read()
        res, arm, bus, photo, root = run_command(["--no-look", "--save"], registered, LONG_TIP)
        assert [s["name"] for s in res["sets"]] == ["high"] and not glob.glob(os.path.join(res["dir"], "low_*")), res["sets"]
        assert res["saved"] is False and open(TOOL_PATH, "rb").read() == before
        assert res["ok"] is False and res["stage"] == "end" and "tip_offset" not in res, res
        assert "was not folded away" in res["reason"] and "nothing confirms where the tip" in res["reason"], res["reason"]
        assert "the photos of this run give no fit" in res["reason"] and "agree on one tip offset" in res["reason"], res["reason"]
        assert "was never measured" in res["reason"] and res["end"]["fold"] is None and res["end"]["unconfirmed"] is True, res["end"]
        assert res["sets"][0]["photos"] == 20 and res["dir"] in res["next"] and "`fd calib-air --sets air,low`" in res["next"]
        assert len(res["notes"]) == 1 and "no fit" in res["notes"][0] and "the next set was not flown" in res["notes"][0], res["notes"]
        # NOT folded: the arm holds where the last photo was taken, torque on (it did not move after that photo)
        assert np.allclose(q_of(bus.pos), photo.shots[-1], atol=math.radians(0.5)) and all(bus.torque[j] == 1 for j in ARM)
        assert res["arm"]["torque"] == "on" and "not folded" in res["arm"]["at"] and res["arm"]["tip"] == "modelled with the registered tip"
        # what to do: look first; measure and register, then again. NO fold is named (round 3b, U1: every fold is
        # planned with the registered tip, the one nothing confirmed): the tool is taken off by hand instead
        for text in (res["next"], res["arm"]["next"]):
            assert text.startswith("LOOK at the arm and at the tool") and "measure the tool with a ruler" in text, text
            assert "`fd calib-air` again" in text and "type NO fold" in text and "`fd tool detach`" in text, text
            assert "fd paint rest" not in text and "fd tool rest" not in text, text
        assert "畳んでいません" in res["say"] and "目で見て確かめて" in res["say"] and "トルク ON" in res["say"], res["say"]


def test_the_arm_is_folded_only_with_a_tip_that_is_confirmed():
    """Round 3, T1 (dangerous). `fd calib-air` folded with the registered tip whenever no fit of the run passed the
    gates, also when nothing had ever measured that tip. The verifier's case: a tool across the fingers (tip really at
    (85, 0, 45) mm), registered on its line of sight as the refused look prints it (93 mm from the tip); the look
    passes; the brush is not seen in 9 of the 21 capture photos, the fit ((84.8, 0, 44.9) mm from 12 photos) fails
    the 60 % gate, the arm is folded with the registered tip: the REAL tip ended 17 mm under the table (10 mm
    without a painting set-up), result ok.

    Now the arm is folded only with a CONFIRMED tip: the tip of a fit of this run that passed the gates, or a tip
    on record as measured that the photos do not speak against. Else it holds where the last photo was taken,
    torque on, ok false, stage end; the measurement is still reported; nothing is stored. (The verifier's second
    case, a tool that lies beside the finger line and passes the look on paint its handle hid, never gets this far
    any more: the two real-desk tests below.)"""
    registered_tip = on_the_line_of_sight(SHORT_GUESS, ACROSS_TIP)
    hidden = (1, 3, 5, 7, 9, 11, 13, 15, 17)                # (of the capture's photos; the look takes two before them)
    hide9 = lambda n: n >= 2 and (n - 2) in hidden          # noqa: E731

    def held(res, bus, photo, real_tip, words):
        """The result of a run that did NOT fold: where the arm is, what the result says. -> the REAL tip's lowest height."""
        assert res["ok"] is False and res["stage"] == "end" and res["saved"] is False, res
        assert "was not folded away" in res["reason"] and all(w in res["reason"] for w in words), res["reason"]
        assert res["end"]["fold"] is None and res["end"]["unconfirmed"] is True and res["end"]["not_folded"] in res["reason"], res["end"]
        assert np.allclose(q_of(bus.pos), photo.shots[-1], atol=math.radians(0.5)), "the arm moved after the last photo"
        assert all(bus.torque[j] == 1 for j in ARM) and res["arm"]["torque"] == "on" and "not folded" in res["arm"]["at"]
        assert res["next"].startswith("LOOK at the arm and at the tool") and "measure the tool with a ruler" in res["next"], res["next"]
        assert res["arm"]["next"].startswith("LOOK at the arm and at the tool"), res["arm"]["next"]
        assert "畳んでいません" in res["say"] and "目で見て確かめて" in res["say"] and "トルク ON" in res["say"], res["say"]
        return min(real_heights(arm, bus, TL.active(), real_tip)), TL.tip_height(arm, q_of(bus.pos), thin_tool(real_tip))

    # --- the verifier's case, with the painting set-up: a fit that fails the gates, 93 mm from an unmeasured registration
    with tool_file(thin_tool(registered_tip)), paint_setup():
        before = open(TOOL_PATH, "rb").read()
        res, arm, bus, photo, root = run_command(["--save"], TL.active(), ACROSS_TIP, prepare=held_at_look_pose(ACROSS_TIP),
                                                 hand_every=0, hide=hide9)
        assert res["look"]["ok"] is True and [x["name"] for x in res["sets"]] == ["high"], res
        low, end = held(res, bus, photo, ACROSS_TIP, ("was never measured", "does not pass the gates", "93 mm"))
        assert low >= 0.020 and end >= 0.050, (low, end)                # (it ended 17 mm UNDER the table)
        # the measurement is reported all the same, and not stored
        assert np.linalg.norm(np.asarray(res["tip_offset"]) - ACROSS_TIP) < 0.0015 and res["gates"]["passed"] is False, res["tip_mm"]
        assert open(TOOL_PATH, "rb").read() == before and any("nothing was stored" in n and "measure again" in n for n in res["notes"])
        # with a set-up too, NO fold is named (round 3b, U1: `fd paint rest` plans with the registered tip and ended
        # with the real tip under the table): measure and register, or take the tool off by hand
        assert "`fd paint rest" not in res["next"] and "`fd tool rest" not in res["next"], res["next"]
        assert "type NO fold" in res["next"] and "`fd tool detach`" in res["next"] and "`fd calib-air` again" in res["next"], res["next"]
        assert "打たないでください" in res["say"], res["say"]
    # --- without a set-up (it ended 10 mm under the table), and with the operator's --no-look: the same
    with tool_file(thin_tool(registered_tip)):
        res, arm, bus, photo, root = run_command(["--no-look"], TL.active(), ACROSS_TIP, prepare=held_at_look_pose(ACROSS_TIP),
                                                 hand_every=0, hide=lambda n: n in hidden)
        low, end = held(res, bus, photo, ACROSS_TIP, ("was never measured", "does not pass the gates"))
        assert low >= 0.020 and end >= 0.050 and res["look"] == {"skipped": "--no-look"}, (low, end)
        assert "`fd tool rest" not in res["next"] and "`fd paint rest" not in res["next"] and "`fd tool detach`" in res["next"], res["next"]
        # no fit at all (the brush in no photo of the capture): the same, and the reason says so
        res, arm, bus, photo, root = run_command([], TL.active(), ACROSS_TIP, prepare=held_at_look_pose(ACROSS_TIP),
                                                 hand_every=0, hide=lambda n: n >= 2)
        low, end = held(res, bus, photo, ACROSS_TIP, ("was never measured", "give no fit"))
        assert low >= 0.020 and end >= 0.050 and "tip_offset" not in res and res["dir"] in res["next"], (low, end)
        # --end stay folds nothing in any case: the result stays ok, and the arm's hint is the same "look first"
        res, arm, bus, photo, root = run_command(["--end", "stay"], TL.active(), ACROSS_TIP, prepare=held_at_look_pose(ACROSS_TIP),
                                                 hand_every=0, hide=hide9)
        assert res["ok"] is True and "stay" in res["arm"]["at"] and res["gates"]["passed"] is False, res
        assert res["arm"]["next"].startswith("LOOK at the arm and at the tool") and "目で確かめて" in res["say"], res["arm"]["next"]
        assert np.allclose(q_of(bus.pos), photo.shots[-1], atol=math.radians(0.5))
    # --- a tip ON RECORD AS MEASURED: the record stands while the photos do not speak against it ...
    with tool_file(measured_tool([0.020, 0.0, 0.050])), short_sets(), paint_setup():
        res, arm, bus, photo, root = run_command(["--no-look"], TL.active(), THIN_TIP, hand_every=0)
        # (eight photos: their fit fails the gates, and lies 7 mm from the record)
        assert res["ok"] is True and res["gates"]["passed"] is False and res["from_registered_mm"] < 10.0, res
        assert res["end"]["fold"]["by"] == "paint" and res["end"]["fold"]["planned_with_tip_mm"] == [20.0, 0.0, 50.0], res["end"]
        assert "rest pose" in res["arm"]["at"] and res["arm"]["tip"] == "modelled with the registered tip"
        assert "FIRST" not in res["arm"]["next"] and "`fd paint rest` switches it off where the arm is folded now" in res["arm"]["next"]
    # ... and no longer once they do: the same record, the tool really elsewhere (re-taped and not registered anew)
    with tool_file(measured_tool(registered_tip)), paint_setup():
        res, arm, bus, photo, root = run_command(["--no-look"], TL.active(), ACROSS_TIP, prepare=held_at_look_pose(ACROSS_TIP),
                                                 hand_every=0, hide=lambda n: n in hidden)
        low, end = held(res, bus, photo, ACROSS_TIP, ("on record as measured", "does not pass the gates", "93 mm"))
        assert low >= 0.020 and end >= 0.050 and res["end"]["tip_measured"] is True, (low, end)
        assert "register the tool anew" in res["next"] and "no longer on record as measured" in res["next"], res["next"]
        # (photos that give no fit speak against a record too: the poses were planned to have that tip in the picture)
        res, arm, bus, photo, root = run_command(["--no-look"], TL.active(), ACROSS_TIP, prepare=held_at_look_pose(ACROSS_TIP),
                                                 hand_every=0, hide=lambda n: True)
        low, end = held(res, bus, photo, ACROSS_TIP, ("on record as measured", "give no fit"))
        assert low >= 0.020 and end >= 0.050, (low, end)
    # --- the verifier's second case, the fold of the sideways-tool run: the tip really at (0, -60, 177) mm (the body 6 cm
    # beside the finger line), registered 8 cm along the fingers, the rig's kind of set-up on record. The first set was
    # flown after a look that had passed on paint the handle hid (that look is refused now: the real-desk test below;
    # here the set is flown on the operator's word, and where the real tip is meanwhile is that word's matter), fewer
    # than half of its 20 photos showed the tip, and the arm was folded with the registered tip: the real tip ended
    # 7 mm UNDER the table, result ok. Now the arm holds where the last photo was taken, and the result is not ok
    side_tip = [0.0, -0.060, 0.1768]
    with tool_file(thin_tool(SHORT_GUESS)), paint_setup():
        res, arm, bus, photo, root = run_command(["--no-look"], TL.active(), side_tip, hand_every=0,
                                                 prepare=_placed(np.radians([-8.0, -47.7, 39.3, 38.4, 0.0])))
        low, end = held(res, bus, photo, side_tip, ("was never measured",))
        assert len(photo.shots) == 20 and [x["name"] for x in res["sets"]] == ["high"] and end >= 0.040, (len(photo.shots), end)
        assert not res.get("gates", {}).get("passed") and "`fd paint rest" not in res["next"] and "`fd tool detach`" in res["next"], res["next"]


def test_the_hint_of_a_folded_arm_names_the_storing_first_while_no_tip_is_on_record_as_measured():
    """Round 3, T4. After an ok `fd calib-air` without --save the arm lies folded as the tip of this run's photos
    allows, torque on, and its hint read "`fd paint rest` switches it off where the arm is folded now". That holds
    only once the tip is stored: while the tool file has no tip on record as measured, `fd paint rest` folds the arm
    AGAIN, as high as it folds (the verifier: the wrist from 44 to 32 deg, the tip from 40 to 80 mm), and switches the
    torque off in that pose, in which torque-off was never tried on the real arm. Now the hint names the storing
    first, as it already did for a solved tip far from the registration."""
    import paintskill as PS      # (part of the tree: a painting module that does not load is a failure, not a skip)
    registered = thin_tool([0.020, 0.0, 0.050])
    with tool_file(registered), paint_setup():
        res, arm, bus, photo, root = run_command(["--no-look"], registered, THIN_TIP, hand_every=0)
        assert res["ok"] is True and res["saved"] is False and res["gates"]["passed"] is True and res["from_registered_mm"] < 10.0, res
        assert res["end"]["fold"]["by"] == "paint" and res["arm"]["tip"] == "modelled with the tip solved in this run", res["end"]
        assert all(bus.torque[j] == 1 for j in ARM) and res["arm"]["torque"] == "on"
        q_end = q_of(bus.pos)
        stores, nxt = f"`fd calib-air --from {res['dir']} --save`", res["arm"]["next"]
        assert nxt.startswith("the torque is ON; FIRST " + stores) and "on record as measured" in nxt, nxt
        assert nxt.index(stores) < nxt.index("THEN `fd paint rest` switches it off where the arm is folded now"), nxt
        assert stores in res["next"] and "先にこの測定結果を保存" in res["say"] and "未測定" in res["say"], res["say"]
        # why: with the tool file as it is (its tip never measured), `fd paint rest` plans another fold, higher ...
        assert CLI._tip_record(TL.active()) is None
        arm.tool = TL.active()
        unstored = PS.highest_rest(arm, PAINT_SETUP)
        assert np.max(np.abs(unstored["q"][:4] - q_end[:4])) > math.radians(5.0) and unstored["clearance_m"] >= 0.06, unstored["clearance_m"]
        # ... and stored as the hint says (no motion), the fold it chooses is where the arm lies
        stored = CLI.main(["--sim", "--from", res["dir"], "--save"])
        assert stored["saved"] is True and CLI._tip_record(TL.active()) is not None, stored
        arm.tool = TL.active()
        assert np.max(np.abs(PS.rest_choice(arm, PAINT_SETUP)["q"][:4] - q_end[:4])) < math.radians(3.0)
    # the words: the storing stands before whatever folds the arm or switches it off, with and without a set-up, folded
    # or not (the same for a tip far from the registration: test_the_fold_at_the_end_...)
    first = CLI._store_first("fd calib-air --from /d --save", None)
    assert first.startswith("FIRST `fd calib-air --from /d --save` (no motion)") and "never measured" in first and first.endswith("THEN "), first
    none, with_setup = {"path": None, "error": None}, {"path": SETUP_PATH, "error": None}
    for paint, fold in ((none, None), (with_setup, None), (with_setup, {"pan_deg": -89.2}), (none, {"pan_deg": 20.0})):
        plain, hint = CLI._torque_on_next(paint, fold, pan_deg=-12.34), CLI._torque_on_next(paint, fold, first, -12.34)
        assert "FIRST" not in plain and hint.replace(first, "") == plain, (plain, hint)
        assert hint.index("--save") < hint.index("`fd paint rest`" if paint["path"] else "`fd tool rest --pan=-12.3 --keep-torque`"), hint
    # (a tip on record as measured and not contradicted, and a tip stored by the same run: the plain hint. These are in
    # test_the_arm_is_folded_only_with_a_tip_that_is_confirmed and test_the_fold_at_the_end_is_planned_with_the_tip_...)


def test_a_tip_far_from_its_registration_is_reported_before_the_low_set_is_flown():
    """Registered 4 cm too long and never looked at (--no-look). The high set finds the tip, well; but a tip that is
    4 cm from its registration has been confirmed by nothing, and the low set (3.5 cm over the sheet, planned with
    it) is not flown: the tip is reported and can be stored, and the next run looks first."""
    registered = thin_tool([0.016, -0.0011, 0.0958])
    with tool_file(registered):
        res, arm, bus, photo, root = run_command(["--no-look"], registered, THIN_TIP)
        assert res["ok"] is True and [s["name"] for s in res["sets"]] == ["high"] and res["gates"]["passed"], res
        assert np.linalg.norm(np.asarray(res["tip_offset"]) - THIN_TIP) < 0.0015 and abs(res["from_registered_mm"] - 40.0) < 1.5
        assert any("was not flown" in n and "40 mm" in n for n in res["notes"]), res["notes"]
        assert not glob.glob(os.path.join(res["dir"], "low_*")) and "--save" in res["next"] and "rest pose" in res["arm"]["at"]
        # what was measured can be stored; with that tip registered (and marked as measured) a run flies both sets
        assert CLI.main(["--sim", "--from", res["dir"], "--save"])["saved"] is True
        measured = TL.active()
        assert CLI._tip_record(measured)["by"] == "calib-air" and np.linalg.norm(np.asarray(measured.tip_offset) - THIN_TIP) < 0.0015
        res, arm, bus, photo, root = run_command([], measured, THIN_TIP)
        assert res["ok"] is True and [s["name"] for s in res["sets"]] == ["high", "low"] and res["look"]["ok"] is True, res
        assert "as measured on" in res["heights"] and res["from_registered_mm"] < 1.0
        # (folded with the tip its own photos gave: no word about an unmeasured registration)
        assert "solved in this run" in res["arm"]["tip"] and not any("REGISTERED tip" in n for n in res.get("notes", [])), res.get("notes")


def test_an_interrupt_and_a_stopped_motion_say_where_the_arm_is():
    registered = thin_tool([0.020, 0.0, 0.050])
    with tool_file(registered):
        # Ctrl-C (or a cancelled tool call) while the sixth calibration photo is taken
        res, arm, bus, photo, root = run_command(["--save"], registered, THIN_TIP, fail_at=(7, KeyboardInterrupt()))
        assert res["ok"] is False and res["stage"] == "capture" and res["interrupted"] is True and res["saved"] is False, res
        a = res["arm"]
        assert a["torque"] == "on" and len(a["q_deg"]) == 5 and a["tip_height_mm"] > 50.0 and "interrupted" in a["at"]
        assert np.allclose(np.radians(a["q_deg"]), q_of(bus.pos), atol=math.radians(0.2))
        # what to do next: fold with the torque left ON, at a base angle the operator chooses, and why; never the
        # bare `fd tool rest`, which folds over whatever lies in front and switches the torque off
        assert folds_where_it_points(res["next"], a) and "wrist can sink" in res["next"], res["next"]
        assert "`fd tool rest`" not in res["next"] and "fd paint rest" not in res["next"]
        assert f"`fd calib-air --from {res['dir']}`" in res["next"] and "トルク ON" in res["say"]
        # the five photos taken before it are on record
        recs, _tool, names = CLI.read_records(res["dir"])
        assert len(recs) == 5 and names == ["records_0_high.json"] and all(os.path.isfile(r["image"]) for r in recs)
        # a motion that is stopped on the way (the monitor, a tracking error): the same
        calls = {"n": 0}
        real = TL.free_move

        def stopping(arm_, q_to, *a_, **k_):
            calls["n"] += 1
            if calls["n"] == 5:
                raise MotionAborted("free move: stopped because the tool tip was 18 mm above the table (measured)")
            return real(arm_, q_to, *a_, **k_)

        TL.free_move = stopping
        try:
            res, arm, bus, photo, root = run_command(["--no-look"], registered, THIN_TIP)
        finally:
            TL.free_move = real
        assert res["ok"] is False and res["stage"] == "capture" and "stopped" in res["reason"] and "18 mm" in res["reason"], res
        assert res["arm"]["torque"] == "on" and "stopped" in res["arm"]["at"]
        assert folds_where_it_points(res["next"], res["arm"]) and "`fd tool rest`" not in res["next"], res["next"]
        assert len(CLI.read_records(res["dir"])[0]) == 4
        # a camera that fails is no reason to leave the operator without the arm's whereabouts
        res, arm, bus, photo, root = run_command(["--no-look"], registered, THIN_TIP, fail_at=(3, RuntimeError("camera returned no frames")))
        assert res["ok"] is False and res["stage"] == "capture" and "camera returned no frames" in res["reason"], res
        assert res["arm"]["torque"] == "on" and len(res["arm"]["q_deg"]) == 5
        assert folds_where_it_points(res["next"], res["arm"]) and "`fd tool rest`" not in res["next"], res["next"]
        # an interrupt that leaves the servo bus unreadable (cut in the middle of a transaction): still a result
        box = {}

        def fragile(arm_, bus_):
            read = bus_.positions

            def positions():
                if box.get("dead"):
                    raise BusError("no reply from the servos")
                return read()

            bus_.positions = positions

        def cut(n):
            if n == 3:
                box["dead"] = True
                raise KeyboardInterrupt
            return False

        res, arm, bus, photo, root = run_command(["--no-look"], registered, THIN_TIP, prepare=fragile, hide=cut)
        assert res["ok"] is False and res["interrupted"] is True and "no reply" in res["arm"]["unread"], res
        assert "fd state" in res["arm"]["next"] and "interrupted" in res["arm"]["at"] and "読めませんでした" in res["say"]
        assert len(CLI.read_records(res["dir"])[0]) == 3


def test_a_tool_held_by_its_jaw_is_held_again_before_it_is_measured():
    """The thin brush of the evening: taped to the moving jaw, the jaw closed against the fixed one and held by a
    squeeze (raw 2041, goal 2016). The squeeze is a servo goal and is gone after `fd tool rest`; without it the tip
    floats. The command puts it back (tooling.hold_jaw) before the first pose, and refuses a jaw that stands
    elsewhere: the tip would be measured where it will not be while painting."""
    held = thin_tool([0.018, 0.001, 0.058], on_jaw="moving_outer", gripper_raw=2041, gripper_goal_raw=2016)
    with tool_file(held):
        res, arm, bus, photo, root = run_command(["--no-look"], held, THIN_TIP, hand_every=0)
        assert res["ok"] is True and np.linalg.norm(np.asarray(res["tip_offset"]) - THIN_TIP) < 0.0015, res
        jaw = res["sets"][0]["jaw"]
        assert jaw["held"] is True and jaw["goal_raw"] == 2016 and int(bus.goal["gripper"]) == 2016 and round(bus.pos["gripper"]) == 2041, jaw
        assert bus.torque["gripper"] == 1
        # every record has the jaw as the servo said it stood for that photo: closed, with the squeeze on
        recs, taken_with, _names = CLI.read_records(res["dir"])
        assert len(recs) == res["photos"] and all(r["gripper_raw"] == 2041 and r["gripper_goal_raw"] == 2016 for r in recs), recs[0]
        assert (taken_with["gripper_raw"], taken_with["gripper_goal_raw"], taken_with["on_jaw"]) == (2041, 2016, "moving_outer")
        # ... and that is what decides where these photos may be stored: into this tool file, as it is
        stored = CLI.main(["--sim", "--from", res["dir"], "--save"])
        assert stored["saved"] is True and stored["tip_measured"]["jaw_raw"] == [2041, 2041] and stored["tip_measured"]["jaw_squeezed"] is True
        # not once the hold is off the record (`fd tool jaw close --squeeze 0`): the same brush, loose in the jaw's backlash
        loose = TL.active()
        loose.gripper_goal_raw = None
        TL.save(loose)
        again = CLI.main(["--sim", "--from", res["dir"], "--save", "--force"])
        assert again["ok"] is False and again["stage"] == "save" and "held shut with a squeeze" in again["reason"], again
        TL.save(held)

        def opened(arm_, bus_):
            bus_.pos["gripper"] = bus_.goal["gripper"] = 2400.0     # somebody opened the jaw

        before = open(TOOL_PATH, "rb").read()
        res, arm, bus, photo, root = run_command(["--no-look", "--save"], held, THIN_TIP, prepare=opened)
        assert res["ok"] is False and res["stage"] == "tool" and "fd tool jaw close" in res["next"] and res["saved"] is False, res
        assert "raw 2400" in res["reason"] or "2400" in res["reason"], res["reason"]
        q0 = TL.rest_q(arm, held)       # (the torque came on: the simulated arm settles by a degree, no more)
        assert all(np.max(np.abs(q_of(p) - q0)) < math.radians(2.0) for p in bus.trace), "the arm moved although the jaw is not held"
        assert not photo.shots and not glob.glob(os.path.join(res["dir"], "*.jpg")) and open(TOOL_PATH, "rb").read() == before
        assert len(res["arm"]["q_deg"]) == 5 and round(bus.pos["gripper"]) == 2400


# ----------------------------------------------------------------------------- --from and --save
def test_from_a_folder_solves_and_saves_to_the_scratch_tool_file_only():
    folder = capture_folder("good")
    pen_cal = dict(contact={"x0": 0.2, "y0": 0.0, "a": 0.004, "bx": 0.0, "by": 0.0, "rms_mm": 0.3, "points": []},
                   corr_map={"centre": [0.2, 0.0], "kind": "affine", "coef": [[0, 0], [1, 0], [0, 1]], "rms_mm": 1.0, "max_mm": 2.0, "n": 9},
                   calibrated={"tip": {"at": "2026-11-13T17:00:00"}})
    registered = thin_tool([0.020, 0.0, 0.050], notes=["thin round brush"], **pen_cal)
    with tool_file(registered):
        before = open(TOOL_PATH, "rb").read()
        made = []
        saved = CLI._make_arm
        CLI._make_arm = lambda sim: made.append(sim)
        try:
            r = CLI.main(["--sim", "--from", folder])
            assert r["ok"] and r["stage"] == "done" and r["mode"] == "tip-only" and r["saved"] is False, r
            assert np.linalg.norm(np.asarray(r["tip_offset"]) - THIN_TIP) < 0.0015 and r["used"] >= 30 and r["photos"] == 35, r
            assert r["registered_tip_mm"] == [20.0, 0.0, 50.0] and r["gates"]["passed"] and "--save" in r["next"]
            assert open(TOOL_PATH, "rb").read() == before, "a solve without --save changed the tool file"
            r = CLI.main(["--sim", "--from", folder, "--save"])
        finally:
            CLI._make_arm = saved
        assert not made, "--from opened the arm"
        assert r["ok"] and r["saved"] is True and r["files"] == {"tool": TOOL_PATH}, r
        t = TL.active()
        assert np.allclose(t.tip_offset, r["tip_offset"], atol=1e-5) and t.name == "brush_thin" and t.pitch_deg == 65.0
        assert t.joint_limits_deg == registered.joint_limits_deg and t.notes == ["thin round brush"] and t.mount == "other"
        rec = CLI._tip_record(t)
        assert rec["by"] == "calib-air" and rec["rms_px"] == r["rms_px"] and rec["used"] == r["used"] and rec["of"] == 35, rec
        # these records carry no time (as the captures of 2026-11-13): "at" is then when the photo files were written
        mtimes = [os.path.getmtime(p) for p in glob.glob(os.path.join(folder, "high_*.jpg"))]
        assert abs(epoch(rec["at"]) - max(mtimes)) <= 1.0 and epoch(rec["saved_at"]) >= epoch(rec["at"]) and rec["photos"] == folder
        assert "jaw_raw" not in rec                 # ... nor where the jaw stood
        assert t.contact is None and t.corr_map is None and t.calibrated == {} and set(r["dropped"]) >= {"contact", "map"}
        assert json.load(open(TOOL_PATH))["tip_measured"]["by"] == "calib-air"
        # saved again: the same tip, so nothing more is dropped and the record is renewed
        r2 = CLI.main(["--sim", "--from", folder, "--save"])
        assert r2["saved"] and r2["tip_changed_mm"] == 0.0 and not r2["dropped"], r2
    # the photos of another tool (or of another taping of it) are not stored into this one
    other = thin_tool([0.020, 0.0, 0.050], attached_at="2026-11-13T19:00:00")
    with tool_file(other):
        before = open(TOOL_PATH, "rb").read()
        for force in ([], ["--force"]):
            r = CLI.main(["--sim", "--from", folder, "--save", *force])
            assert r["ok"] is False and r["stage"] == "save" and r["saved"] is False and "attached at" in r["reason"], r
            assert "--force does not change that" in r["reason"] and "fd calib-air --tip-only --save" in r["next"]
            assert open(TOOL_PATH, "rb").read() == before, "photos of another taping were stored"
    # no tool at all
    with tool_file(None):
        r = CLI.main(["--sim", "--from", folder, "--save"])
        assert r["ok"] is False and r["stage"] == "save" and not os.path.exists(TOOL_PATH), r
    # a folder without records, a folder that is none
    assert CLI.main(["--sim", "--from", _TMP])["stage"] == "load" and CLI.main(["--sim", "--from", os.path.join(_TMP, "nowhere")])["stage"] == "load"


def test_a_bad_fit_is_not_stored_without_force():
    folder = capture_folder("few")              # 20 photos, the brush in 8
    registered = thin_tool([0.020, 0.0, 0.050])
    with tool_file(registered):
        before = open(TOOL_PATH, "rb").read()
        r = CLI.main(["--sim", "--from", folder, "--save"])
        assert r["ok"] is False and r["stage"] == "save" and r["saved"] is False and "not stored" in r["reason"], r
        assert r["used"] == 8 and r["photos"] == 20 and not r["gates"]["passed"] and len(r["gates"]["problems"]) == 2, r["gates"]
        assert "only 8 of the 20 photos" in r["warning"] and len(r["unused"]) == 12 and all(u["why"].startswith("no candidate") for u in r["unused"])
        assert open(TOOL_PATH, "rb").read() == before
        # the numbers are still reported: the eight photos agree on the right tip
        assert np.linalg.norm(np.asarray(r["tip_offset"]) - THIN_TIP) < 0.002
        r = CLI.main(["--sim", "--from", folder, "--save", "--force"])
        assert r["ok"] is True and r["saved"] is True and len(r["forced"]) == 2, r
        assert CLI._tip_record(TL.active())["forced"] == r["forced"]
    # a fit that fails the gates without being "few photos" or "the camera moved": the hint is the whole command
    res = {"tip_offset": THIN_TIP, "tip_mm": [16.0, -1.1, 55.8], "photos": 14, "used": 11, "rms_px": 7.0, "unused": []}
    CLI._verdict_text(res, "tip", {"problems": ["11 photos carry the fit"], "moved": None, "few": False}, "/some/folder")
    assert "`fd calib-air --from /some/folder --save --force` stores it anyway" in res["next"] and "--force only stores" in res["next"]
    full = dict(res, camera={"center": [0.58, 0.08, 0.28]})
    CLI._verdict_text(full, "full", {"problems": ["rms"], "moved": None, "few": False}, "/some/folder")
    assert "`fd calib-air --from /some/folder --sets air,low --save --force`" in full["next"], full["next"]
    CLI._verdict_text(full, "full", {"problems": [], "moved": None, "few": False}, "/some/folder")
    assert full["next"].startswith("`fd calib-air --from /some/folder --sets air,low --save` stores it"), full["next"]
    # the gates themselves, on the numbers of 2026-11-13
    fit = {"used": 34, "photos": 36, "rms_px": 6.6, "sigma": [0.0003, 0.0003, 0.0002], "tip_offset": THIN_TIP, "check": None}
    assert CLI.judge("tip", fit, CAM) == {"problems": [], "moved": None, "few": False}
    assert CLI.judge("tip", dict(fit, used=8, photos=20), CAM)["few"] is True
    assert len(CLI.judge("tip", dict(fit, used=11, photos=12), CAM)["problems"]) == 1
    assert "formal error" in CLI.judge("tip", dict(fit, sigma=[0.0015, 0.0003, 0.0002]), CAM)["problems"][0]
    for rms, moved in ((9.3, False), (11.5, False), (11.7, True), (15.0, True)):      # 1.3 x 8.89 = 11.56
        assert bool(CLI.judge("tip", dict(fit, rms_px=rms), CAM)["moved"]) is moved, rms
    chk = {"moved_mm": 20.0, "turned_deg": 0.3, "d": 9.0, "used": 30}
    assert "20 mm" in CLI.judge("tip", dict(fit, check=chk), CAM)["moved"]
    assert CLI.judge("tip", dict(fit, check=dict(chk, used=9)), CAM)["moved"] is None                # nine numbers from nine photos
    assert CLI.judge("tip", dict(fit, check=dict(chk, d=3.5)), CAM)["moved"] is None                 # the evening: d 1.8 .. 3.5
    assert CLI.judge("tip", dict(fit, check=dict(chk, moved_mm=2.0)), CAM)["moved"] is None
    assert "30 cm" in " ".join(CLI.judge("tip", dict(fit, tip_offset=[0.0, 0.0, 0.4]), CAM)["problems"])
    full = {"used": 55, "photos": 61, "rms_px": 8.9, "sigma": [0.005] * 6 + [0.001, 0.0015, 0.0016], "center_sigma": [0.002, 0.003, 0.002],
            "center": [0.582, 0.080, 0.279], "tip_offset": FLAT_TIP}
    assert CLI.judge("full", full, CAM)["problems"] == []
    assert CLI.judge("full", dict(full, used=29, photos=31, center_sigma=[0.0037, 0.0054, 0.0034], sigma=[0.005] * 6 + [0.0016, 0.0025, 0.0027]), CAM)["problems"] == []
    for bad in (dict(used=22, photos=22), dict(used=30, photos=61), dict(rms_px=14.0), dict(center_sigma=[0.004, 0.007, 0.004]),
                dict(sigma=[0.005] * 6 + [0.002, 0.004, 0.003]), dict(center=[0.5, 0.0, -0.1])):
        assert len(CLI.judge("full", dict(full, **bad), CAM)["problems"]) >= 1, bad


def test_full_solve_from_a_folder_replaces_the_scratch_camera_model_with_a_backup():
    """The model on file is 4 cm and 2 deg from where the photos were taken; the full solve finds the camera the
    photos were drawn with (6 mm) and the tip (3 mm), stores both and keeps the old model."""
    folder = capture_folder("good")
    stale = moved_camera(CAM, (0.03, -0.025, 0.01), (0.0, 2.0, 0.0)).to_dict()
    registered = thin_tool([0.020, 0.0, 0.050])
    with model_file(stale), tool_file(registered):
        stale_bytes = open(MODEL_PATH, "rb").read()
        # tip-only against the stale model: a tip comes out, with the warning, and --save refuses it
        r = CLI.main(["--sim", "--from", folder, "--tip-only", "--save"])
        assert r["saved"] is False and r["ok"] is False and r["stage"] == "save" and open(MODEL_PATH, "rb").read() == stale_bytes, r
        assert "camera seems to have moved" in r["warning"] and "camera seems to have moved" in r["reason"], r
        assert "`fd calib-air --sets air,low`" in r["next"] and r["camera_model"]["check"]["moved_mm"] > 25.0
        assert np.allclose(TL.active().tip_offset, [0.020, 0.0, 0.050])
        r = CLI.main(["--sim", "--from", folder, "--sets", "air,low"])
        assert r["ok"] and r["mode"] == "full" and r["saved"] is False and open(MODEL_PATH, "rb").read() == stale_bytes, r
        assert np.linalg.norm(np.asarray(r["camera"]["center"]) - CAM.center) < 0.006, r["camera"]
        assert np.linalg.norm(np.asarray(r["tip_offset"]) - THIN_TIP) < 0.003, r["tip_mm"]
        assert 30.0 < r["camera"]["moved_mm"] < 50.0 and 1.5 < r["camera"]["turned_deg"] < 2.5 and r["gates"]["passed"], r
        r = CLI.main(["--sim", "--from", folder, "--sets", "air,low", "--save"])
        assert r["ok"] and r["saved"] is True, r
        new = json.load(open(MODEL_PATH))
        assert np.linalg.norm(np.asarray(new["center_base_m"]) - CAM.center) < 0.006 and new["f"] == 1430.0 and new["n_points"] == r["used"]
        assert "fd calib-air" in new["source"] and "brush_thin" in new["source"] and abs(new["rms_px"] - r["rms_px"]) < 0.01
        backup = r["files"]["camera_model_backup"]
        assert os.path.dirname(backup) == os.path.join(_TMP, "backups") and os.path.basename(backup).startswith("camera_model.before_aircal_2")
        assert open(backup, "rb").read() == stale_bytes and r["files"]["camera_model"] == MODEL_PATH and not os.path.exists(MODEL_PATH + ".tmp")
        t = TL.active()
        assert np.allclose(t.tip_offset, r["tip_offset"], atol=1e-5) and CLI._tip_record(t)["solve"] == "full"
        assert C.CameraModel.load().project([[0.2, 0.0, 0.0]]).shape == (1, 2)
    assert json.load(open(MODEL_PATH)) == CAM_DICT
    # --sim never replaces the rig's own model. (Shown with the scratch file standing in for config/camera_model.json
    # and a scratch backup folder: should the refusal ever fail, the rig's files are still not what gets written.)
    assert CLI._backup_dir(C.CAM_MODEL_PATH) == os.path.join(tazuna_paths.ROOT, "backups")
    saved = (C.CAM_MODEL_PATH, CLI._backup_dir)
    C.CAM_MODEL_PATH, CLI._backup_dir = MODEL_PATH, (lambda path: os.path.join(_TMP, "backups"))
    try:
        with tool_file(registered):
            r = CLI.main(["--sim", "--from", folder, "--sets", "air,low", "--save"])
        assert r["ok"] is False and r["stage"] == "save" and "--sim does not protect" in r["reason"], r
        assert json.load(open(MODEL_PATH)) == CAM_DICT and not os.path.exists(os.path.join(_TMP, "backups"))
    finally:
        C.CAM_MODEL_PATH, CLI._backup_dir = saved


def test_store_tip_without_the_newer_tooling():
    """tooling.set_measured_tip arrived on the evening of 2026-11-13; with a tooling that lacks it the command does
    the same steps itself and the record survives a save and a load."""
    setter = getattr(TL, "set_measured_tip", None)
    if setter is not None:
        del TL.set_measured_tip
    try:
        rec = {"at": "2026-11-13T20:00:00", "by": "calib-air", "rms_px": 6.6, "used": 34, "of": 36}
        t = thin_tool(contact={"x0": 0.2, "y0": 0.0, "a": 0.0, "rms_mm": 0.2, "points": []}, calibrated={"contact": {"at": "x"}})
        assert CLI._store_tip(t, THIN_TIP, rec) == [] and t.contact is not None        # the same tip: nothing dropped
        assert CLI._tip_record(t) == rec
        dropped = CLI._store_tip(t, [0.017, -0.001, 0.056], rec)
        assert set(dropped) == {"contact", "calibrated"} and t.contact is None and t.calibrated == {} and t.tip_offset == [0.017, -0.001, 0.056]
        with tool_file(t):
            assert CLI._tip_record(TL.active()) == rec
        raises(SafetyError, CLI._store_tip, t, [0.0, 0.0, 0.5], rec)
    finally:
        if setter is not None:
            TL.set_measured_tip = setter
    # a tip that is typed in afterwards is no longer "measured"
    t = thin_tool()
    CLI._store_tip(t, THIN_TIP, {"at": "x", "by": "calib-air"})
    assert CLI._tip_record(t) is not None
    t.tip_offset = [0.0, 0.0, 0.08]
    assert CLI._tip_record(t) is None


# ----------------------------------------------------------------------------- what the review of 2026-11-13 found
LONGER_TIP = [-0.001, 0.0, 0.1768]       # 17 cm ALONG the fingers: 9 cm longer than the 8 cm of SHORT_GUESS


def test_force_only_stores_and_no_option_flies_a_tool_the_look_refused():
    """Findings 2 and 54. Registered "8 cm along the fingers", really 17 cm along them. Before the split,
    `fd calib-air --save --force` (the words the hint for a failed gate printed) went on after the refused look
    and flew 27 poses with the real tip at or under the table at 3 of them.

    Every combination of the options that does not carry the operator's own --no-look: the command stops at the
    look, the REAL tip is never lower than 2 cm (the way to the look pose and the look pose are made for a tool
    of unknown length), no calibration pose is flown, nothing is stored. The simulated arm starts folded as the
    real tool allows (at_rest_of): in the rest pose of the REGISTERED tool this one would stand inside the table.

    --no-look is the operator's statement that the registered tip was checked: the first set is then flown on
    that word, and nothing in the program can know better (here the real tip comes down to the table: that is
    printed as a note, it is what the look is for). What still holds: --force changes nothing of what is flown,
    and no option flies a low set on a tip that is only registered."""
    registered = thin_tool(SHORT_GUESS)
    start = at_rest_of(LONGER_TIP)
    modes = ([], ["--tip-only"], ["--sets", "air,low"], ["--sets", "low"])
    extras = [list(c) for n in range(4) for c in itertools.combinations((("--save",), ("--save", "--force"), ("--end", "stay")), n)
              if not (("--save",) in c and ("--save", "--force") in c)]
    extras = [[w for part in c for w in part] for c in extras]
    assert sorted(map(tuple, extras)) == sorted([(), ("--save",), ("--save", "--force"), ("--end", "stay"), ("--save", "--end", "stay"),
                                                 ("--save", "--force", "--end", "stay")])
    with tool_file(registered):
        before = open(TOOL_PATH, "rb").read()
        lowest = None
        for mode in modes:
            for extra in extras:
                argv = mode + extra
                res, arm, bus, photo, root = run_command(argv, registered, LONGER_TIP, prepare=start)
                assert res["ok"] is False and res["stage"] == "look" and res["saved"] is False and "forced" not in res["look"], (argv, res)
                assert len(photo.shots) == 2 and res["sets"] == [], (argv, len(photo.shots), res["sets"])
                assert sorted(os.path.basename(p) for p in glob.glob(os.path.join(res["dir"], "*.jpg"))) == ["look.jpg", "look_before.jpg"]
                low = min(real_heights(arm, bus, registered, LONGER_TIP))
                assert low >= 0.020, (argv, low)
                lowest = low if lowest is None else min(lowest, low)
                assert open(TOOL_PATH, "rb").read() == before, argv
        # (the bound the look pose states for a tool along the fingers: fingertip 12 cm up, 30 deg down, 17 cm of tool)
        assert 0.025 <= lowest <= 0.05, lowest
        # --force without --save says nothing: refused before the arm is asked for
        made = []
        saved = CLI._make_arm
        CLI._make_arm = lambda sim: made.append(sim)
        try:
            for argv in (["--force"], ["--no-look", "--force"], ["--sets", "low", "--force"]):
                r = CLI.main(argv)
                assert r["ok"] is False and r["stage"] == "options" and "--no-look" in r["reason"] and "--save --force" in r["next"], (argv, r)
        finally:
            CLI._make_arm = saved
        assert not made
        # the operator's word: --no-look. What is flown does not depend on --force / --save, and it is the high set alone
        flown = {}
        for extra in ([], ["--save", "--force"]):
            res, arm, bus, photo, root = run_command(["--no-look"] + extra, registered, LONGER_TIP, prepare=start, hand_every=0)
            assert [s["name"] for s in res["sets"]] == ["high"] and not glob.glob(os.path.join(res["dir"], "low_*")), (extra, res["sets"])
            assert any("was not flown" in n and "no option flies it so" in n for n in res["notes"]), res["notes"]
            flown[bool(extra)] = (np.array(photo.shots), min(real_heights(arm, bus, registered, LONGER_TIP)))
        assert flown[False][0].shape == flown[True][0].shape and np.allclose(flown[False][0], flown[True][0], atol=1e-6)
        print(f"     note: with --no-look the high set was flown on the operator's word; the REAL tip of a tool 9 cm longer "
              f"than registered was at {flown[False][1] * 1000:+.0f} mm at its lowest (0 = the table): the look is what prevents that")
    # a low set, 3 .. 6 cm over the table, with a tip that is only registered: no option flies it
    with tool_file(registered):
        for extra in ([], ["--save", "--force"], ["--end", "stay"]):
            res, arm, bus, photo, root = run_command(["--no-look", "--sets", "low"] + extra, registered, LONGER_TIP, prepare=start)
            assert res["ok"] is False and res["stage"] == "capture" and "has not been measured" in res["reason"], (extra, res)
            assert not photo.shots and res["sets"] == [] and bus.io["writes"] == 0 and not any(bus.torque.values()), extra
            assert "`fd calib-air --sets low --dry`" in res["next"]
        # ... and an arm that this command did not move is not folded away at the end either: it holds at the look
        # pose (torque on, where a refused look left it), a fold would be planned with the registered tip, and this
        # tool would come down to the table (see test_a_look_never_passes_by_colour_alone)
        res, arm, bus, photo, root = run_command(["--no-look", "--sets", "low"], registered, LONGER_TIP,
                                                 prepare=held_at_look_pose(LONGER_TIP))
        q_look = AC.look_pose(arm, registered, CAM)["q"]
        assert res["stage"] == "capture" and "did not move it" in res["arm"]["at"] and res["arm"]["torque"] == "on", res
        assert all(np.max(np.abs(q_of(p) - q_look)) < math.radians(1.0) for p in bus.trace) and not photo.shots
        assert np.allclose(q_of(bus.pos), q_look, atol=math.radians(1.0)) and min(real_heights(arm, bus, registered, LONGER_TIP) or [1.0]) >= 0.020
        # what is flown is not a matter of --force: the run works with a set of options that has no such word
        for no_look, names, stage in ((True, ["low"], "capture"), (False, ["high", "low"], "look")):
            arm, bus = sim_arm(TL.active())
            start(arm, bus)
            a = argparse.Namespace(no_look=no_look, end="rest", sim=False)
            res = CLI._run(a, "full" if no_look else "tip", names, arm, CAM, camera_for(arm, LONGER_TIP),
                           tempfile.mkdtemp(prefix="run_", dir=_TMP), lambda *m: None)
            assert res["ok"] is False and res["stage"] == stage and "error" not in res and res["sets"] == [], res


def test_a_look_never_passes_by_colour_alone():
    """Finding 43. A look that was refused leaves the arm at the look pose; the next run started from there took
    no photo "before" and judged by colour alone: with the evening's desk as the picture, registrations of 12 and
    8 cm along the fingers (the real tip was 17 cm out) passed with paint 21 and 35 px from where they would be
    seen, and 19 and 20 poses were flown."""
    registered = thin_tool(SHORT_GUESS)
    arm0, _bus = sim_arm(registered)
    look = AC.look_pose(arm0, registered, CAM)
    q_look, px = look["q"], np.array(look["tip_px"])
    # paint on the sheet exactly where the registered tip would be seen: bristle-coloured, as long and wide as a tip
    desk = BASE.copy()
    _seg(desk, px + [-22.0, 4.0], px + [22.0, -4.0], 11, 4, ORANGE)
    # 1) the check itself: without a photo of the scene it is not checked, whatever lies near. (The brush is the
    #    evening's second mount, 17 cm out and beside the finger line: it does not hide the paint.)
    photo = draw_tool(desk.copy(), CAM, q_look, LONG_TIP)
    alone = AC.look_check(photo, CAM, q_look, SHORT_GUESS, bg=None)
    assert alone["ok"] is False and alone["checked"] is False and alone["nearest_px"] < 30.0 < AC.LOOK_PX, alone
    with_bg = AC.look_check(photo, CAM, q_look, SHORT_GUESS, bg=desk)
    assert with_bg["ok"] is False and with_bg["checked"] is True and with_bg["nearest_px"] > 200.0, with_bg
    # (round 3: and such a background is no background for a look at all: look_background)
    assert with_bg["background"]["clean"] is False and alone["background"] is None, with_bg["background"]
    # 2) the command, started at the look pose with the torque on (over the desk without that paint): the arm is
    #    turned aside about its base for the photo "before", comes back, and the look refuses; nothing else is flown
    with tool_file(registered):
        res, arm, bus, shots, root = run_command([], registered, LONG_TIP, prepare=held_at_look_pose())
        lk = res["look"]
        assert res["ok"] is False and res["stage"] == "look" and lk["ok"] is False and lk["checked"] is True, res
        assert lk["nearest_px"] > 200.0 and os.path.isfile(lk["photo_before"]) and os.path.isfile(lk["photo"]), lk
        assert lk["before"]["turn_deg"] in AC.LOOK_ASIDE_DEG and "turned aside" in lk["before"]["taken"], lk["before"]
        assert len(shots.shots) == 2 and not glob.glob(os.path.join(res["dir"], "high_*")) and res["sets"] == []
        aside, back = shots.shots
        assert abs(abs(math.degrees(aside[0] - q_look[0])) - abs(lk["before"]["turn_deg"])) < 1.0, np.degrees(aside)
        assert np.max(np.abs(aside[1:] - q_look[1:])) < math.radians(1.5) and np.allclose(back, q_look, atol=math.radians(1.0))
        assert np.allclose(q_of(bus.pos), q_look, atol=math.radians(1.0)) and all(bus.torque[j] == 1 for j in ARM)
        assert "look pose" in res["arm"]["at"] and res["arm"]["torque"] == "on"
        # in the photo before, the camera model has the tool far from where the look photo has it
        seen = CAM.project([K.fk(aside)[:3, 3] + K.fk(aside)[:3, :3] @ np.asarray(SHORT_GUESS)])[0]
        assert np.linalg.norm(seen - px) >= AC.LOOK_ASIDE_PX
        # the hint after a refused look says what must not be done from here, and names the command that folds
        # (no painting set-up in this test: the tool's own)
        assert "fold only when the photo shows the tool no longer than registered" in res["next"]
        assert folds_where_it_points(res["next"], res["arm"]) and "`fd tool rest`" not in res["next"]
        # (round 3b, U2) the pose is not called harmless any more; the side the arm turned over is named
        assert "harmless" not in res["next"] and "the base turns only with the tool lifted 10 cm straight up" in res["next"], res["next"]
        assert "clear of anything higher than about 2 cm" in res["next"] and res["look"]["before"]["over"] in res["next"], res["next"]
        # --dry from there plans the same turn, and writes no goal
        writes = bus.io["writes"]
        saved = CLI._make_arm
        CLI._make_arm = lambda sim: arm
        try:
            r = CLI.main(["--sim", "--dry"])
        finally:
            CLI._make_arm = saved
        assert r["ok"] and "turned" in r["look"]["photo_before"] and "aside" in r["look"]["photo_before"], r["look"]
        assert bus.io["writes"] == writes
        # 2b) round 3, T2 (b), a clean background: over the desk WITH paint where the registered tip would be seen, no
        #     look is taken at all. The arm turns aside, one photo, and it holds there; nothing is flown
        res, arm, bus, shots, root = run_command([], registered, LONG_TIP, prepare=held_at_look_pose(), base=desk)
        lk = res["look"]
        assert res["ok"] is False and res["stage"] == "look" and lk["ok"] is False and lk["dirty"] is True, res
        assert lk["background"]["clean"] is False and lk["background"]["paint_px"] >= 100 and lk["background"]["nearest_px"] < 10.0, lk
        assert len(shots.shots) == 1 and sorted(os.listdir(res["dir"])) == ["look_before.jpg"] and res["sets"] == [], os.listdir(res["dir"])
        assert abs(math.degrees(shots.shots[0][0] - q_look[0])) > 85.0 and np.max(np.abs(shots.shots[0][1:] - q_look[1:])) < math.radians(1.5)
        assert np.allclose(q_of(bus.pos), shots.shots[0], atol=math.radians(1.0)) and all(bus.torque[j] == 1 for j in ARM)
        assert "turned aside" in res["arm"]["at"] and res["arm"]["torque"] == "on" and min(real_heights(arm, bus, registered, LONG_TIP)) >= 0.020
        assert "bristle-coloured paint" in res["reason"] and "No look photo was taken" in res["reason"], res["reason"]
        assert "cover the sheet with a blank one, or take it away" in res["next"] and "`fd calib-air` again" in res["next"], res["next"]
        assert "fold only when the photo shows the tool no longer than registered" in res["next"] and "白い紙で覆う" in res["say"]
        # ... a handful of such pixels is noise, not paint; paint further away than LOOK_CLEAN_PX is not in the way
        speck = BASE.copy()
        cv2.circle(speck, (int(px[0]) + 40, int(px[1]) - 30), 1, ORANGE, -1)
        cv2.circle(speck, (int(px[0]) + int(AC.LOOK_CLEAN_PX) + 40, int(px[1])), 14, ORANGE, -1)
        ground = AC.look_background(speck, CAM, q_look, SHORT_GUESS)
        assert ground["clean"] is True and 0 < ground["paint_px"] < AC.LOOK_CLEAN_MIN_PX and ground["radius_px"] == AC.LOOK_CLEAN_PX, ground
        assert AC.look_background(BASE, CAM, q_look, SHORT_GUESS) == dict(ground, paint_px=0, nearest_px=None)
        # the same with a tool that is 9 cm longer than registered ALONG the fingers, the one a fold is dangerous for.
        # Only the base turned: its REAL tip kept its height all the way (3.5 cm) ...
        res, arm, bus, shots, root = run_command([], registered, LONGER_TIP, prepare=held_at_look_pose(LONGER_TIP))
        assert res["stage"] == "look" and res["look"]["before"]["turn_deg"] in AC.LOOK_ASIDE_DEG and len(shots.shots) == 2, res
        h_look = TL.tip_height(arm, q_look, thin_tool(LONGER_TIP))
        heights = real_heights(arm, bus, registered, LONGER_TIP)
        assert 0.030 < h_look < 0.040 and min(heights) > h_look - 0.004 and min(heights) >= 0.020, (h_look, min(heights))
        # ... which a fold does not do: it is planned with the registered tip, and brings this tool down to a
        # centimetre above the table (9.6 mm; into the fold `fd paint rest` chooses: 4.6 mm under it)
        bus.trace.clear()
        TL.rest(arm, torque_off=False)
        folded = min(real_heights(arm, bus, registered, LONGER_TIP))
        assert folded < 0.012, folded
    # 3) a registration that is right passes from there too, on the two photos, over a blank sheet ...
    good = thin_tool(np.add(THIN_TIP, [0.003, 0.002, 0.003]))
    with tool_file(good):
        arm, bus = sim_arm(TL.active())
        held_at_look_pose()(arm, bus)
        out = tempfile.mkdtemp(prefix="look_", dir=_TMP)
        lk = CLI._look(arm, arm.tool, CAM, camera_for(arm, THIN_TIP, hand_every=0), out, "tip", lambda *m: None)
        assert lk["ok"] is True and lk["checked"] is True and lk["nearest_px"] < 40.0 and lk["before"]["turn_deg"] in AC.LOOK_ASIDE_DEG, lk
        assert lk["background"]["clean"] is True and "dirty" not in lk
        assert np.allclose(q_of(bus.pos), AC.look_pose(arm, arm.tool, CAM)["q"], atol=math.radians(1.0))
        # ... and not over paint: the same right registration, a stroke 50 px beside where its tip will be seen. The
        # look cannot tell a brush from paint behind it, so it is not taken (and the check itself refuses such a pair)
        q_good, p_good = AC.look_pose(arm, arm.tool, CAM)["q"], np.array(AC.look_pose(arm, arm.tool, CAM)["tip_px"])
        painted = BASE.copy()
        _seg(painted, p_good + [50.0, -20.0], p_good + [50.0, 24.0], 11, 4, RED)
        arm, bus = sim_arm(TL.active())
        held_at_look_pose()(arm, bus)
        shots = camera_for(arm, THIN_TIP, base=painted, hand_every=0)
        lk = CLI._look(arm, arm.tool, CAM, shots, tempfile.mkdtemp(prefix="look_", dir=_TMP), "tip", lambda *m: None)
        assert lk["ok"] is False and lk["dirty"] is True and len(shots.shots) == 1 and lk["photo"] is None, lk
        pair = AC.look_check(draw_tool(painted.copy(), CAM, q_good, THIN_TIP), CAM, q_good, good.tip_offset, bg=painted)
        assert pair["ok"] is False and pair["nearest_px"] < 40.0 and pair["background"]["clean"] is False, pair
    # 4) no turn is possible (the base cannot turn far enough either way): refused before anything moves
    stiff = thin_tool(SHORT_GUESS, joint_limits_deg={"wrist_flex": [20.0, 97.0], "wrist_roll": [-5.0, 5.0], "shoulder_pan": [-30.0, 30.0]})
    with tool_file(stiff):
        res, arm, bus, shots, root = run_command([], stiff, LONGER_TIP, prepare=held_at_look_pose(LONGER_TIP), base=desk)
        assert res["ok"] is False and res["stage"] == "look" and "cannot be turned aside" in res["reason"], res
        assert not shots.shots and np.allclose(q_of(bus.pos), AC.look_pose(arm, stiff, CAM)["q"], atol=math.radians(1.0))
        assert all(np.max(np.abs(q_of(p) - q_of(bus.pos))) < math.radians(0.5) for p in bus.trace), "the arm moved"
        # (round 3b, U5) no fold is named: the turns are judged at the look pose, whatever the start, so "fold, then the
        # same command" cannot help. What is named: the operator's own look and --no-look, or a ruler
        assert "--no-look" in res["next"] and "`fd snap --exposure paper`" in res["next"], res["next"]
        assert "folding the arm first does not help" in res["next"] and "ruler" in res["next"], res["next"]
        assert "`fd tool rest" not in res["next"] and "`fd paint rest" not in res["next"], res["next"]


def test_real_desk_what_passed_by_colour_alone_from_the_look_pose_is_refused():
    """Finding 43 with the evening's desk as the camera's picture (runs/20261113_paint/190853_run.jpg, a local
    photo): started at the look pose, registrations of 12 and 8 cm along the fingers passed with paint 21 and
    35 px from where they would be seen. (Until 2026-11-14 this was the tail of the test above, which printed a
    note and returned where the photo is missing: now it is skipped there, and counted.)"""
    desk_photo = os.path.join(tazuna_paths.ROOT, "runs", "20261113_paint", "190853_run.jpg")
    if not os.path.isfile(desk_photo):
        raise Skip("runs/20261113_paint/190853_run.jpg is not on this computer")
    img = cv2.imread(desk_photo)
    for tip in ([-0.001, 0.0, 0.1268], SHORT_GUESS):
        tool = thin_tool(tip)
        with tool_file(tool):
            arm, bus = sim_arm(TL.active())
            held_at_look_pose()(arm, bus)
            q = arm.read_q()
            assert AC.look_check(img, CAM, q, tip, bg=None)["nearest_px"] < AC.LOOK_PX       # colour alone: "near"
            taken = []

            def desk_only():
                taken.append(arm.read_q())
                return img.copy()

            saved = (CLI._make_arm, CLI._photo_fn, CLI._run_dir)
            root = tempfile.mkdtemp(prefix="run_", dir=_TMP)
            CLI._make_arm, CLI._photo_fn, CLI._run_dir = (lambda sim: arm), (lambda: desk_only), (lambda: root)
            try:
                # as round 2 left it (the photos alone, the clean-background rule switched off): the arm is turned
                # aside, two photos, and nothing differs between them
                with contextlib.redirect_stderr(io.StringIO()), any_background():
                    res = CLI.main([])
                assert res["ok"] is False and res["stage"] == "look" and len(taken) == 2 and res["sets"] == [], (tip, res)
                assert res["look"]["nearest_px"] is None, res["look"]       # nothing differs from the photo before
                # round 3, with the rule: the place of the look lies on the picture's red. No look photo is taken at
                # all: one photo (the arm turned aside, or where the first run left it), and the arm holds
                del taken[:]
                with contextlib.redirect_stderr(io.StringIO()):
                    res = CLI.main([])
            finally:
                CLI._make_arm, CLI._photo_fn, CLI._run_dir = saved
            lk = res["look"]
            assert res["ok"] is False and res["stage"] == "look" and len(taken) == 1 and res["sets"] == [], (tip, res)
            assert lk["dirty"] is True and lk["background"]["clean"] is False and lk["background"]["paint_px"] > 500, lk["background"]
            assert lk["photo"] is None and "cover the sheet with a blank one" in res["next"] and all(bus.torque[j] == 1 for j in ARM)


def test_photos_taken_with_the_jaw_elsewhere_are_never_stored():
    """Findings 7, 24 and 44. `fd calib-air --from DIR --save` compared the tool's name and attach time only. The
    brush of 2026-11-13 had its tip at (-5.1, -2.5, 60.4) mm with the jaw at raw 2182 and at (16.0, -1.1, 55.8) mm
    closed at raw 2041, under the same name and attach time: the first, stored into the second, is 2.2 cm off and
    marked "measured"."""
    held = thin_tool(on_jaw="moving_outer", gripper_raw=2041, gripper_goal_raw=2016)

    def then(**kw):
        return thin_tool(**kw).to_dict()

    why = CLI.jaw_mismatch
    # --- records without a reading of the jaw (the captures of 2026-11-13): what the tool file said then, against now
    assert why(held.to_dict(), [], held) == []
    assert why(then(on_jaw="moving_outer", gripper_raw=2043, gripper_goal_raw=2018), [], held) == []        # tooling.JAW_SAME_TICKS
    got = why(then(on_jaw="moving_outer", gripper_raw=2182), [], held)
    assert len(got) == 2 and "raw 2182" in got[0] and "raw 2041" in got[0] and "no jaw hold" in got[1] and "2016" in got[1], got
    assert len(why(then(gripper_raw=2045, on_jaw="moving_outer", gripper_goal_raw=2020), [], held)) == 2      # 4 ticks: another state
    got = why(then(gripper_raw=2041), [], held)                 # the same place, but no hold on record then
    assert len(got) == 1 and "no jaw hold on record" in got[0], got
    got = why(held.to_dict(), [], thin_tool(gripper_raw=2041))  # ... and the other way round
    assert len(got) == 1 and "held by a squeeze" in got[0] and "no hold" in got[0], got
    assert why(then(), [], thin_tool()) == []                   # a tool that never recorded where its jaw stood
    assert len(why(then(), [], thin_tool(gripper_raw=2041))) == 1 and len(why(then(gripper_raw=2041), [], thin_tool())) == 1
    # on which jaw: "not said" contradicts nothing (`fd tool jaw mount` writes a label), two different answers do
    assert why(then(gripper_raw=2041), [], thin_tool(gripper_raw=2041, on_jaw="moving_outer")) == []
    got = why(then(gripper_raw=2041, on_jaw="fixed"), [], thin_tool(gripper_raw=2041, on_jaw="between"))
    assert len(got) == 1 and "on_jaw = fixed" in got[0] and "between" in got[0], got
    # --- records of this command: where the servo said the jaw stood decides, whatever the file said at that moment
    squeezed = [{"gripper_raw": 2040 + k % 3, "gripper_goal_raw": 2016} for k in range(12)]
    assert why(then(gripper_raw=2182), squeezed, held) == []
    assert why(None, squeezed, held) == []
    got = why(held.to_dict(), [{"gripper_raw": 2182, "gripper_goal_raw": 2182}] * 12, held)
    assert len(got) == 2 and "raw 2182" in got[0] and "read from the servo" in got[0] and "NOT held" in got[1], got
    got = why(held.to_dict(), [{"gripper_raw": 2041, "gripper_goal_raw": 2041}] * 12, held)      # in place, but loose
    assert len(got) == 1 and "NOT held with a squeeze" in got[0], got
    got = why(held.to_dict(), [{"gripper_raw": 2041, "gripper_goal_raw": None}] * 12, held)      # the gripper's torque was off
    assert len(got) == 1 and "NOT held with a squeeze" in got[0], got
    got = why(then(gripper_raw=2041), squeezed, thin_tool(gripper_raw=2041))                     # squeezed, and the file knows no hold
    assert len(got) == 1 and "held shut with a squeeze" in got[0] and "no hold" in got[0], got
    # (a tool gripped BETWEEN the jaws is squeezed by its nature and can have no hold on record: nothing to compare)
    assert why(None, squeezed, thin_tool(gripper_raw=2041, on_jaw="between")) == []
    # a tool taped to the FIXED jaw does not go with the moving one: where that stood says nothing about its tip
    on_fixed = thin_tool(gripper_raw=2041, on_jaw="fixed")
    assert why(None, [{"gripper_raw": 2300, "gripper_goal_raw": 2300}] * 5, on_fixed) == []
    assert why(then(gripper_raw=2300, on_jaw="fixed"), [], on_fixed) == [] and why(then(gripper_raw=2300), [], on_fixed) == []
    assert len(why(then(gripper_raw=2041, on_jaw="between"), [], on_fixed)) == 1
    assert why(None, [{"gripper_raw": 2041, "gripper_goal_raw": "?"}, {"gripper_raw": "x"}], thin_tool(gripper_raw=2041)) == []
    edge = CLI.PHOTO_JAW_TOL_TICKS
    assert why(None, [{"gripper_raw": 2041 + edge, "gripper_goal_raw": 2016}] * 3, held) == []
    assert len(why(None, [{"gripper_raw": 2041 + edge + 1, "gripper_goal_raw": 2016}] * 3, held)) == 1
    assert len(why(None, squeezed + [{"gripper_raw": 2100, "gripper_goal_raw": 2016}], held)) == 1          # one photo elsewhere is enough

    # --- the command. (The same 35 photos are solved again and again: the fit is kept, only what happens to it differs.)
    folder = capture_folder("good")
    rec_path = os.path.join(folder, "records_0_high.json")
    original = open(rec_path).read()
    data = json.loads(original)
    fits = {}
    solve = CLI._solve

    def solve_once(mode, images, records, cam, guess):
        key = (mode, len(records))
        if key not in fits:
            fits[key] = solve(mode, images, records, cam, guess)
        return copy.deepcopy(fits[key])

    def put(tool_block, records=None):
        d = {k: v for k, v in data.items() if k != "tool"}
        if tool_block is not None:
            d["tool"] = tool_block
        if records is not None:
            d["records"] = records
        with open(rec_path, "w") as fh:
            json.dump(d, fh)

    CLI._solve = solve_once
    try:
        # the photos were taken with the jaw at raw 2182; since then `fd tool jaw close`: raw 2041, held (same name, same attach time)
        put(then(tip=[0.020, 0.0, 0.050], on_jaw="moving_outer", gripper_raw=2182))
        now = thin_tool([0.0234, -0.0025, 0.0523], on_jaw="moving_outer", gripper_raw=2041, gripper_goal_raw=2016)
        with tool_file(now):
            before = open(TOOL_PATH, "rb").read()
            for force in ([], ["--force"]):
                r = CLI.main(["--sim", "--from", folder, "--save", *force])
                assert r["ok"] is False and r["stage"] == "save" and r["saved"] is False, r
                assert "raw 2182" in r["reason"] and "raw 2041" in r["reason"] and "--force does not change that" in r["reason"], r["reason"]
                assert len(r["not_this_tool"]) == 2 and "fd tool jaw status" in r["next"] and "fd calib-air --tip-only --save" in r["next"]
                assert r["gates"]["passed"] is True and "保存していません" in r["say"]       # the fit is fine: it is of something else
                assert open(TOOL_PATH, "rb").read() == before, "a tip measured with the jaw elsewhere was stored"
            assert TL.active().tip_measured is None
            # without --save the same photos are solved and reported; the result says what they were taken with, and
            # does not name a --save that would be refused
            r = CLI.main(["--sim", "--from", folder])
            assert r["ok"] and r["taken"]["tool"]["gripper_raw"] == 2182 and r["taken"]["jaw"] is None, r["taken"]
            assert len(r["not_this_tool"]) == 2 and "--save would refuse them" in r["next"] and "stores it" not in r["next"], r["next"]
            assert "`fd calib-air --tip-only --save`" in r["next"] and "保存できません" in r["say"]
        # into the tool file as it was when they were taken, they are stored
        with tool_file(thin_tool([0.020, 0.0, 0.050], on_jaw="moving_outer", gripper_raw=2182)):
            r = CLI.main(["--sim", "--from", folder])
            assert "not_this_tool" not in r and f"`fd calib-air --from {folder} --save` stores it" in r["next"], r["next"]
            r = CLI.main(["--sim", "--from", folder, "--save"])
            assert r["ok"] and r["saved"] is True and "forced" not in r["tip_measured"], r
        # records that name no tool: nothing says what the photos show. Not stored, unless --force says they are of this tool
        put(None)
        with tool_file(thin_tool([0.020, 0.0, 0.050])):
            before = open(TOOL_PATH, "rb").read()
            r = CLI.main(["--sim", "--from", folder, "--save"])
            assert r["ok"] is False and r["stage"] == "save" and "name no tool" in r["reason"] and open(TOOL_PATH, "rb").read() == before, r
            assert f"`fd calib-air --from {folder} --save --force`" in r["next"] and r["gates"]["passed"] is True
            r = CLI.main(["--sim", "--from", folder, "--save", "--force"])
            assert r["ok"] and r["saved"] is True and len(r["forced"]) == 1 and "name no tool" in r["forced"][0], r
            assert TL.active().tip_measured["forced"] == r["forced"]
        # ... but where their records carry the jaw as the servo read it, --force does not help photos of another jaw state
        put(None, [dict(rec, gripper_raw=2182, gripper_goal_raw=2182) for rec in data["records"]])
        with tool_file(now):
            before = open(TOOL_PATH, "rb").read()
            r = CLI.main(["--sim", "--from", folder, "--save", "--force"])
            assert r["ok"] is False and r["stage"] == "save" and "read from the servo" in r["reason"] and open(TOOL_PATH, "rb").read() == before, r
        # two record files of one folder that were taken with different jaw states: not one measurement
        put(then(tip=[0.020, 0.0, 0.050], gripper_raw=2182))
        other = os.path.join(folder, "records_1_low.json")
        with open(other, "w") as fh:
            json.dump({"tool": then(tip=[0.020, 0.0, 0.050], gripper_raw=2041), "records": []}, fh)
        try:
            with tool_file(thin_tool([0.020, 0.0, 0.050], gripper_raw=2182)):
                before = open(TOOL_PATH, "rb").read()
                r = CLI.main(["--sim", "--from", folder, "--save", "--force"])
                assert r["ok"] is False and r["stage"] == "save" and "gripper_raw (2182, 2041)" in r["reason"], r
                assert open(TOOL_PATH, "rb").read() == before
        finally:
            os.remove(other)
    finally:
        CLI._solve = solve
        with open(rec_path, "w") as fh:
            fh.write(original)
    assert len(fits) == 1

    # --- a capture. The jaw of a tool without a hold stands 59 ticks from where its file has it (a limp jaw that was
    # pushed, or closed by something that did not record it). Asked to --save: refused before the arm moves, not
    # after three minutes of poses.
    def pushed(arm_, bus_):
        bus_.pos["gripper"] = bus_.goal["gripper"] = 2100.0

    # (the tip on record as measured: the eight photos of the short sets never pass the gates, and since round 3 an
    # arm whose registered tip was never measured is then not folded and the result not ok. The jaw is the subject here)
    registered = measured_tool([0.020, 0.0, 0.050], gripper_raw=2041)
    with tool_file(registered), short_sets():
        before = open(TOOL_PATH, "rb").read()
        res, arm, bus, photo, root = run_command(["--no-look", "--save"], registered, THIN_TIP, prepare=pushed)
        assert res["ok"] is False and res["stage"] == "tool" and "raw 2100" in res["reason"] and "raw 2041" in res["reason"], res
        assert "Nothing moved" in res["reason"] and "`fd tool jaw close`" in res["next"] and "`fd tool jaw status`" in res["next"]
        assert not photo.shots and bus.io["writes"] == 0 and not any(bus.torque.values()) and res["arm"]["torque"] == "off"
        # the plan says the same (the run it describes would not move): refused
        saved = CLI._make_arm
        CLI._make_arm = lambda sim: arm
        try:
            r = CLI.main(["--sim", "--dry"])
        finally:
            CLI._make_arm = saved
        assert r["ok"] is False and r["stage"] == "tool" and "raw 2100" in r["reason"] and "`fd tool jaw close`" in r["next"], r
        # not asked to store: no pose is flown with it either (tooling.jaw_in_place: the jaw of every tool that records
        # a jaw position, held or not, is compared before the torque comes on)
        res, arm, bus, photo, root = run_command(["--no-look"], registered, THIN_TIP, prepare=pushed, hand_every=0)
        assert res["ok"] is False and res["stage"] == "tool" and "raw 2100" in res["reason"] and not photo.shots, res
        assert bus.io["writes"] == 0 and not any(bus.torque.values())
        # the jaw gives way DURING the capture (in place when it starts, 59 ticks off from the fourth photo on): the
        # records have it, the run does not offer to store what it measured, --save refuses, and so does --from later
        box = {}

        def gives_way(n):
            if n == 3:
                box["bus"].pos["gripper"] = box["bus"].goal["gripper"] = 2100.0
            return False

        res, arm, bus, photo, root = run_command(["--no-look"], registered, THIN_TIP, prepare=lambda a_, b_: box.update(bus=b_),
                                                 hide=gives_way, hand_every=0)
        recs = CLI.read_records(res["dir"])[0]
        assert res["ok"] is True and len(recs) == 8 and [r["gripper_raw"] for r in recs] == [2041] * 3 + [2100] * 5, res
        assert len(res["not_this_tool"]) == 1 and "raw 2041 .. 2100" in res["not_this_tool"][0], res["not_this_tool"]
        assert "--save would refuse them" in res["next"] and "stores it" not in res["next"] and "保存できません" in res["say"], res["next"]
        for force in ([], ["--force"]):
            r = CLI.main(["--sim", "--from", res["dir"], "--save", *force])
            assert r["ok"] is False and r["stage"] == "save" and "read from the servo" in r["reason"], r
            assert open(TOOL_PATH, "rb").read() == before
        res, arm, bus, photo, root = run_command(["--no-look", "--save", "--force"], registered, THIN_TIP,
                                                 prepare=lambda a_, b_: box.update(bus=b_), hide=gives_way, hand_every=0)
        assert res["ok"] is False and res["stage"] == "save" and res["saved"] is False and "read from the servo" in res["reason"], res
        assert len(photo.shots) == 8 and open(TOOL_PATH, "rb").read() == before
        # with the jaw where the file has it, the same command runs to its end and offers to store
        res, arm, bus, photo, root = run_command(["--no-look"], registered, THIN_TIP, hand_every=0)
        assert res["sets"][0]["photos"] == 8 and "not_this_tool" not in res and "--save" in res["next"], res["next"]
        assert not any("cannot be stored" in n for n in res.get("notes", []))


def test_with_a_painting_setup_the_arm_folds_where_fd_paint_rest_would_and_every_hint_says_so():
    """Findings 10, 31 and 50. The tool's default rest pose (pan 20) has the brush 4 cm INSIDE the sheet of
    2026-11-13, and with the torque off the wrist sank there until the brush stood on the paper. calib-air folded
    into that pose and named `fd tool rest` (torque off) after every stop. With a painting set-up on record it
    folds where paintskill.rest_choice says and names `fd paint rest`; without one, the tool's rest pose and
    `fd tool rest --pan=A --keep-torque` (A: where the arm points), with the reason for another angle."""
    import paintskill as PS      # (part of the tree: a painting module that does not load is a failure, not a skip)
    # (the tip on record as measured, 7 mm from the real one: the eight photos of the short sets never pass the gates,
    # and since round 3 an arm whose registered tip was never measured is not folded then. Where it folds is the
    # subject here; test_the_arm_is_folded_only_with_a_tip_that_is_confirmed is about whether)
    registered = measured_tool([0.020, 0.0, 0.050])
    real = thin_tool(THIN_TIP)

    def brush_at(arm, q):
        """(how far the REAL brush tip is outside the canvas, how far the place it would sink to is), metres."""
        tip = K.fk(q, K.tool_offset(*real.tip_offset))[:3, 3]
        _lands, low = PS.sink_of(arm, real, q)
        return PS.canvas_distance(PAINT_SETUP, float(tip[0]), float(tip[1])), PS.canvas_distance(PAINT_SETUP, *low)

    with tool_file(registered), short_sets():
        # --- no painting set-up: the tool's own rest pose, which is over that sheet
        assert CLI._painting() == {"PS": None, "path": None, "setup": None, "error": None}
        res, arm, bus, photo, root = run_command(["--no-look"], registered, THIN_TIP, hand_every=0)
        q = q_of(bus.pos)
        assert res["sets"][0]["photos"] == 8 and res["end"]["fold"]["by"] == "tool" and "rest pose" in res["arm"]["at"], res
        assert abs(math.degrees(q[0]) - TL.REST_PAN_DEG) < 1.0 and all(bus.torque[j] == 1 for j in ARM)
        tip_out, sink_out = brush_at(arm, q)
        assert tip_out < -0.030 and sink_out < -0.030, (tip_out, sink_out)       # 4 and 5.5 cm INSIDE the canvas
        nxt = res["arm"]["next"]
        assert folds_where_it_points(nxt, res["arm"]) and "wrist can sink" in nxt and "`fd tool rest`" not in nxt, nxt
        assert f"`fd tool rest --pan={TL.REST_PAN_DEG:g} --keep-torque`" in nxt                 # (it lies folded at the tool's own pan)
        assert "fd paint rest" not in nxt
        # --- with the set-up of that evening on record
        with paint_setup():
            assert CLI._painting()["path"] == SETUP_PATH and CLI._painting()["setup"]["canvas"]["size_mm"] == [224, 153]
            res, arm, bus, photo, root = run_command(["--no-look"], registered, THIN_TIP, hand_every=0)
            q = q_of(bus.pos)
            fold = res["end"]["fold"]
            assert res["sets"][0]["photos"] == 8 and fold["by"] == "paint" and fold["setup"] == SETUP_PATH, res["end"]
            # where `fd paint rest` would fold this tool: the same pan, and the brush and its sink point off the canvas.
            # (The registered tool: eight photos never pass the gates, so no tip of this run replaces it for the fold.)
            assert fold["planned_with_tip_mm"] == [20.0, 0.0, 50.0] and "registered tip" in res["arm"]["tip"], (fold, res["arm"])
            choice = PS.rest_choice(arm, PAINT_SETUP)
            assert abs(fold["pan_deg"] - choice["pan_deg"]) < 0.11 and abs(math.degrees(q[0]) - choice["pan_deg"]) < 1.0, (fold, choice["pan_deg"])
            assert abs(fold["pan_deg"] - TL.REST_PAN_DEG) > 45.0 and fold["clear_of_canvas_mm"] >= 30.0
            tip_out, sink_out = brush_at(arm, q)
            assert tip_out > 0.030 and sink_out > 0.030, (tip_out, sink_out)
            assert all(bus.torque[j] == 1 for j in ARM) and res["arm"]["torque"] == "on"        # never off here
            assert "fd paint rest" in res["arm"]["at"] and "rest pose" in res["arm"]["at"] and "トルク ON" in res["say"]
            assert "`fd paint rest`" in res["arm"]["next"] and "fd tool rest --pan" not in res["arm"]["next"]
            assert fold["canvas"]["ok"] is True and fold["canvas"]["clear_of_canvas_mm"] >= 30.0, fold["canvas"]     # (paintskill.rest_report)
            # every hint after a stop names `fd paint rest`; `fd tool rest` only to say that it is not the one
            stops = [("an interrupt", dict(fail_at=(3, KeyboardInterrupt()))),
                     ("a failing camera", dict(fail_at=(3, RuntimeError("camera returned no frames"))))]
            for what, camera in stops:
                res, arm, bus, photo, root = run_command(["--no-look"], registered, THIN_TIP, **camera)
                assert res["ok"] is False and res["stage"] == "capture" and res["arm"]["torque"] == "on", (what, res)
                assert "`fd paint rest`" in res["next"] and "fd tool rest --pan" not in res["next"], (what, res["next"])
                assert "not `fd tool rest`" in res["next"] and f"`fd calib-air --from {res['dir']}`" in res["next"]
            real_move, calls = TL.free_move, {"n": 0}

            def stopping(arm_, q_to, *a_, **k_):
                calls["n"] += 1
                if calls["n"] == 3:
                    raise MotionAborted("free move: stopped because the tool tip was 18 mm above the table (measured)")
                return real_move(arm_, q_to, *a_, **k_)

            TL.free_move = stopping
            try:
                res, arm, bus, photo, root = run_command(["--no-look"], registered, THIN_TIP)
            finally:
                TL.free_move = real_move
            assert res["stage"] == "capture" and "18 mm" in res["reason"] and "`fd paint rest`" in res["next"], res
            assert "fd tool rest --pan" not in res["next"]
            # the plan says where it will fold
            r = CLI.main(["--sim", "--dry"])
            assert r["ok"] and r["end"]["fold"]["by"] == "paint" and abs(r["end"]["fold"]["pan_deg"] - fold["pan_deg"]) < 6.0, r["end"]
        # --- a set-up that leaves no fold off the canvas (a sheet as large as the arm's reach): refused before the arm moves
        with paint_setup(dict(PAINT_SETUP, canvas={"origin_xy": [-0.3, -0.3], "size_mm": [600, 600], "pitch_deg": 65.0})):
            res, arm, bus, photo, root = run_command(["--no-look"], registered, THIN_TIP)
            assert res["ok"] is False and res["stage"] == "rest" and "Nothing moved" in res["reason"], res
            assert not photo.shots and bus.io["writes"] == 0 and not any(bus.torque.values())
            assert "`fd calib-air --end stay`" in res["next"] and "--end stay" in res["say"]
            assert CLI.main(["--sim", "--dry"])["stage"] == "rest"
        # --- a set-up file that cannot be read: where the canvas lies is not known. Refused, unless the arm is not folded
        with paint_setup("{ not json"):
            res, arm, bus, photo, root = run_command(["--no-look"], registered, THIN_TIP)
            assert res["ok"] is False and res["stage"] == "setup" and "cannot be used" in res["reason"], res
            assert not photo.shots and bus.io["writes"] == 0 and not any(bus.torque.values())
            assert "`fd paint status`" in res["next"] and "`fd calib-air --end stay`" in res["next"]
            res, arm, bus, photo, root = run_command(["--no-look", "--end", "stay"], registered, THIN_TIP, hand_every=0)
            assert res["sets"][0]["photos"] == 8 and "stay" in res["arm"]["at"] and res["arm"]["torque"] == "on", res
            assert "`fd paint rest`" in res["arm"]["next"] and "`fd paint status`" in res["arm"]["next"]
    # --- a look that refuses, with the set-up: the arm stays at the look pose, and the hint names `fd paint rest`
    wrong = thin_tool(SHORT_GUESS)
    with tool_file(wrong), paint_setup():
        res, arm, bus, photo, root = run_command([], wrong, LONG_TIP)
        assert res["stage"] == "look" and "`fd paint rest`" in res["next"] and "fd tool rest --pan" not in res["next"], res["next"]
        assert "fold only when the photo shows the tool no longer than registered" in res["next"]
    # --- a scratch tool file with the RIG's set-up (no $FD_PAINT_SETUP): that set-up describes another table
    keep = os.environ.pop("FD_PAINT_SETUP")
    try:
        assert CLI._painting()["path"] is None
    finally:
        os.environ["FD_PAINT_SETUP"] = keep


def test_a_camera_model_that_cannot_be_read_is_a_refusal_and_the_full_solve_goes_on_without_it():
    """Finding 25. A camera model file with broken JSON, or without "rvec", ended `fd calib-air` in a Python
    traceback (no stage, reason, next or say), also for the full solve, which is the command that replaces it."""
    folder = capture_folder("good")
    made = []
    saved = CLI._make_arm
    damaged = (("not JSON", "{ not json"), ("no rvec", json.dumps({k: v for k, v in CAM_DICT.items() if k != "rvec"})),
               ("a list", "[1, 2]"), ("no numbers", json.dumps(dict(CAM_DICT, tvec=[None, 0.1, 0.6]))),
               ("an empty file", ""))
    try:
        with tool_file(thin_tool([0.020, 0.0, 0.050])):
            before = open(TOOL_PATH, "rb").read()
            for name, text in damaged:
                with open(MODEL_PATH, "w") as fh:
                    fh.write(text)
                CLI._make_arm = lambda sim: made.append(sim)
                for argv in (["--sim", "--dry"], ["--sim", "--dry", "--tip-only"], ["--no-look"], ["--sim", "--from", folder],
                             ["--sim", "--from", folder, "--save"]):
                    r = CLI.main(argv)
                    assert r["ok"] is False and r["stage"] == "camera" and MODEL_PATH in r["reason"] and r["say"], (name, argv, r)
                    assert "cannot be read" in r["reason"] and "--sets air,low" in r["next"] and "error" not in r and "trace" not in r
                assert not made, name                       # refused before the arm is asked for
                assert open(TOOL_PATH, "rb").read() == before
                # through fd: a result with a stage, not a traceback
                r, rc = fd_calib_air("--dry", sim=True)
                assert rc == 1 and r["stage"] == "camera" and "trace" not in r and "`fd --sim calib-air --sets air,low --dry`" in r["next"], r
                # the full solve keeps nothing of the model: its plan stands, and says what the look will lack
                CLI._make_arm = saved
                r = CLI.main(["--sim", "--dry", "--sets", "air,low"])
                assert r["ok"] and r["stage"] == "planned" and r["mode"] == "full" and r["look"]["registered_px"] is None, (name, r)
                assert len(r["notes"]) == 1 and "cannot be read" in r["notes"][0] and "without a prior" in r["notes"][0]
            # ... and from photos it finds the camera they were taken with (6 mm) and the tip (3 mm) with no model to
            # start from, stores both, and keeps the damaged file in backups/
            with open(MODEL_PATH, "w") as fh:
                fh.write("{ not json")
            r = CLI.main(["--sim", "--from", folder, "--sets", "air,low", "--save"])
            assert r["ok"] and r["saved"] is True and r["camera"]["moved_mm"] is None, r
            assert np.linalg.norm(np.asarray(r["camera"]["center"]) - CAM.center) < 0.006, r["camera"]
            assert np.linalg.norm(np.asarray(r["tip_offset"]) - THIN_TIP) < 0.003, r["tip_mm"]
            assert open(r["files"]["camera_model_backup"]).read() == "{ not json"
            assert np.linalg.norm(C.CameraModel.load().center - CAM.center) < 0.006
    finally:
        CLI._make_arm = saved
        with open(MODEL_PATH, "w") as fh:
            json.dump(CAM_DICT, fh)
        shutil.rmtree(os.path.join(_TMP, "backups"), ignore_errors=True)


def test_sim_stands_before_the_command_and_no_simulated_tool_file_is_written_into_the_project_folder():
    """Finding 26. aircal_cli had a --sim of its own: `fd calib-air -- --sim --from DIR --save` was read by fd as a
    command for the REAL arm (no "sim" in the result), and because fd had not moved the simulated tool file out
    of the tree, tool_active.sim.json was written into the project folder."""
    folder = capture_folder("good")
    made = []
    saved = CLI._make_arm
    CLI._make_arm = lambda sim: made.append(sim)
    try:
        with tool_file(thin_tool([0.020, 0.0, 0.050])):
            before = open(TOOL_PATH, "rb").read()
            for argv in (["--dry", "--sim"], ["--", "--sim", "--dry"], ["--from", folder, "--sim", "--save"],
                         ["--sim", "--dry", "--sim"], ["--sim", "--", "--sim", "--dry"]):
                r = CLI.main(argv)
                assert r["ok"] is False and r["stage"] == "options" and "--sim must stand before the command" in r["reason"], (argv, r)
                assert r["next"].startswith("`fd --sim calib-air ") and "--sim" not in r["next"][len("`fd --sim calib-air "):], r["next"]
            # a caller that knows whether fd was given --sim says so: then it is refused in front as well
            r = CLI.main(["--sim", "--dry"], sim=False)
            assert r["ok"] is False and r["stage"] == "options" and "--sim must stand before the command" in r["reason"], r
            assert not made and open(TOOL_PATH, "rb").read() == before
            # through fd, as it would be typed
            r, rc = fd_calib_air("--dry", "--sim")
            assert rc == 1 and r["stage"] == "options" and "--sim must stand before the command" in r["reason"] and "sim" not in r, r
        # where fd puts it, it is the simulated arm, and the result says so by itself
        CLI._make_arm = saved
        with tool_file(thin_tool()):
            for argv, kw in ((["--sim", "--dry"], {}), (["--dry"], {"sim": True}), (["--sim", "--", "--dry"], {})):
                r = CLI.main(argv, **kw)
                assert r["ok"] and r["stage"] == "planned" and r["sim"] is True, (argv, r)
            r = CLI.main(["--sim", "--from", folder])
            assert r["ok"] and r["sim"] is True
            assert "sim" not in CLI.main(["--from", os.path.join(_TMP, "nowhere")])
    finally:
        CLI._make_arm = saved
    # --- the project folder. A stand-in for it (the tool files and tazuna_paths.ROOT point into a scratch folder,
    # $FD_TOOL is not set): the simulated tool file would lie "in the tree", as it does when fd was not told --sim
    tree = tempfile.mkdtemp(prefix="stand_in_tree_", dir=_TMP)
    keep = (TL.TOOL_FILE, TL.SIM_TOOL_FILE, tazuna_paths.ROOT)
    TL.TOOL_FILE, TL.SIM_TOOL_FILE, tazuna_paths.ROOT = os.path.join(tree, "tool_active.json"), os.path.join(tree, "tool_active.sim.json"), tree
    env = os.environ.pop("FD_TOOL")
    try:
        registered = thin_tool([0.020, 0.0, 0.050])
        with open(TL.TOOL_FILE, "w") as fh:
            json.dump(registered.to_dict(), fh)
        tool_before = open(TL.TOOL_FILE, "rb").read()
        assert TL.tool_path(True) == TL.SIM_TOOL_FILE and CLI._in_project(TL.SIM_TOOL_FILE)
        for argv in (["--sim", "--from", folder, "--save"], ["--sim", "--from", folder], ["--sim", "--dry"]):
            r = CLI.main(argv)
            assert r["ok"] is False and r["stage"] == "options" and "inside the project folder" in r["reason"], (argv, r)
            assert "fd --sim calib-air" in r["next"] and "$FD_TOOL" in r["next"] and r["sim"] is True
        # the last line before the write itself (should the door ever be passed): refused, nothing written
        fit = {"tip_offset": THIN_TIP, "rms_px": 3.0, "used": 30, "photos": 35}
        res = {"tip_sigma_mm": [0.1, 0.1, 0.1], "say": ""}
        CLI._save(res, "tip", fit, {"problems": [], "moved": None, "few": False}, registered, True, True, registered.to_dict(), [], folder)
        assert res["ok"] is False and res["stage"] == "save" and res["saved"] is False and "project folder" in res["reason"], res
        assert sorted(os.listdir(tree)) == ["tool_active.json"] and open(TL.TOOL_FILE, "rb").read() == tool_before
        # where fd keeps the simulated tool (outside the tree), a simulated --save writes THAT file and says so; the
        # real tool file is not touched
        TL.SIM_TOOL_FILE = os.path.join(tempfile.mkdtemp(prefix="sim_state_", dir=_TMP), "tool_active.sim.json")
        r = CLI.main(["--sim", "--from", folder, "--save"])
        assert r["ok"] and r["saved"] is True and r["files"] == {"tool": TL.SIM_TOOL_FILE} and r["simulated_tool"] is True, r
        assert "fd --sim tool detach" in r["next"] and os.path.isfile(TL.SIM_TOOL_FILE) and r["sim"] is True
        # (a copy of the real tool's record: read until the real tool file changes, as tooling says of it)
        assert "until the real tool file changes" in r["next"] and "no longer follows" not in r["next"], r["next"]
        assert TL.sim_tool_state()["copy_of_real"] is True
        assert sorted(os.listdir(tree)) == ["tool_active.json"] and open(TL.TOOL_FILE, "rb").read() == tool_before
        os.remove(TL.SIM_TOOL_FILE)
        TL.SIM_TOOL_FILE = os.path.join(tree, "tool_active.sim.json")
        # the same call for the real tool file is what --save is for (no --sim): it is written
        CLI._save(res, "tip", fit, {"problems": [], "moved": None, "few": False}, registered, False, False, registered.to_dict(), [], folder)
        assert res["saved"] is True and "simulated_tool" not in res and sorted(os.listdir(tree)) == ["tool_active.json"]
        assert open(TL.TOOL_FILE, "rb").read() != tool_before
    finally:
        os.environ["FD_TOOL"] = env
        TL.TOOL_FILE, TL.SIM_TOOL_FILE, tazuna_paths.ROOT = keep
    assert TL.tool_path() == TOOL_PATH and not CLI._in_project(TOOL_PATH)


def test_a_photo_that_cannot_be_stored_stops_the_capture_and_records_are_written_whole():
    """Finding 42. cv2.imwrite returns False for a folder that cannot be written or a disk that is full (this
    computer: 2.7 GB free, 98 % used) and raises nothing: the capture flew every pose, three to five minutes, and
    ended with "the photos give no fit". The record file was written in place: an interrupt in the middle left
    half a file, and `--from DIR` then refused the whole folder."""
    registered = thin_tool([0.020, 0.0, 0.050])
    real_write = cv2.imwrite
    count = {"n": 0}

    def full_after(k, empty=False):
        def write(path, img, *params):
            count["n"] += 1
            if count["n"] <= k:
                return real_write(path, img, *params)
            if empty:
                open(path, "wb").close()        # (a file system that makes the name and has no room for the bytes)
                return True
            return False
        count["n"] = 0
        return write

    try:
        # --- aircal.capture: the fourth photo cannot be stored
        arm, bus = sim_arm(registered)
        poses = AC.plan_set(arm, registered, AC.POSE_SETS["tip"]["high"])[:8]
        out = tempfile.mkdtemp(prefix="cap_full_disk_", dir=_TMP)
        photo = camera_for(arm, THIN_TIP)
        cv2.imwrite = full_after(3)
        cap = AC.capture(arm, poses, photo, out, prefix="high")
        assert len(cap["records"]) == 3 and len(photo.shots) == 4 and cap["interrupted"] is False, (len(cap["records"]), len(photo.shots))
        assert cap["photo_error"] == cap["stopped"] and "pose 3" in cap["stopped"] and "high_03.jpg" in cap["stopped"], cap["stopped"]
        assert "could not be stored" in cap["stopped"] and "OpenCV could not write the file" in cap["stopped"]
        assert sorted(os.listdir(out)) == ["high_00.jpg", "high_01.jpg", "high_02.jpg"]
        assert np.allclose(q_of(bus.pos), photo.shots[3], atol=math.radians(0.3)) and all(bus.torque[j] == 1 for j in ARM)   # it holds there
        # a file of no bytes is no photo either
        arm, bus = sim_arm(registered)
        photo = camera_for(arm, THIN_TIP)
        cv2.imwrite = full_after(1, empty=True)
        cap = AC.capture(arm, poses, photo, tempfile.mkdtemp(prefix="cap_empty_", dir=_TMP), prefix="high")
        assert len(cap["records"]) == 1 and len(photo.shots) == 2 and "the file is empty" in cap["photo_error"], cap["photo_error"]
        assert AC.store_photo(os.path.join(_TMP, "no_such_folder", "x.jpg"), BASE) is not None
        cv2.imwrite = real_write
        ok_path = os.path.join(tempfile.mkdtemp(prefix="stored_", dir=_TMP), "stored.jpg")
        assert AC.store_photo(ok_path, BASE) is None and os.path.getsize(ok_path) > 10000
        # --- the command: it stops there, says so, and what was taken is on record
        with tool_file(registered):
            before = open(TOOL_PATH, "rb").read()
            cv2.imwrite = full_after(3)
            res, arm, bus, photo, root = run_command(["--no-look", "--save"], registered, THIN_TIP)
            cv2.imwrite = real_write
            assert res["ok"] is False and res["stage"] == "capture" and res["error"] == "PhotoNotStored" and res["saved"] is False, res
            assert "could not be stored" in res["reason"] and "high_03.jpg" in res["reason"] and len(photo.shots) == 4, res["reason"]
            assert "`df -h`" in res["next"] and folds_where_it_points(res["next"], res["arm"]), res["next"]
            assert f"`fd calib-air --from {res['dir']}`" in res["next"] and "写真を保存できなかった" in res["say"]
            assert res["arm"]["torque"] == "on" and "could not be stored" in res["arm"]["at"] and len(res["arm"]["q_deg"]) == 5
            assert np.allclose(np.radians(res["arm"]["q_deg"]), photo.shots[3], atol=math.radians(0.3))
            recs, _tool, names = CLI.read_records(res["dir"])
            assert len(recs) == 3 and names == ["records_0_high.json"] and open(TOOL_PATH, "rb").read() == before
            # the look's own photo: the same, before any calibration pose
            cv2.imwrite = full_after(1)
            res, arm, bus, photo, root = run_command([], registered, THIN_TIP)
            cv2.imwrite = real_write
            assert res["ok"] is False and res["stage"] == "look" and res["error"] == "PhotoNotStored" and "look.jpg" in res["reason"], res
            assert len(photo.shots) == 2 and res["sets"] == [] and res["arm"]["torque"] == "on"
    finally:
        cv2.imwrite = real_write
    # --- the record file: an interrupt while it is written leaves the file that was there, whole
    out = tempfile.mkdtemp(prefix="records_whole_", dir=_TMP)
    first = {"records": [{"i": 0, "q": [0.0] * 5, "image": "a.jpg"}], "skipped": [], "stopped": None}
    path = AC.save_records(out, registered, first, "records_0_high.json")
    assert json.load(open(path))["tool"]["name"] == "brush_thin" and os.listdir(out) == ["records_0_high.json"]
    whole = open(path).read()
    real_dump = AC.json.dump

    def cut_short(obj, fh, **kw):
        fh.write('{"tool": {"na')
        fh.flush()
        raise KeyboardInterrupt

    AC.json.dump = cut_short
    try:
        raises(KeyboardInterrupt, AC.save_records, out, registered, dict(first, records=[]), "records_0_high.json")
        raises(KeyboardInterrupt, AC.save_records, out, registered, first, "records_1_low.json")
    finally:
        AC.json.dump = real_dump
    assert open(path).read() == whole and os.listdir(out) == ["records_0_high.json"], os.listdir(out)
    assert CLI.read_records(out)[2] == ["records_0_high.json"]


# ----------------------------------------------------------------------------- round 2 of the fix (2026-11-14)
ACROSS_TIP = [0.085, 0.0, 0.045]         # a tool taped ACROSS the fingers on the fixed-jaw side (the verifier's case E)
ACROSS_TIP_F = [0.070, 0.0, 0.040]       # ... a shorter one, registered as 6 cm along the fingers (case F)
NO_FOLD_TIP = [0.100, 0.0, 0.030]        # ... one that cannot lie folded: no wrist angle inside wrist_flex 20..97 keeps it up


def on_the_line_of_sight(registered_tip, real_tip) -> list:
    """The registration a refused look suggests (its `next` prints it as `fd tool set --tip=...`): the registered tip
    moved onto the line of sight of the tip the look photo shows. One photo cannot tell the two apart, so the next
    look passes with it, however far along that line the real tip is."""
    arm, _bus = sim_arm(thin_tool(registered_tip))
    q = AC.look_pose(arm, arm.tool, CAM)["q"]
    chk = AC.look_check(draw_tool(BASE.copy(), CAM, q, real_tip), CAM, q, registered_tip, bg=BASE)
    assert chk["ok"] is False and chk["suggest_tip"] is not None, chk
    return [float(v) for v in chk["suggest_tip"]]


def test_the_fold_at_the_end_is_planned_with_the_tip_the_photos_gave():
    """Round 2, R1. A tool taped across the fingers (tip really at (85, 0, 45) mm), registered "8 cm along the
    fingers". The look refuses and prints `fd tool set --tip=...`: the registration moved onto the tip's line of
    sight, 93 mm from the real tip. Typed as printed, the next look passes, the high set is flown and solved to a
    fraction of a millimetre, the low set is rightly not flown (the jump rule) -- and until 2026-11-14 the arm was
    then folded with the REGISTERED tip: the real tip ended 17 mm under the table with a painting set-up, 10 mm
    without one, result ok (the second case, 6 cm along / 7 cm across: 3 mm under). Now what moves after a fit that
    passed the gates is planned with its tip, and where no fold is safe with that tip the arm is not folded."""
    import paintskill as PS      # (part of the tree: a painting module that does not load is a failure, not a skip)
    real = thin_tool(ACROSS_TIP)
    registered = thin_tool(SHORT_GUESS)
    with tool_file(registered), paint_setup():
        # 1) the look refuses, and prints the registration to type
        res, arm, bus, photo, root = run_command([], registered, ACROSS_TIP, prepare=at_rest_of(ACROSS_TIP), hand_every=0)
        s = res["look"]["suggest_tip"]
        assert res["stage"] == "look" and f"`fd tool set --tip={s[0]:.4f},{s[1]:.4f},{s[2]:.4f}`" in res["next"], res["next"]
        assert 0.08 < np.linalg.norm(np.subtract(s, ACROSS_TIP)) < 0.11, s
        # 2) typed as printed (what `fd tool set --tip=...` writes), then the same command, from the look pose
        t = TL.active()
        t.tip_offset = [float(v) for v in s]
        TL.save(t)
        suggested = TL.active()
        res, arm, bus, photo, root = run_command([], suggested, ACROSS_TIP, prepare=held_at_look_pose(ACROSS_TIP), hand_every=0)
        assert res["ok"] is True and res["stage"] == "done" and res["look"]["ok"] is True and res["look"]["nearest_px"] < 10.0, res
        assert [x["name"] for x in res["sets"]] == ["high"] and any("was not flown" in n for n in res["notes"]), res["notes"]
        assert np.linalg.norm(np.asarray(res["tip_offset"]) - ACROSS_TIP) < 0.0015 and res["from_registered_mm"] > 80.0, res["tip_mm"]
        # the fold: planned with the tip the photos gave, and the REAL tip never came lower than 2 cm (it was -17 mm)
        fold = res["end"]["fold"]
        assert fold["by"] == "paint" and np.linalg.norm(np.asarray(fold["planned_with_tip_mm"]) / 1000.0 - ACROSS_TIP) < 0.0015, fold
        assert res["arm"]["tip"] == "modelled with the tip solved in this run"
        low = min(real_heights(arm, bus, suggested, ACROSS_TIP))
        assert low >= 0.020, low
        q_end = q_of(bus.pos)
        assert abs(TL.tip_height(arm, q_end, real) - fold["clearance_m"]) < 0.004 and TL.in_rest_pose(arm, real, q_end)[0], np.degrees(q_end)
        assert all(bus.torque[j] == 1 for j in ARM) and res["arm"]["torque"] == "on"
        tip_xy = K.fk(q_end, K.tool_offset(*ACROSS_TIP))[:3, 3]
        assert PS.canvas_distance(PAINT_SETUP, float(tip_xy[0]), float(tip_xy[1])) > 0.030
        # What folds the arm or switches it off next plans with the tip the TOOL FILE has: every hint names the
        # storing first ...
        stores = f"`fd calib-air --from {res['dir']} --save`"
        nxt = res["arm"]["next"]
        assert stores in res["next"] and "FIRST " + stores in nxt and nxt.index(stores) < nxt.index("`fd paint rest`"), nxt
        assert "先にこの測定結果を保存" in res["say"]
        # ... because with the registered tip on file `fd paint rest` would fold the arm AGAIN, into the table
        arm.tool = suggested
        q_registered = PS.rest_choice(arm, PAINT_SETUP)["q"]
        assert np.max(np.abs(q_registered[:4] - q_end[:4])) > math.radians(10.0) and TL.tip_height(arm, q_registered, real) < 0.0
        # stored as the hint says: the fold `fd paint rest` chooses is where the arm lies (it only switches the torque off)
        assert CLI.main(["--sim", "--from", res["dir"], "--save"])["saved"] is True
        arm.tool = TL.active()
        q_stored = PS.rest_choice(arm, PAINT_SETUP)["q"]
        assert np.max(np.abs(q_stored[:4] - q_end[:4])) < math.radians(3.0), np.degrees(q_stored - q_end)
    # --- no painting set-up (the tool's own rest pose: the real tip ended 10 mm under the table), stored in the same run
    with tool_file(thin_tool(s)):
        res, arm, bus, photo, root = run_command(["--save"], TL.active(), ACROSS_TIP, prepare=held_at_look_pose(ACROSS_TIP), hand_every=0)
        assert res["ok"] is True and res["saved"] is True and res["end"]["fold"]["by"] == "tool", res
        low = min(real_heights(arm, bus, thin_tool(s), ACROSS_TIP))
        assert low >= 0.020 and abs(TL.tip_height(arm, q_of(bus.pos), real) - TL.REST_CLEARANCE) < 0.004, low
        assert np.linalg.norm(np.asarray(TL.active().tip_offset) - ACROSS_TIP) < 0.0015
        # the tool file has the tip the arm was folded with: the plain hint, no "store first"
        assert "FIRST" not in res["arm"]["next"] and folds_where_it_points(res["arm"]["next"], res["arm"]), res["arm"]["next"]
        assert "先にこの測定結果を保存" not in res["say"] and "保存しました" in res["say"] and "arm_next_stored" not in res
    # --- the verifier's second case. (Round 3: for THIS registration no turn of the look pose keeps everything a tool of
    # 30 cm can reach clear of the place of the look: the look is refused before anything moves, and the capture runs
    # on the operator's own look, --no-look. The fold is the subject here.)
    with tool_file(thin_tool(on_the_line_of_sight([-0.001, 0.0, 0.0668], ACROSS_TIP_F))), paint_setup():
        res, arm, bus, photo, root = run_command([], TL.active(), ACROSS_TIP_F, prepare=held_at_look_pose(ACROSS_TIP_F), hand_every=0)
        assert res["ok"] is False and res["stage"] == "look" and "cannot be turned aside" in res["reason"] and not photo.shots, res
        assert all(np.max(np.abs(q_of(p) - q_of(bus.pos))) < math.radians(0.5) for p in bus.trace), "the arm moved"
        res, arm, bus, photo, root = run_command(["--no-look"], TL.active(), ACROSS_TIP_F, prepare=held_at_look_pose(ACROSS_TIP_F),
                                                 hand_every=0)
        assert res["ok"] is True and res["look"] == {"skipped": "--no-look"} and res["from_registered_mm"] > 60.0, res
        low = min(real_heights(arm, bus, TL.active(), ACROSS_TIP_F))
        assert low >= 0.020 and "solved in this run" in res["arm"]["tip"], low
    # --- one set only (`--sets air`): its photos are solved BEFORE the fold, for the same reason
    with tool_file(thin_tool(s)), paint_setup():
        res, arm, bus, photo, root = run_command(["--sets", "air"], TL.active(), ACROSS_TIP, prepare=held_at_look_pose(ACROSS_TIP),
                                                 hand_every=0)
        assert res["ok"] is True and res["mode"] == "full" and [x["name"] for x in res["sets"]] == ["air"], res
        low = min(real_heights(arm, bus, TL.active(), ACROSS_TIP))
        assert low >= 0.020 and "solved in this run" in res["arm"]["tip"] and "FIRST `fd calib-air --from" in res["arm"]["next"], low
    # --- no fold is safe with the solved tip (10 cm across the fingers): the arm is NOT folded. It holds where the last
    # photo was taken, torque on; the result is not ok, says why, and names the storing before anything that folds
    arm0, _bus = sim_arm(registered)
    raises(SafetyError, TL.rest_q, arm0, thin_tool(NO_FOLD_TIP))
    box = {"marks": []}

    def at_look(arm_, bus_):        # (this tool cannot lie folded: the arm stands at the look pose, as a refused look leaves it)
        assert TL.sim_place(arm_, AC.look_pose(arm_, arm_.tool, CAM)["q"])
        TL._power(arm_, None)
        box["bus"] = bus_

    def mark(n):                    # (where the trace stood when each photo was taken)
        box["marks"].append(len(box["bus"].trace))
        return False

    with tool_file(thin_tool(on_the_line_of_sight(SHORT_GUESS, NO_FOLD_TIP))), paint_setup():
        before = open(TOOL_PATH, "rb").read()
        # (round 3: for this registration, too, no turn of the look pose keeps the reach of a 30 cm tool clear of the
        # place of the look, and the look is refused before anything moves: the capture runs on --no-look)
        res, arm, bus, photo, root = run_command(["--save", "--no-look"], TL.active(), NO_FOLD_TIP, prepare=at_look, hide=mark,
                                                 hand_every=0)
        assert res["ok"] is False and res["stage"] == "end" and res["look"] == {"skipped": "--no-look"}, res
        assert "was not folded away" in res["reason"] and "the tip solved in this run" in res["reason"], res["reason"]
        assert res["end"]["fold"] is None and "cannot" in res["end"]["not_folded"], res["end"]
        # the measurement is reported all the same
        assert np.linalg.norm(np.asarray(res["tip_offset"]) - NO_FOLD_TIP) < 0.0015 and res["gates"]["passed"] is True, res["tip_mm"]
        # nothing moved after the last photo; the arm holds there
        tail = bus.trace[box["marks"][-1]:]
        assert all(np.max(np.abs(q_of(p) - photo.shots[-1])) < math.radians(0.5) for p in tail), "the arm moved after the last photo"
        assert np.allclose(q_of(bus.pos), photo.shots[-1], atol=math.radians(0.5)) and all(bus.torque[j] == 1 for j in ARM)
        assert res["arm"]["torque"] == "on" and TL.tip_height(arm, q_of(bus.pos), thin_tool(NO_FOLD_TIP)) > 0.030
        stores = f"`fd calib-air --from {res['dir']} --save`"
        assert "`fd state`" in res["next"] and "FIRST " + stores in res["next"], res["next"]
        assert res["next"].index(stores) < res["next"].index("`fd paint rest`")
        assert "畳んでいません" in res["say"] and "トルク ON" in res["say"] and "先にこの測定結果を保存" in res["say"], res["say"]
        # not folded = the command did not end as planned: --save stores nothing by itself
        assert res["saved"] is False and open(TOOL_PATH, "rb").read() == before and any("nothing was stored" in n for n in res["notes"])
    # --- the words (also for --end stay: the arm stands at its last pose)
    first = CLI._store_first("fd calib-air --from /d --save", 0.093)
    assert first.startswith("FIRST `fd calib-air --from /d --save` (no motion)") and "93 mm" in first, first
    none, with_setup = {"path": None, "error": None}, {"path": SETUP_PATH, "error": None}
    for paint, fold in ((none, None), (with_setup, None), (with_setup, {"pan_deg": -89.2}), (none, {"pan_deg": 20.0})):
        plain, far = CLI._torque_on_next(paint, fold, pan_deg=-12.34), CLI._torque_on_next(paint, fold, first, -12.34)
        assert "FIRST" not in plain and far.replace(first, "") == plain and plain.startswith("the torque is ON"), (plain, far)
        assert far.index("--save") < far.index("`fd paint rest`" if paint["path"] else "`fd tool rest --pan=-12.3 --keep-torque`"), far
    # (an arm that could not be read: no command that would have to guess its base angle)
    assert "`fd tool rest" not in CLI._fold_hint(none) and "`fd state`" in CLI._fold_hint(none) and "--pan" in CLI._fold_hint(none)


def test_a_stored_tip_leads_to_fd_paint_status_and_the_refusals_name_the_way_on():
    """Round 2: R2 and R28 (what a stored tip leads to, what goes on record with it) and R6 (photos from before a
    jaw hold went on record). Until 2026-11-14 the result of `fd calib-air --from DIR --save` named `fd tool status`,
    which sent a brush to `fd tool calibrate`, the PEN calibration (its tip stage lowered the brush 9 .. 15 mm below
    first touch inside the canvas, simulated); a tool file without a jaw position was never compared with the jaw
    again; and on the rig's own files the order `fd paint status` gave ended in "the photos were taken with no jaw
    hold on record", with no way on but a new capture."""
    folder = capture_folder("good")             # (its records: no reading of the jaw, the tool without a jaw position)
    data = json.load(open(os.path.join(folder, "records_0_high.json")))
    fits = {}
    solve = CLI._solve

    def solve_once(mode, images, records, cam, guess):      # (the same 35 photos again and again: what happens to the fit differs)
        key = (mode, len(records))
        if key not in fits:
            fits[key] = solve(mode, images, records, cam, guess)
        return copy.deepcopy(fits[key])

    def copy_of(tool_block=None, raws=None):
        """A capture folder with the same photos: another tool in its records, or a jaw reading in every record."""
        d = tempfile.mkdtemp(prefix="capture_as_", dir=_TMP)
        recs = data["records"] if raws is None else [dict(rec, gripper_raw=raw, gripper_goal_raw=raw) for rec, raw in zip(data["records"], raws)]
        with open(os.path.join(d, "records_0_high.json"), "w") as fh:
            json.dump(dict(data, tool=data["tool"] if tool_block is None else tool_block, records=recs), fh)
        return d

    pen_cal = dict(contact={"x0": 0.2, "y0": 0.0, "a": 0.004, "bx": 0.0, "by": 0.0, "rms_mm": 0.3, "points": []},
                   calibrated={"contact": {"at": "2026-11-13T17:40:00"}})
    CLI._solve = solve_once
    try:
        # --- R2 / R28(c). A painting set-up on record: next is `fd paint status`, and what it will find first
        with tool_file(thin_tool([0.020, 0.0, 0.050], **pen_cal)), paint_setup():
            r = CLI.main(["--sim", "--from", folder, "--save"])
            assert r["saved"] is True and r["tip_changed_mm"] > 5.0 and "contact" in r["dropped"], r
            assert r["next"].startswith("`fd paint status`") and "fd tool calibrate" not in r["next"] + r["say"], r["next"]
            assert "fd tool status" not in r["next"] and "measured again, not adopted" in r["next"] and "--adopt`" not in r["next"]
            assert "fd paint status" in r["say"] and "測り直し" in r["say"]
            r = CLI.main(["--sim", "--from", folder, "--save"])         # the same tip again: nothing moved
            assert r["tip_changed_mm"] == 0.0 and r["next"].startswith("`fd paint status`"), r["next"]
            assert "`fd paint contact --adopt`" in r["next"] and "fd tool calibrate" not in r["next"] + r["say"]
        # ... no painting set-up: the tool; a PEN whose calibration the new tip dropped is told the word fd asks for
        with tool_file(thin_tool([0.020, 0.0, 0.050], **pen_cal)):
            r = CLI.main(["--sim", "--from", folder, "--save"])
            assert r["next"].startswith("`fd tool status`") and "`fd tool calibrate --pen`" in r["next"], r["next"]
            assert "`fd tool calibrate`" not in r["next"] and "fd paint status" not in r["next"] + r["say"]
            r = CLI.main(["--sim", "--from", folder, "--save"])
            assert r["next"].startswith("`fd tool status`") and "fd tool calibrate" not in r["next"]
        # --- R28(b). A tool file without a jaw position gets the one the servo reported during the photos
        steady = copy_of(raws=[2040 + k % 3 for k in range(len(data["records"]))])
        with tool_file(thin_tool([0.020, 0.0, 0.050])):
            assert TL.active().gripper_raw is None and TL.jaw_tolerance(TL.active()) is None
            r = CLI.main(["--sim", "--from", steady, "--save"])
            t = TL.active()
            assert r["saved"] is True and r["jaw_recorded_raw"] == 2041 == t.gripper_raw and t.tip_measured["jaw_raw"] == [2040, 2042], r
            assert TL.jaw_tolerance(t) == TL.JAW_END_TOL_TICKS
            assert any("raw 2041" in n and "`fd tool jaw status`" in n for n in r["notes"]), r["notes"]
            # from now on photos are compared with it, like those of any tool that records its jaw
            r = CLI.main(["--sim", "--from", steady, "--save"])
            assert r["saved"] is True and "jaw_recorded_raw" not in r and TL.active().gripper_raw == 2041
            r = CLI.main(["--sim", "--from", copy_of(raws=[2100] * len(data["records"])), "--save"])
            assert r["ok"] is False and r["stage"] == "save" and "raw 2100" in r["reason"] and "raw 2041" in r["reason"], r
        # ... a jaw that moved during the photos names no one position: none is stored, and the result says so
        with tool_file(thin_tool([0.020, 0.0, 0.050])):
            r = CLI.main(["--sim", "--from", copy_of(raws=[2041] * 20 + [2100] * (len(data["records"]) - 20)), "--save"])
            assert r["saved"] is True and "jaw_recorded_raw" not in r and TL.active().gripper_raw is None, r
            assert any("raw 2041 .. 2100" in n and "still not compared" in n for n in r["notes"]), r["notes"]
        # ... a file that has its jaw position keeps it; records without a reading give none
        with tool_file(thin_tool([0.020, 0.0, 0.050], gripper_raw=2043)):
            r = CLI.main(["--sim", "--from", steady, "--save"])
            assert r["saved"] is True and TL.active().gripper_raw == 2043 and "jaw_recorded_raw" not in r
        with tool_file(thin_tool([0.020, 0.0, 0.050])):
            r = CLI.main(["--sim", "--from", folder, "--save"])
            assert r["saved"] is True and TL.active().gripper_raw is None and "jaw_recorded_raw" not in r
        # --- R6. The photos were taken with the jaw closed at raw 2041 and no hold on record (a script had put the
        # squeeze on); `fd tool jaw close` has recorded the hold since. Refused as ever, and the refusal names the
        # way back, command by command, besides the new capture
        data_tool = thin_tool([0.020, 0.0, 0.050], gripper_raw=2041).to_dict()       # (what the tool file said when they were taken)
        before_hold = copy_of(data_tool)
        held = thin_tool([0.020, 0.0, 0.050], on_jaw="moving_outer", gripper_raw=2041, gripper_goal_raw=2016)
        steps = ["`fd tool jaw close --squeeze 0`", f"`fd calib-air --from {before_hold} --save`", "`fd tool jaw close`"]
        assert CLI.hold_recorded_since(data_tool, data["records"], held)
        with tool_file(held):
            before = open(TOOL_PATH, "rb").read()
            for force in ([], ["--force"]):
                r = CLI.main(["--sim", "--from", before_hold, "--save", *force])
                assert r["ok"] is False and r["stage"] == "save" and r["saved"] is False and "no jaw hold on record" in r["reason"], r
                assert "--force does not change that" in r["reason"] and open(TOOL_PATH, "rb").read() == before
                back = r["next"][r["next"].index("The way back, ONLY when"):]
                at = [back.index(x) for x in steps]
                assert at == sorted(at) and r["way_back"] is True, r["next"]
                assert "`fd calib-air --tip-only --save`" in r["next"] and "fd tool jaw close --squeeze 0" in r["say"]
            r = CLI.main(["--sim", "--from", before_hold])              # not asked to store: the same way, not a --save that is refused
            assert r["ok"] and "--save would refuse them" in r["next"] and all(x in r["next"] for x in steps), r["next"]
            # walked: the hold off the record (what `fd tool jaw close --squeeze 0` writes), the photos stored, the
            # hold on record again (`fd tool jaw close`: a close that ends where the file has the jaw keeps the tip)
            t = TL.active()
            t.gripper_goal_raw = None
            TL.save(t)
            r = CLI.main(["--sim", "--from", before_hold, "--save"])
            assert r["saved"] is True and "way_back" not in r, r
            rec = TL.active().tip_measured
            t = TL.active()
            t.gripper_goal_raw = 2016
            TL.save(t)
            assert TL.active().tip_measured == rec and TL.active().gripper_goal_raw == 2016
        # ... no way back where the jaw stood elsewhere, or where the SERVO said that it was not squeezed
        elsewhere = copy_of(thin_tool([0.020, 0.0, 0.050], gripper_raw=2182).to_dict())
        loose = copy_of(raws=[2041] * len(data["records"]))
        assert not CLI.hold_recorded_since(data_tool, [dict(rec, gripper_raw=2041, gripper_goal_raw=2041) for rec in data["records"]], held)
        assert not CLI.hold_recorded_since(dict(data_tool, gripper_raw=2182), data["records"], held)
        assert not CLI.hold_recorded_since(dict(data_tool, attached_at="2026-11-13T19:00:00"), data["records"], held)
        with tool_file(held):
            for d, word in ((elsewhere, "raw 2182"), (loose, "NOT held with a squeeze")):
                r = CLI.main(["--sim", "--from", d, "--save"])
                assert r["ok"] is False and r["stage"] == "save" and word in r["reason"], (word, r)
                assert "way_back" not in r and "--squeeze 0" not in r["next"] and "`fd calib-air --tip-only --save`" in r["next"], r["next"]
    finally:
        CLI._solve = solve
    assert len(fits) == 1


def _placed(q):
    """prepare(arm, bus) for run_command: the simulated arm stands at joint pose q with the torque on (where some
    earlier command left it)."""
    def prepare(arm, bus):
        assert TL.sim_place(arm, q)
        TL._power(arm, None)
    return prepare


def test_the_look_does_not_pass_on_paint_the_arm_hid_in_the_photo_before():
    """Round 2, R4 (the rest of finding 43). The photo "before" was taken where the arm stood unless it stood within
    12 deg of the look pose. An arm 13 .. 25 deg from it stands with its tool over the place of the look all the same:
    what the arm HIDES in that photo, the look photo reveals, and paint among it differs from the photo before and is
    bristle-coloured, right where the registration is judged. On the evening's desk a tool registered as 8 cm and
    really 17 cm along the fingers passed like that (the candidate 16 px from the registered tip's place) and 20
    poses were flown with the real tip down to the table (see the test with that photo, below).

    Here: registered 8 cm, really 13 cm along the fingers; one stroke of paint beside the tool's line where the
    registered tip will be seen; the arm 13 deg lower than the look pose, with the arm itself in the photos."""
    reg, real = SHORT_GUESS, [-0.001, 0.0, 0.1368]
    registered = thin_tool(reg)
    arm0, _bus = sim_arm(registered)
    look = AC.look_pose(arm0, registered, CAM)
    q_look, px, g = look["q"], np.array(look["tip_px"]), np.array(look["gripper_px"])
    along = (px - g) / np.linalg.norm(px - g)
    paint = px + 32.0 * np.array([-along[1], along[0]])
    desk = BASE.copy()
    _seg(desk, paint - 22.0 * along, paint + 22.0 * along, 11, 4, ORANGE)
    q_start = q_look + np.radians([0.0, 13.0, -13.0, 0.0, 0.0])
    assert np.max(np.abs(q_start - q_look)) > math.radians(CLI.LOOK_SAME_DEG)
    assert TL.tip_height(arm0, q_start, thin_tool(real)) > 0.03 and TL.tip_height(arm0, q_start, registered) > 0.05

    def shot(q):
        return draw_tool(draw_arm(desk.copy(), CAM, q), CAM, q, real)

    def shot_of(tip):               # (the look photo of a tool whose tip is at `tip`)
        return draw_tool(draw_arm(desk.copy(), CAM, q_look), CAM, q_look, tip)

    def paint_seen(img) -> bool:
        return any(np.hypot(k["centre"][0] - paint[0], k["centre"][1] - paint[1]) < 15.0 for k in AC.candidates(img, None, None))

    before, look_img = shot(q_start), shot(q_look)
    # 1) the mechanism. The arm hides the paint in the photo taken where it stands; the look photo shows it; judged
    #    against that photo (the rule until 2026-11-14) the paint is the tip, and the look passes. (any_background:
    #    the photos alone. Since round 3 a desk with paint this near the place of the look is refused as it is: 4.)
    assert paint_seen(desk) and paint_seen(look_img) and not paint_seen(before)
    with any_background():
        old = AC.look_check(look_img, CAM, q_look, reg, bg=before)
    assert old["ok"] is True and old["checked"] is True and old["nearest_px"] < AC.LOOK_PX and old["hidden_before"] is None, old
    # ... the truth (the bare desk as the photo before): the tip is 5 cm from where it is registered
    truth = AC.look_check(look_img, CAM, q_look, reg, bg=desk)
    assert truth["ok"] is False and truth["nearest_px"] > 150.0 and truth["background"]["clean"] is False, truth
    # 2) told where the arm stood for the photo before, the check lets nothing pass that lies there. (It stays in the
    #    list, for a person who looks at the photo, and "nearest_clear_px" is the nearest of the rest.)
    with any_background():
        new = AC.look_check(look_img, CAM, q_look, reg, bg=before, q_bg=q_start)
    assert new["ok"] is False and new["checked"] is True and new["hidden_before"] >= 1, new
    assert new["nearest_px"] == old["nearest_px"] and (new["nearest_clear_px"] is None or new["nearest_clear_px"] > 150.0), new
    # ... what lies elsewhere is judged as before: the same photos with the arm turned aside for the photo before
    aside = AC.look_aside(arm0, registered, CAM, q_start, q_look)
    with any_background():
        far = AC.look_check(look_img, CAM, q_look, reg, bg=desk, q_bg=aside["q"])
        fine = AC.look_check(shot_of(np.add(reg, [0.002, 0.0, 0.002])), CAM, q_look, reg, bg=desk, q_bg=aside["q"])
    assert far["ok"] is False and far["nearest_px"] == truth["nearest_px"], far
    assert fine["ok"] is True and fine["hidden_before"] == 0 and fine["nearest_clear_px"] == fine["nearest_px"] < 30.0, fine
    # the drawing (round 3, T2): the arm's links and EVERYTHING a tool of up to 30 cm can reach from the gripper, wherever
    # it points. Until then the tool was two lines (along the fingers, through the registered tip), and the body of a
    # tool that lay 6 cm beside the finger line hid paint outside the drawing
    mask = AC.silhouette(CAM, q_start, reg)
    t = K.fk(q_start)

    def drawn(points) -> bool:      # (every one of these points of the gripper frame that the camera sees is inside the drawing)
        pts = [t[:3, 3] + t[:3, :3] @ np.asarray(p, dtype=float) for p in points]
        pts = [p for p in pts if float((CAM.R @ p + CAM.tvec)[2]) > 0.05]
        uv = [(int(round(u)), int(round(v))) for u, v in CAM.project(pts)]
        return bool(pts) and all(mask[v, u] for u, v in uv if 0 <= u < 1920 and 0 <= v < 1080)

    assert drawn([[0.0, 0.0, 0.0], reg, np.asarray(reg) * 1.5, np.asarray(reg) * 3.0])
    for side in ([0.0, -0.060, 0.177], [0.0, 0.060, 0.177], [0.0, -0.060, 0.130], [0.0, 0.080, 0.140], LONG_TIP, SIDE_TIP, ACROSS_TIP):
        assert drawn([np.asarray(side) * k for k in np.linspace(0.0, 1.0, 15)]), side      # (the verifier's sideways tools)
    rng = np.random.default_rng(5)
    ball = rng.normal(size=(400, 3))
    ball *= (AC.SIL_REACH_M * rng.uniform(0.0, 1.0, (400, 1)) ** (1.0 / 3.0)) / np.linalg.norm(ball, axis=1, keepdims=True)
    assert drawn(ball) and AC.SIL_REACH_M == TL.MAX_OFFSET_M
    assert mask.shape == (1080, 1920) and mask[int(paint[1]), int(paint[0])] and float(np.mean(mask > 0)) > 0.5
    # ... and not the whole picture: with the look pose turned aside it leaves the place of the look clear
    clear = AC.silhouette(CAM, aside["q"], reg)
    assert not clear[int(round(px[1])), int(round(px[0]))] and 0.02 < float(np.mean(clear > 0)) < 0.9, float(np.mean(clear > 0))
    # 3) so a photo taken there is no photo "before", and the arm is turned aside first. Since round 3 a folded arm is
    #    none either (a tool of 30 cm could cover the place of the look from there), nor the look pose turned 60 deg;
    #    turned 90 deg and more to the palette's side it is one. Right in front of the camera it is none
    assert "hide the place of the look" in AC.look_hidden(CAM, q_start, q_look, reg)
    assert AC.look_hidden(CAM, q_look, q_look, reg) is not None
    assert AC.look_hidden(CAM, TL.rest_q(arm0, registered), q_look, reg) is not None
    for turn in (60.0, -60.0, 90.0):
        assert AC.look_hidden(CAM, q_look + np.radians([turn, 0.0, 0.0, 0.0, 0.0]), q_look, reg) is not None, turn
    for turn in (-90.0, -97.0):
        assert AC.look_hidden(CAM, q_look + np.radians([turn, 0.0, 0.0, 0.0, 0.0]), q_look, reg) is None, turn
    assert "from the camera" in AC.look_hidden(CAM, np.radians([14.0, 39.0, -66.0, 21.0, 0.0]), q_look, reg)
    # (a turn aside is asked the same question: with a camera that counts every pose as too near, none qualifies)
    keep = AC.LOOK_CAMERA_M
    AC.LOOK_CAMERA_M = 2.0
    try:
        e = raises(SafetyError, AC.look_aside, arm0, registered, CAM, q_start, q_look)
        assert "cannot be turned aside" in str(e) and "from the camera" in str(e), e
    finally:
        AC.LOOK_CAMERA_M = keep
    # (the pose aside is the LOOK POSE with the base turned, wherever the arm stands: the one pose made for a tool of
    # unknown length)
    assert np.allclose(aside["q"][1:], q_look[1:]) and aside["turn_deg"] in (-90.0, -97.0), aside
    assert AC.look_hidden(CAM, aside["q"], q_look, reg) is None
    # 4) the command from there. Over this desk (paint 32 px from the place of the look) no look is taken at all (round
    #    3, a clean background): the arm goes to the look pose turned aside, ONE photo, and it holds there ...
    with tool_file(registered):
        res, arm, bus, photo, root = run_command([], registered, real, prepare=_placed(q_start), base=desk, links=True, hand_every=0)
        lk = res["look"]
        assert res["ok"] is False and res["stage"] == "look" and lk["dirty"] is True and lk["photo"] is None, res
        assert len(photo.shots) == 1 and res["sets"] == [] and lk["background"]["nearest_px"] < 40.0, lk["background"]
        assert np.max(np.abs(photo.shots[0][1:] - q_look[1:])) < math.radians(1.5) and "turned aside" in res["arm"]["at"], res["arm"]
        assert np.allclose(q_of(bus.pos), photo.shots[0], atol=math.radians(1.0)) and all(bus.torque[j] == 1 for j in ARM)
        assert min(real_heights(arm, bus, registered, real)) >= 0.020
        # ... and by the photos alone (that rule switched off, as round 2 judged): turned aside, photo, back to the look
        # pose, photo: refused, nothing flown
        with any_background():
            res, arm, bus, photo, root = run_command([], registered, real, prepare=_placed(q_start), base=desk, links=True,
                                                     hand_every=0)
        lk = res["look"]
        assert res["ok"] is False and res["stage"] == "look" and lk["ok"] is False and lk["checked"] is True, res
        assert "turned aside" in lk["before"]["taken"] and "place of the look" in lk["before"]["taken"], lk["before"]
        assert lk["before"]["turn_deg"] in AC.LOOK_ASIDE_DEG and (lk["nearest_px"] is None or lk["nearest_px"] > 150.0), lk
        assert lk["hidden_before"] is not None      # (judged with the pose of the photo before: what lies where the arm stood cannot pass)
        assert len(photo.shots) == 2 and res["sets"] == [] and not glob.glob(os.path.join(res["dir"], "high_*"))
        turned, back = photo.shots
        assert abs(abs(math.degrees(turned[0] - q_look[0])) - abs(lk["before"]["turn_deg"])) < 1.0, np.degrees(turned)
        assert np.max(np.abs(turned[1:] - q_look[1:])) < math.radians(1.5) and np.allclose(back, q_look, atol=math.radians(1.0))
        assert min(real_heights(arm, bus, registered, real)) >= 0.020
        assert "fd tool set --tip" in res["next"] and res["arm"]["torque"] == "on"
        # --dry from such a start plans the same turn, and writes no goal
        arm2, bus2 = sim_arm(TL.active())
        _placed(q_start)(arm2, bus2)
        writes = bus2.io["writes"]
        saved = CLI._make_arm
        CLI._make_arm = lambda sim: arm2
        try:
            r = CLI.main(["--sim", "--dry"])
        finally:
            CLI._make_arm = saved
        assert r["ok"] and "aside" in r["look"]["photo_before"] and "place of the look" in r["look"]["photo_before"], r["look"]
        assert bus2.io["writes"] == writes
    # 5) a registration that is right passes from the same kind of start, on the two photos (the paint is in both). By
    #    the photos alone: this desk has a stroke 11 cm in the picture from where THAT tip is seen, and such a look is
    #    not taken since round 3 (a right registration over a blank sheet: test_a_look_never_passes_by_colour_alone, 3)
    good = thin_tool(np.add(real, [0.003, 0.002, 0.003]))
    arm, bus = sim_arm(good)
    q_good = AC.look_pose(arm, good, CAM)["q"]
    _placed(q_good + np.radians([0.0, 13.0, -13.0, 0.0, 0.0]))(arm, bus)
    out = tempfile.mkdtemp(prefix="look_", dir=_TMP)
    with any_background():
        lk = CLI._look(arm, good, CAM, camera_for(arm, real, base=desk, links=True, hand_every=0), out, "tip", lambda *m: None)
    assert lk["ok"] is True and lk["checked"] is True and lk["nearest_px"] < 40.0 and "turned aside" in lk["before"]["taken"], lk
    assert np.allclose(q_of(bus.pos), q_good, atol=math.radians(1.0))


# the starts near the look pose in which the look PASSED for the verifier of 2026-11-14, on the evening's desk
# (registered tip, real tip, the start as degrees from the look pose): (pan, lift, elbow, wrist, roll)
_ALONG_17 = ([-0.001, 0.0, 0.0868], [-0.001, 0.0, 0.1768])
_ALONG_12 = ([-0.001, 0.0, 0.0668], [-0.001, 0.0, 0.1268])
_ALONG_13 = ([-0.001, 0.0, 0.0868], [-0.001, 0.0, 0.1368])
_SECOND_MOUNT = ([-0.001, 0.0, 0.0868], LONG_TIP)
LOOKS_THAT_PASSED = ((_ALONG_17, (0, -25, 0, 25, 0)), (_ALONG_12, (0, -13, 0, 13, 0)), (_ALONG_12, (0, -13, 0, 25, 0)),
                     (_ALONG_13, (0, -13, 13, 0, 0)), (_ALONG_13, (0, 0, -13, 25, 0)), (_ALONG_13, (0, 13, -25, 25, 0)),
                     (_SECOND_MOUNT, (0, 0, 0, 20, 0)), (_SECOND_MOUNT, (0, 13, 0, 0, 0)), (_SECOND_MOUNT, (0, 0, 13, 0, 0)))
NEAR_THE_LOOK_POSE = ((13, 0, 0, 0, 0), (-13, 0, 0, 0, 0), (20, 0, 0, 0, 0), (-20, 0, 0, 0, 0), (0, 0, 0, 13, 0), (0, 0, 0, -13, 0),
                      (0, 0, 0, 20, 0), (0, 13, 0, 0, 0), (0, -13, 0, 0, 0), (0, 0, 13, 0, 0), (0, 0, -13, 0, 0), (0, 13, -13, 0, 0),
                      (0, 8, 8, 13, 0), (13, 0, 0, 13, 0))


# The starts in which the look PASSED for the verifier of round 2 (2026-11-14) with a tool whose BODY lies beside the
# finger line, registered 8 cm along the fingers, on the evening's desk: (real tip, the starts as joint angles in
# degrees). The photo before was taken where the arm stood, the drawing of the tool had two lines, and the handle hid
# the lantern's red beside them. With the first of these tools 20 poses were then flown, the real tip at -3 .. 14 mm.
SIDEWAYS_LOOKS_THAT_PASSED = (
    ([0.0, -0.060, 0.177], ((-8.0, -87.7, 52.3, 65.4), (-8.0, -87.7, 65.3, 50.4), (-8.0, -87.7, 77.3, 25.4), (-8.0, -72.7, 52.3, 50.4),
                            (-8.0, -72.7, 65.3, 25.4), (-8.0, -60.7, 52.3, 38.4), (-8.0, -47.7, 27.3, 65.4), (-8.0, -47.7, 39.3, 38.4))),
    ([0.0, -0.040, 0.177], ((-8.0, -87.7, 65.3, 38.4), (-8.0, -87.7, 77.3, 25.4), (-8.0, -60.7, 27.3, 65.4), (-8.0, -47.7, 27.3, 50.4))),
    ([0.0, 0.060, 0.177], ((8.0, -60.7, 27.3, 65.4), (3.3, -37.0, 16.9, 42.1))),
    ([0.0, -0.060, 0.130], ((-8.0, -87.7, 77.3, 38.4), (-8.0, -72.7, 65.3, 38.4), (-8.0, -60.7, 39.3, 65.4), (-8.0, -60.7, 52.3, 38.4),
                            (-8.0, -60.7, 52.3, 50.4), (-8.0, -47.7, 39.3, 50.4), (-8.0, -34.7, 27.3, 38.4), (-8.0, -34.7, 27.3, 50.4))),
    ([0.0, 0.080, 0.140], ((8.0, -47.7, 39.3, 38.4), (8.0, -34.7, 12.3, 50.4))))


def test_real_desk_the_looks_that_passed_from_near_the_look_pose_are_refused():
    """Round 2, R4, and round 3, T2, with the evening's desk as the scene (runs/20261113_paint/190853_run.jpg, the
    finished picture uncovered: a local photo) and the arm and the brush drawn into it where the camera sees them:
    - the nine starts in which the look passed for the verifier of round 1 (6 of 267 with tools longer than registered
      along the fingers, 3 of 12 with the evening's second mount) and that verifier's fourteen starts around the look
      pose for two of those tools;
    - the 24 starts in which it passed for the verifier of round 2 with five tools whose body lies 4 to 8 cm beside the
      finger line (26 of 1106 looks; the reduced grid of T2).
    0 looks pass. (a) Judged by the photos alone, the clean-background rule switched off: every look is refused after the
    arm went to the look pose turned aside (the drawing covers everything a tool of 30 cm can reach), or before
    anything moves. (b) With the rule, as the command runs (every fourth start): refused after ONE photo, the place of
    the look lies on the picture's red, and no look photo is taken.
    The REAL tip of a tool along the fingers never comes more than 6 mm lower than it stood at the start or stands at
    the look pose (the ways are made for such a tool: aircal.unknown_tool_dip allows 5 mm); that of the sideways tools
    stays above the table (how far: printed)."""
    desk_photo = os.path.join(tazuna_paths.ROOT, "runs", "20261113_paint", "190853_run.jpg")
    if not os.path.isfile(desk_photo):
        raise Skip("runs/20261113_paint/190853_run.jpg is not on this computer")
    desk = cv2.imread(desk_photo)
    starts = [(reg, real, "d", dq) for (reg, real), dq in LOOKS_THAT_PASSED]
    starts += [(reg, real, "d", dq) for (reg, real) in (_ALONG_17, _SECOND_MOUNT) for dq in NEAR_THE_LOOK_POSE]
    n_along = len(starts)
    starts += [(SHORT_GUESS, real, "q", q) for real, qs in SIDEWAYS_LOOKS_THAT_PASSED for q in qs]
    tried = turned = unmoved = dirty = side = 0
    side_low = 1.0

    def placed(registered, q, real):
        arm, bus = sim_arm(registered)
        _placed(q)(arm, bus)
        bus.trace.clear()
        return arm, bus, camera_for(arm, real, base=desk, links=True, hand_every=0)

    for k, (reg, real, how, v) in enumerate(starts):
        registered, real_tool = thin_tool(reg), thin_tool(real)
        arm0, _bus = sim_arm(registered)
        q = AC.look_pose(arm0, registered, CAM)["q"] + np.radians(np.asarray(v, dtype=float)) if how == "d" else np.radians([*v, 0.0])
        passed_then = k >= n_along or ((reg, real), v) in LOOKS_THAT_PASSED
        try:        # (where a real arm can stand with this tool: its real tip 2 cm up)
            TL.check_pose(arm0, real_tool, q, min_height=0.02, what="start")
            TL.check_pose(arm0, registered, q, min_height=0.0, what="start")
        except SafetyError:
            assert not passed_then, (reg, real, v)
            continue
        arm, bus, photo = placed(registered, q, real)
        h_start = TL.tip_height(arm, q_of(bus.pos), real_tool)      # (as it holds: switched on, a simulated arm gives way by a degree)
        tried += 1
        side += k >= n_along
        try:
            with any_background():
                lk = CLI._look(arm, registered, CAM, photo, tempfile.mkdtemp(prefix="look_", dir=_TMP), "tip", lambda *m: None)
        except SafetyError as e:
            # no way to the look pose turned aside can be planned from there: refused before anything moves
            assert "cannot be turned aside" in str(e) and not photo.shots, (reg, real, v, e)
            assert all(np.max(np.abs(q_of(p) - q)) < math.radians(1.5) for p in bus.trace), (v, "the arm moved")
            unmoved += 1
            continue
        assert lk["ok"] is False and lk["checked"] is True and len(photo.shots) == 2, (reg, real, v, lk)
        assert "turned aside" in lk["before"]["taken"] and lk["before"]["turn_deg"] in AC.LOOK_ASIDE_DEG, lk["before"]
        turned += 1
        low = min(real_heights(arm, bus, registered, real))
        if k < n_along:
            assert low >= min(h_start, TL.tip_height(arm, q_of(bus.pos), real_tool)) - 0.006, (reg, real, v, low, h_start)
        else:
            side_low = min(side_low, low)
        if k % 4 == 0:      # (b) as the command runs: no clean background, no look
            arm, bus, photo = placed(registered, q, real)
            lk = CLI._look(arm, registered, CAM, photo, tempfile.mkdtemp(prefix="look_", dir=_TMP), "tip", lambda *m: None)
            assert lk["ok"] is False and lk["dirty"] is True and lk["photo"] is None and len(photo.shots) == 1, (reg, real, v, lk)
            assert lk["background"]["paint_px"] > 500 and all(bus.torque[j] == 1 for j in ARM), lk["background"]
            dirty += 1
    print(f"     note: {tried} starts ({side} with the sideways tools): {turned} looks refused by the photos after a turn aside, "
          f"{unmoved} before anything moved; {dirty} of them again with the clean-background rule: refused after one photo; "
          f"the REAL tip of the sideways tools at its lowest {side_low * 1000:+.0f} mm (0 = the table)")
    # (all 9 + 24 starts in which a look once passed could be started)
    assert side == 24 and tried >= 49 and turned + unmoved == tried and turned >= 40 and dirty >= 10, (tried, side, turned, unmoved, dirty)
    assert side_low > 0.0, side_low


# ----------------------------------------------------------------------------- the base turns only with the tool lifted
RIG_DISH = (0.346, 0.192)                # the water dish of the rig's painting set-up (2026-11-13), 7.5 cm round it
RIG_FOLD = (0.04, -100.0)                # how the rig's arm lies folded: the registered tip 4 cm up, the base at -100 deg
PHOTO_POSE_DEG = (-89.2, -61.7, 36.2, 85.6, 0.0)       # where the painting commands leave the arm: the gripper 60 deg down


def _tool_points(reach: float = TL.MAX_OFFSET_M, step: float = 0.05) -> np.ndarray:
    """Points of whatever may be taped to the gripper (its axes): in 26 directions, every `step` out to `reach`."""
    dirs = [np.array(d, dtype=float) for d in itertools.product((-1, 0, 1), repeat=3) if any(d)]
    return np.array([d / np.linalg.norm(d) * r for d in dirs for r in np.arange(step, reach + 1e-9, step)])


def _point_heights(q, pts) -> np.ndarray:
    """Height above the table of gripper-frame points at joint pose q."""
    t = K.fk(np.asarray(q, dtype=float))
    return (t[:3, 3] + pts @ t[:3, :3].T)[:, 2] - TZ


def _samples(wps, step_deg: float = 1.0) -> list:
    out = [np.asarray(wps[0], dtype=float)]
    for a, b in zip(wps[:-1], wps[1:]):
        n = max(1, int(math.ceil(math.degrees(float(np.max(np.abs(np.asarray(b) - np.asarray(a))))) / step_deg)))
        out += [np.asarray(a) + (np.asarray(b) - np.asarray(a)) * (k / n) for k in range(1, n + 1)]
    return out


def test_the_base_turns_only_with_the_tool_lifted():
    """U2 of the last verification (2026-11-14). For the photo "before", the first look turned the look pose to the
    palette's side and back with the fingertip 12 cm above the table: before the look had judged anything, a tool
    longer than registered passed 12 .. 47 mm above the table within 7.5 cm of the rig's water dish (26 mm for a brush
    9 cm longer than registered, 24 mm for the mistake of 2026-11-13, 12 mm for 20 cm). Now the base turns only after
    the gripper was lifted 10 cm straight up without turning (aircal.straight_lift, look_way, fly_look_way): whatever
    is taped on, of any length and in any direction, is then 10 cm above where it hung, and the arm comes straight down
    onto the pose. Where the arm cannot do that, the look is refused before anything moves."""
    tool = thin_tool(SHORT_GUESS)
    arm, _bus = sim_arm(tool)
    q_look = AC.look_pose(arm, tool, CAM)["q"]
    aside = q_look.copy()
    aside[0] -= math.radians(97.0)
    pts = _tool_points()
    far = pts[np.linalg.norm(pts, axis=1) > TL.MAX_OFFSET_M - 1e-6]
    # 1. the lift: nothing turns, and every point of anything within 30 cm of the gripper goes up by the same 10 cm
    up = AC.straight_lift(arm, tool, q_look)
    t0 = K.fk(q_look)
    for w in up:
        t = K.fk(w)
        assert abs(w[0] - q_look[0]) < math.radians(0.05) and abs(w[4] - q_look[4]) < 1e-6, np.degrees(w)
        assert np.max(np.abs(t[:3, :3] - t0[:3, :3])) < 2e-3 and np.linalg.norm(t[:2, 3] - t0[:2, 3]) < 1e-3, np.degrees(w)
    gain = _point_heights(up[-1], pts) - _point_heights(q_look, pts)
    assert np.all(np.abs(gain - AC.LOOK_LIFT_M) < 0.004), (gain.min(), gain.max())
    assert abs(AC.tool_rise(q_look, up[-1]) - AC.LOOK_LIFT_M) < 0.003 and abs(AC.tool_rise(q_look, aside)) < 1e-5
    # (tool_rise is what the point that gains LEAST gains: a gripper that tips on the way gives part of the lift away)
    tipped = up[-1].copy()
    tipped[3] += math.radians(10.0)
    least = float((_point_heights(tipped, far) - _point_heights(q_look, far)).min())
    assert AC.tool_rise(q_look, tipped) <= least + 1e-6 and AC.tool_rise(q_look, tipped) < AC.LOOK_LIFT_M - 0.04, least
    # a gripper that points steeply down has its wrist at the limit: it cannot be lifted straight up, and says so
    photo_pose = np.radians(PHOTO_POSE_DEG)
    assert "cannot lift" in str(raises(SafetyError, AC.straight_lift, arm, tool, photo_pose))
    # 2. the way, as it is planned: lift, turn, (across,) straight down, and no other leg turns the base. While the
    #    base turns, every point of a tool that could be on the arm where it stood (not inside the table) is 10 cm
    #    above the table, whatever its length and direction
    fold, rig_fold = TL.rest_q(arm, tool), TL.rest_q(arm, tool, *RIG_FOLD)
    for name, q_from, q_to, kinds in (("look pose -> aside", q_look, aside, ["lift", "turn", "lower"]),
                                      ("aside -> look pose", aside, q_look, ["lift", "turn", "lower"]),
                                      ("fold -> aside", fold, aside, ["lift", "turn", "plane", "lower"]),
                                      ("the rig's fold -> aside", rig_fold, aside, ["lift", "turn", "plane", "lower"]),
                                      ("photo pose -> aside", photo_pose, aside, ["plane", "lift", "turn", "lower"])):
        way = AC.look_way(arm, tool, q_from, q_to)
        assert [leg["kind"] for leg in way["legs"]] == kinds and way["turn"] and way["lift_m"] == AC.LOOK_LIFT_M, (name, way["via"])
        assert way["turn_rise_m"] >= AC.LOOK_LIFT_M - 0.005 and np.allclose(way["legs"][-1]["path"][-1], q_to), (name, way)
        assert AC.unknown_tool_dip(arm, tool, AC.plan_waypoints(way)) is None, name
        ground = None
        for leg in way["legs"]:
            path = leg["path"]
            if leg["kind"] == "lift":
                ground = path[0]
            if leg["kind"] != "turn":
                assert max(abs(w[0] - path[0][0]) for w in path) < math.radians(0.05), (name, leg["kind"])
                continue
            could_be = pts[_point_heights(ground, pts) >= 0.0]
            for q in _samples(path):
                assert AC.tool_rise(ground, q) >= AC.LOOK_LIFT_M - 0.005, (name, AC.tool_rise(ground, q))
                assert _point_heights(q, could_be).min() >= AC.LOOK_LIFT_M - 0.005, (name, _point_heights(q, could_be).min())
        down = way["legs"][-1]["path"]                      # ... and it comes down along a vertical line, not turning
        t1 = K.fk(q_to)
        assert all(np.max(np.abs(K.fk(w)[:3, :3] - t1[:3, :3])) < 2e-3 and np.linalg.norm(K.fk(w)[:2, 3] - t1[:2, 3]) < 1e-3 for w in down), name
    # two poses on one base angle need no turn and no lift
    same = AC.look_way(arm, tool, np.radians([0.0, -60.0, 40.0, 60.0, 0.0]), q_look)
    assert not same["turn"] and same["lift_m"] is None and {leg["kind"] for leg in same["legs"]} <= {"plane", "raise"}, same["via"]
    # 3. the verifier's scenario, through the look itself on the simulated arm, the REAL tip traced: tools longer than
    #    registered, from the look pose (where a refused look leaves the arm) and from folds
    cases = (("the rig's brush, 9 cm longer", THIN_TIP, [0.016, -0.0011, 0.146], "look"),
             ("8 cm registered, 17.7 cm along", SHORT_GUESS, LONGER_TIP, "look"),
             ("8 cm registered, 20 cm along", SHORT_GUESS, [-0.001, 0.0, 0.200], "look"),
             ("8 cm registered, (0, -60, 177)", SHORT_GUESS, [0.0, -0.060, 0.1768], "look"),
             ("8 cm registered, 17.7 cm along, folded", SHORT_GUESS, LONGER_TIP, "fold"),
             ("the rig's brush, 5 cm longer, the rig's fold", THIN_TIP, [0.016, -0.0011, 0.106], "rig"))
    lows = []
    for name, reg, real, start in cases:
        registered = thin_tool(reg)
        with tool_file(registered):
            arm, bus = sim_arm(TL.active())
            if start == "look":
                held_at_look_pose(real)(arm, bus)
            elif start == "fold":
                at_rest_of(real)(arm, bus)
            else:
                assert TL.sim_place(arm, TL.rest_q(arm, registered, *RIG_FOLD))
            bus.trace.clear()
            first = []
            real_fly = AC.fly_look_way

            def fly(*a, _real=real_fly, **k):
                first.append(len(bus.trace))                # (how many samples the arm had made when a way began)
                return _real(*a, **k)

            AC.fly_look_way = fly
            shots = camera_for(arm, real, hand_every=0)
            try:
                lk = CLI._look(arm, arm.tool, CAM, shots, tempfile.mkdtemp(prefix="look_", dir=_TMP), "tip", lambda *m: None)
            finally:
                AC.fly_look_way = real_fly
            # the look itself is what it was: a photo with the look pose turned aside, one at the look pose, refused
            assert lk["ok"] is False and lk["checked"] is True and len(shots.shots) == 2, (name, lk)
            assert lk["before"]["turn_deg"] in AC.LOOK_ASIDE_DEG and "turned aside" in lk["before"]["taken"], (name, lk["before"])
            assert np.max(np.abs(shots.shots[0][1:] - q_look[1:])) < math.radians(1.5), (name, np.degrees(shots.shots[0]))
            assert np.allclose(q_of(bus.pos), AC.look_pose(arm, arm.tool, CAM)["q"], atol=math.radians(1.0)), name
            assert "lifted 10 cm straight up" in lk["before"]["way"] and "lifted 10 cm straight up" in lk["way"]["via"], (name, lk)
            assert lk["before"]["rise_at_turn_mm"] >= 90.0 and lk["way"]["rise_at_turn_mm"] >= 90.0, (name, lk["before"], lk["way"])
            qs = [q_of(p) for p in bus.trace]
            tip = np.array([K.fk(q, K.tool_offset(*real))[:3, 3] for q in qs])
            h = tip[:, 2] - TZ
            pan = np.degrees([q[0] for q in qs])
            turning = np.zeros(len(qs), bool)
            turning[1:] = np.abs(np.diff(pan)) > 0.02
            over_dish = np.linalg.norm(tip[:, :2] - RIG_DISH, axis=1) < 0.075
            assert turning.sum() > 200 and over_dish.sum() > 20 and np.all(turning[over_dish]), (name, turning.sum(), over_dish.sum())
            # while the base turns, and over the dish, the REAL tip is 8 cm and more above the table (it was 12 .. 47 mm)
            assert h[turning].min() >= AC.LOOK_CLEAR_M and h[over_dish].min() >= AC.LOOK_CLEAR_M, (name, h[turning].min(), h[over_dish].min())
            # ... and so is every point of a tool of up to 30 cm that could have been on the arm where it started
            could_be = pts[_point_heights(qs[first[0]], pts) >= 0.0]
            worst = min(float(_point_heights(q, could_be).min()) for q in np.array(qs)[turning][::10])
            assert worst >= AC.LOOK_CLEAR_M, (name, worst)
            # the look's own ways never bring the real tip lower than it was when they began (to 3 mm)
            assert h[first[0]:].min() >= h[first[0]] - 0.003, (name, h[first[0]], h[first[0]:].min())
            lows.append((name, round(float(h[turning].min()) * 1000), round(float(h[over_dish].min()) * 1000), round(worst * 1000)))
    print("     note: the REAL tip while the base turns / over the rig's dish / the lowest point of any tool of up to 30 cm "
          "while it turns, in mm above the table: " + "; ".join(f"{n}: {a} / {b} / {c}" for n, a, b, c in lows))
    # 4. what is asked of the MEASURED arm: a lift that ends short does not let the base turn. The arm holds where it is
    with tool_file(thin_tool(SHORT_GUESS)):
        arm, bus = sim_arm(TL.active())
        held_at_look_pose()(arm, bus)
        q0 = arm.read_q()
        way = AC.look_way(arm, arm.tool, q0, aside)
        real_lift = AC.straight_lift
        AC.straight_lift = lambda a, t, q, rise_m=AC.LOOK_LIFT_M: real_lift(a, t, q, 0.03)
        try:
            e = raises(MotionAborted, AC.fly_look_way, arm, aside, way=way)
        finally:
            AC.straight_lift = real_lift
        assert "the base was not turned" in str(e) and "holds" in str(e), str(e)
        q1 = q_of(bus.pos)
        assert abs(math.degrees(q1[0] - q0[0])) < 0.5 and 0.02 < AC.tool_rise(q0, q1) < 0.04 and all(bus.torque[j] == 1 for j in ARM)
        # ... and a way that is flown says what it measured
        flown = AC.fly_look_way(arm, aside)
        assert flown["turned"] and flown["rise_at_turn_m"] >= AC.LOOK_LIFT_M - 0.01 and flown["legs"] >= 3, flown
        assert np.allclose(q_of(bus.pos), aside, atol=math.radians(1.0)) and flown["min_tip_height_m"] >= TL.MIN_FREE_HEIGHT
    # 5. no such way: refused before anything moves. (a) A tool whose own joint limits keep the wrist from following the
    #    lift at the look pose: no turn can be flown back from, whatever the start, and no fold is named
    stiff = thin_tool(SHORT_GUESS, joint_limits_deg={"wrist_flex": [20.0, 45.0], "wrist_roll": [-5.0, 5.0]})
    with tool_file(stiff):
        res, arm, bus, shots, root = run_command([], stiff, SHORT_GUESS)
        assert res["ok"] is False and res["stage"] == "look" and "cannot be turned aside" in res["reason"], res
        assert "lifted" in res["reason"] and "no way back to the look pose" in res["reason"], res["reason"]
        assert not shots.shots and not bus.log and not any(bus.torque[j] for j in ARM), "the arm was powered or moved"
        assert "folding the arm first does not help" in res["next"] and "`fd tool rest" not in res["next"], res["next"]
        assert "アームは動かしていません" in res["say"] or "動かす前" in res["say"], res["say"]
    #    (b) an arm that stands where it can neither lift nor reach the look pose on its base angle without dipping
    #    for a tool of unknown length (the first pose of the high set): there the START is what is in the way
    thin = thin_tool()
    arm, _bus = sim_arm(thin)
    held = poses("tip")[0]
    e = raises(SafetyError, AC.look_aside, arm, thin, CAM, held, AC.look_pose(arm, thin, CAM)["q"])
    assert "no safe way from where it stands" in str(e) and "cannot be turned aside" not in str(e), str(e)


# ----------------------------------------------------------------------------- the day's photos
def _real(folder: str, *opts) -> dict:
    """The command on one of the day's capture folders (read only). The scratch tool file holds the tool those
    photos were taken with, so that --save meets the gates and not a missing tool."""
    path = os.path.join(RUNS, folder)
    files = sorted(glob.glob(os.path.join(path, "*records*.json")))
    if not files or not glob.glob(os.path.join(path, "*.jpg")):
        raise Skip(f"runs/20261113_110358/{folder} is not on this computer")
    with open(files[0]) as fh:
        taken_with = TL.Tool.from_dict(json.load(fh)["tool"])
    with tool_file(taken_with):
        before = open(TOOL_PATH, "rb").read()
        r = CLI.main(["--sim", "--from", path, *opts])
        assert open(TOOL_PATH, "rb").read() == before or r.get("saved") is True
    json.dumps(r, default=fd._jsonable, allow_nan=False)
    return r


def test_real_morning_full_solve_55_of_59():
    r = _real("aircal", "--sets", "air,low")
    assert r["ok"] and r["mode"] == "full" and r["detector"] == "head", r
    assert (r["poses"], r["with_candidate"], r["used"]) == (61, 59, 55) and abs(r["rms_px"] - 8.89) < 0.03, r
    assert np.allclose(r["tip_mm"], [8.5, 0.7, 61.1], atol=0.11) and np.allclose(r["camera"]["center"], [0.5824, 0.0801, 0.2788], atol=3e-4), r
    assert np.allclose(r["tip_sigma_mm"], [0.96, 1.47, 1.60], atol=0.05) and r["camera"]["moved_mm"] < 0.5 and r["gates"]["passed"]
    why = {u["photo"]: u["why"] for u in r["unused"]}
    assert sorted(why) == ["air_21.jpg", "air_22.jpg", "low_04.jpg", "low_25.jpg", "low_26.jpg", "low_27.jpg"], why
    assert why["air_21.jpg"] == why["air_22.jpg"] == "tip at the edge of the picture"         # the head was cut off at the top
    assert all(why[n].startswith("no candidate near") for n in ("low_04.jpg", "low_25.jpg", "low_26.jpg", "low_27.jpg"))


def test_real_evening_tip_only_thin3_and_thin4():
    if os.environ.get("JH_QUICK"):
        raise Skip("JH_QUICK set")
    r = _real("aircal_thin3", "--tip-only")
    assert r["ok"] and (r["photos"], r["used"]) == (33, 33) and np.allclose(r["tip_mm"], [-5.1, -2.5, 60.4], atol=0.5), r
    assert abs(r["rms_px"] - 7.7) < 0.1 and max(r["tip_sigma_mm"]) < 0.4 and r["gates"]["passed"] and "warning" not in r
    assert r["camera_model"]["check"]["moved_mm"] < 5.0
    r = _real("aircal_thin4")                                       # tip-only is the default with a camera model
    assert r["ok"] and r["mode"] == "tip-only" and (r["photos"], r["used"]) == (36, 34), r
    assert np.allclose(r["tip_mm"], [16.0, -1.1, 55.8], atol=0.5) and abs(r["rms_px"] - 6.6) < 0.1 and r["gates"]["passed"]
    assert abs(r["from_registered_mm"] - 8.3) < 0.3 and "warning" not in r      # the jaw model's estimate was 8.3 mm off
    assert sorted(u["photo"] for u in r["unused"]) == ["thinA_10.jpg", "thinA_12.jpg"]


def test_real_evening_first_and_second_mount():
    if os.environ.get("JH_QUICK"):
        raise Skip("JH_QUICK set")
    r = _real("aircal_thin")
    assert r["ok"] and np.allclose(r["tip_mm"], [-90.2, -9.7, 61.0], atol=0.6) and r["used"] >= 20 and r["photos"] == 22, r
    assert r["gates"]["passed"] and "warning" not in r
    r = _real("aircal_thin2", "--save")
    assert np.allclose(r["tip_mm"], [-121.7, -1.6, 114.1], atol=0.5) and (r["used"], r["photos"]) == (8, 20), r
    assert "only 8 of the 20 photos" in r["warning"] and "outside the picture" in r["warning"] and "124 mm" in r["warning"]
    assert sum(1 for u in r["unused"] if u["why"] == "tip outside the picture") == 11
    assert r["ok"] is False and r["stage"] == "save" and r["saved"] is False and not r["gates"]["passed"]
    # (with "=": fd's parser takes `--tip -0.1217,...` for an option without a value; the hint printed that spelling)
    assert "`fd tool set --tip=-0.1217,-0.0016,0.1141`" in r["next"]


def test_real_camera_that_had_moved_is_noticed():
    """The morning's photos against the camera model of 2026-11-11, from before the camera was moved (52 mm)."""
    old = os.path.join(tazuna_paths.ROOT, "backups", "camera_model.before_aircal_1113.json")
    if not os.path.isfile(old) or os.environ.get("JH_QUICK"):
        raise Skip("the camera model of 2026-11-11 is not on this computer" if not os.environ.get("JH_QUICK") else "JH_QUICK set")
    with model_file(json.load(open(old))):
        r = _real("aircal", "--tip-only", "--save")
    # (the figures of that day: 12.7 px against a limit of 11.6 px, and the cross-check puts the camera 52 mm away)
    assert r["saved"] is False and r["ok"] is False and r["stage"] == "save", r
    assert "camera seems to have moved" in r["warning"] and "camera seems to have moved" in r["reason"], r
    assert "`fd calib-air --sets air,low`" in r["next"] and 45.0 < r["camera_model"]["check"]["moved_mm"] < 60.0, r["camera_model"]


def test_real_photos_are_stored_only_into_the_jaw_state_they_were_taken_in():
    """The review's cases with the evening's own folders (local photos). aircal_thin3 was taken with the jaw at raw
    2182, aircal_thin with the jaw open at raw 2576; the tool file has the same name since, and the jaw closed at
    raw 2041: stored there, their tips are 2.2 and 10.7 cm from the real one. aircal_thin4 was taken with the jaw
    closed (raw 2041): it belongs to the rig's tool file as it stood that evening, and is stored into it."""
    if os.environ.get("JH_QUICK"):
        raise Skip("JH_QUICK set")
    thin3, thin4 = os.path.join(RUNS, "aircal_thin3"), os.path.join(RUNS, "aircal_thin4")
    if not all(glob.glob(os.path.join(d, "*.jpg")) and glob.glob(os.path.join(d, "*records*.json")) for d in (thin3, thin4)):
        raise Skip("runs/20261113_110358/aircal_thin3 and aircal_thin4 are not on this computer")
    fits = {}
    solve = CLI._solve

    def solve_once(mode, images, records, cam, guess):
        key = (mode, os.path.dirname(records[0]["image"]), len(records))
        if key not in fits:
            fits[key] = solve(mode, images, records, cam, guess)
        return copy.deepcopy(fits[key])

    # the rig's tool file as it stood at the end of that evening (state_backup/tool_active.json): the jaw closed, the
    # squeeze put on by a script and not on record
    rig = TL.Tool(name="brush_thin", tip_offset=[0.016, -0.0011, 0.0558], mount="other", pitch_deg=65.0, gripper_raw=2041,
                  attached_at="2026-11-13T17:17:45", joint_limits_deg={"wrist_flex": [20.0, 97.0], "wrist_roll": [-5.0, 5.0]})
    CLI._solve = solve_once
    try:
        # thin3 (16:46 .. 16:49, jaw at raw 2182) into a tool file of the same name and attach time with the jaw closed
        closed = TL.Tool.from_dict(dict(rig.to_dict(), attached_at="2026-11-13T16:46:35", on_jaw="moving_outer", gripper_goal_raw=2016,
                                        tip_offset=[0.0234, -0.0025, 0.0523]))
        with tool_file(closed):
            before = open(TOOL_PATH, "rb").read()
            for force in ([], ["--force"]):
                r = CLI.main(["--sim", "--from", thin3, "--save", *force])
                assert r["ok"] is False and r["stage"] == "save" and "raw 2182" in r["reason"] and "raw 2041" in r["reason"], r
                assert r["gates"]["passed"] is True and np.allclose(r["tip_mm"], [-5.1, -2.5, 60.4], atol=0.5)
                assert open(TOOL_PATH, "rb").read() == before
        # thin4 (17:18 .. 17:21, jaw closed at raw 2041) into the rig's file: stored, the tip that was typed in by hand
        # that evening is confirmed to the tenth of a millimetre, and the record says when the photos were taken
        with tool_file(rig), paint_setup():
            r = CLI.main(["--sim", "--from", thin4, "--save"])
            assert r["ok"] and r["saved"] is True and r["tip_changed_mm"] < 0.3 and (r["used"], r["photos"]) == (34, 36), r
            # (a painting set-up is on record, as on the rig: the next step is `fd paint status`, never the pen calibration)
            assert r["next"].startswith("`fd paint status`") and "`fd paint contact --adopt`" in r["next"], r["next"]
            assert "fd tool calibrate" not in r["next"] + r["say"] and "fd tool status" not in r["next"]
            rec = TL.active().tip_measured
            # (these records carry a time of day only: when the photos were taken is what their files say, 17:18 to
            # 17:21 of that evening on this computer)
            taken = sorted(time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(p)))
                           for p in glob.glob(os.path.join(thin4, "*.jpg")))
            assert rec["at"] == taken[-1] and epoch(rec["saved_at"]) >= T_START - 1.0 > epoch(rec["at"]) and rec["photos"] == thin4, rec
            assert r["taken"]["first"] == taken[0] and r["taken"]["last"] == taken[-1] and r["taken"]["jaw"] is None
            assert r["taken"]["tool"] == {"name": "brush_thin", "attached_at": "2026-11-13T17:17:45", "gripper_raw": 2041,
                                          "on_jaw": None, "gripper_goal_raw": None}
            # afterwards the jaw is recorded as taped to the moving jaw and held (`fd tool jaw mount moving_outer`,
            # `fd tool jaw close`: a close that ends where the file has the jaw keeps the tip and its record)
            held = TL.active()
            held.on_jaw, held.gripper_goal_raw = "moving_outer", 2016
            TL.save(held)
            assert TL.active().tip_measured == rec
            # ... but in THAT order: once the hold is on record, the old photos (no hold on record) no longer belong
            # to this file, and are refused
            r = CLI.main(["--sim", "--from", thin4, "--save", "--force"])
            assert r["ok"] is False and r["stage"] == "save" and "no jaw hold on record" in r["reason"], r
            # ... and the refusal names the way back (round 2, R6), in the order that works
            back = r["next"][r["next"].index("The way back, ONLY when"):]
            at = [back.index(x) for x in ("`fd tool jaw close --squeeze 0`", f"`fd calib-air --from {thin4} --save`", "`fd tool jaw close`")]
            assert at == sorted(at) and r["way_back"] is True, r["next"]
    finally:
        CLI._solve = solve


# ----------------------------------------------------------------------------- the words of every hint
def _texts(res) -> list:
    """Every text of a result that aircal_cli composes and in which it may print a command: next, the notes and
    warnings, and what its parts say (the arm's `next`, why a look refused, the note of the planned fold). Not
    `reason` and `end.not_folded`: they quote the refusals of other modules word for word (tooling, paintskill),
    whose own suites check those; and not `say`, which has the commands in Japanese sentences, without backticks."""
    out = [res.get("next"), res.get("warning")]
    if res.get("stage") == "options":       # (a refusal of the options is this module's own text from end to end)
        out.append(res.get("reason"))
    for part in (res.get("arm"), res.get("look")):
        if isinstance(part, dict):
            out += [part.get("next"), part.get("why")]
    if isinstance(res.get("end"), dict):
        out.append(res["end"].get("note"))
    out += list(res.get("notes") or []) + list(res.get("warnings") or []) + list(res.get("not_this_tool") or [])
    return [t for t in out if isinstance(t, str)]


_PLACEHOLDER = re.compile(r"[A-Z][A-Z_:,]*|x,y,z")       # DIR, P, N, LEN, PLAN, LO:HI, X,Y,Z ...


def _refused_by_its_options(sim: bool, words) -> str | None:
    """Does `fd [--sim] calib-air WORDS` get past the command's own checks of its options? Why not, or None. Neither
    the arm nor the photos of a folder are touched: what would open them is replaced for the call."""
    saved = (CLI._on_arm, CLI._from_dir)
    CLI._on_arm = CLI._from_dir = lambda *a, **k: {"ok": True, "stage": "reached"}
    try:
        res = _cli_main((["--sim"] if sim else []) + list(words))
    finally:
        CLI._on_arm, CLI._from_dir = saved
    return None if res.get("stage") == "reached" else f"refused as typed ({res.get('stage')}): {str(res.get('reason'))[:100]}"


def _printed_commands(path: str) -> list:
    """[(line, command)]: every `fd ...` between backticks that the texts of a module spell out, i.e. its string
    literals outside the docstrings, joined as the source joins them (+, f-strings). A command with a part that the
    program fills in is not among them: those are judged where a result prints them."""
    import ast
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    skip = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                skip.add(id(first.value))

    def flat(node) -> str:
        skip.add(id(node))
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.JoinedStr):
            return "".join(flat(v) for v in node.values)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            return flat(node.left) + flat(node.right)
        skip.discard(id(node))
        return "\x00"                # (a value the program fills in; what is inside it is met by the walk)

    out = []
    for node in ast.walk(tree):     # (breadth first: a chain of + is met before its parts)
        if id(node) in skip or not isinstance(node, (ast.Constant, ast.JoinedStr, ast.BinOp)):
            continue
        if isinstance(node, ast.BinOp) and not isinstance(node.op, ast.Add):
            continue
        if isinstance(node, ast.Constant) and not isinstance(node.value, str):
            continue
        out += [(node.lineno, c) for c in re.findall(r"`(fd [^`]+)`", flat(node)) if "\x00" not in c]
    return out


def _accepted(cmd: str):
    """Is `fd ...` a command the parsers take AS IT IS PRINTED? Returns why not (None: it is). Nothing is run: fd's
    own parser reads the words up to the sub-command, the sub-command's parser the rest, and for `calib-air` the
    command's own checks of its options too (_refused_by_its_options: `fd --sim calib-air` parses, and is refused as
    typed, the simulated arm has no camera). No word is replaced: until
    2026-11-14 this check filled in DIR, P, N, LEN and x,y,z before it parsed, and so passed `fd tool rest --pan P
    --keep-torque` and `fd tool attach --name N --along-fingers LEN`, which fd refuses when they are typed."""
    if "|" in cmd or "..." in cmd:
        return "a bar or dots inside the backticks: not one command"
    words = shlex.split(cmd)[1:]
    for w in words:
        if any(_PLACEHOLDER.fullmatch(part) for part in w.lstrip("-").split("=")):
            return f"a placeholder ({w}): fd refuses it as typed"
    argv, own, rest = fd._split_own_words(words)
    sink = io.StringIO()
    try:
        with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
            fd.build_parser().parse_args(argv)
    except SystemExit as e:
        return f"fd's parser: {sink.getvalue().strip().splitlines()[-1] if sink.getvalue().strip() else e.code}"
    rest = rest[1:] if rest[:1] == ["--"] else rest
    try:
        if own == "calib_air_args":
            if "--sim" in rest:
                return "--sim after the command"
            if rest[:1] not in (["--help"], ["-h"]):
                CLI.build_parser().parse_args(rest)
                return _refused_by_its_options("--sim" in argv, rest)
        elif own == "paint_args":
            import paintcli
            paintcli.build_parser().parse_args(rest)
        elif own == "jaw_args":
            import tooljaw
            if rest[:1] not in (["help"], ["--help"]):
                tooljaw.build_parser().parse_args(rest)
    except SystemExit as e:
        return None if e.code == 0 else f"exit code {e.code}"
    except Exception as e:      # (each module's own "cannot read these words")
        return f"{type(e).__name__}: {e}"
    return None


def test_zy_every_command_a_next_names_is_one_the_parsers_accept():
    """Finding 5. Every `next` of the command spelled `fd calib-air --dry`, `fd calib-air --tip-only`,
    `fd calib-air --from DIR --save`, and fd answered each of them with "unrecognized arguments", exit code 2 and
    no result: only `fd calib-air -- --dry` worked, and the tests tried that spelling when the plain one failed.
    Here every command that any result of this suite named (in `next`, in the arm's `next`, in its notes and
    warnings) is handed to fd's parser and to the parser of the sub-command it belongs to, exactly as it is printed
    (round 2, R9: one complete command per pair of backticks, no bar, no dots, no placeholder)."""
    # the results this test needs when it runs alone; with the whole suite before it there are some hundred more
    folder = capture_folder("few")
    with tool_file(thin_tool([0.020, 0.0, 0.050])):
        for argv in (["--sim", "--dry"], ["--sim"], ["--nonsense"], ["--force"], ["--dry", "--sim"], ["--dry", "--save"],
                     ["--sim", "--dry", "--sets", "low"], ["--sim", "--from", folder], ["--sim", "--from", folder, "--save", "--force"],
                     ["--sim", "--from", os.path.join(_TMP, "nowhere")]):
            CLI.main(argv)
        hwlock.lock("test_aircal")
        try:
            CLI.main([])
        finally:
            hwlock.unlock()
        for argv in (["--no-look", "--sim"], ["--dry", "--end", "stay", "--sim"]):      # --sim after the command
            assert CLI.main(argv)["stage"] == "options"
        # (a plan that cannot fold at the end: its hint is a plan too)
        with paint_setup(dict(PAINT_SETUP, canvas={"origin_xy": [-0.3, -0.3], "size_mm": [600, 600], "pitch_deg": 65.0})):
            r = CLI.main(["--sim", "--dry"])
            assert r["stage"] == "rest" and "`fd --sim calib-air --dry --end stay`" in r["next"], r["next"]
    for with_setup in (False, True):                             # a look that refuses, without and with a painting set-up
        with tool_file(thin_tool([0.020, 0.0, 0.050])), (paint_setup() if with_setup else contextlib.nullcontext()):
            res, arm, bus, photo, root = run_command([], thin_tool([0.020, 0.0, 0.050]), LONG_TIP)
            assert res["stage"] == "look"
    gate = {"tip_offset": THIN_TIP, "tip_mm": [16.0, -1.1, 55.8], "photos": 14, "used": 11, "rms_px": 7.0, "unused": []}
    CLI._verdict_text(gate, "tip", {"problems": ["11 photos carry the fit"], "moved": None, "few": False}, folder)
    _RESULTS.append(gate)                                        # (the hint of a fit that fails the gates)
    cmds = sorted({c for res in _RESULTS for t in _texts(res) for c in re.findall(r"`(fd [^`]+)`", t)})
    assert len(cmds) >= 15 and len(_RESULTS) >= 12, (len(cmds), len(_RESULTS))
    bad = [(c, why) for c in cmds for why in [_accepted(c)] if why]
    assert not bad, bad
    # nothing else stands in backticks than a command: fd's, or the two of the shell that the hints name
    other = sorted({c for res in _RESULTS for t in _texts(res) for c in re.findall(r"`([^`]+)`", t) if not c.startswith("fd ")})
    assert all(c in ("df -h",) for c in other), other
    # the ones this command prints most are among them, in the plain spelling
    for want in ("fd calib-air --dry", "fd --sim calib-air --dry", "fd calib-air --sets air,low", "fd tool status", "fd hw status",
                 "fd paint rest", "fd tool rest", f"fd calib-air --from {folder} --save --force"):
        # (round 3b: a held result and a registration without a turn aside name no fold any more, so no fold with a
        # particular --pan is certain to be among the results of this test alone)
        assert any(c == want or c.startswith(want + " ") for c in cmds), (want, cmds)
    # and the check itself can fail: the spellings that did not work are told apart
    assert _accepted("fd calib-air --dry") is None and _accepted(f"fd --sim calib-air --from {folder} --save --force") is None
    assert _accepted("fd calib-air --drx") and _accepted("fd calib-air --dry --sim") and _accepted("fd calibair --dry")
    assert _accepted("fd tool rest --pan=-12.3 --keep-torque") is None and _accepted("fd tool rest --pan") is not None
    for typed in ("fd tool rest --pan P --keep-torque", "fd calib-air --from DIR --save", "fd tool attach --name N --along-fingers LEN",
                  "fd tool set --tip=x,y,z", "fd --sim calib-air ...", "fd tool jaw mount moving_outer|fixed|between",
                  "fd tool set --joint-limit wrist_flex=LO:HI", "fd paint preview PLAN"):
        assert _accepted(typed) is not None, typed
    # round 3, T12: a command that parses and is refused as it is typed is not a command to print either (the three that
    # this module printed: in two reasons, and as the hint of a simulated plan that cannot fold) ...
    for typed in ("fd --sim calib-air", "fd --sim calib-air --save", "fd --sim calib-air --end stay", "fd calib-air --force",
                  "fd calib-air --dry --save"):
        assert "refused as typed" in (_accepted(typed) or ""), typed
    assert _accepted("fd --sim calib-air --dry --end stay") is None and _accepted("fd calib-air --end stay") is None
    assert _accepted(f"fd --sim calib-air --from {folder} --sets air,low --save") is None and _accepted("fd calib-air --no-look") is None
    # ... and no text of the two modules spells one out, whether a result of this suite printed it or not: every `fd ...`
    # between backticks in their string literals (docstrings and comments are not printed)
    spelled = [(os.path.basename(m.__file__), line, c) for m in (CLI, AC) for line, c in _printed_commands(m.__file__)]
    assert len(spelled) >= 40 and any(c == "fd tool detach" for _f, _l, c in spelled), len(spelled)
    bad = [(f, line, c, why) for f, line, c in spelled for why in [_accepted(c)] if why]
    assert not bad, bad
    assert not any("`fd --sim calib-air`" in r.get("reason", "") for r in _RESULTS if r.get("stage") == "options")


# ----------------------------------------------------------------------------- the rig's files
def test_zz_the_rigs_files_were_never_touched():
    assert not _touched
    assert os.environ["FD_TOOL"] == TOOL_PATH and os.environ["FD_CAMERA_MODEL"] == MODEL_PATH
    assert _rig_files() == _RIG_FILES, "a test created, changed or removed a state file of the rig"
    new = [p for p in glob.glob(os.path.join(tazuna_paths.ROOT, "backups", "camera_model.before_aircal_*"))
           if os.path.getmtime(p) >= T_START - 1.0]
    assert not new, new
    made = [p for p in glob.glob(os.path.join(tazuna_paths.ROOT, "runs", "*", "aircal_*")) if os.path.getmtime(p) >= T_START - 1.0]
    assert not made, made


if __name__ == "__main__":
    import traceback
    only = sys.argv[1:]
    fails = skipped = 0
    n = 0
    t_all = time.time()
    try:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn) and (not only or any(o in name for o in only)):
                n += 1
                t0 = time.time()
                try:
                    fn()
                    print(f"PASS {name} ({time.time() - t0:.1f} s)", flush=True)
                except Skip as e:
                    skipped += 1
                    print(f"SKIP {name}: {e}", flush=True)
                except Exception:
                    fails += 1
                    print("FAIL", name, flush=True)
                    traceback.print_exc()
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
    print(f"tests: {n}  passed: {n - fails - skipped}  skipped: {skipped}  failures: {fails}  ({time.time() - t_all:.0f} s)")
    sys.exit(1 if fails else 0)
