"""Tests of the painting commands (`fd paint ...`: tools/paint/brushwork.py, paintskill.py, paintcli.py).
Run: python tests/test_paint.py [name-part ...]      (no hardware: simulated servo bus, saved and synthetic photos)

What they show, on the simulated arm with a brush tool from $FD_TOOL (the thin round brush of 2026-11-13 with
its tip under the fingers, the flat brush of that morning, and a brush taped far to the side):
 - the depth guard: nothing below table_z - 6 mm, no press over 6 mm, press and --lower-mm together no more than
   7.5 mm below the first-touch plane, no rung of a contact ladder more than 6 mm below the plane the set-up
   already has; all refused before anything moves;
 - a hop raises the brush first and stays in the travel band; a hop stopped by the height monitor is done once
   more at half speed and a second stop ends the command; near the base the brush still comes 30 mm up; the
   tool's joint limits hold on every waypoint; a stroke runs at contact(x, y) - press of the set-up's PLANE;
 - recipes run their steps in order, first and reload, with a wash between unrelated paints and none from a
   tint to the stronger one; a round brush is reloaded by the paint it used, never inside a stroke;
 - a layer stops cleanly at its time limit and goes on where it stopped; progress is per plan + layer + content;
 - the set-up is checked when it is loaded; one-dish set-ups; the brush block; a plan for another brush, canvas
   or pan axis is refused; a contact plane measured for another tip, gripper position or pitch stops `run`;
 - `fd paint contact --read` reads the two photos of the evening of 2026-11-13 as the eye did (skipped when the
   photos are not there) and synthetic photos with marks at known rungs; with a synthetic camera the whole
   command measures a hidden plane and stores it with the brush state;
 - relocate_wells and canvas_view on synthetic photos; `fd paint check` names a dish out of reach;
   `fd paint rest` never leaves the brush, or where it would sink to, over the canvas;
 - the time model; every sub-command's result (ok, stage, reason / next, say), refusals without tracebacks,
   --dry without a single bus write; an interrupt or a stopped move ends with the brush up and says where;
   servos that stop answering in the middle are reported as that (the brush's place "unknown", never "lifted"),
   and the strokes painted until then are stored; a state file or a photo that cannot be written is said;
 - what the review of 2026-11-13 found (tests 32 to 47): a brush whose tip was never measured does not move
   (the word --tip-looked-at); `fd paint contact` needs --scrap-laid and refuses a sheet that is not bare paper
   under its ladders (shown on the two real photos of that day); `fd paint locate` stores only a palette, and
   keeps the places before; the record beside the contact photos and what --read / --save / --dry / --adopt do
   with it; a plane without a record once the tool file is newer; the jaw and its squeeze in the record; a jaw
   that is shut without a recorded hold; progress by the real path; a state file that cannot be read; what the
   brush carries, per tool; plans without a brush block; the blotting strip in `check`; strokes near the
   palette; only runs that ended ok in the time means; rest_report.
 - what the second review found (2026-11-14, tests 48 to 53 and what was tightened in 14, 15, 19, 34, 38, 39, 46):
   a plane without a record is never painted with; `locate` puts no well where the photo shows no paint (a
   turned palette, wells that are not located); test strokes on the canvas need --scrap-laid and a look; the
   order `fd paint status` leads through is typed, command by command, on copies of the rig's files (49);
   `rest` with a tip that was never measured, and an arm that already lies folded; a brush whose content is not
   known is washed; a jaw that stands elsewhere; the next of `run`; no --sim in the paint parser.
 - what the third review found (2026-11-14, tests 54 to 56 and what was tightened in 26, 29, 38, 39, 45, 49, 50, 52, zy):
   `locate` writes no well where the photo shows no paint, also with the set-up 3 to 6 deg off the palette's turn
   (a piece whose middle is no paint is nobody's blob, a piece within reach of two wells is given to neither,
   every place that is written passes the paint test; real photos, skipped on a clone, and that review's scan
   reduced); a `rest` that cannot fold at all names the way out, not itself; the continuation of `run` keeps
   --stop; a plane ADOPTED for another brush state names the adoption again; the next of `brush` is status; an
   arm left as it lies is not said to have been folded; no option of a sub-command is abbreviated; and every
   command a result prints in backticks, or begins its next with, works AS IT IS TYPED: the parsers take it with
   nothing filled in, no word of it is a placeholder (PLAN, DIR, x,y, P), none is a bare command name (zy).
 - what the last verification left open (2026-11-14 evening, test 57): `locate` stores no well on its NEIGHBOUR's
   paint. Every well that took a blob must lie where the other wells' blobs put it, else nothing is stored (the
   verifier's case on a drawn palette and on the photos it happened on, skipped on a clone); and a well remembers
   the colour it showed ("hsv") and is never written onto paint of another colour.
What is IN-SAMPLE is said where it is: the time model against the five calls it was fitted to (test 22, with the
leave-one-out figures), the contact reading against the photos its rule was derived from (16) and against
synthetic photos drawn with the rule's inverse (17, 18, 29), the bare-paper look against three of the photos its
thresholds were set on (33, part 3; its synthetic window shows where the limit lies).

Every test uses scratch files ($FD_TOOL, $FD_PAINT_SETUP, $FD_PAINT_STATE, $FD_PAINT_OUT) and its own hardware
lock ($TAZUNA_HW_LOCK, a scratch file that stays set: nothing here can reach the real arm or the camera). The
default tool of the tests is a brush that is READY to paint (tip measured, jaw recorded and held): tool().
A simulated pass is not a pass on the real arm: contact force, paint and paper are not simulated.
"""

import os as _os  # the harness modules live in tools/<feature>/ (see tools/tazuna_paths.py)
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "tools"))
import tazuna_paths  # noqa: E402,F401

import contextlib
import copy
import hashlib
import json
import math
import io
import os
import re
import shlex
import shutil
import signal
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
_TMP = tempfile.mkdtemp(prefix="tz_paint_")
BACKUP = "/home/jetson/lerobot/jetharness_work/paint_update/state_backup"       # the rig as it stood on 2026-11-13 19:10
EVENING_PHOTOS = (os.path.join(tazuna_paths.ROOT, "runs", "20261113_paint", "173916_look.jpg"),
                  os.path.join(tazuna_paths.ROOT, "runs", "20261113_paint", "174247_test.jpg"))
# the camera of that day (config/camera_model.json of the rig, air calibration of 2026-11-13)
CAMERA = {"f": 1430.0, "cx": 960.0, "cy": 540.0, "k1": 0.0,
          "rvec": [1.612758669942364, 1.6178146130486317, -0.8615937575277688],
          "tvec": [-0.08753385134198806, -0.08456418778922864, 0.6391351033336194], "rms_px": 8.89, "n_points": 55}
CAMERA_PATH = os.path.join(_TMP, "camera_model.json")
with open(CAMERA_PATH, "w") as _f:
    json.dump(CAMERA, _f)
os.environ["FD_CONFIG"] = os.path.join(HERE, "testdata", "pen_20261111", "config.json")
os.environ["FD_CAMERA_MODEL"] = CAMERA_PATH
TOOL_PATH = os.path.join(_TMP, "tool_active.json")
SETUP_PATH = os.path.join(_TMP, "paint_setup.json")
STATE_PATH = os.path.join(_TMP, "paint_state.json")
OUT_DIR = os.path.join(_TMP, "out")
LOCK_PATH = os.path.join(_TMP, "hardware.lock")
os.environ["FD_TOOL"] = TOOL_PATH          # (this spelling: tests/test_integration_pen.py looks for it in every suite that builds an Arm)
os.environ.update(FD_PAINT_SETUP=SETUP_PATH, FD_PAINT_STATE=STATE_PATH, FD_PAINT_OUT=OUT_DIR, TAZUNA_HW_LOCK=LOCK_PATH)
with open(LOCK_PATH, "w") as _f:           # this process is locked out of the real hardware, whatever the user's lock says
    json.dump({"reason": "tests/test_paint.py", "since": time.strftime("%Y-%m-%dT%H:%M:%S"), "by": "test"}, _f)

_TRIPPED = []


def _audit(event, args):
    """These tests must never open the servo bus or the camera."""
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        p = os.fsdecode(args[0])
        if p.startswith(("/dev/ttyACM", "/dev/ttyUSB", "/dev/video")):
            _TRIPPED.append((event, p))
            raise RuntimeError(f"test touched hardware: open({p})")
    elif event in ("subprocess.Popen", "os.system", "os.posix_spawn", "os.exec"):
        s = repr(args)
        if "v4l2" in s or "/dev/video" in s or "/dev/ttyACM" in s:
            _TRIPPED.append((event, s))
            raise RuntimeError(f"test touched the camera: {s}")


sys.addaudithook(_audit)


def _tree_state(root=None) -> dict:
    """The rig's own tool, set-up and state files (their bytes), every file of that kind in the project folder,
    the names in config/ and in runs/, and the names in EVERY runs/*_paint folder (where the painting commands
    write photos and pictures: runs/<date>_paint). Not "the folder of the day the suite started": a suite that
    starts at 23:58 and leaks after midnight writes into a folder of the NEW day, which a listing of two dated
    folders, taken before and after, does not see (second review of 2026-11-13)."""
    out = {}
    root = tazuna_paths.ROOT if root is None else root
    for name in sorted(set(n for n in os.listdir(root) if n.startswith(("paint_", "tool_active")))
                       | {"tool_active.json", "tool_active.sim.json", "paint_setup.json", "paint_state.json", "paint_state.sim.json"}):
        try:
            with open(os.path.join(root, name), "rb") as fh:
                out[name] = hashlib.md5(fh.read()).hexdigest()
        except FileNotFoundError:
            out[name] = None
    for d in ("config", "runs"):
        p = os.path.join(root, d)
        out["dir:" + d] = sorted(os.listdir(p)) if os.path.isdir(p) else None
    for name in out["dir:runs"] or []:
        p = os.path.join(root, "runs", name)
        if name.endswith("_paint") and os.path.isdir(p):
            out["dir:runs/" + name] = sorted(os.listdir(p))
    return out


_TREE_BEFORE = _tree_state()

import cv2  # noqa: E402
import numpy as np  # noqa: E402


def _no_capture(*a, **k):
    _TRIPPED.append(("cv2.VideoCapture", a))
    raise RuntimeError("a test opened the camera")


cv2.VideoCapture = _no_capture

import brushwork as BW  # noqa: E402
import camera as C  # noqa: E402
import hwlock  # noqa: E402
import kinematics as K  # noqa: E402
import paintcli  # noqa: E402
import paintskill as PS  # noqa: E402
import render as RD  # noqa: E402
import tooling as TL  # noqa: E402
from robot import ARM, Arm, MotionAborted, SafetyError, load_config  # noqa: E402
from servo_bus import BusError  # noqa: E402
from sim_bus import SimBus  # noqa: E402

np.set_printoptions(precision=4, suppress=True)
CFG = load_config()
TZ = float(CFG["table_z"])
TPR = 4096.0 / (2.0 * math.pi)
CAM = C.CameraModel.load()
PHOTO_Q = [-89.2, -61.7, 36.2, 85.6, 0.0]
LIMITS = {"wrist_flex": [20.0, 97.0], "wrist_roll": [-5.0, 5.0]}       # the brush handle lies back over the forearm
TIPS = {"thin": [0.016, -0.001, 0.056],        # the thin round brush, jaw squeezed shut: tip under the fingers
        "flat": [0.0085, 0.0007, 0.0611],      # the flat brush of the morning (its band about 9 mm wide)
        "side": [-0.090, -0.010, 0.061]}       # the first mount of the evening: 9 cm beside the finger line
# How each of them sits on the gripper, as a brush that is READY to paint has it in its tool file: the thin one on
# the outer face of the moving jaw, the jaw shut at raw 2041 and held with the squeeze (goal 2016); the flat one
# taped to the fixed jaw; the first mount of the evening on the moving jaw, open 35 deg (nothing recorded).
JAWS = {"thin": dict(gripper_raw=2041, on_jaw="moving_outer", gripper_goal_raw=2016),
        "flat": dict(gripper_raw=2041, on_jaw="fixed"),
        "side": dict(gripper_raw=2576)}
MEASURED = {"at": "2026-11-13T17:25:00", "how": "tests/test_paint.py: as `fd calib-air --tip-only --save` records it",
            "photos_used": 34, "rms_px": 6.6}


class Skip(Exception):
    pass


# ----------------------------------------------------------------------------- the scratch set-up
def _base_setup() -> dict:
    """The rig's set-up of 2026-11-13 19:10 (state_backup, read only); where that folder is not there, the
    part of it these tests use."""
    path = os.path.join(BACKUP, "paint_setup.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)

    def rec(well, first, reload_, after=None):
        d = {"first": first, "reload": reload_}
        if after:
            d["after_without_rinse"] = after
        return d

    paints = {"sky": rec("brown", [["water", "clean", 2], ["scrub", "brown", 2]], [["water", "clean", 1], ["scrub", "brown", 2]])}
    for tube in ("red", "black", "blue", "yellow", "crimson"):
        paints[tube] = rec(tube, [["water", "clean", 1], ["scrub", tube, 3]], [["water", "clean", 1], ["scrub", tube, 2]],
                           [tube + "_pale", tube + "_mid"])
        paints[tube + "_mid"] = rec(tube, [["water", "clean", 1], ["scrub", tube, 2]], [["water", "clean", 1], ["scrub", tube, 2]],
                                    [tube + "_pale"])
        paints[tube + "_pale"] = rec(tube, [["water", "clean", 2], ["touch", tube]], [["water", "clean", 1], ["touch", tube]])
    paints["ink"] = rec("black", [["water", "clean", 1], ["scrub", "black", 2], ["blot"]],
                        [["water", "clean", 1], ["scrub", "black", 1], ["blot"]], ["black", "black_pale", "black_mid"])
    paints["ink2"] = rec("black", [["scrub", "black", 3], ["blot"]], [["scrub", "black", 2], ["blot"]],
                         ["ink", "black", "black_pale", "black_mid"])
    wells = {"magenta": [0.0116, 0.2781], "red": [0.0716, 0.2651], "blue": [0.1453, 0.2583], "yellow": [0.1987, 0.1882],
             "crimson": [0.2117, 0.2544], "black": [0.2578, 0.1763], "cerulean": [0.2613, 0.2186], "green": [0.2638, 0.2604],
             "brown": [0.0929, 0.1998], "violet": [0.0096, 0.2351]}
    return {"photo_q_deg": list(PHOTO_Q), "canvas": {"origin_xy": [0.146, -0.118], "size_mm": [224, 153], "pitch_deg": 65.0},
            "contact": {"z0": 0.00542, "x0": 0.2225, "y0": -0.019, "bx": 0.0362, "by": -0.0062}, "press_mm": 5.0,
            "wells": {n: {"xy": xy, "z_mm": 2.5} for n, xy in wells.items()},
            "water": {"rinse": {"xy": [0.346, 0.192], "z_mm": 6.0}, "clean": {"xy": [0.34, 0.186], "z_mm": 6.0}},
            "blot": {"xy": [0.354, 0.012], "size": [0.03, 0.04], "z": 0.006}, "paints": paints}


BASE = _base_setup()
ROUND = {"kind": "round", "modes": {"fine": {"press": 0.6, "width_mm": 2.7}, "broad": {"press": 1.2, "width_mm": 5.0}},
         "load_mm": 140}


def tool(kind="thin", **over):
    """A brush as it is when it is ready to paint: its tip measured (tip_measured), its jaw as JAWS says. Pass
    tip_measured=None for a typed tip, on_jaw=None / gripper_goal_raw=None for a jaw nobody recorded (the rig's
    own file of 2026-11-13: tool(..., **RIG_JAW))."""
    kw = dict(name="brush_" + kind, tip_offset=list(TIPS[kind]), mount="other", pitch_deg=65.0,
              joint_limits_deg=copy.deepcopy(LIMITS), attached_at="2026-11-13T17:17:45", tip_measured=dict(MEASURED),
              **JAWS[kind])
    kw.update(over)
    return TL.Tool(**kw)


RIG_JAW = dict(on_jaw=None, gripper_goal_raw=None)         # jaw shut at 2041, the squeeze only in the file's notes


def use_tool(t="thin", age_s=3600.0):
    """The tool file of the tests ($FD_TOOL) holds this tool; returns it. The file is dated age_s seconds back
    (as a brush that was registered before the set-up was written; which of the two files is the older one
    decides nothing any more: test 39)."""
    t = tool(t) if isinstance(t, str) else t
    TL.clear()
    TL.save(t)
    stamp = time.time() - age_s
    os.utime(TOOL_PATH, (stamp, stamp))
    return t


def make_setup(brush="round", record=None, pitch=65.0, **edits) -> dict:
    """A copy of the rig's set-up: with the round brush block (brush=None: no block = the flat brush of the
    morning), `record`: the tool whose state the contact plane is recorded for, at canvas pitch `pitch` (None: no
    record, as the rig's file is: such a plane is not painted with), and any top-level keys replaced."""
    s = copy.deepcopy(BASE)
    s["contact"].pop("measured", None)
    s.pop("brush", None)
    if brush == "round":
        s["brush"] = copy.deepcopy(ROUND)
    elif brush is not None:
        s["brush"] = brush
    if record is not None:
        s["contact"]["measured"] = PS.brush_state(record, pitch)
    s.update(edits)
    return s


def use_setup(s=None) -> dict:
    """The set-up file of the tests ($FD_PAINT_SETUP) holds this set-up (written as a hand would: no check)."""
    s = make_setup() if s is None else s
    with open(SETUP_PATH, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1, ensure_ascii=False)
    return s


def reset_state():
    for sim in (False, True):
        with contextlib.suppress(FileNotFoundError):
            os.remove(PS.state_path(sim))


def write_plan(name, layers, brush="round", canvas=(224, 153), pan=None, **more) -> str:
    """A plan file in the scratch folder. layers: [(layer name, paint, [(pts, press), ...]), ...]."""
    plan = {"canvas_mm": list(canvas), "pan_axis_mm": list(pan or PS.pan_axis_mm(BASE)), "paints": {},
            "layers": [{"name": n, "paint": p, "strokes": [{"pts": [list(q) for q in pts], "press": press} for pts, press in sts]}
                       for n, p, sts in layers]}
    if brush == "round":
        plan["brush"] = {"kind": "round", "wide_mm": 2.7, "broad_mm": 5.0, "load_mm": 140.0}
    elif brush is not None:
        plan["brush"] = brush
    plan.update(more)
    path = os.path.join(_TMP, name)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(plan, f)
    return path


def bands(n, u0=60.0, v0=50.0, length=40.0, gap=6.0, press=1.2):
    """n parallel strokes in the middle of the canvas, `length` mm long along u, `gap` mm apart in v."""
    return [([(u0, v0 + gap * k), (u0 + length, v0 + gap * k)], press) for k in range(n)]


def bands_as_strokes(n, **kw) -> list:
    """bands(...) as the strokes of a plan's layer: [{"pts", "press"}, ...]."""
    return [{"pts": [list(q) for q in pts], "press": press} for pts, press in bands(n, **kw)]


# ----------------------------------------------------------------------------- the simulated arm
class RecBus(SimBus):
    """SimBus that remembers every pose the simulated arm passes through."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.trace = []

    def _advance(self, dt):
        super()._advance(dt)
        self.trace.append({j: self.pos[j] for j in ARM})


def q_of(pos):
    z, s = CFG["zero_ticks"], CFG["sign"]
    return np.array([s[j] * (pos[j] - z[j]) / TPR for j in ARM])


def place(arm, q):
    raw = arm.q_to_raw_f(q)
    for j, v in raw.items():
        arm.bus.pos[j] = arm.bus.goal[j] = arm.bus.sp[j] = float(v)


def sim_arm(t="thin", q=None, torque=False):
    """A simulated arm as `fd --sim` builds it (limp, in the tool's rest pose) carrying the tool; or at q,
    holding, with the simulated sag settled."""
    t = tool(t) if isinstance(t, str) else t
    bus = RecBus(realtime=False)
    arm = Arm(bus=bus, cfg=load_config(), tool=t).open()
    TL.sim_start(arm)
    if q is not None:
        place(arm, q)
    if torque:
        arm.torque_on()
        bus.sleep(0.6)
    bus.trace.clear()
    return arm, bus


def raises_value(fn, *a, **k):
    """The ValueError (or SafetyError) fn raises; an AssertionError when it raises none."""
    try:
        fn(*a, **k)
    except (ValueError, SafetyError) as e:
        return e
    raise AssertionError(f"{getattr(fn, '__name__', fn)} did not refuse")


def painter(t="thin", setup=None, **kw):
    arm, bus = sim_arm(t)
    return arm, bus, PS.Painter(arm, make_setup() if setup is None else setup, **kw)


def tips(arm, bus, start=0) -> np.ndarray:
    """(N, 3) tool tip positions of the poses the simulated arm went through."""
    tl = K.tool_offset(*arm.tool.tip_offset)
    return np.array([K.fk(q_of(p), tl)[:3, 3] for p in bus.trace[start:]]).reshape(-1, 3)


def moved(bus, w0=0) -> bool:
    """Did anything reach the servos since the bus had w0 writes? (goal writes: the log)"""
    return bus.io["writes"] != w0 or bool(bus.log)


@contextlib.contextmanager
def patched(obj, name, value):
    had = hasattr(obj, name)
    old = getattr(obj, name, None)
    setattr(obj, name, value)
    try:
        yield
    finally:
        if had:
            setattr(obj, name, old)
        else:
            delattr(obj, name)


def cli(*argv, prep=None, camera=None):
    """paintcli.main on a simulated arm the test can look at afterwards: (result, {"arm", "bus"} of the last
    arm the command built). prep(arm): called on every arm it builds. camera: a photo function that stands
    in for the real camera. A call without --sim reaches for the real arm and is stopped by the tests' own
    hardware lock (asserted)."""
    made = {}
    real = paintcli._make_arm

    def make(sim, tool="sim"):
        if not sim:
            assert hwlock.info() is not None and hwlock.path() == LOCK_PATH, "the tests' own hardware lock is gone"
            return real(False)
        bus = RecBus(realtime=False)
        arm = Arm(bus=bus, cfg=load_config(), tool=TL.active(sim=True) if tool == "sim" else tool)
        arm.open()
        TL.sim_start(arm)
        if prep is not None:
            prep(arm)
        made.update(arm=arm, bus=bus)
        return arm

    with contextlib.ExitStack() as stack:
        stack.enter_context(patched(paintcli, "_make_arm", make))
        if camera is not None:
            stack.enter_context(patched(paintcli, "_camera", lambda sim: camera))
        res = paintcli.main([str(a) for a in argv])
    _RESULTS.append(res)
    return res, made


_RESULTS = []          # every result of cli(): the commands they name are parsed at the end (test_zy_...)


def check_result(r, ok=None, stage=None):
    """What every result of `fd paint` has: ok, stage, say (Japanese); reason and next when it did not do its
    job; nothing that is a traceback; printable as JSON."""
    assert isinstance(r, dict) and isinstance(r.get("ok"), bool) and isinstance(r.get("stage"), str) and r["stage"], r
    assert isinstance(r.get("say"), str) and any(ord(c) > 0x3000 for c in r["say"]), r.get("say")
    if not r["ok"]:
        assert isinstance(r.get("reason"), str) and r["reason"] and isinstance(r.get("next"), str) and r["next"], r
    assert "trace" not in r and r["stage"] != "internal" and "Traceback" not in json.dumps(r, default=str), r
    json.dumps(r, default=paintcli._json_default, ensure_ascii=False)
    if ok is not None:
        assert r["ok"] is ok, r
    if stage is not None:
        assert r["stage"] == stage, (r["stage"], r.get("reason"))
    return r


# ----------------------------------------------------------------------------- synthetic photos
PAPER_BGR = (176, 172, 170)
RED_BGR = (48, 52, 205)


def blank_photo(seed=1, paper=PAPER_BGR):
    """A photo of nothing but paper (the whole frame), with a little noise."""
    rng = np.random.default_rng(seed)
    img = np.empty((1080, 1920, 3), np.float32)
    img[:] = np.array(paper, np.float32)
    img += rng.normal(0.0, 1.0, img.shape).astype(np.float32)
    return np.clip(img + 0.5, 0, 255).astype(np.uint8)


def draw_patch(img, x0, x1, y0, y1, colour=RED_BGR, z=TZ, cam=None):
    """Fill the table rectangle x0..x1, y0..y1 (metres, at height z) in a photo."""
    cam = cam or CAM
    uv = cam.project([[x0, y0, z], [x0, y1, z], [x1, y1, z], [x1, y0, z]])
    cv2.fillPoly(img, [np.round(uv * 16).astype(np.int32)], colour, cv2.LINE_AA, 4)


def draw_rung_mark(img, rung, length_mm, width_mm, beyond_mm=3.0, side_mm=-3.0, pieces=1):
    """A mark as the evening's looked from above: it starts beyond_mm past the END of the rung (towards the
    robot), is length_mm long (in `pieces` dabs with gaps of 1.5 mm) and width_mm wide, side_mm beside the rung."""
    x_hi = rung["x1"] - beyond_mm / 1000.0
    y = rung["y"] + side_mm / 1000.0
    piece = (length_mm - 1.5 * (pieces - 1)) / pieces
    for k in range(pieces):
        a = x_hi - (k * (piece + 1.5)) / 1000.0
        draw_patch(img, a - piece / 1000.0, a, y - width_mm / 2000.0, y + width_mm / 2000.0)


def mark_for(depth_mm):
    """(part of the pull that is marked, width in mm) of the mark a rung leaves when it runs depth_mm below first
    touch: the reading rule of paintskill.first_touch, BACKWARDS. So the synthetic photos are drawn with the
    rule's own inverse: tests that read them (17, 18, 29) are IN-SAMPLE for the rule. They show that reader and
    drawer agree, that marks are found, assigned to the right rung and measured through the camera model, and
    what is refused; they do not show that the rule is right on paper (that rests on the one pair of real
    photos of test 16, from which the rule was derived: in-sample too, until a second real ladder is read).
    Nothing above the paper; dabs while the rung
    just touches (depth = 0.75 mm x cover, up to three quarters of the pull); a line over the whole pull,
    depth - 0.8 mm wide (at most 5), from 1.8 mm on; between, a 1 mm line over more and more of the pull."""
    t, b, w_min = PS.TOUCH_PER_COVER_MM, PS.LINE_BASE_MM, 1.0
    if depth_mm < 0.0:
        return None
    if depth_mm < 0.75 * t:
        return depth_mm / t, 1.4
    if depth_mm >= b + w_min:
        return 1.1, min(5.0, depth_mm - b)
    lo, hi = 0.75, 1.0
    for _ in range(30):
        c = 0.5 * (lo + hi)
        k = (c - 0.75) / 0.25
        if (1.0 - k) * t * c + k * (b + w_min) < depth_mm:
            lo = c
        else:
            hi = c
    return 0.5 * (lo + hi), w_min


def ladder_photo(places, touch_of, **kw):
    """The photo after ladders were painted on paper whose first-touch height is touch_of(x, y) (mm, plan z)."""
    img = blank_photo(seed=2)
    for p in places:
        for r in p["rungs"]:
            m = mark_for(touch_of((r["x0"] + r["x1"]) / 2.0, r["y"]) - r["plan_z_mm"])
            if m is not None:
                draw_rung_mark(img, r, max(1.6, (r["x0"] - r["x1"]) * 1000.0 * m[0]), m[1], **kw)
    return img


BLOB_BGR = {"red": (40, 40, 215), "crimson": (50, 30, 140), "blue": (150, 60, 30), "cerulean": (200, 110, 40),
            "yellow": (40, 200, 240), "green": (70, 150, 40), "brown": (50, 65, 120), "violet": (140, 50, 95),
            "magenta": (120, 40, 200), "black": (45, 30, 26)}
DESK_GREY = 62


def desk_photo():
    """A photo of the bare desk: no palette, no paper."""
    return np.full((1080, 1920, 3), DESK_GREY, np.uint8)


def draw_blob(img, xy, colour, r=0.011):
    """A blob of paint of radius r at table point xy (at blob height)."""
    c = CAM.project([[xy[0] + r * math.cos(a), xy[1] + r * math.sin(a), TZ + 0.004] for a in np.linspace(0, 2 * math.pi, 40)])
    cv2.fillPoly(img, [np.round(c).astype(np.int32)], colour)


def palette_places(s, shift=(0.0, 0.0), turn_deg=0.0) -> dict:
    """Where the blobs of set-up s lie after the palette was turned by turn_deg about its middle and pushed by
    `shift`: {well: (x, y)}."""
    pts = np.array([w["xy"] for w in s["wells"].values()], dtype=float)
    c, a = pts.mean(axis=0), math.radians(turn_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    moved = (pts - c) @ rot.T + c + np.array(shift, dtype=float)
    return {name: (float(p[0]), float(p[1])) for name, p in zip(s["wells"], moved)}


def palette_photo(s, shift=(0.0, 0.0), skip=(), turn_deg=0.0, r=0.011):
    """The white palette with a blob of paint in every well of set-up s (but those in `skip`), pushed by `shift`
    (and turned by turn_deg about its middle)."""
    img = desk_photo()
    places = palette_places(s, shift, turn_deg)
    pts = np.array(list(places.values()))
    lo, hi = pts.min(axis=0) - 0.045, pts.max(axis=0) + 0.045
    draw_patch(img, lo[0], hi[0], lo[1], hi[1], colour=(236, 236, 236), z=TZ + 0.004)      # the white palette
    for name, xy in places.items():
        if name not in skip:
            draw_blob(img, xy, BLOB_BGR[name], r)
    return img


def paint_share(img, xy, r_m=0.008) -> float:
    """The part of a photo's pixels within r_m of table place xy (at blob height) that is paint-coloured, judged
    HERE and not with the module's own mask (other thresholds: saturated, or very dark)."""
    c, e = CAM.project([[xy[0], xy[1], TZ + 0.004], [xy[0] + r_m, xy[1], TZ + 0.004]])
    r = max(4, int(round(float(np.linalg.norm(e - c)))))
    x, y = int(round(float(c[0]))), int(round(float(c[1])))
    if not (r <= x < img.shape[1] - r and r <= y < img.shape[0] - r):
        return 0.0
    hsv = cv2.cvtColor(img[y - r:y + r + 1, x - r:x + r + 1], cv2.COLOR_BGR2HSV)
    yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
    disc = (xx * xx + yy * yy) <= r * r
    sat, val = hsv[..., 1][disc].astype(int), hsv[..., 2][disc].astype(int)
    return float((((sat > 90) & (val > 40)) | (val < 70)).mean())


# Hidden paper heights (plan z of first touch, mm) for the five standard places, in their order: near-left,
# near-right, far-left, far-right, centre. Three places just touch a rung (dabs: read to a few tenths), two are
# pressed 2 mm below their first marked rung (a line: read to about a millimetre, the photo's blur widens it).
HIDDEN = (3.3, 5.0, 9.2, 8.0, 6.3)
HIDDEN_TOL = (0.35, 1.2, 0.35, 1.2, 0.35)


def hidden_paper(places, heights=HIDDEN):
    """touch_of(x, y) for ladder_photo: every ladder stands on paper of its own height."""
    def touch_of(x, y):
        k = int(np.argmin([math.hypot(x - (p["rungs"][0]["x0"] - 0.005), y - (p["rungs"][0]["y"] + 0.018)) for p in places]))
        return heights[k]
    return touch_of


# ============================================================================= 1. the moves of the brush
def test_01_the_depth_guard_refuses_before_anything_moves():
    arm, bus, pt = painter()
    br = pt.br
    w0 = bus.io["writes"]
    for what, call in (
            ("a line 6.5 mm under the table", lambda: br.line([np.array([0.25, 0.0, TZ - 0.0065])], 15.0)),
            ("a dab 7 mm under the table", lambda: br.dab((0.25, 0.0), TZ - 0.007)),
            ("a dab whose pattern goes under", lambda: br.dab((0.25, 0.0), TZ + 0.002, pattern=[(0.0, 0.0, -0.009)])),
            ("a press of 6.5 mm", lambda: br.stroke([(0.25, 0.0), (0.24, 0.0)], TZ + 0.004, 0.0065)),
            ("a press that is not a number", lambda: br.stroke([(0.25, 0.0), (0.24, 0.0)], TZ + 0.004, float("nan"))),
            ("a stroke 6.5 mm under the table", lambda: br.stroke([(0.25, 0.0), (0.24, 0.0)], TZ - 0.0015, 0.005)),
            ("a hop that ends 20 mm up", lambda: br.hop((0.25, 0.0), h=0.02)),
            ("press 1.3 x 5 mm through the painter", lambda: pt.stroke_xy([(0.25, 0.0), (0.24, 0.0)], 1.3))):
        try:
            call()
        except SafetyError as e:
            assert "mm" in str(e), (what, e)
        else:
            raise AssertionError(f"{what} was not refused")
        assert not moved(bus, w0) and not arm.is_torqued() and not br.moved, what
    assert BW.MAX_PRESS_M == 0.006 and BW.MAX_BELOW_TABLE_M == 0.006 and abs(br.z_min - (TZ - 0.006)) < 1e-12
    # the same through a plan: refused as a plan problem, nothing moved
    plan = write_plan("deep.json", [("deep", "red", bands(1, press=1.3))])
    with open(plan) as f:
        try:
            pt.run_layer(json.load(f), 0)
        except PS.Refusal as e:
            assert e.stage == "plan" and "6 mm" in str(e), e
        else:
            raise AssertionError("a plan that presses 6.5 mm was not refused")
    assert not moved(bus, w0)
    # what is allowed runs, and exactly as deep as it says: 6 mm below a contact at +4.5 mm
    r = br.stroke([(0.25, 0.0), (0.23, 0.0)], TZ + 0.006, 0.006)
    assert abs(r["z_low"] - TZ) < 1e-4 and abs(br.lowest_z - TZ) < 1e-9, r
    low = float(tips(arm, bus)[:, 2].min())
    assert TZ - 0.0035 < low < TZ + 0.002, low           # (the simulated arm sags a little; the PLAN is the limit)


def test_02_a_hop_raises_first_and_keeps_the_travel_band():
    arm, bus, pt = painter()
    br = pt.br
    calls = []
    real = TL.free_move

    def free_move(arm_, q_to, speed, **kw):
        calls.append((speed, kw.get("min_height")))
        return real(arm_, q_to, speed, **kw)

    with patched(TL, "free_move", free_move):
        br.hop((0.25, 0.0))                              # from the rest pose to above the canvas
        assert abs(br.height() - 0.065) < 0.004 and np.allclose(br.tip()[:2], (0.25, 0.0), atol=0.003)
        br.line([np.array([0.25, 0.0, TZ + 0.008])], 15.0, what="down")            # the brush is low now
        n0 = len(bus.trace)
        calls.clear()
        br.hop((0.0716, 0.2651))                         # a far hop, to the red well
        tr = tips(arm, bus, n0)
        h = tr[:, 2] - TZ
        up = int(np.argmax(h >= 0.060))
        assert up > 0 and float(np.abs(tr[:up, :2] - tr[0, :2]).max()) < 0.002, "it did not rise straight up first"
        assert float(h[up:].min()) >= 0.065 - br.o.sag_m, float(h[up:].min())      # the band of a far move
        assert len(calls) == 1 and calls[0][0] == 20.0 and abs(calls[0][1] - 0.035) < 0.003, calls
        assert np.allclose(br.tip()[:2], (0.0716, 0.2651), atol=0.003) and abs(br.height() - 0.065) < 0.004
        # to a stroke start: it ends 30 mm up and the monitor's limit is the 2 cm of every free move
        calls.clear()
        n0 = len(bus.trace)
        br.hop((0.25, -0.02), h=br.o.above_h)
        assert abs(calls[0][1] - TL.MIN_FREE_HEIGHT) < 1e-9 and abs(br.height() - 0.030) < 0.004
        assert float((tips(arm, bus, n0)[:, 2] - TZ).min()) >= 0.020
    assert BW.BrushOptions().hop_deg_s == 20.0           # (30 until 2026-11-13 18:15: four stops; none after)


def _trip_monitor(arm, n):
    """The next n monitored motions of this arm are stopped by their monitor after 0.8 s (as a measured tip
    height below the limit stops a free move)."""
    left = {"n": n}
    orig = arm.run_path

    def run_path(wps, speed, **kw):
        if kw.get("monitor") is not None and left["n"] > 0:
            left["n"] -= 1
            mon = kw["monitor"]
            kw["monitor"] = lambda t, q_plan, q_meas: bool(mon(t, q_plan, q_meas) or t > 0.8)
        return orig(wps, speed, **kw)

    arm.run_path = run_path
    return left


def test_03_a_stopped_hop_is_tried_once_more_at_half_speed():
    arm, bus, pt = painter()
    br = pt.br
    speeds = []
    real = TL.free_move

    def free_move(arm_, q_to, speed, **kw):
        speeds.append(speed)
        return real(arm_, q_to, speed, **kw)

    with patched(TL, "free_move", free_move):
        br.hop((0.25, 0.0))
        speeds.clear()
        _trip_monitor(arm, 1)
        br.hop((0.0716, 0.2651))
        assert speeds == [20.0, 10.0] and br.hop_retries == 1, (speeds, br.hop_retries)
        assert np.allclose(br.tip()[:2], (0.0716, 0.2651), atol=0.003) and abs(br.height() - 0.065) < 0.004
        # a second stop of the same hop ends the command as before: MotionAborted, the arm holds, brush up
        speeds.clear()
        _trip_monitor(arm, 2)
        try:
            br.hop((0.30, -0.05))
        except MotionAborted as e:
            assert e.info.get("reason") == "monitor" and "tool_tip_height_m" in e.info, e.info
        else:
            raise AssertionError("a hop that was stopped twice went on")
        assert speeds == [20.0, 10.0] and br.hop_retries == 2 and arm.is_torqued() and br.height() > 0.05
        # any other abort is not retried
        speeds.clear()

        def target_not_reached(*a, **k):
            speeds.append("x")
            raise MotionAborted("target not reached: test", info={"reason": "target_not_reached"})

        with patched(TL, "free_move", target_not_reached):
            try:
                br.hop((0.30, -0.05))
            except MotionAborted:
                pass
            else:
                raise AssertionError("an abort that is no monitor stop was swallowed")
        assert speeds == ["x"] and br.hop_retries == 2
    # the way to the photo pose has the same second try, and the report counts it
    _trip_monitor(arm, 1)
    rep = br.retreat(PHOTO_Q)
    assert rep["where"] == "photo pose" and rep["hop_retries"] == 3 and rep["height_mm"] > 50 and rep["torque"] == "on", rep


def test_04_near_the_base_the_brush_still_comes_30_mm_up():
    """Near the robot's base the tip cannot rise where it stands (the wrist is at its limit): raise_to finds
    another way up, or says that there is none; it never goes down and never leaves the tool's joint limits."""
    wells = [(BASE["wells"][n]["xy"][0], BASE["wells"][n]["xy"][1], None) for n in ("violet", "magenta", "red", "brown")]
    assert all(x < 0.11 for x, _y, _p in wells)
    close = {"thin": ((0.105, 0.07, 80.0), (0.115, 0.0, 90.0), (0.095, 0.0, 100.0), (0.10, 0.03, 95.0), (0.085, -0.07, 90.0)),
             "flat": ((0.105, 0.07, 80.0), (0.115, 0.0, 90.0), (0.10, 0.03, 95.0), (0.085, -0.07, 90.0))}
    for kind in ("thin", "flat"):
        sideways = []
        for x, y, pitch in wells + list(close[kind]):
            arm, bus = sim_arm(kind)
            p = np.array([x, y, TZ + (0.010 if pitch is None else 0.008)])
            q = arm.solve(p, BW.BrushRun(arm).pitch_at([p]) if pitch is None else pitch, 0.0)
            arm, bus = sim_arm(kind, q=q, torque=True)       # (the simulated arm sags a few millimetres as it takes the load)
            br = BW.BrushRun(arm)
            h0, p0 = br.height(), br.tip()
            assert h0 < 0.012, h0
            assert br.raise_to(0.065) is True
            t = tips(arm, bus)
            assert br.height() >= 0.030 - 0.004, (kind, x, y, br.height())
            assert pitch is not None or br.height() >= 0.065 - 0.004          # over the palette it comes all the way up
            assert float(t[:, 2].min() - TZ) >= h0 - 0.002, (kind, x, y, float(t[:, 2].min() - TZ), h0)   # never down on the way up
            sideways.append(float(np.hypot(*(br.tip()[:2] - p0[:2]))))
            lim = arm.q_limits()
            for pos in bus.trace:
                qq = q_of(pos)
                assert lim["wrist_flex"][0] - math.radians(1.0) <= qq[3] <= lim["wrist_flex"][1] + math.radians(1.0), math.degrees(qq[3])
            assert arm.is_torqued()
        # over the wells it rises where it stands; a few centimetres from the pan axis only up and AWAY from the robot
        assert max(sideways[:4]) < 0.003 and min(sideways[4:]) > 0.02 and max(sideways) < 0.09, (kind, sideways)
    # where there is no way up at all, that is a refusal with the arm holding where it is, not a fall
    arm, bus = sim_arm("thin")
    q = arm.solve(np.array([0.115, 0.0, TZ + 0.008]), 90.0, 0.0)
    arm, bus = sim_arm("thin", q=q, torque=True)
    br = BW.BrushRun(arm)

    def refused(*a, **k):
        raise SafetyError("refused: test")

    q0 = arm.read_q()
    with patched(br, "line", refused), patched(TL, "check_pose", refused), patched(TL, "lift", refused):
        try:
            br.raise_to(0.065)
        except SafetyError:
            pass
        else:
            raise AssertionError("raise_to reported a way up that was not there")
    assert arm.is_torqued() and float(np.abs(arm.read_q() - q0).max()) < math.radians(0.3)


def test_05_the_tools_joint_limits_hold_on_every_waypoint():
    for kind in ("thin", "flat"):
        arm, bus, pt = painter(kind, make_setup(brush="round" if kind == "thin" else None))
        planned = []
        orig = arm.run_path

        def run_path(wps, speed, _orig=orig, **kw):
            planned.extend(np.asarray(w, dtype=float) for w in wps[1:])
            return _orig(wps, speed, **kw)

        arm.run_path = run_path
        pt.load("ink")                                     # water, the far black well, the blotting strip
        w, h = BASE["canvas"]["size_mm"]
        for u, v in ((8, 8), (w - 8, 8), (8, h - 8), (w - 8, h - 8), (w / 2, h / 2)):      # the corners and the centre
            pt.stroke_canvas([(u, v), (u + (10 if u < w / 2 else -10), v)], 0.6)
            pt.stroke_canvas([(u, v), (u, v + (10 if v < h / 2 else -10))], 1.2)
        pt.load("red")                                     # a wash of the brush in between
        pt.br.retreat(PHOTO_Q)
        lim = arm.q_limits()
        assert abs(math.degrees(lim["wrist_flex"][0]) - 20.0) < 1e-6 and abs(math.degrees(lim["wrist_flex"][1]) - 97.0) < 1e-6
        assert abs(math.degrees(lim["wrist_roll"][1]) - 5.0) < 1e-6
        assert len(planned) > 300
        for q in planned:
            for i, j in enumerate(ARM):
                assert lim[j][0] - 1e-6 <= q[i] <= lim[j][1] + 1e-6, (kind, j, math.degrees(q[i]))
        for p in bus.trace[::5]:                           # and where the simulated arm really was (it lags a little)
            q = q_of(p)
            assert lim["wrist_flex"][0] - math.radians(1.5) <= q[3] <= lim["wrist_flex"][1] + math.radians(1.5)
            assert abs(q[4]) <= math.radians(5.0) + math.radians(1.0)


def test_06_a_stroke_runs_at_the_contact_plane_minus_the_press():
    s = make_setup()
    c = s["contact"]
    assert c["bx"] == 0.0362 and c["by"] == -0.0062 and s["press_mm"] == 5.0          # the plane of 2026-11-13
    arm, bus, pt = painter("thin", s)

    def zc(x, y):
        return c["z0"] + c["bx"] * (x - c["x0"]) + c["by"] * (y - c["y0"])

    assert abs(pt.zc(0.30, 0.10) - zc(0.30, 0.10)) < 1e-12 and abs(zc(0.2225, -0.019) - 0.00542) < 1e-12
    planned = []
    orig = arm.run_path

    def run_path(wps, speed, **kw):
        planned.extend(np.asarray(w, dtype=float) for w in wps[1:])       # (the first one is where the arm stands)
        return orig(wps, speed, **kw)

    arm.run_path = run_path
    for pts, press in (([(0.20, -0.06), (0.26, -0.06)], 0.6),        # away from the robot: the paper rises 2.2 mm
                       ([(0.24, 0.07), (0.24, -0.02)], 1.2),         # sideways
                       ([(0.28, 0.05), (0.17, -0.09)], 1.0)):        # diagonal, towards the robot
        planned.clear()
        pt.br.lowest_z = None
        r = pt.stroke_xy(pts, press)
        want = [zc(x, y) - s["press_mm"] / 1000.0 * press for x, y in pts]
        assert abs(r["z_low"] - min(want)) < 1e-4 and abs(pt.br.lowest_z - min(want)) < 1e-9, (r, want)
        tl = K.tool_offset(*arm.tool.tip_offset)
        tip = np.array([K.fk(q, tl)[:3, 3] for q in planned])
        assert abs(float(tip[:, 2].min()) - min(want)) < 2e-4, (float(tip[:, 2].min()), min(want))
        for (x, y), z in zip(pts, want):                   # at both ends the plan passes through the plane's own height
            d = np.hypot(tip[:, 0] - x, tip[:, 1] - y)
            assert abs(float(tip[d < 0.0015, 2].min()) - z) < 2e-4, (x, y, z)
    assert abs((zc(0.26, -0.06) - zc(0.20, -0.06)) * 1000.0 - 2.17) < 0.01             # (it IS a plane: 36 mm per metre)
    # `--lower-mm`: the plane 1.5 mm lower for one command (what the bands of 2026-11-13 needed at the far side)
    assert pt.set_lower(1.5) == 1.5 and abs(pt.zc(0.25, 0.0) - (zc(0.25, 0.0) - 0.0015)) < 1e-12
    planned.clear()
    pt.br.lowest_z = None
    r = pt.stroke_xy([(0.28, 0.0), (0.28, 0.03)], 1.2)
    want = min(zc(0.28, 0.0), zc(0.28, 0.03)) - 0.006 - 0.0015
    tip = np.array([K.fk(q, K.tool_offset(*arm.tool.tip_offset))[:3, 3] for q in planned])
    assert abs(r["z_low"] - want) < 1e-4 and abs(float(tip[:, 2].min()) - want) < 2e-4, (r, want)
    for bad in (2.5, -0.5, float("nan"), "x"):
        try:
            pt.set_lower(bad)
        except PS.Refusal as e:
            assert e.stage == "options"
        else:
            raise AssertionError(f"lowered the plane by {bad!r}")
    assert pt.lower_m == 0.0015 and PS.MAX_LOWER_MM == 2.0
    # the lowering is no second allowance on top of the press: the two together stay within 7.5 mm below first
    # touch (what the bands of 2026-11-13 ran at: 6 mm with the plane lowered by 1.5). Press 1.2 with 2 mm of
    # lowering would be 8 mm, on this plane 8.2 mm below it at the far end: refused before anything moves
    assert PS.MAX_DEPTH_M == 0.0075 and abs(pt.depth_m(1.2) - 0.0075) < 1e-12 and abs(pt.depth_m(0.6) - 0.0045) < 1e-12
    w0 = bus.io["writes"]
    n0 = len(bus.log)
    pt.set_lower(2.0)
    for call in (lambda: pt.depth_m(1.2), lambda: pt.stroke_xy([(0.28, 0.0), (0.28, 0.03)], 1.2)):
        try:
            call()
        except SafetyError as e:
            assert "8.0 mm below first touch" in str(e) and "7.5 mm" in str(e) and "--lower-mm 1.5 at most" in str(e), e
        else:
            raise AssertionError("press 6 mm with the plane lowered by 2 mm was accepted")
    assert bus.io["writes"] == w0 and len(bus.log) == n0
    assert abs(pt.depth_m(1.1) - 0.0075) < 1e-12           # 5.5 mm of press and 2 mm of lowering: the limit itself
    pt.set_lower(1.5)
    # the depth guard does not move with it: 6 mm of press on a plane lowered to 7.5 mm under a table-high contact
    low = make_setup()
    low["contact"] = {"z0": TZ, "x0": 0.2225, "y0": -0.019, "bx": 0.0, "by": 0.0}
    arm2, bus2, pt2 = painter("thin", low)
    pt2.set_lower(1.5)
    try:
        pt2.stroke_xy([(0.25, 0.0), (0.25, 0.03)], 1.2)
    except SafetyError as e:
        assert "below the limit of brush work" in str(e)
    else:
        raise AssertionError("a stroke 7.5 mm under the table was not refused")
    assert not moved(bus2) and not arm2.is_torqued()


# ============================================================================= 2. recipes, loads, layers
def _spy(pt):
    """Replace the painter's moves by notes of what it would do (no motion): returns the list."""
    seen = []
    pt.water = lambda which="clean", passes=2, amp=0.007: seen.append(("water", which, passes))
    pt.scrub = lambda well, passes=4, amp=0.006, side=0.003: seen.append(("scrub", well, passes))
    pt.touch = lambda well: seen.append(("touch", well))
    pt.blot = lambda strokes=2: seen.append(("blot", strokes))
    pt.clean = lambda: seen.append(("clean",))
    return seen


def test_07_recipes_run_their_steps_in_order():
    s = make_setup()
    arm, bus, pt = painter("thin", s)
    seen = _spy(pt)
    r = pt.load("ink")
    assert seen == [("water", "clean", 1), ("scrub", "black", 2), ("blot", 2)] and r["first"] and not r["washed"], seen
    assert r["steps"] == ["water", "scrub", "blot"] and pt.paint == "ink" and pt.used == 0.0 and pt.blots == 1
    seen.clear()
    r = pt.load("ink")                                     # it carries ink already: the reload steps
    assert seen == [("water", "clean", 1), ("scrub", "black", 1), ("blot", 2)] and not r["first"], seen
    seen.clear()
    r = pt.load("red_pale")                                # an unrelated paint: the brush is washed first
    assert seen == [("clean",), ("water", "clean", 2), ("touch", "red")] and r["washed"] and pt.washes == 1, seen
    seen.clear()
    pt.load("red_mid")                                     # a tint, then the stronger tint of the same paint: no wash
    pt.load("red")
    assert ("clean",) not in seen and seen == [("water", "clean", 1), ("scrub", "red", 2), ("water", "clean", 1), ("scrub", "red", 3)], seen
    assert s["paints"]["red"]["after_without_rinse"] == ["red_pale", "red_mid"] and pt.washes == 1
    seen.clear()
    pt.load("red_pale")                                    # back to the pale one: that needs a clean brush
    assert seen[0] == ("clean",) and pt.washes == 2
    pt.paint = None                                        # a clean brush (after `fd paint rinse`): no wash before the first paint
    seen.clear()
    pt.load("black")
    assert ("clean",) not in seen and pt.loads == 7
    try:
        pt.load("pink")
    except PS.Refusal as e:
        assert e.stage == "paint" and "pink" in str(e) and "fd paint status" in e.next
    else:
        raise AssertionError("an unknown paint was loaded")
    assert not moved(bus)
    # and for real, on the simulated arm: where the brush goes down, in the order of the recipe
    arm, bus, pt = painter("thin", s)
    dabs, real_dab, real_stroke = [], pt.br.dab, pt.br.stroke
    pt.br.dab = lambda xy, z, **kw: (dabs.append((kw.get("what"), tuple(xy), round(z - TZ, 4))), real_dab(xy, z, **kw))[1]
    pt.br.stroke = lambda pts, zc, press, **kw: (dabs.append(("stroke", tuple(pts[0]), press)), real_stroke(pts, zc, press, **kw))[1]
    pt.load("ink")
    assert [d[0] for d in dabs] == ["water (clean)", "paint (black)", "stroke", "stroke"], dabs
    assert dabs[0][1] == tuple(s["water"]["clean"]["xy"]) and dabs[0][2] == 0.006
    assert dabs[1][1] == tuple(s["wells"]["black"]["xy"]) and dabs[1][2] == 0.0025
    assert abs(dabs[2][1][0] - (0.354 + 0.015 - 0.004)) < 1e-9 and dabs[2][2] == 0.003                # the blotting strip
    low = tips(arm, bus)
    near_well = np.hypot(low[:, 0] - 0.2578, low[:, 1] - 0.1763) < 0.012
    assert abs(float(low[near_well, 2].min()) - (TZ + 0.0025)) < 0.003                              # down in the blob


def _dry_painter(setup, t="thin"):
    """A painter whose loads and strokes only take notes: for the reload rule, without waiting for moves."""
    arm, bus, pt = painter(t, setup)
    events = []

    def load(paint):
        events.append(("load", paint))
        pt.paint, pt.since_load, pt.used = paint, 0.0, 0.0
        pt.loads += 1

    def stroke(pts_xy, zc, press_m, **kw):
        ln = RD.length_mm(np.asarray(pts_xy) * 1000.0)
        events.append(("stroke", round(ln, 1)))
        return {"pitch": 65.0, "length_m": ln / 1000.0 + 0.001, "z_low": 0.0, "seconds": 5.0, "travel_s": ln / 20.0}

    pt.load = load
    pt.br.stroke = stroke
    return arm, bus, pt, events


def _loads_before(events):
    """Indices of the strokes that got a fresh load."""
    out, k = [], 0
    for i, e in enumerate(events):
        if e[0] == "stroke":
            if i and events[i - 1][0] == "load":
                out.append(k)
            k += 1
    return out


def test_08_a_round_brush_is_reloaded_by_the_paint_it_used():
    s = make_setup(record=tool("thin"))
    with open(write_plan("reload.json", [("bands", "red", bands(6, press=1.2)), ("lines", "red", bands(9, press=0.6)),
                                         ("mixed", "red", bands(2, press=0.6) + bands(2, press=1.2) + bands(3, press=0.6))])) as f:
        plan = json.load(f)
    # a broad band of 40 mm takes 40 * 5.0 / 2.7 = 74 mm of fine line: one band per load of 140; a fine line 40
    assert abs(PS.use_mm(PS.brush_of(s), 40.0, 1.2) - 74.07) < 0.01 and PS.use_mm(PS.brush_of(s), 40.0, 0.6) == 40.0
    arm, bus, pt, ev = _dry_painter(s)
    r = pt.run_layer(plan, 0)
    assert _loads_before(ev) == [0, 1, 2, 3, 4, 5] and r["loads"] == 6 and r["done"] and r["reload"] == {"by": "paint used", "limit_mm": 140.0}, ev
    ev.clear()
    pt.paint, pt.used = None, 1e9
    r = pt.run_layer(plan, 1)
    assert _loads_before(ev) == [0, 3, 6] and r["loads"] == 3, ev              # three fine lines of 40 mm per load
    ev.clear()
    pt.paint, pt.used = None, 1e9
    pt.run_layer(plan, 2)                                  # 40 + 40 | 74 | 74 + 40 | 40 + 40
    assert _loads_before(ev) == [0, 2, 3, 5], ev
    # the count is the plan's own length, not the travel the arm reports (1 mm more per stroke here)
    assert abs(pt.used - 80.0) < 1e-6 and abs(pt.since_load - 80.0) < 1e-6
    # never inside a stroke: a load comes only between two strokes, and the estimate says the same
    assert all(ev[i + 1][0] == "stroke" for i, e in enumerate(ev) if e[0] == "load")
    for li in range(3):
        b = PS.layer_budget(s, plan["layers"][li])
        ev.clear()
        pt.paint, pt.used, pt.since_load = None, 1e9, 1e9
        r = pt.run_layer(plan, li)
        assert (b["loads"], b["strokes"]) == (r["loads"], r["strokes"]), (li, b, r)
        assert r["travel_mm"] == b["travel_mm"] + r["strokes"]         # (the result has the travel the arm reports)
    # --reload-mm still means plain travel, whatever the brush: 75 mm is one 40 mm stroke per load, 100 mm two
    for reload_mm, want in ((75.0, [0, 1, 2, 3, 4, 5, 6, 7, 8]), (100.0, [0, 2, 4, 6, 8])):
        ev.clear()
        pt.paint, pt.used, pt.since_load = None, 1e9, 1e9
        r = pt.run_layer(plan, 1, reload_mm=reload_mm)
        assert _loads_before(ev) == want and r["reload"] == {"by": "travel", "limit_mm": reload_mm}, (reload_mm, ev)
    # no brush block: the flat brush of the morning, 180 mm of plain travel, press or not
    flat = make_setup(brush=None, record=tool("thin"))
    with open(write_plan("reload_flat.json", [("bands", "red", bands(6, press=1.2))], brush=None)) as f:
        plan_flat = json.load(f)
    arm, bus, pt, ev = _dry_painter(flat)
    r = pt.run_layer(plan_flat, 0)
    assert _loads_before(ev) == [0, 4] and r["reload"] == {"by": "travel", "limit_mm": 180.0} and RD.LOAD_MM == 180.0, ev
    # a recipe's own load (a blotted brush carries less): 60 mm of fine line is one 40 mm line per load
    s2 = make_setup(record=tool("thin"))
    s2["paints"]["red"]["load_mm"] = 60
    arm, bus, pt, ev = _dry_painter(PS.check_setup(s2))
    try:
        pt.run_layer(plan, 1)                              # (the bands of this plan take 74: more than such a load)
    except PS.Refusal as e:
        assert e.stage == "plan" and "more than one load of red (60 mm of fine line)" in str(e) and not ev
    else:
        raise AssertionError("a band that takes more than the recipe's load was accepted")
    with open(write_plan("reload_lines.json", [("lines", "red", bands(9, press=0.6))])) as f:
        lines = json.load(f)
    r = pt.run_layer(lines, 0)
    assert _loads_before(ev) == list(range(9)) and PS.layer_budget(s2, lines["layers"][0])["loads"] == 9 and r["reload"]["limit_mm"] == 60.0, ev
    # and with real moves: the loads of a layer on the simulated arm are the loads of the estimate
    arm, bus, pt = painter("thin", s)
    r = pt.run_layer(plan, 2)
    b = PS.layer_budget(s, plan["layers"][2])
    assert r["ok"] and r["done"] and (r["loads"], r["strokes"]) == (b["loads"], 7) == (4, 7), (r, b)


def test_09_a_layer_stops_cleanly_at_its_time_limit_and_goes_on():
    use_tool("thin")
    s = use_setup(make_setup(record=tool("thin")))
    reset_state()
    plan = write_plan("limit.json", [("six lines", "red", bands(6, press=0.6))])
    r, made = cli("--sim", "run", plan, "--layer", "0", "--max-min", "0.5")          # 30 s: the first load alone takes that
    check_result(r, ok=True, stage="run")
    k = r["next_stroke"]
    assert r["time_limit"] is True and not r["done"] and 1 <= k < 6 and r["strokes"] == k and "time limit" in r["reason"], r
    assert f"continues at stroke {k}" in r["next"] and r["brush"]["where"] == "photo pose" and r["brush"]["height_mm"] > 50
    assert made["arm"].bus.torque["shoulder_lift"] == 1                  # (the arm holds: the brush does not sink)
    state = PS.load_state(sim=True)
    key = PS.progress_key(plan, json.load(open(plan)), 0)
    assert state["progress"] == {key: k} and state["paint"] == "red" and r["seconds"] > 30
    lifted = tips(made["arm"], made["bus"])[-1]
    assert lifted[2] - TZ > 0.05
    r2, made2 = cli("--sim", "run", plan, "--layer", "0")                # the same command goes on where it stopped
    check_result(r2, ok=True, stage="run")
    assert r2["start"] == k and r2["done"] and r2["next_stroke"] == 6 and r2["strokes"] == 6 - k, r2
    assert PS.load_state(sim=True)["progress"][key] == 6
    # the strokes that were painted: every one of the six lines once (the brush low over each of them once)
    for arm_, bus_, rng in ((made["arm"], made["bus"], range(0, k)), (made2["arm"], made2["bus"], range(k, 6))):
        t = tips(arm_, bus_)
        for i in rng:
            x, y = PS.canvas_xy(s, 80.0, 50.0 + 6.0 * i)
            near = (np.hypot(t[:, 0] - x, t[:, 1] - y) < 0.002)
            assert near.any() and float(t[near, 2].min()) < PS.contact_z(s, x, y) - 0.001, i
    # a third call has nothing to paint, says so, and moves nothing
    r3, made3 = cli("--sim", "run", plan, "--layer", "0")
    check_result(r3, ok=True)
    assert r3["done"] and r3["strokes"] == 0 and not moved(made3["bus"]) and "--start 0" in r3["next"], r3
    # run_layer itself: the limit is looked at between strokes only
    arm, bus, pt = painter("thin", s)
    out = pt.run_layer(json.load(open(plan)), 0, max_s=1.0)
    assert out["ok"] and out["strokes"] == 1 and out["next_stroke"] == 1 and out["time_limit"], out


def test_10_progress_is_per_plan_layer_and_content():
    use_tool("thin")
    use_setup(make_setup(record=tool("thin")))
    reset_state()
    a = write_plan("prog_a.json", [("one", "red", bands(2, press=0.6)), ("two", "red", bands(2, v0=80.0, press=0.6))])
    pa = json.load(open(a))
    keys = {PS.progress_key(a, pa, 0), PS.progress_key(a, pa, 1), PS.progress_key(os.path.join(_TMP, "other.json"), pa, 0)}
    assert len(keys) == 3 and all(os.path.abspath(a) in k or "other.json" in k for k in keys)
    r, _ = cli("--sim", "run", a, "--layer", "one")                      # by name
    assert check_result(r, ok=True)["done"] and r["next_stroke"] == 2
    r, made = cli("--sim", "run", a, "--layer", "1", "--dry")            # the other layer starts at its first stroke
    assert r["start"] == 0 and r["stage"] == "planned" and not moved(made["bus"]), r
    r, _ = cli("--sim", "run", a, "--layer", "0", "--dry")
    assert r["start"] == 2
    # the same file made anew with another stroke: the old index does not count
    write_plan("prog_a.json", [("one", "red", bands(2, u0=70.0, press=0.6)), ("two", "red", bands(2, v0=80.0, press=0.6))])
    r, _ = cli("--sim", "run", a, "--layer", "0", "--dry")
    assert r["start"] == 0 and r["budget"]["strokes"] == 2, r
    r, _ = cli("--sim", "run", a, "--layer", "0", "--start", "1", "--dry")           # --start says otherwise
    assert r["start"] == 1 and r["budget"]["strokes"] == 1
    r, _ = cli("--sim", "run", a, "--layer", "0", "--start", "5", "--dry")
    check_result(r, ok=False, stage="options")


def test_11_a_simulated_run_never_writes_the_real_state_file():
    use_tool("thin")
    use_setup(make_setup(record=tool("thin")))
    reset_state()
    assert PS.state_path(False) == STATE_PATH and PS.state_path(True) == os.path.join(_TMP, "paint_state.sim.json")
    with open(STATE_PATH, "w") as f:
        f.write('{"paint": "black", "since_load_mm": 3.0, "marker": "the real arm\'s brush"}')
    before = open(STATE_PATH, "rb").read()
    plan = write_plan("state.json", [("two", "red", bands(2, press=0.6))])
    r, _ = cli("--sim", "run", plan, "--layer", "0")
    assert check_result(r, ok=True)["done"] and r["washes"] == 0         # (it did not think the brush carried black)
    assert open(STATE_PATH, "rb").read() == before
    sim = PS.load_state(sim=True)
    assert sim["paint"] == "red" and "timing" not in sim and "marker" not in sim, sim   # no simulated seconds in the time model
    r, _ = cli("--sim", "status")
    assert r["state"]["paint"] == "red" and r["state_file"].endswith(".sim.json")
    # without the variables the real files are the project's own; what a SIMULATED command writes lies outside
    # the tree, where `fd --sim` keeps its simulated tool (fd.sim_state_dir()): its state file, its pictures
    import fd
    root = os.path.realpath(tazuna_paths.ROOT) + os.sep
    with patched(os, "environ", {k: v for k, v in os.environ.items() if k not in ("FD_PAINT_STATE", "FD_PAINT_OUT")}):
        assert PS.state_path(False) == os.path.join(tazuna_paths.ROOT, "paint_state.json")
        assert PS.sim_dir() == fd.sim_state_dir() and not (os.path.realpath(PS.sim_dir()) + os.sep).startswith(root)
        assert PS.state_path(True) == os.path.join(fd.sim_state_dir(), "paint_state.sim.json")
        assert paintcli._out_dir(True) == os.path.join(fd.sim_state_dir(), "paint_out")
        assert paintcli._out_dir(False) == os.path.join(tazuna_paths.ROOT, "runs", time.strftime("%Y%m%d") + "_paint")
        for p in (PS.state_path(True), paintcli._out_dir(True)):
            assert not (os.path.realpath(p) + os.sep).startswith(root), p
    assert paintcli._out_dir(True) == OUT_DIR and paintcli._out_dir(False) == OUT_DIR         # ($FD_PAINT_OUT names the folder)
    # a dry run and a refusal leave even the simulated file alone
    stamp = open(PS.state_path(True), "rb").read()
    cli("--sim", "run", plan, "--layer", "0", "--dry")
    cli("--sim", "test", "--do", "bogus")
    assert open(PS.state_path(True), "rb").read() == stamp
    reset_state()


# ============================================================================= 3. the set-up
def test_12_a_set_up_is_checked_when_it_is_loaded():
    good = make_setup()
    assert PS.check_setup(copy.deepcopy(good)) and PS.check_setup(make_setup(brush=None))

    def broken(edit):
        s = copy.deepcopy(good)
        edit(s)
        return s

    cases = {
        "no canvas": (lambda s: s.pop("canvas"), "canvas"),
        "a canvas without a size": (lambda s: s["canvas"].pop("size_mm"), "size_mm"),
        "z0 in millimetres": (lambda s: s["contact"].update(z0=5.42), "contact.z0"),
        "no contact": (lambda s: s.pop("contact"), "contact"),
        "a slope of 36 (mm per m typed as m per m)": (lambda s: s["contact"].update(bx=36.2), "contact.bx"),
        "a well without xy": (lambda s: s["wells"]["red"].pop("xy"), "red"),
        "a well in millimetres": (lambda s: s["wells"]["red"].update(xy=[71.6, 265.1]), "red"),
        "water without the rinse dish": (lambda s: s["water"].pop("rinse"), "rinse"),
        "a third dish": (lambda s: s["water"].update(spare={"xy": [0.3, 0.2]}), "spare"),
        "an unknown step": (lambda s: s["paints"]["red"]["first"].append(["stir", "red"]), "stir"),
        "a step whose well is not there": (lambda s: s["paints"]["red"]["reload"].append(["scrub", "pink", 2]), "pink"),
        "a step into a dish that is none": (lambda s: s["paints"]["red"]["first"].insert(0, ["water", "pond", 1]), "pond"),
        "blotting without a strip": (lambda s: s.update(blot=None), "blot"),
        "a recipe without reload steps": (lambda s: s["paints"]["red"].pop("reload"), "reload"),
        "a press of 9 mm": (lambda s: s.update(press_mm=9.0), "press_mm"),
        "a photo pose of four joints": (lambda s: s.update(photo_q_deg=[0, 0, 0, 0]), "photo_q_deg"),
        "a square brush": (lambda s: s.update(brush={"kind": "square"}), "brush.kind"),
        "a round brush without modes": (lambda s: s.update(brush={"kind": "round", "load_mm": 140}), "modes"),
        "a broad mode narrower than the fine one": (lambda s: s["brush"]["modes"]["broad"].update(width_mm=2.0), "broad"),
        "a round brush without a load": (lambda s: s["brush"].pop("load_mm"), "load_mm"),
    }
    for what, (edit, word) in cases.items():
        try:
            PS.check_setup(broken(edit), "X.json")
        except PS.Refusal as e:
            assert e.stage == "setup" and word in str(e) and "X.json" in str(e) and e.next, (what, str(e))
        else:
            raise AssertionError(f"accepted a set-up with {what}")
    # through the command: a refusal with a text, for every sub-command, and nothing is built or moved
    use_tool("thin")
    use_setup(broken(cases["z0 in millimetres"][0]))
    for argv in (("status",), ("check",), ("look",), ("rest",), ("test", "--do", "rinse"), ("contact",), ("brush",),
                 ("run", "x.json", "--layer", "0"), ("preview", "x.json"), ("rinse",), ("locate",)):
        r, made = cli("--sim", *argv)
        check_result(r, ok=False, stage="setup")
        assert "contact.z0" in r["reason"] and not made, (argv, r)
    with open(SETUP_PATH, "w") as f:
        f.write("{ not json")
    r, _ = cli("--sim", "status")
    assert check_result(r, ok=False, stage="setup") and "cannot be read" in r["reason"]
    os.remove(SETUP_PATH)
    r, _ = cli("--sim", "run", "x.json", "--layer", "0")
    assert check_result(r, ok=False, stage="setup")
    # what is written is checked too: a broken set-up is never stored
    try:
        PS.save_setup(broken(cases["a well without xy"][0]))
    except PS.Refusal:
        assert not os.path.exists(SETUP_PATH)
    else:
        raise AssertionError("a broken set-up was written")
    use_setup()


def test_13_one_dish_set_ups():
    one = make_setup(record=tool("thin"))
    one["water"] = {"rinse": {"xy": [0.346, 0.192], "z_mm": 6.0}}
    PS.check_setup(one)                                    # recipes still say ["water", "clean", 1]: that is the same dish
    assert PS.one_dish(one) and PS.one_dish(make_setup())  # (the rig's file of 2026-11-13: two entries, 8 mm apart, one dish)
    two = make_setup()
    two["water"]["clean"]["xy"] = [0.30, 0.10]
    assert not PS.one_dish(two)
    arm, bus, pt = painter("thin", one)
    assert pt.dish("clean") is one["water"]["rinse"] and pt.dish("rinse") is one["water"]["rinse"]
    dabs, real = [], pt.br.dab
    pt.br.dab = lambda xy, z, **kw: (dabs.append((kw.get("what"), tuple(xy))), real(xy, z, **kw))[1]
    pt.water("clean")
    pt.rinse()
    assert [d[0] for d in dabs] == ["water (clean)", "rinse", "rinse (clean dish)"] and {d[1] for d in dabs} == {(0.346, 0.192)}, dabs
    assert pt.paint is None
    try:
        pt.dish("pond")
    except SafetyError:
        pass
    else:
        raise AssertionError("a dish that is none")
    use_tool("thin")
    use_setup(one)
    r, _ = cli("--sim", "status")
    check_result(r, ok=True, stage="status")
    assert r["one_dish"] is True and "水入れは1つ" in r["say"] and any("ONE water dish" in w for w in r["warnings"]), r
    r, made = cli("--sim", "check")
    assert check_result(r, ok=True)["ok"] and [i["name"] for i in r["items"] if i["kind"] == "dish"] == ["rinse"]
    assert any("ONE water dish" in w for w in r["warnings"]) and not moved(made["bus"])
    use_setup(two)
    r, _ = cli("--sim", "status")
    assert r["one_dish"] is False and "水入れは2つ" in r["say"]
    use_setup()


def test_14_the_brush_block_and_plans_for_another_brush():
    flat, rnd = make_setup(brush=None, record=tool("thin")), make_setup()
    b = PS.brush_of(flat)
    assert b == {"kind": "flat", "wide_mm": 9.0, "thin_mm": 2.5, "load_mm": 180.0, "stated": False}      # the morning, as it was
    b = PS.brush_of(rnd)
    assert b["kind"] == "round" and b["stated"] and b["load_mm"] == 140
    assert abs(PS.brush_width_mm(b, 0.6) - 2.7) < 1e-9 and abs(PS.brush_width_mm(b, 1.2) - 5.0) < 1e-9
    assert abs(PS.brush_width_mm(b, 0.9) - 3.85) < 1e-9 and PS.brush_width_mm(PS.brush_of(flat), 1.0) is None
    assert PS.use_mm(PS.brush_of(flat), 50.0, 1.2) == 50.0
    try:                                                   # the design kit of the evening, where this harness has it
        import roundbrush
        assert abs(roundbrush.width_of(0.9, 2.7, 5.0) - PS.brush_width_mm(b, 0.9)) < 1e-9
    except ImportError:
        pass
    # a plan says which kind of brush it was made for; another kind than the set-up's is refused with a text
    round_plan = json.load(open(write_plan("kind_round.json", [("a", "red", bands(2))])))
    flat_plan = json.load(open(write_plan("kind_flat.json", [("a", "red", bands(2))], brush={"kind": "flat"})))
    old_plan = json.load(open(write_plan("kind_none.json", [("a", "red", bands(2))], brush=None)))
    assert PS.plan_problems(round_plan, rnd) == [] and PS.plan_problems(flat_plan, flat) == []
    # a plan WITHOUT a brush block is a plan for the flat brush of the morning (those plans have none): it keeps
    # working with a flat set-up and is refused with a round one (it used to pass for any brush)
    assert PS.plan_brush_kind(old_plan) == "flat" and PS.plan_brush_kind(round_plan) == "round"
    assert PS.plan_problems(old_plan, flat) == []
    assert PS.plan_problems(old_plan, make_setup(brush={"kind": "flat", "wide_mm": 9.0, "thin_mm": 2.5, "load_mm": 180})) == []
    p = PS.plan_problems(old_plan, rnd)
    assert len(p) == 1 and "flat brush" in p[0] and "no \"brush\" block" in p[0] and "the set-up's brush is round" in p[0], p
    # (R20) such a plan does not SAY which brush it was made for, and no text advises to change the set-up's brush
    # kind so that a plan fits: which brush is on is what is on the gripper
    assert "does not say which brush it was made for" in p[0] and "fd paint brush" not in p[0] and "kaminarimon" not in p[0], p
    p = PS.plan_problems(round_plan, flat)
    assert len(p) == 1 and "round brush" in p[0] and "no \"brush\" block" in p[0], p
    # (R27) ... a set-up that does not say its brush at all is told where the command for that is (a whole one)
    assert "`fd paint status`" in p[0] and "fd paint brush --kind" not in p[0] and "..." not in p[0], p
    p = PS.plan_problems(flat_plan, rnd)
    assert len(p) == 1 and "flat brush" in p[0] and "fd paint brush" not in p[0], p
    p = PS.plan_problems(round_plan, make_setup(brush={"kind": "flat", "wide_mm": 9.0, "thin_mm": 2.5, "load_mm": 180}))
    assert len(p) == 1 and "the set-up's brush is flat" in p[0] and "fd paint" not in p[0], p
    # the lettering touch-up of the evening, shipped as the record of what ran (no block): refused on a round set-up,
    # and the refusal names the way that works: the plan kaminarimon.retrace() returns, as ONE command to type
    shipped = os.path.join(os.path.dirname(os.path.abspath(PS.__file__)), "paintings", "kaminarimon_letters_touchup.json")
    if os.path.exists(shipped):
        import subprocess
        assert hashlib.md5(open(shipped, "rb").read()).hexdigest() == "eac4a176d92a12aae2c17a6d5bbead4b"     # (byte for byte what ran)
        use_tool("thin")
        use_setup(make_setup(record=tool("thin")))
        r, _ = cli("--sim", "preview", shipped, "--out", os.path.join(_TMP, "touchup.png"))
        check_result(r, ok=False, stage="plan")
        assert "does not say which brush" in r["reason"] and "kaminarimon.retrace()" in r["reason"] and "fd paint brush" not in r["reason"] + r["next"], r
        cmd = re.search(r"`([^`]*kaminarimon[^`]*)`", r["reason"]).group(1)
        out = os.path.join(_TMP, "retrace_plan.json")
        assert cmd == PS.retrace_command() and " -c " in cmd and "|" not in cmd
        done_ = subprocess.run(shlex.split(PS.retrace_command(out)), capture_output=True, text=True, timeout=120,
                               env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        assert done_.returncode == 0 and os.path.exists(out), done_.stderr[-400:]
        r, _ = cli("--sim", "preview", out, "--out", os.path.join(_TMP, "touchup.png"))
        assert check_result(r, ok=True)["strokes"] == 13 and r["plan_brush"]["kind"] == "round", r      # the thirteen strokes of 雷門
        assert PS.plan_problems(json.load(open(shipped)), flat) == []       # (a set-up that says nothing: both mean the flat brush)
    for bad in ("round", ["round"], {"kind": "oval"}, {"wide_mm": 9}):          # a brush that is no object, or says no kind
        text_plan = dict(round_plan, brush=bad)
        for s_ in (rnd, flat):
            p = PS.plan_problems(text_plan, s_)
            assert len(p) == 1 and "\"brush\" must be an object" in p[0], (bad, p)
    for name in ("akabeko_plan.json", "akabeko_touchup.json"):                  # the morning's own plans, where they are
        path = os.path.join(tazuna_paths.ROOT, "runs", "20261113_paint", name)
        if os.path.exists(path):
            morning = json.load(open(path))
            assert "brush" not in morning and any("flat brush" in q for q in PS.plan_problems(morning, rnd)), name
    use_tool("thin")
    use_setup(flat)
    reset_state()
    r, made = cli("--sim", "run", os.path.join(_TMP, "kind_round.json"), "--layer", "0")
    check_result(r, ok=False, stage="plan")
    assert "round brush" in r["reason"] and not moved(made["bus"]) and r["brush"]["where"] == "not moved"
    # `fd paint brush`: show, store, clear (file only)
    r, made = cli("--sim", "brush")
    check_result(r, ok=True, stage="brush")
    assert r["brush"]["kind"] == "flat" and not r["saved"] and not made and "brush" not in json.load(open(SETUP_PATH))
    r, _ = cli("--sim", "brush", "--kind", "round", "--fine", "0.6:2.7", "--broad", "1.2:5.0", "--load-mm", "140", "--dry")
    assert check_result(r, ok=True, stage="planned")["brush"]["kind"] == "round" and "brush" not in json.load(open(SETUP_PATH))
    r, _ = cli("--sim", "brush", "--kind", "round", "--fine", "0.6:2.7", "--broad", "1.2:5.0", "--load-mm", "140")
    check_result(r, ok=True, stage="brush")
    assert r["saved"] and json.load(open(SETUP_PATH))["brush"] == {
        "kind": "round", "modes": {"fine": {"press": 0.6, "width_mm": 2.7}, "broad": {"press": 1.2, "width_mm": 5.0}},
        "load_mm": 140.0}
    r, made = cli("--sim", "run", os.path.join(_TMP, "kind_round.json"), "--layer", "0", "--dry")
    check_result(r, ok=True, stage="planned")              # now the plan fits
    r, _ = cli("--sim", "brush", "--load-mm", "120")       # one number changed, the rest kept
    assert json.load(open(SETUP_PATH))["brush"]["load_mm"] == 120.0 and json.load(open(SETUP_PATH))["brush"]["modes"]["broad"]["width_mm"] == 5.0
    for argv in (("--kind", "round", "--fine", "wide"), ("--kind", "round", "--broad", "1.2:1.0"), ("--kind", "round", "--wide-mm", "9"),
                 ("--kind", "flat", "--fine", "0.6:2.7"), ("--kind", "oval"), ("--load-mm", "3")):
        before = open(SETUP_PATH, "rb").read()
        r, _ = cli("--sim", "brush", *argv)
        assert check_result(r, ok=False)["stage"] in ("options", "setup") and open(SETUP_PATH, "rb").read() == before, (argv, r)
    r, _ = cli("--sim", "brush", "--kind", "flat", "--wide-mm", "12", "--load-mm", "200")
    assert json.load(open(SETUP_PATH))["brush"] == {"kind": "flat", "wide_mm": 12.0, "thin_mm": 2.5, "load_mm": 200.0}
    r, _ = cli("--sim", "brush", "--clear")
    assert r["saved"] and "brush" not in json.load(open(SETUP_PATH)) and r["brush"]["stated"] is False
    # a simulated command does not write the REAL rig's set-up (here: a stand-in for it, with the variable unset)
    stand_in = os.path.join(_TMP, "rig_setup.json")
    with open(stand_in, "w") as f:
        json.dump(flat, f)
    before = open(stand_in, "rb").read()
    env = {k: v for k, v in os.environ.items() if k != "FD_PAINT_SETUP"}
    with patched(PS, "SETUP_FILE", stand_in), patched(os, "environ", env):
        r, _ = cli("--sim", "brush", "--kind", "round")
        assert check_result(r, ok=False, stage="options") and "REAL rig" in r["reason"], r
        r, _ = cli("--sim", "brush", "--kind", "round", "--dry")
        assert r["ok"] and not r["saved"]
    assert open(stand_in, "rb").read() == before
    use_setup()


def test_15_a_contact_plane_holds_for_the_brush_it_was_measured_with():
    thin = tool("thin")
    s = make_setup(record=thin)
    assert s["contact"]["measured"] == {"tool": "brush_thin", "attached_at": "2026-11-13T17:17:45",
                                        "tip_offset": [0.016, -0.001, 0.056], "gripper_raw": 2041, "pitch_deg": 65.0,
                                        "gripper_goal_raw": 2016, "on_jaw": "moving_outer", "tip_measured": True}
    assert PS.contact_record_state(s, thin, 65.0) == {"recorded": True, "ok": True, "why": [], "measured": s["contact"]["measured"]}
    for what, t, pitch, word in (
            ("a tip 0.6 mm longer", tool("thin", tip_offset=[0.016, -0.001, 0.0566]), 65.0, "tool tip"),
            ("the tip of another mount", tool("thin", tip_offset=[-0.0051, -0.0025, 0.0604]), 65.0, "tool tip"),
            ("a jaw that stands 10 ticks elsewhere", tool("thin", gripper_raw=2051), 65.0, "gripper"),
            ("an open jaw", tool("thin", gripper_raw=2576), 65.0, "gripper"),
            ("no gripper position at all", tool("thin", gripper_raw=None), 65.0, "gripper"),
            ("another pitch", thin, 66.5, "pitch"),
            # the jaw and its squeeze are part of the brush state: first touch was 4 mm higher with the brush loose
            ("the squeeze taken off (`close --squeeze 0`)", tool("thin", gripper_goal_raw=None), 65.0, "not held by a squeeze now"),
            ("a squeeze 15 ticks harder", tool("thin", gripper_goal_raw=2001), 65.0, "squeezed to goal 2001"),
            ("recorded on the other jaw", tool("thin", on_jaw="fixed"), 65.0, "on_jaw = fixed"),
            # R19: which tool it is counts too (the record always named it; the gate did not look)
            ("the same numbers under another name", tool("thin", name="brush_b"), 65.0, "registered tool is 'brush_b'"),
            ("the same brush taped on again", tool("thin", attached_at="2026-11-14T08:00:00"), 65.0, "attached 2026-11-14T08:00:00")):
        st = PS.contact_record_state(s, t, pitch)
        assert st["recorded"] and not st["ok"] and len(st["why"]) == 1 and word in st["why"][0], (what, st)
        assert st["reason"].startswith("the contact plane of the set-up was measured for another brush state: ")
        assert "`fd paint contact --scrap-laid --save`" in st["next"] and st["next"].startswith("lay a scrap sheet"), st["next"]
    for t, pitch in ((tool("thin", tip_offset=[0.0163, -0.001, 0.056]), 65.0), (tool("thin", gripper_raw=2044), 65.0), (thin, 65.8),
                     (tool("thin", gripper_goal_raw=2019), 65.0)):
        assert PS.contact_record_state(s, t, pitch)["ok"]          # measuring noise is not another brush
    # the other way round: a plane measured with the brush LOOSE (the rig's tool file of 2026-11-13: jaw shut,
    # nothing recorded), used after `fd tool jaw mount moving_outer` + `close` put the squeeze on (the jaw ends
    # within 3 ticks, the tip is kept): that is another brush state too
    loose = tool("thin", **RIG_JAW)
    st = PS.contact_record_state(make_setup(record=loose), thin, 65.0)
    assert not st["ok"] and any("held shut by a squeeze now, it was not held" in w for w in st["why"]), st
    assert any("on_jaw = moving_outer now, it was None" in w for w in st["why"]) and len(st["why"]) == 2, st["why"]
    assert PS.contact_record_state(make_setup(record=loose), loose, 65.0)["ok"]
    old = PS.contact_record_state(make_setup(), thin, 65.0)        # no record at all: whose plane it is, nothing says
    assert not old["ok"] and not old["recorded"] and old["next"] == PS.ADOPT_NEXT and "no record of the brush state" in old["reason"]
    assert old["why"] and "warning" not in old
    # `fd paint run`: refused at "contact" with the command to run, before anything moves
    plan = write_plan("contact.json", [("a", "red", bands(2, press=0.6))])
    use_setup(s)
    reset_state()
    for t in (tool("thin", tip_offset=[0.016, -0.001, 0.058]), tool("thin", gripper_raw=2100, gripper_goal_raw=2075),
              tool("thin", **RIG_JAW)):
        use_tool(t)
        for extra in ((), ("--dry",)):
            r, made = cli("--sim", "run", plan, "--layer", "0", *extra)
            check_result(r, ok=False, stage="contact")
            assert "`fd paint contact --scrap-laid --save`" in r["next"] and r["next"].startswith("lay a scrap sheet"), r["next"]
            assert "measured for another brush state" in r["reason"], r
            assert not moved(made["bus"]) and r["brush"]["where"] == "not moved" and not made["arm"].is_torqued()
        r, _ = cli("--sim", "check")
        nxt = check_result(r, ok=False, stage="contact")["next"]
        if t.gripper_goal_raw is not None:
            assert "`fd paint contact --scrap-laid --save`" in nxt, nxt
        else:       # the squeeze is what is missing: `check` names the FIRST missing step, as `status` does (with
            #         the jaw held again the brush is the one the plane was measured with)
            assert nxt.startswith("fd tool jaw mount moving_outer") and "その前にすること" in r["say"], (nxt, r["say"])
            r2, _ = cli("--sim", "status")
            assert r2["next"] == nxt
        r, _ = cli("--sim", "preview", plan, "--out", os.path.join(_TMP, "c.png"))
        assert check_result(r, ok=False, stage="contact") and "contact plane" in r["reason"] and "`fd paint contact --scrap-laid --save`" in r["next"]
        # test STROKES run at the plane minus a press, as a layer does: refused too. A ladder (heights around the
        # plane: how a plane is looked for) is not, and says what it stands on
        r, made = cli("--sim", "test", "--do", "line:0.25,0.0:0.25,0.02", "--dry")
        check_result(r, ok=False, stage="contact")
        assert "Test strokes" in r["reason"] and r["next"].startswith("lay a scrap sheet") and not moved(made["bus"])
        r, made = cli("--sim", "test", "--do", "swatch:red:0.25,0.0")
        assert check_result(r, ok=False, stage="contact") and not moved(made["bus"]) and r["brush"]["where"] == "not moved"
        r, _ = cli("--sim", "test", "--ladder", "0.25,0.0", "--dry")
        assert check_result(r, ok=True) and any("another brush state" in w for w in r["warnings"]), r
    # the pitch: a canvas this brush can only paint at another pitch than the plane was measured at
    use_tool(thin)
    far = make_setup(record=thin)
    far["contact"]["measured"]["pitch_deg"] = 62.0
    use_setup(far)
    r, made = cli("--sim", "run", plan, "--layer", "0")
    assert check_result(r, ok=False, stage="contact") and "pitch" in r["reason"] and not moved(made["bus"])
    # the same brush as recorded: it paints. No record: it does not (test 39), and status names the way
    use_setup(s)
    r, _ = cli("--sim", "run", plan, "--layer", "0", "--dry")
    assert check_result(r, ok=True, stage="planned") and "warnings" not in r
    use_setup(make_setup())
    r, made = cli("--sim", "run", plan, "--layer", "0", "--dry")
    assert check_result(r, ok=False, stage="contact") and "no record of the brush state" in r["reason"] and not moved(made["bus"])
    r, _ = cli("--sim", "status")
    assert r["contact_record"]["recorded"] is False and any("brush state" in w for w in r["warnings"])
    assert r["next"].startswith("fd paint contact --adopt"), r["next"]
    use_setup()


# ============================================================================= 4. the paper height from two photos
EVENING_AT = ((0.166, -0.134), (0.166, 0.060), (0.289, -0.130), (0.289, 0.060), (0.227, -0.037))
EVENING_EYE = (3.3, 3.3, 9.0, 6.5, 5.0)                    # read by eye that evening, plan z in mm
EVENING_PLANE = {"z0": 5.42, "x0": 0.2225, "y0": -0.019, "bx": 36.2, "by": -6.2}       # mm, mm per metre


def _evening_setup():
    s = make_setup()
    s["canvas"]["origin_xy"] = [0.146, -0.131]             # where the canvas was at 17:40 (it was moved by 13 mm later)
    return s


def test_16_contact_read_reproduces_the_evening_of_2026_11_13():
    """IN-SAMPLE: the reading rule (paintskill.first_touch and the mark windows) was derived from these two
    photos. The test pins the rule to what the eye read that evening; it is no check on another sheet, brush
    or paint (the next real `fd paint contact` is: look at the picture it writes)."""
    if not all(os.path.exists(p) for p in EVENING_PHOTOS):
        raise Skip("the photos of the evening (runs/20261113_paint/173916_look.jpg, 174247_test.jpg) are not here")
    rig_cam = os.path.join(BACKUP, "camera_model.json")
    if os.path.exists(rig_cam):                            # the test's camera IS the rig's of that day
        rig = json.load(open(rig_cam))
        assert all(np.allclose(rig[k], CAMERA[k]) for k in ("f", "cx", "cy", "rvec", "tvec"))
    use_tool("thin")
    s = use_setup(_evening_setup())
    before = open(SETUP_PATH, "rb").read()
    argv = ["--sim", "contact", "--read", *EVENING_PHOTOS, "--zs", "12,9,6,3"]
    for x, y in EVENING_AT:
        argv += ["--at", f"{x},{y}"]
    r, made = cli(*argv)
    check_result(r, ok=True, stage="contact")
    assert not made and open(SETUP_PATH, "rb").read() == before and r["saved"] is False       # no arm, nothing written
    got = [p["first_touch_mm"] for p in r["places"]]
    assert r["read"] == 5 and all(p["read"] and p["inside_canvas"] for p in r["places"]), r["places"]
    for z, eye in zip(got, EVENING_EYE):
        assert abs(z - eye) <= 0.5, (got, EVENING_EYE)     # (the task allows 1.5 mm; the rule lands within 0.15)
    # every rung says whether it left a mark, and how long and wide it is
    marks = [[bool(q["mark"]) for q in p["rungs"]] for p in r["places"]]
    assert marks == [[False, False, False, True], [False, False, False, True], [False, True, True, True],
                     [False, False, True, True], [False, False, False, True]], marks
    for p in r["places"]:
        assert [q["plan_z_mm"] for q in p["rungs"]] == [12.0, 9.0, 6.0, 3.0] and -6.0 < p["sideways_mm"] < 2.0 and p["stray"] == 0
        for q in p["rungs"]:
            if q["mark"]:
                assert 1.0 <= q["length_mm"] <= 17.0 and 1.0 <= q["width_mm"] <= 4.0 and q["pieces"] in (1, 2), q
                assert -22.0 <= q["at_mm"][0] < q["at_mm"][1] <= 0.0, q       # beyond the end of the pull, towards the robot
    far_left, centre = r["places"][2]["rungs"], r["places"][4]["rungs"]
    assert far_left[1]["length_mm"] < 2.5 and far_left[1]["faint"]                          # the dot at 9 mm
    assert far_left[3]["length_mm"] > 12 and centre[3]["length_mm"] > 10 and centre[3]["width_mm"] < 2.2
    assert r["places"][3]["rungs"][2]["pieces"] == 2                                         # two dabs at 6 mm, far right
    # the plane, against the one written down that evening
    pl = r["plane"]
    at_centre = (pl["z0"] + pl["bx"] * (EVENING_PLANE["x0"] - pl["x0"]) + pl["by"] * (EVENING_PLANE["y0"] - pl["y0"])) * 1000.0
    assert abs(at_centre - EVENING_PLANE["z0"]) < 0.3, at_centre
    assert abs(pl["bx"] * 1000.0 - EVENING_PLANE["bx"]) < 3.0 and abs(pl["by"] * 1000.0 - EVENING_PLANE["by"]) < 2.0, pl
    assert pl["n"] == 5 and pl["max_residual_mm"] <= 1.0 and len(pl["residuals_mm"]) == 5
    x0, x1, y0, y1 = PS.canvas_rect(s)
    for x, y in ((x0, y0), (x0, y1), (x1, y0), (x1, y1)):  # the two planes agree over the whole canvas
        mine = (pl["z0"] + pl["bx"] * (x - pl["x0"]) + pl["by"] * (y - pl["y0"])) * 1000.0
        theirs = EVENING_PLANE["z0"] + EVENING_PLANE["bx"] * (x - EVENING_PLANE["x0"]) + EVENING_PLANE["by"] * (y - EVENING_PLANE["y0"])
        assert abs(mine - theirs) < 0.5, (x, y, mine, theirs)
    pic = cv2.imread(r["picture"])
    assert r["picture"].startswith(OUT_DIR) and pic is not None and pic.shape[0] > 600 and "3.3" in r["say"]
    assert r["record"] is None and any("no record beside the photos" in w for w in r["warnings"])
    # --save: these two photos have no record of the brush that painted the ladders (they are from before such
    # records were kept). Their plane is not stored, whatever tool is registered: it used to be stamped with the
    # tool file of the moment of reading, and a re-taped brush was certified with the plane of the old one
    for extra in (("--save",), ("--save", "--dry")):
        r, _ = cli(*argv, *extra)
        check_result(r, ok=False, stage="contact")
        assert "no record of the brush state" in r["reason"] and "fd paint contact --adopt" in r["next"], r
        assert open(SETUP_PATH, "rb").read() == before
    # With the record that `fd paint contact` now writes beside its photos (here: beside copies of the two photos,
    # for the brush as it was that evening), --read needs neither --at nor --zs, and --save stores the plane FOR
    # THE BRUSH OF THE RECORD, when that is still the brush on the gripper
    b_copy, a_copy = os.path.join(_TMP, "evening_before.jpg"), os.path.join(_TMP, "evening_after.jpg")
    shutil.copy(EVENING_PHOTOS[0], b_copy)
    shutil.copy(EVENING_PHOTOS[1], a_copy)
    evening_places = PS.contact_places(s, [tuple(p) for p in EVENING_AT], [12.0, 9.0, 6.0, 3.0], 0.010)
    rec_path = PS.write_contact_record(b_copy, PS.brush_state(tool("thin"), 65.0), evening_places, [12.0, 9.0, 6.0, 3.0], 0.010,
                                       "red", s, after_photo=a_copy, at="2026-11-13T17:42:47")
    assert rec_path == os.path.join(_TMP, "evening_before.json") and PS.load_contact_record(b_copy)["brush"]["tool"] == "brush_thin"
    r2, made = cli("--sim", "contact", "--read", b_copy, a_copy)
    check_result(r2, ok=True, stage="contact")
    assert not made and r2["record"] == rec_path and r2["painted_at"] == "2026-11-13T17:42:47" and "warnings" not in r2, r2
    assert [p["first_touch_mm"] for p in r2["places"]] == got and r2["painted_with"]["gripper_goal_raw"] == 2016
    # --dry: read and reported, nothing written (it used to store the plane all the same)
    pictures = set(os.listdir(OUT_DIR))
    r3, _ = cli("--sim", "contact", "--read", b_copy, a_copy, "--save", "--dry")
    check_result(r3, ok=True, stage="planned")
    assert r3["would_save"] is True and r3["saved"] is False and r3["picture"] is None and r3["dry"] is True, r3
    assert open(SETUP_PATH, "rb").read() == before and set(os.listdir(OUT_DIR)) == pictures and "--dry" in r3["say"]
    # the registered tool is not the brush of the record: another tool (re-taped the next morning), the same tool
    # with its jaw elsewhere, the same tool without its squeeze. The plane is not stored for any of them
    for other, word in ((tool("thin", name="brush_new", attached_at="2026-11-14T09:00:00", tip_offset=[0.012, 0.0, 0.064]), "brush_new"),
                        (tool("thin", gripper_raw=2101, gripper_goal_raw=2076), "gripper stands at raw 2101"),
                        (tool("thin", **RIG_JAW), "not held by a squeeze now"),
                        (tool("thin", attached_at="2026-11-14T09:00:00"), "attached 2026-11-14T09:00:00")):
        use_tool(other)
        for extra in (("--save",), ("--save", "--dry")):
            r, made = cli("--sim", "contact", "--read", b_copy, a_copy, *extra)
            check_result(r, ok=False, stage="contact")
            assert word in r["reason"] and "another brush state" in r["reason"] and not made, (word, r["reason"])
            assert "`fd paint contact --scrap-laid --save`" in r["next"] and open(SETUP_PATH, "rb").read() == before
        r, _ = cli("--sim", "contact", "--read", b_copy, a_copy)             # (reading alone is always possible)
        assert check_result(r, ok=True)["read"] == 5
    # other ladders than the record says were painted: read as asked, with a warning, and never stored
    use_tool("thin")
    r, _ = cli("--sim", "contact", "--read", b_copy, a_copy, "--zs", "11,8,5,2")
    assert any("other ladders than the record" in w for w in r["warnings"]), r.get("warnings")
    r, _ = cli("--sim", "contact", "--read", b_copy, a_copy, "--zs", "11,8,5,2", "--save")
    assert check_result(r, ok=False, stage="options") and "as they were painted" in r["reason"] and open(SETUP_PATH, "rb").read() == before
    # the brush of the record is on: the plane goes into the set-up WITH the brush state and the time of the ladders
    r, _ = cli("--sim", "contact", "--read", b_copy, a_copy, "--save")
    check_result(r, ok=True, stage="contact")
    c = json.load(open(SETUP_PATH))["contact"]
    assert r["saved"] and abs(c["z0"] - pl["z0"]) < 1e-9 and c["bx"] == pl["bx"] and c["previous"]["z0"] == 0.00542
    m = c["measured"]
    assert m["tip_offset"] == [0.016, -0.001, 0.056] and m["gripper_raw"] == 2041 and m["pitch_deg"] == 65.0 and len(m["places"]) == 5
    assert m["at"] == "2026-11-13T17:42:47" and m["stored_at"] > "2026-11-13T17:42:47" and m["gripper_goal_raw"] == 2016
    assert m["photos"] == [b_copy, a_copy] and "adopted" not in m
    assert PS.contact_record_state(PS.load_setup(), tool("thin"), 65.0)["ok"]
    # a record that is damaged is said, not ignored
    with open(rec_path, "w") as f:
        f.write('{"brush": 3}')
    r, _ = cli("--sim", "contact", "--read", b_copy, a_copy)
    assert check_result(r, ok=False, stage="contact") and "is not a record" in r["reason"], r
    use_setup()


def _synthetic(heights=HIDDEN, at=None, zs=(12.0, 9.0, 6.0, 3.0), **mark):
    s = make_setup()
    places = PS.contact_places(s, at, list(zs), 0.010)
    touch_of = heights if callable(heights) else hidden_paper(places, heights)
    return s, places, PS.read_contact(blank_photo(), ladder_photo(places, touch_of, **mark), CAM, s, places, TZ)


def test_17_contact_read_on_synthetic_photos():
    """In-sample for the reading rule (the photos are drawn with its inverse: mark_for)."""
    s, places, read = _synthetic()
    assert [p["name"] for p in places] == ["near-left", "near-right", "far-left", "far-right", "centre"]
    assert [p["at"] for p in places] == [[0.166, -0.121], [0.166, 0.073], [0.289, -0.121], [0.289, 0.073], [0.2275, -0.024]]
    assert read["usable"] and read["read"] == 5 and not read["warnings"], read
    # which rungs left a mark is read exactly; how long and wide they are, as far as the photo shows it
    assert [[q["mark"] for q in p["rungs"]] for p in read["places"]] == [
        [False, False, False, True], [False, False, False, True], [False, True, True, True], [False, False, True, True],
        [False, False, True, True]]
    for p, true, tol in zip(read["places"], HIDDEN, HIDDEN_TOL):
        assert abs(p["first_touch_mm"] - true) < tol, (p["name"], p["first_touch_mm"], true, p["why"])
        assert abs(p["sideways_mm"] + 3.0) < 0.6 and p["stray"] == 0
        for q in p["rungs"]:
            if q["mark"]:
                cover, width = mark_for(true - q["plan_z_mm"])
                # (as the photo shows them: its blur adds up to a camera pixel at every edge)
                assert -0.5 <= q["length_mm"] - max(1.6, 10.0 * cover) <= 1.8 and -0.3 <= q["width_mm"] - width <= 1.2, (p["name"], q)
                assert 2.0 <= -q["at_mm"][1] <= 4.5, q     # they were drawn 3 mm beyond the end of the pull
    truth = PS.fit_plane([(p["xy"][0], p["xy"][1], t / 1000.0) for p, t in zip(read["places"], HIDDEN)])
    pl = read["plane"]
    assert abs(pl["z0"] - truth["z0"]) < 0.0006 and abs(pl["bx"] - truth["bx"]) < 0.008 and abs(pl["by"] - truth["by"]) < 0.008, (pl, truth)
    # marks displaced otherwise (another camera fit, another brush): on the rung itself; 9 mm beyond and 4.5 beside
    for mark in ({"beyond_mm": -9.0, "side_mm": 0.0}, {"beyond_mm": 9.0, "side_mm": 4.5}, {"beyond_mm": 5.0, "side_mm": -5.5}):
        _s, _p, rd = _synthetic(**mark)
        assert rd["usable"] and rd["read"] == 5, (mark, [p["why"] for p in rd["places"]])
        assert all(abs(p["first_touch_mm"] - t) < tol for p, t, tol in zip(rd["places"], HIDDEN, HIDDEN_TOL)), (
            mark, [p["first_touch_mm"] for p in rd["places"]])

    def plane(x, y):                                       # a tilted sheet: 6.3 mm in the middle, rising away from the robot
        return 6.3 + 30.0 * (x - 0.2225) - 10.0 * (y + 0.006)

    # what cannot be read is not guessed
    _s, _p, rd = _synthetic(lambda x, y: 1.0)              # the paper lower than the lowest rung: no mark anywhere
    assert not rd["usable"] and rd["read"] == 0 and rd["plane"] is None and "no rung left a mark" in rd["why"], rd["why"]
    _s, _p, rd = _synthetic(lambda x, y: 14.0)             # higher than the highest: every rung a full line
    assert rd["read"] == 0 and "start it higher" in rd["places"][0]["why"]
    _s, _p, rd = _synthetic(lambda x, y: 12.3)             # just touching at the top rung: that is a reading
    assert rd["read"] == 5 and all(abs(p["first_touch_mm"] - 12.3) < 0.4 for p in rd["places"])
    # a ladder whose lower rung stayed dry (a drip, a dry brush)
    s = make_setup()
    places = PS.contact_places(s, None, [12.0, 9.0, 6.0, 3.0], 0.010)
    img = ladder_photo(places, plane)
    r3 = places[4]["rungs"][3]
    draw_patch(img, r3["x1"] - 0.022, r3["x0"] + 0.006, r3["y"] - 0.0095, r3["y"] + 0.003, colour=PAPER_BGR)       # wipe the lowest mark
    rd = PS.read_contact(blank_photo(), img, CAM, s, places, TZ)
    assert not rd["places"][4]["read"] and "did not" in rd["places"][4]["why"] and rd["read"] == 4 and rd["usable"]
    assert any("centre" in w for w in rd["warnings"])
    # one place 6 mm off the plane of the others: the residual says so, and that is not a measurement to paint with
    _s, _p, rd = _synthetic((3.3, 3.3, 3.3, 9.3, 3.3))
    assert rd["read"] == 5 and rd["plane"]["max_residual_mm"] > 1.5 and not rd["usable"] and "residual" in rd["why"], rd["plane"]
    # marks half way between two rungs, and nothing that tells which rungs left them (neither reading is a
    # ladder that touched from some rung on down)
    places = PS.contact_places(s, [(0.25, -0.02)], [12.0, 9.0, 6.0, 3.0], 0.010)
    img = blank_photo(seed=2)
    for k in (0, 2):
        draw_rung_mark(img, places[0]["rungs"][k], 11.0, 2.5, side_mm=6.0)
    rd = PS.read_contact(blank_photo(), img, CAM, s, places, TZ)
    assert not rd["places"][0]["read"] and "half way between" in rd["places"][0]["why"], rd["places"][0]
    # the same shift is no riddle when only ONE reading is such a ladder: marks 6 mm beside rungs 6 and 9 are
    # the marks of rungs 3 and 6 on the other side; marks 5.5 mm beside rungs 6 and 3 are theirs
    img = blank_photo(seed=2)
    for k in (1, 2):
        draw_rung_mark(img, places[0]["rungs"][k], 11.0, 2.5, side_mm=6.0)
    rd = PS.read_contact(blank_photo(), img, CAM, s, places, TZ)
    assert rd["places"][0]["read"] and [q["mark"] for q in rd["places"][0]["rungs"]] == [False, False, True, True], rd["places"][0]
    assert abs(rd["places"][0]["sideways_mm"] + 6.0) < 0.6
    img = blank_photo(seed=2)
    for k in (2, 3):
        draw_rung_mark(img, places[0]["rungs"][k], 11.0, 2.5, side_mm=-5.5)
    rd = PS.read_contact(blank_photo(), img, CAM, s, places, TZ)
    assert rd["places"][0]["read"] and [q["mark"] for q in rd["places"][0]["rungs"]] == [False, False, True, True], rd["places"][0]
    assert abs(rd["places"][0]["sideways_mm"] + 5.5) < 0.6
    # a place that is not paper in the first photo, photos that differ all over, a ladder outside the canvas
    dark = blank_photo()
    draw_patch(dark, 0.20, 0.30, -0.06, 0.04, colour=(40, 40, 40))
    rd = PS.read_contact(dark, img, CAM, s, places, TZ)
    assert not rd["places"][0]["read"] and "no bright paper" in rd["places"][0]["why"]
    rd = PS.read_contact(blank_photo(), blank_photo(seed=5, paper=(150, 150, 150)), CAM, s, places, TZ)       # only darker: no marks
    assert not rd["places"][0]["read"] and "no rung left a mark" in rd["places"][0]["why"]
    outside = PS.contact_places(s, [(0.345, 0.0)], [12.0, 9.0, 6.0, 3.0], 0.010)
    rd = PS.read_contact(blank_photo(), ladder_photo(outside, plane), CAM, s, outside, TZ)
    assert not outside[0]["inside_canvas"] and any("measure where the picture will be" in w for w in rd["warnings"])
    # the plane itself
    assert PS.fit_plane([(0.1, 0.0, 0.003), (0.2, 0.0, 0.004)]) is None
    assert PS.fit_plane([(0.1, 0.0, 0.003), (0.2, 0.0, 0.004), (0.3, 0.001, 0.005)]) is None       # on one line
    fp = PS.fit_plane([(0.1, 0.0, 0.003), (0.3, 0.0, 0.005), (0.1, 0.2, 0.001), (0.3, 0.2, 0.003)])
    assert abs(fp["bx"] - 0.01) < 1e-6 and abs(fp["by"] + 0.01) < 1e-6 and abs(fp["z0"] - 0.003) < 1e-6 and fp["max_residual_mm"] < 1e-6
    assert (fp["x0"], fp["y0"]) == (0.2, 0.1)
    # the rule in numbers
    rungs = [{"plan_z_mm": 12.0, "mark": False}, {"plan_z_mm": 9.0, "mark": False},
             {"plan_z_mm": 6.0, "mark": True, "length_mm": 4.0, "width_mm": 1.6}, {"plan_z_mm": 3.0, "mark": True, "length_mm": 11, "width_mm": 3}]
    assert PS.first_touch(rungs, 10.0)[0] == 6.3           # dabs over 40 % of the pull: just touching, + 0.75 * 0.4
    rungs[2].update(length_mm=11.0, width_mm=1.2)
    assert PS.first_touch(rungs, 10.0)[0] == 8.0           # a line 1.2 mm wide over the whole pull: pressed by 0.8 + 1.2
    rungs[2].update(width_mm=4.0)
    assert PS.first_touch(rungs, 10.0)[0] == 8.9           # never up to the dry rung above
    rungs[2].update(length_mm=8.75, width_mm=1.0)          # between dabs and a line: half of each reading
    assert abs(PS.first_touch(rungs, 10.0)[0] - (6.0 + 0.5 * 0.75 * 0.875 + 0.5 * 1.8)) < 0.006
    for depth in (0.0, 0.2, 0.5, 0.8, 1.5, 1.8, 2.2, 2.6):                # mark_for is that rule backwards
        cover, width = mark_for(depth)
        rungs[2].update(length_mm=10.0 * cover, width_mm=width)
        assert abs(PS.first_touch(rungs, 10.0)[0] - 6.0 - depth) < 0.011, depth


def _camera_for(places, touch_of):
    """A camera for `fd paint contact`: the first photo shows blank paper, every later one the ladders as they
    come out on paper whose first-touch height is touch_of."""
    shots = []

    def photo():
        shots.append(len(shots))
        return blank_photo() if len(shots) == 1 else ladder_photo(places, touch_of)

    photo.shots = shots
    return photo


def test_18_contact_with_the_arm_measures_a_hidden_plane_and_stores_it():
    thin = use_tool("thin")
    s = use_setup(make_setup())
    reset_state()

    places = PS.contact_places(s, None, [12.0, 9.0, 6.0, 3.0], 0.010)
    plane = hidden_paper(places)                           # the paper: every ladder at its own hidden height
    # --dry: the places and their reach; nothing moves, no photo
    cam = _camera_for(places, plane)
    r, made = cli("--sim", "contact", "--paint", "red", "--dry", camera=cam)
    check_result(r, ok=True, stage="planned")
    assert len(r["ladders"]) == 5 and not moved(made["bus"]) and not cam.shots and r["pitch"] == 65.0, r
    assert "--scrap-laid" in r["next"] and "scrap sheet" in r["next"] and "捨て紙" in r["say"]
    # the simulated arm has no camera: the ladders are painted, nothing is read, and --save is refused beforehand
    r, made = cli("--sim", "contact", "--scrap-laid", "--paint", "red", "--at", "0.25,-0.02")
    check_result(r, ok=True, stage="contact")
    assert r["read"] is None and r["ladders"][0]["rungs_done"] == 4 and "読み取りはしていません" in r["say"], r
    assert r["sheet"] is None and any("the sheet was not looked at" in w for w in r["warnings"]) and r["record"] is None
    r, made = cli("--sim", "contact", "--scrap-laid", "--paint", "red", "--save")
    assert check_result(r, ok=False, stage="camera") and not moved(made["bus"])
    # with a camera: before photo, load, five ladders, after photo, the reading, the picture, the set-up
    cam = _camera_for(places, plane)
    r, made = cli("--sim", "contact", "--scrap-laid", "--paint", "red", "--save", camera=cam)
    check_result(r, ok=True, stage="contact")
    assert len(cam.shots) == 2 and r["read"] == 5 and r["usable"] and r["saved"] and r["paint"] == "red", r
    assert all(os.path.exists(p) and p.startswith(OUT_DIR) for p in (r["photos"]["before"], r["photos"]["after"], r["picture"]))
    assert r["sheet"]["ok"] and all(p["bare"] and p["not_paper_pct"] == 0.0 for p in r["sheet"]["places"]), r["sheet"]
    # the record beside the two photos: which brush painted these ladders, where, at which heights
    assert r["record"] == os.path.splitext(r["photos"]["before"])[0] + ".json" and os.path.exists(r["record"])
    rec = json.load(open(r["record"]))
    assert rec["brush"] == PS.brush_state(thin, 65.0) and rec["zs_mm"] == [12.0, 9.0, 6.0, 3.0] and rec["ladder_len_m"] == 0.010
    assert [(p["name"], p["at"], p["rungs"], p["rungs_done"]) for p in rec["places"]] == [(p["name"], p["at"], 4, 4) for p in places]
    assert rec["photos"] == {"before": r["photos"]["before"], "after": r["photos"]["after"]} and rec["paint"] == "red" and rec["sim"]
    assert rec["canvas"]["origin_xy"] == [0.146, -0.118] and rec["at"] <= time.strftime("%Y-%m-%dT%H:%M:%S")
    assert r["reread"] == f"fd paint contact --read {r['photos']['before']} {r['photos']['after']} --save"
    for p, true, tol in zip(r["places"], HIDDEN, HIDDEN_TOL):
        assert abs(p["first_touch_mm"] - true) < tol, (p["name"], p["first_touch_mm"], true)
    assert r["brush"]["where"] == "photo pose" and r["brush"]["height_mm"] > 50 and "保存しました" in r["say"]
    # the arm really made every rung at its plan height (the lowest the tip came near each rung), highest first
    t = tips(made["arm"], made["bus"])
    order = []
    for p in places:
        for q in p["rungs"]:
            near = np.hypot(t[:, 0] - (q["x0"] + q["x1"]) / 2.0, t[:, 1] - q["y"]) < 0.004
            assert near.any(), (p["name"], q)
            assert abs(float(t[near, 2].min()) * 1000.0 - q["plan_z_mm"]) < 2.0, (p["name"], q, float(t[near, 2].min()))
            order.append(int(np.argmax(near)))
    assert order == sorted(order)
    c = json.load(open(SETUP_PATH))["contact"]
    assert c["measured"]["tip_offset"] == [0.016, -0.001, 0.056] and c["measured"]["gripper_raw"] == 2041
    assert c["measured"]["pitch_deg"] == 65.0 and c["measured"]["tool"] == "brush_thin" and len(c["measured"]["photos"]) == 2
    assert c["measured"]["gripper_goal_raw"] == 2016 and c["measured"]["on_jaw"] == "moving_outer"      # the jaw was held
    assert c["measured"]["at"] == rec["at"] and c["measured"]["stored_at"] >= rec["at"] and "adopted" not in c["measured"]
    truth = PS.fit_plane([(p["xy"][0], p["xy"][1], t / 1000.0) for p, t in zip(r["places"], HIDDEN)])
    assert abs(c["z0"] - truth["z0"]) < 0.0006 and abs(c["bx"] - truth["bx"]) < 0.008 and c["measured"]["max_residual_mm"] <= 1.5
    assert PS.load_state(sim=True)["paint"] == "red"
    # painting goes on with that plane, and only with the brush it was measured for
    plan = write_plan("after_contact.json", [("a", "red", bands(1, press=0.6))])
    r, _ = cli("--sim", "run", plan, "--layer", "0", "--dry")
    assert check_result(r, ok=True) and "warnings" not in r
    use_tool(tool("thin", tip_offset=[0.016, -0.001, 0.059]))
    r, _ = cli("--sim", "run", plan, "--layer", "0", "--dry")
    assert check_result(r, ok=False, stage="contact")
    use_tool(thin)
    use_setup(make_setup(record=thin))                     # (the plane of the rig again)
    # default paint: what is on the brush (red now); refusals before anything moves
    r, made = cli("--sim", "contact", "--dry")
    assert r["paint"] == "red"
    for argv, stage in ((("--zs", "3,6,9"), "options"), (("--zs", "12,9,6,-9"), "options"), (("--zs", "x"), "options"),
                        (("--at", "0.9,0.0"), "reach"), (("--at", "there"), "options"), (("--paint", "pink"), "paint"),
                        (("--at", "0.25,0.0", "--save"), "options"), (("--ladder-len", "0.2"), "options")):
        r, made = cli("--sim", "contact", *argv)
        check_result(r, ok=False, stage=stage)
        assert not moved(made["bus"]) and r["brush"]["where"] == "not moved", (argv, r)
    r, made = cli("--sim", "contact", "--at", "0.25,0.0", "--at", "0.2,0.0", "--at", "0.3,0.0", "--save", camera=cam)
    assert check_result(r, ok=False, stage="options") and "at least 4 places" in r["reason"] and not moved(made["bus"])
    # a ladder outside the canvas is painted, with the warning; one in a keep-out is refused
    r, made = cli("--sim", "contact", "--at", "0.345,0.0", "--zs", "13,10,7,4", "--dry")
    assert check_result(r, ok=True) and any("measure where the picture will be" in w for w in r["warnings"]), r
    r, made = cli("--sim", "contact", "--at", "0.25,-0.17", "--dry")           # x >= 0.19, y <= -0.135: the cables
    assert check_result(r, ok=False, stage="reach") and "keep-out" in r["reason"] and not moved(made["bus"]), r
    # a reading that is no measurement is not stored, and the result says why
    before = open(SETUP_PATH, "rb").read()
    r, made = cli("--sim", "contact", "--scrap-laid", "--paint", "red", "--save", camera=_camera_for(places, lambda x, y: 1.0))
    check_result(r, ok=False, stage="contact")
    assert r["saved"] is False and "no rung left a mark" in r["reason"] and open(SETUP_PATH, "rb").read() == before
    assert r["brush"]["where"] == "photo pose" and "picture" in r["next"] and "--scrap-laid" in r["next"]
    use_setup()


# ============================================================================= 5. photos of the table
def test_19_relocate_wells_and_canvas_view_on_synthetic_photos():
    s = make_setup()

    def palette(shift, skip=()):
        return palette_photo(s, shift, skip)

    for shift in ((0.0, 0.0), (0.012, 0.0), (-0.008, 0.009), (0.02, -0.014)):      # 12 mm on 2026-11-13; up to 3 cm
        r = PS.relocate_wells(palette(shift), CAM, s, TZ)
        assert r["missing"] == [] and r["blobs"] == 10, (shift, r)
        assert np.allclose(r["shift_mm"], np.array(shift) * 1000.0, atol=1.5), (shift, r["shift_mm"])
        for name, w in s["wells"].items():
            got = r["wells"][name]["xy"]
            assert math.hypot(got[0] - w["xy"][0] - shift[0], got[1] - w["xy"][1] - shift[1]) < 0.002, (shift, name, got)
        assert len({tuple(w["xy"]) for w in r["wells"].values()}) == 10            # every well its own blob
    r = PS.relocate_wells(palette((0.02, -0.014), skip=("yellow", "black")), CAM, s, TZ)
    assert sorted(r["missing"]) == ["black", "yellow"] == sorted(r["not_located"]) and r["blobs"] == 8 and r["found"] == 8
    for name in ("black", "yellow"):                       # a blob that is gone: the well KEEPS its place, not located
        got = dict(r["wells"][name])
        want = (s["wells"][name]["xy"][0] + 0.02, s["wells"][name]["xy"][1] - 0.014)       # (where the palette's move puts it)
        assert math.hypot(got["expected_xy"][0] - want[0], got["expected_xy"][1] - want[1]) < 0.002, (name, got)
        del got["expected_xy"]
        assert got == {"xy": s["wells"][name]["xy"], "moved_mm": 0.0, "located": False, "by": None}, got
    assert all(w["located"] and w["by"] == "blob" for n_, w in r["wells"].items() if n_ not in ("black", "yellow"))
    r = PS.relocate_wells(np.full((1080, 1920, 3), 62, np.uint8), CAM, s, TZ)       # no palette in the photo: nothing moves
    assert len(r["missing"]) == 10 == len(r["not_located"]) and r["shift_mm"] == [0.0, 0.0] and r["wells"]["red"]["xy"] == s["wells"]["red"]["xy"]
    # R3: a palette that was TURNED as well as pushed. By the shift alone the wells at its ends lost their blobs
    # (they were then moved by the median shift: beside the paint); the fit of shift and turn finds all ten
    for shift, turn in (((0.008, -0.004), 6.0), ((-0.006, 0.005), -5.0), ((0.0, 0.0), 4.0)):
        r = PS.relocate_wells(palette_photo(s, shift, turn_deg=turn), CAM, s, TZ)
        want = palette_places(s, shift, turn)
        assert r["missing"] == [] == r["not_located"] and r["found"] == 10 and abs(r["turn_deg"] - turn) < 0.6, (shift, turn, r)
        for name in s["wells"]:
            got = r["wells"][name]
            assert got["by"] == "blob" and math.hypot(got["xy"][0] - want[name][0], got["xy"][1] - want[name][1]) < 0.002, (turn, name, got)
        assert max(w["moved_mm"] for w in r["wells"].values()) > 14.0 or turn == 4.0          # (further than the shift alone reaches)
    assert PS.locate_usable(PS.relocate_wells(palette_photo(s, (0.008, -0.004), turn_deg=6.0), CAM, s, TZ)) == (True, None)
    # turned, and two blobs gone: those two keep their places; nothing is put on the bare palette
    r = PS.relocate_wells(palette_photo(s, (0.008, -0.004), skip=("magenta", "green"), turn_deg=6.0), CAM, s, TZ)
    assert sorted(r["not_located"]) == ["green", "magenta"] and r["found"] == 8
    assert all(r["wells"][n_]["xy"] == s["wells"][n_]["xy"] and not r["wells"][n_]["located"] for n_ in ("green", "magenta"))
    # the fit itself: shift and turn of points that were moved rigidly; too few or too close together: the shift alone
    src = np.array([[0.0, 0.0], [0.1, 0.0], [0.1, 0.2], [0.0, 0.2]])
    a_ = math.radians(8.0)
    dst = src @ np.array([[math.cos(a_), -math.sin(a_)], [math.sin(a_), math.cos(a_)]]).T + [0.01, -0.02]
    move, deg = PS._palette_fit(src, dst)
    assert abs(deg - 8.0) < 1e-6 and np.allclose(move(src), dst, atol=1e-9) and np.allclose(move([0.05, 0.1]), dst.mean(axis=0), atol=1e-9)
    move, deg = PS._palette_fit(src[:2], dst[:2])
    assert deg == 0.0 and np.allclose(move(src[:2]).mean(axis=0), dst[:2].mean(axis=0))
    assert PS._palette_fit(src * 0.2, dst * 0.2)[1] == 0.0                                 # (all within 4 cm of one another)
    # a real photo of 2026-11-13 (17:39): the violet and the magenta blob run into ONE dark piece there. Each well
    # took "the nearest blob" then, and both were moved onto the middle between them (17 and 28 mm off); now a
    # well without a blob of its own is put where the palette's move puts it, when the photo shows paint there
    if os.path.exists(EVENING_PHOTOS[0]):
        photo = cv2.imread(EVENING_PHOTOS[0])
        r = PS.relocate_wells(photo, CAM, BASE, TZ)
        assert sorted(r["missing"]) == ["magenta", "violet"] and abs(r["shift_mm"][0]) < 2.0 and abs(r["shift_mm"][1]) < 2.0, r
        assert all(w["moved_mm"] < 6.0 for w in r["wells"].values()), r["wells"]
        assert len({tuple(w["xy"]) for w in r["wells"].values()}) == len(BASE["wells"])
        assert r["not_located"] == [] and [r["wells"][n_]["by"] for n_ in ("magenta", "violet")] == ["paint", "paint"]
        assert all(paint_share(photo, w["xy"]) > 0.8 for w in r["wells"].values())             # every well on paint
    # canvas_view: a mark at a canvas place lands at that place of the view
    z0 = s["contact"]["z0"]
    for plane_z, kw in ((z0, {}), (TZ, {"z": TZ})):
        img = blank_photo()
        for u, v in ((150.0, 40.0), (20.0, 130.0)):
            x, y = PS.canvas_xy(s, u, v)
            draw_patch(img, x - 0.003, x + 0.003, y - 0.003, y + 0.003, colour=(20, 20, 20), z=plane_z)
        view = PS.canvas_view(img, CAM, s, px_per_mm=4.0, margin_mm=12.0, **kw)
        assert view.shape[:2] == (int((153 + 24) * 4), int((224 + 24) * 4))
        n, _lab, stats, cent = cv2.connectedComponentsWithStats((view.max(axis=2) < 90).astype(np.uint8))
        got = sorted((float(cent[k][0]), float(cent[k][1])) for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] > 100)
        want = sorted(((u + 12.0) * 4.0, (v + 12.0) * 4.0) for u, v in ((150.0, 40.0), (20.0, 130.0)))
        assert len(got) == 2 and np.allclose(got, want, atol=2.5), (got, want)
    # the strokes of a plan over a photo: on the photo, inside the canvas as the camera sees it
    plan = json.load(open(write_plan("overlay.json", [("a", "red", bands(3))])))
    img = blank_photo()
    over = PS.overlay_plan(img, CAM, s, plan)
    changed = np.argwhere(np.abs(over.astype(int) - img.astype(int)).max(axis=2) > 40)
    quad = CAM.project([[x, y, z0] for x, y in ((0.146, -0.118), (0.146, 0.106), (0.299, 0.106), (0.299, -0.118))])
    assert over.shape == img.shape and len(changed) > 500
    assert changed[:, 1].min() >= quad[:, 0].min() - 8 and changed[:, 1].max() <= quad[:, 0].max() + 8


# ============================================================================= 6. reach, rest
def test_20_check_names_what_is_out_of_reach():
    use_tool("thin")
    use_setup(make_setup(record=tool("thin")))
    r, made = cli("--sim", "check")
    check_result(r, ok=True, stage="check")
    assert not moved(made["bus"]) and not made["arm"].is_torqued() and r["problems"] == [] and r["pitch_canvas"] == 65.0
    kinds = [i["kind"] for i in r["items"]]
    assert kinds.count("well") == len(BASE["wells"]) and kinds.count("dish") == 2 and kinds.count("blot") == 1 and kinds.count("canvas") == 5
    for i in r["items"]:
        assert i["ok"] and 15.0 <= i["pitch"] <= 90.0 and i["wrist_margin_deg"] > 0.0, i
        assert 20.0 <= i["wrist_flex_deg"][0] <= i["wrist_flex_deg"][1] <= 97.0 and i["r_mm"] > 100 and i["heights_mm"]["travel"] in (30, 65)
    red = next(i for i in r["items"] if i["name"] == "red")
    assert red["needed_by"][:1] == ["red"] and red["heights_mm"] == {"low": 2.5, "travel": 65}
    # the evening's second dish: 0.44 m from the pan axis, about 0.40 are in reach
    far = make_setup(record=tool("thin"))
    far["water"]["clean"]["xy"] = [0.355, 0.30]
    use_setup(far)
    r, made = cli("--sim", "check")
    check_result(r, ok=False, stage="check")
    dish = next(i for i in r["items"] if i["name"] == "clean")
    assert r["problems"] == ["clean"] and not dish["ok"] and dish["r_mm"] == 436 and 380 <= dish["reach_mm"] < 436, dish
    assert "clean dish" in r["reason"] and "nearer to the robot" in r["next"] and "ONE dish" in r["next"] and "clean" in r["say"]
    assert not moved(made["bus"])
    r, made = cli("--sim", "run", write_plan("far.json", [("a", "red", bands(1))]), "--layer", "0", "--dry")
    assert r["ok"]                                         # (run --dry looks at the canvas; `check` is where the dishes are judged)
    # a well nobody uses may be out of reach; one that a recipe needs may not; neither may lie in a keep-out
    s = make_setup(record=tool("thin"))
    s["wells"]["magenta"]["xy"] = [0.30, 0.34]
    use_setup(s)
    r, _ = cli("--sim", "check")
    assert check_result(r, ok=True)["problems"] == [] and any("magenta" in w for w in r["warnings"])
    s["wells"]["red"]["xy"] = [0.30, 0.34]
    use_setup(s)
    r, _ = cli("--sim", "check")
    assert check_result(r, ok=False)["problems"] == ["red"] and "the well 'red'" in r["reason"]
    s = make_setup(record=tool("thin"))
    s["wells"]["black"]["xy"] = [0.25, -0.16]              # among the cables
    use_setup(s)
    r, _ = cli("--sim", "check")
    assert r["problems"] == ["black"] and "keep-out 'cables'" in r["reason"] and "out of the keep-out" in r["next"], r
    # the brush taped far to the side: the near part of the sheet is out of reach, and `check` says it first
    use_tool("side")
    use_setup(make_setup())
    r, _ = cli("--sim", "check")
    check_result(r, ok=False, stage="check")
    assert {"near-left", "near-right"} <= set(r["problems"]) and r["pitch_canvas"] is None, r["problems"]
    r, made = cli("--sim", "run", os.path.join(_TMP, "far.json"), "--layer", "0")
    assert check_result(r, ok=False, stage="reach") and not moved(made["bus"])
    # without --sim the real arm is asked for its limits; locked (as these tests are), the simulated arm stands in
    use_tool("thin")
    use_setup(make_setup(record=tool("thin")))
    r, made = cli("check")
    check_result(r, ok=True, stage="check")
    assert "hardware is locked" in r["limits_from"] and "sim" not in r and "シミュレーション" in r["say"], r
    use_setup()


def _rest_ok(s, arm, margin=0.03):
    """Is the brush of this (folded) arm, and the place it sinks to, off the canvas by the margin?"""
    q = arm.read_q()
    tip = K.fk(q, K.tool_offset(*arm.tool.tip_offset))[:3, 3]
    _lands, low = PS.sink_of(arm, arm.tool, q)
    return min(PS.canvas_distance(s, tip[0], tip[1]), PS.canvas_distance(s, *low)) >= margin - 0.002


def test_21_rest_never_leaves_the_brush_over_the_canvas():
    s = use_setup(make_setup())
    for kind in ("thin", "flat", "side"):
        use_tool(kind)
        # the tool's own rest pose (pan 20) is over the picture, and so is where the brush would sink to
        arm, bus = sim_arm(kind)
        own = PS.rest_pose(arm, arm.tool, TL.REST_PAN_DEG, 0.04)
        assert PS.canvas_distance(s, *own["tip_xy"]) < 0.0 or PS.canvas_distance(s, *own["low_xy"]) < 0.03, (kind, own)
        r, made = cli("--sim", "rest", "--dry")
        check_result(r, ok=True, stage="planned")
        assert not moved(made["bus"]) and r["pose"]["clear_of_canvas_mm"] >= 30.0 and r["chosen"] == "nearest to the photo pose", r
        r, made = cli("--sim", "rest")
        check_result(r, ok=True, stage="rest")
        arm = made["arm"]
        assert r["torque"] == "off" and not arm.is_torqued() and r["brush"]["where"] == "rest" and r["moved"], r
        assert _rest_ok(s, arm) and r["brush"]["clear_of_canvas_mm"] >= 30.0 and r["brush"]["height_mm"] > 30.0, (kind, r["brush"])
        assert abs(r["pan_deg"] - PHOTO_Q[0]) <= 15.0 and r["clearance_mm"] == 40 and TL.in_rest_pose(arm, arm.tool)[0]
        assert r["pose"]["lands_xy"] is None or PS.canvas_distance(s, *r["pose"]["lands_xy"]) >= 0.03
        assert "キャンバス" in r["say"] and "トルクOFF" in r["say"]
        h = tips(arm, made["bus"])[:, 2] - TZ
        assert float(h.min()) >= TL.MIN_FREE_HEIGHT           # on the way the brush stayed up
        r, made = cli("--sim", "rest", "--keep-torque")
        assert check_result(r, ok=True)["torque"] == "on" and made["arm"].is_torqued() and _rest_ok(s, made["arm"])
        # from the photo pose, torque on, a wet brush: the same fold
        def at_photo(arm_):
            place(arm_, np.radians(PHOTO_Q))
            arm_.torque_on()
            arm_.bus.sleep(0.6)
        r, made = cli("--sim", "rest", prep=at_photo)
        assert check_result(r, ok=True)["torque"] == "off" and _rest_ok(s, made["arm"])
        # a pan that puts the brush over the picture is refused, and nothing moves
        for pan in ("0", "20", "-30"):
            r, made = cli("--sim", "rest", "--pan", pan)
            check_result(r, ok=False, stage="rest")
            assert "canvas" in r["reason"] and not moved(made["bus"]) and not made["arm"].is_torqued(), (kind, pan, r)
        r, made = cli("--sim", "rest", "--pan", "-100", "--clearance", "0.04")            # what was done by hand that evening
        assert check_result(r, ok=True)["pan_deg"] == -100.0 and r["chosen"] == "asked" and _rest_ok(s, made["arm"])
    # the set-up may name the fold; one that is over the picture is refused there too
    use_tool("thin")
    use_setup(make_setup(rest={"pan_deg": 95.0, "clearance_m": 0.05}))
    r, made = cli("--sim", "rest")
    assert check_result(r, ok=True)["pan_deg"] == 95.0 and r["chosen"] == "set-up" and r["clearance_mm"] == 50 and _rest_ok(s, made["arm"])
    use_setup(make_setup(rest={"pan_deg": 10.0}))
    r, made = cli("--sim", "rest")
    assert check_result(r, ok=False, stage="rest") and "set-up" in r["next"] and not moved(made["bus"])
    # a sheet so large that no fold is off it: refused with a text, the torque stays as it was
    big = make_setup()
    big["canvas"] = {"origin_xy": [-0.26, -0.29], "size_mm": [580, 600], "pitch_deg": 65.0}       # 30 cm around the pan axis
    use_setup(big)
    r, made = cli("--sim", "rest")
    check_result(r, ok=False, stage="rest")
    assert "no folded pose" in r["reason"] and not moved(made["bus"]) and "torque on" in r["next"], r
    # (R27: with the canvas still in the set-up a plain `fd tool rest` refuses the same fold: the word is named)
    assert "`fd tool rest --over-canvas`" in r["next"] and "then `fd tool rest`" not in r["next"], r["next"]
    # no tool: there is nothing to keep off the picture
    TL.clear()
    use_setup(make_setup())
    r, _ = cli("--sim", "rest")
    assert check_result(r, ok=False, stage="tool") and "fd park" in r["next"]
    use_tool("thin")


# ============================================================================= 7. time
def test_22_the_time_model():
    """IN-SAMPLE where it compares the model with the five timed calls of 2026-11-13: those are the calls it was
    fitted to (four numbers from five calls; the seconds per blotting and per wash rest on ONE call each, which
    therefore cannot miss). What it is worth on a call it has not seen is the leave-one-out part below."""
    fit = PS.fit_times()
    v = BW.BrushOptions().v_stroke
    assert v == 20.0 and fit["per_mm_s"] == round(1.0 / v, 4)          # a millimetre costs what the stroke speed says
    assert 3.0 <= fit["per_stroke_s"] <= 5.0 and 34.0 <= fit["per_load_s"] <= 40.0, fit
    assert 10.0 <= fit["per_blot_s"] <= 18.0 and 65.0 <= fit["per_wash_s"] <= 85.0 and fit["worst_error"] <= 0.06, fit
    # leave one out: fitted on four of the calls and asked for the fifth, the model is off by -20, +10, -10, -21
    # and -22 % (the calls with the blottings and with the wash lose exactly those seconds). Within a quarter:
    # that, not the 5 % in-sample, is what an estimate is good for until real runs have fed the means
    loo = PS.fit_times_loo()
    assert [round(100.0 * e) for e in loo] == [-20, 10, -10, -21, -22], loo
    assert max(abs(e) for e in loo) <= 0.25 and max(abs(e) for e in loo) > 4.0 * fit["worst_error"]
    assert "IN-SAMPLE" in PS.fit_times.__doc__ and "fit_times_loo" in PS.fit_times.__doc__
    for n, mm, loads, blots, washes, sec in PS.EVENING_CALLS:        # the five timed calls, each within 6 % (in-sample)
        est = PS.estimate_s({"strokes": n, "travel_mm": mm, "loads": loads, "blots": blots, "washes": washes}, fit)
        assert abs(est / sec - 1.0) <= 0.06, (n, mm, loads, est, sec)
        old = 6.0 * n + mm / 20.0 + 22.0 * loads               # what the estimate of that day said (render.plan_stats)
        assert old < est
    assert [c[5] for c in PS.EVENING_CALLS] == [289.0, 469.0, 694.0, 409.0, 342.0]
    # the Kaminarimon as it was painted: about 40 minutes of the robot in its calls (the day's estimate: 26 to 35)
    ref = "/home/jetson/lerobot/jetharness_work/paint_update/reference/kaminarimon_plan.json"
    s = make_setup()
    if os.path.exists(ref):
        b = PS.plan_budget(s, json.load(open(ref)))
        assert b["strokes"] == 70 and 36.0 <= b["minutes"] <= 44.0 and b["washes"] == 3 and len(b["layers"]) == 8, b
        assert b["layers"][7]["blots"] == b["layers"][7]["loads"] and b["blots"] == b["layers"][7]["loads"]      # the written lines
    # the running means: a real run that needed 60 s per load pulls the estimate up, weighted by how much was seen
    state = {}
    t0 = PS.times_of(state)
    assert t0["runs"] == 0 and {k: t0[k] for k in PS.TIME_KEYS} == {k: fit[k] for k in PS.TIME_KEYS}
    assert PS.note_run(state, {"strokes": 0, "travel_mm": 0, "loads": 1}, 40.0, {"load_s": 35.0}) is None      # no finished stroke
    counts = {"strokes": 20, "travel_mm": 800, "loads": 43, "blots": 0, "washes": 1}
    spent = {"travel_s": 56.0, "load_s": 43 * 60.0, "blot_s": 0.0, "wash_s": 100.0}
    total = 20 * 7.0 + 56.0 + 43 * 60.0 + 100.0
    t = PS.note_run(state, counts, total, spent)
    assert t["runs"] == 1 and t["loads"] == 43 and t["loads_s"] == 2580.0 and t["strokes_s"] == 140.0 and t["mm_s"] == 56.0
    t1 = PS.times_of(state)
    assert abs(t1["per_load_s"] - (fit["per_load_s"] * 43 + 2580.0) / 86) < 1e-3          # as many loads as the evening had
    assert abs(t1["per_wash_s"] - (fit["per_wash_s"] + 100.0) / 2) < 1e-3 and t1["per_blot_s"] == fit["per_blot_s"]
    assert abs(t1["per_mm_s"] - (0.05 * 2794 + 56.0) / (2794 + 800)) < 1e-4 and t1["runs"] == 1 and "1 real run" in t1["source"]
    assert abs(PS.estimate_s(counts, {**t1, "per_stroke_s": 7.0, "per_mm_s": 0.07, "per_load_s": 60.0, "per_wash_s": 100.0}) - total) < 1e-6
    # `run --dry` and `preview` report minutes from the state file's means
    use_tool("thin")
    use_setup(make_setup(record=tool("thin")))
    reset_state()
    plan = write_plan("time.json", [("lines", "red", bands(6, press=0.6)), ("ink", "ink", bands(4, v0=100.0, press=0.6))])
    r, _ = cli("--sim", "run", plan, "--layer", "0", "--dry")
    want = (6 * fit["per_stroke_s"] + 240 * fit["per_mm_s"] + 2 * fit["per_load_s"]) / 60.0
    assert r["budget"] == {"strokes": 6, "travel_mm": 240, "loads": 2, "blots": 0, "washes": 0} and abs(r["minutes"] - want) < 0.06, r
    assert r["calls"] == 1 and r["times"]["runs"] == 0 and r["minutes_all"] > r["minutes"]
    r2, _ = cli("--sim", "preview", plan, "--out", os.path.join(_TMP, "time.png"))
    assert r2["layers"][0]["minutes"] == r["minutes"] and abs(r2["minutes"] - r["minutes_all"]) < 0.06 and r2["blots"] == 2, r2
    PS.save_state({"timing": state["timing"]}, sim=True)
    r3, _ = cli("--sim", "run", plan, "--layer", "0", "--dry")
    assert r3["minutes"] > r["minutes"] + 0.3 and r3["times"]["runs"] == 1, (r3["minutes"], r["minutes"])
    r4, _ = cli("--sim", "run", plan, "--layer", "0", "--max-min", "1", "--dry")
    assert r4["calls"] >= 2
    reset_state()
    # a simulated arm keeps its own clock: the seconds of a result are simulated seconds, not the test's
    t_wall = time.time()
    r, made = cli("--sim", "run", plan, "--layer", "0")
    assert r["done"] and 60.0 < r["seconds"] < 400.0 and time.time() - t_wall < r["seconds"] and abs(made["bus"].now - r["seconds"]) < 5.0
    assert "timing" not in PS.load_state(sim=True)
    reset_state()


# ============================================================================= 8. the commands
def test_23_every_sub_command_answers_in_the_house_way():
    thin = use_tool("thin")
    use_setup(make_setup(record=thin))
    reset_state()
    plan = write_plan("shape.json", [("a", "red", bands(2, press=0.6)), ("b", "black", bands(1, v0=90.0, press=1.2))])
    done = {}
    for name, argv in (("status", ("status",)), ("brush", ("brush",)), ("check", ("check",)), ("look", ("look",)),
                       ("preview", ("preview", plan, "--out", os.path.join(_TMP, "shape.png"))),
                       ("run --dry", ("run", plan, "--layer", "0", "--dry")), ("run", ("run", plan, "--layer", "0")),
                       ("test --dry", ("test", "--do", "load:red", "--do", "line:0.25,-0.05:0.25,-0.02:0.6", "--dry")),
                       ("test", ("test", "--scrap-laid", "--do", "load:red", "--do", "line:0.25,-0.05:0.25,-0.02:0.6", "--do", "blot:1",
                                 "--do", "stroke:0.26,0.0", "--swatch", "black:0.27,0.02", "--ladder", "0.2,0.04")),
                       ("contact --dry", ("contact", "--dry")), ("rinse", ("rinse", "--blot-first", "1")),
                       ("rest --dry", ("rest", "--dry")), ("rest", ("rest",))):
        r, made = cli("--sim", *argv)
        check_result(r, ok=True)
        assert r.get("sim") is True or name in ("brush",), (name, r)
        assert r["cmd"] == argv[0] and ("シミュレーション" in r["say"] or name in ("status", "brush", "check", "preview")), (name, r["say"])
        done[name] = (r, made)
    # --dry moves nothing: not one write reaches the simulated servos, the torque stays off
    for name in ("run --dry", "test --dry", "contact --dry", "rest --dry", "check", "preview", "status"):
        r, made = done[name]
        if made:
            assert not moved(made["bus"]) and not made["arm"].is_torqued(), name
        assert name in ("check", "preview", "status") or r["stage"] == "planned", (name, r["stage"])
    # everything that moved says where the brush is, and left it up, at the photo pose, holding
    for name in ("look", "run", "test", "rinse"):
        r, made = done[name]
        b = r["brush"]
        assert b["where"] == "photo pose" and b["torque"] == "on" and b["lifted"] and b["height_mm"] > 50, (name, b)
        assert np.allclose(np.degrees(made["arm"].read_q()), PHOTO_Q, atol=1.5) and "撮影姿勢" in r["say"] and r["seconds"] > 5
    r = done["test"][0]
    assert r["did"] == ["load:red", "line:0.25,-0.05:0.25,-0.02:0.6", "blot:1", "stroke:0.26,0.0", "swatch:black:0.27,0.02", "ladder:0.2,0.04"]
    assert len(r["swatches"]) == 3 and len(r["ladders"][0]["rungs"]) == 4 and r["swatches"][2]["load"]["washed"] is True
    assert done["run"][0]["done"] and done["preview"][0]["picture"].endswith("shape.png") and os.path.exists(os.path.join(_TMP, "shape.png"))
    st = done["status"][0]
    assert st["brush"]["kind"] == "round" and st["tool"]["name"] == "brush_thin" and st["contact_record"]["ok"] and st["setup"] == SETUP_PATH
    assert "fd paint check" in st["next"]
    # refusals: a text, a stage, the next command; never a traceback; nothing moved
    cases = (
        (("run", plan, "--layer", "7"), "plan", "no layer"),                    # an unknown layer, by index
        (("run", plan, "--layer", "sky"), "plan", "no layer"),                  # ... and by name
        (("run", os.path.join(_TMP, "nowhere.json"), "--layer", "0"), "plan", "cannot be read"),
        (("run", SETUP_PATH, "--layer", "0"), "plan", "not a painting plan"),
        (("run", write_plan("canvas.json", [("a", "red", bands(1))], canvas=(297, 210)), "--layer", "0"), "plan", "canvas of"),
        (("run", write_plan("pan.json", [("a", "red", bands(1))], pan=(131.0, -107.2)), "--layer", "0"), "plan", "pan axis"),
        (("run", write_plan("paint.json", [("a", "pink", bands(1))]), "--layer", "0"), "plan", "pink"),
        (("run", write_plan("leave.json", [("a", "red", [([(200, 50), (240, 50)], 0.6)])]), "--layer", "0"), "plan", "leaves the canvas"),
        (("run", write_plan("long.json", [("a", "red", [([(20, 50), (200, 50)], 1.2)])]), "--layer", "0"), "plan", "more than one load"),
        (("run", plan, "--layer", "0", "--reload-mm", "1"), "options", "reload-mm"),
        (("run", plan), "options", "--layer"),
        (("preview", os.path.join(_TMP, "nowhere.json")), "plan", "cannot be read"),
        (("test", "--do", "load:red", "--do", "wiggle:1,2"), "options", "unknown test action"),
        (("test", "--do", "load:pink"), "paint", "pink"),
        (("test", "--swatch", "pink:0.25,0.0"), "paint", "pink"),
        (("test", "--do", "line:0.25,0.0"), "options", "line:x0,y0:x1,y1"),
        (("test", "--do", "line:0.25,0.0:0.25,0.03:1.3"), "options", "6 mm"),
        (("test", "--do", "stroke:0.9,0.0"), "reach", "out of reach"),
        (("test", "--do", "stroke:far"), "options", "two numbers"),
        (("test",), "options", "nothing to do"),
        (("rinse", "--blot-first", "20"), "options", "blot-first"),
        (("frobnicate",), "options", "invalid choice"),
        (("run", plan, "--layer", "0", "--speed", "9"), "options", "unrecognized"),
    )
    for argv, stage, word in cases:
        r, made = cli("--sim", *argv)
        check_result(r, ok=False, stage=stage)
        assert word in r["reason"], (argv, r["reason"])
        if made:
            assert not moved(made["bus"]) and not made["arm"].is_torqued(), argv
        if "brush" in r:
            assert r["brush"]["where"] == "not moved" and "動かしていません" in r["say"], (argv, r)
    # no brush registered: every command that needs one says how to register it
    TL.clear()
    for argv in (("look",), ("check",), ("run", plan, "--layer", "0"), ("test", "--do", "rinse"), ("contact",), ("rinse",)):
        r, made = cli("--sim", *argv)
        assert check_result(r, ok=False, stage="tool") and "fd tool attach" in r["next"], (argv, r)
    r, _ = cli("--sim", "status")
    assert check_result(r, ok=True)["tool"] is None and any("no brush" in w for w in r["warnings"])
    use_tool(thin)
    # without --sim: the real arm, which the hardware lock keeps shut (here the tests' own lock)
    for argv in (("look",), ("run", plan, "--layer", "0"), ("rest",)):
        r, _ = cli(*argv)
        check_result(r, ok=False, stage="hardware")
        assert "hardware is locked" in r["reason"] and "fd --sim paint" in r["next"] and "ロック" in r["say"], r
    reset_state()


def test_24_the_line_action_goes_any_way():
    use_tool("thin")
    s = use_setup(make_setup(record=tool("thin")))
    reset_state()
    lines = {"sideways (+y)": ((0.25, -0.05), (0.25, -0.01), 0.6), "away from the robot": ((0.20, 0.03), (0.24, 0.03), 1.2),
             "diagonal, towards the robot": ((0.28, 0.08), (0.25, 0.05), 0.6)}
    argv = ["--sim", "test", "--scrap-laid", "--do", "load:red"]         # (on the canvas: a scrap sheet lies there)
    for a, b, press in lines.values():
        argv += ["--do", f"line:{a[0]},{a[1]}:{b[0]},{b[1]}:{press}"]
    r, made = cli(*argv)
    check_result(r, ok=True, stage="test")
    t = tips(made["arm"], made["bus"])
    for (name, (a, b, press)), sw in zip(lines.items(), r["swatches"]):
        assert sw["at"] == list(a) and sw["to"] == list(b) and sw["press"] == press and sw["paint"] == "red"
        assert abs(sw["length_m"] - math.hypot(b[0] - a[0], b[1] - a[1])) < 0.002, sw
        # down at the start, then along the line to its end at contact - press, in that order
        za, zb = (PS.contact_z(s, *p) - 0.005 * press for p in (a, b))
        i_a = np.flatnonzero((np.hypot(t[:, 0] - a[0], t[:, 1] - a[1]) < 0.002) & (np.abs(t[:, 2] - za) < 0.0025))
        i_b = np.flatnonzero((np.hypot(t[:, 0] - b[0], t[:, 1] - b[1]) < 0.002) & (np.abs(t[:, 2] - zb) < 0.0025))
        assert len(i_a) and len(i_b) and i_a[0] < i_b[0], name
        seg = t[i_a[0]:i_b[0] + 1]
        ab, rel = np.array(b) - np.array(a), seg[:, :2] - np.array(a)
        d = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / math.hypot(*ab)
        assert float(d.max()) < 0.003 and float(seg[:, 2].max()) < max(za, zb) + 0.004, (name, float(d.max()))
    assert "line:x0,y0:x1,y1[:press]" in paintcli.NEW_BRUSH_HELP and "how a new brush is measured" in paintcli.NEW_BRUSH_HELP
    for word in ("fd calib-air", "fd paint contact", "fd paint brush --kind round", "fd paint check"):
        assert word in paintcli.NEW_BRUSH_HELP, word
    reset_state()


def _break_stroke(n, exc, seen=None):
    """The n-th stroke on the paper raises `exc` instead of moving: the brush is down at its start then."""
    seen = {} if seen is None else seen
    seen["n"] = 0
    real = BW.BrushRun.line

    def line(self, pts, v_mm_s, pitch=None, settle=False, step=0.003, what="move"):
        if what == "stroke":
            seen["n"] += 1
            if seen["n"] == n:
                seen["broken"] = True
                raise exc
        return real(self, pts, v_mm_s, pitch=pitch, settle=settle, step=step, what=what)

    return patched(BW.BrushRun, "line", line)


def test_25_an_interrupt_or_a_stop_ends_with_the_brush_up():
    use_tool("thin")
    s = use_setup(make_setup(record=tool("thin")))
    plan = write_plan("stop.json", [("six", "red", bands(6, press=0.6))])
    key = PS.progress_key(plan, json.load(open(plan)), 0)
    for exc, word in ((KeyboardInterrupt(), "中断"), (MotionAborted("target not reached: test", info={"reason": "target_not_reached"}), "止まりました"),
                      (SafetyError("refused: test"), "止まりました")):
        reset_state()
        with _break_stroke(3, exc):                        # in the middle of a layer
            r, made = cli("--sim", "run", plan, "--layer", "0")
        check_result(r, ok=False, stage="run")
        assert r["next_stroke"] == 2 and r["strokes"] == 2 and not r["done"] and word in r["say"], r
        assert bool(r.get("interrupted")) == isinstance(exc, KeyboardInterrupt) and "continues at stroke 2" in r["next"]
        b = r["brush"]
        assert b["where"] == "photo pose" and b["lifted"] and b["height_mm"] > 50 and b["torque"] == "on", b
        t = tips(made["arm"], made["bus"])
        x, y = PS.canvas_xy(s, 60.0, 62.0)                 # the third stroke's start: the brush had gone down there ...
        near = np.hypot(t[:, 0] - x, t[:, 1] - y) < 0.002
        assert float(t[near, 2].min()) < PS.contact_z(s, x, y) - 0.001 and t[-1][2] - TZ > 0.05      # ... and is up now
        st = PS.load_state(sim=True)
        assert st["progress"] == {key: 2} and st["paint"] == "red"       # the progress is stored: the next call goes on
        with _break_stroke(99, exc):
            r2, _ = cli("--sim", "run", plan, "--layer", "0")
        assert check_result(r2, ok=True)["start"] == 2 and r2["done"]
    # in a ladder of `fd paint test` and of `fd paint contact`, and in a single test stroke
    for argv in (("test", "--scrap-laid", "--do", "load:red", "--ladder", "0.22,0.0"),
                 ("contact", "--scrap-laid", "--paint", "red", "--at", "0.22,0.0"),
                 ("test", "--scrap-laid", "--do", "load:red", "--do", "line:0.22,0.0:0.22,0.03", "--do", "line:0.23,0.0:0.23,0.03",
                  "--do", "line:0.24,0.0:0.24,0.03")):
        for exc in (KeyboardInterrupt(), MotionAborted("target not reached: test", info={"reason": "target_not_reached"})):
            reset_state()
            with _break_stroke(3, exc):
                r, made = cli("--sim", *argv)
            check_result(r, ok=False, stage=argv[0])
            assert r["brush"]["where"] == "photo pose" and r["brush"]["height_mm"] > 50 and made["arm"].is_torqued(), (argv, r["brush"])
            assert bool(r.get("interrupted")) == isinstance(exc, KeyboardInterrupt) and "fd paint rest" in r["next"], r
            done = r["ladders"][0].get("rungs_done", len(r["ladders"][0].get("rungs", []))) if r.get("ladders") else len(r["swatches"])
            assert done == 2, (argv, r)                    # what was painted before the stop is in the result
            assert PS.load_state(sim=True)["paint"] == "red"
    # a lift that fails is reported, not hidden: the brush stays where it stopped, torque on, and the result says so
    reset_state()

    seen, real_raise = {}, BW.BrushRun.raise_to

    def no_lift(self, h, v_mm_s=None, at_least=0.03):
        if seen.get("broken"):
            raise SafetyError("no way up: test")
        return real_raise(self, h, v_mm_s=v_mm_s, at_least=at_least)

    with _break_stroke(2, MotionAborted("target not reached: test", info={"reason": "target_not_reached"}), seen):
        with patched(BW.BrushRun, "raise_to", no_lift):
            r, made = cli("--sim", "run", plan, "--layer", "0")
    check_result(r, ok=False, stage="run")
    b = r["brush"]
    assert b["where"] == "stopped" and "lift_error" in b and b["torque"] == "on" and b["height_mm"] < 20 and not b["lifted"], b
    assert "持ち上げられませんでした" in r["say"] and made["arm"].is_torqued()
    # a cancelled call arrives as SIGTERM: it is turned into the same interrupt
    assert signal.getsignal(signal.SIGTERM) is signal.default_int_handler
    reset_state()


def test_26_power_puts_the_squeeze_of_the_jaw_back():
    calls = []

    def hold(arm, tool_):
        calls.append((arm, tool_))
        return {"held": True, "wrote": True, "goal_raw": 2016}

    arm, bus = sim_arm("thin")
    with patched(TL, "hold_jaw", hold):
        br = BW.BrushRun(arm)
        assert calls == [] and br.jaw is None              # nothing before the torque is on
        br.power()
        assert len(calls) == 1 and calls[0][0] is arm and calls[0][1] is arm.tool and br.jaw["goal_raw"] == 2016 and arm.is_torqued()
        br.power()
        br.hop((0.25, 0.0))
        assert len(calls) == 1                             # once per command while the torque stays on
        arm.torque_off(force=True)
        br.power()
        assert len(calls) == 2                             # and again whenever the torque had to be switched on
    # a jaw that cannot be held is a refusal of the command: a brush with play does not paint
    arm, bus = sim_arm("thin")

    def refuse(arm_, tool_):
        raise SafetyError("the jaw stands elsewhere: test")

    with patched(TL, "hold_jaw", refuse):
        br = BW.BrushRun(arm)
        try:
            br.hop((0.25, 0.0))
        except SafetyError as e:
            assert "jaw" in str(e)
        else:
            raise AssertionError("painting went on with a jaw that is not held")
        assert not br.moved and not arm.is_torqued()       # limp again, in its rest pose: as it was found
        assert float(np.abs(arm.read_q() - TL.rest_q(arm, arm.tool)).max()) < math.radians(2.0)
        use_tool("thin")
        use_setup(make_setup(record=tool("thin")))
        r, made = cli("--sim", "look")
        assert check_result(r, ok=False, stage="tool") and "jaw" in r["reason"] and r["brush"]["where"] == "not moved", r
        assert r["next"].startswith("`fd tool jaw status`") and "爪" in r["say"]          # (R11: its own way out)
    # a harness without tooling.hold_jaw (before that change) simply powers on
    arm, bus = sim_arm("thin")
    saved = getattr(TL, "hold_jaw", None)
    if saved is not None:
        delattr(TL, "hold_jaw")
    try:
        br = BW.BrushRun(arm)
        br.power()
        assert br.jaw is None and arm.is_torqued()
    finally:
        if saved is not None:
            TL.hold_jaw = saved
    # with the real tooling.hold_jaw of this harness and a tool that records a held jaw: the result carries it.
    # (T13, third review: where one of these names was missing, this half was left out by a silent `return`: with
    # tooling.hold_jaw renamed the test PASSED in 0.2 s without ever putting a squeeze back. They are this
    # harness' own names: asserted, so that a missing one is a failure that is seen)
    assert callable(saved), "tooling.hold_jaw is missing: nothing puts the squeeze of a held jaw back"
    assert all(hasattr(TL.Tool, k) for k in ("on_jaw", "gripper_goal_raw")), "tooling.Tool has no jaw fields (on_jaw, gripper_goal_raw)"
    try:
        held = tool("thin", on_jaw="moving_outer", gripper_goal_raw=2016)
        use_tool(held)
    except (SafetyError, TypeError) as e:
        raise Skip(f"this harness' tooling does not take a held jaw as these tests write it ({e})") from None
    reset_state()
    r, made = cli("--sim", "look")
    check_result(r, ok=True, stage="look")
    assert r["jaw"]["held"] is True and r["jaw"]["goal_raw"] == 2016 and round(made["bus"].goal["gripper"]) == 2016, r.get("jaw")
    rec = PS.brush_state(held, 65.0)
    assert rec["gripper_goal_raw"] == 2016 and rec["on_jaw"] == "moving_outer"
    use_tool("thin")
    reset_state()


def test_27_preview_gives_the_picture_the_minutes_and_every_check():
    use_tool("thin")
    use_setup(make_setup(record=tool("thin")))
    reset_state()
    plan = write_plan("preview.json", [("wash", "red_pale", bands(4)), ("glaze", "red_mid", bands(3, v0=90.0)),
                                       ("letters", "ink", bands(5, u0=120.0, length=20.0, press=0.6))])
    out = os.path.join(_TMP, "pv", "picture.png")
    r, made = cli("--sim", "preview", plan, "--out", out)
    check_result(r, ok=True, stage="preview")
    img = cv2.imread(out)
    assert r["picture"] == out and img is not None and img.shape[:2] == (153 * 4, 224 * 4) and not moved(made["bus"])
    paper = img[10:60, 10:60].reshape(-1, 3).mean(axis=0)
    band = img[int(50 * 4) - 4:int(50 * 4) + 4, int(70 * 4):int(90 * 4)].reshape(-1, 3).mean(axis=0)
    assert float(paper.min()) > 215 and float(paper[1] - band[1]) > 20, (paper, band)            # red on white paper
    assert r["renderer"].split()[0] in ("roundbrush.render", "render.render") and r["design_stats_from"] in ("roundbrush.stats", "render.plan_stats")
    try:
        import roundbrush  # noqa: F401
        assert r["renderer"] == "roundbrush.render" and r["design_stats"]["strokes"] == 12
    except ImportError:
        assert "roundbrush.py is not in this harness" in r["renderer"]
    assert [ly["name"] for ly in r["layers"]] == ["wash", "glaze", "letters"] and all(ly["ok"] for ly in r["layers"])
    assert [(ly["strokes"], ly["loads"], ly["blots"], ly["washes"]) for ly in r["layers"]] == [(4, 4, 0, 0), (3, 3, 0, 0), (5, 1, 1, 1)], r["layers"]
    assert r["strokes"] == 12 and r["loads"] == 8 and r["washes"] == 1 and abs(r["minutes"] - sum(ly["minutes"] for ly in r["layers"])) < 0.2
    assert r["overlay"] is None and "simulation" in r["overlay_note"] and "写真への重ね描きはありません" in r["say"]
    assert r["next"].startswith("fd paint run") and r["pitch"] == 65.0 and r["contact_record"]["ok"]
    # without --sim, locked: no photo is taken (said so), the simulated picture is still made
    r, made = cli("preview", plan, "--out", out)
    check_result(r, ok=True, stage="preview")
    assert r["overlay"] is None and "locked" in r["overlay_note"] and "hardware is locked" in r["limits_from"] and os.path.exists(out)
    # default place of the picture: the photo folder of the run
    r, _ = cli("--sim", "preview", plan)
    assert r["picture"].startswith(OUT_DIR) and r["picture"].endswith("_preview.png") and os.path.exists(r["picture"])
    # a plan that does not fit: refused at "plan" with every problem, and the picture is made all the same
    bad = write_plan("preview_bad.json", [("ok", "red", bands(2)), ("off", "red", [([(200, 50), (240, 50)], 0.6)]),
                                          ("strange", "pink", bands(1))], canvas=(224, 153))
    r, _ = cli("--sim", "preview", bad, "--out", out)
    check_result(r, ok=False, stage="plan")
    assert len(r["problems"]) == 2 and [ly["ok"] for ly in r["layers"]] == [True, False, False] and r["picture"] == out
    assert "leaves the canvas" in r["layers"][1]["reason"] and "pink" in r["layers"][2]["reason"]
    # a stroke the arm cannot make is found here, before paint is on the brush
    use_tool("side")
    r, _ = cli("--sim", "preview", plan, "--out", out)
    assert check_result(r, ok=False, stage="plan") and "reaches the whole canvas" in r["reason"]
    # the flat brush of the morning: its own simulation
    use_tool("flat")
    use_setup(make_setup(brush=None, record=tool("flat"), pitch=72.1))        # (this brush paints the canvas at 72.1 deg)
    flat = write_plan("preview_flat.json", [("a", "red", bands(2, press=1.0))], brush=None, paints={"red": {"rgb": [214, 38, 44], "opacity": 0.9}})
    r, _ = cli("--sim", "preview", flat, "--out", out)
    assert check_result(r, ok=True)["renderer"] == "render.render" and r["design_stats_from"] == "render.plan_stats" and r["loads"] == 1
    use_tool("thin")
    use_setup()


def test_28_lower_mm_hop_retries_and_the_fd_command():
    thin = use_tool("thin")
    s = use_setup(make_setup(record=thin))
    reset_state()
    plan = write_plan("lower.json", [("bands", "red", bands(2))])
    # a layer painted with the plane 1.5 mm lower: the strokes run that much deeper, and the result says so
    lows = {}
    for extra in ((), ("--lower-mm", "1.5")):
        reset_state()
        r, made = cli("--sim", "run", plan, "--layer", "0", *extra)
        check_result(r, ok=True, stage="run")
        t = tips(made["arm"], made["bus"])
        x, y = PS.canvas_xy(s, 80.0, 50.0)
        near = np.hypot(t[:, 0] - x, t[:, 1] - y) < 0.003
        lows[extra] = float(t[near, 2].min()) - (PS.contact_z(s, x, y) - 0.006)
        assert (r.get("contact_lowered_mm") == 1.5) == bool(extra), r
    assert abs(lows[()]) < 0.0012 and abs(lows[("--lower-mm", "1.5")] + 0.0015) < 0.0012, lows
    r, made = cli("--sim", "run", plan, "--layer", "0", "--start", "0", "--stop", "1", "--lower-mm", "1.5")
    assert "--lower-mm 1.5" in r["next"] and r["next_stroke"] == 1 and not r["done"]
    r, made = cli("--sim", "run", plan, "--layer", "0", "--lower-mm", "2.5")
    assert check_result(r, ok=False, stage="options") and not moved(made["bus"])
    # test strokes too; a ladder of `fd paint test` stays relative to the plane as it was measured
    reset_state()
    r, made = cli("--sim", "test", "--scrap-laid", "--do", "load:red", "--do", "line:0.25,0.0:0.25,0.03:1.2", "--do", "ladder:0.22,0.0",
                  "--zs", "2,0", "--lower-mm", "1.5")
    check_result(r, ok=True, stage="test")
    assert r["contact_lowered_mm"] == 1.5 and abs(r["swatches"][0]["z_low"] - (PS.contact_z(s, 0.25, 0.0) - 0.006 - 0.0015)) < 2e-4
    assert [q["plan_z_mm"] for q in r["ladders"][0]["rungs"]] == [round((PS.contact_z(s, 0.22, 0.0) + 0.002) * 1000.0, 2),
                                                                   round(PS.contact_z(s, 0.22, 0.012) * 1000.0, 2)]
    # a hop that the height monitor stopped once is in the result of the command
    reset_state()
    r, made = cli("--sim", "run", plan, "--layer", "0", prep=lambda arm: _trip_monitor(arm, 1))
    check_result(r, ok=True, stage="run")
    assert r["hop_retries"] == 1 and r["brush"]["hop_retries"] == 1 and r["done"], r
    r, made = cli("--sim", "look", prep=lambda arm: _trip_monitor(arm, 2))       # stopped twice: the command ends, brush up
    check_result(r, ok=False, stage="look")
    assert "MotionAborted" in r["reason"] and r["hop_retries"] >= 1 and r["brush"]["lifted"] and r["brush"]["torque"] == "on", r
    # `fd paint ...` itself: one JSON line, exit code 0 / 1, no run log from a test
    import io

    import fd
    with patched(fd, "log_event", lambda *a, **k: None):
        for argv, rc_want, stage in ((["--sim", "paint", "status"], 0, "status"),
                                     (["--sim", "paint", "run", plan, "--layer", "0", "--dry"], 0, "planned"),
                                     (["--sim", "paint", "run", plan, "--layer", "9"], 1, "plan"),
                                     (["--sim", "paint", "check"], 0, "check"),
                                     (["--sim", "paint", "nonsense"], 1, "options"),
                                     (["--sim", "paint", "--", "status"], 0, "status"),          # one leading "--" is dropped
                                     (["--sim", "paint", "contact", "--dry"], 0, "planned"),
                                     (["--sim", "paint", "contact"], 1, "contact"),              # no --scrap-laid: nothing painted
                                     (["--sim", "paint", "contact", "--adopt", "--dry"], 0, "contact"),      # (recorded already)
                                     (["--sim", "paint", "brush", "--carries", "none", "--dry"], 0, "planned"),
                                     (["--sim", "paint", "rest", "--clearance", "9"], 1, "options"),
                                     (["paint", "look"], 1, "hardware")):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = fd.main(list(argv))
            lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
            r = json.loads(lines[-1])
            assert len(lines) == 1 and rc == rc_want and r["stage"] == stage and "wall_s" in r, (argv, rc, r)
            assert (r.get("sim") is True) == (argv[0] == "--sim")
            check_result(r, ok=rc_want == 0)
    reset_state()


def test_29_a_camera_that_fails_and_a_second_interrupt_still_give_a_result():
    thin = use_tool("thin")
    s = use_setup(make_setup(record=thin))
    reset_state()
    places = PS.contact_places(s, None, [12.0, 9.0, 6.0, 3.0], 0.010)
    plan = write_plan("camera.json", [("a", "red", bands(2, press=0.6))])

    def dead_camera():
        raise RuntimeError("camera returned no frames")

    # no photo at all: `look` says so at "camera" (the arm went to the photo pose and holds there)
    r, made = cli("--sim", "look", camera=dead_camera)
    check_result(r, ok=False, stage="camera")
    assert "camera returned no frames" in r["reason"] and "fd snap" in r["next"] and r["brush"]["where"] == "photo pose", r
    # a layer is painted and stays painted when only the photo after it fails
    r, made = cli("--sim", "run", plan, "--layer", "0", camera=dead_camera)
    check_result(r, ok=True, stage="run")
    assert r["done"] and "camera returned no frames" in r["photo_error"] and "photo" not in r
    assert PS.load_state(sim=True)["progress"][PS.progress_key(plan, json.load(open(plan)), 0)] == 2
    r, made = cli("--sim", "run", plan, "--layer", "0", "--start", "0", "--no-photo", camera=dead_camera)
    assert check_result(r, ok=True)["done"] and "photo_error" not in r                       # (--no-photo: it is not even asked)
    # contact: the ladders are painted, the second photo fails: nothing is read, and the result says how to go on
    good = _camera_for(places, hidden_paper(places))

    def dies_after_one():
        if good.shots:
            raise RuntimeError("camera returned no frames")
        return good()

    before = open(SETUP_PATH, "rb").read()
    r, made = cli("--sim", "contact", "--scrap-laid", "--paint", "red", "--save", camera=dies_after_one)
    check_result(r, ok=False, stage="camera")
    assert all(ld["rungs_done"] == 4 for ld in r["ladders"]) and "fd paint contact --read" in r["next"] and "fd paint look" in r["next"], r
    # (T12: the photo that is still to be taken is a file the program does not know: the reading command stands in
    # words, with the photo it does know; `... --read BEFORE AFTER.jpg` typed as it was printed is refused)
    assert re.findall(r"`(fd [^`]+)`", r["next"]) == ["fd paint look"] and "AFTER.jpg" not in r["next"], r["next"]
    assert f"fd paint contact --read {r['photos']['before']} (the new photo) --save" in r["next"], r["next"]
    assert os.path.exists(r["photos"]["before"]) and r["photos"]["after"] is None and open(SETUP_PATH, "rb").read() == before
    assert r["brush"]["where"] == "photo pose"
    # with both photos the result carries the command that reads them again, and that gives the same plane
    r, made = cli("--sim", "contact", "--scrap-laid", "--paint", "red", camera=_camera_for(places, hidden_paper(places)))
    check_result(r, ok=True, stage="contact")
    words = r["reread"].split()
    assert words[:4] == ["fd", "paint", "contact", "--read"] and words[4:6] == [r["photos"]["before"], r["photos"]["after"]], words
    r2, made2 = cli("--sim", *words[2:])
    check_result(r2, ok=True, stage="contact")
    assert not made2 and r2["read"] == 5 and [p["first_touch_mm"] for p in r2["places"]]
    for a, b in zip(r["places"], r2["places"]):              # (the saved photos are JPEGs of what was read in memory)
        assert abs(a["first_touch_mm"] - b["first_touch_mm"]) < 0.3, (a["name"], a["first_touch_mm"], b["first_touch_mm"])
    assert r2["record"] == r["record"] and r2["painted_with"] == PS.brush_state(thin, 65.0)
    # the ladders are stopped in the middle (the second rung of the second ladder): the record beside the photos
    # says how far they got, and reading them later takes only the ladder that was finished: no plane to store
    reset_state()
    with _break_stroke(6, MotionAborted("target not reached: test", info={"reason": "target_not_reached"})):
        r, made = cli("--sim", "contact", "--scrap-laid", "--paint", "red", "--save", camera=_camera_for(places, hidden_paper(places)))
    check_result(r, ok=False, stage="contact")
    rec = json.load(open(r["record"]))
    assert [p["rungs_done"] for p in rec["places"]] == [4, 1, 0, 0, 0] == [ld["rungs_done"] for ld in r["ladders"]], rec["places"]
    assert rec["photos"]["after"] is None and os.path.exists(r["photo"]) and open(SETUP_PATH, "rb").read() == before
    r2, _ = cli("--sim", "contact", "--read", r["photos"]["before"], r["photo"])
    assert [p["name"] for p in r2["places"]] == ["near-left"] and any("were not finished" in w for w in r2["warnings"]), r2
    r2, _ = cli("--sim", "contact", "--read", r["photos"]["before"], r["photo"], "--save")
    check_result(r2, ok=False, stage="contact")
    assert "only 1 place(s) were read" in r2["reason"] and open(SETUP_PATH, "rb").read() == before
    # a second interrupt, while the brush is being lifted after the first: the result still says where the brush is
    reset_state()
    seen = {}
    real_raise = BW.BrushRun.raise_to

    def interrupted_lift(self, h, v_mm_s=None, at_least=0.03):
        if seen.get("broken") and not seen.get("again"):
            seen["again"] = True
            raise KeyboardInterrupt()
        return real_raise(self, h, v_mm_s=v_mm_s, at_least=at_least)

    with _break_stroke(2, KeyboardInterrupt(), seen), patched(BW.BrushRun, "raise_to", interrupted_lift):
        r, made = cli("--sim", "run", plan, "--layer", "0", "--start", "0")
    check_result(r, ok=False, stage="run")
    b = r["brush"]                                       # it is still down where the stroke was to begin: said, not hidden
    assert r["interrupted"] is True and b["where"] == "stopped" and "lift_error" in b and b["torque"] == "on", r
    assert b["height_mm"] < 20 and b["lifted"] is False and "持ち上げられませんでした" in r["say"], b
    assert r["next_stroke"] == 1 and PS.load_state(sim=True)["progress"][PS.progress_key(plan, json.load(open(plan)), 0)] == 1
    reset_state()
    use_setup()


def _dying_bus(arm, when):
    """The servo bus of this arm stops answering for good once when() says so (a cable pulled out, the supply
    gone): every read and every write raises BusError from then on."""
    bus, dead = arm.bus, {"is": False}
    for name in ("read", "write", "sync_read", "sync_write", "positions", "feedback", "snapshot", "faults", "limits",
                 "torque_on", "torque_off"):
        def gone(*a, _real=getattr(bus, name), **k):
            if dead["is"] or when():
                dead["is"] = True
                raise BusError("no answer from servo 2 (shoulder_lift): test")
            return _real(*a, **k)
        setattr(bus, name, gone)
    return dead


def test_30_servos_that_stop_answering_are_not_reported_as_a_lifted_brush():
    thin = use_tool("thin")
    s = use_setup(make_setup(record=thin))
    reset_state()
    plan = write_plan("dead.json", [("four lines", "red", bands(4, press=0.6))])
    key = PS.progress_key(plan, json.load(open(plan)), 0)
    # 1. the bus dies while the third stroke is on the paper: nothing can be lifted and nothing read. The result
    #    says so, and the two finished strokes are stored
    count = {"n": 0}
    real_line = BW.BrushRun.line

    def line(self, pts, v_mm_s, pitch=None, settle=False, step=0.003, what="move"):
        count["n"] += what == "stroke"
        return real_line(self, pts, v_mm_s, pitch=pitch, settle=settle, step=step, what=what)

    with patched(BW.BrushRun, "line", line):
        r, made = cli("--sim", "run", plan, "--layer", "0", prep=lambda arm: _dying_bus(arm, lambda: count["n"] >= 3))
    check_result(r, ok=False, stage="hardware")
    b = r["brush"]
    assert "BusError" in r["reason"] and r["next_stroke"] == 2 and r["of"] == 4 and "LOOK at the arm" in r["next"], r
    assert b["where"] == "unknown" and b["torque"] == "unknown" and "BusError" in b["lift_error"], b
    assert "height_mm" not in b and "lifted" not in b and "tip_xyz" not in b, b
    assert "読めません" in r["say"] and "持ち上げられませんでした" in r["say"] and "mm 上" not in r["say"], r["say"]
    state = PS.load_state(sim=True)
    assert state["progress"] == {key: 2} and state["paint"] == "red" and "timing" not in state, state
    # ... and the same command, the arm answering again, paints the two that are missing (no second coat)
    r2, made2 = cli("--sim", "run", plan, "--layer", "0")
    check_result(r2, ok=True, stage="run")
    assert r2["start"] == 2 and r2["strokes"] == 2 and r2["done"] and r2["brush"]["where"] == "photo pose", r2
    t = tips(made2["arm"], made2["bus"])
    for i in range(4):
        x, y = PS.canvas_xy(s, 80.0, 50.0 + 6.0 * i)
        near = np.hypot(t[:, 0] - x, t[:, 1] - y) < 0.002
        low = bool(near.any() and float(t[near, 2].min()) < PS.contact_z(s, x, y) - 0.001)
        assert low == (i >= 2), (i, low)
    # 2. one servo that does not answer ONCE (the second stroke), the bus alive after it: the brush is lifted
    #    and taken to the photo pose as after any stop, and the result says where it is
    reset_state()
    with _break_stroke(2, BusError("no answer from servo 4 (wrist_flex): test")):
        r, made = cli("--sim", "run", plan, "--layer", "0")
    check_result(r, ok=False, stage="hardware")
    b = r["brush"]
    assert r["next_stroke"] == 1 and b["where"] == "photo pose" and b["lifted"] and b["torque"] == "on", r
    assert "fd paint rest" in r["next"] and "LOOK at the arm" not in r["next"] and "撮影姿勢" in r["say"], r
    assert PS.load_state(sim=True)["progress"] == {key: 1}
    assert tips(made["arm"], made["bus"])[-1][2] - TZ > 0.05
    # 3. `rest`: the bus dies in the middle of the fold. Not "the arm was not moved"
    def at_photo_then_dies(arm_):
        place(arm_, np.radians(PHOTO_Q))
        arm_.torque_on()
        arm_.bus.sleep(0.6)
        n0 = len(arm_.bus.log)
        _dying_bus(arm_, lambda: len(arm_.bus.log) > n0 + 300)

    r, made = cli("--sim", "rest", prep=at_photo_then_dies)
    check_result(r, ok=False, stage="hardware")
    assert r["brush"] == {"where": "unknown", "torque": "unknown"} and "LOOK at the arm" in r["next"], r
    assert "読めません" in r["say"] and "動かしていません" not in r["say"] and moved(made["bus"]), r
    r, made = cli("--sim", "rest", "--dry", prep=lambda arm: _dying_bus(arm, lambda: True))     # (a --dry moves nothing)
    check_result(r, ok=False, stage="hardware")
    assert "brush" not in r and "動かしていません" in r["say"] and not made["bus"].log, r
    # 4. an arm that cannot be opened at all (switched off): a refusal, nothing moved; what only computes goes on
    #    with the simulated arm and says so
    real_make = paintcli._make_arm

    def off(sim, tool="sim"):
        if not sim:
            raise BusError("no answer from servo 1 (shoulder_pan): test")
        return real_make(sim, tool)

    with patched(paintcli, "_make_arm", off):
        for argv in (["look"], ["run", plan, "--layer", "0"], ["rinse"], ["rest"], ["contact"]):
            r = paintcli.main(argv)
            check_result(r, ok=False, stage="hardware")
            assert "could not be opened" in r["reason"] and "fd state" in r["next"] and "動かしていません" in r["say"], r
            assert "brush" not in r and "sim" not in r, r
        r = paintcli.main(["check"])
        assert "could not be opened" in check_result(r)["limits_from"] and "sim" not in r, r
    assert PS.load_state(sim=False) == {} or "progress" not in PS.load_state(sim=False)
    reset_state()


def test_31_a_file_that_cannot_be_written_is_said_and_the_result_stands():
    thin = use_tool("thin")
    use_setup(make_setup(record=thin))
    reset_state()
    plan = write_plan("disk.json", [("two lines", "red", bands(2, press=0.6))])

    def full(*a, **k):
        raise OSError(28, "No space left on device")

    # the state file cannot be written after a layer (a full disk): the layer is painted, the brush is up, and the
    # result says that the progress is NOT stored and how to go on
    with patched(PS, "save_state", full):
        r, made = cli("--sim", "run", plan, "--layer", "0")
    check_result(r, ok=False, stage="state")
    assert r["done"] and r["strokes"] == 2 and r["next_stroke"] == 2 and r["brush"]["where"] == "photo pose", r
    assert "could not be written" in r["reason"] and "--start 2" in r["next"] and "状態ファイル" in r["say"], r
    assert "progress" not in PS.load_state(sim=True)
    # ... and when the layer itself was stopped, it is one more warning of that result
    with patched(PS, "save_state", full), _break_stroke(2, MotionAborted("stopped: test", info={"reason": "target_not_reached"})):
        r, made = cli("--sim", "run", plan, "--layer", "0")
    check_result(r, ok=False, stage="run")
    assert r["next_stroke"] == 1 and "状態ファイル" in r["say"] and r["brush"]["lifted"], r
    assert any("could not be written" in w and "--start 1" in w for w in r["warnings"]), r["warnings"]
    # a photo that cannot be stored: `look` says so (it is not passed off as a photo); after a layer it is a note
    # beside a result that stands
    blocked = os.path.join(_TMP, "not_a_folder")
    open(blocked, "w").close()
    os.environ["FD_PAINT_OUT"] = os.path.join(blocked, "photos")
    try:
        r, made = cli("--sim", "look", camera=blank_photo)
        check_result(r, ok=False, stage="camera")
        assert "could not be stored" in r["reason"] and "df -h" in r["next"] and r["brush"]["where"] == "photo pose", r
        reset_state()
        r, made = cli("--sim", "run", plan, "--layer", "0", camera=blank_photo)
        check_result(r, ok=True, stage="run")
        assert r["done"] and "photo" not in r and "could not be stored" in r["photo_error"], r
        r, made = cli("--sim", "preview", plan)                # (its picture goes to the same folder)
        check_result(r, ok=False, stage="options")
        assert "cannot be written" in r["reason"], r
    finally:
        os.environ["FD_PAINT_OUT"] = OUT_DIR
        os.remove(blocked)
    reset_state()


# ============================================================================= 9. what the review of 2026-11-13 found
UNCOVERED_PHOTO = os.path.join(tazuna_paths.ROOT, "runs", "20261113_paint", "190853_run.jpg")      # the finished Kaminarimon, 19:08


def test_32_a_brush_whose_tip_was_never_measured_does_not_move():
    """Finding 45. `fd tool attach --along-fingers 0.06` (a guess) and then `fd paint rinse` ran down to 1.5 mm
    above the table; that evening such a guess was 9 cm short of the real tip."""
    guess = tool("thin", name="brush_new", tip_measured=None, attached_at="2026-11-14T09:00:00")
    use_tool(guess)
    s = use_setup(make_setup(record=guess))
    reset_state()
    plan = write_plan("unmeasured.json", [("a", "red", bands(2, press=0.6))])
    state_before = os.path.exists(PS.state_path(True))
    moving = (("look",), ("locate",), ("rinse",), ("test", "--do", "load:red", "--do", "line:0.25,-0.05:0.25,-0.02:0.6"),
              ("test", "--do", "rinse", "--dry"), ("contact", "--scrap-laid", "--paint", "red"), ("contact", "--dry"),
              ("run", plan, "--layer", "0"), ("run", plan, "--layer", "0", "--dry"))
    for argv in moving:
        r, made = cli("--sim", *argv, camera=blank_photo)
        check_result(r, ok=False, stage="tool")
        assert r["next"].startswith("fd calib-air") and "--tip-looked-at" in r["next"] and "never measured" in r["reason"], (argv, r)
        assert not made and "測定されていません" in r["say"] and "動かしていません" in r["say"], (argv, r)      # no arm was even built
    assert os.path.exists(PS.state_path(True)) == state_before
    # `rest` is the way to put the arm away whatever the tool file says; what only computes says what is missing
    r, made = cli("--sim", "rest")
    assert check_result(r, ok=True)["torque"] == "off"
    r, _ = cli("--sim", "status")
    assert check_result(r, ok=True)["next"].startswith("fd calib-air") and r["tool"]["tip_measured"] is False
    assert [(st["step"], st["done"]) for st in r["steps"]] == [("jaw", True), ("tip", False), ("brush", True), ("contact", True)]
    r, made = cli("--sim", "check")
    assert check_result(r, ok=True)["next"].startswith("fd calib-air") and any("tip was measured" in w for w in r["warnings"])
    r, _ = cli("--sim", "preview", plan, "--out", os.path.join(_TMP, "unmeasured.png"))
    assert check_result(r, ok=True)["strokes"] == 2
    # the word of the operator who has looked at a photo of gripper and brush: it moves, and the result says on whose word
    r, made = cli("--sim", "look", "--tip-looked-at")
    check_result(r, ok=True, stage="look")
    assert moved(made["bus"]) and any("--tip-looked-at" in w and "not measured" in w for w in r["warnings"]), r["warnings"]
    r, made = cli("--sim", "run", plan, "--layer", "0", "--dry", "--tip-looked-at")
    assert check_result(r, ok=True, stage="planned") and any("on your word" in w for w in r["warnings"])
    # a photo from where the arm stands moves nothing: `locate --dry` needs no word
    r, made = cli("--sim", "locate", "--dry", camera=lambda: palette_photo(s))
    assert check_result(r, ok=True) and r["saved"] is False and not moved(made["bus"])
    # the layers below ask the same question: the painter (with the next command), the brush run
    arm, bus = sim_arm(guess)
    try:
        PS.Painter(arm, s)
    except PS.Refusal as e:
        assert e.stage == "tool" and e.next.startswith("fd calib-air") and e.detail["tip_measured"] is False
    else:
        raise AssertionError("a painter was built for a brush whose tip was never measured")
    try:
        BW.BrushRun(arm)
    except SafetyError as e:
        assert "never measured" in str(e) and "fd calib-air" in str(e)
    else:
        raise AssertionError("brush work started with a tip that was never measured")
    assert PS.Painter(arm, s, unmeasured_tip_ok=True).tip_measured is False and BW.BrushRun(arm, unmeasured_tip_ok=True)
    assert not moved(bus) and not arm.is_torqued()
    # a measured tip that is then changed (typed over, the jaw moved) is no measured tip any more
    t = tool("thin")
    assert BW.tip_is_measured(t)
    t.tip_offset = [0.016, -0.001, 0.060]
    assert not BW.tip_is_measured(t)
    use_tool(t)
    r, made = cli("--sim", "rinse")
    assert check_result(r, ok=False, stage="tool") and not made
    use_tool("thin")
    use_setup()


def _painted_sheet(windows):
    """A photo of a sheet with paint inside the given ladder windows [(x0, x1, y0, y1), ...]: a red band 12 mm
    wide across the middle of each."""
    img = blank_photo()
    for x0, x1, y0, y1 in windows:
        xm = (x0 + x1) / 2.0
        draw_patch(img, xm - 0.006, xm + 0.006, y0 + 0.004, y1 - 0.004)
    return img


def test_33_contact_needs_a_scrap_sheet_and_looks_at_it_first():
    """Finding 46. `fd paint contact --save` was the `next` of four commands; run as given over the finished
    picture it painted 20 rungs on it, although its own first photo showed the picture.

    IN-SAMPLE where it reads the three real photos (part 3): the thresholds of the look (paintskill.PAINT_SAT_STEP,
    PAPER_REL_GREY, BARE_MAX_PART) were SET on the 51 photos of 2026-11-13, and these three are among them. What
    the look is worth under another light or exposure no test here shows; the synthetic window below shows where
    its limit lies (a wash paler than 20 saturation steps over the paper passes for bare paper). The first real
    `fd paint contact --scrap-laid` is the check: read "sheet" in its result."""
    thin = use_tool("thin")
    s = use_setup(make_setup(record=thin))
    reset_state()
    places = PS.contact_places(s, None, [12.0, 9.0, 6.0, 3.0], 0.010)
    wins = [PS.ladder_window(p) for p in places]
    # 1. the word: without --scrap-laid nothing is painted, whatever the sheet looks like (a blank painting sheet
    #    cannot be told from scrap)
    good = _camera_for(places, hidden_paper(places))
    r, made = cli("--sim", "contact", "--paint", "red", camera=good)
    check_result(r, ok=False, stage="contact")
    assert "scrap sheet" in r["reason"] and "--scrap-laid" in r["next"] and r["next"].startswith("lay a scrap sheet"), r
    assert not moved(made["bus"]) and not good.shots and r["brush"]["where"] == "not moved" and "捨て紙" in r["say"]
    assert all(ld["rungs_done"] == 0 for ld in r["ladders"])
    r, made = cli("--sim", "contact", "--paint", "red", "--save", camera=good)
    assert check_result(r, ok=False, stage="contact") and "--scrap-laid --save" in r["next"] and not moved(made["bus"])
    # 2. the look: with the word, the first photo is looked at before any paint is loaded. A sheet with paint in ONE
    #    ladder window (the centre): refused, the brush never went down, nothing was loaded, no second photo
    shots = []

    def painted_centre():
        shots.append(1)
        return _painted_sheet([wins[4]])

    r, made = cli("--sim", "contact", "--scrap-laid", "--paint", "red", "--save", camera=painted_centre)
    check_result(r, ok=False, stage="contact")
    assert "not bare paper" in r["reason"] and "centre" in r["reason"] and "lay a scrap sheet" in r["reason"], r["reason"]
    assert "FRESH scrap sheet" in r["next"] and "捨て紙" in r["say"] and "何も描いていません" in r["say"], r
    assert [p["bare"] for p in r["sheet"]["places"]] == [True, True, True, True, False] and r["sheet"]["ok"] is False
    assert len(shots) == 1 and "photo" not in r and r["loads_total"] == 0 and all(ld["rungs_done"] == 0 for ld in r["ladders"])
    assert r["brush"]["where"] == "photo pose" and float((tips(made["arm"], made["bus"])[:, 2] - TZ).min()) > 0.03
    assert PS.load_state(sim=True).get("paint") is None and "record" not in r
    assert json.load(open(SETUP_PATH))["contact"] == s["contact"]
    # one old mark (10 x 2.5 mm, what a rung leaves) in a window is not bare paper either: a used scrap sheet
    used = blank_photo()
    q = places[0]["rungs"][3]
    draw_rung_mark(used, q, 10.0, 2.5)
    chk = PS.sheet_check(used, CAM, places, TZ)
    assert not chk["ok"] and [p["bare"] for p in chk["places"]] == [False, True, True, True, True] and chk["places"][0]["not_paper_pct"] > 0.5
    # a fresh sheet: every window 0 %, the ladders are painted
    chk = PS.sheet_check(blank_photo(), CAM, places, TZ)
    assert chk["ok"] and all(p["not_paper_pct"] == 0.0 for p in chk["places"]) and chk["why"] is None
    # a window that is not paper at all (the desk, a dark underlay)
    dark = PS.bare_paper(desk_photo(), CAM, wins[0], TZ)
    assert not dark["bare"] and "no bright paper" in dark["why"]
    # (R24) WHERE THE LIMIT LIES, on a synthetic window: a patch of pale wash over a sixth of it. The paper of these
    # photos has a saturation of 9; the look takes for paint what is more than 20 steps over that, and for a mark
    # what is darker than 0.85 of the paper. A wash just under that passes for bare paper: by the module's own
    # note a real pale wash has a saturation of 40 to 60 under the light of 2026-11-13, not far above the limit
    assert (PS.PAINT_SAT_STEP, PS.PAPER_MAX_SAT, PS.PAPER_REL_GREY, PS.BARE_MAX_PART) == (20.0, 25.0, 0.85, 0.005)
    win = (0.20, 0.24, -0.03, 0.03)
    paper_sat = float(np.median(cv2.cvtColor(blank_photo()[300:700, 600:1300], cv2.COLOR_BGR2HSV)[..., 1]))
    assert 8.0 <= paper_sat <= 10.0, paper_sat

    def with_patch(colour):
        img = blank_photo()
        draw_patch(img, 0.212, 0.228, -0.012, 0.012, colour=colour)
        return PS.bare_paper(img, CAM, win, TZ)

    def wash(sat):                                         # a reddish wash of that saturation, as bright as the paper
        low = 176.0 * (1.0 - sat / 255.0)
        return with_patch((int(round(low)), int(round((176.0 + low) / 2.0)), 176))

    for sat, bare in ((24, True), (27, True), (32, False), (36, False), (50, False)):
        got = wash(sat)
        assert got["bare"] is bare and (got["not_paper_pct"] > 10.0) is (not bare), (sat, got)
    for rel, bare in ((0.95, True), (0.88, True), (0.82, False), (0.70, False)):
        g = int(round(172.7 * rel))
        got = with_patch((g, g, g))
        assert got["bare"] is bare, (rel, got)
    assert "IN-SAMPLE" in test_33_contact_needs_a_scrap_sheet_and_looks_at_it_first.__doc__
    # 3. the two real photos of 2026-11-13: the finished picture, uncovered (19:08), and the clean scrap sheet that
    #    lay over the sheet when the plane was measured (17:39)
    if not (os.path.exists(UNCOVERED_PHOTO) and os.path.exists(EVENING_PHOTOS[0])):
        raise Skip("the photos of 2026-11-13 (190853_run.jpg, 173916_look.jpg) are not here; the synthetic part passed")
    picture, scrap = cv2.imread(UNCOVERED_PHOTO), cv2.imread(EVENING_PHOTOS[0])
    chk = PS.sheet_check(picture, CAM, places, TZ)
    pct = {p["name"]: p["not_paper_pct"] for p in chk["places"]}
    assert not chk["ok"] and pct["centre"] > 80.0 and pct["near-left"] > 25.0 and pct["near-right"] > 18.0, pct
    assert not any(p["bare"] for p in chk["places"]), pct           # (the far corners: the ends of the pillars, 4 to 5 %)
    chk = PS.sheet_check(scrap, CAM, places, TZ)
    assert chk["ok"] and all(p["not_paper_pct"] == 0.0 for p in chk["places"]), chk
    used = PS.sheet_check(cv2.imread(EVENING_PHOTOS[1]), CAM, places, TZ)         # the same scrap sheet with its ladders on
    assert not used["ok"] and sum(1 for p in used["places"] if not p["bare"]) >= 2, used
    # ... and through the command: over the picture nothing is painted; over the scrap sheet the ladders are
    r, made = cli("--sim", "contact", "--scrap-laid", "--paint", "red", camera=lambda: picture)
    check_result(r, ok=False, stage="contact")
    assert all(ld["rungs_done"] == 0 for ld in r["ladders"]) and r["loads_total"] == 0 and "centre" in r["reason"], r
    assert float((tips(made["arm"], made["bus"])[:, 2] - TZ).min()) > 0.03
    r, made = cli("--sim", "contact", "--scrap-laid", "--paint", "red", camera=lambda: scrap)
    assert r["sheet"]["ok"] and all(ld["rungs_done"] == 4 for ld in r["ladders"]) and r["loads_total"] == 1, r.get("reason")
    reset_state()
    use_setup()


def test_34_locate_stores_only_what_shows_the_palette():
    """Finding 1. `fd paint locate` wrote every well from whatever one photo gave: nothing found, or ONE stray
    blob 25 mm beside the red well, and all ten wells were moved; the places before were gone."""
    use_tool("thin")
    s = use_setup(make_setup(record=tool("thin")))
    reset_state()
    old = {n: list(w["xy"]) for n, w in s["wells"].items()}
    before = open(SETUP_PATH, "rb").read()
    # an empty photo, a single stray blob 25 mm from the red well, half a palette: refused at "locate", the file
    # byte for byte what it was, the picture named
    stray = desk_photo()
    draw_blob(stray, (old["red"][0] + 0.025 * math.cos(1.0), old["red"][1] + 0.025 * math.sin(1.0)), BLOB_BGR["red"])
    five = palette_photo(s, (0.012, 0.0), skip=("red", "blue", "yellow", "black", "green"))
    for what, photo, n in (("no palette", desk_photo(), 0), ("one stray blob", stray, 1), ("five of ten", five, 5)):
        r, made = cli("--sim", "locate", camera=lambda _p=photo: _p.copy())
        check_result(r, ok=False, stage="locate")
        assert r["saved"] is False and r["found"] == n and f"only {n} of 10 wells" in r["reason"], (what, r["reason"])
        assert open(SETUP_PATH, "rb").read() == before and "set-up keeps the places it had" in r["reason"], what
        assert os.path.exists(r["picture"]) and r["picture"].endswith("_wells.jpg") and "picture" in r["next"], what
        assert r["brush"]["where"] == "photo pose" and "保存しませんでした" in r["say"], (what, r["say"])
    rr = PS.relocate_wells(stray, CAM, s, TZ)              # (and what the gate keeps out no longer moves the other nine)
    assert rr["found"] == 1 and rr["wells"]["green"] == {**rr["wells"]["green"], "moved_mm": 0.0, "located": False, "xy": old["green"]}
    # several blobs that do not agree on one shift are no palette either
    ok, why = PS.locate_usable({"wells": dict.fromkeys(old), "missing": ["a"] * 4, "found": 6, "agree": 2, "blobs": 6})
    assert not ok and "agree" in why
    assert PS.locate_usable({"wells": dict.fromkeys(old), "missing": ["a"] * 4, "found": 6, "agree": 3, "blobs": 6}) == (True, None)
    assert PS.locate_usable({"wells": {"red": 1}, "missing": [], "found": 1, "agree": 1}) == (True, None)        # a one-well set-up
    assert PS.locate_usable({"wells": {"red": 1}, "missing": ["red"], "found": 0, "agree": 0})[0] is False
    # the palette pushed by 12 mm (2026-11-13): --dry shows it and writes nothing; without --dry it is stored, and
    # every well keeps where it was before
    pushed = palette_photo(s, (0.012, 0.0))
    r, made = cli("--sim", "locate", "--dry", camera=lambda: pushed.copy())
    check_result(r, ok=True, stage="locate")
    assert r["usable"] and r["saved"] is False and r["found"] == 10 and open(SETUP_PATH, "rb").read() == before and not moved(made["bus"])
    r, made = cli("--sim", "locate", camera=lambda: pushed.copy())
    check_result(r, ok=True, stage="locate")
    now = json.load(open(SETUP_PATH))
    assert r["saved"] and r["found"] == 10 and r["agree"] == 10 and np.allclose(r["shift_mm"], [12.0, 0.0], atol=1.5), r
    for name, w in now["wells"].items():
        assert w["previous_xy"] == old[name] and abs(w["xy"][0] - old[name][0] - 0.012) < 0.002 and abs(w["xy"][1] - old[name][1]) < 0.002, name
    assert now["located"]["found"] == 10 and now["located"]["photo"] == r["photo"] and now["located"]["missing"] == []
    assert "previous_xy" in r["say"] and PS.check_setup(now)
    assert r["not_located"] == [] and not any("located" in w for w in now["wells"].values()) and "warnings" not in r
    # R3. Six of ten visible: stored, and the four WITHOUT paint in the photo keep the places they had, marked
    # "located": false (they were moved with the others: onto bare palette). A second locate keeps the places of
    # just before it
    mid = {n: list(w["xy"]) for n, w in now["wells"].items()}
    gone = ("red", "blue", "yellow", "black")
    six = palette_photo(s, (0.012, 0.010), skip=gone)
    r, _ = cli("--sim", "locate", camera=lambda: six.copy())
    check_result(r, ok=True, stage="locate")
    now = json.load(open(SETUP_PATH))
    assert r["saved"] and r["found"] == 6 and sorted(r["missing"]) == ["black", "blue", "red", "yellow"] == sorted(r["not_located"])
    assert all(now["wells"][n]["previous_xy"] == mid[n] for n in mid)
    for n in mid:
        if n in gone:                                      # where it was, and said so: file, result, warning, sentence
            assert now["wells"][n]["xy"] == mid[n] and now["wells"][n]["located"] is False and r["wells"][n]["located"] is False, n
            want = (old[n][0] + 0.012, old[n][1] + 0.010)  # (where the palette's move would put it: looked for there next)
            assert math.hypot(now["wells"][n]["expected_xy"][0] - want[0], now["wells"][n]["expected_xy"][1] - want[1]) < 0.002, n
        else:
            assert "located" not in now["wells"][n] and abs(now["wells"][n]["xy"][1] - old[n][1] - 0.010) < 0.002, n
    assert sorted(now["located"]["not_located"]) == sorted(gone) and any("not located: " in w and "red" in w for w in r["warnings"])
    assert "位置未確認" in r["say"] and all(n in r["say"] for n in gone), r["say"]
    # ... and no paint is loaded from a well that is not located: refused at "wells" before anything moves
    plan = write_plan("wells.json", [("a", "red", bands(2, press=0.6)), ("b", "green", bands(1, v0=90.0, press=0.6))])
    for argv in (("test", "--do", "load:red"), ("test", "--do", "rinse", "--do", "swatch:red_pale:0.25,0.0"), ("test", "--do", "load:ink", "--dry"),
                 ("run", plan, "--layer", "0"), ("run", plan, "--layer", "0", "--dry"), ("contact", "--scrap-laid", "--paint", "blue"),
                 ("contact", "--paint", "yellow", "--dry")):
        r, made = cli("--sim", *argv, camera=blank_photo)
        check_result(r, ok=False, stage="wells")
        assert "did not find" in r["reason"] and "`fd paint locate`" in r["next"] and "by hand" in r["next"], (argv, r)
        assert not moved(made["bus"]) and r["brush"]["where"] == "not moved" and not made["arm"].is_torqued(), argv
        assert r["wells_not_located"] and set(r["wells_not_located"]) <= set(gone), (argv, r)
        assert "fd paint locate" in r["say"] and r["say"].count("動かしていません") == 1, (argv, r["say"])
    r, made = cli("--sim", "run", plan, "--layer", "1", "--dry")           # a paint whose well WAS found
    assert check_result(r, ok=True, stage="planned")
    r, _ = cli("--sim", "status")
    assert sorted(r["wells_not_located"]) == sorted(gone) and any("wells not located" in w for w in r["warnings"])
    r, _ = cli("--sim", "check")
    assert any("wells not located" in w for w in r["warnings"])
    arm, bus, pt = painter("thin", PS.load_setup())        # the painter itself (code that does not come through a command)
    for call in (lambda: pt.load("red"), lambda: pt.run_layer(json.load(open(plan)), 0)):
        try:
            call()
        except PS.Refusal as e:
            assert e.stage == "wells" and e.detail["wells_not_located"] == ["red"]
        else:
            raise AssertionError("paint was loaded from a well that is not located")
    assert not moved(bus) and not arm.is_torqued() and PS.wells_of(now, "ink") == ["black"] and PS.wells_of(now, "red_pale") == ["red"]
    # a later photo that shows all ten: the marks are gone, every well on its blob; red loads again
    whole = palette_photo(s, (0.012, 0.010))
    r, _ = cli("--sim", "locate", camera=lambda: whole.copy())
    now = json.load(open(SETUP_PATH))
    assert check_result(r, ok=True)["saved"] and r["not_located"] == [] and r["found"] == 10, r
    assert not any("located" in w or "expected_xy" in w for w in now["wells"].values())
    assert abs(now["wells"]["red"]["xy"][1] - old["red"][1] - 0.010) < 0.002 and now["wells"]["red"]["previous_xy"] == mid["red"]
    r, made = cli("--sim", "test", "--do", "load:red", "--dry")
    assert check_result(r, ok=True, stage="planned")
    # the way by hand: the place written into the set-up, the mark taken out; a mark that is no true / false is said
    marked = make_setup(record=tool("thin"))
    marked["wells"]["red"]["located"] = False
    use_setup(marked)
    r, made = cli("--sim", "test", "--do", "load:red", "--dry")
    assert check_result(r, ok=False, stage="wells") and not moved(made["bus"])
    del marked["wells"]["red"]["located"]
    use_setup(marked)
    assert check_result(cli("--sim", "test", "--do", "load:red", "--dry")[0], ok=True, stage="planned")
    marked["wells"]["red"]["located"] = "no"
    use_setup(marked)
    r, _ = cli("--sim", "status")
    assert check_result(r, ok=False, stage="setup") and "\"located\" must be true or false" in r["reason"]
    # a palette that was turned as well as pushed: found by the fit of shift and turn, stored, nothing lost
    use_setup(make_setup(record=tool("thin")))
    turned = palette_photo(s, (0.008, -0.004), turn_deg=6.0)
    want = palette_places(s, (0.008, -0.004), 6.0)
    r, _ = cli("--sim", "locate", camera=lambda: turned.copy())
    now = json.load(open(SETUP_PATH))
    assert check_result(r, ok=True)["saved"] and r["found"] == 10 and r["not_located"] == [] and abs(r["turn_deg"] - 6.0) < 0.6, r
    assert all(math.hypot(now["wells"][n]["xy"][0] - want[n][0], now["wells"][n]["xy"][1] - want[n][1]) < 0.002 for n in want)
    assert abs(now["located"]["turn_deg"] - 6.0) < 0.6 and "回転" in r["say"]
    # the palette as it lay on the MORNING of 2026-11-13 (about 17 mm from where the set-up of the evening has it,
    # turned, smaller blobs), the real photo: stored with four wells 14 to 18 mm beside their blobs, on bare
    # palette. Now every well that is written is on paint, the others keep their place and nothing loads from them
    morning = os.path.join(tazuna_paths.ROOT, "runs", "20261113_paint", "132819_locate.jpg")
    if os.path.exists(morning):
        use_setup(make_setup(record=tool("thin")))
        photo = cv2.imread(morning)
        r, _ = cli("--sim", "locate", camera=lambda: photo.copy())
        check_result(r, ok=True, stage="locate")
        now = json.load(open(SETUP_PATH))
        lost = sorted(r["not_located"])
        assert r["saved"] and lost and set(lost) <= {"magenta", "red", "brown", "violet"}, r
        for n, w in now["wells"].items():
            if n in lost:
                assert w["xy"] == s["wells"][n]["xy"] and w["located"] is False, n
            else:
                assert "located" not in w and paint_share(photo, w["xy"]) >= 0.5, (n, paint_share(photo, w["xy"]))
        if "red" in lost:
            r, made = cli("--sim", "test", "--do", "load:red")
            assert check_result(r, ok=False, stage="wells") and not moved(made["bus"])
    # a simulated command never writes the set-up of the real rig
    use_setup(make_setup(record=tool("thin")))
    stand_in = os.path.join(_TMP, "rig_setup_locate.json")
    shutil.copy(SETUP_PATH, stand_in)
    stamp = open(stand_in, "rb").read()
    env = {k: v for k, v in os.environ.items() if k != "FD_PAINT_SETUP"}
    with patched(PS, "SETUP_FILE", stand_in), patched(os, "environ", env):
        r, made = cli("--sim", "locate", camera=lambda: pushed.copy())
        assert check_result(r, ok=False, stage="options") and "REAL rig" in r["reason"]
    assert open(stand_in, "rb").read() == stamp
    use_setup()


def test_35_rungs_and_lowering_are_counted_from_the_plane_the_set_up_has():
    """Findings 9 and 55. The ladders of `fd paint contact` were bounded by table_z - 6 mm only: 15 mm below the
    paper the set-up already knew at the far-left corner, as deep as the bristles are long. And --lower-mm came
    on top of the 6 mm of press."""
    thin = use_tool("thin")
    s = use_setup(make_setup(record=thin))
    reset_state()
    assert abs(PS.contact_z(s, 0.289, -0.121) * 1000.0 - 8.46) < 0.02            # the rig's plane at the far-left corner
    for zs, deep in (("9,5,1,-3,-7", 15.2), ("6,3,0,-3,-6", 14.2), ("12,9,6,2", 6.2)):
        for extra in (("--dry",), ("--scrap-laid",)):
            r, made = cli("--sim", "contact", "--zs", zs, *extra)
            check_result(r, ok=False, stage="options")
            assert f"{deep:.1f} mm below the paper height" in r["reason"] and "far-left" in r["reason"], (zs, r["reason"])
            assert "--distrust-plane" in r["next"] and not moved(made["bus"]) and r["brush"]["where"] == "not moved"
    # the deepest ladder that is accepted: no rung more than 6 mm below the stored plane at its own place
    for zs in ("12,9,6,3", "12,9,6,2.3"):
        r, made = cli("--sim", "contact", "--zs", zs, "--dry")
        check_result(r, ok=True, stage="planned")
        places = PS.contact_places(s, None, [float(v) for v in zs.split(",")], 0.010)
        worst = max(PS.contact_z(s, x, q["y"]) * 1000.0 - q["plan_z_mm"] for p in places for q in p["rungs"] for x in (q["x0"], q["x1"]))
        assert worst <= 6.0 + 1e-6, (zs, worst)
    assert 5.9 < worst <= 6.0                               # ("12,9,6,2.3" is the limit itself, at the far-left corner)
    PS.check_rungs(s, PS.contact_places(s, None, [12.0, 9.0, 6.0, 3.0], 0.010))
    try:
        PS.check_rungs(s, PS.contact_places(s, [(0.345, 0.0)], [12.0, 9.0, 6.0, 3.0], 0.010))
    except PS.Refusal as e:
        assert e.stage == "options" and "6 mm" in str(e)   # (outside the canvas the plane says 9.5 mm: 3 mm is 6.5 below)
    else:
        raise AssertionError("a rung 6.5 mm below the stored plane was accepted")
    # the word for a stored plane that is known to be wrong: the rungs are then held against the table only
    r, made = cli("--sim", "contact", "--zs", "9,5,1,-3,-7", "--distrust-plane", "--dry")
    check_result(r, ok=True, stage="planned")
    assert any("--distrust-plane" in w for w in r["warnings"]) and r["zs_mm"][-1] == -7.0
    r, made = cli("--sim", "contact", "--zs", "9,5,1,-3,-8", "--distrust-plane", "--dry")       # 6.5 mm under the table
    assert check_result(r, ok=False, stage="options") and "not below -7.5" in r["reason"]
    # --lower-mm: counted together with the press, 7.5 mm at most below first touch
    r, made = cli("--sim", "test", "--do", "load:red", "--do", "line:0.25,0.0:0.25,0.03:1.2", "--lower-mm", "2")
    check_result(r, ok=False, stage="options")
    assert "8.0 mm below first touch" in r["reason"] and "7.5 mm" in r["reason"] and not moved(made["bus"]), r["reason"]
    r, made = cli("--sim", "test", "--scrap-laid", "--do", "load:red", "--do", "line:0.25,0.0:0.25,0.03:1.2", "--lower-mm", "1.5")
    check_result(r, ok=True, stage="test")
    plane_low = min(PS.contact_z(s, 0.25, 0.0), PS.contact_z(s, 0.25, 0.03))
    assert abs((plane_low - r["swatches"][0]["z_low"]) * 1000.0 - 7.5) < 0.05                        # the deepest stroke there is
    r, made = cli("--sim", "test", "--do", "line:0.25,0.0:0.25,0.03:0.6", "--lower-mm", "2", "--dry")
    assert check_result(r, ok=True, stage="planned")       # (a line: 3 mm of press and 2 mm of lowering)
    reset_state()
    bandplan = write_plan("depth.json", [("bands", "red", bands(2)), ("lines", "red", bands(2, v0=90.0, press=0.6))])
    for extra in ((), ("--dry",)):
        r, made = cli("--sim", "run", bandplan, "--layer", "0", "--lower-mm", "2", *extra)
        check_result(r, ok=False, stage="options")
        assert "bands[0]" in r["reason"] and "8.0 mm below first touch" in r["reason"] and not moved(made["bus"]), r["reason"]
    r, made = cli("--sim", "run", bandplan, "--layer", "1", "--lower-mm", "2", "--dry")
    assert check_result(r, ok=True, stage="planned") and r["contact_lowered_mm"] == 2.0
    # the same through the painter: a layer is refused before its first stroke, not at the one that asks too much
    arm, bus, pt = painter("thin", s)
    pt.set_lower(2.0)
    mixed = json.load(open(write_plan("depth_mixed.json", [("mixed", "red", bands(2, press=0.6) + bands(1, v0=80.0, press=1.2))])))
    try:
        pt.run_layer(mixed, 0)
    except PS.Refusal as e:
        assert e.stage == "options" and "mixed[2]" in str(e)
    else:
        raise AssertionError("a layer whose third stroke runs 8 mm below first touch was started")
    assert not moved(bus) and not arm.is_torqued()
    reset_state()
    use_setup()


def test_36_the_brush_carries_a_paint_from_the_first_step_that_touches_a_well():
    """Finding 19. A load stopped after the scrub (the hop to the blotting strip: hops were stopped four times
    that evening) left the state at "clean"; pale yellow then went onto a brush full of black, without a wash."""
    use_tool("thin")
    s = use_setup(make_setup(record=tool("thin")))
    reset_state()
    stop = MotionAborted("stopped by the height monitor: test", info={"reason": "monitor"})

    def stopped(*a, **k):
        raise stop

    with patched(PS.Painter, "blot", stopped):              # the recipe of "ink": water, scrub black, blot
        r, made = cli("--sim", "test", "--do", "load:ink")
    check_result(r, ok=False, stage="test")
    st = PS.load_state(sim=True)
    assert st["paint"] == "ink" and st["used_mm"] >= 1e6 - 1 and r["brush"]["where"] == "photo pose", st
    t = tips(made["arm"], made["bus"])
    assert float(t[np.hypot(t[:, 0] - 0.2578, t[:, 1] - 0.1763) < 0.012, 2].min()) < TZ + 0.006      # it HAD been in the black
    r, _ = cli("--sim", "status")
    assert r["carries"] == "ink" and "「ink」が付いています" in r["say"]
    cleaned = []
    real_clean = PS.Painter.clean
    with patched(PS.Painter, "clean", lambda self: (cleaned.append(1), real_clean(self))[1]):
        r, made = cli("--sim", "test", "--do", "load:yellow_pale")
    check_result(r, ok=True, stage="test")
    assert cleaned == [1] and PS.load_state(sim=True)["paint"] == "yellow_pale"         # washed before the pale yellow
    # stopped BEFORE a well was touched (in the water dip): the brush carries what it carried
    reset_state()
    with patched(PS.Painter, "water", stopped):
        r, _ = cli("--sim", "test", "--do", "load:ink")
    assert check_result(r, ok=False, stage="test") and PS.load_state(sim=True).get("paint") is None
    # in a layer: interrupted in the middle of its first load
    reset_state()
    plan = write_plan("carry.json", [("a", "ink", bands(2, press=0.6))])

    def interrupted(*a, **k):
        raise KeyboardInterrupt()

    with patched(PS.Painter, "blot", interrupted):
        r, _ = cli("--sim", "run", plan, "--layer", "0")
    assert check_result(r, ok=False, stage="run")["interrupted"] and PS.load_state(sim=True)["paint"] == "ink"
    assert r["next_stroke"] == 0
    # the painter itself: the same paint again is a RELOAD (it carries it), and the amount on it is not trusted
    arm, bus, pt = painter("thin", s)
    seen = _spy(pt)
    pt.blot = stopped
    try:
        pt.load("ink")
    except MotionAborted:
        pass
    assert seen == [("water", "clean", 1), ("scrub", "black", 2)] and pt.paint == "ink" and pt.used == 1e9 and pt.loads == 0
    pt.blot = lambda strokes=2: seen.append(("blot", strokes))
    seen.clear()
    r = pt.load("ink")
    assert not r["first"] and seen == [("water", "clean", 1), ("scrub", "black", 1), ("blot", 2)] and pt.used == 0.0
    reset_state()


def test_37_progress_is_kept_by_the_real_path_of_the_plan():
    """Finding 20. Keyed by the path as typed, the plan reached through the symbolic link ~/lerobot/tazuna (the
    documented command) had no progress: the 19 written lines of the Kaminarimon would have been painted again."""
    use_tool("thin")
    use_setup(make_setup(record=tool("thin")))
    reset_state()
    real_dir, link_dir = os.path.join(_TMP, "plans_real"), os.path.join(_TMP, "plans_link")
    os.makedirs(real_dir, exist_ok=True)
    if not os.path.islink(link_dir):
        os.symlink(real_dir, link_dir)
    real = os.path.join(real_dir, "sym.json")
    shutil.move(write_plan("sym.json", [("a", "red", bands(3, press=0.6)), ("b", "red", bands(2, v0=90.0, press=0.6))]), real)
    link = os.path.join(link_dir, "sym.json")
    plan = json.load(open(real))
    assert os.path.realpath(link) == os.path.realpath(real) and os.path.abspath(link) != os.path.abspath(real)
    assert PS.progress_key(link, plan, 0) == PS.progress_key(real, plan, 0) and os.path.realpath(real) in PS.progress_key(real, plan, 0)
    # painted through the real path: the same file through the link is done
    r, _ = cli("--sim", "run", real, "--layer", "0", "--stop", "2")
    assert check_result(r, ok=True)["next_stroke"] == 2
    r, made = cli("--sim", "run", link, "--layer", "0", "--dry")
    assert r["start"] == 2 and r["budget"]["strokes"] == 1 and not moved(made["bus"]), r
    r, _ = cli("--sim", "run", link, "--layer", "0")                      # ... and goes on there, never from stroke 0
    assert check_result(r, ok=True)["start"] == 2 and r["done"] and r["strokes"] == 1
    assert PS.load_state(sim=True)["progress"] == {PS.progress_key(real, plan, 0): 3}
    r, made = cli("--sim", "run", real, "--layer", "0")
    assert r["done"] and r["strokes"] == 0 and not moved(made["bus"])
    # keys an earlier version stored under the OTHER spelling (the path as it was typed, through the link) still count
    typed = f"{os.path.abspath(link)}|b|{PS.progress_key(real, plan, 1).rsplit('|', 1)[1]}"
    state = PS.load_state(sim=True)
    state["progress"][typed] = 2
    PS.save_state(state, sim=True)
    assert PS.progress_of(state, real, plan, 1) == 2 and PS.progress_of(state, link, plan, 1) == 2
    for path in (real, link):
        r, made = cli("--sim", "run", path, "--layer", "1", "--dry")
        assert r["start"] == 2 and r["of"] == 2, (path, r["start"])
    r, made = cli("--sim", "run", real, "--layer", "1")
    assert r["done"] and r["strokes"] == 0 and not moved(made["bus"]), r                 # no second coat
    # one layer, one number: a new run stores it under the real path and takes the other spelling out
    r, _ = cli("--sim", "run", link, "--layer", "1", "--start", "1")
    assert check_result(r, ok=True)["done"] and r["strokes"] == 1
    prog = PS.load_state(sim=True)["progress"]
    assert typed not in prog and prog[PS.progress_key(real, plan, 1)] == 2 and len(prog) == 2, prog
    # another file with the same content is another plan; the same file changed starts anew (as before)
    other = os.path.join(_TMP, "sym_copy.json")
    shutil.copy(real, other)
    assert PS.progress_of(PS.load_state(sim=True), other, plan, 0) == 0
    # (R27) ... but that the same strokes are recorded as painted under ANOTHER path is said, by `run --dry` and by
    # `preview` (the shipped copy of the Kaminarimon's plan led to "paint layer 0" over the finished picture)
    r, made = cli("--sim", "run", other, "--layer", "0", "--dry")
    assert check_result(r, ok=True, stage="planned")["start"] == 0 and not moved(made["bus"])
    assert r["painted_elsewhere"] == [{"layer": "a", "strokes": 3, "of": 3, "plan": os.path.realpath(real)}], r.get("painted_elsewhere")
    assert any("same strokes" in w and os.path.realpath(real) in w for w in r["warnings"]), r["warnings"]
    r, _ = cli("--sim", "preview", other, "--out", os.path.join(_TMP, "sym_copy.png"))
    assert check_result(r, ok=True) and [e["layer"] for e in r["painted_elsewhere"]] == ["a", "b"] and any("same strokes" in w for w in r["warnings"])
    for path in (real, link):                              # (the plan itself, however it is spelled: nothing "elsewhere")
        r, _ = cli("--sim", "run", path, "--layer", "0", "--dry")
        assert "painted_elsewhere" not in r, path
    assert PS.painted_elsewhere({}, other, plan) == [] and PS.painted_elsewhere(PS.load_state(sim=True), real, plan) == []
    # the rig's own state file of 2026-11-13: its keys are real paths already, and the Kaminarimon's written lines
    # are done through either spelling
    rig_state = os.path.join(BACKUP, "paint_state.json")
    rig_plan = os.path.join(tazuna_paths.ROOT, "runs", "20261113_paint", "kaminarimon_plan.json")
    if os.path.exists(rig_state) and os.path.exists(rig_plan):
        st = PS.check_state(json.load(open(rig_state)))
        kp = json.load(open(rig_plan))
        via_link = os.path.join(_TMP, "tazuna")
        if not os.path.islink(via_link):
            os.symlink(tazuna_paths.ROOT, via_link)
        for path in (rig_plan, os.path.join(via_link, "runs", "20261113_paint", "kaminarimon_plan.json")):
            assert PS.progress_of(st, path, kp, 7) == 19 and len(kp["layers"][7]["strokes"]) == 19, path
    reset_state()


def test_38_a_jaw_that_is_shut_without_a_recorded_hold_is_said():
    """Findings 21 and 49 (the painting side). The rig's own tool file of 2026-11-13 has the jaw shut at raw 2041
    and the squeeze only in its notes: nothing put the squeeze back, no command said so, and `fd paint status`
    sent the operator to the contact plane first."""
    rig = tool("thin", tip_measured=None, **RIG_JAW)       # as the rig's file stands: jaw shut, nothing recorded, tip typed
    use_tool(rig)
    use_setup(make_setup(brush=None))                      # ... and its set-up: no brush block, no contact record
    reset_state()
    assert CFG["gripper_touch_raw"] == 2050 and rig.gripper_raw == 2041
    # `fd paint status`: the first missing step, in the order it has to be done
    r, _ = cli("--sim", "status")
    check_result(r, ok=True, stage="status")
    # (R6: the jaw is shut WITHOUT a hold on record and the tip is not measured: the tip comes first. Photos taken
    # with the jaw as it stands are stored BEFORE `fd tool jaw close`; with a hold on record they are refused)
    assert [(st["step"], st["done"]) for st in r["steps"]] == [("tip", False), ("jaw", False), ("brush", False), ("contact", False)]
    # (T12, third review: the folder of those photos is a value the program does not know, so this step stands in
    # WORDS; `fd calib-air --from DIR --save`, which stood at the beginning of this next, is refused as it is typed)
    assert r["next"].startswith("ONLY when photos of this brush exist that were taken with the jaw as it stands now: fd calib-air "
                                "with --from, the folder of those photos, and --save   (no motion"), r["next"]
    assert "BEFORE the jaw is closed and held" in r["next"] and "DIR" not in r["next"] and "--from DIR" not in r["say"], r["next"]
    assert "`fd tool jaw mount moving_outer`" in r["next"] and "`fd tool jaw close`" in r["next"] and "`fd calib-air --save`" in r["next"]
    assert any("jaw is shut" in w and "no jaw hold is recorded" in w for w in r["warnings"]) and "爪" in r["say"] and "写真" in r["say"]
    order = []
    for t_, s_ in ((tool("thin", tip_measured=None, on_jaw="moving_outer", gripper_goal_raw=None), make_setup(brush=None)),
                   (tool("thin", **RIG_JAW), make_setup(brush=None)),             # the tip stored: now the jaw
                   (tool("thin", gripper_goal_raw=None), make_setup(brush=None)),
                   (tool("thin", tip_measured=None), make_setup(brush=None)),     # held, the tip not measured (the jaw moved)
                   (tool("side", tip_measured=None), make_setup(brush=None)),     # an open jaw, nothing to hold
                   (tool("thin"), make_setup(brush=None)),
                   (tool("thin"), make_setup()),
                   (tool("thin"), make_setup(record=tool("thin", tip_offset=[0.016, -0.001, 0.058]))),
                   (tool("thin"), make_setup(record=tool("thin")))):
        use_tool(t_)
        use_setup(s_)
        r, _ = cli("--sim", "status")
        order.append(r["next"].split("   ")[0].split(" measures")[0])
        r2, _ = cli("--sim", "check")                      # `fd paint check` names the same step, whatever its stage
        if not r2["problems"] and any(not st["done"] for st in r["steps"]):    # (what is out of reach comes first in `check`)
            assert r2["next"] == r["next"], (r["next"], r2["next"])
    assert order == ["ONLY when photos of this brush exist that were taken with the jaw as it stands now: " + PS.FROM_PHOTOS,
                     "fd tool jaw mount moving_outer", "fd tool jaw close", "fd calib-air --save",
                     "fd calib-air --save", "fd paint brush --kind round --fine 0.6:2.7 --broad 1.2:5.0 --load-mm 140",
                     "fd paint contact --adopt", "lay a scrap sheet over the whole canvas, then `fd paint contact --scrap-laid --save` (it",
                     "`fd paint check` (reach); " + paintcli.PLAN_NEXT], order
    # (T12: a plan's file is a value the program does not know: no `fd paint preview PLAN` in backticks any more)
    assert "PLAN" not in order[-1] and "`fd paint preview" not in order[-1] and "`fd paint run" not in order[-1], order[-1]
    # once a hold is on record, photos from before it are not offered any more (they would be refused); without a
    # hold and with nothing to hold (an open jaw) they still are; and only --save is named as what records the tip
    for t_, offered in ((tool("thin", tip_measured=None), False), (tool("side", tip_measured=None), True),
                        (tool("thin", tip_measured=None, on_jaw="moving_outer", gripper_goal_raw=None), True)):
        nxt = PS.tip_next(t_, CFG)
        assert (PS.FROM_PHOTOS in nxt) == offered and ("--from" in nxt) == offered and "DIR" not in nxt, nxt
        assert "--save" in nxt.split("   ")[0] and ("only --save writes the record" in nxt or nxt.startswith("ONLY when photos")), nxt
    # T9 (third review): with a hold on record this text said that such photos "cannot be stored any more", while the
    # refusal of `fd calib-air` for them names a way back that works as typed. Every text gives the one order
    held = PS.tip_next(tool("thin", tip_measured=None), CFG)
    assert held is PS.TIP_NEXT_HELD and held.startswith("fd calib-air --save   (")
    assert "are refused as they are; the refusal names the way back" in held and "cannot be stored any more" not in held, held
    assert "`fd tool jaw close`, then" in PS.tip_next(tool("thin", tip_measured=None, on_jaw="moving_outer", gripper_goal_raw=None), CFG)
    assert "mount" not in PS.tip_next(tool("thin", tip_measured=None, on_jaw="moving_outer", gripper_goal_raw=None), CFG)
    # the refusal of a moving command names the same step as status (ONE order in every text)
    for t_ in (rig, tool("thin", tip_measured=None), tool("side", tip_measured=None)):
        use_tool(t_)
        r, made = cli("--sim", "look")
        r2, _ = cli("--sim", "status")
        assert check_result(r, ok=False, stage="tool")["next"].startswith(r2["next"]) and "--tip-looked-at" in r["next"] and not made, (r["next"], r2["next"])
    # `fd paint check` says it too, and names the same next step (its stage is "contact": the plane has no record)
    use_tool(rig)
    use_setup(make_setup(brush=None))
    r, made = cli("--sim", "check")
    check_result(r, ok=False, stage="contact")
    assert any("jaw is shut" in w for w in r["warnings"]) and r["next"].startswith("ONLY when photos of this brush exist"), r
    assert "その前にすること" in r["say"] and not moved(made["bus"])
    # the note itself: shut and not held, unless the tool is recorded on the fixed jaw or between the jaws
    assert BW.jaw_note(tool("thin"), 2050) is None                                        # held
    assert "does not say on which jaw" in BW.jaw_note(tool("thin", **RIG_JAW), 2050)
    assert "recorded on the outer face of the moving jaw" in BW.jaw_note(tool("thin", gripper_goal_raw=None), 2050)
    assert BW.jaw_note(tool("flat"), 2050) is None and BW.jaw_note(tool("thin", on_jaw="between", gripper_goal_raw=None), 2050) is None
    assert BW.jaw_note(tool("side"), 2050) is None and BW.jaw_note(tool("thin", gripper_raw=None, **RIG_JAW), 2050) is None
    assert PS.jaw_hold_note(tool("thin", **RIG_JAW), CFG) == BW.jaw_note(tool("thin", **RIG_JAW), 2050)
    # BrushRun.power(): nothing can be held (tooling.hold_jaw: "no jaw hold is recorded"), and that is SAID
    loose = tool("thin", **RIG_JAW)                        # (the tip measured: the jaw is what this is about)
    arm, bus = sim_arm(loose)
    br = BW.BrushRun(arm)
    assert br.jaw_note is None
    assert callable(TL.jaw_in_place) and callable(TL.hold_jaw) and TL.jaw_tolerance(loose) == TL.JAW_END_TOL_TICKS
    br.power()              # (the jaw stands where the tool file has it: powered; what is missing is the hold)
    assert arm.is_torqued() and "no jaw hold is recorded" in br.jaw_note and not (br.jaw or {}).get("held")
    assert round(bus.goal["gripper"]) == 2041          # (the goal is where the jaw stands: no squeeze)
    assert br.jaw_place == {"pos_raw": 2041, "from_record_ticks": 0, "tol_ticks": TL.JAW_END_TOL_TICKS, "held": False}
    arm, bus = sim_arm("thin")
    br = BW.BrushRun(arm)
    br.power()
    assert br.jaw_note is None and br.jaw["held"] and round(bus.goal["gripper"]) == 2016
    # ... and reaches the operator through every command that moves the brush
    use_tool(loose)
    use_setup(make_setup(record=loose))
    for argv in (("look",), ("rinse",)):
        r, made = cli("--sim", *argv)
        check_result(r, ok=True, stage=argv[0])
        assert any("no jaw hold is recorded" in w and "fd tool jaw mount moving_outer" in w for w in r["warnings"]), r["warnings"]
        assert r["jaw"] == {"held": False, "why": "no jaw hold is recorded although the jaw is shut"} and "保持" in r["say"]
    # tooling.jaw_in_place is asked BEFORE the torque comes on (reads only), once per command; what it refuses is
    # a refusal of brush work with the arm as it was found
    seen = []

    def in_place(arm_, tool_):
        seen.append((arm_.is_torqued(), tool_.name))
        return None

    arm, bus = sim_arm("thin")
    with patched(TL, "jaw_in_place", in_place):
        br = BW.BrushRun(arm)
        br.power()
        br.power()
        br.hop((0.25, 0.0))
    assert seen == [(False, "brush_thin")], seen

    def elsewhere(arm_, tool_):
        raise SafetyError("the jaw stands at raw 2141, 100 ticks from where it was when the tip was measured: test")

    arm, bus = sim_arm("thin")
    w0 = bus.io["writes"]
    with patched(TL, "jaw_in_place", elsewhere):
        br = BW.BrushRun(arm)
        try:
            br.hop((0.25, 0.0))
        except SafetyError as e:
            assert "100 ticks" in str(e)
        else:
            raise AssertionError("brush work started with the jaw elsewhere than the tool file says")
    assert not arm.is_torqued() and not br.moved and not moved(bus, w0)
    # with this harness' own tooling: a HELD jaw that stands 100 ticks from its record stops the command before
    # the torque comes on (tooling.jaw_in_place as it is today; for a jaw without a recorded hold see the hooks)
    use_tool("thin")
    use_setup(make_setup(record=tool("thin")))

    def limp_elsewhere(arm_):
        arm_.bus.pos["gripper"] = arm_.bus.goal["gripper"] = 2141.0

    # (R11: the refusal names its own way out. With the general hint the operator went from `status` (ok) to `check`
    # (ok) to `run --dry` ("planned") and back to this refusal; and the sentence for the operator is Japanese)
    plan = write_plan("jaw.json", [("a", "red", bands(2, press=0.6))])
    for argv in (("look",), ("rinse",), ("run", plan, "--layer", "0"), ("run", plan, "--layer", "0", "--dry"),
                 ("test", "--do", "load:red"), ("contact", "--scrap-laid", "--paint", "red"), ("locate",)):
        r, made = cli("--sim", *argv, prep=limp_elsewhere, camera=blank_photo)
        check_result(r, ok=False, stage="tool")
        assert "raw 2141" in r["reason"] and r["brush"]["where"] == "not moved" and not made["arm"].is_torqued(), (argv, r)
        assert round(made["bus"].goal["gripper"]) == 2141 and not made["bus"].log, argv
        assert r["next"].startswith("`fd tool jaw status`") and "fd paint check" not in r["next"], (argv, r["next"])
        assert "爪" in r["say"] and "fd tool jaw status" in r["say"] and "raw 2141" not in r["say"] and "SafetyError" not in r["say"], r["say"]
        assert r["say"].count("動かしていません") == 1
    # status and check only read files (they must answer under the hardware lock): they do not ask the jaw
    for argv in (("status",), ("check",)):
        r, made = cli("--sim", *argv, prep=limp_elsewhere)
        assert check_result(r, ok=True) and "raw 2141" not in json.dumps(r), argv
    # a jaw refusal is a SafetyError of its own kind, from both of tooling's questions
    arm, bus = sim_arm("thin")
    arm.bus.pos["gripper"] = arm.bus.goal["gripper"] = 2141.0
    br = BW.BrushRun(arm)
    for call in (br.check_jaw, br.power, lambda: br.hop((0.25, 0.0))):
        try:
            call()
        except BW.JawRefusal as e:
            assert isinstance(e, SafetyError) and "raw 2141" in str(e)
        else:
            raise AssertionError("brush work went on with the jaw 100 ticks from its record")
    assert not arm.is_torqued() and not br.moved

    def cannot_hold(arm_, tool_=None):
        raise SafetyError("with its holding goal written again the jaw stands at raw 2070: test")

    arm, bus = sim_arm("thin")
    with patched(TL, "hold_jaw", cannot_hold):
        br = BW.BrushRun(arm)
        try:
            br.power()
        except BW.JawRefusal as e:
            assert "raw 2070" in str(e)
        else:
            raise AssertionError("brush work went on with a jaw that cannot be held")
    assert not arm.is_torqued() and not br.moved           # (limp again where it rests)
    with patched(TL, "hold_jaw", cannot_hold):
        r, made = cli("--sim", "look")
    assert check_result(r, ok=False, stage="tool")["next"].startswith("`fd tool jaw status`") and r["brush"]["where"] == "not moved"
    # a harness without tooling.jaw_in_place (before that change) simply powers on
    arm, bus = sim_arm("thin")
    saved = TL.jaw_in_place
    del TL.jaw_in_place
    try:
        br = BW.BrushRun(arm)
        br.power()
        assert arm.is_torqued() and br.jaw_place is None
    finally:
        TL.jaw_in_place = saved
    use_setup()
    reset_state()


def _newer_tool(t):
    """The tool file holds t and is NEWER than the set-up file (as after `fd tool jaw close`, `fd calib-air --save`
    or a re-taping that came after the plane was written)."""
    use_tool(t, age_s=-60.0)
    assert os.path.getmtime(TOOL_PATH) > os.path.getmtime(SETUP_PATH)


def test_39_a_plane_without_a_record_is_not_painted_with_until_it_is_adopted_or_measured():
    """Finding 47 and R5. A contact plane without a "measured" record passed for every brush state: a tip 6 mm
    longer, even a tool of another name, painted with the plane of the old brush (bands pressed 11 mm instead of
    6). The first fix dated such a plane by its file and refused it once the tool file was the newer one; the
    tool measured anew and THEN the set-up edited by hand (the hand-over asked for that edit) made the old plane
    the newer file again, and the stroke was painted. Now no date is asked: no record, no stroke."""
    thin = tool("thin")
    use_tool(thin)
    s = use_setup(make_setup())                            # the rig's set-up as it stands: a plane, no record
    reset_state()
    plan = write_plan("adopt.json", [("a", "red", bands(2, press=0.6))])

    def refused_everywhere(what):
        for argv in (("run", plan, "--layer", "0"), ("run", plan, "--layer", "0", "--dry"),
                     ("test", "--do", "load:red", "--do", "line:0.25,0.0:0.25,0.03:1.2"),
                     ("test", "--do", "stroke:0.25,0.0", "--dry")):
            r, made = cli("--sim", *argv)
            check_result(r, ok=False, stage="contact")
            assert "no record of the brush state" in r["reason"] and r["contact"] == {"recorded": False, "ok": False, "why": r["contact"]["why"]}, (what, r)
            assert r["next"].startswith("fd paint contact --adopt") and "--scrap-laid --save" in r["next"], (what, r["next"])
            assert not moved(made["bus"]) and r["brush"]["where"] == "not moved" and not made["arm"].is_torqued(), (what, argv)
            assert "warnings" not in r or not any("contact plane" in w for w in r["warnings"]), (what, r["warnings"])
        r, _ = cli("--sim", "check")
        assert check_result(r, ok=False, stage="contact")["next"].startswith("fd paint contact --adopt") and "--adopt" in r["say"], what
        r, _ = cli("--sim", "preview", plan, "--out", os.path.join(_TMP, "adopt.png"))
        assert check_result(r, ok=False, stage="contact")["next"].startswith("fd paint contact --adopt"), what
        r, _ = cli("--sim", "status")
        assert (r["contact_record"]["recorded"], r["contact_record"]["ok"]) == (False, False) and r["contact_record"]["why"], what
        assert r["next"].startswith("fd paint contact --adopt") and any("no record of the brush state" in w for w in r["warnings"]), what
        r, _ = cli("--sim", "test", "--ladder", "0.25,0.0", "--dry")          # a ladder is how a plane is looked for
        assert check_result(r, ok=True) and any("no record of the brush state" in w for w in r["warnings"]), what

    # 1. the tool file is the OLDER of the two (use_tool dates it an hour back): this is the verifier's order, the
    #    tool measured first and the set-up written after it. It used to paint, with a warning
    assert os.path.getmtime(TOOL_PATH) < os.path.getmtime(SETUP_PATH)
    refused_everywhere("the tool file is the older one")
    # 2. the set-up edited by hand after the tool changed (the hand-over: "load_mm" in the recipes ink and ink2),
    #    with a longer tip, another tool, or the brush as it was
    for t in (tool("thin", tip_offset=[0.016, -0.001, 0.062]), tool("thin", name="brush_new", attached_at="2026-11-14T09:00:00"), thin):
        use_tool(t, age_s=1800.0)
        edited = copy.deepcopy(s)
        edited["paints"]["ink"]["load_mm"], edited["paints"]["ink2"]["load_mm"] = 110, 60
        use_setup(edited)
        refused_everywhere(f"hand edit after the tool changed ({t.name}, {t.tip_offset})")
    # 3. `fd paint brush` after the edit, `fd paint locate`: rewriting the file gives the plane no record and no date
    use_tool(thin, age_s=1800.0)
    r, _ = cli("--sim", "brush", "--load-mm", "120")
    c = json.load(open(SETUP_PATH))["contact"]
    assert check_result(r, ok=True)["saved"] and "measured" not in c and "plane_at" not in c, c
    refused_everywhere("after `fd paint brush`")
    r, _ = cli("--sim", "locate", camera=lambda: palette_photo(s, (0.012, 0.0)))
    assert check_result(r, ok=True)["saved"] and "plane_at" not in json.load(open(SETUP_PATH))["contact"]
    refused_everywhere("after `fd paint locate`")
    # 4. the tool file is the newer one (as after `fd tool jaw close` or `fd calib-air --save`): refused as before
    use_setup(s)
    _newer_tool(thin)
    refused_everywhere("the tool file is the newer one")
    # 5. a date in the block (what a version of 2026-11-13 wrote: "plane_at") decides nothing, and is taken out
    #    when the file is next written
    dated = copy.deepcopy(s)
    dated["contact"]["plane_at"] = "2099-01-01T00:00:00"
    use_setup(dated)
    use_tool(thin)
    refused_everywhere("a plane dated in the future")
    r, _ = cli("--sim", "brush", "--load-mm", "140")
    assert check_result(r, ok=True)["saved"] and "plane_at" not in json.load(open(SETUP_PATH))["contact"]
    use_setup(s)
    # the painter itself (built in code, no files): the same rule, before anything moves
    arm, bus, pt = painter("thin", s)
    st = pt.contact_state()
    assert st == {**st, "recorded": False, "ok": False} and st["next"] == PS.ADOPT_NEXT and not hasattr(pt, "tool_file")
    try:
        pt.run_layer(json.load(open(plan)), 0)
    except PS.Refusal as e:
        assert e.stage == "contact" and e.next == PS.ADOPT_NEXT
    else:
        raise AssertionError("a layer was painted with a plane that has no record")
    assert not moved(bus) and not arm.is_torqued()
    assert PS.contact_record_state(s, thin, 65.0) == PS.contact_record_state(s, tool("flat"), None)        # whoever the brush is
    # `fd paint contact --adopt`: no motion, nothing measured; the operator's word that this plane was measured
    # with the brush as it is. --dry writes nothing
    before = open(SETUP_PATH, "rb").read()
    r, made = cli("--sim", "contact", "--adopt", "--dry")
    check_result(r, ok=True, stage="planned")
    assert r["saved"] is False and open(SETUP_PATH, "rb").read() == before and not moved(made["bus"]), r
    r, made = cli("--sim", "contact", "--adopt")
    check_result(r, ok=True, stage="contact")
    c = json.load(open(SETUP_PATH))["contact"]
    m = c["measured"]
    assert r["saved"] and not moved(made["bus"]) and not made["arm"].is_torqued() and "測定はしていません" in r["say"]
    assert {k: c[k] for k in ("z0", "x0", "y0", "bx", "by")} == {k: s["contact"][k] for k in ("z0", "x0", "y0", "bx", "by")}
    assert m["adopted"] is True and m["tool"] == "brush_thin" and m["tip_offset"] == [0.016, -0.001, 0.056] and m["pitch_deg"] == 65.0
    assert m["gripper_raw"] == 2041 and m["gripper_goal_raw"] == 2016 and m["on_jaw"] == "moving_outer" and "note" in m
    assert c.get("note") == s["contact"].get("note") and "plane_at" not in c           # (what the set-up said about its plane stays)
    for argv in (("run", plan, "--layer", "0", "--dry"), ("test", "--scrap-laid", "--do", "line:0.25,0.0:0.25,0.03:1.2", "--dry")):
        r, _ = cli("--sim", *argv)
        assert check_result(r, ok=True, stage="planned") and "warnings" not in r, r.get("warnings")
    r, _ = cli("--sim", "status")
    assert r["contact_record"] == {"recorded": True, "ok": True, "why": []} and "fd paint check" in r["next"]
    # adopted for this brush already: nothing to write
    stamp = open(SETUP_PATH, "rb").read()
    r, _ = cli("--sim", "contact", "--adopt")
    assert check_result(r, ok=True)["saved"] is False and open(SETUP_PATH, "rb").read() == stamp and "すでに" in r["say"]
    # R19: the same numbers under another name, or the same brush registered anew (another attach time), are
    # another brush state: not "recorded already", and no layer; adopting says whose record it replaces
    for other in (tool("thin", name="brush_b"), tool("thin", attached_at="2026-11-14T08:00:00")):
        use_tool(other)
        r, made = cli("--sim", "run", plan, "--layer", "0", "--dry")
        assert check_result(r, ok=False, stage="contact") and "the registered tool is" in r["reason"] and not moved(made["bus"]), r
        r, _ = cli("--sim", "status")
        # (T10: the record that differs is an ADOPTED one: the adoption is named beside the measurement)
        assert r["contact_record"]["recorded"] and r["contact_record"]["ok"] is False and r["next"] == PS.ADOPT_NEXT, r["next"]
        assert "fd paint contact --scrap-laid --save" in r["next"] and r["contact_record"]["adopted"] is True
        r, _ = cli("--sim", "contact", "--adopt", "--dry")
        assert check_result(r, ok=True, stage="planned") and r["replaces"]["tool"] == "brush_thin" and "すでに" not in r["say"], r
    assert open(SETUP_PATH, "rb").read() == stamp
    # the brush changes afterwards: the adopted record stops the layer like a measured one ...
    use_tool(tool("thin", tip_offset=[0.016, -0.001, 0.062]))
    r, made = cli("--sim", "run", plan, "--layer", "0")
    assert check_result(r, ok=False, stage="contact") and "adopted for another brush state" in r["reason"] and not moved(made["bus"])
    assert "measured for another brush state" not in r["reason"] and r["next"] == PS.ADOPT_NEXT, (r["reason"], r["next"])
    # ... and, being the operator's word, may be given again for the brush as it is now (said in the result)
    r, _ = cli("--sim", "contact", "--adopt")
    assert check_result(r, ok=True)["saved"] and r["replaces"]["tip_offset"] == [0.016, -0.001, 0.056]
    assert json.load(open(SETUP_PATH))["contact"]["measured"]["tip_offset"] == [0.016, -0.001, 0.062]
    # a record that came from LADDERS is not written over by a word: another brush state needs a measurement
    use_setup(make_setup(record=thin))
    stamp = open(SETUP_PATH, "rb").read()
    for t in (tool("thin", tip_offset=[0.016, -0.001, 0.062]), tool("thin", attached_at="2026-11-14T08:00:00")):
        use_tool(t)
        for extra in ((), ("--dry",)):
            r, _ = cli("--sim", "contact", "--adopt", *extra)
            check_result(r, ok=False, stage="contact")
            assert "MEASURED record" in r["reason"] and "`fd paint contact --scrap-laid --save`" in r["next"], r
    assert open(SETUP_PATH, "rb").read() == stamp
    # it stands alone, needs a brush, and a simulated command does not write the real rig's set-up
    for argv in (("--adopt", "--save"), ("--adopt", "--at", "0.25,0.0"), ("--adopt", "--scrap-laid"), ("--adopt", "--zs", "9,6")):
        r, _ = cli("--sim", "contact", *argv)
        assert check_result(r, ok=False, stage="options") and "stands alone" in r["reason"], argv
    use_tool(thin)
    use_setup(make_setup())
    stand_in = os.path.join(_TMP, "rig_setup_adopt.json")
    shutil.copy(SETUP_PATH, stand_in)
    stamp = open(stand_in, "rb").read()
    env = {k: v for k, v in os.environ.items() if k != "FD_PAINT_SETUP"}
    with patched(PS, "SETUP_FILE", stand_in), patched(os, "environ", env):
        r, _ = cli("--sim", "contact", "--adopt")
        assert check_result(r, ok=False, stage="options") and "REAL rig" in r["reason"]
        r, _ = cli("--sim", "contact", "--adopt", "--dry")
        assert check_result(r, ok=True, stage="planned")
    assert open(stand_in, "rb").read() == stamp
    TL.clear()
    r, _ = cli("--sim", "contact", "--adopt")
    assert check_result(r, ok=False, stage="tool")
    use_tool(thin)
    # the rig's own files of 2026-11-13 19:25, as they stand (copies): the hand-over. The plane has no record: no
    # stroke is painted with it. Once the jaw is held and the tip measured (test 49 goes that way command by
    # command), the plane is adopted for the held brush
    rig_tool, rig_setup = os.path.join(BACKUP, "tool_active.json"), os.path.join(BACKUP, "paint_setup.json")
    if os.path.exists(rig_tool) and os.path.exists(rig_setup):
        shutil.copy(rig_setup, SETUP_PATH)
        t = TL.load(rig_tool)
        assert t.on_jaw is None and t.gripper_goal_raw is None and not BW.tip_is_measured(t) and t.gripper_raw == 2041
        handed = TL.Tool.from_dict({**t.to_dict(), "on_jaw": "moving_outer", "gripper_goal_raw": 2016, "tip_measured": dict(MEASURED)})
        _newer_tool(handed)
        r, _ = cli("--sim", "status")
        assert [st["done"] for st in r["steps"]] == [True, True, False, False] and r["next"].startswith("fd paint brush --kind round")
        r, _ = cli("--sim", "contact", "--adopt")
        m = json.load(open(SETUP_PATH))["contact"]["measured"]
        assert check_result(r, ok=True)["saved"] and m["tip_offset"] == [0.016, -0.0011, 0.0558] and m["gripper_goal_raw"] == 2016
        assert m["pitch_deg"] == 65.0 and json.load(open(SETUP_PATH))["contact"]["z0"] == 0.00542
    use_tool(thin)
    use_setup()
    reset_state()


def test_40_the_state_file_is_validated_and_an_unreadable_one_is_never_written_over():
    """Finding 27. A state file that was not JSON counted silently as empty and was written over by the next
    command: the progress of 19 layers gone, a finished layer would start at stroke 0 again."""
    use_tool("thin")
    use_setup(make_setup(record=tool("thin")))
    reset_state()
    path = PS.state_path(True)
    plan = write_plan("state_bad.json", [("a", "red", bands(2, press=0.6))])
    good = {"paint": None, "since_load_mm": 1000000.0, "at": "2026-11-13T19:10:15",
            "progress": {PS.progress_key(plan, json.load(open(plan)), 0): 2}, "blot_k": 3,
            "timing": {"strokes": 20.0, "strokes_s": 140.0, "runs": 1, "at": "2026-11-13T19:10:15"},
            "tool": {"name": "brush_thin", "attached_at": "2026-11-13T17:17:45"}}
    PS.save_state(good, sim=True)
    assert PS.load_state(sim=True) == good and PS.check_state(copy.deepcopy(good)) == good
    rig = os.path.join(BACKUP, "paint_state.json")
    if os.path.exists(rig):                                # the rig's own state file is a state
        assert len(PS.check_state(json.load(open(rig)))["progress"]) == 19
    whole = open(path, "rb").read()
    bad_files = {"cut off in the middle": whole[:-40], "not JSON at all": b"paint: black\n", "a list": b"[1, 2]",
                 "empty": b""}
    bad_values = {"blot_k": "three", "progress": [1, 2], "timing": {"strokes": "many"}, "paint": 3, "tool": "brush_thin",
                  "since_load_mm": "far"}
    cases = list(bad_files.items()) + [(f"{k} = {v!r}", json.dumps({**good, k: v}).encode()) for k, v in bad_values.items()]
    cases.append(("a negative progress", json.dumps({**good, "progress": {"x|a|0123456789": -1}}).encode()))
    for what, raw in cases:
        with open(path, "wb") as f:
            f.write(raw)
        for argv in (("status",), ("look",), ("rinse",), ("run", plan, "--layer", "0", "--dry"), ("run", plan, "--layer", "0"),
                     ("preview", plan, "--out", os.path.join(_TMP, "state_bad.png")), ("brush", "--carries", "none")):
            r, made = cli("--sim", *argv)
            check_result(r, ok=False, stage="state")       # (no traceback: check_result)
            assert path in r["reason"] and ("repair the file" in r["next"] and "move it aside" in r["next"]), (what, argv, r)
            assert not made, (what, argv)                  # refused before any arm was built
        assert open(path, "rb").read() == raw, what        # ... and the file is byte for byte what it was
    assert not os.path.exists(os.path.join(_TMP, "state_bad.png"))
    # save_state never writes over a file it could not read: it is kept beside the new one
    with open(path, "wb") as f:
        f.write(bad_files["cut off in the middle"])
    PS.save_state({"paint": "red"}, sim=True)
    kept = [n for n in os.listdir(_TMP) if n.startswith("paint_state.sim.damaged_")]
    assert len(kept) == 1 and open(os.path.join(_TMP, kept[0]), "rb").read() == bad_files["cut off in the middle"]
    assert PS.load_state(sim=True) == {"paint": "red"}
    os.remove(os.path.join(_TMP, kept[0]))
    # ... and stores nothing that it could not read back
    stamp = open(path, "rb").read()
    for bad in ({"progress": [1]}, {"paint": 7}, [1, 2]):
        try:
            PS.save_state(bad, sim=True)
        except PS.Refusal as e:
            assert e.stage == "state"
        else:
            raise AssertionError(f"stored {bad!r} as a painting state")
    assert open(path, "rb").read() == stamp
    # no file at all is an empty state, as it always was
    reset_state()
    assert PS.load_state(sim=True) == {} and PS.load_state(sim=False) == {}
    r, _ = cli("--sim", "status")
    assert check_result(r, ok=True)["state"] == {}


def test_41_check_takes_the_blotting_strip_as_the_paper_it_is():
    """Finding 29. `fd paint check` put the floor line of a free pose on the blotting strip, where the executor
    makes contact strokes: the strip at table height that the Akabeko was painted with was "out of reach"."""
    use_tool("flat")
    morning = make_setup(brush=None, record=tool("flat"), pitch=72.1)      # (the flat brush paints this canvas at 72.1 deg)
    morning["blot"] = {"xy": [0.354, 0.082], "size": [0.03, 0.066], "z": 0.0}      # paint_setup.before_thin_1700.json
    use_setup(morning)
    reset_state()
    r, made = cli("--sim", "check")
    check_result(r, ok=True, stage="check")
    blot = next(i for i in r["items"] if i["kind"] == "blot")
    assert blot["ok"] and blot["heights_mm"]["low"] == -1.5 and r["problems"] == [] and not moved(made["bus"]), blot
    r, made = cli("--sim", "test", "--do", "blot:2")       # what the executor does there, on the simulated arm
    assert check_result(r, ok=True, stage="test")
    low = tips(made["arm"], made["bus"])
    on_strip = (np.abs(low[:, 0] - 0.354) < 0.016) & (np.abs(low[:, 1] - 0.082) < 0.034)
    assert float(low[on_strip, 2].min()) < 0.0             # the brush was pressed onto it
    # a strip that lies too low for brush work IS a problem, and the advice is not "move it nearer"
    morning["blot"]["z"] = -0.006
    use_setup(morning)
    r, _ = cli("--sim", "check")
    check_result(r, ok=False, stage="check")
    blot = next(i for i in r["items"] if i["kind"] == "blot")
    assert r["problems"] == ["blot"] and blot["cause"] == "too low" and "below the limit of brush work" in blot["why"], blot
    assert "lies too low" in r["next"] and "blot.z" in r["next"] and "nearer to the robot" not in r["next"] and "低すぎ" in r["say"]
    # a well that is too low for a free pose keeps its floor check; one that is too far its "nearer to the robot"
    far = make_setup(brush=None, record=tool("flat"), pitch=72.1)
    far["blot"]["xy"] = [0.44, 0.28]                       # 49 cm from the pan axis
    use_setup(far)
    r, _ = cli("--sim", "check")
    blot = next(i for i in r["items"] if i["kind"] == "blot")
    assert check_result(r, ok=False)["problems"] == ["blot"] and blot["cause"] == "reach" and "nearer to the robot" in r["next"]
    use_tool("thin")
    use_setup()
    reset_state()


def test_42_preview_reports_its_picture_when_the_tool_cannot_be_used():
    """Finding 34. `fd paint preview` wrote its picture, then met a missing or damaged tool file and answered
    "the preview was not run": 620 kB on the disk and no word of it; the `next` was `fd paint status`, which
    refused the same file."""
    use_setup(make_setup(record=tool("thin")))
    reset_state()
    plan = write_plan("no_tool.json", [("a", "red", bands(3))])
    for what, content in (("no tool file", None), ("an empty tool file", ""), ("a tool file without a tip", "{}"),
                          ("a tool file that is no JSON", "{ name: brush")):
        TL.clear()
        if content is not None:
            with open(TOOL_PATH, "w") as f:
                f.write(content)
        out = os.path.join(_TMP, "pv_tool", "p.png")
        if os.path.exists(out):
            os.remove(out)
        r, made = cli("--sim", "preview", plan, "--out", out)
        check_result(r, ok=False, stage="tool")
        assert r["picture"] == out and cv2.imread(out) is not None and r["reach_checked"] is False, (what, r)
        assert "絵は作りました" in r["say"] and "確認していません" in r["say"] and r["layers"] == [], (what, r["say"])
        if content is None:
            assert "fd tool attach" in r["next"], r["next"]
        else:
            assert "cannot be used" in r["reason"] and "fd tool detach" in r["next"] and "repair the tool file" in r["next"], r
            # the other commands meet the same file: a refusal at "tool" with that next, not "start" and `fd paint status`
            for argv in (("status",), ("check",), ("look",), ("rest",), ("run", plan, "--layer", "0", "--dry"), ("contact", "--adopt")):
                r2, made2 = cli("--sim", *argv)
                check_result(r2, ok=False, stage="tool")
                assert "fd tool detach" in r2["next"] and "fd paint status" not in r2["next"] and "ツールのファイル" in r2["say"], (what, argv, r2)
    os.remove(TOOL_PATH)
    use_tool("thin")
    # with a tool it is the whole preview again
    r, _ = cli("--sim", "preview", plan, "--out", os.path.join(_TMP, "pv_tool", "p.png"))
    assert check_result(r, ok=True)["strokes"] == 3 and "reach_checked" not in r


def real_cli(*argv, prep=None):
    """paintcli.main WITHOUT --sim, the simulated arm standing in for the real one: the command believes that it
    drives the real arm (its state file is the tests' real one, its seconds go into the time model); there is
    no camera. Returns (result, {"arm", "bus"})."""
    made = {}

    def make(sim, tool="sim"):
        bus = RecBus(realtime=False)
        arm = Arm(bus=bus, cfg=load_config(), tool=TL.active(sim=False))
        arm.open()
        TL.sim_start(arm)
        if prep is not None:
            prep(arm)
        made.update(arm=arm, bus=bus)
        return arm

    with patched(paintcli, "_make_arm", make), patched(paintcli, "_camera", lambda sim: None):
        res = paintcli.main([str(a) for a in argv])
    _RESULTS.append(res)
    return res, made


def test_43_only_runs_that_ended_ok_feed_the_time_means():
    """Finding 35. A stopped run was added to the running means whole: the seconds of the load that failed and
    of the way out became overhead per stroke (4.0 s -> 9 s after four such runs), and nothing took them out."""
    use_tool("thin")
    use_setup(make_setup(record=tool("thin")))
    reset_state()
    plan = write_plan("means.json", [("six", "red", bands(6, press=0.6))])
    fit = PS.fit_times()
    # the rule itself
    state = {}
    counts, spent = {"strokes": 1, "travel_mm": 40, "loads": 1, "blots": 0, "washes": 0}, {"travel_s": 2.0, "load_s": 37.0}
    assert PS.note_run(state, counts, 140.0, spent, ok=False) is None and state == {}
    assert PS.note_run(state, counts, 140.0, spent, ok=True)["runs"] == 1
    # through the command, as on the real arm: a run that ends ok is counted
    r, made = real_cli("run", plan, "--layer", "0", "--stop", "2")
    check_result(r, ok=True, stage="run")
    assert "sim" not in r and r["strokes"] == 2
    t1 = PS.load_state(sim=False)["timing"]
    assert t1["runs"] == 1 and t1["strokes"] == 2.0 and t1["loads"] == 1.0 and t1["strokes_s"] > 0.0, t1
    # a run that is stopped, interrupted, or loses its servos is NOT: the means stay what they were
    for exc in (MotionAborted("target not reached: test", info={"reason": "target_not_reached"}), KeyboardInterrupt(),
                BusError("no answer from servo 4 (wrist_flex): test")):
        start = PS.progress_of(PS.load_state(sim=False), plan, json.load(open(plan)), 0)
        with _break_stroke(2, exc):
            r, made = real_cli("run", plan, "--layer", "0")
        assert r["ok"] is False and r["next_stroke"] == start + 1 and r["seconds"] > 30, r      # one stroke, then the stop
        st = PS.load_state(sim=False)
        assert st["timing"] == t1, (type(exc).__name__, st["timing"])
        assert PS.times_of(st)["per_stroke_s"] == PS.times_of({"timing": t1})["per_stroke_s"]
    # (the progress of the stopped runs IS stored: one stroke each)
    assert PS.progress_of(PS.load_state(sim=False), plan, json.load(open(plan)), 0) == 5
    # a run that ends at its time limit ended ok: counted
    r, made = real_cli("run", plan, "--layer", "0", "--start", "0", "--max-min", "0.5")
    assert check_result(r, ok=True)["time_limit"] and r["strokes"] >= 1
    t2 = PS.load_state(sim=False)["timing"]
    assert t2["runs"] == 2 and t2["strokes"] == 2.0 + r["strokes"]
    # a simulated run never feeds them (as before)
    r, _ = cli("--sim", "run", plan, "--layer", "0", "--start", "0", "--stop", "1")
    assert r["ok"] and "timing" not in PS.load_state(sim=True) and PS.load_state(sim=False)["timing"] == t2
    assert abs(PS.times_of({})["per_stroke_s"] - fit["per_stroke_s"]) < 1e-9
    reset_state()


def test_44_strokes_that_are_no_pairs_and_paints_that_are_no_text_are_plan_problems():
    """Finding 38. "pts": [[10, 10, 3], [20, 20, 3]] (reshaped into three points) and "paint": ["x"] passed
    plan_problems; preview and run then stopped with "internal error" and a trace."""
    use_tool("thin")
    s = use_setup(make_setup(record=tool("thin")))
    reset_state()
    good = json.load(open(write_plan("pairs_ok.json", [("a", "red", bands(2)), ("b", "red", bands(1, v0=90.0))])))
    assert PS.plan_problems(good, s) == []
    cases = {"three numbers per point": (lambda p: p["layers"][0]["strokes"][0].update(pts=[[10, 10, 3], [20, 20, 3]]), "a[0]: not a stroke"),
             "a flat list of numbers": (lambda p: p["layers"][0]["strokes"][1].update(pts=[10, 10, 20, 20]), "a[1]: not a stroke"),
             "points of unequal length": (lambda p: p["layers"][1]["strokes"][0].update(pts=[[10, 10], [20]]), "b[0]: not a stroke"),
             "a paint that is a list": (lambda p: p["layers"][0].update(paint=["x"]), "layer 'a': its paint must be the name"),
             "a paint that is a number": (lambda p: p["layers"][1].update(paint=7), "layer 'b': its paint must be the name")}
    for what, (edit, word) in cases.items():
        plan = copy.deepcopy(good)
        edit(plan)
        probs = PS.plan_problems(plan, s)
        assert len(probs) == 1 and word in probs[0], (what, probs)
        path = os.path.join(_TMP, "pairs_bad.json")
        with open(path, "w") as f:
            json.dump(plan, f)
        for argv in (("preview", path, "--out", os.path.join(_TMP, "pairs.png")), ("run", path, "--layer", "0", "--dry"),
                     ("run", path, "--layer", "1")):
            r, made = cli("--sim", *argv)
            check_result(r, ok=False, stage="plan")        # (check_result: no trace, no "internal")
            assert word in (r["reason"] + " ".join(r.get("problems", []))) and not moved(made["bus"]), (what, argv, r["reason"])


def test_45_rest_checks_its_clearance_writes_its_own_next_and_reports_a_fold():
    """Findings 40 and 50 (the painting side)."""
    thin = use_tool("thin")
    s = use_setup(make_setup(record=thin))
    reset_state()
    # 40: a clearance out of range is said as that, not as "no folded pose keeps the brush off the canvas"
    for c in ("9", "0.02", "0.2", "nan"):
        for extra in ((), ("--dry",)):
            r, made = cli("--sim", "rest", "--clearance", c, *extra)
            check_result(r, ok=False, stage="options")
            assert "--clearance" in r["reason"] and "0.03 .. 0.15" in r["reason"] and "take the sheet away" not in r["next"], r
            assert not moved(made["bus"]) and not made["arm"].is_torqued()
    arm, bus = sim_arm("thin")
    try:
        PS.rest_choice(arm, s, None, 0.2)
    except PS.Refusal as e:
        assert e.stage == "options"
    else:
        raise AssertionError("a clearance of 20 cm was taken")
    for c in ("0.03", "0.06"):
        r, _ = cli("--sim", "rest", "--clearance", c, "--dry")
        assert check_result(r, ok=True, stage="planned")["clearance_mm"] == round(float(c) * 1000)
    # in range, but no fold of this arm holds the tip that high: said as that (not "take the sheet away")
    r, made = cli("--sim", "rest", "--clearance", "0.15")
    check_result(r, ok=False, stage="rest")
    assert "cannot fold with the brush tip 150 mm above the table" in r["reason"] and "take the sheet away" not in r["next"], r
    assert not moved(made["bus"]) and "`fd paint rest` without --clearance" in r["next"]          # (a height WAS asked for)
    # T5 (third review). A tool the arm cannot fold with at ALL (10 cm across the fingers; no height was asked
    # for): the next of the refusal was `fd paint rest` without --clearance, the very command that was refused,
    # and `fd calib-air`, which holds the arm when no fold is safe, leads here. Now: what tooling says for such a
    # tool (the arm stays as it is; support it, take the tool off, `fd tool detach`), in next and in the sentence
    mark = {}

    def holds(arm_):                   # (where `fd calib-air` leaves the arm when it does not fold: holding)
        place(arm_, np.radians(PHOTO_Q))
        arm_.torque_on()
        arm_.bus.sleep(0.6)
        mark.update(writes=arm_.bus.io["writes"], log=len(arm_.bus.log), q=arm_.read_q())

    for across in (tool("thin", tip_offset=[0.0999, 0.0, 0.045]), tool("thin", tip_offset=[0.0999, 0.0, 0.045], tip_measured=None)):
        use_tool(across)
        for extra in ((), ("--dry",), ("--keep-torque",)):
            r, made = cli("--sim", "rest", *extra, prep=holds)
            check_result(r, ok=False, stage="rest")
            assert "cannot fold with the brush tip 40 mm above the table at any pan" in r["reason"] and r["cannot_fold"] is True, r
            assert not re.findall(r"`fd paint rest[^`]*`", r["next"]) and "--clearance" not in r["next"], r["next"]
            assert "`fd tool detach`" in r["next"] and "upport the arm by hand" in r["next"] and "`fd tool status`" in r["next"], r["next"]
            assert "fd tool detach" in r["say"] and "手で支えて" in r["say"] and "動かしていません" in r["say"], r["say"]
            assert made["arm"].is_torqued() and made["bus"].io["writes"] == mark["writes"] and len(made["bus"].log) == mark["log"], extra
            assert float(np.max(np.abs(made["arm"].read_q() - mark["q"]))) < math.radians(0.2), extra       # it holds where it stood
    use_tool(thin)
    use_setup(make_setup(record=thin, rest={"clearance_m": 0.15}))             # the height comes from the set-up: said as that
    r, made = cli("--sim", "rest")
    assert check_result(r, ok=False, stage="rest") and "\"rest\"" in r["next"] and "in the set-up" in r["next"] and not moved(made["bus"]), r["next"]
    use_setup(s)
    # 50: `fd paint rest` writes its own `next`. tooling.rest's names `fd tool rest`, whose fold is over the picture
    def short_of_the_fold(arm_, **kw):
        return {"ok": False, "stage": "rest", "torque": "on", "moved": True, "reason": "the arm did not end in the rest pose: test",
                "next": "the torque is still ON and the arm holds; `fd state`, then `fd tool rest` again"}

    def sagged(arm_, **kw):
        return {"ok": False, "stage": "rest", "torque": "off", "moved": True, "sag_after_release_mm": 31.0,
                "reason": "after the torque went off the arm sagged: the tool tip is 12 mm above the table",
                "next": "support the arm by hand; `fd torque on` holds it where it is, then `fd tool rest --clearance 0.07`"}

    for fake, words in ((short_of_the_fold, ("fd state", "fd paint rest --pan -89.2")),
                        (sagged, ("fd torque on", "fd paint rest --pan -89.2 --clearance 0.06"))):
        with patched(TL, "rest", fake):
            r, _ = cli("--sim", "rest")
        check_result(r, ok=False, stage="rest")
        assert all(w in r["next"] for w in words) and "`fd tool rest`" not in r["next"], r["next"]
        assert "fd tool rest --clearance" not in r["next"] and "then `fd tool rest` again" not in r["next"]
    # rest_report: the question `fd tool rest` and `fd calib-air` have to ask before they fold a brush
    arm, bus = sim_arm("thin")
    rep = PS.rest_report(arm, arm.tool, s, None, None)     # tooling's own fold: pan 20, 5 cm
    assert rep["ok"] is False and rep["pan_deg"] == TL.REST_PAN_DEG == 20.0 and rep["clearance_m"] == TL.REST_CLEARANCE
    assert rep["margin_mm"] == 30.0 and rep["clear_of_canvas_mm"] < 30.0 and "canvas" in rep["why"] and "fd paint rest" in rep["next"]
    assert abs(min(PS.canvas_distance(s, *rep["tip_xy"]), PS.canvas_distance(s, *rep["sink_xy"])) * 1000.0 - rep["clear_of_canvas_mm"]) < 0.06
    assert PS.canvas_distance(s, *rep["tip_xy"]) < 0.0 and "INSIDE the canvas" in rep["why"]         # 40 mm inside (2026-11-13)
    assert set(rep) >= {"ok", "why", "tip_xy", "sink_xy", "margin_mm"}
    good = PS.rest_report(arm, arm.tool, s, -100.0, 0.04)   # where the arm was put away by hand that evening
    assert good["ok"] is True and good["why"] == "" and good["clear_of_canvas_mm"] >= 30.0
    chosen = PS.rest_choice(arm, s)
    assert PS.rest_report(arm, arm.tool, s, chosen["pan_deg"], chosen["clearance_m"])["ok"]
    assert PS.rest_report(arm, None, None, 20.0, 0.05)["ok"] is False                      # the arm's tool, the set-up file
    assert PS.rest_report(arm, arm.tool, s, 20.0, 9.0)["ok"] is False                      # a fold the arm cannot make
    os.remove(SETUP_PATH)
    none = PS.rest_report(arm, arm.tool, None, 20.0, 0.05)
    assert none["ok"] is True and "no painting set-up" in none["why"]
    with open(SETUP_PATH, "w") as f:
        f.write("{ not json")
    assert PS.rest_report(arm, arm.tool, None, 20.0, 0.05)["ok"] is True                   # (nothing known: nothing to refuse)
    assert not moved(bus) and not arm.is_torqued()
    for kind in ("flat", "side"):                          # with any brush the tool's own fold is over this canvas
        arm, bus = sim_arm(kind)
        assert PS.rest_report(arm, arm.tool, s, None, None)["ok"] is False, kind
    use_setup()


def test_46_what_the_brush_carries_belongs_to_the_tool_it_was_written_for():
    """Finding 41. The recorded brush content was not tied to the tool: a brush that was swapped kept the old
    paint (and was "reloaded" with the short recipe), and only a wash by the arm could clear the record."""
    thin = use_tool("thin")
    use_setup(make_setup(record=thin))
    reset_state()
    r, _ = cli("--sim", "test", "--do", "load:black")
    st = PS.load_state(sim=True)
    assert check_result(r, ok=True) and st["paint"] == "black" and st["tool"] == {"name": "brush_thin", "attached_at": "2026-11-13T17:17:45"}
    r, _ = cli("--sim", "status")
    assert r["carries"] == "black" and "「black」が付いています" in r["say"]
    # another brush is taped on the next morning: what the old one carried is not what this one carries
    new = tool("thin", name="brush_round2", attached_at="2026-11-14T08:30:00")
    use_tool(new)
    use_setup(make_setup(record=new))
    r, _ = cli("--sim", "status")
    check_result(r, ok=True, stage="status")
    assert r["carries"] is None and r["state"]["paint"] == "black" and "不明" in r["say"], r["say"]
    assert any("that was tool 'brush_thin'" in w and "--carries" in w for w in r["warnings"]), r["warnings"]
    scrubs, washed = [], []
    real_scrub, real_clean = PS.Painter.scrub, PS.Painter.clean

    def scrub(self, well, passes=4, amp=0.006, side=0.003):
        scrubs.append((well, passes))
        return real_scrub(self, well, passes, amp, side)

    with patched(PS.Painter, "scrub", scrub), patched(PS.Painter, "clean", lambda self: (washed.append(1), real_clean(self))[1]):
        r, _ = cli("--sim", "test", "--do", "load:black")
    check_result(r, ok=True, stage="test")
    # the FIRST recipe (3 passes), not the reload (2); and, R10, WASHED before it: what this brush carries is not
    # known, which is not the same as "nothing"
    assert scrubs == [("black", 3)] and washed == [1], (scrubs, washed)
    assert any("that was tool 'brush_thin'" in w and "NOT KNOWN" in w for w in r["warnings"]) and "not known" in r["washed_first"]
    assert "洗いました" in r["say"] and "paint_unknown" not in PS.load_state(sim=True)
    assert PS.load_state(sim=True)["tool"] == {"name": "brush_round2", "attached_at": "2026-11-14T08:30:00"}
    r, _ = cli("--sim", "status")
    assert r["carries"] == "black" and not any("that was tool" in w for w in r["warnings"])
    # `fd paint brush --carries none|PAINT`: the record by hand (a brush rinsed at the tap), file only
    stamp = open(PS.state_path(True), "rb").read()
    r, made = cli("--sim", "brush", "--carries", "none", "--dry")
    assert check_result(r, ok=True, stage="planned") and r["saved"] is False and open(PS.state_path(True), "rb").read() == stamp and not made
    r, made = cli("--sim", "brush", "--carries", "none")
    check_result(r, ok=True, stage="brush")
    st = PS.load_state(sim=True)
    assert r["saved"] and r["carried"] == "black" and r["carries"] is None and st["paint"] is None and not made and r["sim"] is True
    assert st["tool"]["name"] == "brush_round2" and "何も付いていない" in r["say"]
    r, _ = cli("--sim", "brush", "--carries", "red_pale")
    st = PS.load_state(sim=True)
    assert check_result(r, ok=True)["carries"] == "red_pale" and st["paint"] == "red_pale" and st["used_mm"] >= 1e6 - 1
    plan = write_plan("carries.json", [("a", "red_pale", bands(1, press=0.6)), ("b", "black", bands(1, v0=90.0, press=0.6))])
    r, _ = cli("--sim", "run", plan, "--layer", "0", "--dry")
    assert r["budget"]["loads"] == 1 and r["budget"]["washes"] == 0      # it carries the paint, how much is not known: a reload
    r, _ = cli("--sim", "run", plan, "--layer", "1", "--dry")
    assert r["budget"]["washes"] == 1                                     # another paint: washed first
    for argv, stage in ((("--carries", "pink"), "paint"), (("--carries", "none", "--kind", "round"), "options")):
        before = open(PS.state_path(True), "rb").read()
        r, _ = cli("--sim", "brush", *argv)
        assert check_result(r, ok=False, stage=stage) and open(PS.state_path(True), "rb").read() == before, argv
    assert not os.path.exists(STATE_PATH)                  # (never the real arm's file)
    # an older state file does not say whose brush it describes: taken at its word, unless it was written before
    # the tool on the gripper was attached
    for at, known in (("2026-11-14T09:10:00", True), ("2026-11-13T19:10:15", False), (None, True)):
        PS.save_state({"paint": "black", **({"at": at} if at else {})}, sim=True)
        r, _ = cli("--sim", "status")
        assert (r["carries"] == "black") == known, (at, r["carries"])
    assert PS.carried_paint({"paint": None}, new) == (None, None) and PS.carried_paint({"paint": "red"}, None) == ("red", None)
    assert PS.carried_paint({}, new) == (None, None)                     # no state at all: an empty brush, as always
    TL.clear()
    r, _ = cli("--sim", "brush", "--carries", "none")
    assert check_result(r, ok=False, stage="tool")
    use_tool("thin")
    use_setup()
    reset_state()


def test_51_a_brush_whose_content_is_not_known_is_washed_before_its_first_load():
    """R10 (second review), a side effect of the fix for finding 41: a brush whose content is "not known" was
    treated as clean. The brush carries black, is taped on again and registered anew (only its attach time
    changes: that happened three times on 2026-11-13), and `load:yellow_pale` (water twice, a touch of the blob)
    went onto the black brush without a wash: dirty water, a spoiled well and tint."""
    thin = use_tool("thin")
    use_setup(make_setup(record=thin))
    again = tool("thin", attached_at="2026-11-14T08:30:00")              # the same brush, registered anew
    calls = []
    real = {name: getattr(PS.Painter, name) for name in ("clean", "water", "touch", "scrub")}

    def spy(name):
        def call(self, *a, **k):
            calls.append(name)
            return real[name](self, *a, **k)
        return call

    def load(*argv):
        del calls[:]
        with contextlib.ExitStack() as stack:
            for name in real:
                stack.enter_context(patched(PS.Painter, name, spy(name)))
            return cli("--sim", *argv)

    def black_then_retaped():
        reset_state()
        use_tool(thin)
        use_setup(make_setup(record=thin))
        r, _ = cli("--sim", "test", "--do", "load:black")
        assert check_result(r, ok=True) and PS.load_state(sim=True)["paint"] == "black"
        use_tool(again)
        use_setup(make_setup(record=again))

    # before the re-registration the wash comes first (as always: black is no tint of yellow)
    reset_state()
    cli("--sim", "test", "--do", "load:black")
    r, _ = load("test", "--do", "load:yellow_pale")
    assert check_result(r, ok=True) and calls[0] == "clean" and "washed_first" not in r, calls
    # after it: washed first all the same, and the result says why
    black_then_retaped()
    r, made = load("test", "--do", "load:yellow_pale")
    check_result(r, ok=True, stage="test")
    assert calls == ["clean", "water", "touch"], calls                    # the wash, THEN the recipe (water twice in one dip, a touch)
    assert "not known" in r["washed_first"] and "洗いました" in r["say"] and any("NOT KNOWN" in w and "washed before its first load" in w for w in r["warnings"])
    st = PS.load_state(sim=True)
    assert st["paint"] == "yellow_pale" and st["tool"] == PS.tool_id(again) and "paint_unknown" not in st
    r, _ = load("test", "--do", "load:yellow_pale")                       # known now: a reload, no wash, no word
    assert check_result(r, ok=True) and "clean" not in calls and "washed_first" not in r and "warnings" not in r
    # in a layer: the first load of the layer is preceded by the wash, the budget counts it, the result says it
    black_then_retaped()
    plan = write_plan("unknown.json", [("a", "yellow_pale", bands(2, press=0.6))])
    r, _ = cli("--sim", "run", plan, "--layer", "0", "--dry")
    assert check_result(r, ok=True, stage="planned")["budget"]["washes"] == 1 and any("NOT KNOWN" in w for w in r["warnings"])
    r, _ = cli("--sim", "preview", plan, "--out", os.path.join(_TMP, "unknown.png"))
    assert check_result(r, ok=True)["washes"] == 1
    r, _ = load("run", plan, "--layer", "0")
    assert check_result(r, ok=True)["washes"] == 1 and calls[0] == "clean" and "washed_first" in r and "洗いました" in r["say"]
    # a command that moves the brush without washing or loading it leaves it NOT KNOWN (it used to write "nothing"
    # into the state for the new tool): the next load still washes
    black_then_retaped()
    r, _ = cli("--sim", "look")
    st = PS.load_state(sim=True)
    assert check_result(r, ok=True) and st["paint"] is None and st["paint_unknown"] is True and st["tool"] == PS.tool_id(again), st
    r, _ = cli("--sim", "status")
    assert r["carries"] is None and "不明" in r["say"] and any("NOT KNOWN" in w for w in r["warnings"])
    r, _ = load("test", "--do", "load:yellow_pale")
    assert check_result(r, ok=True) and calls[0] == "clean" and "washed_first" in r
    # what makes it known without that wash: a wash by the arm (`fd paint rinse`), or the operator's word
    black_then_retaped()
    r, _ = cli("--sim", "rinse")
    assert check_result(r, ok=True) and PS.load_state(sim=True).get("paint") is None and "paint_unknown" not in PS.load_state(sim=True)
    r, _ = load("test", "--do", "load:yellow_pale")
    assert check_result(r, ok=True) and "clean" not in calls and "washed_first" not in r
    black_then_retaped()
    cli("--sim", "look")
    r, _ = cli("--sim", "brush", "--carries", "none")
    assert check_result(r, ok=True)["saved"] and "paint_unknown" not in PS.load_state(sim=True)
    r, _ = cli("--sim", "run", plan, "--layer", "0", "--dry")
    assert r["budget"]["washes"] == 0
    r, _ = load("test", "--do", "load:yellow_pale")
    assert check_result(r, ok=True) and "clean" not in calls and "washed_first" not in r
    # the record says NOTHING for the other tool (it was rinsed, then another brush was taped on): not known either
    reset_state()
    use_tool(thin)
    cli("--sim", "rinse")
    assert PS.load_state(sim=True)["paint"] is None and PS.load_state(sim=True)["tool"] == PS.tool_id(thin)
    use_tool(again)
    assert PS.carried_paint(PS.load_state(sim=True), again)[1] and PS.carried_paint(PS.load_state(sim=True), thin) == (None, None)
    r, _ = load("test", "--do", "load:yellow_pale")
    assert check_result(r, ok=True) and calls[0] == "clean" and "washed_first" in r
    # no state at all: an empty brush, as it always was (no wash); a state that is no true / false is refused
    reset_state()
    r, _ = load("test", "--do", "load:yellow_pale")
    assert check_result(r, ok=True) and "clean" not in calls and "washed_first" not in r
    try:
        PS.check_state({"paint": None, "paint_unknown": "yes"})
    except PS.Refusal as e:
        assert e.stage == "state"
    else:
        raise AssertionError("a paint_unknown that is no true / false was taken")
    # the painter itself
    arm, bus, pt = painter("thin", make_setup(record=thin))
    seen = _spy(pt)
    cleaned = []
    pt.clean = lambda: (cleaned.append(1), setattr(pt, "unknown", False))[0]
    pt.unknown = True
    r = pt.load("yellow_pale")
    assert cleaned == [1] and r["washed"] and r["content_was_unknown"] and pt.washed_unknown and not pt.unknown and pt.washes == 1
    r = pt.load("yellow_pale")
    assert cleaned == [1] and not r["washed"] and "content_was_unknown" not in r
    assert PS.UNKNOWN_PAINT not in BASE["paints"] and PS.layer_budget(make_setup(), {"paint": "red", "strokes": bands_as_strokes(1)}, PS.UNKNOWN_PAINT)["washes"] == 1
    use_tool("thin")
    use_setup()
    reset_state()


def test_47_test_strokes_and_ladders_keep_off_the_palette_and_the_dishes():
    """Finding 56. Test strokes were checked for reach and the two keep-outs only: a line ran through three wells
    of the palette 0.7 mm above the table, another across the rim into the water dish, a ladder in the red well."""
    thin = use_tool("thin")
    s = use_setup(make_setup(record=thin))
    reset_state()
    refused = (("line:0.2578,0.170:0.2638,0.262:1.2", "well"), ("line:0.290,0.150:0.340,0.186:1.2", "dish"),
               ("ladder:0.0716,0.2651", "the well 'red'"), ("stroke:0.21,0.19", "the well 'yellow'"),
               ("swatch:red:0.33,0.13", "dish"))
    for act, word in refused:
        for extra in (("--dry",), ()):
            r, made = cli("--sim", "test", "--do", act, *extra)
            check_result(r, ok=False, stage="options")
            assert word in r["reason"] and "--near-palette" in r["next"] and act in r["reason"], (act, r["reason"])
            assert not moved(made["bus"]) and r["brush"]["where"] == "not moved"
    r, made = cli("--sim", "test", "--ladder", "0.0716,0.2651", "--dry")
    assert check_result(r, ok=False, stage="options") and "the well 'red'" in r["reason"]
    for argv in (("--at", "0.0716,0.2651", "--dry"), ("--at", "0.0716,0.2651", "--scrap-laid")):
        r, made = cli("--sim", "contact", *argv)
        check_result(r, ok=False, stage="options")
        assert "the well 'red'" in r["reason"] and "--near-palette" in r["next"] and not moved(made["bus"]), r["reason"]
    # the word for a stroke that is meant to go there
    for act, _word in refused:
        r, made = cli("--sim", "test", "--do", act, "--near-palette", "--dry")
        assert check_result(r, ok=True, stage="planned"), act
    r, _ = cli("--sim", "contact", "--at", "0.0716,0.2651", "--near-palette", "--dry")
    assert check_result(r, ok=True, stage="planned")
    # what was always meant stays as it was: strokes on the sheet, the standard ladders of the canvas
    r, _ = cli("--sim", "test", "--do", "line:0.25,-0.05:0.25,-0.02:0.6", "--do", "ladder:0.2,0.04", "--dry")
    assert check_result(r, ok=True, stage="planned")
    r, _ = cli("--sim", "contact", "--dry")
    assert check_result(r, ok=True, stage="planned")
    # the rule: the whole stroke counts, not its ends; 4 cm from a well, 8 cm from a dish centre
    tiny = {"wells": {"a": {"xy": [0.20, 0.20]}}, "water": {"rinse": {"xy": [0.40, 0.40]}, "note": "one dish"}}
    assert PS.NEAR_WELL_M == 0.04 and PS.NEAR_DISH_M == 0.08
    assert "the well 'a'" in PS.near_palette(tiny, [(0.10, 0.20), (0.30, 0.20)])           # through it, both ends 10 cm away
    assert "35 mm" in PS.near_palette(tiny, [(0.10, 0.235), (0.25, 0.235)])
    assert PS.near_palette(tiny, [(0.10, 0.245), (0.25, 0.245)]) is None                   # 45 mm beside it
    assert "the rinse dish" in PS.near_palette(tiny, [(0.30, 0.40), (0.33, 0.40)]) and PS.near_palette(tiny, [(0.30, 0.40), (0.31, 0.40)]) is None
    assert PS.near_palette(tiny, [(0.22, 0.22)]) is not None and PS.near_palette(tiny, [(0.10, 0.10)]) is None
    assert PS.near_palette(s, [PS.canvas_xy(s, u, v) for u, v in ((0, 0), (224, 0), (224, 153), (0, 153), (0, 0))]) is None
    use_setup()
    reset_state()


def test_48_test_strokes_on_the_canvas_need_a_scrap_sheet_and_a_look():
    """R8 (second review). `fd paint test` painted strokes and ladders at typed coordinates without the word and
    without the look at the sheet that `fd paint contact` has, and the example in its own help lay inside the
    rig's canvas: typed as printed with the finished Kaminarimon uncovered, both lines were painted on it."""
    thin = use_tool("thin")
    s = use_setup(make_setup(record=thin))
    reset_state()
    x0, x1, y0, y1 = PS.canvas_rect(s)
    old = ("line:0.25,-0.05:0.25,-0.02:0.6", "line:0.26,-0.05:0.29,-0.05:0.6")       # the example the help used to give
    shots = []

    def camera_of(img):
        def shoot():
            shots.append(1)
            return img.copy()
        return shoot

    # 1. on the canvas, without the word: refused before anything moves, whatever the sheet looks like; --dry plans
    #    it and says what it needs
    cases = ((("--do", "load:red", "--do", old[0], "--do", old[1]), list(old)),
             (("--do", "ladder:0.2275,-0.024"), ["ladder:0.2275,-0.024"]),
             (("--do", "stroke:0.20,0.0"), ["stroke:0.20,0.0"]),
             (("--swatch", "red:0.2,0.05", "--ladder", "0.24,0.04"), ["swatch:red:0.2,0.05", "ladder:0.24,0.04"]),
             # a line that only clips a corner of the canvas, both ends outside it; one that runs 6 mm beside its edge
             (("--do", f"line:{x1 - 0.02:.3f},{y1 + 0.02:.3f}:{x1 + 0.02:.3f},{y1 - 0.02:.3f}:0.6"), None),
             (("--do", f"line:{x1 + 0.006:.3f},-0.05:{x1 + 0.006:.3f},-0.02:0.6"), None))
    for argv, names in cases:
        del shots[:]
        r, made = cli("--sim", "test", *argv, camera=camera_of(blank_photo()))
        check_result(r, ok=False, stage="contact")
        assert (names is None and len(r["on_canvas"]) == 1) or r["on_canvas"] == names, (argv, r["on_canvas"])
        assert "--scrap-laid" in r["next"] and r["next"].startswith("lay a scrap sheet") and "scrap sheet" in r["reason"], r
        assert not moved(made["bus"]) and not shots and r["brush"]["where"] == "not moved" and not made["arm"].is_torqued(), argv
        assert "捨て紙" in r["say"] and r["loads_total"] == 0 and PS.load_state(sim=True).get("paint") is None
        r, made = cli("--sim", "test", *argv, "--dry")
        check_result(r, ok=True, stage="planned")
        assert r["next"].startswith("lay a scrap sheet") and "--scrap-laid" in r["next"] and not moved(made["bus"]), r["next"]
        assert any("on the canvas of the set-up" in w for w in r["warnings"]) and "捨て紙" in r["say"]
        r, made = cli("--sim", "test", *argv, "--dry", "--scrap-laid")
        assert check_result(r, ok=True, stage="planned")["next"] == "the same command without --dry" and "warnings" not in r
    # 2. with the word, the sheet is looked at BEFORE anything is loaded. Paint where ONE of the strokes would
    #    land: refused, nothing loaded, the brush never came down, no second photo
    wins = [PS.stroke_window([(0.25, -0.05), (0.25, -0.02)]), PS.stroke_window([(0.26, -0.05), (0.29, -0.05)])]
    assert wins[1][0] < 0.26 - 0.02 and wins[1][1] > 0.29 and wins[1][2] < -0.05 < wins[1][3]         # (beyond its ends, beside it)
    # (paint where only the second line would land: its window, where it reaches beyond the first one's)
    stained = blank_photo()
    draw_patch(stained, 0.275, 0.287, wins[0][3] + 0.002, wins[1][3] - 0.002)
    del shots[:]
    r, made = cli("--sim", "test", "--scrap-laid", "--do", "load:red", "--do", old[0], "--do", old[1], camera=camera_of(stained))
    check_result(r, ok=False, stage="contact")
    assert [p["bare"] for p in r["sheet"]["places"]] == [True, False] and r["sheet"]["ok"] is False and old[1] in r["reason"], r["sheet"]
    assert "not bare paper" in r["reason"] and "FRESH scrap sheet" in r["next"] and "何も描いていません" in r["say"], r
    assert len(shots) == 1 and "photo" not in r and r["loads_total"] == 0 and r["did"] == [] and r["swatches"] == []
    assert r["brush"]["where"] == "photo pose" and float((tips(made["arm"], made["bus"])[:, 2] - TZ).min()) > 0.03
    assert PS.load_state(sim=True).get("paint") is None
    # a bare sheet: painted, and the look came first (photo, then the load, the strokes, the photo after)
    order = []
    real_load = PS.Painter.load

    def shoot():
        order.append("photo")
        return blank_photo()

    with patched(PS.Painter, "load", lambda self, paint: (order.append("load"), real_load(self, paint))[1]):
        r, made = cli("--sim", "test", "--scrap-laid", "--do", "load:red", "--do", old[0], "--do", old[1], camera=shoot)
    check_result(r, ok=True, stage="test")
    assert order == ["photo", "load", "photo"] and r["sheet"] == {"ok": True, "places": [
        {"name": old[0], "bare": True, "not_paper_pct": 0.0}, {"name": old[1], "bare": True, "not_paper_pct": 0.0}]}, (order, r["sheet"])
    assert r["did"] == ["load:red", *old] and len(r["swatches"]) == 2 and r["photos"]["before"] and r["photo"]
    low = tips(made["arm"], made["bus"])
    assert float(low[(np.abs(low[:, 0] - 0.25) < 0.003) & (np.abs(low[:, 1] + 0.035) < 0.012), 2].min()) < PS.contact_z(s, 0.25, -0.035)
    # the simulated arm has no camera: painted on the word alone, and the result says that nobody looked
    reset_state()
    r, made = cli("--sim", "test", "--scrap-laid", "--do", "load:red", "--do", old[0])
    assert check_result(r, ok=True, stage="test")["sheet"] is None and any("was not looked at" in w for w in r["warnings"])
    # 3. outside the canvas nothing changes: no word, no look (one photo: the one after)
    reset_state()
    out = ("line:0.325,-0.06:0.325,-0.03:0.6", "line:0.32,-0.02:0.345,-0.02:0.6")
    del shots[:]
    r, made = cli("--sim", "test", "--do", "load:red", "--do", out[0], "--do", out[1], camera=camera_of(desk_photo()))
    check_result(r, ok=True, stage="test")
    assert "on_canvas" not in r and "sheet" not in r and len(shots) == 1 and r["did"] == ["load:red", *out], r
    # the rule: the whole stroke counts; nearer than 1 cm to the canvas is on it; the window reaches where the paint lands
    assert PS.TEST_CANVAS_MARGIN_M == 0.010
    assert PS.on_canvas(s, [(0.25, 0.0), (0.26, 0.0)]) and PS.on_canvas(s, [(x1 + 0.009, 0.0), (x1 + 0.009, 0.03)])
    assert not PS.on_canvas(s, [(x1 + 0.011, 0.0), (x1 + 0.011, 0.03)]) and not PS.on_canvas(s, [(0.33, y0 - 0.05)])
    assert PS.on_canvas(s, [(x0 - 0.03, 0.0), (x1 + 0.03, 0.0)]) and PS.on_canvas(s, [(x1 - 0.02, y0 - 0.02), (x1 + 0.02, y0 + 0.02)])
    assert not PS.on_canvas(s, [(x1 + 0.02, y0 - 0.02), (x1 + 0.06, y0 - 0.02)])
    # 4. the help of the command: every line it spells lies OUTSIDE the rig's canvas, and its example, typed as
    #    printed, needs no word
    spelled = re.findall(r"line:(-?[0-9.]+),(-?[0-9.]+):(-?[0-9.]+),(-?[0-9.]+)", paintcli.NEW_BRUSH_HELP)
    assert len(spelled) >= 2 and "--scrap-laid" in paintcli.NEW_BRUSH_HELP and "ON THE CANVAS" in paintcli.NEW_BRUSH_HELP
    for a0, b0, a1, b1 in spelled:
        assert not PS.on_canvas(BASE, [(float(a0), float(b0)), (float(a1), float(b1))]), (a0, b0, a1, b1)
    example = re.search(r"fd paint test (--do load:red(?: --do line:[^ \n]+)+)", paintcli.NEW_BRUSH_HELP).group(1).split()
    r, made = cli("--sim", "test", *example, "--dry")
    assert check_result(r, ok=True, stage="planned") and "on_canvas" not in r and r["next"] == "the same command without --dry"
    assert old[0] not in paintcli.NEW_BRUSH_HELP
    # 5. the two real photos of 2026-11-13: the finished picture, uncovered (19:08), and the clean scrap sheet (17:39)
    reset_state()
    if not (os.path.exists(UNCOVERED_PHOTO) and os.path.exists(EVENING_PHOTOS[0])):
        raise Skip("the photos of 2026-11-13 (190853_run.jpg, 173916_look.jpg) are not here; the synthetic part passed")
    picture, scrap = cv2.imread(UNCOVERED_PHOTO), cv2.imread(EVENING_PHOTOS[0])
    for argv in (("--do", "load:red", "--do", old[0], "--do", old[1]), ("--do", "ladder:0.2275,-0.024")):
        r, made = cli("--sim", "test", "--scrap-laid", *argv, camera=lambda: picture.copy())
        check_result(r, ok=False, stage="contact")
        assert r["sheet"]["ok"] is False and r["loads_total"] == 0 and r["did"] == [], (argv, r["sheet"])
        assert float((tips(made["arm"], made["bus"])[:, 2] - TZ).min()) > 0.03
    r, made = cli("--sim", "test", "--scrap-laid", "--do", "load:red", "--do", old[0], "--do", old[1], camera=lambda: scrap.copy())
    assert check_result(r, ok=True, stage="test")["sheet"]["ok"] and r["did"] == ["load:red", *old]
    reset_state()
    use_setup()


THIN4 = os.path.join(tazuna_paths.ROOT, "runs", "20261113_110358", "aircal_thin4")      # the last tip measurement of the evening


def fd_cmd(*words) -> dict:
    """`fd WORDS` through fd.main itself (one JSON result on stdout; no run log from a test) -> the result + "_rc"."""
    import fd
    buf = io.StringIO()
    with patched(fd, "log_event", lambda *a, **k: None), contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        rc = fd.main([str(w) for w in words])
    lines = [ln for ln in buf.getvalue().splitlines() if ln.strip()]
    res = json.loads(lines[-1])
    res["_rc"] = rc
    return res


def test_49_the_order_status_leads_through_works_as_typed_on_the_rigs_own_files():
    """R6, the acceptance test of the hand-over. On copies of the rig's files as they stand (the jaw shut at raw
    2041 with nothing recorded, the tip typed, no brush block, a plane without a record), every `next` of
    `fd --sim paint status` is typed as printed, through fd itself: `fd calib-air` and `fd tool jaw` are the
    commands as they are, nothing is written by hand. The ONE step that needs a value the program does not know
    (the folder of the evening's last tip measurement) is said in words since the third review (T12: no
    placeholder to fill in): the operator types what the words say. It ends at check / preview / run, and no
    command on the way refuses. The order status gave before (jaw first) ended in a refusal at the third command:
    the evening's photos then count as taken "with no jaw hold on record", and the operator had to fly a new
    measurement on the real arm."""
    rig = {name: os.path.join(BACKUP, name) for name in ("tool_active.json", "paint_setup.json", "paint_state.json")}
    if not (all(os.path.exists(p) for p in rig.values()) and os.path.isdir(THIN4)):
        raise Skip("the rig's files (state_backup) or the photos runs/20261113_110358/aircal_thin4 are not here")

    def rig_copies():
        reset_state()
        shutil.copy(rig["tool_active.json"], TOOL_PATH)
        shutil.copy(rig["paint_setup.json"], SETUP_PATH)
        for sim in (False, True):                          # (the progress of the evening, for the real and the simulated arm)
            shutil.copy(rig["paint_state.json"], PS.state_path(sim))

    def typed(nxt):
        """The command a `next` gives, as the operator types it: the command it begins with, up to the remark,
        exactly as printed; and for the step that is said in words (PS.FROM_PHOTOS: "fd calib-air with --from, the
        folder of those photos, and --save") that command with the folder the operator knows."""
        if nxt.startswith("ONLY when photos of this brush exist") and nxt.split("   ")[0].endswith(PS.FROM_PHOTOS):
            return shlex.join(["fd", "calib-air", "--from", THIN4, "--save"])
        assert nxt.startswith("fd "), nxt
        return nxt.split("   ")[0].strip()

    rig_copies()
    t0 = TL.load(TOOL_PATH)
    assert t0.on_jaw is None and t0.gripper_goal_raw is None and t0.gripper_raw == 2041 and not BW.tip_is_measured(t0)
    done = []
    for _ in range(10):
        r = fd_cmd("--sim", "paint", "status")
        assert r["_rc"] == 0 and r["ok"] and r["sim"] is True, r
        if r["next"].startswith("`fd paint check`"):
            break
        cmd = typed(r["next"])
        assert _accepted(cmd) is None and not _placeholders(cmd), (cmd, _accepted(cmd))      # (a spelling the parsers take)
        res = fd_cmd("--sim", *shlex.split(cmd)[1:])
        assert res["_rc"] == 0 and res["ok"] is True, (cmd, res.get("stage"), res.get("reason") or res.get("message"))
        assert "trace" not in res, res
        if cmd.startswith("fd paint brush"):
            # T16 (third review): the next of an ok `fd paint brush` named `fd paint preview PLAN`, which is refused
            # at stage "contact" here (the plane has no record yet). It names status, which leads on
            assert res["next"].startswith("`fd paint status`") and "preview" not in res["next"], res["next"]
        done.append(cmd.replace(THIN4, "(the folder)"))
    else:
        raise AssertionError(f"`fd paint status` never came to check / preview / run: {done}")
    assert done == ["fd calib-air --from (the folder) --save", "fd tool jaw mount moving_outer", "fd tool jaw close",
                    "fd paint brush --kind round --fine 0.6:2.7 --broad 1.2:5.0 --load-mm 140", "fd paint contact --adopt"], done
    assert r["next"] == "`fd paint check` (reach); " + paintcli.PLAN_NEXT and "PLAN" not in r["next"], r["next"]
    assert "fd paint preview and that file" in r["next"] and "fd paint run with that file, --layer 0 and --dry" in r["next"]
    assert all(st["done"] for st in r["steps"]) and r["contact_record"] == {"recorded": True, "ok": True, "why": []}, r["steps"]
    # what the files say now: the tip measured from the photos (by calib-air itself), the jaw held, the plane
    # adopted for exactly that brush state; the rig's own files are what they were (the last test looks again)
    t = TL.load(TOOL_PATH)
    m = json.load(open(SETUP_PATH))["contact"]["measured"]
    assert BW.tip_is_measured(t) and t.on_jaw == "moving_outer" and t.gripper_goal_raw is not None and abs(t.gripper_raw - 2041) <= 3
    assert max(abs(a - b) for a, b in zip(t.tip_offset, t0.tip_offset)) < 0.0005                 # (the evening's tip, measured)
    assert m["adopted"] is True and m["tool"] == t.name and m["gripper_goal_raw"] == t.gripper_goal_raw and m["tip_measured"] is True
    assert json.load(open(SETUP_PATH))["brush"]["kind"] == "round"
    # ... and what status then names works: the reach, and a layer of the Kaminarimon as it was painted
    r = fd_cmd("--sim", "paint", "check")
    assert r["_rc"] == 0 and r["ok"] and r["stage"] == "check", (r.get("stage"), r.get("reason"))
    plan = os.path.join(tazuna_paths.ROOT, "runs", "20261113_paint", "kaminarimon_plan.json")
    if os.path.exists(plan):
        r = fd_cmd("--sim", "paint", "run", plan, "--layer", "7", "--dry")
        assert r["ok"] and r["stage"] == "planned" and r["start"] == r["of"] == 19, r       # the written lines: all 19 on the paper
        r = fd_cmd("--sim", "paint", "run", plan, "--layer", "0", "--dry")
        assert r["ok"] and r["stage"] == "planned", (r.get("stage"), r.get("reason"))
    # the other order (the jaw first, what status used to print): with a hold on record the photos of the evening
    # are refused, and status does not offer them any more: a new capture is what is left
    rig_copies()
    for words in (("tool", "jaw", "mount", "moving_outer"), ("tool", "jaw", "close")):
        res = fd_cmd("--sim", *words)
        assert res["ok"], (words, res.get("reason"))
    res = fd_cmd("--sim", "calib-air", "--from", THIN4, "--save")
    assert res["ok"] is False and res["stage"] == "save" and not BW.tip_is_measured(TL.load(TOOL_PATH)), res
    r = fd_cmd("--sim", "paint", "status")
    assert typed(r["next"]) == "fd calib-air --save" and "--from" not in r["next"], r["next"]
    # T9 (third review): status said here that those photos "cannot be stored any more", while the refusal just
    # above names the way back. Now every text gives the one order: refused as they are, the refusal says how
    assert "are refused as they are; the refusal names the way back" in r["next"] and "cannot be stored any more" not in r["next"], r["next"]
    assert [(st["step"], st["done"]) for st in r["steps"]][:2] == [("jaw", True), ("tip", False)]
    assert isinstance(res.get("next"), str) and res["next"], res         # (that refusal has a next of its own: aircal's text)
    use_tool("thin")
    use_setup()
    reset_state()


def test_50_rest_with_an_unmeasured_tip_folds_high_and_a_folded_arm_is_left_as_it_lies():
    """R7 and R21 (second review). `fd paint rest` is the one moving paint command without the tip gate, and every
    hint after a stop names it: it folded a tool whose tip was never measured by the REGISTERED tip, 4 cm above
    the table, and said nothing; a tool 9 cm longer than registered (what happened on 2026-11-13) ended 5 mm
    under the table surface. And an arm that already lay folded and limp was switched on and turned to another pan."""
    s = use_setup(make_setup(record=tool("thin")))
    reset_state()
    registered, real = [-0.001, 0.0, 0.0868], [-0.001, 0.0, 0.1768]         # 8 cm along the fingers; really 17 cm
    guess = tool("thin", name="brush_new", tip_offset=registered, mount="along_fingers", tip_measured=None,
                 attached_at="2026-11-14T09:00:00", **RIG_JAW)
    use_tool(guess)

    def real_heights(made):
        tl = K.tool_offset(*real)
        return np.array([K.fk(q_of(p), tl)[2, 3] - TZ for p in made["bus"].trace])

    def holding(arm_):                 # a place the tool as it REALLY is can be in (its own fold), holding
        place(arm_, TL.rest_q(arm_, tool("thin", tip_offset=real, mount="along_fingers", tip_measured=None)))
        arm_.torque_on()
        arm_.bus.sleep(0.6)
        arm_.bus.trace.clear()

    def limp(arm_):                    # ... and limp there (over the canvas: not a fold `fd paint rest` leaves)
        place(arm_, TL.rest_q(arm_, tool("thin", tip_offset=real, mount="along_fingers", tip_measured=None)))
        arm_.bus.trace.clear()

    for extra, torque, prep in (((), "off", holding), (("--keep-torque",), "on", holding), ((), "off", limp)):
        r, made = cli("--sim", "rest", *extra, prep=prep)
        check_result(r, ok=True, stage="rest")
        assert r["torque"] == torque and r["moved"] and r["tip_measured"] is False and r["brush"]["where"] == "rest", r
        assert r["clearance_mm"] >= 70 and r["brush"]["height_mm"] >= 70.0 and TL.in_rest_pose(made["arm"], made["arm"].tool)[0], r["clearance_mm"]
        assert any("never measured" in w and "REGISTERED tip" in w and "LOOK at the arm" in w for w in r["warnings"]), r["warnings"]
        assert r["next"].startswith("LOOK at the arm") and "`fd paint status`" in r["next"] and "fd paint look" not in r["next"], r["next"]
        assert "未測定" in r["say"] and "目で確認" in r["say"] and _rest_ok(s, made["arm"])
        h = real_heights(made)                             # the tool as it REALLY is: never on the table
        assert float(h.min()) > 0.015 and float(h[-1]) > 0.04, (extra, float(h.min()), float(h[-1]))
    r, made = cli("--sim", "rest", "--dry", prep=limp)
    check_result(r, ok=True, stage="planned")
    assert r["clearance_mm"] >= 70 and any("never measured" in w for w in r["warnings"]) and "未測定" in r["say"] and not moved(made["bus"])
    # the highest fold is one tooling takes for a rest pose (only there is the torque switched off), found from the top
    arm, bus = sim_arm(guess)
    high, low = PS.highest_rest(arm, s), PS.rest_choice(arm, s)
    assert high["clearance_m"] > low["clearance_m"] == 0.04 and TL.in_rest_pose(arm, guess, high["q"])[0] and high["clear_of_canvas_mm"] >= 30.0
    try:
        over = PS.rest_choice(arm, s, None, round(high["clearance_m"] + 0.01, 3))
    except PS.Refusal:
        over = None
    assert over is None or not TL.in_rest_pose(arm, guess, over["q"])[0]                # (a centimetre more is none)
    assert not moved(bus)
    # a height that is asked for is taken as asked, and the warning stands; a pan over the picture is still refused
    r, made = cli("--sim", "rest", "--clearance", "0.05", prep=holding)
    assert check_result(r, ok=True)["clearance_mm"] == 50 and any("never measured" in w for w in r["warnings"]) and "未測定" in r["say"]
    r, made = cli("--sim", "rest", "--pan", "20", prep=holding)
    assert check_result(r, ok=False, stage="rest") and "canvas" in r["reason"] and made["arm"].is_torqued()
    # a brush whose tip WAS measured: the fold of always (4 cm), no such word
    use_tool("thin")
    r, made = cli("--sim", "rest", prep=holding)
    check_result(r, ok=True, stage="rest")
    assert r["clearance_mm"] == 40 and "warnings" not in r and "tip_measured" not in r and "未測定" not in r["say"] and "fd paint look" in r["next"]
    # R21. The arm as it lies on 2026-11-14: folded at pan -100, 4 cm up, the torque off (put there by hand-typed
    # options the evening before), 157 mm clear of the canvas. It is left as it lies, and the result says so
    for t in (tool("thin"), tool("thin", tip_measured=None, **RIG_JAW)):
        use_tool(t)

        def lies(arm_):
            place(arm_, TL.rest_q(arm_, arm_.tool, clearance_m=0.04, pan_deg=-100.0))
            arm_.bus.trace.clear()

        for argv in (("rest",), ("rest", "--pan", "-100"), ("rest", "--dry"), ("rest", "--keep-torque")):
            r, made = cli("--sim", *argv, prep=lies)
            check_result(r, ok=True, stage="planned" if "--dry" in argv else "rest")
            assert r["moved"] is False and r["torque"] == "off" and not made["arm"].is_torqued() and not moved(made["bus"]), (argv, r)
            assert r["chosen"] == "as it lies" and r["pan_deg"] == -100.0 and r["brush"]["where"] == "rest" and "nothing moved" in r["note"]
            assert abs(r["brush"]["clear_of_canvas_mm"] - 157.0) < 3.0 and abs(r["brush"]["height_mm"] - 40.0) < 1.0, r["brush"]
            assert "すでに畳んであります" in r["say"] and "何も動かしていません" in r["say"] and "畳みました" not in r["say"], r["say"]
            assert (BW.tip_is_measured(t) and "warnings" not in r) or any("left as it lies" in w for w in r["warnings"])
            # T11 (third review): nothing was planned or folded here, and the next says so. It said "the fold was
            # planned with the registered tip"; and the next of --dry spelled `fd paint rest --pan P`, which typed
            # as printed is refused ('P' is no angle): the pan stands in words
            assert "the fold was planned" not in r["next"] and not re.findall(r"`[^`]*--pan P[^`]*`", r["next"]), r["next"]
            assert all(_accepted(c) is None for c in re.findall(r"`(fd [^`]+)`", r["next"])), r["next"]
            if "--dry" in argv:
                assert "leaves the arm as it lies too" in r["next"] and "--pan and an angle in degrees" in r["next"], r["next"]
            elif BW.tip_is_measured(t):
                assert "`fd paint look`" in r["next"] and "LOOK at the arm" not in r["next"], r["next"]
            else:
                assert r["next"].startswith("LOOK at the arm: it was left as it lies (nothing was planned or folded)"), r["next"]
                assert "is not known" in r["next"] and "never measured" in r["next"] and "`fd paint status`" in r["next"], r["next"]
        # another pan, or a height, is asked for: then it is folded there
        r, made = cli("--sim", "rest", "--pan", "-60", prep=lies)
        assert check_result(r, ok=True)["moved"] and r["pan_deg"] == -60.0 and r["chosen"] == "asked" and "畳みました" in r["say"]
        r, made = cli("--sim", "rest", "--clearance", "0.06", prep=lies)
        assert check_result(r, ok=True)["moved"] and r["clearance_mm"] == 60
    # limp, but over the picture (the tool's own rest pose, pan 20: where the simulated arm starts), or folded with
    # the torque ON: not "as it lies"
    use_tool("thin")
    arm, bus = sim_arm("thin")
    assert PS.rest_as_it_lies(arm, s) is None and not arm.is_torqued()
    place(arm, TL.rest_q(arm, arm.tool, clearance_m=0.04, pan_deg=-100.0))
    lying = PS.rest_as_it_lies(arm, s)
    assert lying["chosen"] == "as it lies" and lying["pan_deg"] == -100.0 and lying["clear_of_canvas_mm"] >= 30.0
    arm.torque_on()
    assert PS.rest_as_it_lies(arm, s) is None
    r, made = cli("--sim", "rest")                         # (from the tool's own rest pose, over the canvas: folded off it)
    assert check_result(r, ok=True)["moved"] and r["chosen"] == "nearest to the photo pose" and _rest_ok(s, made["arm"])
    # a fold tooling.rest itself leaves alone (asked for by pan AND height, the arm already there): said as that
    def there(arm_):
        place(arm_, TL.rest_q(arm_, arm_.tool, clearance_m=0.05, pan_deg=-100.0))

    r, made = cli("--sim", "rest", "--pan", "-100", "--clearance", "0.05", prep=there)
    assert check_result(r, ok=True)["moved"] is False and "すでに" in r["say"] and "畳みました" not in r["say"] and not moved(made["bus"])
    use_setup()


def test_52_the_next_of_run_keeps_what_the_layer_needs():
    """R12 (second review). After `fd paint run PLAN --layer 0 --lower-mm 1.5` the next was `fd paint run PLAN
    --layer 1`: the other bands were then painted without the lowered plane (they came out thin that way on
    2026-11-13). After `... --layer 7 --reload-mm 110 --max-min 0.5` it was `fd paint run PLAN --layer 7`: the
    written lines were then reloaded every 140 mm instead of every 110. And the plan's own "run" list (the
    command of every layer, written by the module that made the plan) was shown nowhere."""
    thin = use_tool("thin")
    use_setup(make_setup(record=thin))
    reset_state()
    layers = [("band0", "red", bands(2)), ("band1", "red", bands(2, v0=80.0)), ("lines", "ink", bands(6, v0=110.0, gap=5.0, press=0.6))]
    runs = ["fd paint run PLAN --layer 0 --lower-mm 1.5", "fd paint run PLAN --layer 1 --lower-mm 1.5", "fd paint run PLAN --layer 2 --reload-mm 110"]
    left_open = ["the far light is thin: `fd paint run PLAN --layer 0 --lower-mm 1.5`", "the lettering waits until the red is dry"]
    plan = write_plan("next.json", layers, run=runs, open=left_open)
    own = [c.replace("PLAN", plan) for c in runs]

    def named(r):
        """The `fd paint run ...` commands a result's next spells (in backticks, or at its beginning)."""
        found = re.findall(r"`(fd paint run [^`]+)`", r["next"])
        return found + ([r["next"].split("   ")[0]] if r["next"].startswith("fd paint run") else [])

    # a layer that is finished: the next layer's command is the PLAN's own (the plane lowered for the next band too)
    r, _ = cli("--sim", "run", plan, "--layer", "0", "--lower-mm", "1.5")
    assert check_result(r, ok=True, stage="run")["done"] and named(r) == [own[1]] and "warnings" not in r, r["next"]
    r, _ = cli("--sim", *shlex.split(named(r)[0])[2:])                     # ... and typed as printed it paints that layer
    assert check_result(r, ok=True)["done"] and r["contact_lowered_mm"] == 1.5 and named(r) == [own[2]], r["next"]
    # a layer that stops at its time limit: the same command again WITH its options
    r, _ = cli("--sim", "run", plan, "--layer", "2", "--reload-mm", "110", "--max-min", "0.5")
    check_result(r, ok=True, stage="run")
    k = r["next_stroke"]
    assert r["time_limit"] and 0 < k < 6 and r["reload"] == {"by": "travel", "limit_mm": 110.0}
    assert named(r) == [f"fd paint run {plan} --layer 2 --reload-mm 110 --max-min 0.5"] and f"continues at stroke {k}" in r["next"], r["next"]
    r2, _ = cli("--sim", *shlex.split(named(r)[0])[2:], "--dry")           # typed as printed: it goes on there, reloading as before
    assert check_result(r2, ok=True, stage="planned")["start"] == k and r2["next"] == named(r)[0] + "   (without --dry)"
    # an interrupted or stopped layer names it the same way
    with _break_stroke(1, KeyboardInterrupt()):
        r, _ = cli("--sim", "run", plan, "--layer", "2", "--reload-mm", "110", "--max-min", "3", "--lower-mm", "1")
    assert check_result(r, ok=False, stage="run")["interrupted"]
    assert named(r) == [f"fd paint run {plan} --layer 2 --lower-mm 1 --reload-mm 110 --max-min 3"], r["next"]
    assert any("the plan's own command for this layer is" in w and own[2] in w for w in r["warnings"]), r["warnings"]
    # --dry and preview show the plan's own commands and what its maker left open
    r, _ = cli("--sim", "run", plan, "--layer", "0", "--dry")
    check_result(r, ok=True, stage="planned")
    # (T12: what the plan's maker left open names its command with PLAN for the file; here the file is known)
    opened = [t.replace("PLAN", plan) for t in left_open]
    assert r["run"] == own and r["open"] == opened and r["next"] == f"fd paint run {plan} --layer 0   (without --dry)"
    assert [c for t in r["open"] for c in re.findall(r"`(fd [^`]+)`", t)] == [own[0]] and _accepted(own[0]) is None
    assert any(own[0] in w for w in r["warnings"])                          # (this call did not lower the plane: said)
    r, _ = cli("--sim", "run", plan, "--layer", "0", "--lower-mm", "1.5", "--reload-mm", "75", "--dry")
    assert r["next"] == f"fd paint run {plan} --layer 0 --lower-mm 1.5 --reload-mm 75   (without --dry)"
    r, _ = cli("--sim", "preview", plan, "--out", os.path.join(_TMP, "next.png"))
    assert check_result(r, ok=True)["run"] == own and r["open"] == opened and r["next"] == own[0] + " --dry"
    assert _accepted(r["next"]) is None and all(_accepted(c) is None for c in own)
    # a layer that is already painted: the same, to paint it again
    r, _ = cli("--sim", "run", plan, "--layer", "0", "--lower-mm", "1.5")
    assert r["strokes"] == 0 and named(r) == [f"fd paint run {plan} --layer 0 --lower-mm 1.5 --start 0"], r["next"]
    # a plan WITHOUT a "run" list: the plain command of the next layer, as before; the options of the call are
    # carried for the same layer only
    reset_state()
    plain = write_plan("next_plain.json", layers)
    r, _ = cli("--sim", "run", plain, "--layer", "0", "--lower-mm", "1.5", "--stop", "1")
    assert check_result(r, ok=True)["next_stroke"] == 1 and not r["done"] and "time_limit" not in r
    assert r["next"] == (f"this call ended at its --stop 1; `fd paint run {plain} --layer 0 --lower-mm 1.5` paints the rest "
                         "of the layer, from stroke 1"), r["next"]      # (the call did what it was limited to)
    r, _ = cli("--sim", "run", plain, "--layer", "0", "--lower-mm", "1.5")
    assert r["done"] and named(r) == [f"fd paint run {plain} --layer 1"] and "run" not in r
    # T6 (third review). A call that was limited by --stop and ended at its TIME limit named the command without
    # --stop: typed as printed it painted the rest of the layer, past the stroke the call was limited to (15
    # strokes instead of 4). The continuation carries --stop: typed as printed, again and again, it ends there
    r, _ = cli("--sim", "run", plain, "--layer", "2", "--stop", "4", "--max-min", "0.3")
    k = check_result(r, ok=True, stage="run")["next_stroke"]
    assert r["time_limit"] and k < 4 and not r["done"], (k, r.get("reason"))
    assert named(r) == [f"fd paint run {plain} --layer 2 --max-min 0.3 --stop 4"], r["next"]
    assert f"continues at stroke {k}" in r["next"] and "ends at stroke 4" in r["next"] and "--stop 4" in r["say"], (r["next"], r["say"])
    for _ in range(8):
        r, _ = cli("--sim", *shlex.split(named(r)[0])[2:])
        assert check_result(r, ok=True, stage="run")["next_stroke"] <= 4 and not r["done"], (r["next_stroke"], r["next"])
        if r["next_stroke"] == 4:
            break
    assert r["next_stroke"] == 4 and PS.progress_of(PS.load_state(sim=True), plain, json.load(open(plain)), 2) == 4
    assert r["next"] == (f"this call ended at its --stop 4; `fd paint run {plain} --layer 2 --max-min 0.3` paints the rest "
                         "of the layer, from stroke 4") and "--stop 4" in r["say"], (r["next"], r["say"])
    # the limited command once more paints nothing and names both: the strokes again, the rest of the layer
    r, made = cli("--sim", "run", plain, "--layer", "2", "--stop", "4", "--max-min", "0.3")
    assert check_result(r, ok=True)["strokes"] == 0 and r["next_stroke"] == 4 and not r["done"]
    assert named(r) == [f"fd paint run {plain} --layer 2 --max-min 0.3 --stop 4 --start 0", f"fd paint run {plain} --layer 2 --max-min 0.3"], r["next"]
    assert "strokes 0 to 3 again" in r["next"] and "limited to stroke 4" in r["next"]
    # --dry and an interrupted call keep the --stop too; a layer that is done names the next layer without it
    r, _ = cli("--sim", "run", plain, "--layer", "2", "--stop", "5", "--dry")
    assert check_result(r, ok=True, stage="planned")["next"] == f"fd paint run {plain} --layer 2 --stop 5   (without --dry)"
    # (round 3b, U4) ... and the --start of a dry run: without it the printed command began at the saved progress,
    # not at the stroke the dry run had planned from
    r, _ = cli("--sim", "run", plain, "--layer", "2", "--start", "1", "--stop", "3", "--dry")
    assert check_result(r, ok=True, stage="planned")["start"] == 1, r
    assert r["next"] == f"fd paint run {plain} --layer 2 --stop 3 --start 1   (without --dry)", r["next"]
    with _break_stroke(1, KeyboardInterrupt()):
        r, _ = cli("--sim", "run", plain, "--layer", "2", "--stop", "5")
    assert check_result(r, ok=False, stage="run")["interrupted"] and named(r) == [f"fd paint run {plain} --layer 2 --stop 5"], r["next"]
    r, _ = cli("--sim", "run", plain, "--layer", "1", "--stop", "2")       # (--stop at the layer's end: the layer is done)
    assert check_result(r, ok=True)["done"] and named(r) == [f"fd paint run {plain} --layer 2"], r["next"]
    r, _ = cli("--sim", "preview", plain, "--out", os.path.join(_TMP, "next.png"))
    assert r["next"] == f"fd paint run {plain} --layer 0 --dry" and "run" not in r and "open" not in r
    # a "run" list that does not fit the plan's layers is not trusted
    for bad in (runs[:2], ["fd paint run PLAN --layer 1"] * 3, ["rm -rf PLAN", "x", "y"], "fd paint run PLAN --layer 0", [1, 2, 3]):
        assert paintcli._plan_runs({"layers": [1, 2, 3], "run": bad}, plan) is None, bad
    assert paintcli._plan_runs({"layers": [1, 2, 3], "run": runs}, "/tmp/a plan.json")[2] == "fd paint run '/tmp/a plan.json' --layer 2 --reload-mm 110"
    # the Kaminarimon's remade plan, where this harness can build it: the commands of its nine layers
    try:
        import kaminarimon as KM
        nxt = KM.build(variant="next")
    except Exception:
        nxt = None
    if nxt is not None and nxt.get("run"):
        got = paintcli._plan_runs(nxt, "k.json")
        assert got and len(got) == len(nxt["layers"]) and got[0].startswith("fd paint run k.json --layer 0") and "--lower-mm 1.5" in got[0]
    reset_state()
    use_setup()


def test_53_the_paint_parser_has_no_sim_of_its_own():
    """R16 (second review). The paint parser had an option --sim of its own; argparse took `--si` and `--s` for it:
    `fd paint --si status` gave a simulated answer to a call fd took for a real one (fd refuses a --sim typed after
    the command), and the usage of `fd paint --help` showed [--sim] after `paint`, the spelling that is refused."""
    use_tool("thin")
    use_setup(make_setup(record=tool("thin")))
    reset_state()
    plan = write_plan("sim_words.json", [("a", "red", bands(1, press=0.6))])
    for argv in (("--si", "status"), ("--s", "status"), ("status", "--sim"), ("--sim", "--sim", "status"), ("--si", "look"),
                 ("run", plan, "--layer", "0", "--si"), ("look", "--s")):
        r, made = cli(*argv)
        check_result(r, ok=False, stage="options")
        assert not made and "sim" not in r and "`fd --sim paint status`" in r["next"] and "fd のすぐ後ろ" in r["say"], (argv, r)
    # fd's own --sim, handed over in front of the words, is the simulated arm as before (one leading "--" is dropped)
    for argv in (("--sim", "status"), ("--sim", "--", "status")):
        r, _ = cli(*argv)
        assert check_result(r, ok=True, stage="status")["sim"] is True, argv
    r, _ = cli("status")                                   # (files only: no arm, and no simulation)
    assert check_result(r, ok=True, stage="status") and "sim" not in r
    # the parser: no option --sim, no abbreviations at the top; the usage shows fd's option where it works
    parser = paintcli.build_parser()
    assert not any("--sim" in act.option_strings for act in parser._actions)
    sub = next(act for act in parser._actions if getattr(act, "choices", None) and "status" in act.choices)
    for name, p in [("", parser), *sub.choices.items()]:
        usage = " ".join(p.format_usage().split())
        assert usage.startswith(f"usage: fd [--sim] paint {name}".rstrip()) and "--sim" not in usage.partition(" paint")[2], usage
        assert not any(ln.strip().startswith("--sim") for ln in p.format_help().splitlines()), name
    # through fd itself: refused, not answered by the simulation
    for words in (("paint", "--si", "status"), ("paint", "--s", "status")):
        r = fd_cmd(*words)
        assert r["_rc"] == 1 and r["ok"] is False and r["stage"] == "options" and "sim" not in r, r
    r = fd_cmd("--sim", "paint", "status")
    assert r["_rc"] == 0 and r["sim"] is True


def test_54_locate_writes_no_well_where_the_photo_shows_no_paint():
    """T3 (third review). With the set-up 3 to 6 deg off the palette's turn, `fd paint locate` still wrote wells
    beside their paint: a well took the middle of the piece that two wells share, or a stray piece, as its own
    blob (places taken "by blob" were not tested for paint), and that ONE pair bent the fit of all the others:
    in 33 of 968 turns and pushes of eight real photos a well was written onto the white divider, in 52 more
    over 8 mm from its blob. Now: a piece whose middle is no paint is nobody's blob, a piece within reach of two
    wells is given to neither, and EVERY place that is written shows paint in the photo (else the well keeps
    its place, is not located, and nothing is loaded from it)."""
    use_tool("thin")
    s = use_setup(make_setup(record=tool("thin")))
    reset_state()

    def turned(turn, shift=(0.0, 0.0)):
        """The set-up as it stands after the palette was turned by -turn deg and pushed by -shift: its wells
        turned about the palette's middle and pushed."""
        t = copy.deepcopy(s)
        for n, xy in palette_places(s, shift, turn).items():
            t["wells"][n]["xy"] = [round(xy[0], 4), round(xy[1], 4)]
        return t

    def off_mm(found, n, truth=None):
        want = (truth or s)["wells"][n]["xy"]
        return math.hypot(found["wells"][n]["xy"][0] - want[0], found["wells"][n]["xy"][1] - want[1]) * 1000.0

    # 1. the last gate. Red's well is scrubbed empty in its middle (a ring of paint is left): the middle of that
    #    piece was written as red's place "by blob", with no paint within 3 mm of it. Through the command
    ring = palette_photo(s)
    draw_blob(ring, s["wells"]["red"]["xy"], (236, 236, 236), r=0.006)
    assert paint_share(ring, s["wells"]["red"]["xy"], PS.LOCATE_PAINT_R_M) == 0.0
    r, _ = cli("--sim", "locate", camera=lambda: ring.copy())
    now = json.load(open(SETUP_PATH))
    assert check_result(r, ok=True, stage="locate")["saved"] and r["found"] == 9 and r["not_located"] == ["red"] == r["missing"], r
    assert r["wells"]["red"]["by"] is None and now["wells"]["red"]["xy"] == s["wells"]["red"]["xy"] and now["wells"]["red"]["located"] is False
    assert "位置未確認" in r["say"] and "red" in r["say"] and any(w.startswith("not located: red.") for w in r["warnings"]), r
    for n, w in now["wells"].items():                      # every place that WAS written shows paint
        assert n == "red" or ("located" not in w and paint_share(ring, w["xy"], PS.LOCATE_PAINT_R_M) >= 0.9), (n, w)
    r, made = cli("--sim", "test", "--do", "load:red")
    assert check_result(r, ok=False, stage="wells") and not moved(made["bus"]) and r["brush"]["where"] == "not moved"
    # 2. a piece that two wells share. Violet's and magenta's paint in ONE piece, joined by paint (its middle IS
    #    paint: the gate alone does not see it), the set-up turned: the well the turn brought near that middle
    #    took it as its own blob, 21 mm from its paint, and the fit made with that pair lost red and brown
    use_setup(s)
    both = palette_photo(s)
    m, v = s["wells"]["magenta"]["xy"], s["wells"]["violet"]["xy"]
    draw_patch(both, min(m[0], v[0]) - 0.004, max(m[0], v[0]) + 0.004, v[1], m[1], colour=BLOB_BGR["violet"], z=TZ + 0.004)
    assert paint_share(both, ((m[0] + v[0]) / 2.0, (m[1] + v[1]) / 2.0), PS.LOCATE_PAINT_R_M) == 1.0
    for turn in (-6.0, -4.0, 4.0, 6.0):
        for shift in ((0.0, 0.0), (0.0, 0.006), (0.0, -0.006)):
            f = PS.relocate_wells(both, CAM, turned(turn, shift), TZ)
            assert PS.locate_usable(f) == (True, None) and f["not_located"] == [] and f["found"] == 8, (turn, shift, f)
            assert sorted(f["missing"]) == ["magenta", "violet"] and [f["wells"][n]["by"] for n in ("magenta", "violet")] == ["paint", "paint"]
            assert abs(f["turn_deg"] + turn) < 0.6 and max(off_mm(f, n) for n in s["wells"]) < 2.0, (turn, shift, f["turn_deg"], f["wells"])
    # 3. the real photos of the evening, the set-up turned about the palette's middle. The third review's cases
    #    through the command: 173916_look.jpg at 4 deg stored magenta 18.8 mm from its paint, on the divider
    #    (fitted turn -1.0), 165340_look.jpg at 6 deg brown on a stray piece 23.4 mm away (fitted turn -0.2)
    names = ("173916_look.jpg", "165340_look.jpg", "174247_test.jpg")
    photos = [os.path.join(tazuna_paths.ROOT, "runs", "20261113_paint", n) for n in names]
    if not all(os.path.exists(p) for p in photos):
        use_setup()
        raise Skip(f"the photos of 2026-11-13 ({', '.join(names)}) are not here; the synthetic parts passed")
    imgs = [cv2.imread(p) for p in photos]
    truth = [PS.relocate_wells(img, CAM, s, TZ) for img in imgs]           # (where the set-up as it is finds them)
    assert all(t["not_located"] == [] for t in truth)
    for img, tr, turn in ((imgs[0], truth[0], 4.0), (imgs[1], truth[1], 6.0)):
        was = use_setup(turned(turn))
        r, _ = cli("--sim", "locate", camera=lambda _i=img: _i.copy())
        now = json.load(open(SETUP_PATH))
        assert check_result(r, ok=True, stage="locate")["saved"] and abs(r["turn_deg"] + turn) < 1.0, (turn, r)
        for n, w in now["wells"].items():
            if w.get("located") is False:                  # (none today; one that is not located keeps its place)
                assert w["xy"] == was["wells"][n]["xy"], (turn, n)
                continue
            assert paint_share(img, w["xy"], PS.LOCATE_PAINT_R_M) >= 0.5 and off_mm(now, n, tr) <= 8.0, \
                (turn, n, w, paint_share(img, w["xy"], PS.LOCATE_PAINT_R_M), off_mm(now, n, tr))
        r, made = cli("--sim", "test", "--do", "load:violet", "--dry")
        assert check_result(r, ok=True, stage="planned")
    # ... and that review's scan, reduced (3 of 8 photos, 5 of 11 turns, 4 of 11 pushes): in every result that
    # may be stored, every well that is written lies on paint (judged here, with other thresholds than the
    # module's) and within 8 mm of where the unturned set-up finds it. Before: 43 of these 593 places did not
    written = 0
    for img, tr in zip(imgs, truth):
        for turn in (-6.0, -4.0, 3.0, 4.0, 6.0):
            for shift in ((0.0, 0.0), (0.0, 0.006), (0.0, -0.012), (-0.008, -0.008)):
                f = PS.relocate_wells(img, CAM, turned(turn, shift), TZ)
                if not PS.locate_usable(f)[0]:
                    continue
                for n, w in f["wells"].items():
                    if w["located"]:
                        written += 1
                        share = paint_share(img, w["xy"], PS.LOCATE_PAINT_R_M)
                        assert share >= 0.5 and off_mm(f, n, tr) <= 8.0, (turn, shift, n, w, share, off_mm(f, n, tr))
                    else:
                        assert w["xy"] == turned(turn, shift)["wells"][n]["xy"] and w["by"] is None, (turn, shift, n, w)
    assert written >= 500, written
    reset_state()
    use_setup()


def test_55_a_plane_adopted_for_another_brush_state_names_the_adoption_again():
    """T10 and T16 (third review), on the tool file as the rig has it. An adoption made EARLY, before the tip was
    stored and the jaw recorded and held, is a record of another brush state once those steps are done. Status
    and check then named only a new measurement (ladders on the real arm) and called the plane "measured for
    another brush state", although nothing was measured for that record and `fd paint contact --adopt` typed
    again was accepted. Now the record's own next is the adoption beside the measurement (PS.ADOPT_NEXT), and the
    reason says "adopted for". And the next of an ok `fd paint brush` named `fd paint preview PLAN`, which is
    refused at stage "contact" while the plane has no record: it names `fd paint status`."""
    early = tool("thin", tip_measured=None, **RIG_JAW)     # the rig's tool file: the tip typed, nothing said of the jaw
    use_tool(early)
    use_setup(make_setup(brush=None))
    reset_state()
    plan = write_plan("adopted.json", [("a", "red", bands(2, press=0.6))])
    r, _ = cli("--sim", "brush", "--kind", "round", "--fine", "0.6:2.7", "--broad", "1.2:5.0", "--load-mm", "140")
    assert check_result(r, ok=True, stage="brush")["saved"], r
    assert r["next"].startswith("`fd paint status` names what this brush still needs") and "preview" not in r["next"], r["next"]
    r2, _ = cli("--sim", "preview", plan, "--out", os.path.join(_TMP, "adopted.png"))      # (what it named before: refused here)
    assert check_result(r2, ok=False, stage="contact")
    r, _ = cli("--sim", "contact", "--adopt")              # the adoption made early: accepted, with two warnings
    assert check_result(r, ok=True, stage="contact")["saved"] and len(r["warnings"]) == 2, r.get("warnings")
    # ... then the tip is stored and the jaw recorded and held (what status names; test 49 types it): the same
    # brush, the same numbers, and the adopted record now names another brush state
    ready = use_tool(tool("thin"))
    assert (ready.name, ready.attached_at, ready.tip_offset) == (early.name, early.attached_at, early.tip_offset)
    r, _ = cli("--sim", "status")
    assert [(st["step"], st["done"]) for st in r["steps"]] == [("jaw", True), ("tip", True), ("brush", True), ("contact", False)]
    assert r["next"] == PS.ADOPT_NEXT and r["next"].startswith("fd paint contact --adopt   (") and "--scrap-laid --save" in r["next"]
    rec = r["contact_record"]
    assert rec["recorded"] and rec["ok"] is False and rec["adopted"] is True and len(rec["why"]) == 2, rec
    assert all("measured" not in w for w in rec["why"]) and any("when the plane was adopted" in w for w in rec["why"]), rec["why"]
    assert any(w.startswith("contact: the contact plane of the set-up was adopted for another brush state") for w in r["warnings"])
    assert not any("measured for another brush state" in w for w in r["warnings"]), r["warnings"]
    r, made = cli("--sim", "check")
    check_result(r, ok=False, stage="contact")
    assert r["next"] == PS.ADOPT_NEXT and "adopted for another brush state" in r["reason"] and "measured for" not in r["reason"], r
    assert "fd paint contact --adopt をもう一度" in r["say"] and "--scrap-laid --save" in r["say"] and not moved(made["bus"]), r["say"]
    for argv in (("run", plan, "--layer", "0"), ("run", plan, "--layer", "0", "--dry"),
                 ("test", "--scrap-laid", "--do", "line:0.25,0.0:0.25,0.03:0.6")):
        r, made = cli("--sim", *argv)                      # no stroke with it: refused as before, with the new next
        assert check_result(r, ok=False, stage="contact")["next"] == PS.ADOPT_NEXT and not moved(made["bus"]), (argv, r["next"])
        assert "adopted for another brush state" in r["reason"] and r["contact"]["recorded"] is True, r["reason"]
    r, _ = cli("--sim", "preview", plan, "--out", os.path.join(_TMP, "adopted.png"))
    assert check_result(r, ok=False, stage="contact")["next"] == PS.ADOPT_NEXT
    r, _ = cli("--sim", "test", "--ladder", "0.25,0.0", "--dry")       # (a ladder is not refused: its warning names both)
    assert check_result(r, ok=True) and any("adopted for another brush state" in w and "`fd paint contact --adopt` again" in w
                                            for w in r["warnings"]), r["warnings"]
    # typed as printed: the word again, for the brush as it is now, and nothing is missing any more
    cmd = PS.ADOPT_NEXT.split("   ")[0]
    assert cmd == "fd paint contact --adopt" and _accepted(cmd) is None
    r, made = cli("--sim", *shlex.split(cmd)[2:])
    assert check_result(r, ok=True, stage="contact")["saved"] and "warnings" not in r and not moved(made["bus"]), r
    assert r["replaces"]["tool"] == "brush_thin" and r["replaces"]["on_jaw"] is None and r["replaces"]["gripper_goal_raw"] is None
    r, _ = cli("--sim", "check")
    assert check_result(r, ok=True, stage="check") and r["contact_record"] == {"recorded": True, "ok": True, "why": []}
    r, _ = cli("--sim", "status")
    assert all(st["done"] for st in r["steps"]) and r["next"].startswith("`fd paint check`")
    r, _ = cli("--sim", "run", plan, "--layer", "0", "--dry")
    assert check_result(r, ok=True, stage="planned")
    # a record that came from LADDERS and names another brush state: the measurement alone, as before (the word is
    # refused for such a record: test 39)
    use_setup(make_setup(record=early))
    r, _ = cli("--sim", "status")
    assert r["next"] == PS.CONTACT_NEXT and "adopted" not in r["contact_record"] and "--adopt" not in r["next"], r["next"]
    assert any("measured for another brush state" in w for w in r["warnings"])
    r, _ = cli("--sim", "check")
    assert check_result(r, ok=False, stage="contact")["next"] == PS.CONTACT_NEXT and "--adopt" not in r["say"], r["say"]
    r, _ = cli("--sim", "contact", "--adopt")
    assert check_result(r, ok=False, stage="contact") and "MEASURED record" in r["reason"]
    use_tool("thin")
    use_setup()
    reset_state()


def test_56_the_options_of_the_paint_commands_are_typed_in_full():
    """T17 (third review). argparse completes an abbreviated option by itself: `fd paint contact --scrap` was
    --scrap-laid and `fd paint look --tip` was --tip-looked-at. Both are the operator's WORD for something the
    program cannot see (a scrap sheet lies over the picture; a photo of gripper and brush was looked at), and
    half a word is none. No sub-command takes an abbreviation any more: refused before an arm is built, and the
    refusal names the option in full."""
    thin = use_tool("thin")
    use_setup(make_setup(record=thin))
    reset_state()
    plan = write_plan("abbrev.json", [("a", "red", bands(2, press=0.6))])
    line = "line:0.25,0.0:0.25,0.03:0.6"                   # on the canvas: needs the word --scrap-laid

    def refused(argv, *full):
        r, made = cli("--sim", *argv)
        check_result(r, ok=False, stage="options")
        assert not made and "sim" not in r, (argv, r)                           # (no arm was built: nothing moved)
        assert "typed in full" in r["reason"] and all(f in r["reason"] and f in r["next"] and f in r["say"] for f in full), (argv, r)
        assert "省略できません" in r["say"] and "動かしていません" in r["say"], r["say"]
        return r

    # the two declarations
    for argv in (("contact", "--scrap"), ("contact", "--scrap-l", "--save"), ("contact", "--at", "0.25,0.0", "--scrap-lai"),
                 ("test", "--do", line, "--scrap"), ("test", "--ladder", "0.25,0.0", "--scrap-la")):
        refused(argv, "--scrap-laid")
    use_tool(tool("thin", tip_measured=None))              # a tip that was never measured: the word --tip-looked-at
    for argv in (("look", "--tip"), ("rinse", "--tip-looked"), ("locate", "--tip-l"), ("run", plan, "--layer", "0", "--tip"),
                 ("test", "--do", "rinse", "--tip-looke"), ("contact", "--scrap-laid", "--tip")):
        refused(argv, "--tip-looked-at")
    r, made = cli("--sim", "look", "--tip-looked-at")      # the whole word is taken as before: the arm moves on it
    assert check_result(r, ok=True, stage="look") and moved(made["bus"]) and any("--tip-looked-at" in w for w in r["warnings"])
    use_tool(thin)
    r, made = cli("--sim", "test", "--do", line)           # (without the word: the refusal the abbreviation got round)
    assert check_result(r, ok=False, stage="contact") and not moved(made["bus"])
    r, made = cli("--sim", "test", "--do", line, "--scrap-laid", "--no-photo")
    assert check_result(r, ok=True, stage="test") and moved(made["bus"]) and r["did"] == [line]
    # every other option too; a beginning that fits several names them all
    refused(("run", plan, "--layer", "0", "--low", "1.5"), "--lower-mm")
    refused(("run", plan, "--lay", "0"), "--layer")
    refused(("rest", "--keep"), "--keep-torque")
    refused(("rest", "--clear", "0.05"), "--clearance")
    refused(("brush", "--carr", "none"), "--carries")
    refused(("contact", "--ad"), "--adopt")
    r = refused(("test", "--l", "0.01", "--do", "rinse"), "--ladder", "--len", "--ladder-len", "--lower-mm")
    assert "one of" in r["next"]
    assert paintcli._abbreviated(["contact", "--scrap-laid", "--save"]) == [] and paintcli._abbreviated(["frobnicate", "--scrap"]) == []
    assert paintcli._abbreviated(["contact", "--scrap=1"]) == [("--scrap", ["--scrap-laid"])]
    # an option that is none at all is refused as before, without that hint
    r, made = cli("--sim", "look", "--frobnicate")
    assert check_result(r, ok=False, stage="options") and not made and "typed in full" not in r["reason"]
    # the parsers themselves; and through fd
    parser = paintcli.build_parser()
    sub = next(act for act in parser._actions if getattr(act, "choices", None) and "status" in act.choices)
    assert parser.allow_abbrev is False and all(p.allow_abbrev is False for p in sub.choices.values())
    r = fd_cmd("--sim", "paint", "contact", "--scrap")
    assert r["_rc"] == 1 and r["ok"] is False and r["stage"] == "options" and "--scrap-laid" in r["next"], r
    r = fd_cmd("--sim", "paint", "contact", "--scrap-laid", "--dry")
    assert r["_rc"] == 0 and r["ok"] is True and r["stage"] == "planned", r
    reset_state()
    use_setup()


NEIGHBOUR_PHOTOS = ("173916_look.jpg", "172135_look.jpg", "174247_test.jpg", "175520_run.jpg")      # (where the last verifier met it)


def displaced(s, turn_deg, shift, about=None) -> dict:
    """Set-up s with its wells turned by turn_deg about the well `about` (None: about the palette's middle) and pushed
    by `shift` (m): what the set-up looks like, against a photo, after the palette was moved the other way."""
    t = copy.deepcopy(s)
    pts = np.array([w["xy"] for w in s["wells"].values()], dtype=float)
    c = pts.mean(axis=0) if about is None else np.array(s["wells"][about]["xy"], dtype=float)
    a = math.radians(turn_deg)
    rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
    for name, q in zip(s["wells"], (pts - c) @ rot.T + c + np.array(shift, dtype=float)):
        t["wells"][name]["xy"] = [round(float(q[0]), 4), round(float(q[1]), 4)]
    return t


def test_57_locate_stores_no_well_on_its_neighbours_paint():
    """U3 of the last verification (2026-11-14). With the set-up turned 8 deg about the black well and pushed 17 mm,
    `fd paint locate` stored magenta on VIOLET's paint (ok, saved; 4 of 2,640 turns and pushes of ten real photos):
    the search area had cut the piece the two wells share, magenta took the middle of what was left, and that ONE pair
    turned the fit by 6 deg (fitted -2.3 for a true -8), so red, blue and brown were written 12 to 20 mm beside the
    middle of their paint as well. Two rules close it. (1) Every well that took a blob must lie where the OTHER wells'
    blobs put it (LOCATE_CONFIRM_M); a result with a well that does not is not stored at all: stage "locate", next:
    put the palette back, or correct the wells by hand. (2) A well remembers the colour it showed where it was last
    written ("hsv" in the set-up) and is never written onto paint of another colour, whatever the fit says."""
    use_tool("thin")
    s = use_setup(make_setup(record=tool("thin")))
    reset_state()
    names = list(s["wells"])

    def off_mm(found, n, truth):
        want = truth["wells"][n]["xy"]
        return math.hypot(found["wells"][n]["xy"][0] - want[0], found["wells"][n]["xy"][1] - want[1]) * 1000.0

    # 1. two colours: what other_paint tells apart, and what it leaves alone (a statement only where both have a hue)
    red, magenta, violet, yellow, blue, cerulean, black = ([179, 212, 141], [173, 214, 123], [133, 136, 58], [18, 179, 165],
                                                           [113, 180, 68], [104, 245, 124], [120, 57, 40])     # (173916_look.jpg)
    assert PS.other_paint(magenta, violet) and PS.other_paint(violet, magenta) and PS.other_paint(red, yellow)
    assert PS.other_paint(black, red) and PS.other_paint(yellow, black) and PS.other_paint(blue, violet)
    assert not PS.other_paint(red, [1, 229, 157]) and not PS.other_paint([1, 229, 157], red)        # (the same red, 90 minutes later)
    assert not PS.other_paint(red, magenta) and not PS.other_paint(blue, cerulean)                  # (too near to say)
    assert not PS.other_paint(black, [111, 92, 39]) and not PS.other_paint(blue, [117, 121, 40])    # (dark: no hue to go by)
    # 2. the colour memory, on a palette that is drawn. A locate that stores writes what each well shows where it is
    #    written, and the set-up with those colours is a set-up
    pushed = palette_photo(s, (0.012, 0.0))
    r, _ = cli("--sim", "locate", camera=lambda: pushed.copy())
    now = json.load(open(SETUP_PATH))
    assert check_result(r, ok=True, stage="locate")["saved"] and r["found"] == 10 and r["other_colour"] == [] and r["apart_mm"] == {}, r
    assert all(len(w["hsv"]) == 3 for w in now["wells"].values()) and PS.check_setup(now), now["wells"]
    hue = {n: w["hsv"][0] for n, w in now["wells"].items()}
    assert min(hue["red"], 180 - hue["red"]) <= 4 and 20 <= hue["yellow"] <= 28 and 130 <= hue["violet"] <= 140 and 160 <= hue["magenta"] <= 170, hue
    bad = copy.deepcopy(now)
    bad["wells"]["red"]["hsv"] = [0, 300]
    assert "hsv" in str(raises_value(PS.check_setup, bad))
    # ... with the colours on record the palette is found again as before, turned and pushed (no well is held back)
    coloured = copy.deepcopy(now)
    for turn in (-6.0, 6.0):
        for shift in ((0.0, 0.0), (0.008, -0.008)):
            f = PS.relocate_wells(palette_photo(coloured, shift, turn_deg=turn), CAM, coloured, TZ)
            assert PS.locate_usable(f) == (True, None) and f["not_located"] == [] and f["other_colour"] == [] and f["found"] == 10, (turn, shift, f)
            assert abs(f["turn_deg"] - turn) < 0.6, (turn, shift, f["turn_deg"])
    # ... a well that shows ANOTHER paint where it would be written is not written: violet's well refilled with yellow.
    #    It keeps its place and its colour, nothing is loaded from it, and the result says why
    refilled = palette_photo(coloured)
    draw_blob(refilled, coloured["wells"]["violet"]["xy"], BLOB_BGR["yellow"])
    r, _ = cli("--sim", "locate", camera=lambda: refilled.copy())
    now = json.load(open(SETUP_PATH))
    assert check_result(r, ok=True, stage="locate")["saved"] and r["other_colour"] == ["violet"] == r["not_located"], r
    assert r["wells"]["violet"]["other_colour"]["on_record"] == coloured["wells"]["violet"]["hsv"], r["wells"]["violet"]
    assert now["wells"]["violet"]["located"] is False and now["wells"]["violet"]["xy"] == coloured["wells"]["violet"]["xy"]
    assert now["wells"]["violet"]["hsv"] == coloured["wells"]["violet"]["hsv"], now["wells"]["violet"]
    assert any(w.startswith("another colour: violet") for w in r["warnings"]) and "記録と違う色" in r["say"], (r["warnings"], r["say"])
    r, made = cli("--sim", "test", "--do", "load:violet")
    assert check_result(r, ok=False, stage="wells") and not moved(made["bus"])
    # ... once the operator has taken the colour out (the well holds another paint now), it is found and learnt anew
    now["wells"]["violet"].pop("hsv")
    use_setup(now)
    r, _ = cli("--sim", "locate", camera=lambda: refilled.copy())
    now = json.load(open(SETUP_PATH))
    assert check_result(r, ok=True, stage="locate")["saved"] and r["not_located"] == [] and "located" not in now["wells"]["violet"], r
    assert 20 <= now["wells"]["violet"]["hsv"][0] <= 28, now["wells"]["violet"]
    # ... and the neighbour's paint. The set-up has magenta beside VIOLET's blob (its own paint is used up), violet
    #    elsewhere: every other well says "nothing moved", so the first rule has nothing to object to. Without a colour
    #    on record magenta is written onto violet's paint; with one, it is not written at all
    use_setup(coloured)
    v = coloured["wells"]["violet"]["xy"]
    beside = copy.deepcopy(coloured)
    beside["wells"]["magenta"]["xy"] = [round(v[0] + 0.004, 4), round(v[1] + 0.004, 4)]
    beside["wells"]["violet"]["xy"] = [round(v[0] + 0.005, 4), round(v[1] - 0.045, 4)]
    no_magenta = palette_photo(coloured, skip=("magenta",))
    blind = copy.deepcopy(beside)
    for w in blind["wells"].values():
        w.pop("hsv")
    f = PS.relocate_wells(no_magenta, CAM, blind, TZ)
    assert PS.locate_usable(f) == (True, None) and f["wells"]["magenta"]["located"] and f["apart_mm"] == {}, f
    assert math.hypot(f["wells"]["magenta"]["xy"][0] - v[0], f["wells"]["magenta"]["xy"][1] - v[1]) < 0.008, f["wells"]["magenta"]
    f = PS.relocate_wells(no_magenta, CAM, beside, TZ)
    assert not f["wells"]["magenta"]["located"] and f["wells"]["magenta"]["xy"] == beside["wells"]["magenta"]["xy"], f["wells"]["magenta"]
    assert "magenta" in f["not_located"] and all(off_mm(f, n, coloured) < 2.0 for n in names if f["wells"][n]["located"]), f
    # 3. the first rule on a palette that is drawn as the real one looked to the function: five wells in one corner,
    #    magenta 23 cm away with its own paint out of the search area, and a piece of VIOLET's paint where the cut
    #    piece had its middle. The set-up turned 8 deg about the black well and pushed 17 mm, as the verifier had it:
    #    magenta takes that piece, the fit follows it (-2.3 deg for a true -8, as on the photo), and all six "agree"
    use_setup(s)
    mini = copy.deepcopy(s)
    mini["wells"] = {n: s["wells"][n] for n in ("yellow", "crimson", "black", "cerulean", "green", "magenta")}
    decoy = palette_photo(mini, skip=("magenta",))
    draw_blob(decoy, (0.0089, 0.2506), BLOB_BGR["violet"], r=0.008)
    case = displaced(mini, 8.0, (0.012, -0.012), "black")
    with patched(PS, "LOCATE_CONFIRM_M", 1.0):          # (what it was: stored, magenta 27 mm from its place)
        f = PS.relocate_wells(decoy, CAM, case, TZ)
    assert PS.locate_usable(f) == (True, None) and f["found"] == 6 and abs(f["turn_deg"] + 2.3) < 0.5, (f["turn_deg"], f["found"])
    assert f["wells"]["magenta"]["by"] == "blob" and 25.0 < off_mm(f, "magenta", mini) < 30.0, f["wells"]["magenta"]
    f = PS.relocate_wells(decoy, CAM, case, TZ)         # ... the other five put magenta 27 mm from that piece: not stored
    assert list(f["apart_mm"]) == ["magenta"] and 25.0 < f["apart_mm"]["magenta"] < 30.0, f["apart_mm"]
    ok, why = PS.locate_usable(f)
    assert ok is False and "do not agree on ONE move" in why and "magenta" in why and "12 mm" in why, why
    # (the same palette with magenta's own paint in reach, turned and pushed as far as works today: all six, stored)
    f = PS.relocate_wells(palette_photo(mini), CAM, displaced(mini, 4.0, (0.008, -0.008), "black"), TZ)
    assert PS.locate_usable(f) == (True, None) and f["found"] == 6 and f["apart_mm"] == {} and abs(f["turn_deg"] + 4.0) < 0.6, f
    assert all(off_mm(f, n, mini) < 3.0 for n in mini["wells"]), f["wells"]
    # (witnesses that cannot say how the palette was turned: with violet's whole blob as the decoy the fit turns the
    # other way, two of the five lose their blobs, and three wells in one corner are all that is left to confirm
    # magenta: they can only say where the palette was pushed to, and magenta's blob must lie THERE to 8 mm)
    decoy = palette_photo(mini, skip=("magenta",))
    draw_blob(decoy, s["wells"]["violet"]["xy"], BLOB_BGR["violet"])
    f = PS.relocate_wells(decoy, CAM, case, TZ)
    assert f["found"] == 4 and list(f["apart_mm"]) == ["magenta"] and PS.LOCATE_AGREE_M * 1000.0 < f["apart_mm"]["magenta"] < 12.0, f
    assert PS.locate_usable(f)[0] is False
    # (a set-up of three wells has nobody to confirm a well with: the rule keeps out of it)
    few = copy.deepcopy(s)
    few["wells"] = {n: s["wells"][n] for n in ("red", "yellow", "green")}
    f = PS.relocate_wells(palette_photo(few, (0.010, 0.0)), CAM, few, TZ)
    assert f["apart_mm"] == {} and PS.locate_usable(f) == (True, None), f
    # 4. the verifier's case, through the command, on the photo it happened on (the rig's set-up has no colours: the
    #    first rule alone)
    photos = [os.path.join(tazuna_paths.ROOT, "runs", "20261113_paint", n) for n in NEIGHBOUR_PHOTOS]
    if not all(os.path.exists(p) for p in photos):
        reset_state()
        use_setup()
        raise Skip(f"the photos of 2026-11-13 ({', '.join(NEIGHBOUR_PHOTOS)}) are not here; the parts on drawn palettes passed")
    imgs = [cv2.imread(p) for p in photos]
    truth = [PS.relocate_wells(img, CAM, s, TZ) for img in imgs]           # (where the set-up as it is finds them)
    case = displaced(s, 8.0, (0.012, -0.012), "black")
    use_setup(case)
    before = open(SETUP_PATH, "rb").read()
    r, made = cli("--sim", "locate", camera=lambda: imgs[0].copy())
    check_result(r, ok=False, stage="locate")
    assert r["saved"] is False and open(SETUP_PATH, "rb").read() == before and "set-up keeps the places it had" in r["reason"], r
    assert "do not agree on ONE move" in r["reason"] and list(r["apart_mm"]) == ["magenta"] and r["apart_mm"]["magenta"] > 20.0, r["reason"]
    assert "put the palette back" in r["next"] and "by hand" in r["next"] and "`fd paint locate` again" in r["next"], r["next"]
    assert "元の位置に戻して" in r["say"] and "保存しませんでした" in r["say"] and os.path.exists(r["picture"]), r["say"]
    assert r["brush"]["where"] == "photo pose"
    # ... what the rule keeps out: with it switched off the same photo is stored, magenta on violet's paint
    for img, tr in zip(imgs, truth):
        with patched(PS, "LOCATE_CONFIRM_M", 1.0):
            f = PS.relocate_wells(img, CAM, case, TZ)
        assert PS.locate_usable(f) == (True, None) and f["wells"]["magenta"]["located"], f
        assert off_mm(f, "magenta", tr) > 20.0 and off_mm(f, "magenta", {"wells": {"magenta": tr["wells"]["violet"]}}) < 20.0, f["wells"]["magenta"]
        f = PS.relocate_wells(img, CAM, case, TZ)
        assert PS.locate_usable(f)[0] is False and f["apart_mm"]["magenta"] > 20.0, f["apart_mm"]
        # ... and with the colours of the wells on record, the colour alone keeps magenta off it (the first rule off)
        known = copy.deepcopy(case)
        for n in names:
            if "hsv" in tr["wells"][n]:
                known["wells"][n]["hsv"] = tr["wells"][n]["hsv"]
        with patched(PS, "LOCATE_CONFIRM_M", 1.0):
            f = PS.relocate_wells(img, CAM, known, TZ)
        assert all(off_mm(f, n, tr) <= 8.0 for n in names if f["wells"][n]["located"]) or not PS.locate_usable(f)[0], f["wells"]
        assert not f["wells"]["magenta"]["located"], f["wells"]["magenta"]
    # ... while what works today still stores, and stores the same: the set-up turned up to 6 deg about that corner
    #    well and pushed 12 mm each way (the wells within 8 mm of where the unturned set-up finds them)
    stored = 0
    for turn in (-6.0, -4.0, 4.0, 6.0):
        for shift in ((0.0, 0.0), (0.012, -0.012), (-0.012, 0.012)):
            f = PS.relocate_wells(imgs[0], CAM, displaced(s, turn, shift, "black"), TZ)
            assert PS.locate_usable(f) == (True, None) and f["apart_mm"] == {} and f["other_colour"] == [], (turn, shift, PS.locate_usable(f))
            stored += sum(1 for n in names if f["wells"][n]["located"])
            assert all(off_mm(f, n, truth[0]) <= 8.0 for n in names if f["wells"][n]["located"]), (turn, shift, f["wells"])
    assert stored >= 100, stored
    reset_state()
    use_setup()


# ============================================================================= 10. the commands a result names
# T12 (third review). What a result prints in backticks, or at the beginning of a `next`, is ONE command that works
# AS IT IS TYPED: no word of it stands for a value the operator has to put in (PLAN, DIR, x,y, P: such a command
# is refused by a check when it parses, `fd paint rest --pan P` does not even parse), and it is no bare name of a
# command that needs more (`fd paint run`). Where the program knows the value (a plan's path, a pan angle) the
# whole command is printed; where it does not, the step stands in words. Until then this test filled such
# words in before it parsed, and let a bare name pass.
_PLACEHOLDER = re.compile(r"[A-Z][A-Z0-9_]*(\.[A-Za-z0-9]+)?"                   # PLAN, DIR, N, P, NAME, PLAN.json, BEFORE.jpg
                          r"|.*\bx\d?,y\d?.*"                                   # x,y   x,y,z   x0,y0:x1,y1
                          r"|.*[A-Z]{2,}[:=][A-Z]{2,}.*|.*=[A-Z]{2,}.*|[A-Z]{2,}=.*"      # PRESS:WIDTH, JOINT=LO:HI, JOINT=none
                          r"|.*\.\.\..*|.*[|<>].*")                             # ..., none|PAINT, <value>


def _placeholders(cmd: str) -> list:
    """The words of a printed command that stand for a value instead of being one ([]: none)."""
    try:
        words = shlex.split(cmd)
    except ValueError:
        return [cmd]
    return [w for w in words[1:] if _PLACEHOLDER.fullmatch(w)]


def _commands(res) -> set:
    """Every `fd ...` command a result gives the operator: in backticks anywhere in its next / reason / warnings /
    steps, and a `next` that itself begins with the command (up to the three spaces before its remark)."""
    nexts = [res.get("next"), res.get("reread")] + [st.get("next") for st in res.get("steps") or [] if isinstance(st, dict)]
    out = set()
    for text in nexts + [res.get("reason")] + list(res.get("warnings") or []):
        if isinstance(text, str):
            out.update(re.findall(r"`(fd [^`]+)`", text))
    for text in nexts:
        if isinstance(text, str) and text.startswith("fd "):
            out.add(text.split("   ")[0].strip())
    return out


def _accepted(cmd: str):
    """Is `fd ...` a command the parsers take, exactly AS IT IS PRINTED? Why not (None: it is). Nothing is run and
    nothing is filled in: fd's own parser reads the words up to the sub-command, the sub-command's parser (paint,
    tool jaw, calib-air) the rest."""
    import fd
    try:
        words = shlex.split(cmd)[1:]
    except ValueError as e:
        return f"cannot be split into words: {e}"
    argv, own, rest = fd._split_own_words(words)
    sink = io.StringIO()

    def verdict(code):
        text = sink.getvalue().strip().splitlines()[-1] if sink.getvalue().strip() else f"exit code {code}"
        return None if code == 0 else text

    try:
        with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
            fd.build_parser().parse_args(argv)
    except SystemExit as e:
        return verdict(e.code)
    rest = rest[1:] if rest[:1] == ["--"] else rest
    try:
        with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
            if own == "paint_args":
                paintcli.build_parser().parse_args(rest)
            elif own == "jaw_args":
                import tooljaw
                if rest[:1] not in (["help"], ["--help"]):
                    tooljaw.build_parser().parse_args(rest)
            elif own == "calib_air_args":
                import aircal_cli
                if rest[:1] not in (["--help"], ["-h"]):
                    aircal_cli.build_parser().parse_args(rest)
    except SystemExit as e:
        return verdict(e.code)
    except Exception as e:      # (each module's own "cannot read these words")
        return f"{type(e).__name__}: {e}"
    return None


def _printed_texts(path: str) -> list:
    """Every text a module can PRINT: its string constants, an f-string as one text with "{}" where a value is
    filled in (adjacent literals are one constant already). Docstrings and comments are not printed: left out."""
    import ast
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    skip, out = set(), []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.body:
            first = node.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                skip.add(id(first.value))
        if isinstance(node, ast.JoinedStr):
            skip.update(id(v) for v in node.values)
            out.append("".join(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "{}" for v in node.values))
    out += [node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in skip]
    return out


def test_zy_every_command_a_next_names_is_one_the_parsers_accept():
    """`next` is the next command in a spelling that works. Every `fd ...` command that a result of this suite
    named (some hundred results when the whole suite ran before; the ones below when this test runs alone), and
    every one the three paint modules can print (their string constants), is handed to fd's parser and to the
    parser of the sub-command it belongs to EXACTLY AS IT IS PRINTED, and none of its words is a placeholder
    (T12, third review). Nothing is run."""
    plan = write_plan("words.json", [("a", "red", bands(2, press=0.6))])
    reset_state()
    for t_, s_ in ((tool("thin", tip_measured=None, **RIG_JAW), make_setup(brush=None)),          # the rig's files as they stand
                   (tool("thin", tip_measured=None, gripper_goal_raw=None), make_setup(brush=None)),
                   (tool("thin", tip_measured=None), make_setup()), (tool("thin"), make_setup(brush=None)), (tool("thin"), make_setup()),
                   (tool("thin"), make_setup(record=tool("thin", tip_offset=[0.016, -0.001, 0.058]))),
                   (tool("thin"), make_setup(record=tool("thin")))):
        use_tool(t_)
        use_setup(s_)
        for argv in (("status",), ("check",), ("look",), ("rinse",), ("run", plan, "--layer", "0"), ("run", plan, "--layer", "0", "--dry"),
                     ("preview", plan, "--out", os.path.join(_TMP, "words.png")), ("contact",), ("contact", "--dry"),
                     ("contact", "--scrap-laid", "--save"), ("contact", "--adopt", "--dry"), ("contact", "--zs", "9,5,1,-3,-7"),
                     ("test", "--do", "line:0.25,0.0:0.25,0.03:1.2", "--lower-mm", "2"), ("test", "--do", "ladder:0.0716,0.2651"),
                     ("rest", "--clearance", "9"), ("rest", "--dry"), ("rest", "--pan", "20"), ("locate",), ("brush",),
                     ("brush", "--carries", "pink"), ("frobnicate",)):
            cli("--sim", *argv)
    _newer_tool(tool("thin"))
    use_setup(make_setup())
    _newer_tool(tool("thin"))
    for argv in (("run", plan, "--layer", "0"), ("check",), ("status",), ("look",)):
        cli("--sim", *argv)
    cli("look")                                            # (the hardware lock)
    with open(PS.state_path(True), "w") as f:
        f.write("{ no json")
    cli("--sim", "status")
    reset_state()
    with open(TOOL_PATH, "w") as f:
        f.write("{}")
    cli("--sim", "status")
    use_tool("thin")
    use_setup(make_setup(record=tool("thin")))
    with _break_stroke(2, KeyboardInterrupt()):
        cli("--sim", "run", plan, "--layer", "0")
    r, _ = cli("--sim", "run", plan, "--layer", "0")
    assert r["done"]
    TL.clear()                                             # (no tool at all: what registers one)
    for argv in (("status",), ("look",), ("preview", plan, "--out", os.path.join(_TMP, "words.png"))):
        cli("--sim", *argv)
    use_tool("thin")
    cmds = sorted({c for res in _RESULTS for c in _commands(res)})
    assert len(cmds) >= 25 and len(_RESULTS) >= 150, (len(cmds), len(_RESULTS))
    # T12: ONE complete command per pair of backticks (and at the beginning of a `next`), typed as printed: the
    # parsers take it with nothing filled in, and no word of it is a placeholder. (R9 before it: no alternatives
    # joined by a bar, `fd paint brush --carries none|PAINT` recorded an empty brush, and no "..." to guess.)
    bad = [(c, why) for c in cmds for why in [_accepted(c)] if why]
    assert not bad, bad
    held = [(c, _placeholders(c)) for c in cmds if _placeholders(c)]
    assert not held, held
    # the texts the three modules can print (their string constants: nexts, reasons, help; not comments): every
    # command in backticks, and every text that begins with its command
    spelled = set()
    for mod in (BW, PS, paintcli):
        for text in _printed_texts(mod.__file__):
            spelled.update(c for c in re.findall(r"`(fd [^`]+)`", text) if "{" not in c)
            if text.startswith("fd ") and "   (" in text and "{" not in text.split("   ")[0]:      # (a command, then its remark)
                spelled.add(text.split("   ")[0].strip())
    for text in (PS.CONTACT_NEXT, PS.ADOPT_NEXT, PS.TIP_NEXT, PS.TIP_NEXT_HELD, PS.TIP_NEXT_FIRST.format(jaw="`fd tool jaw close`"),
                 paintcli.JAW_MOUNT_NEXT, paintcli.JAW_CLOSE_NEXT, paintcli.TIP_NEXT, paintcli.JAW_ELSEWHERE_NEXT, paintcli.SIM_NEXT,
                 paintcli.WELLS_LOST, paintcli.TOOL_FILE_NEXT, paintcli.BRUSH_NEXT, paintcli.SCRAP_NEXT.format(more=" --save"),
                 paintcli.INTERRUPTED_NEXT, paintcli.STOPPED_NEXT, paintcli.BUS_NEXT, paintcli.PLAN_NEXT, paintcli.READ_NEXT, BW.JAW_NEXT):
        spelled.update(re.findall(r"`(fd [^`]+)`", text))
        if text.startswith("fd "):                         # (a `next` that begins with its command)
            spelled.add(text.split("   ")[0])
    assert PS.CONTACT_NEXT.startswith("lay a scrap sheet") and paintcli.SCRAP_NEXT.startswith("lay a scrap sheet")     # the sheet first
    assert len(spelled) >= 30, len(spelled)
    bad = [(c, why) for c in sorted(spelled) for why in [_accepted(c)] if why]
    assert not bad, bad
    held = [(c, _placeholders(c)) for c in sorted(spelled) if _placeholders(c)]
    assert not held, held
    # what needs a value the program does not know stands in WORDS: not in backticks, not at the beginning of a next
    for text in (PS.TIP_NEXT_FIRST, paintcli.PLAN_NEXT, paintcli.READ_NEXT, paintcli.BUS_NEXT):
        assert not text.startswith("fd ") and not any(_placeholders(c) for c in re.findall(r"`(fd [^`]+)`", text)), text
    assert PS.FROM_PHOTOS in PS.TIP_NEXT and PS.FROM_PHOTOS in PS.TIP_NEXT_FIRST and "`" not in PS.FROM_PHOTOS + paintcli.PLAN_NEXT
    # the ones the operator meets first are among them, as they are typed
    for want in ("fd tool jaw mount moving_outer", "fd tool jaw close", "fd calib-air", "fd calib-air --save", "fd tool jaw status",
                 "fd tool attach --help", "fd --sim paint status", "fd paint locate",
                 "fd paint brush --kind round --fine 0.6:2.7 --broad 1.2:5.0 --load-mm 140", "fd paint contact --adopt",
                 "fd paint contact --scrap-laid --save", "fd paint contact --scrap-laid", "fd paint rest", "fd paint status",
                 "fd paint check", "fd tool detach", "fd hw status", f"fd paint run {plan} --layer 0"):
        assert want in cmds or want in spelled, (want, [c for c in cmds if c.startswith(want[:14])])
    # and the check itself can fail: spellings that do not work, bare names and placeholders are told apart
    assert _accepted("fd paint contact --scrap-laid --save") is None and _accepted("fd --sim paint status") is None
    assert _accepted("fd paint contact --save --scraplaid") and _accepted("fd paint contakt") and _accepted("fd paint run PLAN")
    assert _accepted("fd paint brush --carries") and _accepted("fd tool jaw mount outside") and _accepted("fd paint status --sim")
    assert _accepted("fd paint run") and _accepted("fd paint") and _accepted("fd paint preview")           # (bare names)
    assert _accepted("fd paint rest --pan P") and _accepted("fd paint brush --kind round ...") and _accepted("fd paint contact --scrap")
    assert _accepted("fd paint preview PLAN") is None and _placeholders("fd paint preview PLAN") == ["PLAN"]       # (it parses: the words tell)
    for cmd, words in (("fd calib-air --from DIR --save", ["DIR"]), ("fd paint run PLAN.json --layer N --dry", ["PLAN.json", "N"]),
                       ("fd paint contact --read BEFORE.jpg AFTER.jpg", ["BEFORE.jpg", "AFTER.jpg"]), ("fd paint test --ladder x,y", ["x,y"]),
                       ("fd tool attach --name N --tip x,y,z", ["N", "x,y,z"]), ("fd tool set --joint-limit JOINT=LO:HI", ["JOINT=LO:HI"]),
                       ("fd tool set --joint-limit wrist_flex=LO:HI", ["wrist_flex=LO:HI"]), ("fd paint brush --carries none|PAINT", ["none|PAINT"]),
                       ("fd paint brush --kind round ...", ["..."]), ("fd paint brush --fine PRESS:WIDTH", ["PRESS:WIDTH"]),
                       ("fd tool jaw open --to DEG", ["DEG"]), ("fd paint rest --pan P", ["P"])):
        assert _placeholders(cmd) == words, (cmd, _placeholders(cmd))
    for cmd in ("fd paint run /tmp/tz_paint_Ab3_x/next.json --layer 0 --lower-mm 1.5 --stop 4", "fd paint rest --pan -89.2 --clearance 0.06",
                "fd tool jaw mount moving_outer", "fd paint brush --kind round --fine 0.6:2.7 --broad 1.2:5.0 --load-mm 140",
                "fd paint test --do load:red --do line:0.325,-0.06:0.325,-0.03:0.6", "fd tool set --joint-limit wrist_flex=20:97",
                "fd paint brush --carries none", f"fd calib-air --from {THIN4} --save", "fd tool set --tip=0.0008,0.0212,0.0777"):
        assert _placeholders(cmd) == [], (cmd, _placeholders(cmd))
    reset_state()
    use_setup()


def test_zz_no_hardware_and_nothing_of_the_rig_was_written():
    """Runs last: no serial port, no camera, no v4l2-ctl; the rig's tool, set-up and state files are byte for
    byte what they were, no such file was added to the project folder, and no photo or picture went into runs/."""
    assert _TRIPPED == [], _TRIPPED
    try:
        open("/dev/ttyACM0")
    except RuntimeError:
        _TRIPPED.clear()                                   # the guard works (and nothing was opened)
    except OSError:
        raise AssertionError("the audit hook is not armed")
    else:
        raise AssertionError("the audit hook is not armed")
    now = _tree_state()
    assert now == _TREE_BEFORE, {k: (now.get(k), _TREE_BEFORE.get(k)) for k in set(now) | set(_TREE_BEFORE)
                                 if now.get(k) != _TREE_BEFORE.get(k)}
    # R22: the comparison sees a leak wherever it goes, also into the photo folder of a NEW day (a suite that
    # crosses midnight), shown on a stand-in tree: a new folder in runs/, a file added to any *_paint folder
    stand_in = os.path.join(_TMP, "tree")
    for d in ("config", os.path.join("runs", "20261113_paint"), os.path.join("runs", "20261110_230151")):
        os.makedirs(os.path.join(stand_in, d), exist_ok=True)
    open(os.path.join(stand_in, "runs", "20261113_paint", "173916_look.jpg"), "w").close()
    before = _tree_state(stand_in)
    assert before["dir:runs"] == ["20261110_230151", "20261113_paint"] and before["dir:runs/20261113_paint"] == ["173916_look.jpg"]
    assert _tree_state(stand_in) == before
    tomorrow = os.path.join(stand_in, "runs", time.strftime("%Y%m%d", time.localtime(time.time() + 86400.0)) + "_paint")
    os.makedirs(tomorrow)                                  # (an empty folder of the new day is a difference already)
    assert _tree_state(stand_in) != before
    open(os.path.join(tomorrow, "000001_look.jpg"), "w").close()
    after = _tree_state(stand_in)
    assert after != before and after["dir:runs/" + os.path.basename(tomorrow)] == ["000001_look.jpg"]
    shutil.rmtree(tomorrow)
    open(os.path.join(stand_in, "runs", "20261113_paint", "000002_preview.png"), "w").close()
    assert _tree_state(stand_in) != before
    assert "dir:runs" in _TREE_BEFORE and any(k.startswith("dir:runs/") and k.endswith("_paint") for k in _TREE_BEFORE) \
        or not os.path.isdir(os.path.join(tazuna_paths.ROOT, "runs", "20261113_paint"))
    # this suite pins $FD_TOOL with the line tests/test_integration_pen.py looks for in every suite that builds an
    # Arm (it read `os.environ.update(FD_TOOL=...)` as "not pinned" and failed), before any harness module is loaded
    with open(os.path.abspath(__file__), encoding="utf-8") as fh:
        src = fh.read()
    pin = 'os.environ["FD_TOOL"] = '
    assert src.count("\n" + pin + "TOOL_PATH") == 1 and src.index("\n" + pin) < src.index("\nimport brushwork")
    assert os.environ["FD_TOOL"] == TOOL_PATH and os.environ["FD_PAINT_SETUP"] == SETUP_PATH and os.environ["FD_PAINT_STATE"] == STATE_PATH
    assert os.environ["TAZUNA_HW_LOCK"] == LOCK_PATH and hwlock.info() is not None
    assert PS.SETUP_FILE == os.path.join(tazuna_paths.ROOT, "paint_setup.json")
    assert PS.STATE_FILE == os.path.join(tazuna_paths.ROOT, "paint_state.json")


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
