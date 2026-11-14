"""Tests for a tool taped to the gripper (tooling.py and the guards in robot.py / skills.py / fd.py).
Run: python tests/test_tooling.py [name-part ...]      (no hardware: simulated servo bus, saved photos)

What they show (SPEC section 7, TOOL): with a tool file present, in simulation, grip / grasp / pick / take /
do / park / probe / bench refuse without a single bus write; `ready` ends with the tool tip at the stand-by
point and never below 2 cm on the way, also from the folded rest pose and from the paper; `fd tool rest`
reaches a pose with the tool tip 4-7 cm above the table for tools 2-12 cm long, along and across the
fingers, or refuses cleanly; torque off is refused in a raised pose; after `detach` the old behaviour is
back; without a tool file nothing changes.

Joint limits of a tool (2026-11-13: `fd tool attach|set --joint-limit wrist_flex=20:97` for a brush whose
handle lies over the forearm): they are stored, shown and removed by the commands and survive a load / save
round; limits that cannot be applied are an error, never "no limits"; Arm.q_limits is the intersection of
servo, soft and tool limits; IK, tip paths and pose checks refuse beyond them and name the joint; the rest
pose and the stand-by pose lie inside them or are refused with the way out; free moves stay inside them; an
arm that was moved by hand to a pose outside a limit can be lifted and comes back inside, never further out;
tools whose tip is beside or under the fingers keep it above the table.

Added after the review of 2026-11-13: a simulated copy of the real tool (what `fd --sim tool set` stores) is
read only while the real tool file is what was copied, and a tool attached in simulation stays; `ready`
refuses, before the torque comes on, when the jaw stands elsewhere than where the tool file has it (8 ticks),
except for a tool recorded on the fixed jaw or without a jaw position; the example of a brush's wrist limit
is the measured 97.

Added with round 3 (2026-11-14): the refusal of a locked command names the fold its caller gives (`fd paint rest`
beside a painting set-up) and is word for word what it was without one; `fd --sim tool detach` says that a mirrored
real tool is still on; a pen command refused for the jaw (`fd tool calibrate`, `fd draw`) has the way out for that
tool as its next and a say that names the jaw; a joint limit to change is named in words, not as a command with
JOINT=LO:HI in backticks.

Every test uses $FD_TOOL (a scratch file), so the real tool_active.json is never read or written.
"""

import os as _os  # the harness modules live in tools/<feature>/ (see tools/tazuna_paths.py)
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "tools"))
import tazuna_paths  # noqa: E402,F401

import contextlib
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["FD_CONFIG"] = os.path.join(HERE, "testdata", "pen_20261111", "config.json")
os.environ["FD_CAMERA_MODEL"] = os.path.join(HERE, "testdata", "pen_20261111", "camera_model.json")
_TMP = tempfile.mkdtemp(prefix="jh_tooling_")
TOOL_PATH = os.path.join(_TMP, "tool_active.json")
os.environ["FD_TOOL"] = TOOL_PATH


def _tree_tool_files() -> dict:
    """The tree's own tool files as they are now (path -> bytes, None = no file). A tool may really be
    attached while these tests run: they must leave these files exactly as they found them."""
    out = {}
    for name in ("tool_active.json", "tool_active.sim.json"):
        p = os.path.join(tazuna_paths.ROOT, name)
        try:
            with open(p, "rb") as fh:
                out[p] = fh.read()
        except FileNotFoundError:
            out[p] = None
    return out


_TREE_TOOL_FILES = _tree_tool_files()  # before any test ran


def _no_hardware(event, args):
    """Audit hook: these tests must never open the servo bus or the camera."""
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        p = os.fsdecode(args[0])
        if p.startswith("/dev/ttyACM") or p.startswith("/dev/ttyUSB") or p.startswith("/dev/video"):
            raise RuntimeError(f"test touched hardware: open({p})")
    if event == "subprocess.Popen" and args and "v4l2" in str(args[0]) + str(args[1] if len(args) > 1 else ""):
        raise RuntimeError(f"test touched the camera: {args[:2]}")


sys.addaudithook(_no_hardware)

import numpy as np  # noqa: E402

import fd  # noqa: E402
import kinematics as K  # noqa: E402
import skills  # noqa: E402
import tooling as TL  # noqa: E402
from robot import ARM, Arm, MotionAborted, SafetyError, load_config  # noqa: E402
from servo_bus import JOINTS  # noqa: E402
from sim_bus import REST_RAW, SimBus  # noqa: E402

fd.log_event = lambda *a, **k: None  # no run log from tests
np.set_printoptions(precision=4, suppress=True)
CFG = load_config()
TABLE_Z = float(CFG["table_z"])
TPR = 4096.0 / (2.0 * math.pi)
FRAME = os.path.join(HERE, "testdata", "0030_overview_raw.jpg")
PEN_TIP = [0.0015, -0.0081, 0.0621]            # the marker of 2026-11-11, measured on the paper
NIGHT_REST_DEG = [20.0, -100.2, 85.2, 42.0, 0.0]  # what held on the real arm with it (tip 56 mm up, torque off)
LENGTHS = (0.02, 0.04, 0.06, 0.08, 0.10, 0.12)


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


def pen(**over):
    kw = dict(name="pen2", tip_offset=list(PEN_TIP), mount="along_fingers", pitch_deg=75.0)
    kw.update(over)
    return TL.Tool(**kw)


def straight(mount, length):
    return TL.Tool(name=f"{mount[:6]}{int(round(length * 100))}", tip_offset=TL.offset_for(CFG, mount, length),
                   mount=mount, pitch_deg=TL.DEFAULT_PITCH[mount])


def q_of(pos, cfg=CFG):
    z, s = cfg["zero_ticks"], cfg["sign"]
    return np.array([s[j] * (pos[j] - z[j]) / TPR for j in ARM])


def place(arm, q, gripper=None):
    """Put the simulated arm at joint pose q (as if it had been left there)."""
    bus = arm.bus
    raw = arm.q_to_raw_f(q)
    if gripper is not None:
        raw["gripper"] = gripper
    for j, v in raw.items():
        bus.pos[j] = bus.goal[j] = bus.sp[j] = float(v)


def sim_arm(tool=None, q=None, torque=False, cfg=None):
    """A simulated arm (as fd.make_arm(True) builds it) carrying `tool`; at q (default: the plain rest
    pose), limp or holding. With torque the simulated sag has settled before the test starts."""
    bus = RecBus(realtime=False)
    arm = Arm(bus=bus, cfg=cfg if cfg is not None else load_config(), tool=tool).open()
    if q is not None:
        place(arm, q)
    if torque:
        arm.torque_on()
        bus.sleep(0.6)
    bus.trace.clear()
    return arm, bus


def heights(arm, bus, tool, start=0):
    return [TL.tip_height(arm, q_of(p), tool) for p in bus.trace[start:]]


def writes(bus):
    return bus.io["writes"], len(bus.log)


def gripper_goals(bus, start=0):
    return [v for (_, j, v) in bus.log[start:] if j == "gripper"]


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


def fd_run(argv, bus=None, cfg=None):
    """fd.main(["--sim", ...]) on a simulated arm the test holds (same construction as fd.make_arm(True),
    without its start pose). Returns (result dict, exit code)."""
    assert argv[0] == "--sim", "tests run the simulated arm only"
    real = fd.make_arm
    if bus is not None:
        def make(sim):
            assert sim, "a test asked for the REAL arm"
            return Arm(bus=bus, cfg=cfg if cfg is not None else load_config(), tool=TL.active(sim=True))
        fd.make_arm = make
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            rc = fd.main(list(argv))
    finally:
        fd.make_arm = real
    return json.loads(buf.getvalue().strip().splitlines()[-1]), rc


# ----------------------------------------------------------------------------- the tool file
def test_no_file_means_no_tool_and_the_old_arm():
    with tool_file(None):
        assert TL.active() is None and TL.active(sim=True) is None and TL.source() is None
        arm, bus = sim_arm(tool="active")
        assert arm.tool is None
        assert np.allclose(arm.tip_tool(), K.tool_offset(*CFG["tip_offset"]))
        # the poses are the plain ones, computed as before
        lim = arm.q_limits()
        q = arm.raw_to_q(arm._c("rest_raw"))
        want = np.array([min(max(q[i], lim[j][0] + 0.005), lim[j][1] - 0.005) for i, j in enumerate(ARM)])
        assert np.array_equal(arm.rest_q(), want)
        r = arm._c("ready")
        assert np.array_equal(arm.ready_q(), arm.solve(np.array(r["xyz"], float), float(r["pitch"]), float(r["roll"])))
        # the gripper works, park works (through skills: fd's park also resets the camera)
        skills.go_ready(arm)
        assert arm.set_gripper(0.5)["open_frac"] > 0.4
        skills.park(arm)
        assert not arm.is_torqued() and skills._is_parked(arm.read_q())
        assert not os.path.exists(TOOL_PATH)


def test_save_load_roundtrip_keeps_unknown_keys():
    with tool_file(None):
        t = pen(gripper_raw=2061, region=[0.17, -0.06, 0.31, 0.07], notes=["marker, fine side"],
                contact={"x0": 0.24, "y0": 0.0, "a": -0.0015, "bx": 0.0, "by": -0.004, "rms_mm": 0.3, "points": []},
                corr_map={"centre": [0.223, 0.003], "kind": "affine", "coef": [[0.21, 0.0], [1.02, 0.05], [-0.1, 1.01]],
                          "rms_mm": 1.96, "max_mm": 3.26, "n": 9},
                calibrated={"tip": {"at": "2026-11-11T22:44:00"}})
        t.extra = {"added_by_a_later_version": {"k": 1}}
        TL.save(t)
        assert os.path.exists(TOOL_PATH) and not os.path.exists(TOOL_PATH + ".tmp")
        back = TL.active()
        assert back.to_dict() == t.to_dict(), (back.to_dict(), t.to_dict())
        assert back.extra == {"added_by_a_later_version": {"k": 1}}
        assert abs(TL.contact_z(back, (0.24, 0.05)) - (-0.0015 - 0.004 * 0.05)) < 1e-12
        assert TL.calibration_state(back) == {"tip": True, "contact": True, "map": True}
        assert TL.calibration_state(pen()) == {"tip": False, "contact": False, "map": False}
        TL.clear()
        assert TL.active() is None
        TL.clear()  # clearing twice is fine


def test_a_damaged_file_is_an_error_never_no_tool():
    with tool_file(None):
        for text in ("{not json", "[]", json.dumps({"name": "pen"}), json.dumps({"name": "", "tip_offset": [0, 0, 0.06]}),
                     json.dumps({"name": "pen", "tip_offset": [0, 0, "x"]}),
                     json.dumps({"name": "pen", "tip_offset": [0, 0, 60]}),      # millimetres typed as metres
                     json.dumps({"name": "pen", "tip_offset": [0, 0, 0.06], "pitch_deg": "steep"})):
            with open(TOOL_PATH, "w") as f:
                f.write(text)
            for fn in (TL.active, lambda: Arm(bus=SimBus(), cfg=load_config())):
                try:
                    fn()
                    raise AssertionError(f"accepted {text!r}")
                except SafetyError as e:
                    assert "fd tool detach" in str(e), e
            r, rc = fd_run(["--sim", "tool", "status"])
            assert rc == 1 and r["ok"] is False and r["stage"] == "tool" and r["attached"] is None, r
            r, rc = fd_run(["--sim", "grip", "0.5"], RecBus())
            assert rc == 1 and r["error"] == "SafetyError", r
        # ... and detach gets rid of it
        r, rc = fd_run(["--sim", "tool", "detach"])
        assert rc == 0 and r["removed"] and not os.path.exists(TOOL_PATH), r
        # saving something that could not be read back is refused too
        for bad in (pen(tip_offset=[0, 0, float("nan")]), pen(name=" "), pen(pitch_deg=200.0)):
            try:
                TL.save(bad)
                raise AssertionError("saved a bad tool")
            except SafetyError:
                pass
        assert not os.path.exists(TOOL_PATH)


def test_config_with_tool_is_a_copy():
    cfg = load_config()
    before = json.dumps(cfg, sort_keys=True)
    out = TL.config_with_tool(cfg, pen())
    assert out["tip_offset"] == PEN_TIP and out is not cfg and out["keepouts"] is not cfg["keepouts"]
    assert json.dumps(cfg, sort_keys=True) == before
    same = TL.config_with_tool(cfg, None)
    assert same == cfg and same is not cfg


def test_mount_geometry():
    tip = np.array(CFG["tip_offset"])
    assert np.allclose(TL.offset_for(CFG, "along_fingers", 0.06), tip + [0, 0, 0.06], atol=1e-6)
    assert np.allclose(TL.offset_for(CFG, "across_fingers", 0.06), tip + [0.06, 0, 0], atol=1e-6)
    for bad in (0.0, -0.01, 5.0, float("nan"), "long"):
        try:
            TL.offset_for(CFG, "along_fingers", bad)
            raise AssertionError(bad)
        except SafetyError:
            pass
    assert TL.infer_mount(CFG, PEN_TIP) == "along_fingers"
    assert TL.infer_mount(CFG, [0.05, 0.004, 0.01]) == "across_fingers"
    assert TL.infer_mount(CFG, [-0.05, 0.0, 0.0068]) == "across_fingers"
    assert TL.infer_mount(CFG, [0.03, 0.0, 0.04]) == "other"
    assert TL.infer_mount(CFG, list(tip)) == "other"
    # the axis of a tool: along the fingers = the gripper's approach axis, whatever sideways offset the tape gives
    assert np.allclose(TL.tool_axis(CFG, pen()), K._APPROACH_LOCAL)
    assert np.allclose(TL.tool_axis(CFG, straight("across_fingers", 0.06)), -np.asarray(K._JAW_OPEN_LOCAL))
    other = TL.Tool("hook", [0.03, 0.0, 0.0468], "other")
    assert abs(np.linalg.norm(TL.tool_axis(CFG, other)) - 1.0) < 1e-9 and TL.tool_axis(CFG, other)[0] > 0.5
    assert abs(TL.length_of(CFG, straight("along_fingers", 0.08)) - 0.08) < 1e-6
    # an across-fingers tool points DOWN when the gripper is level at roll 0 (that is how such a pen is used)
    arm, _ = sim_arm()
    q = arm.solve(np.array([0.25, 0.0, 0.10]), 30.0, 0.0)
    axis = K.fk(q)[:3, :3] @ TL.tool_axis(CFG, straight("across_fingers", 0.06))
    assert axis[2] < -0.8, axis


def test_sim_writes_its_own_file_and_mirrors_the_real_one():
    """`fd --sim tool ...` must never change what the REAL arm believes is on its gripper."""
    real, simf = os.path.join(_TMP, "real_tool.json"), os.path.join(_TMP, "sim_tool.json")
    saved = (TL.TOOL_FILE, TL.SIM_TOOL_FILE, os.environ.pop("FD_TOOL"))
    TL.TOOL_FILE, TL.SIM_TOOL_FILE = real, simf
    try:
        assert TL.tool_path() == real and TL.tool_path(sim=True) == simf
        assert TL.active() is None and TL.active(sim=True) is None
        TL.save(pen(name="real_pen"))
        assert TL.active().name == "real_pen" and TL.active(sim=True).name == "real_pen"  # the simulation mirrors
        assert TL.source(sim=True) == real
        r, rc = fd_run(["--sim", "tool", "attach", "--name", "sim_pen", "--along-fingers", "0.10"])
        assert rc == 0 and r["file"] == simf and os.path.exists(simf), r
        assert TL.active().name == "real_pen" and TL.active(sim=True).name == "sim_pen"
        assert fd.make_arm(True).tool.name == "sim_pen"
        r, rc = fd_run(["--sim", "tool", "set", "--pitch", "70"])
        assert rc == 0 and TL.active(sim=True).pitch_deg == 70.0 and TL.active().pitch_deg == 75.0
        r, rc = fd_run(["--sim", "tool", "detach"])
        assert rc == 0 and not os.path.exists(simf) and TL.active().name == "real_pen", r
        r, rc = fd_run(["--sim", "tool", "detach"])  # nothing simulated left: the real tool is NOT touched, and still mirrored
        assert rc == 0 and r["removed"] is False and TL.active().name == "real_pen" and TL.active(sim=True).name == "real_pen", r
        # (round 3, T8: this answered "attached": false and 「ツールは登録されていませんでした」 while the simulated arm
        # carried the mirrored tool; the truth stood in a key "note" only)
        assert r["attached"] is True and r["mirrors"] == "real_pen" and "note" not in r and "stay locked" in r["next"], r
        assert "real_pen" in r["say"] and "ロックされたまま" in r["say"] and "登録されていませんでした" not in r["say"], r["say"]
        r, rc = fd_run(["--sim", "park"])            # ... which is what the next command meets
        assert rc == 1 and r["stage"] == "tool" and r["tool"] == "real_pen", r
        # a change made in simulation to the mirrored tool is a copy, not a change of the real one
        r, rc = fd_run(["--sim", "tool", "set", "--name", "renamed"])
        assert rc == 0 and TL.active().name == "real_pen" and TL.active(sim=True).name == "renamed", r
    finally:
        TL.TOOL_FILE, TL.SIM_TOOL_FILE = saved[0], saved[1]
        os.environ["FD_TOOL"] = saved[2]
        for p in (real, simf):
            if os.path.exists(p):
                os.remove(p)


# ----------------------------------------------------------------------------- the arm with a tool
def test_arm_learns_the_tool_when_it_is_built():
    with tool_file(pen()):
        arm = Arm(bus=SimBus(), cfg=load_config()).open()
        assert arm.tool is not None and arm.tool.name == "pen2"
        assert np.allclose(arm.tip_tool(), K.tool_offset(*PEN_TIP))
        assert arm.cfg["tip_offset"] == CFG["tip_offset"], "cfg keeps the fingertip (the jaw line needs it)"
        assert arm._check_ctx()["jaw_slack"] == Arm(bus=SimBus(), cfg=load_config(), tool=None).open()._check_ctx()["jaw_slack"]
        st = arm.state()
        q = arm.read_q()
        assert np.allclose(st["tip_xyz"], K.fk(q, K.tool_offset(*PEN_TIP))[:3, 3], atol=1e-4)
        assert np.allclose(arm.rest_q(), TL.rest_q(arm, arm.tool)) and np.allclose(arm.ready_q(), TL.ready_q(arm, arm.tool))
        assert fd.make_arm(True).tool.name == "pen2"
    assert Arm(bus=SimBus(), cfg=load_config()).tool is None
    assert Arm(bus=SimBus(), cfg=load_config(), tool=pen()).tool.name == "pen2"  # given explicitly: no file needed
    try:
        Arm(bus=SimBus(), cfg=load_config(), tool="pen")
        raise AssertionError("a tool name is not a tool")
    except SafetyError:
        pass


def test_floor_and_plan_are_about_the_tool_tip():
    arm, _ = sim_arm(tool=pen())
    plain, _ = sim_arm(tool=None)
    # fingertip 4 cm above the table, gripper pointing down: fine for the bare gripper, the pen is in the table
    q = plain.solve(np.array([0.25, 0.0, TABLE_Z + 0.04]), 80.0, 0.0)
    plain.check_q(q)
    try:
        arm.check_q(q)
        raise AssertionError("the tool tip is below the table")
    except SafetyError as e:
        assert "below floor limit" in str(e), e
    # solve() puts the TOOL tip where it is asked
    q = arm.solve(np.array([0.25, 0.02, 0.03]), 75.0, 0.0)
    assert np.allclose(K.fk(q, K.tool_offset(*PEN_TIP))[:3, 3], [0.25, 0.02, 0.03], atol=1e-4)
    arm.check_q(q)


def test_gripper_is_locked_before_any_bus_traffic():
    arm, bus = sim_arm(tool=pen(), q=None)
    place(arm, TL.ready_q(arm, arm.tool))
    arm.torque_on()
    bus.sleep(0.5)
    q = arm.read_q()
    before, reads = writes(bus), bus.io["reads"]
    calls = {
        "set_gripper": lambda: arm.set_gripper(0.5),
        "set_gripper raw": lambda: arm.set_gripper(raw=3000, wait=False),
        "grasp": lambda: arm.grasp(),
        "pinch": lambda: arm.pinch(),
        "grasp_fast": lambda: arm.grasp_fast(),
        "gripper_raw_for": lambda: arm.gripper_raw_for(0.1),
        "_gripper_clamp": lambda: arm._gripper_clamp(2500),
        "check_path": lambda: arm.check_path([q, q], 20.0, gripper_goal_raw=2500),
        "run_path": lambda: arm.run_path([q, q], 20.0, gripper_goal_raw=2500),
        "skills.park": lambda: skills.park(arm),
    }
    for name, fn in calls.items():
        try:
            fn()
            raise AssertionError(f"{name} was not refused")
        except SafetyError as e:
            assert "pen2" in str(e) and "fd tool rest" in str(e) and "fd tool detach" in str(e), (name, e)
        assert writes(bus) == before, f"{name} wrote to the bus"
    assert arm._gripper_clamp() == Arm(bus=SimBus(), cfg=load_config(), tool=None)._gripper_clamp()  # the range is still told
    assert arm.is_torqued() and bus.pos["gripper"] == REST_RAW["gripper"]
    # the same calls on an arm without a tool go through
    plain, pbus = sim_arm(tool=None)
    skills.go_ready(plain)
    assert plain.set_gripper(0.5)["target_raw"] == plain.gripper_raw_for(0.5)
    del reads


# ----------------------------------------------------------------------------- fd guards
GUARDED = [
    ["grip", "0.5"], ["grasp"], ["grasp", "--fast"], ["grasp", "--thin"],
    ["pick", "--color", "blue", "--frame", FRAME], ["pick", "--id", "1", "--dry", "--frame", FRAME],
    ["take", "--color", "white", "--hollow", "--frame", FRAME],
    ["do", "青いキューブを箱に入れて", "--frame", FRAME], ["do", "箱の中の白いキャップを取り出して", "--frame", FRAME],
    ["park"], ["park", "--tier", "slow"], ["probe"], ["bench", "--tier", "slow"],
    ["repl", "--frame", FRAME], ["calib-at", "0.22", "0.0"], ["servo-tip", "0.22", "0.0", "0.03"],
]


def _guard_round(start: str):
    tool = pen()
    with tool_file(tool):
        arm, bus = sim_arm(tool=tool)
        if start == "resting":      # limp in the tool's rest pose, as the arm was left on 2026-11-11
            place(arm, TL.rest_q(arm, tool), gripper=2061)
        else:                       # holding at the tool stand-by pose
            place(arm, TL.ready_q(arm, tool), gripper=2061)
            arm.torque_on()
            bus.sleep(0.6)
        for argv in GUARDED:
            before, pos = writes(bus), dict(bus.pos)
            torque = dict(bus.torque)
            r, rc = fd_run(["--sim"] + argv, bus)
            assert rc == 1 and r["ok"] is False and r.get("stage") == "tool", (argv, r)
            assert r["error"] == "SafetyError" and r["tool"] == "pen2", (argv, r)
            assert "fd tool rest" in r["next"] and "fd tool detach" in r["next"], (argv, r)
            assert "pen2" in r["say"] and "fd tool rest" in r["say"], (argv, r)
            assert writes(bus) == before, f"{argv}: {writes(bus)[0] - before[0]} bus write(s) before the refusal"
            assert bus.torque == torque and bus.pos == pos, argv
        assert gripper_goals(bus) in ([], [2061]), gripper_goals(bus)  # (the hold of torque_on, if any; never a goal)


def test_guarded_commands_refuse_without_a_bus_write_when_resting():
    _guard_round("resting")


def test_guarded_commands_refuse_without_a_bus_write_when_holding():
    _guard_round("holding")


def test_the_refusal_names_the_fold_the_caller_gives():
    """Round 3 (T14). The refusal of a locked command named `fd tool rest` for every tool; with a painting set-up on
    record that command is refused over the canvas, and fd names `fd paint rest` instead (fd._fold_words). Here:
    tooling's half. Without a fold the texts are what they were, word for word; with one, next, reason and say
    name it, and the detach command in the caller's spelling."""
    t = pen()
    old_next = ("`fd tool rest` folds the arm away with the tool on (torque off); once the tool is off the "
                "gripper, `fd tool detach` unlocks the gripper and `park`")
    assert TL.NEXT_LOCKED == TL.next_locked() == old_next
    r = TL.refusal(t, "park")
    assert r["next"] == old_next and r["reason"] == TL.lock_message(t, "park") and r["reason"].endswith("Nothing moved. " + old_next), r
    assert "片付けるときは fd tool rest、ツールを外したあとは fd tool detach を使ってください。" in r["say"] and "paint" not in r["say"], r["say"]
    r = TL.refusal(t, "grip", fold="fd --sim paint rest", detach="fd --sim tool detach")
    assert r["ok"] is False and r["stage"] == "tool" and r["tool"] == "pen2" and r["message"] == r["reason"], r
    assert r["next"].startswith("`fd --sim paint rest` folds the arm away with the tool on, clear of the canvas"), r["next"]
    assert "`fd --sim tool detach` unlocks the gripper" in r["next"] and "fd tool rest" not in r["next"] and r["next"] in r["reason"], r
    assert "fd paint rest" in r["say"] and "fd tool rest はキャンバスの上に畳むため断られます" in r["say"] and "grip" in r["say"], r["say"]
    assert "`fd paint rest` folds" in TL.next_locked("fd paint rest") and "`fd tool detach`" in TL.next_locked("fd paint rest")
    # torque off outside the rest pose: the same word of the caller
    arm, bus = sim_arm(tool=t)
    place(arm, TL.ready_q(arm, t))
    arm.torque_on()
    bus.sleep(0.5)
    assert "Use `fd tool rest` (folds the arm away" in TL.torque_off_block(arm)
    why = TL.torque_off_block(arm, "fd --sim paint rest")
    assert "Use `fd --sim paint rest` (folds the arm away" in why and "fd tool rest" not in why, why


def test_do_that_would_not_move_is_still_answered():
    """`do` is refused only when it would pick or take: a sentence it cannot run is handed back as before."""
    tool = pen()
    with tool_file(tool):
        arm, bus = sim_arm(tool=tool)
        place(arm, TL.rest_q(arm, tool))
        before = writes(bus)
        real = fd.cmd_scene
        fd.cmd_scene = lambda arm, a: {"objects": [], "warnings": [], "view": None}  # (perception is not under test)
        try:
            r, rc = fd_run(["--sim", "do", "青いキューブを箱に入れないで", "--frame", FRAME], bus)
        finally:
            fd.cmd_scene = real
        assert rc == 1 and r.get("need_llm") is True and r.get("stage") != "tool", r
        assert writes(bus) == before


def test_commands_that_stay_open_with_a_tool():
    """state, check, tip and joints work on the TOOL tip; nothing of it goes to the gripper."""
    tool = pen()
    with tool_file(tool):
        arm, bus = sim_arm(tool=tool)
        place(arm, TL.ready_q(arm, tool), gripper=2061)
        r, rc = fd_run(["--sim", "torque", "on"], bus)
        assert rc == 0, r
        r, rc = fd_run(["--sim", "state"], bus)
        assert rc == 0 and abs(r["tip_xyz"][2] - 0.12) < 0.012, r
        r, rc = fd_run(["--sim", "check", "0.25", "0.0", "0.03", "--pitch", "75"], bus)
        assert rc == 0 and r["ok"], r
        r, rc = fd_run(["--sim", "tip", "0.25", "0.0", "0.03", "--pitch", "75"], bus)
        assert rc == 0, r
        tip = K.fk(q_of(bus.pos), K.tool_offset(*PEN_TIP))[:3, 3]
        assert np.allclose(tip, [0.25, 0.0, 0.03], atol=0.004), tip
        r, rc = fd_run(["--sim", "tip", "0.25", "0.0", "-0.02", "--pitch", "75"], bus)  # 2 cm into the table
        assert rc == 1 and "below floor limit" in r["message"], r
        # down to the table height by hand: yes. Pressing below it (--contact): not with a tool on
        r, rc = fd_run(["--sim", "tip", "0.25", "0.0", str(TABLE_Z + 0.0005), "--pitch", "75", "--straight"], bus)
        assert rc == 0, r
        for argv in (["tip", "0.25", "0.0", "-0.005", "--pitch", "75", "--contact"],
                     ["tip", "0", "0", "-0.003", "--rel", "--contact"], ["joints", "wrist_flex=2", "--rel", "--contact"]):
            before, pos = writes(bus), dict(bus.pos)
            r, rc = fd_run(["--sim"] + argv, bus)
            assert rc == 1 and r["stage"] == "tool" and "--contact" in r["reason"] and "fd draw" in r["reason"], (argv, r)
            # (round 3, T12: the pen commands are named in words; the one command in backticks works as it is typed)
            assert "the pen commands (`fd tool status` says" in r["next"] and "`fd draw`" not in r["next"] + r["reason"], (argv, r)
            assert writes(bus) == before and bus.pos == pos, argv
        r, rc = fd_run(["--sim", "tip", "0", "0", "0.03", "--rel"], bus)                 # and up again
        assert rc == 0, r
        assert all(v == 2061 for v in gripper_goals(bus)) and round(bus.pos["gripper"]) == 2061, gripper_goals(bus)


# ----------------------------------------------------------------------------- ready
def _check_ready(arm, bus, tool, r, h_start, label):
    """The tool tip ended at the stand-by point; on the way it only rose until it was clear and never came
    lower than 2 cm above the table after that; the gripper got no goal but the hold."""
    hs = heights(arm, bus, tool)
    assert hs, label
    tip = K.fk(q_of(bus.pos), K.tool_offset(*tool.tip_offset))[:3, 3]
    want = np.array(r["standby_xyz"])
    assert np.linalg.norm(tip - want) < 0.004, (label, tip, want)
    if h_start >= TL.MIN_FREE_HEIGHT + TL.PLAN_MARGIN:
        assert min(hs) >= TL.MIN_FREE_HEIGHT, (label, min(hs))
    else:
        k = next(i for i, h in enumerate(hs) if h >= TL.MIN_FREE_HEIGHT + TL.PLAN_MARGIN)
        assert min(hs[:k + 1]) >= h_start - 0.0015, (label, "it dipped while lifting", min(hs[:k + 1]), h_start)
        assert min(hs[k:]) >= TL.MIN_FREE_HEIGHT, (label, min(hs[k:]))
    assert abs(r["min_tip_height_m"] - min(min(hs), h_start)) < 0.004, (label, r["min_tip_height_m"], min(hs))
    grip = round(bus.pos["gripper"])
    assert all(v == grip for v in gripper_goals(bus)), (label, gripper_goals(bus))


def test_ready_from_the_folded_tool_rest_pose():
    tool = pen()
    with tool_file(tool):
        arm, bus = sim_arm(tool=tool)
        place(arm, np.radians(NIGHT_REST_DEG), gripper=2061)     # exactly where the real arm stands, limp
        h0 = TL.tip_height(arm, arm.read_q(), tool)
        assert abs(h0 - 0.0565) < 0.001, h0
        r, rc = fd_run(["--sim", "ready"], bus)
        assert rc == 0 and r["ok"] and r["tool"] == "pen2" and r["standby_moved_m"] == 0.0, r
        assert r["standby_xyz"] == [0.2, 0.0, 0.12] and not r["lifted_first"], r
        assert np.allclose(r["tip_xyz"], [0.2, 0.0, 0.12], atol=0.004), r["tip_xyz"]
        _check_ready(arm, bus, tool, r, h0, "night rest pose")
        assert arm.is_torqued() and round(bus.pos["gripper"]) == 2061
        # and again from there: nothing to do, nothing goes wrong
        bus.trace.clear()
        r, rc = fd_run(["--sim", "ready"], bus)
        assert rc == 0 and r["ok"], r


def test_ready_from_everywhere_the_pen_was_that_night():
    tool = pen()
    arm0, _ = sim_arm(tool=tool)
    starts = {"aside": np.radians([43.15, -20.31, 10.46, 63.28, -0.05]),
              "found": np.radians([-7.998, -85.526, 86.309, 6.24, 9.088]),
              "tool rest": TL.rest_q(arm0, tool),
              "plain rest (tool inside the table: simulation only)": Arm(bus=SimBus(), cfg=load_config(), tool=None).open().rest_q()}
    for (x, y) in ((0.17, -0.06), (0.2575, 0.003), (0.31, 0.06), (0.22, 0.09), (0.30, -0.06)):
        for h in (-0.002, 0.001, 0.008, 0.02, 0.03):
            starts[f"paper ({x}, {y}) h={h}"] = TL.work_q(arm0, tool, (x, y), h)[0]
    lifted = 0
    for label, q in starts.items():
        with tool_file(tool):
            arm, bus = sim_arm(tool=tool, q=q, torque=True)
            h0 = TL.tip_height(arm, arm.read_q(), tool)
            r = skills.go_ready(arm)
            assert r["tool"] == "pen2", label
            assert r["lifted_first"] == (h0 < TL.LIFT_HEIGHT - 0.002), (label, h0, r["lifted_first"])
            lifted += bool(r["lifted_first"])
            _check_ready(arm, bus, tool, r, h0, label)
    assert lifted >= 20, lifted


def test_ready_with_tools_of_every_length():
    moved = {}
    for mount in ("along_fingers", "across_fingers"):
        for length in LENGTHS:
            tool = straight(mount, length)
            arm, bus = sim_arm(tool=tool)
            place(arm, TL.rest_q(arm, tool))
            h0 = TL.tip_height(arm, arm.read_q(), tool)
            r = skills.go_ready(arm)
            _check_ready(arm, bus, tool, r, h0, tool.name)
            moved[tool.name] = r["standby_moved_m"]
            pose = TL.ready_pose(arm, tool)
            assert pose["xyz"] == r["standby_xyz"] and pose["moved_m"] == r["standby_moved_m"]
    # up to 8 cm the tip stands at the configured point; a longer tool along the fingers stands by further out
    assert all(moved[f"along_{n}"] == 0.0 for n in (2, 4, 6, 8)), moved
    assert 0.0 < moved["along_12"] <= 0.12, moved
    assert all(moved[f"across{n}"] == 0.0 for n in (2, 4, 6, 8, 10, 12)), moved


def test_ready_goes_around_what_stands_in_the_direct_way():
    """A keep-out on the straight joint-space way: the wrist is moved first or last instead (two pieces,
    a stop between them), nothing of the tool enters the keep-out, and the tip never comes below 2 cm."""
    tool = pen()
    cfg = load_config()
    box = [0.215, 0.239, -0.041, -0.017, 0.082, 0.106]     # 24 mm cube around the tip's direct path
    cfg["keepouts"]["post"] = box
    arm, bus = sim_arm(tool=tool, cfg=cfg)
    place(arm, TL.rest_q(arm, tool))
    h0 = TL.tip_height(arm, arm.read_q(), tool)
    plan = TL.plan_free_move(arm, tool, arm.read_q(), TL.ready_q(arm, tool))
    assert plan["via"] in ("wrist first", "wrist last") and len(plan["path"]) == 3 and plan["lift"] is None, plan["via"]
    r = skills.go_ready(arm)
    assert r["via"] == plan["via"] and r["pieces"] == 2, r
    _check_ready(arm, bus, tool, r, h0, "around the post")
    for pos in bus.trace:
        for pt in TL.body_points(arm, q_of(pos), tool):
            assert not (box[0] <= pt[0] <= box[1] and box[2] <= pt[1] <= box[3] and box[4] <= pt[2] <= box[5]), pt
    # the same arm without the post goes straight
    arm, bus = sim_arm(tool=tool)
    place(arm, TL.rest_q(arm, tool))
    assert skills.go_ready(arm)["via"] == "direct"


def test_ready_refuses_before_the_torque_goes_on():
    # a "tool" the arm cannot stand by with: nothing may be written
    tool = TL.Tool("pole", TL.offset_for(CFG, "along_fingers", 0.25), "along_fingers", 75.0)
    arm, bus = sim_arm(tool=tool)
    before = writes(bus)
    try:
        skills.go_ready(arm)
        raise AssertionError("a 25 cm pole has no stand-by pose")
    except SafetyError as e:
        assert "stand-by" in str(e), e
    assert writes(bus) == before and not arm.is_torqued()
    # something in the way of every route: refused with the arm still limp where it was
    tool = pen()
    cfg = load_config()
    cfg["keepouts"]["shelf"] = [0.10, 0.40, -0.30, 0.30, 0.095, 0.11]   # a slab between the rest pose and the stand-by point
    arm, bus = sim_arm(tool=tool, cfg=cfg)
    place(arm, TL.rest_q(arm, tool))
    before, pos = writes(bus), dict(bus.pos)
    try:
        skills.go_ready(arm)
        raise AssertionError("the way is blocked")
    except SafetyError as e:
        assert "Nothing moved" in str(e) and "shelf" in str(e), e
    assert writes(bus) == before and bus.pos == pos and not arm.is_torqued()
    # an unknown speed tier refuses first, as without a tool
    arm, bus = sim_arm(tool=pen())
    place(arm, TL.rest_q(arm, arm.tool))
    try:
        skills.go_ready(arm, "warp")
        raise AssertionError
    except SafetyError as e:
        assert "unknown speed tier" in str(e)
    assert writes(bus) == (0, 0)


def test_a_free_move_stops_when_the_tool_comes_down():
    """The planner keeps the tool up; if the arm nevertheless brings it below 2 cm, the motion is stopped."""
    tool = pen()
    arm, bus = sim_arm(tool=tool)
    place(arm, TL.ready_q(arm, tool))
    arm.torque_on()
    bus.sleep(0.6)
    bus.trace.clear()
    q = arm.read_q()
    q_low = TL.work_q(arm, tool, (0.25, 0.0), 0.003)[0]
    try:
        TL.plan_free_move(arm, tool, q, q_low)
        raise AssertionError("a free move must not end 3 mm above the table")
    except SafetyError:
        pass
    forced = {"lift": None, "path": [q, q_low], "via": "forced by the test", "min_height_m": 0.003}
    try:
        TL.free_move(arm, q_low, 25.0, plan=forced)
        raise AssertionError("the motion was not stopped")
    except MotionAborted as e:
        assert e.info["reason"] == "monitor" and "tool tip" in str(e), e
    hs = heights(arm, bus, tool)
    assert 0.012 < min(hs) < TL.MIN_FREE_HEIGHT, min(hs)       # stopped within a few millimetres of the limit
    bus.sleep(1.0)
    assert TL.tip_height(arm, arm.read_q(), tool) > 0.012 and arm.is_torqued()   # and it holds there


def test_lift_raises_a_low_tool_straight_up():
    tool = pen()
    arm0, _ = sim_arm(tool=tool)
    for xy, h in (((0.2575, 0.003), -0.002), ((0.17, -0.06), 0.004), ((0.31, 0.06), 0.015)):
        arm, bus = sim_arm(tool=tool, q=TL.work_q(arm0, tool, xy, h)[0], torque=True)
        p0 = K.fk(arm.read_q(), arm.tip_tool())[:3, 3]
        r = TL.lift(arm)
        p1 = K.fk(arm.read_q(), arm.tip_tool())[:3, 3]
        assert r["lifted"] and abs(r["tip_height_m"] - TL.LIFT_HEIGHT) < 0.003, r
        assert np.linalg.norm(p1[:2] - p0[:2]) < 0.004, (p0, p1)            # straight up
        hs = heights(arm, bus, tool)
        assert min(hs) >= (p0[2] - TABLE_Z) - 0.0015 and all(b >= a - 0.0015 for a, b in zip(hs, hs[1:])), min(hs)
        assert gripper_goals(bus) == [REST_RAW["gripper"]]                   # (the hold of torque_on)
        assert TL.lift(arm) == {"lifted": False, "tip_height_m": r["tip_height_m"]}   # already up: nothing moves
    plain, _ = sim_arm(tool=None)
    try:
        TL.lift(plain)
        raise AssertionError
    except SafetyError:
        pass


# ----------------------------------------------------------------------------- the rest pose
def test_rest_pose_is_the_one_that_held_on_the_real_arm():
    tool = pen()
    arm, _ = sim_arm(tool=tool)
    q = TL.rest_q(arm, tool)
    assert abs(TL.tip_height(arm, q, tool) - 0.05) < 2e-4
    d = np.degrees(q) - np.array(NIGHT_REST_DEG)
    assert np.all(np.abs(d) < 2.5), d          # same fold; the wrist 2 deg lower (tip at 50 instead of 56 mm)
    night = np.radians(NIGHT_REST_DEG)
    assert abs(TL.tip_height(arm, night, tool) - 0.0565) < 0.001
    TL.check_pose(arm, tool, night, min_height=0.05)
    assert TL.in_rest_pose(arm, tool, night) == (True, "")
    # asking for the height it had that night gives that night's pose
    q56 = TL.rest_q(arm, tool, clearance_m=0.0565)
    assert np.all(np.abs(np.degrees(q56) - np.array(NIGHT_REST_DEG)) < 0.35), np.degrees(q56)
    # the plain rest pose and the stand-by pose are NOT rest poses for an arm with this pen
    plain = Arm(bus=SimBus(), cfg=load_config(), tool=None).open().rest_q()
    assert TL.tip_height(arm, plain, tool) < -0.01
    assert not TL.in_rest_pose(arm, tool, plain)[0] and not TL.in_rest_pose(arm, tool, TL.ready_q(arm, tool))[0]
    for bad in ({"clearance_m": 0.01}, {"clearance_m": 0.5}, {"pan_deg": 170.0}, {"clearance_m": "high"}):
        try:
            TL.rest_q(arm, tool, **bad)
            raise AssertionError(bad)
        except SafetyError:
            pass


def test_tool_rest_for_tools_2_to_12_cm_along_and_across():
    done, refused = [], []
    for mount in ("along_fingers", "across_fingers"):
        for length in LENGTHS + (0.03, 0.07, 0.11):
            tool = straight(mount, length)
            with tool_file(tool):
                arm, bus = sim_arm(tool=tool)
                place(arm, TL.ready_q(arm, tool), gripper=2061)
                arm.torque_on()
                bus.sleep(0.6)
                bus.trace.clear()
                n_log = len(bus.log)
                before, pos = writes(bus), dict(bus.pos)
                r, rc = fd_run(["--sim", "tool", "rest"], bus)
                if not r["ok"]:
                    # a clean refusal: nothing moved, the torque is as it was, and it says what to do
                    assert rc == 1 and r.get("error") == "SafetyError" and r["stage"] == "plan" and r["next"], (tool.name, r)
                    assert writes(bus) == before and bus.pos == pos and arm.is_torqued(), tool.name
                    refused.append(tool.name)
                    continue
                assert rc == 0 and r["stage"] == "rest" and r["torque"] == "off" and r["moved"], (tool.name, r)
                h_end = TL.tip_height(arm, q_of(bus.pos), tool)
                assert 0.04 <= h_end <= 0.07, (tool.name, h_end)
                assert abs(r["tip_height_mm"] - h_end * 1000) < 1.0 and r["sag_after_release_mm"] < 1.0, (tool.name, r)
                assert min(heights(arm, bus, tool)) >= TL.MIN_FREE_HEIGHT, (tool.name, min(heights(arm, bus, tool)))
                assert not any(bus.torque.values()), tool.name
                assert TL.in_rest_pose(arm, tool, q_of(bus.pos))[0], tool.name
                assert gripper_goals(bus, n_log) == [] and round(bus.pos["gripper"]) == 2061, tool.name
                assert "fd tool detach" in r["next"] and tool.name in r["say"], r
                done.append(tool.name)
    assert "along_6" in done and len(done) + len(refused) == 18, (done, refused)
    assert not refused, f"refused (cleanly, but unexpected with this arm): {refused}"


def test_tool_rest_from_the_paper_dry_keep_torque_and_twice():
    tool = pen()
    with tool_file(tool):
        arm, bus = sim_arm(tool=tool)
        place(arm, TL.work_q(arm, tool, (0.26, 0.02), 0.0)[0], gripper=2061)   # the pen stands on the paper
        arm.torque_on()
        bus.sleep(0.6)
        h0 = TL.tip_height(arm, arm.read_q(), tool)
        bus.trace.clear()
        before, pos = writes(bus), dict(bus.pos)
        r, rc = fd_run(["--sim", "tool", "rest", "--dry"], bus)
        assert rc == 0 and r["stage"] == "planned" and r["moved"] is False and r["lift_first"], r
        assert abs(r["rest_tip_height_mm"] - 50.0) < 0.5 and r["planned_min_tip_height_mm"] >= 25.0, r
        assert writes(bus) == before and bus.pos == pos and arm.is_torqued(), "a dry run moved or wrote something"
        r, rc = fd_run(["--sim", "tool", "rest", "--keep-torque"], bus)
        assert rc == 0 and r["ok"] and r["torque"] == "on" and arm.is_torqued(), r
        assert abs(r["tip_height_mm"] - TL.tip_height(arm, q_of(bus.pos), tool) * 1000) < 1.0, r   # where it IS now
        assert abs(r["start_tip_height_mm"] - h0 * 1000) < 1.0 and 40.0 < r["tip_height_mm"] < 60.0, r
        hs = heights(arm, bus, tool)
        k = next(i for i, h in enumerate(hs) if h >= TL.MIN_FREE_HEIGHT + TL.PLAN_MARGIN)
        assert min(hs[:k + 1]) >= h0 - 0.0015 and min(hs[k:]) >= TL.MIN_FREE_HEIGHT, (h0, min(hs[:k + 1]), min(hs[k:]))
        r, rc = fd_run(["--sim", "tool", "rest"], bus)                        # now let go
        assert rc == 0 and r["ok"] and r["torque"] == "off" and not arm.is_torqued(), r
        before, pos = writes(bus), dict(bus.pos)
        r, rc = fd_run(["--sim", "tool", "rest"], bus)                        # already resting
        assert rc == 0 and r["ok"] and r["moved"] is False and "already" in r["note"], r
        assert writes(bus) == before and bus.pos == pos
        r, rc = fd_run(["--sim", "tool", "rest", "--clearance", "0.065", "--pan", "-15"], bus)
        assert rc == 0 and r["ok"] and r["moved"] is False, r      # a limp arm that rests is left alone
        assert writes(bus) == before and bus.pos == pos
        # ... `ready` first, then it folds into the other rest pose, and the torque may go off there too
        r, rc = fd_run(["--sim", "ready"], bus)
        assert rc == 0, r
        r, rc = fd_run(["--sim", "tool", "rest", "--clearance", "0.12", "--pan", "-15", "--keep-torque"], bus)
        assert rc == 0 and r["ok"] and r["moved"] and abs(r["rest_tip_height_mm"] - 120.0) < 0.5, r
        assert abs(math.degrees(q_of(bus.pos)[0]) + 15.0) < 1.0 and arm.is_torqued()
        r, rc = fd_run(["--sim", "torque", "off"], bus)
        assert rc == 0 and not arm.is_torqued(), r
        assert abs(TL.tip_height(arm, q_of(bus.pos), tool) - 0.12) < 0.012
        assert all(v == 2061 for v in gripper_goals(bus)) and round(bus.pos["gripper"]) == 2061
    r, rc = fd_run(["--sim", "tool", "rest"], RecBus())                       # no tool: park is the command
    assert rc == 1 and r["stage"] == "tool" and "fd park" in r["next"], r


def test_tool_rest_refuses_cleanly_when_the_arm_cannot_rest():
    tool = pen()
    cfg = load_config()
    cfg["keepouts"]["box"] = [0.15, 0.40, -0.25, 0.05, -0.05, 0.20]     # something stands where the tool would rest
    arm, bus = sim_arm(tool=tool, cfg=cfg)
    place(arm, np.radians([-30.0, -40.0, 30.0, 70.0, 0.0]))
    arm.torque_on()
    bus.sleep(0.6)
    before, pos = writes(bus), dict(bus.pos)
    for dry in (True, False):
        try:
            TL.rest(arm, dry=dry)
            raise AssertionError("no rest pose is free")
        except SafetyError as e:
            assert "cannot rest" in str(e) and "fd tool detach" in str(e), e
        assert writes(bus) == before and bus.pos == pos and arm.is_torqued()
    assert TL.in_rest_pose(arm, tool)[0] is False and "no rest pose" in TL.in_rest_pose(arm, tool)[1]
    with tool_file(tool):                # the command says the same as a complete result
        for extra in ([], ["--dry"]):
            r, rc = fd_run(["--sim", "tool", "rest"] + extra, bus, cfg=cfg)
            assert rc == 1 and r["ok"] is False and r["stage"] == "plan" and r["moved"] is False, r
            assert r["error"] == "SafetyError" and "cannot rest" in r["reason"] and "fd tool detach" in r["next"], r
            assert r["torque"] == "on" and "動かしていません" in r["say"], r
            assert writes(bus) == before and bus.pos == pos and arm.is_torqued()
    try:
        arm.torque_off()                 # ... and so the torque cannot simply be switched off either
        raise AssertionError
    except SafetyError as e:
        assert "refusing torque off" in str(e)
    assert arm.is_torqued()


def test_tool_rest_that_is_stopped_on_the_way_says_so_and_keeps_the_torque():
    tool = pen()
    with tool_file(tool):
        arm, bus = sim_arm(tool=tool)
        place(arm, TL.ready_q(arm, tool), gripper=2061)
        arm.torque_on()
        bus.sleep(0.6)
        lift_stop = arm.q_to_raw_f(np.radians([0.0, -80.0, 0.0, 0.0, 0.0]))["shoulder_lift"]
        bus.blocker = lambda pos: pos["shoulder_lift"] < lift_stop       # something keeps the arm from folding back
        r, rc = fd_run(["--sim", "tool", "rest"], bus)
        assert rc == 1 and r["ok"] is False and r["error"] == "MotionAborted" and r["stage"] == "move", r
        assert r["moved"] is True and r["torque"] == "on" and arm.is_torqued(), r
        assert "fd tool rest" in r["next"] and "止まっています" in r["say"], r
        # with one joint held back the others would have swung the tool down: stopped at the 2 cm limit
        assert min(heights(arm, bus, tool)) > 0.012 and TL.tip_height(arm, q_of(bus.pos), tool) > 0.012
        assert all(v == 2061 for v in gripper_goals(bus))
        bus.blocker = None                                                # the way is free again: it finishes
        r, rc = fd_run(["--sim", "tool", "rest"], bus)
        assert rc == 0 and r["ok"] and r["torque"] == "off", r


# ----------------------------------------------------------------------------- torque off
def test_torque_off_is_refused_in_a_raised_pose():
    tool = pen()
    with tool_file(tool):
        arm, bus = sim_arm(tool=tool)
        raised = {"stand-by": TL.ready_q(arm, tool), "aside": np.radians([43.15, -20.31, 10.46, 63.28, -0.05]),
                  "on the paper": TL.work_q(arm, tool, (0.25, 0.0), 0.0)[0],
                  "plain rest pose": Arm(bus=SimBus(), cfg=load_config(), tool=None).open().rest_q()}
        for label, q in raised.items():
            arm, bus = sim_arm(tool=tool, q=q, torque=True)
            before = writes(bus)
            r, rc = fd_run(["--sim", "torque", "off"], bus)
            assert rc == 1 and r["stage"] == "tool" and "fd tool rest" in r["next"] and "pen2" in r["say"], (label, r)
            assert writes(bus) == before and all(bus.torque.values()), label
            try:
                arm.torque_off()         # the library call is refused too
                raise AssertionError(label)
            except SafetyError as e:
                assert "refusing torque off" in str(e) and "fd tool rest" in str(e), e
            assert writes(bus) == before and all(bus.torque.values()), label
        # the emergency stop stays: --force releases anywhere
        r, rc = fd_run(["--sim", "torque", "off", "--force"], bus)
        assert rc == 0 and not any(bus.torque.values()), r
        # in the tool's rest pose it is simply allowed
        arm, bus = sim_arm(tool=tool, q=TL.rest_q(arm, tool), torque=True)
        r, rc = fd_run(["--sim", "torque", "off"], bus)
        assert rc == 0 and r["torque"] == "off" and r["tool"] == "pen2" and not any(bus.torque.values()), r
        r, rc = fd_run(["--sim", "torque", "off"], bus)      # already limp: nothing to protect
        assert rc == 0, r
        assert gripper_goals(bus) in ([], [REST_RAW["gripper"]])
    # without a tool the old guard is the one that speaks
    arm, bus = sim_arm(tool=None)
    skills.go_ready(arm)
    r, rc = fd_run(["--sim", "torque", "off"], bus)
    assert rc == 1 and "use `park`" in r["message"] and "stage" not in r, r


# ----------------------------------------------------------------------------- lean
def test_lean_points_where_the_pen_leaned():
    """2026-11-11: gripper pitch 75, the pen leaned 15 deg with its tip pointing away from the robot."""
    tool = pen()
    arm, _ = sim_arm(tool=tool)
    for xy in ((0.2575, 0.003), (0.20, -0.057), (0.315, 0.063), (0.17, 0.0)):
        v = TL.lean_xy(arm, tool, xy)
        radial = np.array([xy[0] - K._X_PAN, xy[1]])
        radial /= np.linalg.norm(radial)
        assert v.shape == (2,) and abs(np.linalg.norm(v) - 1.0) < 1e-9
        assert float(v @ radial) > 0.97, (xy, v, radial)
        q, pitch = TL.work_q(arm, tool, xy)
        axis = K.fk(q)[:3, :3] @ TL.tool_axis(CFG, tool)
        assert abs(math.degrees(math.atan2(math.hypot(axis[0], axis[1]), -axis[2])) - (90.0 - pitch)) < 0.5
    # across the fingers at pitch 30: the tool leans 30 deg the other way (tip towards the robot)
    across = straight("across_fingers", 0.06)
    v = TL.lean_xy(arm, across, (0.26, 0.0))
    assert v[0] < -0.97, v
    # an upright tool has no lean: the direction away from the robot is returned
    up = straight("across_fingers", 0.10)
    up.pitch_deg = 0.0
    q, pitch = TL.work_q(arm, up, (0.26, 0.03))
    if abs(pitch) < 0.5:
        v = TL.lean_xy(arm, up, (0.26, 0.03))
        radial = np.array([0.26 - K._X_PAN, 0.03])
        assert np.allclose(v, radial / np.linalg.norm(radial), atol=0.05), v
    try:
        TL.lean_xy(arm, tool, (0.60, 0.0))
        raise AssertionError("out of reach")
    except SafetyError as e:
        assert "out of reach" in str(e)


# ----------------------------------------------------------------------------- the tool commands
def test_attach_set_status_detach():
    with tool_file(None):
        bus = RecBus()
        bus.pos["gripper"] = bus.goal["gripper"] = 2061.0
        r, rc = fd_run(["--sim", "tool", "status"])
        assert rc == 0 and r["attached"] is False and "fd tool attach" in r["next"] and r["say"], r
        # --dry shows, writes nothing
        r, rc = fd_run(["--sim", "tool", "attach", "--name", "pen2", "--tip", "0.0015,-0.0081,0.0621", "--dry"], bus)
        assert rc == 0 and r["stage"] == "planned" and r["saved"] is False and not os.path.exists(TOOL_PATH), r
        r, rc = fd_run(["--sim", "tool", "attach", "--name", "pen2", "--tip", "0.0015,-0.0081,0.0621", "--note", "marker"], bus)
        assert rc == 0 and r["stage"] == "attached" and r["saved"], r
        t = TL.active()
        assert (t.name, t.tip_offset, t.mount, t.pitch_deg, t.gripper_raw, t.notes) == (
            "pen2", PEN_TIP, "along_fingers", 75.0, 2061, ["marker"]), t
        assert t.attached_at and t.contact is None and t.corr_map is None and t.calibrated == {}
        assert r["arm"]["rest"]["tip_height_mm"] == 50.0 and "park" in r["locked"] and "fd tool rest" in r["next"], r
        assert writes(bus) == (0, 0), "attach only looks at the arm"
        # a second attach is refused; --replace is the way
        r, rc = fd_run(["--sim", "tool", "attach", "--name", "other", "--along-fingers", "0.08"], bus)
        assert rc == 1 and r["stage"] == "tool" and "--replace" in r["next"] and TL.active().name == "pen2", r
        r, rc = fd_run(["--sim", "tool", "status"])
        assert rc == 0 and r["attached"] and r["tool"]["name"] == "pen2" and "fd tool calibrate" in r["next"], r
        assert r["tool"]["calibrated"] == {"tip": False, "contact": False, "map": False}
        # set: a note and a region keep the calibration, a new tip drops it
        t.contact = {"x0": 0.24, "y0": 0.0, "a": -0.0015, "bx": 0.0, "by": 0.0, "rms_mm": 0.2, "points": []}
        t.corr_map = {"centre": [0.22, 0.0], "kind": "affine", "coef": [[0, 0], [1, 0], [0, 1]], "rms_mm": 1.0, "max_mm": 2.0, "n": 9}
        t.calibrated = {"tip": {"at": "x"}, "contact": {"at": "x"}, "map": {"at": "x"}}
        TL.save(t)
        r, rc = fd_run(["--sim", "tool", "status"])
        assert "calibrated" in r["next"] and "fd draw" in r["next"] and r["tool"]["corr_map"]["rms_mm"] == 1.0, r
        r, rc = fd_run(["--sim", "tool", "set", "--region", "0.17,-0.06,0.31,0.07", "--note", "new sheet"], bus)
        assert rc == 0 and r["changed"] == ["region", "notes"] and not r["calibration_dropped"], r
        assert TL.active().corr_map is not None and TL.active().region == [0.17, -0.06, 0.31, 0.07]
        r, rc = fd_run(["--sim", "tool", "set", "--pitch", "80", "--dry"], bus)
        assert rc == 0 and r["calibration_dropped"] and not r["saved"] and TL.active().pitch_deg == 75.0, r
        r, rc = fd_run(["--sim", "tool", "set", "--pitch", "80", "--keep-calibration"], bus)
        assert rc == 0 and TL.active().pitch_deg == 80.0 and TL.active().corr_map is not None, r
        r, rc = fd_run(["--sim", "tool", "set", "--tip", "0.002,-0.008,0.060"], bus)
        assert rc == 0 and r["calibration_dropped"] and "fd tool calibrate" in r["next"], r
        t = TL.active()
        assert t.tip_offset == [0.002, -0.008, 0.06] and t.contact is None and t.corr_map is None and t.calibrated == {}
        assert t.gripper_raw == 2061 and t.region == [0.17, -0.06, 0.31, 0.07]
        r, rc = fd_run(["--sim", "tool", "set", "--region", "none", "--name", "marker"], bus)
        assert rc == 0 and TL.active().region is None and TL.active().name == "marker", r
        r, rc = fd_run(["--sim", "tool", "set"], bus)
        assert rc == 0 and r["changed"] == [] and r["saved"] is False, r
        # replace, across the fingers, without looking at the arm
        r, rc = fd_run(["--sim", "tool", "attach", "--name", "stylus", "--across-fingers", "0.05", "--replace", "--no-robot"])
        assert rc == 0, r
        t = TL.active()
        assert (t.name, t.mount, t.pitch_deg, t.gripper_raw) == ("stylus", "across_fingers", 30.0, None), t
        assert np.allclose(t.tip_offset, np.array(CFG["tip_offset"]) + [0.05, 0, 0], atol=1e-6)
        # detach: dry first, then for real
        r, rc = fd_run(["--sim", "tool", "detach", "--dry"])
        assert rc == 0 and r["removed"] is False and TL.active() is not None, r
        r, rc = fd_run(["--sim", "tool", "detach"])
        assert rc == 0 and r["removed"] and r["tool"] == "stylus" and TL.active() is None and "park" in r["say"], r
        r, rc = fd_run(["--sim", "tool", "detach"])
        assert rc == 0 and r["removed"] is False, r
        r, rc = fd_run(["--sim", "tool", "set", "--pitch", "70"])
        assert rc == 1 and "fd tool attach" in r["next"], r
        assert writes(bus) == (0, 0)


def test_attach_rejects_nonsense():
    with tool_file(None):
        for argv in (["--name", "p", "--tip", "0.01,0.02"], ["--name", "p", "--tip", "a,b,c"],
                     ["--name", "p", "--tip", "0,0,62"], ["--name", "p", "--along-fingers", "6"],
                     ["--name", "p", "--along-fingers", "-0.06"], ["--name", "p", "--across-fingers", "0"],
                     ["--name", " ", "--along-fingers", "0.06"], ["--name", "p", "--along-fingers", "0.06", "--pitch", "500"]):
            r, rc = fd_run(["--sim", "tool", "attach", "--no-robot"] + argv)
            assert rc == 1 and r["error"] == "SafetyError" and not os.path.exists(TOOL_PATH), (argv, r)
        r, rc = fd_run(["--sim", "tool", "calibrate"])       # stage 2 fills this in
        assert rc == 1 and r["stage"] == "tool", r


def test_after_detach_everything_is_as_before():
    with tool_file(None):
        bus = RecBus()
        r, rc = fd_run(["--sim", "tool", "attach", "--name", "pen2", "--tip", "0.0015,-0.0081,0.0621"], bus)
        assert rc == 0
        r, rc = fd_run(["--sim", "grip", "0.5"], bus)
        assert rc == 1 and r["stage"] == "tool"
        r, rc = fd_run(["--sim", "tool", "detach"], bus)
        assert rc == 0
        arm = fd.make_arm(True).open()
        assert arm.tool is None and tuple(round(arm.bus.pos[j]) for j in JOINTS) == tuple(REST_RAW[j] for j in JOINTS)
        assert np.allclose(arm.tip_tool(), K.tool_offset(*CFG["tip_offset"]))
        # ready, grip, grasp and park as always (ready is the plain one: the FINGERTIP at the stand-by point)
        r, rc = fd_run(["--sim", "ready"], bus)
        assert rc == 0 and "tool" not in r and np.allclose(r["tip_xyz"], [0.2, 0.0, 0.12], atol=0.004), r
        r, rc = fd_run(["--sim", "grip", "0.5"], bus)
        assert rc == 0 and abs(r["open_frac"] - 0.5) < 0.03, r
        r, rc = fd_run(["--sim", "grasp", "--fast"], bus)
        assert rc == 0, r
        r, rc = fd_run(["--sim", "tip", "0.25", "0.0", str(TABLE_Z - 0.003), "--pitch", "80", "--contact"], bus)
        assert rc == 0, r                                           # --contact is the fingertip's again
        r, rc = fd_run(["--sim", "ready"], bus)
        assert rc == 0, r
        a2 = Arm(bus=bus, cfg=load_config()).open()
        skills.park(a2)
        assert not a2.is_torqued() and skills._is_parked(a2.read_q())
        fingertip = K.fk(a2.read_q(), a2.tip_tool())[:3, 3]
        assert 0.02 < fingertip[2] - TABLE_Z < 0.04, fingertip      # the plain rest pose: fingertip about 3 cm up


def test_old_suite_passes_after_attach_and_detach():
    """An attach / detach pair leaves no trace: the simulation suite of the tree passes with the same
    $FD_TOOL (97 tests, about 15 s)."""
    if os.environ.get("JH_QUICK"):
        raise Skip("JH_QUICK set")
    with tool_file(None):
        fd_run(["--sim", "tool", "attach", "--name", "pen2", "--along-fingers", "0.06", "--no-robot"])
        assert os.path.exists(TOOL_PATH)
        # with the tool on, the old suite is NOT expected to pass (its gripper and park tests are refused)
        fd_run(["--sim", "tool", "detach"])
        assert not os.path.exists(TOOL_PATH)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        env.pop("FD_CONFIG", None)
        env.pop("FD_CAMERA_MODEL", None)
        p = subprocess.run([sys.executable, os.path.join(HERE, "test_sim.py")], capture_output=True, text=True,
                           timeout=600, env=env, cwd=HERE)
        tail = p.stdout.strip().splitlines()[-1] if p.stdout.strip() else p.stderr[-400:]
        assert p.returncode == 0 and "failures: 0" in tail and "tests: 97" in tail, tail


# ----------------------------------------------------------------------------- joint limits of a tool
# 2026-11-13: a flat brush with a 30 cm handle lying back over the forearm. The handle hit the forearm when the
# wrist straightened (below about 20 deg) and when the gripper rolled: `--joint-limit wrist_flex=20:97`,
# `wrist_roll=-5:5`. The limits were written that morning without a test; these are the tests.
BRUSH_TIP = [0.0085, 0.0007, 0.0611]                    # the flat brush, measured in the air (+- 1.5 mm)
BRUSH_LIMITS = {"wrist_flex": [20.0, 97.0], "wrist_roll": [-5.0, 5.0]}
SIDE_TIP = [-0.09, -0.01, 0.06]                         # a brush on the opened moving jaw: 9 cm beside the finger line
UNDER_TIP = [0.016, -0.001, 0.056]                      # the same brush with the jaw closed: the tip under the fingers


def brush(**over):
    kw = dict(name="brush", tip_offset=list(BRUSH_TIP), mount="along_fingers", pitch_deg=65.0,
              joint_limits_deg={j: list(v) for j, v in BRUSH_LIMITS.items()})
    kw.update(over)
    return TL.Tool(**kw)


def lim_deg(arm):
    return {j: (math.degrees(a), math.degrees(b)) for j, (a, b) in arm.q_limits().items()}


def outside_deg(arm, q, tool=None) -> float:
    """How far (deg) pose q lies outside the arm's joint limits with `tool` on it (<= 0: inside)."""
    lim = TL._limits(arm, arm.tool if tool is None else tool)
    return max(max(math.degrees(lim[j][0] - q[i]), math.degrees(q[i] - lim[j][1])) for i, j in enumerate(ARM))


def outside_tool_deg(tool, q) -> float:
    """How far (deg) pose q lies outside the TOOL's own joint limits (<= 0: inside). The simulated shoulder and elbow
    droop by a degree under torque, past the arm's own limits where a pose sits at them (the fold): the tool's
    limits are what a trace of the simulated arm is held to."""
    return max([max(lo - math.degrees(q[ARM.index(j)]), math.degrees(q[ARM.index(j)]) - hi)
                for j, (lo, hi) in (tool.joint_limits_deg or {}).items()] or [-1e9])


def refused(fn, *texts) -> str:
    try:
        fn()
    except SafetyError as e:
        for t in texts:
            assert t in str(e), (t, str(e))
        return str(e)
    raise AssertionError("not refused")


def test_joint_limits_load_and_save_unchanged():
    with tool_file(None):
        t = brush(gripper_raw=2054, notes=["flat brush, its band about 9 mm wide"], contact={"x0": 0.24, "y0": 0.0, "a": 0.0, "rms_mm": 0.4, "points": []})
        TL.save(t)
        text = open(TOOL_PATH).read()
        assert json.loads(text)["joint_limits_deg"] == BRUSH_LIMITS
        back = TL.active()
        assert back.joint_limits_deg == BRUSH_LIMITS and back.to_dict() == t.to_dict()
        TL.save(back)
        assert open(TOOL_PATH).read() == text, "a load / save round changed the file"
        # written by hand with whole numbers, in another order, with a joint more: the same limits come back
        d = json.loads(text)
        d["joint_limits_deg"] = {"wrist_roll": [-5, 5], "shoulder_pan": [-90, 90.5], "wrist_flex": [20, 97]}
        with open(TOOL_PATH, "w") as f:
            json.dump(d, f)
        back = TL.active()
        assert back.joint_limits_deg == {"wrist_roll": [-5.0, 5.0], "shoulder_pan": [-90.0, 90.5], "wrist_flex": [20.0, 97.0]}
        assert list(back.joint_limits_deg) == ["wrist_roll", "shoulder_pan", "wrist_flex"]
        TL.save(back)
        assert TL.active().joint_limits_deg == back.joint_limits_deg and TL.active().contact == t.contact
        # the arm that is built with the file has them
        arm = Arm(bus=SimBus(), cfg=load_config()).open()
        got = lim_deg(arm)
        assert np.allclose(got["wrist_flex"], (20.0, 97.0)) and np.allclose(got["wrist_roll"], (-5.0, 5.0))
        assert np.allclose(got["shoulder_pan"], (-90.0, 90.5))
        assert fd.make_arm(True).tool.joint_limits_deg == back.joint_limits_deg


def test_bad_joint_limits_are_an_error_never_no_limits():
    bad = [{"wrist": [20, 97]}, {"gripper": [0, 50]}, {"wrist_flex": [97, 20]}, {"wrist_flex": [20, 20]}, {"wrist_flex": [20]},
           {"wrist_flex": [20, 97, 100]}, {"wrist_flex": "20:97"}, {"wrist_flex": None}, {"wrist_flex": ["a", "b"]},
           {"wrist_flex": [float("nan"), 97]}, {"wrist_flex": [-400, 97]}, {"wrist_flex": [20, 361]}, {"wrist_flex": 97},
           {"wrist_flex": {"lo": 20, "hi": 97}}, [["wrist_flex", 20, 97]], "wrist_flex=20:97", 5,
           {"wrist_flex": [20, 97], "elbow": [0, 1]}]
    for v in bad:
        msg = refused(lambda: TL.check_joint_limits(v), "joint limits")
        assert ("unknown joint" in msg) == (isinstance(v, dict) and any(k not in ARM for k in v)), (v, msg)
    assert TL.check_joint_limits(None) is None and TL.check_joint_limits({}) is None
    assert TL.check_joint_limits({"wrist_flex": (20, 97)}) == {"wrist_flex": [20.0, 97.0]}
    with tool_file(None):
        good = brush().to_dict()
        for v in bad:
            d = dict(good, joint_limits_deg=v)
            with open(TOOL_PATH, "w") as f:
                f.write(json.dumps(d).replace("NaN", "null"))
            # never read as a tool without limits: not by the file functions, not by an arm, not by a command
            for fn in (TL.active, lambda: TL.active(sim=True), lambda: Arm(bus=SimBus(), cfg=load_config()), lambda: fd.make_arm(True)):
                msg = refused(fn, "cannot be used", "fd tool detach")
                assert "joint limits" in msg, (v, msg)
            r, rc = fd_run(["--sim", "tool", "status"])
            assert rc == 1 and r["ok"] is False and r["stage"] == "tool" and r["attached"] is None, (v, r)
            assert "joint limits" in r["reason"], (v, r)
            bus = RecBus()
            r, rc = fd_run(["--sim", "ready"], bus)
            assert rc == 1 and r["error"] == "SafetyError" and writes(bus) == (0, 0), (v, r)
        TL.clear()
        # the command line: refused, nothing stored
        attach = ["--sim", "tool", "attach", "--name", "brush", "--tip", "0.0085,0.0007,0.0611", "--no-robot"]
        for arg in ("wrist=20:97", "wrist_flex=97:20", "wrist_flex=20", "wrist_flex=a:b", "wrist_flex", "gripper=0:40",
                    "wrist_flex=20:97:3"):
            r, rc = fd_run(attach + ["--joint-limit", arg])
            assert rc == 1 and r["error"] == "SafetyError" and not os.path.exists(TOOL_PATH), (arg, r)
        TL.save(brush())
        text = open(TOOL_PATH).read()
        for arg in ("wrist=20:97", "wrist_flex=97:20", "elbow_flex=x:1"):
            r, rc = fd_run(["--sim", "tool", "set", "--joint-limit", arg])
            assert rc == 1 and r["error"] == "SafetyError" and open(TOOL_PATH).read() == text, (arg, r)
    # a Tool made in code with limits the arm cannot apply: the arm says so, it does not drop them
    for v, texts in (({"wrist": [20.0, 97.0]}, ("unknown joint", "wrist")), ({"wrist_flex": [97.0, 20.0]}, ("wrist_flex", "lo < hi")),
                     ({"wrist_flex": "20:97"}, ("wrist_flex",)), ([20.0, 97.0], ("joint limits",))):
        arm, bus = sim_arm(tool=brush(joint_limits_deg=v))
        q = Arm(bus=SimBus(), cfg=load_config(), tool=None).open().ready_q()
        for call in (arm.q_limits, lambda: arm.check_q(q), lambda: arm.solve(np.array([0.25, 0.0, 0.08]), 65.0, 0.0),
                     lambda: arm.plan_tip_path([(np.array([0.25, 0.0, 0.08]), 65.0, 0.0)], q_start=q), lambda: skills.go_ready(arm)):
            refused(call, *texts)
        assert writes(bus) == (0, 0)
    # limits that leave nothing of what the arm itself allows: an error that names the tool, the joint and the way out
    for v, joint in (({"wrist_flex": [150.0, 170.0]}, "wrist_flex"), ({"shoulder_pan": [120.0, 140.0]}, "shoulder_pan"),
                     ({"wrist_flex": [20.0, 97.0], "elbow_flex": [-200.0, -150.0]}, "elbow_flex")):
        tool = brush(joint_limits_deg=v)
        arm, bus = sim_arm(tool=tool)
        msg = refused(arm.q_limits, "tool 'brush'", joint, "no pose would be left", f"fd tool set --joint-limit {joint}=LO:HI")
        for call in (lambda: arm.check_q(np.zeros(5)), lambda: TL.rest_q(arm, tool), lambda: TL.ready_pose(arm, tool),
                     lambda: skills.go_ready(arm), lambda: TL.rest(arm)):
            assert refused(call) == msg or "no pose would be left" in refused(call)
        assert writes(bus) == (0, 0) and TL.in_rest_pose(arm, tool)[0] is False
        with tool_file(tool):                              # the commands still answer, and `tool set` can repair it
            r, rc = fd_run(["--sim", "ready"])
            assert rc == 1 and "no pose would be left" in r["message"], r
            r, rc = fd_run(["--sim", "tool", "status"])
            assert rc == 0 and r["tool"]["joint_limits_deg"] == v, r
            r, rc = fd_run(["--sim", "tool", "set", "--joint-limit", "none"])
            assert rc == 0 and TL.active().joint_limits_deg is None, r
            r, rc = fd_run(["--sim", "ready"])
            assert rc == 0, r


def test_a_tool_file_without_joint_limits_behaves_as_before():
    with tool_file(None):
        plain = pen()
        TL.save(plain)
        d = json.load(open(TOOL_PATH))
        assert d["joint_limits_deg"] is None
        for variant in ("absent", None, {}):
            dd = dict(d)
            if variant == "absent":
                del dd["joint_limits_deg"]                 # as every file written before 2026-11-13
            else:
                dd["joint_limits_deg"] = variant
            with open(TOOL_PATH, "w") as f:
                json.dump(dd, f)
            t = TL.active()
            assert t.joint_limits_deg is None and t.to_dict() == plain.to_dict(), variant
            arm, _ = sim_arm(tool="active")
            bare, _ = sim_arm(tool=None)
            assert arm.q_limits() == bare.q_limits(), variant
            assert np.array_equal(TL.rest_q(arm, t), TL.rest_q(arm, plain)) and np.array_equal(TL.ready_q(arm, t), TL.ready_q(arm, plain))
            assert abs(TL.tip_height(arm, TL.rest_q(arm, t), t) - 0.05) < 2e-4
        r, rc = fd_run(["--sim", "tool", "status"])
        assert rc == 0 and "joint_limits_deg" not in r["tool"], r
        r, rc = fd_run(["--sim", "ready"])
        want = TL.ready_pose(sim_arm(tool=plain)[0], plain)
        assert rc == 0 and r["standby_xyz"] == [0.2, 0.0, 0.12] == want["xyz"] and r["standby_pitch_deg"] == want["pitch"], r


def test_q_limits_is_the_intersection_of_servo_soft_and_tool_limits():
    cfg = load_config()
    raw = SimBus().limits()
    m, z = cfg["limit_margin_ticks"], cfg["zero_ticks"]
    servo = {j: ((raw[j][0] + m - z[j]) / TPR, (raw[j][1] - m - z[j]) / TPR) for j in ARM}        # sign +1 on every joint
    soft = {j: tuple(math.radians(v) for v in cfg["soft_limits_deg"][j]) for j in cfg["soft_limits_deg"]}
    assert set(soft) == {"shoulder_pan", "wrist_roll"} and servo["shoulder_pan"][1] > soft["shoulder_pan"][1]

    def want(tool_lim):
        out = {}
        for j in ARM:
            lo, hi = servo[j]
            for extra in (soft.get(j), tool_lim.get(j) and tuple(math.radians(v) for v in tool_lim[j])):
                if extra:
                    lo, hi = max(lo, extra[0]), min(hi, extra[1])
            out[j] = (lo, hi)
        return out

    cases = [{}, BRUSH_LIMITS,
             {"wrist_flex": [-170.0, 170.0], "wrist_roll": [-180.0, 180.0]},      # wider than servo and soft limits: no effect
             {"shoulder_pan": [-120.0, 30.0]},                                    # one side from the tool, the other from the soft limit
             {"shoulder_lift": [-150.0, 0.0], "elbow_flex": [0.0, 150.0]},        # one side from the tool, the other from the servo
             {"wrist_roll": [100.0, 170.0]},                                      # the tool's lower end, the soft limit's upper end
             {j: [-1.0, 1.0] for j in ARM}]
    for tool_lim in cases:
        arm, _ = sim_arm(tool=brush(joint_limits_deg=tool_lim or None))
        got, exp = arm.q_limits(), want(tool_lim)
        assert list(got) == list(ARM)
        for j in ARM:
            assert np.allclose(got[j], exp[j], atol=1e-12), (tool_lim, j, got[j], exp[j])
            # never wider than without the tool
            assert got[j][0] >= want({})[j][0] - 1e-12 and got[j][1] <= want({})[j][1] + 1e-12
    bare, _ = sim_arm(tool=None)
    assert all(np.allclose(bare.q_limits()[j], want({})[j], atol=1e-12) for j in ARM)
    # the numbers of that morning
    got = lim_deg(sim_arm(tool=brush())[0])
    assert np.allclose(got["wrist_flex"], (20.0, 97.0)) and np.allclose(got["wrist_roll"], (-5.0, 5.0))
    assert np.allclose(got["shoulder_pan"], (-100.0, 100.0)) and abs(got["elbow_flex"][1] - 85.43) < 0.01
    # other soft limits, a joint whose servo counts the other way: still the intersection of the three
    cfg2 = load_config()
    cfg2["soft_limits_deg"] = {"wrist_flex": [-30, 60]}
    cfg2["sign"]["wrist_flex"] = -1
    arm2, _ = sim_arm(tool=brush(joint_limits_deg={"wrist_flex": [10.0, 80.0], "shoulder_pan": [-150.0, 150.0]}), cfg=cfg2)
    a, b = sorted((-(raw["wrist_flex"][0] + m - z["wrist_flex"]) / TPR, -(raw["wrist_flex"][1] - m - z["wrist_flex"]) / TPR))
    assert a < math.radians(-30) and b > math.radians(60)
    assert np.allclose(arm2.q_limits()["wrist_flex"], (math.radians(10.0), math.radians(60.0)), atol=1e-12)
    assert np.allclose(arm2.q_limits()["shoulder_pan"], servo["shoulder_pan"], atol=1e-12)      # (no soft limit on pan in cfg2)
    # the limits are those of the tool that is on the arm NOW
    arm, _ = sim_arm(tool=brush())
    arm.tool = None
    assert arm.q_limits() == bare.q_limits()
    arm.tool = pen()
    assert arm.q_limits() == bare.q_limits()


def test_ik_plan_and_check_refuse_a_pose_beyond_the_tool_limit_and_name_the_joint():
    arm, bus = sim_arm(tool=brush())
    free, _ = sim_arm(tool=brush(joint_limits_deg=None))
    # the same tip pose: fine without the limits (the wrist at 17.7 deg), refused with them
    xyz = np.array([0.32, 0.0, 0.15])
    q_low = free.solve(xyz, 10.0, 0.0)
    assert abs(math.degrees(q_low[3]) - 17.7) < 0.2
    msg = refused(lambda: arm.solve(xyz, 10.0, 0.0), "IK failed", "wrist_flex=17.7 deg outside [20.0, 97.0]")
    assert "feasible pitch range" in msg                               # ... and it says which pitches the limits leave
    q_roll = free.solve(np.array([0.25, 0.0, 0.05]), 65.0, 10.0)
    refused(lambda: arm.solve(np.array([0.25, 0.0, 0.05]), 65.0, 10.0), "wrist_roll=10.0 deg outside [-5.0, 5.0]")
    refused(lambda: arm.solve(np.array([0.25, 0.0, 0.05]), 65.0, -5.5), "wrist_roll=-5.5 deg outside [-5.0, 5.0]")
    assert np.allclose(arm.solve(np.array([0.25, 0.0, 0.05]), 65.0, 5.0), free.solve(np.array([0.25, 0.0, 0.05]), 65.0, 5.0))
    # a straight line that runs the wrist past 97 deg on its way: the bare limits allow it (101.9), the tool's do not
    q0 = arm.solve(np.array([0.25, 0.0, 0.05]), 65.0, 0.0)
    line = [(np.array([0.15, 0.0, 0.05]), 65.0, 0.0)]
    wps = free.plan_tip_path(line, q_start=q0)
    assert 97.0 < max(math.degrees(w[3]) for w in wps) < 103.0
    refused(lambda: arm.plan_tip_path(line, q_start=q0), "IK failed", "wrist_flex=", "outside [20.0, 97.0]")
    inside = arm.plan_tip_path([(np.array([0.28, 0.02, 0.07]), 65.0, 0.0)], q_start=q0)
    assert max(outside_deg(arm, w) for w in inside) <= 1e-9
    # check_q: the pose itself, with the joint, its value and the limits in the text
    for q, text in ((q_low, "wrist_flex=17.7 deg outside limits [20.0, 97.0]"),
                    (q_roll, "wrist_roll=10.0 deg outside limits [-5.0, 5.0]"), (wps[-1], "wrist_flex=")):
        free.check_q(q)
        msg = refused(lambda: arm.check_q(q, what="test pose"), "test pose", text)
        refused(lambda: TL.check_pose(arm, arm.tool, q), text)
    # a path through such a pose is refused before anything moves; the motion commands say the same
    place(arm, q0)
    arm.torque_on()
    bus.sleep(0.6)
    q_now = arm.read_q()
    before, pos = writes(bus), dict(bus.pos)
    for call in (lambda: arm.check_path([q_now, q_low], 20.0), lambda: arm.run_path([q_now, q_low], 20.0),
                 lambda: arm.move_joints(q_roll, 20.0), lambda: arm.move_tip(xyz, 10.0, 0.0),
                 lambda: arm.move_tip_path(line, 15.0)):
        msg = refused(call)
        assert ("wrist_flex" in msg or "wrist_roll" in msg) and "outside" in msg, msg
    with tool_file(brush()):
        for argv, joint in ((["check", "0.32", "0.0", "0.15", "--pitch", "10", "--roll", "0"], "wrist_flex"),
                            (["tip", "0.32", "0.0", "0.15", "--pitch", "10", "--roll", "0"], "wrist_flex"),
                            (["tip", "0.25", "0.0", "0.05", "--pitch", "65", "--roll", "10"], "wrist_roll"),
                            (["joints", "wrist_flex=15"], "wrist_flex"), (["joints", "wrist_roll=-8"], "wrist_roll")):
            r, rc = fd_run(["--sim"] + argv, bus)
            text = json.dumps(r)
            assert rc == 1 and r["ok"] is False and joint in text and "outside" in text, (argv, r)
    assert writes(bus) == before and bus.pos == pos, "a refused pose moved the arm"
    # just inside the limit it goes
    r = arm.move_joints(np.array([*q_now[:3], math.radians(96.5), math.radians(4.5)]), 20.0)
    assert not r["aborted"] and abs(math.degrees(arm.read_q()[3]) - 96.5) < 0.5


def test_rest_pose_is_inside_the_tool_limits_or_refuses_cleanly():
    found, refusals = 0, 0
    sets = [BRUSH_LIMITS, {"wrist_flex": [20.0, 60.0]}, {"wrist_flex": [20.0, 40.0]}, {"wrist_flex": [-20.0, 30.0]},
            {"shoulder_pan": [-30.0, 60.0]}, {"shoulder_lift": [-110.0, 60.0], "elbow_flex": [-90.0, 90.0]},
            {"wrist_flex": [80.0, 97.0]}, {"wrist_flex": [60.0, 66.0]}, {"shoulder_lift": [-60.0, 60.0]}, {"elbow_flex": [-90.0, 60.0]},
            {"wrist_roll": [10.0, 20.0]}, {"shoulder_pan": [30.0, 60.0]}]
    for tip in (BRUSH_TIP, PEN_TIP, UNDER_TIP):
        for lims in sets:
            tool = brush(tip_offset=list(tip), joint_limits_deg=lims)
            arm, bus = sim_arm(tool=tool)
            try:
                q = TL.rest_q(arm, tool)
            except SafetyError as e:
                # a clean refusal: it says that the arm cannot rest (or that the pan asked for is not allowed), names
                # the limit that is in the way, and how to change it; and the arm did not move
                refusals += 1
                msg = str(e)
                joint = next(j for j in lims if j in msg)
                lo, hi = lims[joint]
                assert f"[{lo:g}, {hi:g}]" in msg or f"{lo:g}..{hi:g}" in msg, (lims, msg)
                if joint == "shoulder_pan":
                    assert "rest pan" in msg and "outside the pan limits" in msg, msg
                    q = TL.rest_q(arm, tool, pan_deg=45.0)            # ... and another base angle does it
                    assert outside_deg(arm, q) <= 1e-9 and abs(math.degrees(q[0]) - 45.0) < 1e-9
                else:
                    assert "cannot rest" in msg and f"fd tool set with --joint-limit {joint}=LO:HI" in msg and "fd tool detach" in msg, msg
                    # (round 3, T12: the range is the operator's to give, so that command stands in words; what is in
                    # backticks works as it is typed)
                    assert "`fd tool status`" in msg and "`fd tool set" not in msg, msg
                    assert "(None)" not in msg and TL.in_rest_pose(arm, tool)[0] is False
                    place(arm, TL.ready_q(arm, tool) if joint != "wrist_roll" else np.radians([0.0, -50.0, 20.0, 70.0, 12.0]))
                    arm.torque_on()
                    bus.sleep(0.6)
                    before, pos = writes(bus), dict(bus.pos)
                    refused(lambda: TL.rest(arm), "cannot rest")
                    with tool_file(tool):
                        r, rc = fd_run(["--sim", "tool", "rest"], bus)
                    assert rc == 1 and r["stage"] == "plan" and r["moved"] is False and "cannot rest" in r["reason"], r
                    assert writes(bus) == before and bus.pos == pos and arm.is_torqued()
                continue
            found += 1
            assert outside_deg(arm, q) <= 1e-9, (tip, lims, np.degrees(q))
            h = TL.check_pose(arm, tool, q, min_height=0.05 - 1e-6)
            # 5 cm up, or as little higher as the tool's wrist limit forces the gripper to be raised
            assert h < 0.0502 or abs(math.degrees(q[3]) - (lims["wrist_flex"][1] - math.degrees(0.005))) < 0.01, (tip, lims, h)
            assert TL.in_rest_pose(arm, tool, q) == (True, "") and np.allclose(arm.rest_q(), q)
            # the fold is the one the arm is known to rest in: shoulder and elbow on their stops, roll 0
            assert np.allclose(np.degrees(q)[[1, 2, 4]], [-100.09, 85.14, 0.0], atol=0.02), np.degrees(q)
    assert found == 3 * 6 and refusals == 3 * 6, (found, refusals)
    # with the brush's limits the rest pose is the pose without them (the limits do not touch the fold)
    arm, _ = sim_arm(tool=brush())
    assert np.allclose(TL.rest_q(arm, arm.tool), TL.rest_q(arm, brush(joint_limits_deg=None)))
    # a limit that takes the wrist's last degrees: the wrist is raised to the limit, the tip stands higher
    tool = brush(joint_limits_deg={"wrist_flex": [20.0, 40.0]})
    arm, _ = sim_arm(tool=tool)
    q = TL.rest_q(arm, tool)
    assert math.degrees(q[3]) <= 40.0 and TL.tip_height(arm, q, tool) > 0.05
    # the limits a pose is judged by are those of the tool that is ASKED about, not of the one the arm carries
    carrier, _ = sim_arm(tool=None)
    for lims in sets:
        tool = brush(joint_limits_deg=lims)
        mine, _ = sim_arm(tool=tool)
        for fn in (TL.rest_q, TL.ready_q, lambda a, t: TL.work_q(a, t, (0.25, 0.0), 0.02)[0]):
            try:
                want = fn(mine, tool)
            except SafetyError as e:
                assert refused(lambda: fn(carrier, tool)) == str(e), lims
            else:
                assert np.allclose(fn(carrier, tool), want), lims


def test_ready_with_limits_that_exclude_the_standby_pose_is_a_refusal_with_a_next_step():
    for lims, joint in (({"shoulder_pan": [30.0, 60.0]}, "shoulder_pan"), ({"wrist_roll": [10.0, 20.0]}, "wrist_roll"),
                        ({"wrist_flex": [-100.0, -20.0]}, "wrist_flex"), ({"shoulder_lift": [20.0, 90.0]}, "shoulder_lift")):
        tool = brush(joint_limits_deg=lims)
        with tool_file(tool):
            for start in ("resting", "holding"):
                arm, bus = sim_arm(tool=tool)
                q0 = np.radians([40.0, -100.0, 85.0, 42.0, 12.0 if joint == "wrist_roll" else 0.0])     # folded, as it was left
                place(arm, q0, gripper=2061)
                if start == "holding":
                    arm.torque_on()
                    bus.sleep(0.6)
                before, pos, torque = writes(bus), dict(bus.pos), dict(bus.torque)
                r, rc = fd_run(["--sim", "ready"], bus)
                assert rc == 1 and r["ok"] is False and r["error"] == "SafetyError", (lims, r)
                said = r.get("reason") or r["message"]
                nxt = r.get("next") or r["message"]                       # (fd gives the text; a "next" key when it has one)
                lo, hi = lims[joint]
                assert "stand-by" in said and joint in said and f"[{lo:g}, {hi:g}]" in said and "brush" in said, said
                assert f"{joint} {lo:g}..{hi:g} deg" in said and "tool's own joint limits" in said, said
                assert "fd tool set with --joint-limit" in nxt and "none (remove them all)" in nxt and "`fd tool rest`" in nxt, nxt
                assert "`fd tool status`" in nxt and "`fd tool set" not in nxt and "LO:HI`" not in nxt, nxt      # (round 3, T12)
                assert writes(bus) == before and bus.pos == pos and bus.torque == torque, f"{lims} {start}: something moved"
            arm, _ = sim_arm(tool=tool)
            refused(lambda: TL.ready_pose(arm, tool), "stand-by", joint)
            msg = refused(lambda: skills.go_ready(arm), "fd tool set and --joint-limit JOINT=LO:HI", "`--joint-limit none` removes them")
            assert "`fd tool set" not in msg, msg
    # limits that leave a stand-by pose at another pitch: the arm stands by there, inside them
    for lims in ({"wrist_flex": [20.0, 60.0]}, {"wrist_flex": [20.0, 97.0], "wrist_roll": [-5.0, 5.0]}, {"elbow_flex": [-100.0, 30.0]},
                 {"shoulder_pan": [-10.0, 60.0]}):
        tool = brush(joint_limits_deg=lims)
        with tool_file(tool):
            arm, bus = sim_arm(tool=tool)
            try:
                q_start = TL.rest_q(arm, tool)
            except SafetyError:                                # (limits that exclude the fold: from a pose over the table)
                q_start = TL.work_q(arm, tool, (0.25, 0.0), 0.06)[0]
            place(arm, q_start, gripper=2061)
            pose = TL.ready_pose(arm, tool)
            assert outside_deg(arm, pose["q"]) <= 1e-9, (lims, np.degrees(pose["q"]))
            r, rc = fd_run(["--sim", "ready"], bus)
            assert rc == 0 and r["ok"] and r["standby_xyz"] == pose["xyz"] and r["standby_pitch_deg"] == pose["pitch"], (lims, r)
            assert max(outside_tool_deg(tool, q_of(p)) for p in bus.trace) < 0.4, lims
            assert np.linalg.norm(K.fk(q_of(bus.pos), K.tool_offset(*BRUSH_TIP))[:3, 3] - np.array(pose["xyz"])) < 0.004
    # a tip offset that puts the point out of reach is still told as that
    pole = TL.Tool("pole", TL.offset_for(CFG, "along_fingers", 0.25), "along_fingers", 75.0, joint_limits_deg=dict(BRUSH_LIMITS))
    arm, _ = sim_arm(tool=pole)
    assert "Check the tip offset" in refused(lambda: TL.ready_pose(arm, pole), "stand-by")


def test_free_moves_keep_every_waypoint_inside_the_tool_limits():
    tool = brush()
    arm0, _ = sim_arm(tool=tool)
    rest, ready = TL.rest_q(arm0, tool), TL.ready_q(arm0, tool)
    starts = {"rest": rest, "ready": ready, "aside": np.radians([43.0, -20.0, 10.0, 63.0, 0.0]),
              "photo pose": np.radians([-89.2, -61.7, 36.2, 85.6, 0.0])}
    for (x, y) in ((0.20, -0.08), (0.25, 0.0), (0.30, 0.08), (0.17, 0.02), (0.33, -0.05)):
        for h in (0.002, 0.012, 0.03, 0.065):
            starts[f"({x}, {y}) h={h}"] = TL.work_q(arm0, tool, (x, y), h)[0]
    assert all(outside_deg(arm0, q) <= 1e-9 for q in starts.values())
    plans = lifted = 0
    vias = set()
    for label, q in starts.items():
        for goal, q_to in (("ready", ready), ("rest", rest), ("aside", starts["aside"])):
            if np.allclose(q, q_to):
                continue
            plan = TL.plan_free_move(arm0, tool, q, q_to)
            plans += 1
            lifted += bool(plan["lift"])
            vias.add(plan["via"])
            # every waypoint, and every 1.5 deg between two of them, inside the limits
            for chain in ([plan["lift"]] if plan["lift"] else []) + [plan["path"]]:
                for a, b in zip(chain, chain[1:]):
                    n = max(1, int(math.ceil(math.degrees(float(np.max(np.abs(b - a)))) / 1.5)))
                    worst = max(outside_deg(arm0, a + (b - a) * k / n) for k in range(n + 1))
                    assert worst <= 1e-9, (label, goal, plan["via"], worst)
            assert np.allclose(plan["path"][-1], q_to) and np.allclose((plan["lift"] or plan["path"])[0], q)
    assert plans >= 60 and lifted >= 20 and "direct" in vias, (plans, lifted, vias)
    # ... and so does the arm when it runs them: every pose it passes through (0.4 deg for the tracking)
    ran = 0
    for label in ("rest", "photo pose", "(0.2, -0.08) h=0.002", "(0.3, 0.08) h=0.012", "(0.17, 0.02) h=0.002", "(0.33, -0.05) h=0.03"):
        for q_to in (ready, rest):
            arm, bus = sim_arm(tool=tool, q=starts[label], torque=True)
            h0 = TL.tip_height(arm, arm.read_q(), tool)
            r = TL.free_move(arm, q_to, 25.0)
            ran += 1
            worst = max(outside_tool_deg(tool, q_of(p)) for p in bus.trace)
            assert worst < 0.4 and max(outside_deg(arm, q_of(p)) for p in bus.trace) < 1.2, (label, r["via"], worst)
            hs = heights(arm, bus, tool)
            k = next(i for i, h in enumerate(hs) if h >= TL.MIN_FREE_HEIGHT + TL.PLAN_MARGIN)
            assert min(hs[:k + 1]) >= h0 - 0.0015 and min(hs[k:]) >= TL.MIN_FREE_HEIGHT, (label, min(hs), h0)
            assert float(np.max(np.abs(arm.read_q() - q_to))) < math.radians(1.5), label
            assert r["lifted_first"] == (h0 < TL.LIFT_HEIGHT - 0.002) and ("lift_via" in r) == r["lifted_first"], (label, r)
    assert ran == 12
    # something in the direct way (a 24 mm cube around the tip's direct path): the detour keeps to the limits as well
    cfg = load_config()
    mid = K.fk(rest + (ready - rest) * 0.5, K.tool_offset(*BRUSH_TIP))[:3, 3]
    cfg["keepouts"]["post"] = [v for c in mid for v in (float(c) - 0.012, float(c) + 0.012)]
    arm, bus = sim_arm(tool=tool, cfg=cfg)
    place(arm, rest)
    plan = TL.plan_free_move(arm, tool, arm.read_q(), ready)
    assert plan["via"] != "direct" and len(plan["path"]) >= 3, plan["via"]
    for a, b in zip(plan["path"], plan["path"][1:]):
        n = max(1, int(math.ceil(math.degrees(float(np.max(np.abs(b - a)))) / 1.5)))
        assert max(outside_tool_deg(tool, a + (b - a) * k / n) for k in range(n + 1)) <= 1e-9, plan["via"]
    arm.torque_on()
    bus.sleep(0.6)
    bus.trace.clear()
    r = TL.free_move(arm, ready, 25.0)
    assert r["via"] == plan["via"] and max(outside_tool_deg(tool, q_of(p)) for p in bus.trace) < 0.4
    assert max(outside_deg(arm, q_of(p)) for p in bus.trace) < 1.2


def test_a_start_outside_a_tool_limit_may_come_back_inside_and_not_go_further_out():
    """2026-11-13 afternoon: the arm had been moved by hand, wrist_roll stood at +15.6 deg under a limit of +-5.
    The lift refused (no IK solution keeps a roll the limits exclude) and the brush was raised by the elbow
    joint from a scratch script."""
    tool = brush()
    arm0, _ = sim_arm(tool=tool)
    cases = 0
    for xy, h in (((0.25, 0.0), 0.004), ((0.30, -0.10), 0.010), ((0.18, 0.05), 0.020)):
        q_in = TL.work_q(arm0, tool, xy, h)[0]
        for roll in (15.6, -12.0):
            cases += 1
            q0 = q_in.copy()
            q0[4] = math.radians(roll)
            assert outside_deg(arm0, q0) > 6.9
            # the checks: back inside is allowed, staying is allowed, further out is not; without the start nothing is
            back, further = q0.copy(), q0.copy()
            back[4], further[4] = math.radians(roll * 0.5), math.radians(roll * 1.1)
            arm0.check_q(back, q_start=q0)
            arm0.check_q(q0, q_start=q0)
            refused(lambda: arm0.check_q(further, q_start=q0), "wrist_roll", "outside limits")
            refused(lambda: arm0.check_q(back), "wrist_roll")
            refused(lambda: arm0.check_q(q0), "wrist_roll")
            # the lift works from there: straight up, the roll staying where the hand left it
            arm, bus = sim_arm(tool=tool, q=q0, torque=True)
            p0 = K.fk(arm.read_q(), arm.tip_tool())[:3, 3]
            h0 = TL.tip_height(arm, arm.read_q(), tool)
            r = TL.lift(arm)
            p1 = K.fk(arm.read_q(), arm.tip_tool())[:3, 3]
            assert r["lifted"] and r["via"] == "straight up" and abs(r["tip_height_m"] - TL.LIFT_HEIGHT) < 0.003, r
            assert np.linalg.norm(p1[:2] - p0[:2]) < 0.004, (p0, p1)
            rolls = [math.degrees(q_of(p)[4]) for p in bus.trace]
            assert max(abs(v - roll) for v in rolls) < 0.2, (roll, min(rolls), max(rolls))
            hs = heights(arm, bus, tool)
            assert min(hs) >= h0 - 0.0015 and all(b >= a - 0.0015 for a, b in zip(hs, hs[1:]))
            # and the free move after it brings the roll back inside, never further out
            for q_to, name in ((TL.ready_q(arm, tool), "ready"), (TL.rest_q(arm, tool), "rest")):
                arm, bus = sim_arm(tool=tool, q=q0, torque=True)
                r = skills.go_ready(arm) if name == "ready" else TL.rest(arm)
                rolls = [math.degrees(q_of(p)[4]) for p in bus.trace]
                assert max(abs(v) for v in rolls) <= abs(roll) + 0.2, (name, roll, max(rolls), min(rolls))       # never further out
                assert all(v * roll >= -0.5 * abs(roll) for v in rolls) and abs(rolls[-1]) < 0.5, (name, rolls[-1])
                for i, j in enumerate(ARM[:4]):                       # the joints that were inside stay inside
                    lo, hi = arm.q_limits()[j]
                    vals = [q_of(p)[i] for p in bus.trace]
                    assert min(vals) >= lo - math.radians(0.4) and max(vals) <= hi + math.radians(0.4), (name, j)
                assert (r.get("lifted_first") if name == "ready" else r["ok"]), r
    assert cases == 6
    # by hand: `fd joints` may turn the roll back, not further
    with tool_file(tool):
        q0 = TL.work_q(arm0, tool, (0.25, 0.0), 0.03)[0]
        q0[4] = math.radians(15.6)
        arm, bus = sim_arm(tool=tool, q=q0, torque=True)
        before = writes(bus)
        r, rc = fd_run(["--sim", "joints", "wrist_roll=20"], bus)
        assert rc == 1 and "wrist_roll" in r["message"] and "outside limits" in r["message"] and writes(bus) == before, r
        r, rc = fd_run(["--sim", "joints", "wrist_roll=4.5"], bus)
        assert rc == 0 and abs(math.degrees(q_of(bus.pos)[4]) - 4.5) < 0.3, r
        r, rc = fd_run(["--sim", "joints", "wrist_roll=8"], bus)        # inside now: the limit holds again
        assert rc == 1 and "wrist_roll" in r["message"], r
    # the wrist bent past its limit (101 deg under a limit of 97), where the tip cannot go straight up: one joint alone
    # raises it, as the elbow did that afternoon. Where that arc carries the tip 4 cm sideways or less the lift takes
    # it; the wrist is never bent further
    for xy, h in (((0.30, -0.10), 0.010), ((0.28, 0.05), 0.012)):
        q0 = TL.work_q(arm0, tool, xy, h)[0]
        q0[3] = math.radians(101.0)
        arm, bus = sim_arm(tool=tool, q=q0, torque=True)
        h0 = TL.tip_height(arm, arm.read_q(), tool)
        p0 = K.fk(arm.read_q(), arm.tip_tool())[:3, 3]
        assert h0 < TL.LIFT_HEIGHT - 0.01
        r = TL.lift(arm)
        p1 = K.fk(arm.read_q(), arm.tip_tool())[:3, 3]
        assert r["lifted"] and r["via"].startswith("shoulder_lift alone") and "sideways" in r["via"], r
        assert abs(r["tip_height_m"] - TL.LIFT_HEIGHT) < 0.004 and np.linalg.norm(p1[:2] - p0[:2]) <= TL.LIFT_ARC_MAX_M, (r, p0, p1)
        hs = heights(arm, bus, tool)
        assert min(hs) >= h0 - 0.0015 and max(math.degrees(q_of(p)[3]) for p in bus.trace) <= 101.2
        moved = np.degrees(np.abs(q_of(bus.pos) - q0))
        assert moved[1] > 5.0 and max(moved[[0, 3, 4]]) < 0.3 and moved[2] < 1.3, moved       # (1 deg: the simulated elbow's droop)
        r = skills.go_ready(arm)
        wr = [math.degrees(q_of(p)[3]) for p in bus.trace]
        assert max(wr) <= 101.2 and wr[-1] <= 97.0, (max(wr), wr[-1])
    # ... and where the arc is longer (7 cm here), it is too much to be taken unasked: the refusal says exactly which
    # joint move raises the tip, and which brings the wrist back inside; the move it names works
    for xy, h in (((0.25, 0.0), 0.004), ((0.22, 0.06), 0.006)):
        q0 = TL.work_q(arm0, tool, xy, h)[0]
        q0[3] = math.radians(101.0)
        arm, bus = sim_arm(tool=tool, q=q0, torque=True)
        h0 = TL.tip_height(arm, arm.read_q(), tool)
        before, pos = writes(bus), dict(bus.pos)
        msg = refused(lambda: TL.lift(arm), "cannot be lifted straight up", "Nothing moved", "One joint alone would raise it to 35 mm",
                      "sideways", "wrist_flex stands at 101.0 deg, outside the limit [20, 97] deg of tool 'brush'", "not further out")
        # (the wrist alone may turn back where that keeps the tip off the table; where not, the tip has to come up first)
        assert ("`fd joints wrist_flex=96.5` turns it back inside (checked" in msg) == (xy == (0.25, 0.0)), msg
        assert ("raise the tip first" in msg and "below floor limit" in msg) == (xy != (0.25, 0.0)), msg
        assert writes(bus) == before and bus.pos == pos
        cmd = re.search(r"`fd joints (elbow_flex=[-+][0-9.]+) --rel`", msg)
        side = re.search(r"carries the tip (\d+) mm sideways", msg)
        assert cmd and side and int(side.group(1)) > TL.LIFT_ARC_MAX_M * 1000, msg
        with tool_file(tool):
            r, rc = fd_run(["--sim", "joints", cmd.group(1), "--rel", "--speed", "12"], bus)
        assert rc == 0, r
        hs = heights(arm, bus, tool)
        assert abs(TL.tip_height(arm, arm.read_q(), tool) - TL.LIFT_HEIGHT) < 0.004 and min(hs) >= h0 - 0.0015, (hs[-1], min(hs))
        assert max(math.degrees(q_of(p)[3]) for p in bus.trace) <= 101.2
        moved = np.degrees(np.abs(q_of(bus.pos) - q0))
        assert moved[2] > 5.0 and max(moved[[0, 1, 3, 4]]) < 1.3, moved          # (1 deg: the simulated shoulder's droop)
        r = skills.go_ready(arm)
        wr = [math.degrees(q_of(p)[3]) for p in bus.trace]
        assert max(wr) <= 101.2 and wr[-1] <= 97.0 and r["tool"] == "brush", (max(wr), wr[-1])
    # when nothing lifts it: the refusal names the joint that is outside and the move that frees it, and that move works
    cfg = load_config()
    q0 = TL.work_q(arm0, tool, (0.25, 0.0), 0.012)[0]
    q0[4] = math.radians(15.6)
    tip0 = K.fk(q0, K.tool_offset(*BRUSH_TIP))[:3, 3]
    cfg["keepouts"]["shelf"] = [0.0, 0.5, -0.3, 0.3, float(tip0[2]) + 0.006, float(tip0[2]) + 0.012]      # a slab just over the tip
    arm, bus = sim_arm(tool=tool, q=q0, torque=True, cfg=cfg)
    before, pos = writes(bus), dict(bus.pos)
    msg = refused(lambda: TL.lift(arm), "cannot be lifted", "shelf", "Nothing moved", "wrist_roll stands at 15.6 deg",
                  "outside the limit [-5, 5] deg of tool 'brush'", "not further out", "`fd joints wrist_roll=4.5`", "checked")
    assert refused(lambda: TL.plan_free_move(arm, tool, arm.read_q(), TL.ready_q(arm, tool))) == msg
    assert writes(bus) == before and bus.pos == pos
    with tool_file(tool):
        r, rc = fd_run(["--sim", "joints", "wrist_roll=4.5"], bus, cfg=cfg)
    assert rc == 0 and abs(math.degrees(q_of(bus.pos)[4]) - 4.5) < 0.3, r
    assert TL.tip_height(arm, arm.read_q(), tool) > 0.0, "the freeing move put the tip into the table"
    # a pose inside the limits keeps the old advice
    q0 = TL.work_q(arm0, tool, (0.25, 0.0), 0.012)[0]
    arm, bus = sim_arm(tool=tool, q=q0, torque=True, cfg=cfg)
    msg = refused(lambda: TL.lift(arm), "cannot be lifted", "fd tip 0 0 0.01 --rel", "fd torque off --force")
    assert "fd joints" not in msg


def test_tools_with_the_tip_beside_and_under_the_fingers_stay_above_the_table():
    """The thin brush of 2026-11-13: taped to the opened moving jaw its tip stood 9 cm beside the finger line
    (mount "other"); with the jaw closed it lay under the fingers. Rest pose, lifts and free moves must keep
    the tip of either above the table."""
    for name, tip in (("beside", SIDE_TIP), ("under", UNDER_TIP)):
        for lims in (None, BRUSH_LIMITS):
            tool = TL.Tool(name=name, tip_offset=list(tip), mount="other", pitch_deg=65.0, joint_limits_deg=lims)
            arm0, _ = sim_arm(tool=tool)
            # the rest pose: tip 5 cm up, nothing of the tool under the floor line, inside the limits, the fold as ever
            q_rest = TL.rest_q(arm0, tool)
            assert abs(TL.tip_height(arm0, q_rest, tool) - 0.05) < 2e-4 and outside_deg(arm0, q_rest) <= 1e-9, (name, lims)
            assert float(TL.body_points(arm0, q_rest, tool)[:, 2].min()) > TABLE_Z + 0.02
            assert np.allclose(np.degrees(q_rest)[[1, 2, 4]], [-100.09, 85.14, 0.0], atol=0.02)
            pose = TL.ready_pose(arm0, tool)
            assert TL.tip_height(arm0, pose["q"], tool) > 0.11 and outside_deg(arm0, pose["q"]) <= 1e-9
            with tool_file(tool):
                # stand-by -> rest -> stand-by through the commands: the tip never under 2 cm, the torque off at rest
                arm, bus = sim_arm(tool=tool)
                place(arm, q_rest, gripper=2576)
                r, rc = fd_run(["--sim", "ready"], bus)
                assert rc == 0 and r["ok"] and np.allclose(r["tool_tip_xyz"], pose["xyz"], atol=0.004), (name, r)
                r, rc = fd_run(["--sim", "tool", "rest"], bus)
                assert rc == 0 and r["ok"] and r["torque"] == "off" and 40.0 <= r["tip_height_mm"] <= 60.0, (name, r)
                assert min(heights(arm, bus, tool)) >= TL.MIN_FREE_HEIGHT, (name, min(heights(arm, bus, tool)))
                assert all(v == 2576 for v in gripper_goals(bus)), "the gripper got a goal"
            # low over the paper at several places: lifted without going down, then to the stand-by pose and to rest
            done = 0
            for xy in ((0.25, 0.0), (0.20, -0.08), (0.30, 0.08), (0.22, 0.10), (0.17, 0.0), (0.33, 0.0)):
                for h in (0.003, 0.012):
                    try:
                        q_low = TL.work_q(arm0, tool, xy, h)[0]
                        TL.check_pose(arm0, tool, q_low, what="work pose")
                    except SafetyError:
                        continue                             # (not a place this tool can work at)
                    done += 1
                    arm, bus = sim_arm(tool=tool, q=q_low, torque=True)
                    h0 = TL.tip_height(arm, arm.read_q(), tool)
                    r = TL.lift(arm)
                    hs = heights(arm, bus, tool)
                    assert r["lifted"] and abs(r["tip_height_m"] - TL.LIFT_HEIGHT) < 0.004, (name, xy, h, r)
                    assert min(hs) >= h0 - 0.0015, (name, xy, h, r["via"], min(hs), h0)
                    assert max(outside_tool_deg(tool, q_of(p)) for p in bus.trace) < 0.4, (name, xy, r["via"])
                    for goal in ("ready", "rest"):
                        arm, bus = sim_arm(tool=tool, q=q_low, torque=True)
                        h0 = TL.tip_height(arm, arm.read_q(), tool)
                        r = skills.go_ready(arm) if goal == "ready" else TL.rest(arm)
                        hs = heights(arm, bus, tool)
                        k = next(i for i, v in enumerate(hs) if v >= TL.MIN_FREE_HEIGHT + TL.PLAN_MARGIN)
                        assert min(hs[:k + 1]) >= h0 - 0.0015 and min(hs[k:]) >= TL.MIN_FREE_HEIGHT, (name, xy, h, goal, min(hs))
                        assert (r["lifted_first"] if goal == "ready" else r["ok"]), (name, xy, goal, r)
                        # the fingers and the tool's body stay off the table too
                        low = min(float(TL.body_points(arm, q_of(p), tool)[:, 2].min()) for p in bus.trace[k:])
                        assert low > TABLE_Z + 0.004, (name, xy, goal, low)
            assert done >= 6, (name, lims, done)
    # where the tip cannot rise straight up (the tool beside the fingers, close to the base, the wrist at its
    # limit), the lift is done by one joint, and says so
    tool = TL.Tool(name="beside", tip_offset=list(SIDE_TIP), mount="other", pitch_deg=65.0, joint_limits_deg=dict(BRUSH_LIMITS))
    arm0, _ = sim_arm(tool=tool)
    q_low = TL.work_q(arm0, tool, (0.20, -0.08), 0.012)[0]
    refused(lambda: arm0.plan_tip_path([(K.fk(q_low, K.tool_offset(*SIDE_TIP))[:3, 3] + [0.0, 0.0, 0.025], None, None)], q_start=q_low),
            "wrist_flex", "outside [20.0, 97.0]")
    wps, via = TL._lift_plan(arm0, tool, q_low)
    assert via.endswith("sideways)") and ("elbow_flex alone" in via or "shoulder_lift alone" in via), via
    assert int(re.search(r"moves (\d+) mm sideways", via).group(1)) <= TL.LIFT_ARC_MAX_M * 1000, via
    assert TL.lift_path(arm0, tool, q_low)[-1].tolist() == wps[-1].tolist()
    arm, bus = sim_arm(tool=tool, q=q_low, torque=True)
    p0 = K.fk(arm.read_q(), arm.tip_tool())[:3, 3]
    r = TL.lift(arm)
    p1 = K.fk(arm.read_q(), arm.tip_tool())[:3, 3]
    assert r["via"].split(" (")[0] == via.split(" (")[0] and np.linalg.norm(p1[:2] - p0[:2]) <= TL.LIFT_ARC_MAX_M, (r, p0, p1)


def test_fd_tool_attach_set_status_store_remove_and_show_joint_limits():
    with tool_file(None):
        bus = RecBus()
        bus.pos["gripper"] = bus.goal["gripper"] = 2054.0
        r, rc = fd_run(["--sim", "tool", "attach", "--name", "brush", "--tip", "0.0085,0.0007,0.0611", "--pitch", "65",
                        "--joint-limit", "wrist_flex=20:97", "--joint-limit", "wrist_roll=-5:5", "--dry"], bus)
        assert rc == 0 and r["stage"] == "planned" and r["tool"]["joint_limits_deg"] == BRUSH_LIMITS and not os.path.exists(TOOL_PATH), r
        r, rc = fd_run(["--sim", "tool", "attach", "--name", "brush", "--tip", "0.0085,0.0007,0.0611", "--pitch", "65",
                        "--joint-limit", "wrist_flex=20:97", "--joint-limit", "wrist_roll=-5:5"], bus)
        assert rc == 0 and r["stage"] == "attached" and r["tool"]["joint_limits_deg"] == BRUSH_LIMITS, r
        assert json.load(open(TOOL_PATH))["joint_limits_deg"] == BRUSH_LIMITS and TL.active().joint_limits_deg == BRUSH_LIMITS
        # what attach shows of the arm is computed WITH the limits (the stand-by pose leans the brush back to keep the wrist under 97)
        arm, _ = sim_arm(tool=TL.active())
        assert r["arm"]["standby"]["q_deg"] == [round(v, 1) for v in np.degrees(TL.ready_q(arm, arm.tool))], r["arm"]
        assert r["arm"]["standby"]["q_deg"][3] <= 97.0 and r["arm"]["rest"]["tip_height_mm"] == 50.0, r["arm"]
        r, rc = fd_run(["--sim", "tool", "status"])
        assert rc == 0 and r["attached"] and r["tool"]["joint_limits_deg"] == BRUSH_LIMITS, r
        # a measured contact plane and map: changing ONLY the limits keeps them
        t = TL.active()
        t.contact = {"x0": 0.24, "y0": 0.0, "a": -0.0015, "bx": 0.0, "by": 0.0, "rms_mm": 0.2, "points": []}
        t.corr_map = {"centre": [0.22, 0.0], "kind": "affine", "coef": [[0, 0], [1, 0], [0, 1]], "rms_mm": 1.0, "max_mm": 2.0, "n": 9}
        t.calibrated = {"tip": {"at": "x"}, "contact": {"at": "x"}, "map": {"at": "x"}}
        t.region = [0.17, -0.06, 0.31, 0.07]
        TL.save(t)
        kept = {k: json.load(open(TOOL_PATH))[k] for k in ("contact", "corr_map", "calibrated", "region", "tip_offset", "pitch_deg",
                                                             "gripper_raw", "attached_at")}

        def same_but_limits(limits):
            d = json.load(open(TOOL_PATH))
            assert {k: d[k] for k in kept} == kept and d["joint_limits_deg"] == limits, d

        r, rc = fd_run(["--sim", "tool", "set", "--joint-limit", "wrist_flex=25:90", "--dry"], bus)
        assert rc == 0 and r["changed"] == ["joint_limits_deg"] and r["saved"] is False and r["calibration_dropped"] is False, r
        same_but_limits(BRUSH_LIMITS)
        r, rc = fd_run(["--sim", "tool", "set", "--joint-limit", "wrist_flex=25:90"], bus)
        assert rc == 0 and r["changed"] == ["joint_limits_deg"] and r["saved"] and r["calibration_dropped"] is False, r
        assert r["tool"]["joint_limits_deg"] == dict(BRUSH_LIMITS, wrist_flex=[25.0, 90.0]) and r["before"]["joint_limits_deg"] == BRUSH_LIMITS, r
        assert r["tool"]["calibrated"] == {"tip": True, "contact": True, "map": True} and "fd tool status" in r["next"], r
        same_but_limits(dict(BRUSH_LIMITS, wrist_flex=[25.0, 90.0]))     # (the joint that is named changes, wrist_roll keeps its limit)
        r, rc = fd_run(["--sim", "tool", "set", "--joint-limit", "wrist_flex=25:90"], bus)
        assert rc == 0 and r["changed"] == [] and r["saved"] is False, r
        r, rc = fd_run(["--sim", "tool", "set", "--joint-limit", "wrist_flex=20:97", "--joint-limit", "wrist_roll=-5:5",
                        "--joint-limit", "shoulder_pan=-95:95", "--note", "handle taped lower"], bus)
        assert rc == 0 and r["changed"] == ["joint_limits_deg", "notes"] and not r["calibration_dropped"], r
        same_but_limits(dict(BRUSH_LIMITS, shoulder_pan=[-95.0, 95.0]))
        r, rc = fd_run(["--sim", "tool", "status"])
        assert r["tool"]["joint_limits_deg"] == dict(BRUSH_LIMITS, shoulder_pan=[-95.0, 95.0]) and "calibrated" in r["next"], r
        # none: removed from the file's limits and from what status shows; the calibration is still there
        r, rc = fd_run(["--sim", "tool", "set", "--joint-limit", "none"], bus)
        assert rc == 0 and r["changed"] == ["joint_limits_deg"] and "joint_limits_deg" not in r["tool"], r
        assert not r["calibration_dropped"], r
        same_but_limits(None)
        bare = Arm(bus=SimBus(), cfg=load_config(), tool=None).open()
        assert TL.active().joint_limits_deg is None and fd.make_arm(True).open().q_limits() == bare.q_limits()
        r, rc = fd_run(["--sim", "tool", "status"])
        assert rc == 0 and "joint_limits_deg" not in r["tool"], r
        assert r["tool"]["calibrated"] == {"tip": True, "contact": True, "map": True}, r
        r, rc = fd_run(["--sim", "tool", "set", "--joint-limit", "none"], bus)
        assert rc == 0 and r["changed"] == [], r
        # a new tip together with new limits: the tip is what drops the calibration
        r, rc = fd_run(["--sim", "tool", "set", "--tip", "0.009,0.001,0.062", "--joint-limit", "wrist_flex=20:97"], bus)
        assert rc == 0 and r["changed"] == ["tip_offset", "joint_limits_deg"] and r["calibration_dropped"], r
        assert TL.active().contact is None and TL.active().joint_limits_deg == {"wrist_flex": [20.0, 97.0]}
        assert writes(bus) == (0, 0), "attach / set / status only look at the arm"
        # the arm of the next command moves inside them
        r, rc = fd_run(["--sim", "ready"])
        assert rc == 0 and r["q_deg"]["wrist_flex"] <= 97.3, r


# ----------------------------------------------------------------------------- simulation start, real files
def test_simulated_arm_starts_in_the_tool_rest_pose():
    with tool_file(None):
        arm = fd.make_arm(True).open()
        assert {j: round(arm.bus.pos[j]) for j in JOINTS} == REST_RAW
    with tool_file(pen(gripper_raw=2061)):
        arm = fd.make_arm(True).open()
        q = arm.read_q()
        assert np.allclose(q, TL.rest_q(arm, arm.tool), atol=math.radians(0.1)), np.degrees(q)
        assert abs(TL.tip_height(arm, q, arm.tool) - 0.05) < 0.002 and round(arm.bus.pos["gripper"]) == 2061
        assert arm.bus.io["writes"] == 0
        # the whole morning round on the command line, each call a fresh simulated arm
        r, rc = fd_run(["--sim", "ready"])
        assert rc == 0 and r["tool"] == "pen2" and r["min_tip_height_m"] >= TL.MIN_FREE_HEIGHT, r
        r, rc = fd_run(["--sim", "tool", "rest", "--dry"])
        assert rc == 0 and r["ok"], r
        r, rc = fd_run(["--sim", "park"])
        assert rc == 1 and r["stage"] == "tool", r


# ----------------------------------------------------------------------------- review of 2026-11-13 (added tests)
def test_a_simulated_copy_follows_the_real_tool_until_the_real_tool_changes():
    """Review finding 32. A simulated command that changes the tool the simulation mirrors (`fd --sim tool set`)
    stores a copy of the real tool as the simulated tool. That copy used to be the simulated tool for good: the
    real tool could be changed or replaced, `fd --sim` went on with the old one. Now the copy is marked as one and
    is read only while the real tool file is what it was copied from; a tool attached IN simulation is no copy and
    stays until `fd --sim tool detach`."""
    real, simf = os.path.join(_TMP, "copy_real_tool.json"), os.path.join(_TMP, "copy_sim_tool.json")
    saved = (TL.TOOL_FILE, TL.SIM_TOOL_FILE, os.environ.pop("FD_TOOL"))
    TL.TOOL_FILE, TL.SIM_TOOL_FILE = real, simf
    try:
        TL.save(pen(name="real_pen", attached_at="2026-11-12T03:00:00"))
        real_text = open(real).read()
        assert TL.sim_tool_state() == {"simulated_tool": False, "file": real, "copy_of_real": False, "dropped": None, "note": None}
        # a simulated change of the mirrored tool: a copy, marked with what it is a copy of; the real file is as it was
        r, rc = fd_run(["--sim", "tool", "set", "--pitch", "70"])
        assert rc == 0 and TL.active(sim=True).pitch_deg == 70.0 and TL.active().pitch_deg == 75.0 and open(real).read() == real_text, r
        mark = json.load(open(simf))[TL.SIM_COPY_KEY]
        assert (mark["file"], mark["name"], mark["attached_at"]) == (real, "real_pen", "2026-11-12T03:00:00") and mark["sha1"], mark
        st = TL.sim_tool_state()
        assert st["simulated_tool"] and st["copy_of_real"] and st["file"] == simf and st["dropped"] is None, st
        assert "copy of the real tool" in st["note"] and "fd --sim tool detach" in st["note"], st["note"]
        assert fd.make_arm(True).tool.pitch_deg == 70.0
        # renamed in simulation: still a copy of the real pen (the mark says of what, not the copy's own name)
        r, rc = fd_run(["--sim", "tool", "set", "--name", "renamed"])
        assert rc == 0 and TL.active(sim=True).name == "renamed" and json.load(open(simf))[TL.SIM_COPY_KEY] == mark, r
        # the real tool file is written again with the same content: nothing changed, the copy holds
        TL.save(TL.active())
        assert open(real).read() == real_text and TL.active(sim=True).name == "renamed"
        # the real tool CHANGES: the copy is left behind (not read, not deleted), the simulation mirrors the real tool
        t = TL.active()
        t.notes.append("the tape was renewed")
        TL.save(t)
        got = TL.active(sim=True)
        assert (got.name, got.pitch_deg, got.notes) == ("real_pen", 75.0, ["the tape was renewed"]) and TL.source(sim=True) == real
        assert os.path.exists(simf) and fd.make_arm(True).tool.name == "real_pen"
        st = TL.sim_tool_state()
        assert not st["simulated_tool"] and st["file"] == real and st["dropped"]["file"] == simf, st
        assert "copy of the real tool 'real_pen'" in st["dropped"]["why"] and "mirrors the real tool again" in st["note"], st
        r, rc = fd_run(["--sim", "tool", "status"])
        assert rc == 0 and r["tool"]["name"] == "real_pen" and r["tool"]["pitch_deg"] == 75.0 and r["file"] == real, r
        # a new simulated change: a new copy, of the real tool as it is now
        r, rc = fd_run(["--sim", "tool", "set", "--pitch", "68"])
        now = json.load(open(simf))
        assert rc == 0 and now["pitch_deg"] == 68.0 and now["notes"] == ["the tape was renewed"] and now[TL.SIM_COPY_KEY] != mark, now
        assert TL.sim_tool_state()["dropped"] is None and TL.active(sim=True).pitch_deg == 68.0
        # the real tool is registered anew (another name, another attach time): the copy of the old one is left behind
        TL.save(pen(name="pen3", attached_at="2026-11-14T09:00:00"))
        assert TL.active(sim=True).name == "pen3" and "the real tool is now 'pen3'" in TL.sim_tool_state()["dropped"]["why"]
        # ... and when the real tool is taken off, the simulation has no tool either (it mirrors the real arm)
        TL.clear()
        assert TL.active(sim=True) is None and TL.source(sim=True) is None
        assert "no tool any more" in TL.sim_tool_state()["dropped"]["why"]
        # a tool attached IN SIMULATION is no copy: it stays, whatever the real arm carries
        r, rc = fd_run(["--sim", "tool", "detach"])                       # (removes what was left behind)
        assert rc == 0 and not os.path.exists(simf), r
        TL.save(pen(name="real_pen", attached_at="2026-11-12T03:00:00"))
        r, rc = fd_run(["--sim", "tool", "attach", "--name", "sim_pen", "--along-fingers", "0.10", "--no-robot"])
        assert rc == 0 and TL.SIM_COPY_KEY not in json.load(open(simf)), r
        for change in (lambda: TL.save(pen(name="real_pen", attached_at="2026-11-12T03:00:00", notes=["changed"])),
                       lambda: TL.save(pen(name="pen4", attached_at="2026-11-14T10:00:00")), TL.clear):
            change()
            assert TL.active(sim=True).name == "sim_pen" and TL.source(sim=True) == simf
            st = TL.sim_tool_state()
            assert st["simulated_tool"] and not st["copy_of_real"] and st["dropped"] is None and "tool of its own" in st["note"], st
        # a copy made before the mark existed (same name and attach time as the real tool, no mark): read while it is
        # the newer file, left behind once the real tool file is newer
        old_copy = pen(name="real_pen", attached_at="2026-11-12T03:00:00", pitch_deg=60.0).to_dict()
        TL.save(pen(name="real_pen", attached_at="2026-11-12T03:00:00"))
        with open(simf, "w") as f:
            json.dump(old_copy, f)
        os.utime(simf, ns=(os.stat(real).st_mtime_ns + 10 ** 9,) * 2)
        assert TL.active(sim=True).pitch_deg == 60.0
        os.utime(simf, ns=(os.stat(real).st_mtime_ns - 10 ** 9,) * 2)
        assert TL.active(sim=True).pitch_deg == 75.0 and "older than the real tool file" in TL.sim_tool_state()["dropped"]["why"]
        # a real tool file never carries the mark, whatever tool object is stored into it
        TL.save(pen(name="real_pen", attached_at="2026-11-12T03:00:00"))
        r, rc = fd_run(["--sim", "tool", "set", "--pitch", "66"])
        copy = TL.active(sim=True)
        assert rc == 0 and TL.SIM_COPY_KEY in copy.extra
        TL.save(copy)
        assert TL.SIM_COPY_KEY not in json.load(open(real)) and TL.active().pitch_deg == 66.0
    finally:
        TL.TOOL_FILE, TL.SIM_TOOL_FILE = saved[0], saved[1]
        os.environ["FD_TOOL"] = saved[2]
        for p in (real, simf):
            if os.path.exists(p):
                os.remove(p)
    # with $FD_TOOL there is one file for both: a simulated save marks nothing
    with tool_file(pen(attached_at="2026-11-12T03:00:00")):
        r, rc = fd_run(["--sim", "tool", "set", "--pitch", "70"])
        assert rc == 0 and TL.SIM_COPY_KEY not in json.load(open(TOOL_PATH)) and TL.active().pitch_deg == 70.0, r
        assert TL.sim_tool_state() == {"simulated_tool": False, "file": TOOL_PATH, "copy_of_real": False, "dropped": None, "note": None}


def test_a_jaw_that_was_moved_since_the_tool_was_attached_stops_ready_and_names_the_way_out():
    """Review finding 49, for the tools of this suite: a pen whose file records where the jaw stood when it was
    attached (`fd tool attach` reads it) and says nothing about the jaw it is taped to. The jaw was compared with
    that record only for a tool held by a jaw squeeze. Now `ready` refuses, before the torque comes on, when the
    jaw stands more than 8 ticks (0.7 deg) from the record; folding away is never held up; a pen recorded on the
    FIXED jaw, and one attached without a look at the arm, are not compared at all."""
    tool = pen(gripper_raw=2061)
    assert TL.jaw_tolerance(tool) == TL.JAW_END_TOL_TICKS == 8 and TL.jaw_tolerance(pen()) is None and TL.jaw_tolerance(None) is None
    # where it stood when the pen was attached (and a few ticks beside it): `ready` as it always was
    for jaw in (2061, 2061 + 8, 2061 - 8):
        arm, bus = sim_arm(tool=tool)
        place(arm, TL.rest_q(arm, tool), gripper=jaw)
        h0 = TL.tip_height(arm, arm.read_q(), tool)
        r = TL.go_ready(arm)
        _check_ready(arm, bus, tool, r, h0, f"jaw at {jaw}")
        assert "jaw" not in r and gripper_goals(bus) in ([], [jaw]), (jaw, r.get("jaw"), gripper_goals(bus))   # (the hold of torque on)
    # the jaw was opened since (by hand, while the arm lay limp): refused before anything is switched on or written
    for jaw in (2184, 2061 + 9, 2061 - 9):
        arm, bus = sim_arm(tool=tool)
        place(arm, TL.rest_q(arm, tool), gripper=jaw)
        before, pos = writes(bus), dict(bus.pos)
        msg = refused(lambda: TL.go_ready(arm), "pen2", f"raw {jaw}", "raw 2061", f"{abs(jaw - 2061)} ticks",
                      "no longer where its tip was measured", "Nothing was written", "does not say on which jaw",
                      "fd tool jaw status", "fd tool jaw mount moving_outer", "fd tool jaw close", "fd tool jaw mount fixed")
        assert writes(bus) == before and bus.pos == pos and not any(bus.torque.values()), msg
        refused(lambda: TL.jaw_in_place(arm), f"raw {jaw}")
        refused(lambda: TL.hold_jaw(arm), f"raw {jaw}")
        assert writes(bus) == before and not any(bus.torque.values())
    # `fd ready` gives the refusal as a result: nothing moved, and the text names the jaw command
    with tool_file(tool):
        bus = RecBus()
        arm = Arm(bus=bus, cfg=load_config(), tool=tool).open()
        place(arm, TL.rest_q(arm, tool), gripper=2184)
        before, pos = writes(bus), dict(bus.pos)
        r, rc = fd_run(["--sim", "ready"], bus)
        assert rc == 1 and r["ok"] is False and r["error"] == "SafetyError" and "raw 2184" in r["message"], r
        assert "fd tool jaw status" in r.get("next", "") + r["message"] and writes(bus) == before and bus.pos == pos, r
        # folding the arm away is never held up by the jaw: `fd tool rest` works, and its result says what is wrong
        place(arm, TL.ready_q(arm, tool), gripper=2184)
        r, rc = fd_run(["--sim", "torque", "on"], bus)
        assert rc == 0, r
        bus.sleep(0.6)
        r, rc = fd_run(["--sim", "tool", "rest"], bus)
        assert rc == 0 and r["ok"] and r["torque"] == "off" and "no longer where its tip was measured" in r["jaw"]["error"], r
        assert round(bus.pos["gripper"]) == 2184 and all(v == 2184 for v in gripper_goals(bus)), gripper_goals(bus)
    # the way out for a pen that is taped to the fixed jaw: say so, and the moving jaw does not matter any more
    fixed = pen(gripper_raw=2061, on_jaw="fixed")
    assert TL.jaw_tolerance(fixed) is None
    arm, bus = sim_arm(tool=fixed)
    place(arm, TL.rest_q(arm, fixed), gripper=2184)
    reads = bus.io["reads"]
    assert TL.jaw_in_place(arm) is None and bus.io["reads"] == reads
    r = TL.go_ready(arm)
    assert "jaw" not in r and round(bus.pos["gripper"]) == 2184, r
    # a pen attached without a look at the arm has no jaw position on record: nothing to compare, as before
    arm, bus = sim_arm(tool=pen())
    place(arm, TL.rest_q(arm, arm.tool), gripper=2600)
    reads = bus.io["reads"]
    assert TL.jaw_in_place(arm) is None and TL.hold_jaw(arm)["held"] is False and bus.io["reads"] == reads
    assert "jaw" not in TL.go_ready(arm)


def test_the_pen_commands_refuse_a_jaw_that_moved_before_the_torque_comes_on():
    """Round 2 (R26 b). `fd ready` and the painting commands compare the jaw with the tool file before they switch
    the arm on (review finding 49); `fd draw` and `fd tool calibrate` switched it on without asking
    (penskill._Run.power): a pen whose jaw had been opened since it was attached, its tip elsewhere than the
    file's, was lowered onto the paper all the same. Now power() asks tooling.jaw_in_place first, once per
    command, and a refusal leaves the arm exactly as it was."""
    import penskill
    tool = pen(gripper_raw=2061)

    def a_run(t, jaw, torque=False):
        arm, bus = sim_arm(tool=t)
        place(arm, TL.rest_q(arm, t), gripper=jaw)
        if torque:
            arm.torque_on()
            bus.sleep(0.4)
        sess = penskill.PenSession(arm, None, lambda exposure="paper": None)
        return penskill._Run(sess, penskill.PenOptions().check(), None), arm, bus

    # the jaw where it stood when the pen was attached (8 ticks beside it still count): the arm is switched on
    for jaw in (2061, 2061 + 8, 2061 - 8):
        run, arm, bus = a_run(tool, jaw)
        run.power()
        assert arm.is_torqued() and run.moved is True and run.sp and round(bus.pos["gripper"]) == jaw, jaw
    # elsewhere: refused before anything is switched on or written, limp or holding, and the run says "not moved"
    for jaw in (2061 + 9, 2061 - 9, 2184):
        for torque in (False, True):
            run, arm, bus = a_run(tool, jaw, torque)
            before, pos = writes(bus), dict(bus.pos)
            msg = refused(run.power, "pen2", f"raw {jaw}", "raw 2061", "no longer where its tip was measured",
                          "Nothing was written", "fd tool jaw status", "fd tool jaw mount fixed")
            assert writes(bus) == before and bus.pos == pos and arm.is_torqued() is torque and run.moved is False, (jaw, torque, msg)
            assert run.pen()["where"] == "not moved", run.pen()
    # asked once per command, not at every power()
    run, arm, bus = a_run(tool, 2061)
    real, asked = TL.jaw_in_place, []
    TL.jaw_in_place = lambda *a, **k: asked.append(1) or real(*a, **k)
    try:
        for _ in range(3):
            run.power()
    finally:
        TL.jaw_in_place = real
    assert asked == [1]
    # a pen recorded on the FIXED jaw, and one attached without a look at the arm: not compared, as everywhere
    for t in (pen(gripper_raw=2061, on_jaw="fixed"), pen()):
        run, arm, bus = a_run(t, 2400)
        run.power()
        assert arm.is_torqued() and round(bus.pos["gripper"]) == 2400
    # the command: `fd tool calibrate` answers with a result, the pen "not moved", the torque off, no photo taken
    with tool_file(tool):
        bus = RecBus()
        arm = Arm(bus=bus, cfg=load_config(), tool=tool).open()
        place(arm, TL.rest_q(arm, tool), gripper=2184)
        before, pos = writes(bus), dict(bus.pos)
        r, rc = fd_run(["--sim", "tool", "calibrate", "--stage", "tip", "--new-sheet"], bus)
        assert rc == 1 and r["ok"] is False and r["stage"] == "start" and "raw 2184" in r["reason"], r
        assert "no longer where its tip was measured" in r["reason"] and "fd tool jaw status" in r["reason"], r["reason"]
        assert r["pen"]["where"] == "not moved" and r["pen"]["torque"] == "off" and r["photos"] == [], r
        assert writes(bus) == before and bus.pos == pos and TL.active().tip_offset == PEN_TIP
        # Round 3 (T15): next and say name the jaw and the way out for THAT tool. They were the general ones of a stop
        # on the way ("look at the reason and at `fd state`", 「校正の途中で止まりました（start: SafetyError）」): the
        # way out stood only at the end of the reason. The next is the refusal's own last sentence, as `fd ready` gives it.
        way = r["reason"].partition("Next: ")[2]
        assert way and r["next"] == "nothing moved. " + way[0].upper() + way[1:], r["next"]
        assert r["next"].startswith("nothing moved. `fd tool jaw status` shows the jaw") and "`fd tool jaw mount fixed`" in r["next"], r["next"]
        assert "fd state" not in r["next"] and "lifted" not in r["next"], r["next"]
        assert "爪" in r["say"] and "fd tool jaw status" in r["say"] and "校正を始めませんでした" in r["say"], r["say"]
        assert "途中で止まりました" not in r["say"] and "アームは動かしていません" in r["say"], r["say"]
    # ... and the same for `fd draw` with a pen that carries the pen calibration (its jaw is asked at the first photo)
    measured = {"at": "2026-11-12T03:10:00", "tip_offset": list(PEN_TIP), "pitch_deg": 75.0}
    drawn = pen(gripper_raw=2061, calibrated={"tip": dict(measured), "contact": dict(measured), "map": dict(measured)},
                contact={"x0": 0.24, "y0": 0.0, "a": -0.0015, "bx": 0.0, "by": 0.0, "rms_mm": 0.2, "points": []},
                corr_map={"centre": [0.22, 0.0], "kind": "affine", "coef": [[0, 0], [1, 0], [0, 1]], "rms_mm": 1.0, "max_mm": 2.0, "n": 9})
    with tool_file(drawn):
        bus = RecBus()
        arm = Arm(bus=bus, cfg=load_config(), tool=drawn).open()
        place(arm, TL.rest_q(arm, drawn), gripper=2184)
        before, pos = writes(bus), dict(bus.pos)
        r, rc = fd_run(["--sim", "draw", "cat", "--new-sheet"], bus)
        assert rc == 1 and r["ok"] is False and r["stage"] == "photo" and "raw 2184" in r["reason"] and r["pen"]["where"] == "not moved", r
        assert r["next"].startswith("nothing moved. `fd tool jaw status` shows the jaw") and "`fd tool jaw mount fixed`" in r["next"], r["next"]
        assert "爪" in r["say"] and "fd tool jaw status" in r["say"] and "描画を始めませんでした" in r["say"], r["say"]
        assert writes(bus) == before and bus.pos == pos and not any(bus.torque.values())
        # a stop of another kind keeps the general words: the jaw is named only when the jaw is what refused
        import penskill as PSK
        real = PSK._Run.look
        PSK._Run.look = lambda self, *a, **k: (_ for _ in ()).throw(SafetyError("the photo pose is outside the tool's joint limits"))
        try:
            place(arm, TL.rest_q(arm, drawn), gripper=2061)
            r, rc = fd_run(["--sim", "draw", "cat", "--new-sheet"], bus)
        finally:
            PSK._Run.look = real
        assert rc == 1 and r["stage"] == "photo" and r["next"].startswith("look at the reason and at `fd state`"), r
        assert "描画の途中で止まりました" in r["say"] and "爪" not in r["say"], r["say"]


def test_the_example_of_a_tool_joint_limit_is_the_cap_that_was_measured():
    """Review finding 58. The examples of a brush's wrist limit said 20..103; 103 is the wrist's own limit, where a
    move stopped 3.7 deg short with the handle against the arm on 2026-11-13. The cap that was measured is 97,
    and an operator copies the example."""
    import inspect
    text = inspect.getsource(TL.Tool)
    assert '{"wrist_flex": [20, 97]}' in text and "20, 103]" not in text and "20:103" not in text
    whole = inspect.getsource(TL) + inspect.getsource(Arm.q_limits)
    assert "wrist_flex=20:103" not in whole and "[20, 103]" not in whole and "20..103" not in whole
    # ... and it is a limit the arm applies: inside the wrist's own range, with the 6 deg to spare that were measured
    arm, _ = sim_arm(tool=brush())
    lo, hi = (math.degrees(v) for v in arm.q_limits()["wrist_flex"])
    bare, _ = sim_arm(tool=None)
    assert abs(lo - 20.0) < 1e-9 and abs(hi - 97.0) < 1e-9 and math.degrees(bare.q_limits()["wrist_flex"][1]) > 100.0


def test_zz_the_real_tool_files_were_never_touched():
    """(Runs LAST: the runner sorts the tests by name, and six tests used to run after this one.)"""
    assert (TL.TOOL_FILE == os.path.join(tazuna_paths.ROOT, "tool_active.json")
            and TL.SIM_TOOL_FILE == os.path.join(tazuna_paths.ROOT, "tool_active.sim.json"))
    assert os.environ["FD_TOOL"] == TOOL_PATH
    # (integration: "no tool file exists" became "the tool files are what they were before the tests": the
    # suite must pass in a tree whose arm really carries a tool, or people detach the tool to run it)
    now = _tree_tool_files()
    assert set(now) == {TL.TOOL_FILE, TL.SIM_TOOL_FILE}
    assert now == _TREE_TOOL_FILES, "a test created, changed or removed a tool file of the tree"


if __name__ == "__main__":
    import shutil
    import time
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
                    print(f"PASS {name} ({time.time() - t0:.1f} s)")
                except Skip as e:
                    skipped += 1
                    print(f"SKIP {name}: {e}")
                except Exception:
                    fails += 1
                    print("FAIL", name)
                    traceback.print_exc()
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
    print(f"tests: {n}  passed: {n - fails - skipped}  skipped: {skipped}  failures: {fails}  ({time.time() - t_all:.0f} s)")
    sys.exit(1 if fails else 0)
