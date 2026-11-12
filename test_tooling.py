"""Tests for a tool taped to the gripper (tooling.py and the guards in robot.py / skills.py / fd.py).
Run: python test_tooling.py [name-part ...]      (no hardware: simulated servo bus, saved photos)

What they show (SPEC section 7, TOOL): with a tool file present, in simulation, grip / grasp / pick / take /
do / park / probe / bench refuse without a single bus write; `ready` ends with the tool tip at the stand-by
point and never below 2 cm on the way, also from the folded rest pose and from the paper; `fd tool rest`
reaches a pose with the tool tip 4-7 cm above the table for tools 2-12 cm long, along and across the
fingers, or refuses cleanly; torque off is refused in a raised pose; after `detach` the old behaviour is
back; without a tool file nothing changes.

Every test uses $FD_TOOL (a scratch file), so the real tool_active.json is never read or written.
"""
import contextlib
import io
import json
import math
import os
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
        p = os.path.join(HERE, name)
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
        r, rc = fd_run(["--sim", "tool", "detach"])  # nothing simulated left: the real tool is NOT touched
        assert rc == 0 and r["removed"] is False and "mirrors" in r["note"] and TL.active().name == "real_pen", r
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
            assert rc == 1 and r["stage"] == "tool" and "--contact" in r["reason"] and "fd draw" in r["next"], (argv, r)
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
    $FD_TOOL (92 tests, about 15 s)."""
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
        assert p.returncode == 0 and "failures: 0" in tail and "tests: 92" in tail, tail


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


def test_the_real_tool_files_were_never_touched():
    assert TL.TOOL_FILE == os.path.join(HERE, "tool_active.json") and TL.SIM_TOOL_FILE == os.path.join(HERE, "tool_active.sim.json")
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
