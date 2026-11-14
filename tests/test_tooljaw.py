"""Tests for the jaw of a tool that is taped to the outer face of the moving jaw: tooling.move_jaw / hold_jaw and
the jaw fields of the tool file, robot.Arm.jaw_goal, and the command `fd tool jaw` (tooljaw.py).
Run: python tests/test_tooljaw.py [name-part ...]      (no hardware: simulated servo bus; about 15 s)

What they show. The jaw of such a tool is walked in steps of at most 15 ticks every 0.12 s and stops at its
goal; a jaw that meets the fixed jaw stops there and is held with the squeeze (the numbers of 2026-11-13: raw
2041, goal 2016, load about 110); a jaw that something keeps open, or that lags, stops with the push taken
off and a result that says so; a tool between the jaws, on the fixed jaw, or of which the file does not say
where it is, is refused before the bus is touched. After a jaw move the tool file has the new jaw position
and holding goal, the tip the jaw model predicts, and neither a tip measurement nor a pen calibration.
hold_jaw puts the squeeze back after the torque was off and refuses a jaw that stands elsewhere. Old tool
files load and are written back unchanged. `fd tool jaw ...` answers with house results, also as a
process of its own, and the hardware lock refuses the real `close` before the servo bus is opened. One test
runs the same calls on the real bus class (servo_bus.Bus) with six simulated servos behind a pty.

Added after the review of 2026-11-13: a servo bus that dies in the middle of a jaw move leaves the tool file
saying that the jaw's position is not known and the tip not measured; a jaw WITHOUT a recorded hold is
compared with the tool file too (8 ticks), before the torque comes on, and the refusal names the `fd tool
jaw` command for that tool; a damaged tool file is answered in the house shape by every sub-command; a
simulated command that stores the tool says that it wrote the simulated tool file, and a simulated copy is
left behind once the real tool is changed or replaced; `--help` / `-h` are answered with the usage.

Added with round 2 (2026-11-14): every command a result names is ONE command that the parsers take (a bar inside
the backticks is a shell pipe when it is typed: `fd tool jaw mount moving_outer|fixed|between`); a finished close
names the command that comes next; the test of the simulated tool file works in a stand-in project folder; the
check of brush work is a test of its own, and where it cannot be made it is a counted SKIP, not a printed note.

A SIMULATED PASS IS NOT A PASS ON THE REAL ARM: the simulated jaw follows its goal exactly and its fixed jaw
is a rigid stop. How far the real jaw lags in free travel, how it creeps under the squeeze and where it meets
the fixed jaw were measured on 2026-11-13 only with the scratch scripts these functions replace.

Every test uses $FD_TOOL (a scratch file) and $TAZUNA_HW_LOCK (a scratch lock that stays set for the whole
run). The rig's own files (tool file, painting set-up, configuration) are only read, to show at the end that
they are what they were; the user's lock is not looked at; the run log of the `fd` processes goes to a
scratch folder, not to runs/.
"""

import os as _os  # the harness modules live in tools/<feature>/ (see tools/tazuna_paths.py)
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "tools"))
import tazuna_paths  # noqa: E402,F401

import contextlib
import hashlib
import io
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
os.environ["FD_CONFIG"] = os.path.join(HERE, "testdata", "pen_20261111", "config.json")
os.environ["FD_CAMERA_MODEL"] = os.path.join(HERE, "testdata", "pen_20261111", "camera_model.json")
_TMP = tempfile.mkdtemp(prefix="tz_tooljaw_")
TOOL_PATH = os.path.join(_TMP, "tool_active.json")
HW_LOCK = os.path.join(_TMP, "hardware.lock")
os.environ["FD_TOOL"] = TOOL_PATH
os.environ["TAZUNA_HW_LOCK"] = HW_LOCK
with open(HW_LOCK, "w") as _f:        # locked from the first line to the last: no test here can reach the real arm
    json.dump({"reason": "tests/test_tooljaw.py", "since": "test"}, _f)

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


def _tree_files() -> dict:
    """The rig's own state files as they are now (path -> md5, None = no file): the tests must leave them alone."""
    out = {}
    for name in ("tool_active.json", "tool_active.sim.json", "paint_setup.json", "paint_state.json",
                 os.path.join("config", "config.json"), os.path.join("config", "camera_model.json")):
        p = os.path.join(tazuna_paths.ROOT, name)
        try:
            with open(p, "rb") as fh:
                out[p] = hashlib.md5(fh.read()).hexdigest()
        except FileNotFoundError:
            out[p] = None
    return out


_TREE_FILES = _tree_files()  # before any test ran

import numpy as np  # noqa: E402

import fd  # noqa: E402
import gripper_model as GM  # noqa: E402
import hwlock  # noqa: E402
import kinematics as K  # noqa: E402
import tooling as TL  # noqa: E402
import tooljaw  # noqa: E402
from robot import ARM, Arm, MotionAborted, SafetyError, load_config  # noqa: E402
from servo_bus import LOCK_PATH as BUS_FLOCK, BusError  # noqa: E402
from sim_bus import SimBus  # noqa: E402

fd.log_event = lambda *a, **k: None  # no run log from tests
np.set_printoptions(precision=4, suppress=True)
CFG = load_config()
TOUCH = int(CFG["gripper_touch_raw"])            # 2050: where the fingertips touch in free air
FIXED = 2041                                     # where the jaw met the fixed jaw under the detection push (17:21)
AT_ZERO = 2182                                   # the jaw at 0.2 deg, where the brush lay along the fingers (17:01)
TIP_AT_ZERO = [-0.0051, -0.0025, 0.0604]         # the brush tip measured there (16:50)
TIP_CLOSED = [0.0160, -0.0011, 0.0558]           # ... and with the jaw closed and squeezed (17:25)
BRUSH_LIMITS = {"wrist_flex": [20.0, 97.0], "wrist_roll": [-5.0, 5.0]}
BACKUP = "/home/jetson/lerobot/jetharness_work/paint_update/state_backup"
TOOLS_DIR = os.path.join(tazuna_paths.ROOT, "tools")


class Skip(Exception):
    pass


class JawBus(SimBus):
    """SimBus that remembers every gripper goal with the time and where the jaw stood then, and can have a jaw
    that moves slower than its goal (jaw_speed, ticks/s), a servo fault that appears at a jaw position, or a
    bus that stops answering after so many gripper goals (die_after_goals; writes_only: the servos still answer
    reads, as with a supply that sags under load)."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.goals = []              # (time, goal, jaw position when it was written)
        self.jaw_speed = None
        self.fault_below = None      # status bit 32 (overload) once the jaw is below this position
        self.interrupt_at = None     # raise KeyboardInterrupt in the n-th sleep from now
        self.die_after_goals = None  # every bus transaction fails once this many gripper goals were written
        self.writes_only = False     # ... only the writes do
        self.die_reading = None      # the bus dies at the first read of this register of the gripper
        self.dead = False
        self.refused = 0             # bus transactions that got no answer

    def _answer(self, writes: bool):
        """BusError when the bus is dead (called before every transaction)."""
        if self.die_after_goals is not None and len(self.goals) >= self.die_after_goals:
            self.dead = True
        if self.dead and (writes or not self.writes_only):
            self.refused += 1
            raise BusError("no answer from servo 6 (gripper): test")

    def _set(self, j, reg, v):
        if j == "gripper" and reg == "Goal_Position":
            self.goals.append((self.now, int(v), self.pos["gripper"]))
        super()._set(j, reg, v)

    def _advance_joint(self, j, dt):
        if j == "gripper" and self.jaw_speed is not None and self.torque[j]:
            step = max(-self.jaw_speed * dt, min(self.jaw_speed * dt, self._eff_goal(j) - self.pos[j]))
            self.pos[j] += self._blocked_step(j, step)
        else:
            super()._advance_joint(j, dt)
        if j == "gripper" and self.fault_below is not None and self.pos[j] < self.fault_below:
            self.fault[j] = 32

    def sleep(self, dt):
        if self.interrupt_at is not None:
            self.interrupt_at -= 1
            if self.interrupt_at <= 0:
                self.interrupt_at = None
                raise KeyboardInterrupt
        super().sleep(dt)


def _mortal(name: str, writes: bool):
    real = getattr(SimBus, name)

    def call(self, *a, **k):
        if name == "read" and self.die_reading is not None and tuple(a[:2]) == ("gripper", self.die_reading):
            self.dead = True
        self._answer(writes)
        return real(self, *a, **k)
    call.__name__ = name
    return call


for _name, _writes in (("read", False), ("sync_read", False), ("positions", False), ("feedback", False), ("snapshot", False),
                       ("limits", False), ("faults", False), ("write", True), ("sync_write", True), ("torque_on", True),
                       ("torque_off", True), ("set_p_gain", True)):
    setattr(JawBus, _name, _mortal(_name, _writes))       # every transaction of a JawBus can find the bus dead
del _name, _writes


def brush(**over):
    """The thin brush of 2026-11-13 on the outer face of the moving jaw, the jaw at 0.2 deg."""
    kw = dict(name="brush_thin", tip_offset=list(TIP_AT_ZERO), mount="along_fingers", pitch_deg=65.0, gripper_raw=AT_ZERO,
              on_jaw="moving_outer", joint_limits_deg={j: list(v) for j, v in BRUSH_LIMITS.items()})
    kw.update(over)
    return TL.Tool(**kw)


def held(**over):
    """The same brush as the evening left it: jaw closed on the fixed jaw, held with 25 ticks, tip measured."""
    kw = dict(tip_offset=list(TIP_CLOSED), gripper_raw=FIXED, gripper_goal_raw=FIXED - 25, mount="other",
              tip_measured={"at": "2026-11-13T17:25:00", "photos": 34, "rms_px": 6.6})
    kw.update(over)
    return brush(**kw)


def place(arm, q, gripper=None):
    bus = arm.bus
    raw = arm.q_to_raw_f(q)
    if gripper is not None:
        raw["gripper"] = gripper
    for j, v in raw.items():
        bus.pos[j] = bus.goal[j] = bus.sp[j] = float(v)


def rig(tool, jaw=None, fixed_jaw=FIXED, pose="standby", torque=False, cfg=None):
    """A simulated arm with `tool`, standing at the tool's stand-by pose (or its rest pose), the jaw at `jaw`
    (default: where the tool file has it) and a fixed jaw at `fixed_jaw` (None: none at all)."""
    bus = JawBus(realtime=False)
    arm = Arm(bus=bus, cfg=cfg if cfg is not None else load_config(), tool=tool).open()
    q = TL.rest_q(arm, tool) if pose == "rest" else TL.ready_q(arm, tool)
    place(arm, q, gripper=tool.gripper_raw if jaw is None else jaw)
    bus.blocker = (lambda pos: False) if fixed_jaw is None else (lambda pos, s=float(fixed_jaw): pos["gripper"] < s)
    if torque:
        arm.torque_on()
        bus.sleep(0.6)
    bus.goals.clear()
    return arm, bus


def traffic(bus):
    """(reads, writes, goal writes) of the simulated bus so far."""
    return bus.io["reads"], bus.io["writes"], len(bus.log)


def jaw_of(bus):
    return int(round(bus.pos["gripper"]))


def advances(bus):
    """The goal writes that pushed the goal further ahead of the jaw (the steps of a walk; not a release or the
    squeeze that follows a stop): [(time, goal, how far ahead of the jaw it was), ...]."""
    out, prev = [], None
    for t, g, pos in bus.goals:
        if prev is not None and abs(g - pos) > abs(prev - pos) - 1e-6 and g != prev:
            out.append((t, g, abs(g - pos), abs(g - prev)))
        prev = g
    return out


def check_walk(bus, max_ahead=45):
    """Every step of a walk: at most 15 ticks, not sooner than 0.12 s after the one before, and the goal never
    further ahead of the jaw than the lag that stops a walk plus one step."""
    steps = advances(bus)
    assert steps, "no goal was written"
    assert max(s[3] for s in steps) <= TL.JAW_STEP_TICKS, max(s[3] for s in steps)
    gaps = [b[0] - a[0] for a, b in zip(steps, steps[1:])]
    assert not gaps or min(gaps) >= TL.JAW_STEP_S - 1e-9, min(gaps)
    assert max(s[2] for s in steps) <= max_ahead + 1e-6, max(s[2] for s in steps)
    return steps


@contextlib.contextmanager
def tool_file(tool):
    TL.clear()
    if tool is not None:
        TL.save(tool)
    try:
        yield
    finally:
        TL.clear()


def cli(words, arm=None):
    """tooljaw.main(words) with the test's arm in the place of the one it would open."""
    real = tooljaw.make_arm
    if arm is not None:
        def make(sim):
            assert sim, "a test asked for the REAL arm"
            arm.tool = TL.active(sim=True)          # as a fresh process reads it
            return arm
        tooljaw.make_arm = make
    try:
        res = tooljaw.main(list(words))
        _RESULTS.append(res)
        return res
    finally:
        tooljaw.make_arm = real


_RESULTS = []           # every result a test of this suite looked at (the commands they name: test_zy_...)


def house(r, ok=None, stage=None):
    """A result in the house style: ok, stage, next (English, one line), say (Japanese), reason when not ok."""
    _RESULTS.append(r)
    assert isinstance(r.get("ok"), bool) and r.get("stage"), r
    assert isinstance(r.get("next"), str) and r["next"] and "\n" not in r["next"], r
    assert isinstance(r.get("say"), str) and any(0x3040 <= ord(ch) <= 0x30FF or 0x4E00 <= ord(ch) <= 0x9FFF for ch in r["say"]), r
    if not r["ok"]:
        assert r.get("reason"), r
    if ok is not None:
        assert r["ok"] is ok, r
    if stage is not None:
        assert r["stage"] == stage, r
    json.dumps(r, default=str)
    return r


_FD_CODE = ("import os, sys\n"
            "def _no_hardware(event, args):\n"
            "    if event == 'open' and args and isinstance(args[0], (str, bytes)):\n"
            "        p = os.fsdecode(args[0])\n"
            "        if p.startswith(('/dev/ttyACM', '/dev/ttyUSB', '/dev/video')):\n"
            "            raise RuntimeError('test touched hardware: open(%s)' % p)\n"
            "sys.addaudithook(_no_hardware)\n"
            "sys.path.insert(0, sys.argv[1])\n"
            "import tazuna_paths\n"
            "import fd\n"
            "fd.RUNS = sys.argv[2]\n"            # the run log of this test goes to a scratch folder, not to the rig's runs/
            "sys.exit(fd.main(sys.argv[3:]))\n")


def fd_process(*argv):
    """`fd ARGV` in a process of its own: fd.main as the launcher starts it, with the run log in a scratch folder
    and the same no-hardware hook. Returns (result, exit code)."""
    assert os.path.exists(HW_LOCK) and os.environ["TAZUNA_HW_LOCK"] == HW_LOCK and os.environ["FD_TOOL"] == TOOL_PATH
    p = subprocess.run([sys.executable, "-c", _FD_CODE, TOOLS_DIR, os.path.join(_TMP, "runs"), *argv], capture_output=True,
                       text=True, timeout=120, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    assert lines, (argv, p.returncode, p.stderr[-600:])
    _RESULTS.append(json.loads(lines[-1]))
    return _RESULTS[-1], p.returncode


def refused(fn, *texts):
    try:
        fn()
    except SafetyError as e:
        _RESULTS.append({"reason": str(e)})             # (a refusal's text reaches the operator as a result's reason)
        for t in texts:
            assert t in str(e), (t, str(e))
        return str(e)
    raise AssertionError("not refused")


# ----------------------------------------------------------------------------- the jaw model
def test_estimate_is_the_rotation_about_the_jaw_pivot_and_was_a_centimetre_off():
    """2026-11-13: tip (-5.1, -2.5, 60.4) mm at 0.2 deg became (16.0, -1.1, 55.8) mm at -12.2 deg; the rotation
    about the pivot predicted (23.4, -2.5, 52.3)."""
    est = TL.jaw_tip_estimate(TIP_AT_ZERO, 0.2, -12.2)
    assert np.allclose(np.array(est) * 1000.0, [23.4, -2.5, 52.3], atol=0.06), est
    off = float(np.linalg.norm(np.array(est) - np.array(TIP_CLOSED)))
    assert 0.007 < off < 0.0095 and off <= TL.JAW_ESTIMATE_ERR_M, off      # 8.3 mm: "about 1 cm", and the constant says so
    # it IS the rigid rotation of the jaw model: the same point through gripper_model.jaw_point, y untouched
    x, z = GM.jaw_point(-12.4, (TIP_AT_ZERO[0] * 1000.0, TIP_AT_ZERO[2] * 1000.0))
    assert np.allclose(np.array(est) * 1000.0, [x, TIP_AT_ZERO[1] * 1000.0, z], atol=0.006)
    r0 = math.hypot(TIP_AT_ZERO[0] * 1000.0 - GM.PIVOT_X, TIP_AT_ZERO[2] * 1000.0 - GM.PIVOT_Z)
    assert abs(math.hypot(est[0] * 1000.0 - GM.PIVOT_X, est[2] * 1000.0 - GM.PIVOT_Z) - r0) < 0.02     # same distance from the pivot
    assert TL.jaw_tip_estimate(TIP_AT_ZERO, 5.0, 5.0) == [round(v, 5) for v in TIP_AT_ZERO]               # no turn, no change
    back = TL.jaw_tip_estimate(TL.jaw_tip_estimate(TIP_AT_ZERO, 0.0, 30.0), 30.0, 0.0)
    assert np.allclose(back, TIP_AT_ZERO, atol=3e-5), back
    # opening carries the tip to the side of the moving jaw (-x): 9 cm beside the finger line at 35 deg
    assert TL.jaw_tip_estimate(TIP_AT_ZERO, 0.2, 35.0)[0] < -0.08
    # raw <-> degrees as the arm has them, and the estimate for a tool file
    arm, _ = rig(brush())
    for raw in (FIXED, TOUCH, AT_ZERO, 2576):
        assert abs(TL.jaw_q_deg(CFG, raw) - arm.gripper_q_deg(raw)) < 1e-9
    assert abs(TL.jaw_q_deg(CFG, FIXED) + 12.2) < 0.01 and abs(TL.jaw_q_deg(CFG, AT_ZERO) - 0.19) < 0.01
    assert TL.jaw_range_raw(CFG) == arm._gripper_clamp()
    e = TL.jaw_estimate(CFG, brush(), FIXED)
    assert e["tip_offset_mm"] == [23.4, -2.5, 52.3] and e["expect_error_mm"] == 10.0 and abs(e["turned_deg"] + 12.39) < 0.01, e
    assert e["from_raw"] == AT_ZERO and e["to_raw"] == FIXED and 28.0 < e["tip_moves_mm"] < 31.0, e
    assert TL.jaw_estimate(CFG, brush(gripper_raw=None), FIXED) is None       # nothing to turn from


# ----------------------------------------------------------------------------- the tool file
def test_jaw_fields_survive_save_and_load():
    with tool_file(None):
        t = held(notes=["closed by hand-written scripts that evening"])
        t.extra = {"added_by_a_later_version": 1}
        TL.save(t)
        back = TL.active()
        assert (back.on_jaw, back.gripper_raw, back.gripper_goal_raw) == ("moving_outer", FIXED, FIXED - 25)
        assert back.tip_measured == {"at": "2026-11-13T17:25:00", "photos": 34, "rms_px": 6.6}
        assert back.to_dict() == t.to_dict() and back.extra == {"added_by_a_later_version": 1}
        on_disk = json.load(open(TOOL_PATH))
        assert on_disk["on_jaw"] == "moving_outer" and on_disk["gripper_goal_raw"] == FIXED - 25
        assert on_disk["tip_measured"]["photos"] == 34
        for where in ("fixed", "between"):
            TL.save(brush(on_jaw=where))
            assert TL.active().on_jaw == where and TL.active().gripper_goal_raw is None
        # a record that would be a danger is an error, never a tool with the jaw unlocked or a squeeze nobody measured
        good = held().to_dict()
        for change, text in (({"on_jaw": "outer"}, "on_jaw"), ({"on_jaw": True}, "on_jaw"),
                             ({"on_jaw": "between"}, "gripper_goal_raw"), ({"on_jaw": None}, "gripper_goal_raw"),
                             ({"gripper_goal_raw": FIXED + 5}, "below"), ({"gripper_goal_raw": FIXED}, "below"),
                             ({"gripper_goal_raw": FIXED - 61}, "below"), ({"gripper_goal_raw": "2016"}, "gripper_goal_raw"),
                             ({"gripper_goal_raw": 2016.5}, "gripper_goal_raw"), ({"gripper_raw": None}, "gripper_raw"),
                             ({"tip_measured": "yes"}, "tip_measured"), ({"tip_measured": [1]}, "tip_measured")):
            d = dict(good)
            d.update(change)
            with open(TOOL_PATH, "w") as f:
                json.dump(d, f)
            msg = refused(TL.active, text, "fd tool detach")
            assert "cannot be used" in msg, msg
            refused(lambda: Arm(bus=SimBus(), cfg=load_config()))
        for bad in (held(on_jaw="between"), held(gripper_goal_raw=FIXED + 1), brush(on_jaw="sideways"), held(tip_measured=3)):
            refused(lambda: TL.save(bad))
        TL.clear()
        assert not os.path.exists(TOOL_PATH)


def test_old_tool_files_load_and_are_written_back_unchanged():
    """The rig's own tool files of 2026-11-13, written before these fields existed: read with the jaw not said
    (so: locked), and written back byte for byte."""
    seen = 0
    for name in ("tool_active.json", "tool_active.brush_thin.json"):
        path = os.path.join(BACKUP, name)
        if not os.path.exists(path):
            continue
        seen += 1
        text = open(path, encoding="utf-8").read()
        old = json.loads(text)
        assert not any(k in old for k in TL.Tool._JAW_FIELDS), name
        t = TL.load(path)
        assert (t.on_jaw, t.gripper_goal_raw, t.tip_measured) == (None, None, None) and t.extra == {}, name
        assert t.to_dict() == old and list(t.to_dict()) == list(old), name          # same keys, same order, same values
        with tool_file(None):
            TL.save(t)
            assert open(TOOL_PATH, encoding="utf-8").read() == text, f"{name} is not written back byte for byte"
            # ... and behaves as before: the gripper locked, no jaw command, nothing written by hold_jaw. (Since the
            # review of 2026-11-13 the jaw of such a tool IS looked at: these files record where it stood, one read)
            arm, bus = rig(TL.active(), jaw=t.gripper_raw)
            before = traffic(bus)
            assert TL.hold_jaw(arm) == {"held": False, "wrote": False, "why": "no jaw hold is recorded for the tool on the gripper",
                                        "pos_raw": t.gripper_raw, "from_record_ticks": 0}
            assert TL.jaw_in_place(arm) == {"pos_raw": t.gripper_raw, "from_record_ticks": 0, "tol_ticks": TL.JAW_END_TOL_TICKS,
                                            "held": False}
            assert traffic(bus) == (before[0] + 2, before[1], before[2]), "more than two reads, or a write"
            before = traffic(bus)
            for call in (lambda: arm.jaw_goal(t.gripper_raw - 5), lambda: TL.move_jaw(arm), lambda: arm.set_gripper(0.5)):
                msg = refused(call)
                assert t.name in msg and traffic(bus) == before, msg
            assert "fd tool jaw" not in TL.lock_message(t, "park")                 # the old text, word for word
    if not seen:
        raise Skip(f"no tool file of the rig in {BACKUP}")
    # the same for a file made here without the keys
    with tool_file(None):
        plain = TL.Tool(name="pen2", tip_offset=[0.0015, -0.0081, 0.0621], mount="along_fingers", pitch_deg=75.0, gripper_raw=2061)
        TL.save(plain)
        assert not any(k in json.load(open(TOOL_PATH)) for k in TL.Tool._JAW_FIELDS)
        assert TL.active().to_dict() == plain.to_dict()


def test_a_changed_tip_clears_the_tip_measurement():
    t = held()
    t.tip_offset = list(TIP_CLOSED)                    # the same tip again: not a change
    assert t.tip_measured is not None
    t.notes.append("x")
    t.pitch_deg = 70.0
    assert t.tip_measured is not None                  # nothing but the tip clears it
    t.tip_offset = [0.016, -0.0011, 0.0568]            # 1 mm longer
    assert t.tip_measured is None
    # the way to give a tool a measured tip: offset first, record after it
    t = held(contact={"x0": 0.2, "y0": 0.0, "a": 0.005, "rms_mm": 0.5, "points": []}, calibrated={"contact": {"at": "x"}})
    dropped = TL.set_measured_tip(t, [0.0157, -0.0008, 0.0559], {"at": "now", "rms_px": 6.2})
    assert t.tip_offset == [0.0157, -0.0008, 0.0559] and t.tip_measured == {"at": "now", "rms_px": 6.2}
    assert set(dropped) == {"tip_measured", "contact", "calibrated"} and t.contact is None and t.calibrated == {}
    assert TL.set_measured_tip(t, [0.0157, -0.0008, 0.0559], {"at": "again"}) == [] and t.tip_measured == {"at": "again"}
    refused(lambda: TL.set_measured_tip(t, [0.0, 0.0, 9.0], {"at": "x"}))
    refused(lambda: TL.set_measured_tip(t, [0.0157, -0.0008, 0.0559], "measured"))
    # through the command that changes a tip by hand (fd.py does not know the field: the tool object does)
    with tool_file(held()):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = fd.main(["--sim", "tool", "set", "--note", "looked at it", "--no-robot"])
        assert rc == 0 and TL.active().tip_measured is not None, buf.getvalue()
        with contextlib.redirect_stdout(io.StringIO()):
            rc = fd.main(["--sim", "tool", "set", "--tip", "0.017,-0.001,0.056", "--no-robot"])
        now = TL.active()
        assert rc == 0 and now.tip_offset == [0.017, -0.001, 0.056] and now.tip_measured is None
        assert now.on_jaw == "moving_outer" and now.gripper_goal_raw == FIXED - 25        # the jaw record is not the tip's


# ----------------------------------------------------------------------------- the gate in the arm
def test_jaw_goal_is_the_one_gripper_goal_and_only_a_small_step():
    arm, bus = rig(brush(), torque=True)
    before = traffic(bus)
    assert arm.jaw_goal(AT_ZERO - 15) == AT_ZERO - 15 and bus.goal["gripper"] == AT_ZERO - 15
    bus.sleep(0.3)
    writes = bus.io["writes"]
    for bad, text in ((AT_ZERO - 120, "ticks from where the jaw stands"), (AT_ZERO + 80, "ticks from where the jaw stands"),
                      (2000, "ticks from where the jaw stands"), (1990, "limits"), (3700, "limits"), (2100.5, "whole number"),
                      (float("nan"), "whole number"), ("soon", "whole number"), (None, "whole number"), (True, "whole number")):
        refused(lambda: arm.jaw_goal(bad), text, "othing was written")
        assert bus.io["writes"] == writes and int(bus.goal["gripper"]) == AT_ZERO - 15, bad
    assert arm.jaw_goal(AT_ZERO - 15 - 85) == AT_ZERO - 100          # 85 ticks ahead of the jaw: the most a walk ever asks for
    bus.sleep(0.5)
    writes = bus.io["writes"]
    refused(lambda: arm.jaw_goal(AT_ZERO - 20, max_ahead_ticks=200), "max_ahead_ticks")
    # every other gripper command stays locked, with the pointer to the jaw command
    for call in (lambda: arm.set_gripper(0.5), lambda: arm.grasp(), lambda: arm.pinch(), lambda: arm.grasp_fast(),
                 lambda: arm._gripper_clamp(2500), lambda: arm.gripper_raw_for(0.2)):
        msg = refused(call, "brush_thin", "fd tool rest", "fd tool detach", "`fd tool jaw close` (`fd tool jaw help` lists")
        assert "DEG" not in msg, msg                                    # (no placeholder inside backticks: round 3)
        assert bus.io["writes"] == writes and "|" not in msg, msg
    # a goal must never be what switches a limp servo on
    bus.torque_off(("gripper",))
    writes = bus.io["writes"]
    refused(lambda: arm.jaw_goal(jaw_of(bus) - 5), "gripper torque is off")
    assert bus.io["writes"] == writes and bus.torque["gripper"] == 0
    # no tool, a tool between the jaws, on the fixed jaw, or not said: the lock, before any bus traffic
    for tool in (None, brush(on_jaw="between"), brush(on_jaw="fixed"), brush(on_jaw=None)):
        a, b = rig(brush(), torque=True)
        a.tool = tool
        before = traffic(b)
        msg = refused(lambda: a.jaw_goal(AT_ZERO - 5))
        assert traffic(b) == before and ("no tool is attached" in msg if tool is None else "fd tool rest" in msg), msg
    del before


# ----------------------------------------------------------------------------- the walk
def test_the_jaw_follows_in_small_steps_and_stops_at_its_goal():
    """`close --to DEG` and `open --to DEG`: thin_close.py without the person watching."""
    tool = brush(gripper_raw=2576, tip_offset=[-0.0905, -0.0097, 0.0611], mount="other")     # as taped: 35 deg open, 9 cm aside
    arm, bus = rig(tool)
    assert not any(bus.torque.values())
    target = arm.gripper_raw_for_q(0.0)
    r = TL.move_jaw(arm, "close", 0.0)
    assert r["ok"] and not r["stopped_early"] and r["met_fixed_jaw"] is None and r["to"] == 0.0, r
    assert r["from_raw"] == 2576 and r["pos_raw"] == target == jaw_of(bus) == 2180 and r["goal_raw"] == target, r
    assert abs(r["q_deg"]) < 0.05 and r["squeeze_ticks"] == 0 and r["load"] == 0 and r["temp_c"] == 25, r
    steps = check_walk(bus, max_ahead=TL.JAW_STEP_TICKS)
    assert len(steps) == math.ceil((2576 - target) / 15) == r["steps"] and r["worst_lag_ticks"] <= 2, (len(steps), r)
    assert [g for _, g, _ in bus.goals][-1] == target and all(target <= g <= 2576 for _, g, _ in bus.goals)
    # only the gripper was switched on; the arm lies as it lay
    assert r["torque_switched_on"] and bus.torque["gripper"] == 1 and not any(bus.torque[j] for j in ARM)
    # the tool went with the jaw: the file's facts are about the new place, the hold is none
    assert (tool.gripper_raw, tool.gripper_goal_raw) == (target, None) and r["record"]["tool_moved"], r["record"]
    want = TL.jaw_tip_estimate([-0.0905, -0.0097, 0.0611], arm.gripper_q_deg(2576), arm.gripper_q_deg(target))
    assert np.allclose(tool.tip_offset, want), (tool.tip_offset, want)
    # and open again, to 20 deg
    bus.goals.clear()
    r = TL.move_jaw(arm, "open", 20.0)
    up = arm.gripper_raw_for_q(20.0)
    assert r["ok"] and r["pos_raw"] == up == jaw_of(bus) and r["direction"] == "open" and not r["torque_switched_on"], r
    check_walk(bus, max_ahead=TL.JAW_STEP_TICKS)
    assert tool.gripper_raw == up and tool.gripper_goal_raw is None
    # `close` never opens, `open` never closes, and the angle must be one the gripper has
    before, at = traffic(bus), jaw_of(bus)
    refused(lambda: TL.move_jaw(arm, "close", 40.0), "would OPEN the jaw", "fd tool jaw open --to 40")
    refused(lambda: TL.move_jaw(arm, "open", 5.0), "would CLOSE the jaw", "fd tool jaw close --to 5")
    refused(lambda: TL.move_jaw(arm, "open", "closed"), "--to DEG")
    refused(lambda: TL.move_jaw(arm, "close", -12.2), "close --to closed")
    refused(lambda: TL.move_jaw(arm, "open", 140.0), "deg")
    refused(lambda: TL.move_jaw(arm, "close", "wide"), "--to")
    refused(lambda: TL.move_jaw(arm, "shut", 0.0), "close")
    refused(lambda: TL.move_jaw(arm, "close", 0.0, squeeze_ticks=10), "--squeeze")
    for kw in ({"step_ticks": 16}, {"step_ticks": 0}, {"step_ticks": 2.5}, {"lag_ticks": 5}, {"lag_ticks": 41}, {"step_ticks": "fast"}):
        refused(lambda: TL.move_jaw(arm, "close", 0.0, **kw), "whole number")
    assert jaw_of(bus) == at and traffic(bus)[1:] == before[1:], "a refusal wrote to the bus"
    # a smaller step is a slower walk
    bus.goals.clear()
    r = TL.move_jaw(arm, "close", 10.0, step_ticks=5)
    assert r["ok"] and max(s[3] for s in advances(bus)) == 5 and r["steps"] == math.ceil((up - arm.gripper_raw_for_q(10.0)) / 5), r


def test_a_jaw_that_meets_the_fixed_jaw_stops_there_and_is_held_with_the_squeeze():
    """thin_squeeze.py: the evening's numbers come out of the simulated servo too (raw 2041, goal 2016, load
    about 110)."""
    tool = brush()
    arm, bus = rig(tool)
    r = TL.move_jaw(arm)                                   # close, to "closed", squeeze 25
    assert r["ok"] and r["met_fixed_jaw"] is True and not r["stopped_early"] and r["to"] == "closed", r
    assert r["pos_raw"] == FIXED == jaw_of(bus) and r["goal_raw"] == FIXED - 25 == int(bus.goal["gripper"]), r
    assert r["squeeze_ticks"] == 25 and 100 <= r["load"] <= 120 and r["temp_c"] == 25 and abs(r["q_deg"] + 12.2) < 0.01, r
    # the walk: 15 ticks a step far from the fixed jaw, 10 near it; the jaw never went past the fixed jaw
    steps = check_walk(bus, max_ahead=TL.JAW_LAG_TICKS + TL.JAW_NEAR_STEP_TICKS)
    assert {s[3] for s in steps if s[1] > TOUCH + TL.JAW_NEAR_TICKS} == {15} and {s[3] for s in steps if s[1] < TOUCH + 40} == {10}
    assert min(pos for _, _, pos in bus.goals) >= FIXED - 0.01 and r["worst_lag_ticks"] <= TL.JAW_LAG_TICKS + 10, r
    assert bus.goals[-1][1] == FIXED - 25 and min(g for _, g, _ in bus.goals) >= arm.raw_limits["gripper"][0]
    bus.sleep(5.0)
    assert jaw_of(bus) == FIXED and bus.goal["gripper"] == FIXED - 25, "the hold did not stay"
    # the file's facts
    assert (tool.gripper_raw, tool.gripper_goal_raw) == (FIXED, FIXED - 25) and r["record"]["gripper_goal_raw"] == FIXED - 25
    TL.Tool.from_dict(tool.to_dict())
    st = TL.jaw_state(arm)
    assert st["squeeze_on"] and st["squeeze_ticks"] == 25 and st["from_record_ticks"] == 0 and st["torque"], st
    # other squeezes: lighter is allowed, none at all too (then nothing holds the jaw and no hold is recorded)
    for sq in (10, 40, 0):
        tool = brush()
        arm, bus = rig(tool)
        r = TL.move_jaw(arm, "close", "closed", squeeze_ticks=sq)
        assert r["ok"] and r["met_fixed_jaw"] and r["pos_raw"] == FIXED and r["squeeze_ticks"] == sq, (sq, r)
        assert int(bus.goal["gripper"]) == FIXED - sq, (sq, bus.goal["gripper"])
        assert tool.gripper_goal_raw == (FIXED - sq if sq else None), (sq, tool.gripper_goal_raw)
    # more than the harness lets a grasp hold is refused before anything is written
    arm, bus = rig(brush())
    before = traffic(bus)
    refused(lambda: TL.move_jaw(arm, "close", "closed", squeeze_ticks=41), "squeeze of 41", "40", "Nothing moved")
    refused(lambda: TL.move_jaw(arm, "close", "closed", squeeze_ticks=-3), "squeeze")
    cfg = load_config()
    cfg["squeeze_ticks"] = 20                             # a rig whose limit is lower: its limit counts, and is the default
    arm2, bus2 = rig(brush(), cfg=cfg)
    refused(lambda: TL.move_jaw(arm2, "close", "closed", squeeze_ticks=25), "20 ticks")
    assert traffic(bus) == before and TL.jaw_plan(cfg, brush())["squeeze_ticks"] == 20
    r = TL.move_jaw(arm2)
    assert r["ok"] and r["squeeze_ticks"] == 20 and int(bus2.goal["gripper"]) == FIXED - 20, r
    # tape between the jaws: met a little above where bare fingertips touch, still the fixed jaw
    arm, bus = rig(brush(), fixed_jaw=TOUCH + 20)
    r = TL.move_jaw(arm)
    assert r["ok"] and r["met_fixed_jaw"] and r["pos_raw"] == TOUCH + 20 and r["goal_raw"] == TOUCH - 5, r
    # `open` to the very angle a held jaw stands at: no step to walk, but the squeeze goes off, in the servo and on record
    tool = held(gripper_raw=TOUCH, gripper_goal_raw=TOUCH - 25)
    arm, bus = rig(tool, fixed_jaw=TOUCH, torque=True)
    assert TL.hold_jaw(arm)["wrote"] and int(bus.goal["gripper"]) == TOUCH - 25
    r = TL.move_jaw(arm, "open", TL.jaw_q_deg(CFG, TOUCH))
    assert r["ok"] and r["steps"] == 0 and r["pos_raw"] == TOUCH and int(bus.goal["gripper"]) == TOUCH and r["squeeze_ticks"] == 0, r
    assert tool.gripper_goal_raw is None and tool.tip_measured is not None and r["record"]["tool_moved"] is False, tool


def test_a_jaw_blocked_early_stops_with_a_clear_result_and_no_squeeze():
    """Something in the way (the tool's handle against the arm, an object between the jaws after all)."""
    tool = brush(tip_measured={"at": "x"})
    arm, bus = rig(tool)
    stop = 2130                                           # 80 ticks before the fingertips would touch
    bus.blocker = lambda pos: pos["gripper"] < stop
    r = TL.move_jaw(arm)
    assert r["ok"] is False and r["stopped_early"] is True and r["met_fixed_jaw"] is False, r
    assert r["pos_raw"] == stop == jaw_of(bus) and "something is in its way" in r["why"] and "not held" in r["why"], r
    assert f"raw {stop}" in r["why"] and f"{stop - TOUCH} ticks before" in r["why"], r["why"]
    # the push: never more than the stopping lag plus a step ahead of the jaw, and none left at the end
    steps = check_walk(bus, max_ahead=TL.JAW_LAG_TICKS + TL.JAW_STEP_TICKS)
    assert max(s[2] for s in steps) > TL.JAW_LAG_TICKS, "the walk stopped before the jaw was behind"
    assert int(bus.goal["gripper"]) == stop and r["goal_raw"] == stop and r["squeeze_ticks"] == 0 and abs(r["load"]) <= 16, r
    assert abs(bus.goal["gripper"] - bus.pos["gripper"]) <= 1.0 <= int(arm._c("squeeze_ticks"))
    bus.sleep(3.0)
    assert jaw_of(bus) == stop
    # the tool went as far as the jaw went: the file says so, there is no hold, and the tip is no longer measured
    assert (tool.gripper_raw, tool.gripper_goal_raw, tool.tip_measured) == (stop, None, None) and r["record"]["tool_moved"]
    # the same on the way to an angle, and when opening
    for direction, to, block in (("close", -5.0, lambda pos: pos["gripper"] < 2150), ("open", 30.0, lambda pos: pos["gripper"] > 2300)):
        tool = brush()
        arm, bus = rig(tool)
        bus.blocker = block
        r = TL.move_jaw(arm, direction, to)
        at = 2150 if direction == "close" else 2300
        assert r["ok"] is False and r["stopped_early"] and r["pos_raw"] == at and "something is in its way" in r["why"], r
        assert int(bus.goal["gripper"]) == at, (direction, bus.goal["gripper"])
        assert tool.gripper_raw == at and tool.gripper_goal_raw is None, (direction, tool)
        check_walk(bus, max_ahead=TL.JAW_LAG_TICKS + TL.JAW_STEP_TICKS)
    # a jaw that is stopped a hair short of an angle does not pretend to be there
    tool = brush()
    arm, bus = rig(tool)
    bus.blocker = lambda pos: pos["gripper"] < 2144       # 21 ticks before -5 deg (raw 2123): less than the stopping lag
    r = TL.move_jaw(arm, "close", -5.0)
    assert r["ok"] is False and r["stopped_early"] and r["pos_raw"] == 2144 and int(bus.goal["gripper"]) == 2144, r


def test_lag_larger_than_the_limit_stops_the_stepping():
    """A jaw that moves but falls behind (a servo that is weak, a tool that drags): the walk does not run away
    from it."""
    def wide():      # the brush as it was first taped: the jaw 35 deg open, 396 ticks to walk to 0 deg
        return brush(gripper_raw=2576, tip_offset=[-0.0905, -0.0097, 0.0611], mount="other")

    tool = wide()
    arm, bus = rig(tool)
    target = arm.gripper_raw_for_q(0.0)
    bus.jaw_speed = 62.5                                  # half of the 125 ticks/s the goal walks at
    r = TL.move_jaw(arm, "close", 0.0)
    assert r["ok"] is False and r["stopped_early"] and "stopped following" in r["why"] and "push was taken off" in r["why"], r
    steps = check_walk(bus, max_ahead=TL.JAW_LAG_MAX_TICKS + TL.JAW_STEP_TICKS)
    assert TL.JAW_LAG_MAX_TICKS < r["worst_lag_ticks"] <= TL.JAW_LAG_MAX_TICKS + TL.JAW_STEP_TICKS, r
    assert len(steps) < math.ceil((2576 - target) / 15) - 5, "the walk went on to its end"
    assert abs(bus.goal["gripper"] - bus.pos["gripper"]) <= 1.0 and r["pos_raw"] > target + 100 and r["pos_raw"] == jaw_of(bus), r
    assert tool.gripper_raw == r["pos_raw"] and tool.gripper_goal_raw is None
    # a jaw that hardly moves at all counts as standing: it is stopped at the lower limit, sooner
    tool = wide()
    arm, bus = rig(tool)
    bus.jaw_speed = 20.0                                  # 2.4 ticks a step
    r = TL.move_jaw(arm, "close", 0.0)
    assert r["ok"] is False and r["stopped_early"] and TL.JAW_LAG_TICKS < r["worst_lag_ticks"] <= TL.JAW_LAG_TICKS + TL.JAW_STEP_TICKS, r
    assert len(check_walk(bus, max_ahead=TL.JAW_LAG_TICKS + TL.JAW_STEP_TICKS)) <= 4
    # the limit can be set lower (never above 40, never below 10)
    tool = wide()
    arm, bus = rig(tool)
    bus.jaw_speed = 20.0
    r = TL.move_jaw(arm, "close", 0.0, lag_ticks=12)
    assert r["stopped_early"] and 12 < r["worst_lag_ticks"] <= 12 + TL.JAW_STEP_TICKS, r
    # a jaw that keeps up with a slower walk is not stopped, and a walk short enough for it to catch up arrives
    tool = wide()
    arm, bus = rig(tool)
    bus.jaw_speed = 62.5
    r = TL.move_jaw(arm, "close", 0.0, step_ticks=7)
    assert r["ok"] and r["pos_raw"] == target and r["worst_lag_ticks"] <= 2, r
    tool = brush()
    arm, bus = rig(tool)
    bus.jaw_speed = 62.5
    r = TL.move_jaw(arm, "close", -8.0)                   # 93 ticks: it is 41 behind at the end of the walk and arrives
    assert r["ok"] and abs(r["pos_raw"] - arm.gripper_raw_for_q(-8.0)) <= 12 and 30 < r["worst_lag_ticks"] <= TL.JAW_LAG_MAX_TICKS, r


def test_the_fixed_jaw_that_is_not_there_is_not_squeezed_at():
    """A jaw that closes past where the fingertips should touch (a wrong gripper_touch_raw, a broken jaw): the
    search ends, no squeeze, and the result names the config key."""
    tool = brush()
    arm, bus = rig(tool, fixed_jaw=None)
    r = TL.move_jaw(arm)
    lo = arm.raw_limits["gripper"][0]
    assert r["ok"] is False and r["met_fixed_jaw"] is False and not r["stopped_early"], r
    assert "gripper_touch_raw" in r["why"] and "without a squeeze" in r["why"], r["why"]
    assert r["pos_raw"] == jaw_of(bus) >= lo and min(g for _, g, _ in bus.goals) >= max(lo, TOUCH - TL.JAW_BELOW_TOUCH_TICKS)
    assert abs(bus.goal["gripper"] - bus.pos["gripper"]) <= 1.0 and tool.gripper_goal_raw is None and r["squeeze_ticks"] == 0


def test_a_fault_or_an_interrupt_on_the_way_takes_the_push_off():
    # the gripper servo reports overload while the jaw is walked
    tool = brush(tip_measured={"at": "x"})
    arm, bus = rig(tool)
    bus.fault_below = 2120
    try:
        TL.move_jaw(arm)
        raise AssertionError("a servo fault did not stop the jaw")
    except MotionAborted as e:
        assert e.info["reason"] == "servo_fault" and e.info["faults"] == {"gripper": 32} and "overload" in str(e), e
    assert abs(bus.goal["gripper"] - bus.pos["gripper"]) <= 1.0, "a push was left on a faulted servo"
    assert tool.gripper_raw == jaw_of(bus) < 2125 and tool.gripper_goal_raw is None and tool.tip_measured is None
    # Ctrl-C / a cancelled tool call in the middle of the walk
    tool = brush()
    arm, bus = rig(tool)
    bus.interrupt_at = 6
    try:
        TL.move_jaw(arm)
        raise AssertionError("not interrupted")
    except KeyboardInterrupt:
        pass
    assert abs(bus.goal["gripper"] - bus.pos["gripper"]) <= 1.0 and AT_ZERO - 90 < jaw_of(bus) < AT_ZERO - 30
    assert tool.gripper_raw == jaw_of(bus) and tool.gripper_goal_raw is None
    # an interrupt while a jaw that was already held is closed again: the hold on record stays for hold_jaw
    tool = held()
    arm, bus = rig(tool, torque=True)
    bus.interrupt_at = 2
    try:
        TL.move_jaw(arm)
        raise AssertionError("not interrupted")
    except KeyboardInterrupt:
        pass
    assert jaw_of(bus) == FIXED and int(bus.goal["gripper"]) == FIXED and tool.gripper_goal_raw == FIXED - 25 and tool.tip_measured
    assert TL.hold_jaw(arm)["wrote"] and int(bus.goal["gripper"]) == FIXED - 25
    # a servo that is too hot: refused before anything is written
    tool = brush()
    arm, bus = rig(tool)
    bus.temp["gripper"] = 70
    before = traffic(bus)
    refused(lambda: TL.move_jaw(arm), "temperature")
    assert traffic(bus)[1:] == before[1:] and bus.torque["gripper"] == 0 and tool.gripper_raw == AT_ZERO


def test_a_bus_that_dies_in_the_middle_of_a_jaw_move_leaves_the_tool_file_saying_so():
    """Review finding 8 (2026-11-13). The servo bus stops answering after six goal writes of `fd tool jaw close`.
    The jaw then stands 75 ticks from where the tool file has it (the tip about 1.5 cm from the file's), nothing
    can be read or released, and the file went on saying "jaw at raw 2182, tip measured". Now it says what is
    true: the jaw's position is not known, no hold, the tip not measured, and a note why."""
    contact = {"x0": 0.2225, "y0": -0.019, "a": 0.00542, "bx": 0.0362, "by": -0.0062, "rms_mm": 0.8, "points": []}

    def measured():
        return brush(tip_measured={"at": "2026-11-13T16:50:00", "rms_px": 7.7}, contact=dict(contact),
                     calibrated={"contact": {"at": "x"}})

    # tooling.move_jaw: the error goes on to the caller, with what was found on the way out
    tool = measured()
    arm, bus = rig(tool)
    bus.die_after_goals = 6
    try:
        TL.move_jaw(arm)
        raise AssertionError("a dead bus did not end the jaw move")
    except BusError as e:
        left = e.jaw_left
    really = jaw_of(bus)
    assert bus.dead and 60 <= AT_ZERO - really <= 90 and len(bus.goals) == 6, (really, bus.goals)      # 75 ticks on its way
    assert left["pos_raw"] is None and left["released"] is False and left["record"]["position_known"] is False, left
    assert set(left["record"]["dropped"]) == {"tip_measured", "contact", "calibrated"}, left
    assert (tool.gripper_raw, tool.gripper_goal_raw, tool.tip_measured, tool.contact, tool.calibrated) == (None, None, None, None, {})
    assert tool.tip_offset == TIP_AT_ZERO and tool.on_jaw == "moving_outer"          # (nothing better to put there; the mount stays)
    note = tool.notes[-1]
    assert "NOT known" in note and "servo bus stopped answering" in note and f"the tool file had it at raw {AT_ZERO}" in note, note
    assert "NOT measured" in note and "fd tool jaw close" in note and "fd calib-air --tip-only" in note, note
    seen = left["record"]["last_seen_raw"]
    assert seen is not None and abs(seen - really) <= 2 * TL.JAW_STEP_TICKS and f"last read at raw {seen}" in note, (seen, really)
    TL.Tool.from_dict(tool.to_dict())                                                # ... and that is a file that can be stored
    # such a tool has no jaw position left to compare or to hold: no check, no squeeze put back, nothing written
    assert TL.jaw_tolerance(tool) is None and TL.jaw_estimate(CFG, tool, FIXED) is None
    # a bus that is dead before the first goal: nothing was commanded, and the tool says what it said
    tool = measured()
    arm, bus = rig(tool)
    was = tool.to_dict()
    bus.dead = True
    try:
        TL.move_jaw(arm)
        raise AssertionError("a dead bus did not end the jaw move")
    except BusError as e:
        assert not hasattr(e, "jaw_left")
    assert tool.to_dict() == was and not bus.goals
    # writes fail, reads still answer: the position is known, so the file gets it (and the result says that the
    # push could not be taken off)
    tool = measured()
    arm, bus = rig(tool)
    bus.die_after_goals, bus.writes_only = 6, True
    try:
        TL.move_jaw(arm)
        raise AssertionError("a bus that takes no writes did not end the jaw move")
    except BusError as e:
        left = e.jaw_left
    assert left["pos_raw"] == jaw_of(bus) and left["released"] is False and left["record"]["tool_moved"] is True, left
    assert (tool.gripper_raw, tool.gripper_goal_raw, tool.tip_measured) == (jaw_of(bus), None, None) and "ESTIMATE" in tool.notes[-1]
    # the bus dies at the very end, after the jaw has met the fixed jaw and the squeeze is on (at the look at the goal
    # for the report): the jaw stands 141 ticks from the file's record, and the file must not go on saying otherwise
    tool = measured()
    arm, bus = rig(tool)
    bus.die_reading = "Goal_Position"
    try:
        TL.move_jaw(arm)
        raise AssertionError("a dead bus did not end the jaw move")
    except BusError as e:
        left = getattr(e, "jaw_left", None)
    assert jaw_of(bus) == FIXED and int(bus.goal["gripper"]) == FIXED - 25, "the walk had not finished when the bus died"
    assert left is not None and left["pos_raw"] is None, left
    assert (tool.gripper_raw, tool.gripper_goal_raw, tool.tip_measured) == (None, None, None) and "NOT known" in tool.notes[-1], tool

    # the command: a complete result, and the FILE says it
    with tool_file(measured()):
        arm, bus = rig(TL.active())
        text = open(TOOL_PATH).read()
        bus.die_after_goals = 6
        r = house(cli(["--sim", "close"], arm), ok=False, stage="hardware")
        assert r["error"] == "BusError" and r["position_known"] is False and r["saved"] is True and r["jaw"] is None, r
        assert "NOT known" in r["reason"] and "could not be taken off" in r["reason"] and "no answer from servo 6" in r["reason"], r
        assert all(t in r["next"] for t in ("fd state", "fd tool jaw status", "fd tool jaw close", "fd calib-air --tip-only")), r["next"]
        assert "分かりません" in r["say"] and "未測定" in r["say"] and r["recorded"]["gripper_raw"] is None, r
        assert r["recorded"]["tip_measured"] is False and r["file"] == TOOL_PATH and r["tool"] == "brush_thin", r
        really = jaw_of(bus)
        now = TL.active()
        assert open(TOOL_PATH).read() != text, "the tool file still says that the jaw is where it was"
        assert (now.gripper_raw, now.gripper_goal_raw, now.tip_measured, now.contact) == (None, None, None, None), now
        assert now.tip_offset == TIP_AT_ZERO and "NOT known" in now.notes[-1] and len(now.notes) == 1
        # the servos answer again: status shows a jaw nobody has on record; closing finds the fixed jaw and records
        # it; the tip stays the old one, said to be wrong, until it is measured
        arm, bus = rig(TL.active(), jaw=really)
        r = house(cli(["--sim", "status"], arm), ok=True, stage="status")
        assert r["recorded"]["gripper_raw"] is None and r["jaw"]["pos_raw"] == really and r["jaw"]["from_record_ticks"] is None, r
        assert r["in_place"] is None and r["recorded"]["tip_measured"] is False, r
        r = house(cli(["--sim", "close"], arm), ok=True, stage="close")
        now = TL.active()
        assert (now.gripper_raw, now.gripper_goal_raw, now.tip_measured) == (FIXED, FIXED - 25, None) and r["tool_moved"], (r, now)
        assert r["estimate"] is None and now.tip_offset == TIP_AT_ZERO and "OLD tip offset" in r["next"], r["next"]
        assert "fd calib-air --tip-only" in r["next"] and "NOT corrected" in now.notes[-1], now.notes
    # a jaw that was held, closed again after the torque was off, when the bus dies: the hold must not stay on record
    with tool_file(held()):
        arm, bus = rig(TL.active(), jaw=FIXED + 30)
        bus.die_after_goals = 2
        r = house(cli(["--sim", "close"], arm), ok=False, stage="hardware")
        now = TL.active()
        assert r["position_known"] is False and (now.gripper_raw, now.gripper_goal_raw, now.tip_measured) == (None, None, None), (r, now)
        assert TL.hold_jaw(rig(now, jaw=FIXED + 12, torque=True)[0])["held"] is False        # nothing left to put back
    # the bus dies before the command has commanded anything (at the look at the jaw): a result, the file untouched
    with tool_file(measured()):
        arm, bus = rig(TL.active())
        text = open(TOOL_PATH).read()
        bus.dead = True
        r = house(cli(["--sim", "close"], arm), ok=False, stage="hardware")
        assert r["position_known"] is None and r["saved"] is False and "did not move the jaw" in r["reason"], r
        assert "fd state" in r["next"] and "fd tool jaw close" in r["next"] and open(TOOL_PATH).read() == text and not bus.goals
        # ... and with writes failing only: the jaw was read afterwards, the file has it
        arm, bus = rig(TL.active())
        bus.die_after_goals, bus.writes_only = 6, True
        r = house(cli(["--sim", "close"], arm), ok=False, stage="hardware")
        now = TL.active()
        assert r["position_known"] is True and f"raw {jaw_of(bus)}" in r["reason"] and "could NOT be taken off" in r["reason"], r
        assert now.gripper_raw == jaw_of(bus) and now.gripper_goal_raw is None and now.tip_measured is None and r["saved"], now


# ----------------------------------------------------------------------------- who may, and where
def test_refusals_for_a_tool_between_the_jaws_and_for_one_that_does_not_say():
    for where, texts in (("between", ("BETWEEN the jaws", "crush", "drop", "fd tool jaw mount moving_outer")),
                         (None, ("does not say on which jaw", "crushed", "dropped", "fd tool jaw mount moving_outer",
                                 "fd tool jaw mount fixed", "fd tool jaw mount between", "photo")),
                         ("fixed", ("FIXED jaw", "fd tool jaw mount moving_outer"))):
        tool = brush(on_jaw=where, tip_measured={"at": "x"})
        arm, bus = rig(tool, torque=True)
        before, was = traffic(bus), tool.to_dict()
        for call in (lambda: TL.move_jaw(arm), lambda: TL.move_jaw(arm, "close", 0.0), lambda: TL.move_jaw(arm, "open", 30.0),
                     lambda: TL.jaw_plan(CFG, tool), lambda: arm.jaw_goal(AT_ZERO - 5)):
            refused(call, "brush_thin")
            assert traffic(bus) == before, f"{where}: bus traffic before the refusal"
        refused(lambda: TL.move_jaw(arm), *texts, "Nothing moved")
        assert tool.to_dict() == was and jaw_of(bus) == AT_ZERO
        assert TL.jaw_locked(tool) == TL.jaw_locked(tool, "moving the jaw") and where != "moving_outer"
        # the command: a complete result, the file untouched, the arm not even opened
        with tool_file(tool):
            text = open(TOOL_PATH).read()
            for words in (["close"], ["close", "--to", "0"], ["close", "--dry"], ["open", "--to", "20"]):
                opened = []
                real = tooljaw.make_arm
                tooljaw.make_arm = lambda sim: opened.append(sim) or (_ for _ in ()).throw(AssertionError("the arm was opened"))
                try:
                    r = tooljaw.main(["--sim"] + words)
                finally:
                    tooljaw.make_arm = real
                house(r, ok=False, stage="tool")
                assert r["error"] == "SafetyError" and r["tool"] == "brush_thin" and r["on_jaw"] == where and not opened, r
                assert texts[0] in r["reason"] and "fd tool jaw mount moving_outer" in r["next"] and "brush_thin" in r["say"], r
                assert open(TOOL_PATH).read() == text
    assert TL.jaw_locked(brush()) is None and "no tool is attached" in TL.jaw_locked(None)
    with tool_file(None):                                   # nothing registered: attach is the next step
        for words in (["close"], ["open", "--to", "10"], ["mount", "moving_outer"], ["close", "--dry"]):
            r = house(tooljaw.main(["--sim"] + words), ok=False, stage="tool")
            assert "fd tool attach" in r["next"], r
        r = house(tooljaw.main(["--sim", "status"]), ok=True, stage="status")
        assert r["attached"] is False


def test_the_tool_must_be_clear_of_the_table_while_it_swings():
    """The tool swings with the jaw while the arm stands still, towards the side that points down."""
    # at the stand-by pose (tip 12 cm up) there is room
    tool = brush()
    arm, bus = rig(tool)
    sw = TL.jaw_swing(CFG, tool, arm.read_q(), AT_ZERO, TOUCH)
    assert 110 < sw["tip_height_mm"]["now"] < 125 and sw["tip_height_mm"]["lowest"] > 80 and sw["turn_deg"] < -11, sw
    assert sw["tip_height_mm"]["end"] < sw["tip_height_mm"]["now"] - 10, "closing swings the tip DOWN here"
    # the arm low over the paper: closing would put the tip into the table
    tool = brush()
    arm, bus = rig(tool)
    q_low = TL.work_q(arm, tool, (0.25, 0.0), 0.022)[0]
    place(arm, q_low, gripper=AT_ZERO)
    before = traffic(bus)
    msg = refused(lambda: TL.move_jaw(arm), "would swing the tip", "above the table", "Nothing moved", "fd ready",
                  "the same jaw command (close) again")
    assert "`fd tool jaw" not in msg, msg       # (in words: this text does not know the options the command was typed with)
    assert traffic(bus)[1:] == before[1:] and jaw_of(bus) == AT_ZERO and tool.gripper_raw == AT_ZERO and bus.torque["gripper"] == 0, msg
    # ... and with the tip already under 2 cm the jaw is not moved at all, whichever way
    place(arm, TL.work_q(arm, tool, (0.25, 0.0), 0.010)[0], gripper=AT_ZERO)
    for direction, to in (("close", "closed"), ("open", 20.0)):
        refused(lambda: TL.move_jaw(arm, direction, to), "is not moved with the tool that low", "fd ready")
    assert jaw_of(bus) == AT_ZERO and bus.torque["gripper"] == 0
    # the command says the same, as a result, for the dry run and for the real one
    with tool_file(tool):
        for words in (["close", "--dry"], ["close"]):
            r = house(cli(["--sim"] + words, arm), ok=False, stage="plan")
            assert "above the table" in r["reason"] and "fd ready" in r["next"] and r["jaw"]["pos_raw"] == AT_ZERO, r
        assert TL.active().gripper_raw == AT_ZERO and jaw_of(bus) == AT_ZERO
    # the brush as it was first taped (35 deg open, 9 cm aside) on an arm folded in its rest pose: no room to close there.
    # Every simulated call starts folded like that; --at-ready stands the simulated arm where `fd ready` leaves the real one
    wide = brush(gripper_raw=2576, tip_offset=[-0.0905, -0.0097, 0.0611], mount="other")
    with tool_file(wide):
        arm, bus = rig(TL.active(), pose="rest")
        q_rest = arm.read_q()
        for words in (["close", "--dry"], ["close"]):
            r = house(cli(["--sim"] + words, arm), ok=False, stage="plan")
            assert "would swing the tip" in r["reason"] and "--at-ready" in r["next"] and "fd ready" in r["next"], r
        assert jaw_of(bus) == 2576 and np.allclose(arm.read_q(), q_rest) and bus.torque["gripper"] == 0
        r = house(cli(["--sim", "close", "--dry", "--at-ready"], arm), ok=True, stage="planned")
        assert r["swing"]["tip_height_mm"]["now"] > 110 and r["swing"]["tip_height_mm"]["lowest"] > 40, r["swing"]
        r = house(cli(["--sim", "close", "--at-ready"], arm), ok=True, stage="close")
        assert r["met_fixed_jaw"] and jaw_of(bus) == FIXED and TL.active().gripper_goal_raw == FIXED - 25, r
        assert np.allclose(arm.read_q(), TL.ready_q(arm, wide), atol=1e-3), "the simulated arm was not stood at the stand-by pose"
        assert not any(bus.torque[j] for j in ARM), "the arm joints were switched on"
        # on the real arm it is `fd ready` that does it: the option is refused before the arm is opened
        del _touched[:]
        r = house(tooljaw.main(["close", "--at-ready"]), ok=False, stage="options")
        assert "fd ready" in r["reason"] and not _touched, r
    # opening at the stand-by pose swings the tool up and out: allowed
    tool = brush()
    arm, bus = rig(tool)
    assert TL.jaw_swing(CFG, tool, arm.read_q(), AT_ZERO, arm.gripper_raw_for_q(35.0))["tip_height_mm"]["lowest"] > 100
    # a tool file that does not say where the jaw stood for its tip: "where it stands now"
    blank = brush(gripper_raw=None)
    a = TL.jaw_swing(CFG, blank, arm.read_q(), AT_ZERO, TOUCH)
    assert a == TL.jaw_swing(CFG, brush(), arm.read_q(), AT_ZERO, TOUCH)


# ----------------------------------------------------------------------------- what a jaw move does to the tool
def test_after_close_the_tip_is_unmeasured_and_the_pen_calibration_dropped():
    contact = {"x0": 0.2225, "y0": -0.019, "a": 0.00542, "bx": 0.0362, "by": -0.0062, "rms_mm": 0.8, "points": []}
    corr = {"centre": [0.22, 0.0], "kind": "affine", "coef": [[0, 0], [1, 0], [0, 1]], "rms_mm": 1.0, "max_mm": 2.0, "n": 9}
    tool = brush(tip_measured={"at": "2026-11-13T16:50:00", "rms_px": 7.7}, contact=dict(contact), corr_map=dict(corr),
                 calibrated={"tip": {"at": "x"}, "contact": {"at": "x"}, "map": {"at": "x"}}, region=[0.15, -0.13, 0.37, 0.03])
    with tool_file(tool):
        arm, bus = rig(TL.active())
        r = house(cli(["--sim", "close"], arm), ok=True, stage="close")
        now = TL.active()
        # the jaw record ...
        assert (now.gripper_raw, now.gripper_goal_raw, now.on_jaw) == (FIXED, FIXED - 25, "moving_outer"), now
        # ... the tip: the jaw model's estimate (the evening's numbers), and nothing that was measured with the old one
        assert np.allclose(np.array(now.tip_offset) * 1000.0, [23.4, -2.5, 52.3], atol=0.06), now.tip_offset
        assert now.tip_measured is None and now.contact is None and now.corr_map is None and now.calibrated == {}
        # (the region and the joint limits stay, as `fd tool set --tip` leaves them)
        assert now.region == [0.15, -0.13, 0.37, 0.03] and now.joint_limits_deg == BRUSH_LIMITS
        assert TL.calibration_state(now) == {"tip": False, "contact": False, "map": False}
        assert "ESTIMATE" in now.notes[-1] and "fd calib-air --tip-only" in now.notes[-1] and "[-5.1, -2.5, 60.4]" in now.notes[-1]
        # what the operator is told
        assert r["tool_moved"] and r["moved"] and r["saved"] and r["tip_measured"] is False, r
        assert r["met_fixed_jaw"] and not r["stopped_early"], r
        assert set(r["dropped"]) == {"tip_measured", "contact", "map", "calibrated"} and r["tip_offset_mm"] == [23.4, -2.5, 52.3], r
        assert r["estimate"]["tip_offset_mm"] == [23.4, -2.5, 52.3] and r["estimate"]["expect_error_mm"] == 10.0, r["estimate"]
        assert "fd calib-air --tip-only" in r["next"] and "(23, -2, 52) mm" in r["next"] and "10 mm off" in r["next"], r["next"]
        assert "about the jaw pivot" in r["next"] and "-12.4" in r["next"].replace("-12.39", "-12.4"), r["next"]
        assert all(t in r["say"] for t in ("fd calib-air --tip-only", "1 cm", "raw 2041", "25 ticks")), r["say"]
        assert r["jaw"]["pos_raw"] == FIXED and 100 <= r["jaw"]["load"] <= 120, r["jaw"]
        assert r["jaw"]["temp_c"] == 25 and r["jaw"]["squeeze_on"], r["jaw"]
        assert r["recorded"]["gripper_goal_raw"] == FIXED - 25 and r["recorded"]["tip_measured"] is False, r["recorded"]
        # the pen commands see an uncalibrated tool, as after any changed tip
        import penskill
        gate = penskill._tool_gate(type("S", (), {"arm": arm})(), "draw", need=("contact", "map"))
        assert gate and gate["stage"] == "tool" and "not calibrated" in gate["reason"], gate


def test_closing_a_jaw_that_is_closed_keeps_the_tip_that_was_measured():
    """The squeeze is gone after `fd tool rest`, the jaw is where it was: closing it again moves no tool."""
    tool = held(contact={"x0": 0.2, "y0": 0.0, "a": 0.005, "rms_mm": 0.8, "points": []})
    with tool_file(tool):
        arm, bus = rig(TL.active(), jaw=FIXED - 1)             # (2040: where the limp jaw stood at 19:25)
        bus.blocker = lambda pos: pos["gripper"] < FIXED - 1
        # the dry run says it beforehand: the jaw is expected to end where the file has it, nothing swings
        r = house(cli(["--sim", "close", "--dry"], arm), ok=True, stage="planned")
        assert r["tool_would_move"] is False and r["plan"]["expect_raw"] == FIXED and r["estimate"]["tip_moves_mm"] == 0.0, r
        assert "does not move" in r["next"] and "squeeze is put on" in r["next"] and "そのまま使える" in r["say"], r
        assert abs(r["swing"]["turn_deg"]) <= 0.2 and "calib-air" not in r["say"], r       # (one tick: the limp jaw at 2040)
        r = house(cli(["--sim", "close"], arm), ok=True, stage="close")
        now = TL.active()
        assert r["tool_moved"] is False and r["tip_measured"] is True and r["dropped"] == [] and r["estimate"] is None, r
        assert now.tip_offset == TIP_CLOSED and now.tip_measured == tool.tip_measured and now.contact == tool.contact
        assert now.gripper_raw == FIXED and now.gripper_goal_raw == FIXED - 26 and int(bus.goal["gripper"]) == FIXED - 26, now
        assert "did not move" in r["next"] and "そのまま使えます" in r["say"] and "calib-air" not in r["say"], r
        assert r["jaw"]["squeeze_ticks"] == 25 and r["jaw"]["pos_raw"] == FIXED - 1, r["jaw"]
    # the rig's own file of that evening (no on_jaw yet): mount, close, and the measured tip is still the tip
    path = os.path.join(BACKUP, "tool_active.json")
    if os.path.exists(path):
        with tool_file(TL.load(path)):
            arm, bus = rig(TL.active(), jaw=2040)
            bus.blocker = lambda pos: pos["gripper"] < 2040
            house(cli(["--sim", "close"], arm), ok=False, stage="tool")
            house(cli(["--sim", "mount", "moving_outer"], arm), ok=True, stage="mount")
            r = house(cli(["--sim", "close", "--dry"], arm), ok=True, stage="planned")
            assert r["tool_would_move"] is False and r["plan"]["expect_raw"] == 2041, r     # (recorded beyond the touch position)
            r = house(cli(["--sim", "close"], arm), ok=True, stage="close")
            now = TL.active()
            assert not r["tool_moved"] and now.tip_offset == [0.016, -0.0011, 0.0558] and now.gripper_raw == 2041, (r, now)
            assert now.gripper_goal_raw == 2015 and now.joint_limits_deg == BRUSH_LIMITS and len(now.notes) == 3


# ----------------------------------------------------------------------------- the squeeze after the torque was off
def test_hold_jaw_restores_the_goal_and_refuses_a_moved_jaw():
    tool = held()
    arm, bus = rig(tool, pose="rest")                       # as `fd tool rest` leaves it: limp, the jaw where it was
    refused(lambda: TL.hold_jaw(arm), "gripper torque is off")
    assert traffic(bus)[1:] == (0, 0)
    arm.torque_on()
    bus.sleep(0.3)
    assert int(bus.goal["gripper"]) == FIXED, "after torque on the goal is where the jaw stands: the squeeze is lost"
    bus.goals.clear()
    r = TL.hold_jaw(arm, tool)
    assert r["held"] and r["wrote"] and r["goal_raw"] == FIXED - 25 and r["pos_raw"] == FIXED and r["squeeze_ticks"] == 25, r
    assert int(bus.goal["gripper"]) == FIXED - 25 and jaw_of(bus) == FIXED and 100 <= r["load"] <= 120 and r["from_record_ticks"] == 0
    steps = check_walk(bus, max_ahead=25)
    assert [g for _, g, _ in bus.goals] == [FIXED - 10, FIXED - 20, FIXED - 25], bus.goals
    del steps
    # again: nothing to do, nothing written
    writes = bus.io["writes"]
    r = TL.hold_jaw(arm)
    assert r["held"] and r["wrote"] is False and bus.io["writes"] == writes, r
    # the limp jaw stood a little open (the tape pulls): it is drawn back onto the fixed jaw
    arm, bus = rig(tool, jaw=FIXED + 30, torque=True)
    r = TL.hold_jaw(arm)
    assert r["wrote"] and jaw_of(bus) == FIXED and int(bus.goal["gripper"]) == FIXED - 25 and r["from_record_ticks"] == 0, r
    check_walk(bus, max_ahead=25 + TL.JAW_STEP_TICKS)
    # a jaw that stands elsewhere: the tool is no longer where its tip was measured. Nothing is written.
    for jaw in (AT_ZERO, FIXED + 41, 2576):
        arm, bus = rig(tool, jaw=jaw, torque=True)
        writes = bus.io["writes"]
        msg = refused(lambda: TL.hold_jaw(arm), "brush_thin", "no longer where its tip was measured", f"raw {jaw}", f"raw {FIXED}",
                      "Nothing was written", "fd tool jaw close", "fd calib-air --tip-only")
        assert bus.io["writes"] == writes and jaw_of(bus) == jaw and int(bus.goal["gripper"]) == jaw, msg
        refused(lambda: TL.jaw_in_place(arm), "no longer where its tip was measured")
        assert bus.io["writes"] == writes
    # something came between the jaws: the jaw does not come back, the push is taken off, and that is a refusal
    arm, bus = rig(tool, jaw=FIXED + 30, torque=True)
    bus.blocker = lambda pos: pos["gripper"] < FIXED + 30
    msg = refused(lambda: TL.hold_jaw(arm), "something keeps it open", "push was taken off", "fd tool jaw close")
    assert jaw_of(bus) == FIXED + 30 and int(bus.goal["gripper"]) == FIXED + 30, msg
    assert max(abs(g - pos) for _, g, pos in bus.goals) <= 45, "the goal ran too far ahead of a blocked jaw"
    # a jaw that closes further than it did (what it lay on is gone): not where the tip was measured either
    arm, bus = rig(tool, torque=True)
    bus.blocker = lambda pos: pos["gripper"] < FIXED - 15
    msg = refused(lambda: TL.hold_jaw(arm), "closes further than it did", "push was taken off")
    assert abs(bus.goal["gripper"] - bus.pos["gripper"]) <= 1.0, msg
    # a tool whose record is not a squeeze the harness allows is refused, not obeyed
    refused(lambda: TL.hold_jaw(rig(held(), torque=True)[0], held(gripper_goal_raw=FIXED - 50)), "squeeze of 50", "40")
    odd = held()
    odd.on_jaw = "between"
    refused(lambda: TL.hold_jaw(rig(held(), torque=True)[0], odd), "gripper_goal_raw")
    # no tool, a tool whose file has no jaw position, a tool on the fixed jaw: nothing, and not one bus transaction
    for t in (None, TL.Tool(name="pen", tip_offset=[0.0, 0.0, 0.06]), brush(on_jaw="fixed"), brush(gripper_raw=None)):
        arm, bus = rig(brush(), torque=True)
        arm.tool = t
        before = traffic(bus)
        assert TL.hold_jaw(arm) == {"held": False, "wrote": False, "why": "no jaw hold is recorded for the tool on the gripper"}
        assert TL.jaw_in_place(arm) is None and TL.jaw_tolerance(t) is None and traffic(bus) == before
    # a tool without a hold that records where its jaw stood: compared with it (reads only), never written to
    arm, bus = rig(brush(), torque=True)
    before = traffic(bus)
    assert TL.hold_jaw(arm) == {"held": False, "wrote": False, "why": "no jaw hold is recorded for the tool on the gripper",
                                "pos_raw": AT_ZERO, "from_record_ticks": 0}
    assert TL.jaw_in_place(arm) == {"pos_raw": AT_ZERO, "from_record_ticks": 0, "tol_ticks": TL.JAW_END_TOL_TICKS, "held": False}
    assert traffic(bus) == (before[0] + 2, before[1], before[2])


def test_power_puts_the_squeeze_back_and_ready_refuses_a_moved_jaw():
    """tooling._power is what `fd ready`, `fd tool rest` and the air calibration switch the arm on with."""
    tool = held()
    arm, bus = rig(tool, pose="rest")
    r = TL.go_ready(arm)
    assert r["jaw"]["held"] and r["jaw"]["wrote"] and int(bus.goal["gripper"]) == FIXED - 25 and jaw_of(bus) == FIXED, r["jaw"]
    assert np.linalg.norm(K.fk(arm.read_q(), arm.tip_tool())[:3, 3] - np.array(r["standby_xyz"])) < 0.004
    sp = TL._power(arm, None)                                 # already on, already held: speeds as ever, plus the report
    assert sp["jaw"] == TL.hold_jaw(arm) and sp["jaw"]["wrote"] is False and sp["transit"] == arm.speeds(None)["transit"]
    assert "jaw" not in arm.speeds(None) and "jaw" not in arm._active, "the tier's own record was changed"
    r = TL.rest(arm)                                          # fold away: the torque goes off, the squeeze with it
    assert r["ok"] and r["torque"] == "off" and r["jaw"]["held"] and jaw_of(bus) == FIXED and bus.torque["gripper"] == 0, r
    # the jaw was moved by hand while the arm lay limp: `ready` refuses before the torque goes on ...
    bus.pos["gripper"] = bus.goal["gripper"] = float(AT_ZERO)
    before, pos = traffic(bus)[1:], dict(bus.pos)
    msg = refused(lambda: TL.go_ready(arm), "no longer where its tip was measured", "fd tool jaw close", "fd calib-air --tip-only")
    assert traffic(bus)[1:] == before and bus.pos == pos and not any(bus.torque.values()), msg
    buf = io.StringIO()
    with tool_file(tool), contextlib.redirect_stdout(buf):
        real = fd.make_arm
        fd.make_arm = lambda sim: arm
        try:
            rc = fd.main(["--sim", "ready"])
        finally:
            fd.make_arm = real
    res = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert rc == 1 and res["ok"] is False and "fd tool jaw close" in res["message"] and bus.pos == pos, res
    # ... while folding the arm away is never held up by it: the result says what is wrong with the jaw
    place(arm, TL.ready_q(arm, tool), gripper=AT_ZERO)
    arm.torque_on()
    bus.sleep(0.6)
    r = TL.rest(arm)
    assert r["ok"] and r["torque"] == "off" and r["jaw"]["held"] is False and "no longer where its tip" in r["jaw"]["error"], r
    assert jaw_of(bus) == AT_ZERO, "the jaw of a tool that is elsewhere was commanded"
    # a tool without a hold: the speeds dict is the arm's own, and the gripper sees no goal but the hold of torque on
    plain = brush()
    arm, bus = rig(plain, pose="rest")
    r = TL.go_ready(arm)
    assert "jaw" not in r and [g for _, g, _ in bus.goals] == [AT_ZERO], bus.goals
    assert TL._power(arm, None) is arm._active


def test_a_jaw_without_a_recorded_hold_is_compared_with_its_record_too():
    """Review finding 49 (2026-11-13). The rig's own tool file of that evening has the jaw shut at raw 2041, the
    squeeze only in its notes and nothing about the jaw the brush is on. With the limp jaw 100 ticks (8.8 deg)
    from that record the tip is 21 mm from the file's by the jaw model; `ready` and the painting went on, because
    the jaw was compared only for a tool with a recorded hold. Now every tool that records a jaw position is
    compared, before the torque comes on, and the refusal names the `fd tool jaw` command that puts it right."""
    closed = dict(gripper_raw=FIXED, tip_offset=list(TIP_CLOSED), mount="other", tip_measured={"at": "2026-11-13T17:25:00"})
    tools = {"not said": brush(on_jaw=None, **closed), "moving_outer": brush(**closed), "between": brush(on_jaw="between", **closed)}
    rig_file = os.path.join(BACKUP, "tool_active.json")
    if os.path.exists(rig_file):
        tools["the rig's own file"] = TL.load(rig_file)
        assert (tools["the rig's own file"].gripper_raw, tools["the rig's own file"].on_jaw) == (FIXED, None)
    way_out = {"not said": ("does not say on which jaw", "fd tool jaw mount moving_outer", "fd tool jaw close", "fd tool jaw mount fixed",
                            "photo", "21 mm"),
               "moving_outer": ("no squeeze holds it", "fd tool jaw close", "fd calib-air --tip-only", "21 mm"),
               "between": ("BETWEEN the jaws", "fd tool attach with --replace", "`fd tool attach --help`", "fd tool jaw mount moving_outer")}
    way_out["the rig's own file"] = way_out["not said"]
    for label, tool in tools.items():
        # the tolerance of a jaw that no squeeze pulls back: what a held jaw must end within (8 ticks, 1.7 mm at this tip)
        assert TL.jaw_tolerance(tool) == TL.JAW_END_TOL_TICKS == 8 < TL.JAW_HOLD_TOL_TICKS, label
        for off in (0, 8, -8):                               # where the file has it: an answer, by one read
            arm, bus = rig(tool, jaw=FIXED + off, pose="rest")
            before = traffic(bus)
            assert TL.jaw_in_place(arm) == {"pos_raw": FIXED + off, "from_record_ticks": off, "tol_ticks": 8, "held": False}, label
            h = TL.hold_jaw(arm)                             # (nothing to hold, nothing to write: no torque needed)
            assert h["held"] is False and h["wrote"] is False and h["pos_raw"] == FIXED + off and h["from_record_ticks"] == off, h
            assert traffic(bus) == (before[0] + 2, before[1], before[2]) and not any(bus.torque.values()), label
        arm, bus = rig(tool, pose="rest")
        r = TL.go_ready(arm)                                 # ... and the arm stands by as it always did
        assert "jaw" not in r and [g for _, g, _ in bus.goals] == [FIXED], (label, bus.goals)
        for off in (100, 9, -9):                             # elsewhere: refused before the torque comes on
            arm, bus = rig(tool, jaw=FIXED + off, pose="rest")
            was, pos = traffic(bus)[1:], dict(bus.pos)
            for call in (lambda: TL.jaw_in_place(arm), lambda: TL.go_ready(arm), lambda: TL.hold_jaw(arm)):
                msg = refused(call, tool.name, "no longer where its tip was measured", f"raw {FIXED + off}", f"raw {FIXED}",
                              f"{abs(off)} ticks", "Nothing was written", "fd tool jaw status")
                assert traffic(bus)[1:] == was and bus.pos == pos and not any(bus.torque.values()), (label, off, msg)
            if off == 100:                                   # the reviewer's case: the way out, and how far the tip is off
                for text in way_out[label]:
                    assert text in msg, (label, text, msg)
            # what switches the arm on without refusing (the way to fold it away) says it, and commands no jaw
            place(arm, TL.ready_q(arm, tool), gripper=FIXED + off)
            arm.torque_on()
            bus.sleep(0.6)
            bus.goals.clear()
            sp = TL._power(arm, None)
            assert sp["jaw"]["held"] is False and "no longer where its tip was measured" in sp["jaw"]["error"], (label, sp["jaw"])
            r = TL.rest(arm)
            assert r["ok"] and r["torque"] == "off" and "fd tool jaw" in r["jaw"]["error"], (label, r)
            assert jaw_of(bus) == FIXED + off and not bus.goals, (label, "the jaw of a tool that is elsewhere was commanded")
    # `fd ready` says it as a result, with the jaw command as the next step, and nothing was switched on
    tool = tools["not said"]
    arm, bus = rig(tool, jaw=FIXED + 100, pose="rest")
    pos = dict(bus.pos)
    buf = io.StringIO()
    with tool_file(tool), contextlib.redirect_stdout(buf):
        real = fd.make_arm
        fd.make_arm = lambda sim: arm
        try:
            rc = fd.main(["--sim", "ready"])
        finally:
            fd.make_arm = real
    res = json.loads(buf.getvalue().strip().splitlines()[-1])
    assert rc == 1 and res["ok"] is False and "raw 2141" in res["message"] and bus.pos == pos and not any(bus.torque.values()), res
    assert "fd tool jaw status" in res.get("next", "") + res["message"], res
    # recorded on the FIXED jaw: the moving jaw does not carry the tool, wherever it stands; and a tool file without a
    # jaw position has nothing to compare: no check, no bus traffic, and `ready` goes
    for tool in (brush(on_jaw="fixed", **closed), brush(on_jaw=None, **dict(closed, gripper_raw=None))):
        assert TL.jaw_tolerance(tool) is None
        arm, bus = rig(tool, jaw=FIXED + 300, pose="rest")
        before = traffic(bus)
        assert TL.jaw_in_place(arm) is None and TL.hold_jaw(arm)["held"] is False and traffic(bus) == before
        r = TL.go_ready(arm)
        assert "jaw" not in r and jaw_of(bus) == FIXED + 300, r
    # status is the command the refusals name: it says that the jaw stands elsewhere, and what to do, per tool
    for label, texts in (("not said", ("fd tool jaw mount moving_outer", "fd tool jaw close", "fd tool jaw mount fixed")),
                         ("moving_outer", ("fd tool jaw close", "calib-air --tip-only")),
                         ("between", ("fd tool attach with --replace", "`fd tool attach --help`"))):
        with tool_file(tools[label]):
            arm, bus = rig(TL.active(), jaw=FIXED + 100, pose="rest")
            r = house(cli(["--sim", "status"], arm), ok=True, stage="status")
            assert r["in_place"] is False and r["jaw"]["from_record_ticks"] == 100 and "NOT where the tool file has it" in r["next"], r
            assert all(t in r["next"] for t in texts) and "ずれています" in r["say"] and "100 ticks" in r["say"], (label, r)
            arm, bus = rig(TL.active(), pose="rest")
            r = house(cli(["--sim", "status"], arm), ok=True, stage="status")
            assert r["in_place"] is True and "NOT where" not in r["next"] and "ずれています" not in r["say"], (label, r)
    # `fd ready` is what raises an arm that stands too low for a jaw move, and `fd ready` now refuses a jaw that stands
    # elsewhere: the refusal of the jaw command then names the way that is open (the real arm; --sim has --at-ready)
    low = SafetyError("with the arm where it stands, closing the jaw would swing the tip down to 12 mm above the table")
    opened = brush()                                         # recorded at 0.2 deg (raw 2182), the jaw now 100 ticks further open
    r = house(tooljaw._too_low(low, "close", False, False, (opened, CFG), tool=opened.name, jaw={"from_record_ticks": 100}),
              ok=False, stage="plan")
    assert "`fd ready` refuses" in r["next"] and "`fd tool jaw close --to 0.2`" in r["next"] and "100 ticks" in r["next"], r["next"]
    r = house(tooljaw._too_low(low, "open --to 35", False, False, (opened, CFG), tool=opened.name, jaw={"from_record_ticks": -100}),
              ok=False, stage="plan")
    assert "`fd tool jaw open --to 0.2`" in r["next"] and "fd tool jaw open --to 0.2" in r["say"], r
    shut = tools["moving_outer"]                             # recorded shut on the fixed jaw: no angle `--to` can name
    r = house(tooljaw._too_low(low, "close", False, False, (shut, CFG), tool=shut.name, jaw={"from_record_ticks": 100}),
              ok=False, stage="plan")
    assert "`fd tool rest`" in r["next"] and "`fd paint rest`" in r["next"] and "--to" not in r["next"], r["next"]
    for same in ({"from_record_ticks": 8}, None):            # the jaw where the file has it, or not read: the plain hint
        r = tooljaw._too_low(low, "close", False, False, (opened, CFG), tool=opened.name, jaw=same)
        assert r["next"] == "`fd ready` raises the arm; then `fd tool jaw close` again", r["next"]


def test_brush_work_refuses_a_jaw_that_stands_elsewhere_before_the_torque_comes_on():
    """Brush work (tools/paint, another owner's module): BrushRun.power() is where the painting commands switch
    the arm on, and it asks tooling.jaw_in_place first. A test of its own since round 2: it was the tail of the
    test above and ended with a printed note and `return` when brushwork did not load or did not take the tool,
    so the suite stayed green and the count the same with the refusal unchecked. Now that is a SKIP, which the
    count shows, and the BrushRun is built in one way only (it used to fall back to a call without the option
    `unmeasured_tip_ok` for a brushwork that did not know it)."""
    try:
        import brushwork as BW
    except Exception as e:           # (the painting modules are not this suite's to repair)
        raise Skip(f"brushwork could not be imported ({type(e).__name__}: {e}): its refusal of a moved jaw was not checked") from None
    # the rig's file of that evening: the jaw shut at raw 2041, nothing said about the jaw the brush is on
    tool = brush(on_jaw=None, gripper_raw=FIXED, tip_offset=list(TIP_CLOSED), mount="other", tip_measured={"at": "2026-11-13T17:25:00"})
    arm, bus = rig(tool, jaw=FIXED + 100, pose="rest")
    try:
        br = BW.BrushRun(arm, unmeasured_tip_ok=True)
    except SafetyError as e:         # (it wants something else of the tool first: not this suite's to judge)
        raise Skip(f"brushwork.BrushRun did not take the test's tool ({e}): its refusal of a moved jaw was not checked") from None
    before = traffic(bus)[1:]
    refused(br.power, "raw 2141", "no longer where its tip was measured", "fd tool jaw")
    assert not arm.is_torqued() and jaw_of(bus) == FIXED + 100 and all(abs(g - p) < 1.0 for _, g, p in bus.goals), bus.goals
    assert traffic(bus)[1:] == before, "a refused BrushRun.power() wrote to the servos"
    # the same jaw where the file has it: the arm is switched on
    arm, bus = rig(tool, pose="rest")
    BW.BrushRun(arm, unmeasured_tip_ok=True).power()
    assert arm.is_torqued() and jaw_of(bus) == FIXED


# ----------------------------------------------------------------------------- the command
def test_status_and_mount_are_about_the_file_and_read_the_arm_only():
    tool = brush(on_jaw=None, tip_measured={"at": "x"})
    with tool_file(tool):
        arm, bus = rig(TL.active(), torque=True)
        before = traffic(bus)[1:]
        r = house(cli(["--sim", "status"], arm), ok=True, stage="status")
        assert r["attached"] and r["tool"] == "brush_thin" and r["on_jaw"] is None and r["may_move"] is False, r
        assert r["recorded"] == {"gripper_raw": AT_ZERO, "jaw_q_deg": 0.19, "gripper_goal_raw": None, "squeeze_ticks": None,
                                 "tip_offset_mm": [-5.1, -2.5, 60.4], "tip_measured": True}, r["recorded"]
        assert r["jaw"]["pos_raw"] == AT_ZERO and r["jaw"]["q_deg"] == 0.19, r["jaw"]
        assert r["jaw"]["squeeze_on"] is False and r["jaw"]["torque"], r["jaw"]
        assert "fd tool jaw mount" in r["next"] and "photo" in r["next"] and "まだ記録されていません" in r["say"], r
        # mount: --dry writes nothing; then each value; the tip and its measurement are not the mount's business
        text = open(TOOL_PATH).read()
        r = house(cli(["--sim", "mount", "moving_outer", "--dry"], arm), ok=True, stage="planned")
        assert r["saved"] is False and r["changed"] and open(TOOL_PATH).read() == text, r
        r = house(cli(["--sim", "mount", "between"], arm), ok=True, stage="mount")
        assert r["saved"] and r["before"] is None and r["on_jaw"] == "between" and r["may_move"] is False, r
        assert TL.active().on_jaw == "between"
        r = house(cli(["--sim", "status"], arm), ok=True)
        assert r["on_jaw"] == "between" and r["may_move"] is False and "locked" in r["next"] and "ロック" in r["say"], r
        r = house(cli(["--sim", "mount", "fixed"], arm), ok=True, stage="mount")
        assert TL.active().on_jaw == "fixed" and "固定爪" in r["say"], r
        r = house(cli(["--sim", "mount", "moving_outer"], arm), ok=True, stage="mount")
        assert r["may_move"] and r["before"] == "fixed", r
        assert "fd tool jaw close --dry" in r["next"] and "NOTHING may be between" in r["next"], r
        assert "写真" in r["say"] and TL.active().on_jaw == "moving_outer" and TL.active().tip_measured == {"at": "x"}
        r = house(cli(["--sim", "mount", "moving_outer"], arm), ok=True, stage="mount")
        assert r["changed"] is False and r["saved"] is False, r
        r = house(cli(["--sim", "status"], arm), ok=True)
        assert r["may_move"] and "fd tool jaw close" in r["next"] and "0.2 度" in r["say"], r
        assert traffic(bus)[1:] == before and jaw_of(bus) == AT_ZERO, "status / mount wrote to the bus"
        # --no-robot: the file part only, the arm is not opened
        r = house(cli(["--sim", "status", "--no-robot"], None), ok=True)
        assert r["jaw"] is None and "--no-robot" in r["jaw_note"] and r["recorded"]["gripper_raw"] == AT_ZERO, r
    # a held jaw: squeeze on / off / the jaw elsewhere
    with tool_file(held()):
        arm, bus = rig(TL.active(), torque=True)
        TL.hold_jaw(arm)
        r = house(cli(["--sim", "status"], arm), ok=True)
        assert r["jaw"]["squeeze_on"] and r["jaw"]["squeeze_ticks"] == 25 and 100 <= r["jaw"]["load"] <= 120, r["jaw"]
        assert r["recorded"]["squeeze_ticks"] == 25 and r["recorded"]["gripper_goal_raw"] == FIXED - 25 and "保持しています" in r["say"], r
        assert "held" in r["next"] and "`fd tool jaw help`" in r["next"] and "DEG" not in r["next"], r["next"]
        bus.torque_off(("gripper",))
        r = house(cli(["--sim", "status"], arm), ok=True)
        assert r["jaw"]["squeeze_on"] is False and "puts it back" in r["next"] and "押し付けは切れています" in r["say"], r
        bus.pos["gripper"] = bus.goal["gripper"] = 2300.0
        r = house(cli(["--sim", "status"], arm), ok=True)
        assert r["jaw"]["from_record_ticks"] == 2300 - FIXED and "fd tool jaw close" in r["next"] and "calib-air" in r["next"], r
        assert "ずれています" in r["say"], r["say"]
        # taking the mount away from a held jaw drops the hold with it (the gripper is locked again)
        r = house(cli(["--sim", "mount", "between"], arm), ok=True)
        assert TL.active().gripper_goal_raw is None and r["notes"], r
    # a damaged file is an error here too
    with tool_file(None):
        with open(TOOL_PATH, "w") as f:
            f.write("{not json")
        r = house(tooljaw.main(["--sim", "status"]), ok=False, stage="tool")
        assert r["attached"] is None and "fd tool detach" in r["next"], r
        TL.clear()


def test_dry_run_shows_the_plan_and_the_estimate_and_moves_nothing():
    with tool_file(brush(tip_measured={"at": "x"})):
        arm, bus = rig(TL.active(), torque=True)
        text, before = open(TOOL_PATH).read(), traffic(bus)[1:]
        r = house(cli(["--sim", "close", "--dry"], arm), ok=True, stage="planned")
        assert r["moved"] is False and r["saved"] is False and r["jaw"]["pos_raw"] == AT_ZERO, r
        plan = r["plan"]
        assert plan["to"] == "closed" and plan["squeeze_ticks"] == 25 and plan["step_ticks"] == 15 and plan["from_raw"] == AT_ZERO, plan
        assert plan["expect_raw"] == TOUCH and plan["floor_raw"] >= arm.raw_limits["gripper"][0] and plan["seconds"] < 6, plan
        # where the tool ends up: the jaw model, with its error said (to where bare fingertips touch: 2050)
        e = r["estimate"]
        assert e["from_q_deg"] == 0.19 and e["to_q_deg"] == -11.41 and e["expect_error_mm"] == 10.0, e
        assert 20.0 < e["tip_offset_mm"][0] < 23.0, e
        assert e["tip_offset_mm"] == [round(v * 1000.0, 1) for v in TL.jaw_tip_estimate(TIP_AT_ZERO, 0.19140625, -11.41)], e
        assert r["swing"]["tip_height_mm"]["lowest"] > 80 and r["tool_would_move"] is True, r["swing"]
        assert "fd tool jaw close" in r["next"] and "calib-air --tip-only" in r["next"], r["next"]
        assert "何も動かしていません" in r["say"] and "10 mm" in r["say"] and "calib-air --tip-only" in r["say"], r["say"]
        r = house(cli(["--sim", "close", "--to", "-5", "--squeeze", "25", "--dry"], arm), ok=False, stage="options")
        assert "--squeeze" in r["reason"], r
        r = house(cli(["--sim", "close", "--to", "-5", "--dry"], arm), ok=True, stage="planned")
        assert r["plan"]["target_raw"] == arm.gripper_raw_for_q(-5.0) and "--to -5" in r["next"] and r["plan"]["squeeze_ticks"] is None, r
        r = house(cli(["--sim", "open", "--to", "35", "--dry"], arm), ok=True, stage="planned")
        assert r["estimate"]["tip_offset_mm"][0] < -80 and r["plan"]["direction"] == "open" and "開く" in r["say"], r
        for words, text_ in ((["close", "--to", "30", "--dry"], "would OPEN"), (["open", "--to", "-5", "--dry"], "would CLOSE"),
                             (["close", "--squeeze", "60", "--dry"], "squeeze of 60"),
                             (["close", "--step", "20", "--dry"], "whole number"),
                             (["close", "--to", "soon", "--dry"], "--to"), (["close", "--lag", "90", "--dry"], "whole number")):
            r = house(cli(["--sim"] + words, arm), ok=False, stage="options")
            assert text_ in r["reason"] and "動かしていません" in r["say"], (words, r)
        assert open(TOOL_PATH).read() == text and traffic(bus)[1:] == before and jaw_of(bus) == AT_ZERO, "a dry run wrote something"
        # without the arm: the plan from the file alone, and it says that the swing was not checked
        r = house(cli(["--sim", "close", "--dry", "--no-robot"], None), ok=True, stage="planned")
        assert r["jaw"] is None and "swing" not in r and "not checked against the table" in r["jaw_note"] and r["estimate"], r
    with tool_file(brush(gripper_raw=None)):                  # attached with --no-robot: nothing to turn the tip from
        r = house(cli(["--sim", "close", "--dry", "--no-robot"], None), ok=True, stage="planned")
        assert r["estimate"] is None and "推定できません" in r["say"], r


def test_close_and_open_through_the_command_store_what_happened():
    with tool_file(brush()):
        arm, bus = rig(TL.active())
        # blocked early: a complete result, the file says where the jaw is, nothing is held
        bus.blocker = lambda pos: pos["gripper"] < 2130
        r = house(cli(["--sim", "close"], arm), ok=False, stage="close")
        assert r["stopped_early"] and r["met_fixed_jaw"] is False and "something is in its way" in r["reason"], r
        assert r["jaw"]["pos_raw"] == 2130 and r["jaw"]["squeeze_on"] is False and r["jaw"]["goal_raw"] == 2130, r["jaw"]
        assert "fd snap" in r["next"] and "fd calib-air --tip-only" in r["next"] and "止まりました" in r["say"] and "解除" in r["say"], r
        now = TL.active()
        assert (now.gripper_raw, now.gripper_goal_raw) == (2130, None) and r["tool_moved"] and r["saved"], now
        # the way is free: the rest of the way, held; the estimate is turned on from where the first move ended
        bus.blocker = lambda pos: pos["gripper"] < FIXED
        r = house(cli(["--sim", "close", "--squeeze", "30", "--step", "10"], arm), ok=True, stage="close")
        now = TL.active()
        assert (now.gripper_raw, now.gripper_goal_raw) == (FIXED, FIXED - 30) and r["jaw"]["squeeze_ticks"] == 30, (now, r["jaw"])
        assert np.allclose(np.array(now.tip_offset) * 1000.0, [23.4, -2.5, 52.3], atol=0.15), now.tip_offset
        assert len(now.notes) == 2 and r["move"]["steps"] >= 9 and r["move"]["duration_s"] < 10, r["move"]
        # open to an angle: the hold is gone from the record
        r = house(cli(["--sim", "open", "--to", "0"], arm), ok=True, stage="open")
        now = TL.active()
        assert now.gripper_raw == arm.gripper_raw_for_q(0.0) and now.gripper_goal_raw is None and r["tool_moved"], now
        assert "開きました" in r["say"] and "backlash" in r["next"] and "fd calib-air --tip-only" in r["next"], r
        # (back within 2 ticks of where it started: the tip the model gives is the tip it started with, turned by those 2 ticks)
        want = TL.jaw_tip_estimate(TIP_AT_ZERO, arm.gripper_q_deg(AT_ZERO), arm.gripper_q_deg(now.gripper_raw))
        assert np.allclose(np.array(now.tip_offset) * 1000.0, np.array(want) * 1000.0, atol=0.05), (now.tip_offset, want)
        # an interrupt in the middle: the answer still comes, and the file has the truth
        bus.interrupt_at = 4
        r = house(cli(["--sim", "close"], arm), ok=False, stage="close")
        assert r["interrupted"] is True and r["error"] == "KeyboardInterrupt" and "fd tool jaw status" in r["next"], r
        now = TL.active()
        assert now.gripper_raw == jaw_of(bus) and abs(bus.goal["gripper"] - bus.pos["gripper"]) <= 1.0 and now.gripper_goal_raw is None
        # a servo that is too hot: refused with a result, nothing moved
        bus.temp["gripper"] = 70
        at = jaw_of(bus)
        r = house(cli(["--sim", "close"], arm), ok=False, stage="jaw")
        assert "temperature" in r["reason"] and jaw_of(bus) == at and TL.active().gripper_raw == at, r


def test_words_the_command_cannot_read_are_answered_not_obeyed():
    with tool_file(brush()):
        arm, bus = rig(TL.active())
        before = traffic(bus)
        for words in (["wiggle"], ["close", "--hard"], ["mount"], ["mount", "inside"], ["open"], ["close", "--to"], ["status", "extra"]):
            r = house(cli(["--sim"] + words, arm), ok=False, stage="options")
            assert r["error"] == "SafetyError" and "usage" in r["next"] and "fd tool jaw status" in r["next"], (words, r)
        for words in ([], ["help"], ["--help"], ["-h"]):
            r = house(cli(["--sim"] + words, arm), ok=True, stage="help")
            assert "close" in r["usage"] and "moving_outer" in r["usage"], r
        # --sim anywhere but in front means that fd took the command for the REAL arm: never guess
        for words in (["close", "--sim"], ["status", "--sim"], ["mount", "between", "--sim"]):
            opened = []
            real = tooljaw.make_arm
            tooljaw.make_arm = lambda sim: opened.append(sim) or arm
            try:
                r = tooljaw.main(words)
            finally:
                tooljaw.make_arm = real
            house(r, ok=False, stage="options")
            assert "fd --sim tool jaw" in r["next"] and not opened and TL.active().on_jaw == "moving_outer", (words, r)
        assert traffic(bus) == before and jaw_of(bus) == AT_ZERO
    # A simulated command that is not told where its tool file is does not write into the project folder. The
    # "project folder" here is a stand-in: tooling's two tool files and tazuna_paths.ROOT point into the scratch
    # folder for the duration ($FD_TOOL is not set), as tests/test_aircal.py does it. Until round 2 this ran with
    # tooling.SIM_TOOL_FILE on the RIG's tool_active.sim.json and relied on the refusal under test to keep that file
    # from being written: had the guard regressed, the test itself would have created it, in the folder no test may write.
    tree = tempfile.mkdtemp(prefix="stand_in_tree_", dir=_TMP)
    saved = (os.environ.pop("FD_TOOL"), TL.SIM_TOOL_FILE, TL.TOOL_FILE, tazuna_paths.ROOT)
    TL.TOOL_FILE, TL.SIM_TOOL_FILE, tazuna_paths.ROOT = os.path.join(tree, "tool_active.json"), os.path.join(tree, "tool_active.sim.json"), tree
    try:
        with open(TL.TOOL_FILE, "w") as f:
            json.dump(brush(on_jaw=None).to_dict(), f)
        assert TL.tool_path(True) == TL.SIM_TOOL_FILE and os.path.dirname(TL.SIM_TOOL_FILE) == tazuna_paths.ROOT == tree
        assert not (os.path.realpath(tree) + os.sep).startswith(os.path.realpath(saved[3]) + os.sep), "the stand-in lies in the rig's folder"
        for words in (["mount", "moving_outer"], ["mount", "fixed"]):
            refused(lambda: tooljaw.main(["--sim"] + words), "project folder", "fd --sim tool jaw", "$FD_TOOL")
        # nothing was written, neither into the stand-in nor (the last test of the suite looks again) into the rig's folder
        assert sorted(os.listdir(tree)) == ["tool_active.json"] and json.load(open(TL.TOOL_FILE)).get("on_jaw") is None
        # (the guard is about the simulated file inside the project folder: where fd keeps it, outside, the same call stores)
        TL.SIM_TOOL_FILE = os.path.join(tempfile.mkdtemp(prefix="sim_state_", dir=_TMP), "tool_active.sim.json")
        r = house(tooljaw.main(["--sim", "mount", "moving_outer"]), ok=True, stage="mount")
        assert r["saved"] and r["file"] == TL.SIM_TOOL_FILE and r["simulated_tool"] is True and sorted(os.listdir(tree)) == ["tool_active.json"], r
    finally:
        os.environ["FD_TOOL"], TL.SIM_TOOL_FILE, TL.TOOL_FILE, tazuna_paths.ROOT = saved
    assert _tree_files() == _TREE_FILES and TL.tool_path(True) == TOOL_PATH


def test_help_words_are_answered_with_the_usage_and_nothing_else_happens():
    """`fd tool jaw --help` used to end in fd's own parser (exit code 2, no result). fd hands the words through
    since 2026-11-13, so the answer is this module's: the usage, as a result, whatever else stands on the line."""
    with tool_file(brush()):
        text = open(TOOL_PATH).read()
        opened = []
        real = tooljaw.make_arm
        tooljaw.make_arm = lambda sim: opened.append(sim) or (_ for _ in ()).throw(AssertionError("the arm was opened"))
        try:
            for words in (["--help"], ["-h"], ["help"], [], ["--sim", "--help"], ["--sim", "-h"], ["--", "--help"],
                          ["--sim", "--", "-h"], ["close", "--help"], ["close", "--to", "closed", "-h"], ["mount", "--help"],
                          ["--sim", "open", "--help"], ["status", "-h"], ["wiggle", "--help"]):
                r = house(tooljaw.main(list(words)), ok=True, stage="help")
                assert r["usage"] == tooljaw.USAGE and "moving_outer" in r["usage"] and "fd tool jaw status" in r["next"], (words, r)
                named = [w for w in words if w in tooljaw.COMMANDS]
                assert r["command"] == (named[0] if named else None), (words, r)
                flat = " ".join(" ".join(r["help"]).split())     # (argparse wraps its lines by the width of the terminal)
                for cmd in named or tooljaw.COMMANDS:            # the options of what was asked about, with their texts
                    assert f"fd tool jaw {cmd}" in flat, (words, cmd, flat[:200])
                assert ("--squeeze" in flat) == (not named or named[0] == "close"), (words, flat[:300])
                if not named or named[0] in ("close", "open"):
                    assert "--to" in flat and "jaws parallel" in flat and "--at-ready" in flat, flat[:300]
                assert "動かしていません" in r["say"], r["say"]
        finally:
            tooljaw.make_arm = real
        assert not opened and open(TOOL_PATH).read() == text
        # through fd, in this process and as a process of its own: exit code 0, the same answer
        for argv in (["tool", "jaw", "--help"], ["--sim", "tool", "jaw", "-h"], ["tool", "jaw", "close", "--help"]):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                rc = fd.main(list(argv))
            res = json.loads(buf.getvalue().strip().splitlines()[-1])
            assert rc == 0 and res["stage"] == "help" and res["usage"] == tooljaw.USAGE and ("sim" in res) == (argv[0] == "--sim"), res
        r, rc = fd_process("tool", "jaw", "--help")
        assert rc == 0 and r["ok"] is True and r["stage"] == "help" and r["usage"] == tooljaw.USAGE and "sim" not in r, r
        r, rc = fd_process("--sim", "tool", "jaw", "open", "-h")
        assert rc == 0 and r["stage"] == "help" and r["command"] == "open" and r["sim"] is True, r
        assert open(TOOL_PATH).read() == text


def test_a_damaged_tool_file_is_answered_in_the_house_shape_by_every_jaw_command():
    """Review finding 37 (2026-11-13). `status` said it as a result; `mount`, `close` and `open` let the SafetyError
    through, and fd printed {"ok": false, "error", "message"} without stage, next and say."""
    good = brush().to_dict()
    with tool_file(None):
        for text in ("", "{not json", "{}", "[]", json.dumps(dict(good, on_jaw="sideways")),
                     json.dumps(dict(good, gripper_goal_raw=AT_ZERO - 25, on_jaw="between")), json.dumps(dict(good, tip_offset=[0, 0, 60]))):
            with open(TOOL_PATH, "w") as f:
                f.write(text)
            for words in (["status"], ["status", "--no-robot"], ["mount", "moving_outer"], ["mount", "fixed", "--dry"], ["close"],
                          ["close", "--dry"], ["close", "--to", "0"], ["open", "--to", "20"], ["open", "--to", "20", "--dry"]):
                opened = []
                real = tooljaw.make_arm
                tooljaw.make_arm = lambda sim: opened.append(sim) or (_ for _ in ()).throw(AssertionError("the arm was opened"))
                try:
                    r = tooljaw.main(["--sim"] + words)                        # (a SafetyError here is the finding)
                finally:
                    tooljaw.make_arm = real
                house(r, ok=False, stage="tool")
                assert r["error"] == "SafetyError" and r["attached"] is None and r["file"] == TOOL_PATH and not opened, (text, words, r)
                assert "cannot be used" in r["reason"] and "fd tool detach" in r["next"] and "repair" in r["next"], (text, words, r)
                assert "読めません" in r["say"] and "動かしていません" in r["say"], r["say"]
                assert open(TOOL_PATH).read() == text, "a command wrote over a tool file it could not read"
            # without --sim (the real arm's tool file): the same answer, before the arm is asked for
            del _touched[:]
            for words in (["mount", "moving_outer"], ["close"], ["open", "--to", "20"]):
                r = house(tooljaw.main(words), ok=False, stage="tool")
                assert "fd tool detach" in r["next"] and "--sim" not in r["next"] and not _touched, (words, r)
        # through fd, as a process: the exit code of a refusal and the complete result
        with open(TOOL_PATH, "w") as f:
            f.write("{}")
        for words in (["mount", "moving_outer"], ["close", "--dry"], ["close"], ["status"]):
            r, rc = fd_process("--sim", "tool", "jaw", *words)
            house(r, ok=False, stage="tool")
            assert rc == 1 and r["sim"] is True and "fd tool detach" in r["next"] and r["message"] == r["reason"], (words, r)
        TL.clear()


@contextlib.contextmanager
def two_tool_files():
    """The real and the simulated tool file as `fd --sim` has them without $FD_TOOL, both in the scratch folder:
    (real path, simulated path)."""
    saved = (os.environ.pop("FD_TOOL"), TL.SIM_TOOL_FILE, TL.TOOL_FILE)
    real, simf = os.path.join(_TMP, "two", "tool_active.json"), os.path.join(_TMP, "two", "simstate", "tool_active.sim.json")
    TL.TOOL_FILE, TL.SIM_TOOL_FILE = real, simf
    try:
        yield real, simf
    finally:
        os.environ["FD_TOOL"], TL.SIM_TOOL_FILE, TL.TOOL_FILE = saved
        for p in (real, simf):
            if os.path.exists(p):
                os.remove(p)


def test_a_simulated_copy_of_the_tool_is_said_and_left_behind_when_the_real_tool_changes():
    """Review finding 32 (2026-11-13). `fd --sim tool jaw mount moving_outer` on a simulation that mirrors the real
    tool stored a copy of it as the simulated tool and answered 「…と記録しました」; from then on `fd --sim` no
    longer followed the real tool file, also after the real tool had been registered anew, and no result said so."""
    with two_tool_files() as (real, simf):
        TL.save(brush(on_jaw=None, attached_at="2026-11-13T17:17:45", tip_measured={"at": "x"}))       # the REAL tool
        real_text = open(real).read()
        assert TL.SIM_COPY_KEY not in json.loads(real_text)
        # the simulation mirrors it: nothing simulated about the tool yet, and the result says that too
        r = house(tooljaw.main(["--sim", "status", "--no-robot"]), ok=True, stage="status")
        assert r["simulated_tool"] is False and r["file"] == real and "sim_note" not in r and "sim_copy_dropped" not in r, r
        r = house(tooljaw.main(["--sim", "mount", "moving_outer", "--dry"]), ok=True, stage="planned")
        assert r["simulated_tool"] is False and not os.path.exists(simf), r                     # (a dry run stores nothing)
        # a simulated command that changes the tool stores a COPY, and says so
        r = house(tooljaw.main(["--sim", "mount", "moving_outer"]), ok=True, stage="mount")
        assert r["saved"] and r["file"] == simf and r["simulated_tool"] is True, r
        assert "stored the tool in the SIMULATED tool file" in r["sim_note"] and "copy of the real tool" in r["sim_note"], r["sim_note"]
        assert "fd --sim tool detach" in r["sim_note"] and "until the real tool file changes" in r["sim_note"], r["sim_note"]
        assert "シミュレーション用のツール記録に保存しました" in r["say"] and "実機のツール記録は変えていません" in r["say"], r["say"]
        assert open(real).read() == real_text and TL.active().on_jaw is None, "a simulated command changed the REAL tool file"
        mark = json.load(open(simf))[TL.SIM_COPY_KEY]
        assert (mark["name"], mark["attached_at"], mark["file"]) == ("brush_thin", "2026-11-13T17:17:45", real), mark
        assert TL.active(sim=True).on_jaw == "moving_outer" and TL.source(sim=True) == simf
        st = TL.sim_tool_state()
        assert st["simulated_tool"] and st["copy_of_real"] and st["dropped"] is None and st["file"] == simf, st
        r = house(tooljaw.main(["--sim", "status", "--no-robot"]), ok=True, stage="status")
        assert r["simulated_tool"] is True and r["on_jaw"] == "moving_outer" and "copy of the real tool" in r["sim_note"], r
        # ... and the next simulated command works on that copy (a simulated close, each call a new simulated arm)
        r = house(tooljaw.main(["--sim", "close"]), ok=True, stage="close")
        assert r["simulated_tool"] is True and r["file"] == simf and "シミュレーション用のツール記録に保存しました" in r["say"], r
        sim_now = TL.active(sim=True)
        assert (sim_now.gripper_raw, sim_now.gripper_goal_raw) == (TOUCH, TOUCH - 25) and sim_now.tip_measured is None
        assert json.load(open(simf))[TL.SIM_COPY_KEY] == mark and open(real).read() == real_text
        assert TL.active().gripper_raw == AT_ZERO and TL.active().tip_measured == {"at": "x"}
        # the mark never reaches a real tool file, whatever tool object is stored there
        TL.save(sim_now)
        assert TL.SIM_COPY_KEY not in json.load(open(real)) and TL.active().gripper_raw == TOUCH
        TL.save(TL.Tool.from_dict(json.loads(real_text)))                                    # (the real tool as it was)
        assert open(real).read() == real_text and TL.active(sim=True).gripper_raw == TOUCH, "the same real record: the copy still holds"
        # the real tool is CHANGED (its jaw record, by a real command): the copy is left behind, the simulation
        # follows the real tool again, and the command that finds it so says it
        t = TL.active()
        t.on_jaw = "fixed"
        TL.save(t)
        assert TL.active(sim=True).on_jaw == "fixed" and TL.source(sim=True) == real and os.path.exists(simf)
        st = TL.sim_tool_state()
        assert st["simulated_tool"] is False and st["dropped"]["file"] == simf and "last changed" in st["dropped"]["why"], st
        r = house(tooljaw.main(["--sim", "status", "--no-robot"]), ok=True, stage="status")
        assert r["on_jaw"] == "fixed" and r["simulated_tool"] is False and r["file"] == real, r
        assert "last changed" in r["sim_copy_dropped"]["why"] and "使っていません" in r["say"], r
        # a new simulated change is a new copy of the real tool as it is NOW (over the one left behind); the result
        # still says that an older copy was left behind
        r = house(tooljaw.main(["--sim", "mount", "moving_outer"]), ok=True, stage="mount")
        assert r["simulated_tool"] is True and r["before"] == "fixed" and "sim_copy_dropped" in r, r
        assert TL.active(sim=True).gripper_raw == AT_ZERO and TL.active(sim=True).on_jaw == "moving_outer"
        # the real tool is registered ANEW (another brush): the reviewer's case. The simulation answers for the new one
        TL.save(TL.Tool(name="brush_round2", tip_offset=[0.012, 0.0, 0.064], mount="other", pitch_deg=65.0, gripper_raw=FIXED,
                        attached_at="2026-11-14T09:00:00"))
        assert TL.active(sim=True).name == "brush_round2" and TL.source(sim=True) == real
        r = house(tooljaw.main(["--sim", "status", "--no-robot"]), ok=True, stage="status")
        assert r["tool"] == "brush_round2" and "brush_round2" in r["sim_copy_dropped"]["why"] and r["simulated_tool"] is False, r
        # the real tool is taken off: a copy of it is no tool of the simulation either
        TL.clear()
        assert TL.active(sim=True) is None and TL.source(sim=True) is None and "no tool any more" in TL.sim_tool_state()["dropped"]["why"]
        # a tool attached IN SIMULATION is not a copy: it stays whatever the real tool does, and is said to be simulated
        os.remove(simf)
        TL.save(TL.Tool(name="sim_pen", tip_offset=[0.0015, -0.0081, 0.0621], mount="along_fingers", attached_at="2026-11-14T10:00:00"),
                sim=True)
        assert TL.SIM_COPY_KEY not in json.load(open(simf))
        for real_tool in (None, brush(), brush(name="other", attached_at="2026-11-14T11:00:00")):
            TL.clear()
            if real_tool is not None:
                TL.save(real_tool)
            assert TL.active(sim=True).name == "sim_pen" and TL.source(sim=True) == simf
            st = TL.sim_tool_state()
            assert st["simulated_tool"] and not st["copy_of_real"] and st["dropped"] is None and "tool of its own" in st["note"], st
        r = house(tooljaw.main(["--sim", "status", "--no-robot"]), ok=True, stage="status")
        assert r["tool"] == "sim_pen" and r["simulated_tool"] is True and "fd --sim tool detach" in r["sim_note"], r
    # with $FD_TOOL there is one file for both: nothing is a copy, nothing is marked, and the results say "not simulated"
    with tool_file(brush(on_jaw=None)):
        r = house(tooljaw.main(["--sim", "mount", "moving_outer"]), ok=True, stage="mount")
        assert r["simulated_tool"] is False and "sim_note" not in r and "シミュレーション用" not in r["say"], r
        assert TL.SIM_COPY_KEY not in json.load(open(TOOL_PATH))
        r = house(tooljaw.main(["mount", "between"]), ok=True, stage="mount")                 # (not simulated: not said at all)
        assert "simulated_tool" not in r, r


def test_fd_as_a_process_gives_house_results():
    """`fd --sim tool jaw status | mount | close --dry | close` as the launcher runs them: each call a new
    process with a new simulated arm, the tool file ($FD_TOOL) the only thing they share."""
    with tool_file(brush(on_jaw=None, tip_measured={"at": "2026-11-13T16:50:00"})):
        r, rc = fd_process("--sim", "tool", "jaw", "status")
        house(r, ok=True, stage="status")
        assert rc == 0 and r["sim"] is True and r["on_jaw"] is None and r["may_move"] is False and r["file"] == TOOL_PATH, r
        assert r["jaw"]["pos_raw"] == AT_ZERO and r["recorded"]["tip_measured"] is True and "fd tool jaw mount" in r["next"], r
        r, rc = fd_process("--sim", "tool", "jaw", "close", "--dry")
        house(r, ok=False, stage="tool")
        assert rc == 1 and "does not say on which jaw" in r["reason"] and "fd tool jaw mount moving_outer" in r["next"], r
        r, rc = fd_process("--sim", "tool", "jaw", "mount", "moving_outer")
        house(r, ok=True, stage="mount")
        assert rc == 0 and r["saved"] and r["may_move"] and TL.active().on_jaw == "moving_outer", r
        text = open(TOOL_PATH).read()
        r, rc = fd_process("--sim", "tool", "jaw", "close", "--dry")
        house(r, ok=True, stage="planned")
        assert rc == 0 and r["moved"] is False and r["plan"]["to"] == "closed" and r["estimate"]["expect_error_mm"] == 10.0, r
        assert r["swing"]["tip_height_mm"]["lowest"] >= 20.0 and "calib-air --tip-only" in r["next"] and open(TOOL_PATH).read() == text, r
        # the move itself, in the tool's rest pose where the simulated arm starts; the simulated jaws touch at 2050
        r, rc = fd_process("--sim", "tool", "jaw", "close")
        house(r, ok=True, stage="close")
        assert rc == 0 and r["met_fixed_jaw"] and r["jaw"]["pos_raw"] == TOUCH and r["jaw"]["squeeze_ticks"] == 25 and r["tool_moved"], r
        now = TL.active()
        assert (now.gripper_raw, now.gripper_goal_raw, now.tip_measured) == (TOUCH, TOUCH - 25, None), now
        # the next process finds the jaw where the file has it, the squeeze gone with the torque, and puts it back
        r, rc = fd_process("--sim", "tool", "jaw", "status")
        assert rc == 0 and r["jaw"]["pos_raw"] == TOUCH and r["jaw"]["squeeze_on"] is False and r["recorded"]["squeeze_ticks"] == 25, r
        r, rc = fd_process("--sim", "tool", "jaw", "close")
        assert rc == 0 and r["ok"] and r["tool_moved"] is False and r["jaw"]["squeeze_on"] and r["jaw"]["pos_raw"] == TOUCH, r
        # (a finished close names the command that comes next: this tool's tip is not on record as measured)
        assert "Next: " in r["next"] and "`fd calib-air --tip-only --save`" in r["next"] and "`fd paint status`" in r["next"], r["next"]
        r, rc = fd_process("--sim", "ready")
        assert rc == 0 and r["ok"] and r["tool"] == "brush_thin", r
        r, rc = fd_process("--sim", "tool", "jaw")
        assert rc == 0 and r["stage"] == "help" and "usage" in r, r
        # the gripper itself stays locked, and says where the jaw commands are (one command per pair of backticks)
        r, rc = fd_process("--sim", "grip", "0.5")
        assert rc == 1 and r["stage"] == "tool" and "`fd tool jaw close` (`fd tool jaw help` lists" in r["reason"], r
        assert "DEG" not in r["reason"] + r["next"], r                  # (no placeholder inside backticks: round 3)
        assert "|" not in r["reason"] + r["next"], r
    assert os.path.isdir(os.path.join(_TMP, "runs")), "the processes logged somewhere else than the scratch folder"


def test_the_hardware_lock_refuses_the_real_close_before_the_bus_is_opened():
    assert hwlock.path() == HW_LOCK and hwlock.info() is not None, "the scratch lock is not set"
    with tool_file(held()):
        text = open(TOOL_PATH).read()
        del _touched[:]

        def bus_lock():      # the flock file every process on the real bus shares: written when the bus is taken
            return os.stat(BUS_FLOCK).st_mtime_ns if os.path.exists(BUS_FLOCK) else None

        taken = bus_lock()
        # the real arm, in this process: refused by the lock, and the serial port was never asked for
        for words in (["close"], ["close", "--to", "closed", "--squeeze", "20"], ["open", "--to", "10"]):
            try:
                tooljaw.main(words)
                raise AssertionError(f"the real `{' '.join(words)}` was not refused")
            except BusError as e:
                assert "hardware is locked" in str(e) and "fd hw unlock" in str(e) and "nothing moved" in str(e), e
        assert not _touched and open(TOOL_PATH).read() == text
        assert bus_lock() == taken, "the bus lock file was taken: the refusal came after the bus was asked for"
        # status and the dry run fall back to the file; mount is the file only
        r = house(tooljaw.main(["status"]), ok=True, stage="status")
        assert r["jaw"] is None and "hardware is locked" in r["jaw_note"] and r["recorded"]["gripper_goal_raw"] == FIXED - 25, r
        r = house(tooljaw.main(["close", "--dry"]), ok=True, stage="planned")
        assert r["jaw"] is None and "hardware is locked" in r["jaw_note"] and r["plan"]["to"] == "closed", r
        assert not _touched
        # through fd, as a process: the house result of a locked rig
        r, rc = fd_process("tool", "jaw", "close")
        assert rc == 1 and r["ok"] is False and r["stage"] == "hardware" and "fd hw unlock" in r["next"] and "ロック" in r["say"], r
        assert "hardware is locked" in r["message"] and "touched hardware" not in r["message"] and "sim" not in r, r
        r, rc = fd_process("tool", "jaw", "status")
        assert rc == 0 and r["ok"] and r["jaw"] is None and "hardware is locked" in r["jaw_note"], r
        assert open(TOOL_PATH).read() == text and bus_lock() == taken


def test_the_simulated_gripper_has_a_fixed_jaw_for_a_tool_on_the_moving_jaw():
    """SimBus has no fixed jaw; fd's simulated arm gets one from the tool file (tooling.sim_start)."""
    with tool_file(held()):
        arm = fd.make_arm(True).open()
        assert round(arm.bus.pos["gripper"]) == FIXED and arm.bus.blocker is not None and arm.bus.blocker({"gripper": FIXED - 1})
        assert not arm.bus.blocker({"gripper": FIXED}) and arm.bus.io["writes"] == 0
        arm.torque_on()
        arm.bus.sleep(0.3)
        assert TL.hold_jaw(arm)["wrote"] and round(arm.bus.pos["gripper"]) == FIXED and int(arm.bus.goal["gripper"]) == FIXED - 25
    with tool_file(brush(gripper_raw=2576)):                 # not held yet: the fingertips touch where the config says
        arm = fd.make_arm(True).open()
        assert round(arm.bus.pos["gripper"]) == 2576
        assert arm.bus.blocker({"gripper": TOUCH - 1}) and not arm.bus.blocker({"gripper": TOUCH})
    with tool_file(brush(gripper_raw=FIXED)):                # a jaw recorded below that (the rig's file of 2026-11-13)
        arm = fd.make_arm(True).open()
        assert arm.bus.blocker({"gripper": FIXED - 1}) and not arm.bus.blocker({"gripper": FIXED})
    for tool in (brush(on_jaw=None), brush(on_jaw="between"), None):
        with tool_file(tool):                                 # everything else: the simulated bus is what it was
            arm = fd.make_arm(True).open()
            assert arm.bus.blocker is None
    # an obstacle a test put there is left alone, also by the lazy path of move_jaw
    arm, bus = rig(brush())
    mine = bus.blocker
    TL.move_jaw(arm, "close", 0.0)
    assert bus.blocker is mine


def test_the_real_bus_class_carries_the_jaw_commands():
    """servo_bus.Bus itself on six simulated servos behind a pty (fakebus.py, as in test_bus_fake.py): the
    registers, the torque of the gripper alone and the feedback read are those of the real bus, not of
    SimBus's look-alike. Real time: about 6 s."""
    import fakebus
    fake = fakebus.FakeBus()
    bus = fake.bus(timeout_ms=150.0)
    tool = brush()
    grip = fakebus.IDS["gripper"]
    try:
        arm = Arm(bus=bus, cfg=load_config(), tool=tool).open()
        assert bus.port.startswith("/dev/pts/") and arm.raw_limits["gripper"] == (1998, 3648)
        for j, v in arm.q_to_raw(TL.ready_q(arm, tool)).items():      # limp at the tool's stand-by pose, the jaw at 0.2 deg
            fake.set_pos(j, v)
        fake.set_pos("gripper", AT_ZERO)
        fake.blocker = lambda sid, pos: sid == grip and pos < FIXED       # the fixed jaw
        st = TL.jaw_state(arm)
        assert st["pos_raw"] == AT_ZERO and st["torque"] is False and st["temp_c"] == 25 + grip and st["squeeze_on"] is False, st
        refused(lambda: arm.jaw_goal(AT_ZERO - 5), "gripper torque is off")
        assert not fake.goal_log and fake.view()["torque"] == [0] * 6, "a goal was written to a limp gripper"
        r = TL.move_jaw(arm)
        assert r["ok"] and r["met_fixed_jaw"] and not r["stopped_early"] and r["torque_switched_on"], r
        assert FIXED <= r["pos_raw"] <= FIXED + 12 and r["goal_raw"] == r["pos_raw"] - 25 and r["temp_c"] == 25 + grip, r
        assert fake.reg("gripper", "Goal_Position") == r["goal_raw"] and round(fake.pos[grip]) == r["pos_raw"]
        assert fake.view()["torque"] == [0, 0, 0, 0, 0, 1], "more than the gripper was switched on"
        goals = [g for _, sid, g in fake.goal_log if sid == grip]
        steps = [abs(b - a) for a, b in zip(goals, goals[1:-1])]             # (the last write is the squeeze)
        assert steps and max(steps) <= TL.JAW_STEP_TICKS and min(goals) >= 1998, goals
        times = [t for t, sid, _ in fake.goal_log if sid == grip]
        assert min(b - a for a, b in zip(times[1:], times[2:-1])) > 0.1, "the steps came faster than one in 0.12 s"
        assert (tool.gripper_raw, tool.gripper_goal_raw) == (r["pos_raw"], r["goal_raw"]) and r["record"]["tool_moved"]
        st = TL.jaw_state(arm)
        assert st["squeeze_on"] and st["squeeze_ticks"] == 25 and st["goal_raw"] == r["goal_raw"], st
        # the torque off and on again (all six, as a command does): the squeeze is lost, hold_jaw brings it back
        bus.torque_off(("gripper",))
        arm.torque_on()
        assert fake.reg("gripper", "Goal_Position") == round(fake.pos[grip]) and fake.view()["torque"] == [1] * 6
        h = TL.hold_jaw(arm)
        assert h["held"] and h["wrote"] and fake.reg("gripper", "Goal_Position") == tool.gripper_goal_raw, h
        assert abs(round(fake.pos[grip]) - tool.gripper_raw) <= 1 and TL.hold_jaw(arm)["wrote"] is False
        # and a jaw that was pushed open meanwhile is refused without a write
        bus.torque_off(("gripper",))
        fake.set_pos("gripper", AT_ZERO)
        arm.torque_on()
        n = len(fake.goal_log)
        refused(lambda: TL.hold_jaw(arm), "no longer where its tip was measured")
        assert len(fake.goal_log) == n
        arm.torque_off(force=True)
    finally:
        bus.close()
        fake.close()
    assert not _touched


# ----------------------------------------------------------------------------- one order for every text (round 3)
PAINT_SETUP = os.path.join(_TMP, "paint_setup.json")


@contextlib.contextmanager
def paint_setup(on_record: bool = True):
    """A painting set-up on record for the duration: $FD_PAINT_SETUP names a scratch file that exists (on_record
    False: the name is pinned and no file is there). The jaw commands ask only whether one is on record
    (tooljaw._paint_setup, the rule of fd._paint_setup_on_record); what it holds is the painting commands' business."""
    had = os.environ.get("FD_PAINT_SETUP")
    os.environ["FD_PAINT_SETUP"] = PAINT_SETUP
    if on_record:
        with open(PAINT_SETUP, "w", encoding="utf-8") as f:
            json.dump({"canvas": {"origin_xy": [0.146, -0.118], "size_mm": [224, 153], "pitch_deg": 65.0}}, f)
    try:
        yield PAINT_SETUP
    finally:
        if os.path.exists(PAINT_SETUP):
            os.remove(PAINT_SETUP)
        if had is None:
            del os.environ["FD_PAINT_SETUP"]
        else:
            os.environ["FD_PAINT_SETUP"] = had


def test_with_the_tip_not_on_record_the_jaw_commands_name_fd_paint_status_first():
    """Round 3 (T9), on copies of the rig's files of 2026-11-13 (the jaw shut at raw 2041, nothing said of the jaw,
    the tip not on record as measured, a painting set-up on record, the evening's photos of the tip in a folder):
    `fd tool jaw status` named `fd tool jaw mount moving_outer`, that one `fd tool jaw close`, and that one
    "`fd calib-air --tip-only --save` measures it": jaw first, to a new flight on the real arm, the photos named
    nowhere on the way. `fd paint status` names them first (a hold on record makes them photos of another jaw
    state). Every text of the jaw commands now gives that one order: for a tool without a measured tip beside a
    painting set-up, `fd paint status` is the first command they name."""
    rigs = dict(on_jaw=None, gripper_raw=FIXED, tip_offset=list(TIP_CLOSED), mount="other")       # the rig's file of that night
    with tool_file(brush(**rigs)), paint_setup() as setup:
        assert tooljaw._paint_setup() == fd._paint_setup_on_record() == setup
        arm, bus = rig(TL.active(), pose="rest")
        r = house(cli(["--sim", "status"], arm), ok=True, stage="status")
        assert r["in_place"] is True and r["on_jaw"] is None and r["recorded"]["tip_measured"] is False, r
        assert _commands(r["next"]) == ["fd paint status", "fd tool jaw mount moving_outer", "fd tool jaw mount fixed",
                                        "fd tool jaw mount between"], r["next"]
        assert "FIRST" in r["next"] and "photos" in r["next"] and "before the jaw is closed and held" in r["next"], r["next"]
        assert "fd paint status" in r["say"] and "photo of the gripper" in r["next"], r
        for dry in (["--dry"], []):
            r = house(cli(["--sim", "mount", "moving_outer"] + dry, arm), ok=True, stage="planned" if dry else "mount")
            assert _commands(r["next"]) == ["fd paint status", "fd tool jaw close --dry", "fd tool jaw close"], r["next"]
            assert "NOTHING may be between" in r["next"] and "FIRST" in r["next"] and "fd paint status" in r["say"], r
        for words in (["status"], ["status", "--no-robot"]):
            r = house(cli(["--sim"] + words, arm if len(words) == 1 else None), ok=True, stage="status")
            assert _commands(r["next"])[:2] == ["fd paint status", "fd tool jaw close --dry"] and "fd paint status" in r["say"], (words, r["next"])
        # jaw first all the same: the close puts the hold on record, and from then on `fd paint status` is named for
        # the new measurement, with what becomes of photos taken before the hold
        r = house(cli(["--sim", "close"], arm), ok=True, stage="close")
        assert r["tool_moved"] is False and r["met_fixed_jaw"] and TL.active().gripper_goal_raw == FIXED - 25, r
        way = r["next"].partition("Next: ")[2]
        assert _commands(way) == ["fd paint status", "fd calib-air --tip-only --save"], r["next"]
        assert "refused as they are" in way and "way back" in way and "fd paint status" in r["say"], r
        r = house(cli(["--sim", "status"], arm), ok=True, stage="status")
        assert r["jaw"]["squeeze_on"] and _commands(r["next"])[:2] == ["fd paint status", "fd calib-air --tip-only --save"], r["next"]
        bus.torque_off(("gripper",))
        r = house(cli(["--sim", "status"], arm), ok=True, stage="status")
        assert r["jaw"]["squeeze_on"] is False and "`fd paint status` names the next step" in r["next"], r["next"]
        # a close that puts no hold on record (the way back the refusal of `fd calib-air --from` names): photos first
        r = house(cli(["--sim", "close", "--squeeze", "0"], arm), ok=True, stage="close")
        assert TL.active().gripper_goal_raw is None and r["tool_moved"] is False, r
        assert _commands(r["next"].partition("Next: ")[2]) == ["fd paint status"] and "FIRST" in r["next"], r["next"]
    # a tool whose tip IS on record: the jaw commands lead as they did
    with tool_file(brush(tip_measured={"at": "2026-11-13T17:25:00"}, **rigs)), paint_setup():
        arm, bus = rig(TL.active(), pose="rest")
        r = house(cli(["--sim", "status"], arm), ok=True, stage="status")
        assert _commands(r["next"])[0] == "fd tool jaw mount moving_outer" and "fd paint status" not in r["next"] + r["say"], r["next"]
        r = house(cli(["--sim", "mount", "moving_outer"], arm), ok=True, stage="mount")
        assert _commands(r["next"]) == ["fd tool jaw close --dry", "fd tool jaw close"], r["next"]
        r = house(cli(["--sim", "close"], arm), ok=True, stage="close")
        assert r["tool_moved"] is False and r["next"].endswith("Next: " + tooljaw.NEXT_PAINT), r["next"]
    # no painting set-up on record (a pen; a rig that has not painted yet; a scratch tool file beside the rig's own
    # set-up): `fd paint status` would be refused there, and the jaw commands do not send the tool to it first
    for pinned in (True, False):
        with tool_file(brush(**rigs)), (paint_setup(on_record=False) if pinned else contextlib.nullcontext()):
            assert tooljaw._paint_setup() is None and fd._paint_setup_on_record() is None
            arm, bus = rig(TL.active(), pose="rest")
            r = house(cli(["--sim", "status"], arm), ok=True, stage="status")
            assert _commands(r["next"]) == ["fd tool jaw mount moving_outer", "fd tool jaw mount fixed", "fd tool jaw mount between"], r["next"]
            r = house(cli(["--sim", "mount", "moving_outer"], arm), ok=True, stage="mount")
            assert _commands(r["next"]) == ["fd tool jaw close --dry", "fd tool jaw close"] and "fd paint status" not in r["say"], r["next"]
            r = house(cli(["--sim", "close"], arm), ok=True, stage="close")
            assert r["next"].endswith("Next: " + tooljaw.NEXT_TIP) and _commands(tooljaw.NEXT_TIP)[0] == "fd calib-air --tip-only --save", r["next"]
    assert not _touched


def test_an_abbreviated_option_is_refused_not_completed():
    """Round 3 (T17). argparse completes an abbreviated option by itself: `fd tool jaw close --sq 0` was --squeeze 0,
    `--at` was --at-ready, `mount between --d` a dry run. A word the operator did not write out is not guessed: the
    parser of these words and of every sub-command takes an option only as it is spelled."""
    with tool_file(brush()):
        arm, bus = rig(TL.active())
        before, text = traffic(bus), open(TOOL_PATH).read()
        for words in (["close", "--sq", "0"], ["close", "--squee", "30", "--dry"], ["close", "--at"], ["close", "--d"],
                      ["close", "--no", "--dry"], ["close", "--la", "20"], ["close", "--st", "5"], ["close", "--t", "5"],
                      ["open", "--t", "10"], ["open", "--to", "10", "--at-r"], ["status", "--no"], ["mount", "between", "--d"]):
            r = house(cli(["--sim"] + words, arm), ok=False, stage="options")
            assert r["error"] == "SafetyError" and ("unrecognized arguments" in r["reason"] or "required" in r["reason"]), (words, r)
            assert "usage" in r["next"] and "動かしていません" in r["say"], (words, r)
            assert traffic(bus) == before and jaw_of(bus) == AT_ZERO and open(TOOL_PATH).read() == text, f"{words}: obeyed"
        for parser in [tooljaw.build_parser()] + list(tooljaw.build_parser().commands.values()):
            assert parser.allow_abbrev is False, parser.prog
        # written out, the same words are taken
        r = house(cli(["--sim", "close", "--squeeze", "0", "--dry"], arm), ok=True, stage="planned")
        assert r["plan"]["squeeze_ticks"] == 0, r["plan"]
        r = house(cli(["--sim", "mount", "between", "--dry"], arm), ok=True, stage="planned")
        assert r["saved"] is False and open(TOOL_PATH).read() == text, r


# ----------------------------------------------------------------------------- the commands a result names
# For the texts of the SOURCES only (comments and docstrings explain with `fd tool jaw close --to DEG`): what stands
# for a value there, and a value to parse it with. What a RESULT prints is parsed as it is printed (round 3).
_NEXT_WORDS = {"DEG": "0", "N": "brush", "LEN": "0.06", "x,y,z": "0.01,0,0.06"}
# a word that stands for a value the operator has to give: N, DEG, LEN, DIR, PLAN, BEFORE.jpg, x,y,z, JOINT=LO:HI
_PLACEHOLDER = re.compile(r"[A-Z][A-Z0-9_]*(\.[a-z]{2,4})?|[A-Z]{2,}=.*|.*(x,y|x0,y0|LO:HI).*")


def _texts(res) -> list:
    """Every text of a result that tells the operator what to type."""
    out = [res.get(k) for k in ("next", "reason", "say", "sim_note", "jaw_note")] + list(res.get("notes") or [])
    return [t for t in out if isinstance(t, str)]


def _commands(text: str) -> list:
    return re.findall(r"`(fd [^`]+)`", text)


def _accepted(cmd: str, fill: bool = False):
    """Is `fd ...` ONE command the parsers take as it is typed? Why not (None: it is). Nothing is run: fd's own
    parser reads the words up to the sub-command, the sub-command's parser (tool jaw, calib-air, paint) the rest.
    A bar, an ellipsis or brackets inside the backticks are no command: typed as printed, `a|b` is a shell pipe.
    Neither is a command with a word that only stands for a value (`fd tool jaw open --to DEG`: refused when it is
    typed), or the bare name of a command that reads words of its own (`fd tool jaw`). fill: the lenient reading
    for the texts of the sources, where placeholders are filled in (_NEXT_WORDS)."""
    if "|" in cmd or "..." in cmd or re.search(r"[()\[\]<>{}]", cmd):
        return "not one command: alternatives, an ellipsis or brackets inside the backticks"
    try:
        words = shlex.split(cmd)[1:]
    except ValueError as e:
        return f"cannot be split into words ({e})"
    if fill:
        words = [_NEXT_WORDS.get(w, w) for w in words]
        unknown = [w for w in words if re.fullmatch(r"[A-Z][A-Z_]+", w)]
    else:
        unknown = [w for w in words for v in [w.partition("=")[2] if w.startswith("--") and "=" in w else w]
                   if not v.startswith("-") and _PLACEHOLDER.fullmatch(v)]
    if unknown:
        return f"a placeholder that stands for no value here: {unknown}"
    argv, own, rest = fd._split_own_words(words)
    if not fill and own == "jaw_args" and not rest:
        return "only the name of a command (the words after it are missing)"
    sink = io.StringIO()

    def last(code):
        return sink.getvalue().strip().splitlines()[-1] if sink.getvalue().strip() else f"exit code {code}"

    try:
        with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
            fd.build_parser().parse_args(argv)
    except SystemExit as e:
        return None if e.code == 0 else "fd's parser: " + last(e.code)
    rest = rest[1:] if rest[:1] == ["--"] else rest
    if "--sim" in rest:
        return "--sim after the command"
    try:
        with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
            if own == "jaw_args":
                if rest and rest[0] != "help" and not any(w in ("--help", "-h") for w in rest):      # (those are answered with the usage)
                    tooljaw.build_parser().parse_args(rest)
            elif own == "calib_air_args":
                import aircal_cli
                if rest[:1] not in (["--help"], ["-h"]):
                    aircal_cli.build_parser().parse_args(rest)
            elif own == "paint_args":
                import paintcli
                paintcli.build_parser().parse_args(rest)
    except SystemExit as e:
        return None if e.code == 0 else f"{own}: " + last(e.code)
    except Exception as e:      # (each module's own "cannot read these words")
        return f"{type(e).__name__}: {e}"
    return None


def _named_only(cmd: str, why: str) -> bool:
    """A text may NAME a command or an option without spelling a whole command line ("the bare gripper is moved
    with `fd grip`", "`fd tool set --tip`"): bare command words, or a command that ends with an option. What that
    lacks is no misspelling."""
    bare = re.fullmatch(r"fd( --sim)?( [a-z][a-z-]*){0,3}", cmd) is not None
    return ("required" in why and bare) or ("expected one argument" in why and cmd.split()[-1].startswith("--"))


def test_zy_every_command_a_result_names_is_one_command_that_parses():
    """Round 2 (R9). `fd tool jaw status` on the rig's file said next '`fd tool jaw mount moving_outer|fixed|between`
    records on which jaw ...'. Typed as it was printed, the shell ran `fd tool jaw mount moving_outer` and piped
    it into "fixed" (command not found, exit 127): the tool file then said moving_outer, which lets `fd tool jaw
    close` move the jaw of a tool nobody had looked at. One complete command per pair of backticks: every `fd ...`
    that a result of this suite named (next, reason, say, notes; some hundred results when the whole suite ran
    before, the ones below when this test runs alone), every one that tooljaw.py, tooling.py and hwlock.py spell
    in their own texts, and the texts those modules build for a tool, are handed to fd's parser and to the
    parser of the sub-command they belong to. Nothing is run."""
    closed = dict(gripper_raw=FIXED, tip_offset=list(TIP_CLOSED), mount="other", tip_measured={"at": "2026-11-13T17:25:00"})
    tools = {None: brush(on_jaw=None, **closed), "moving_outer": brush(**closed), "between": brush(on_jaw="between", **closed),
             "fixed": brush(on_jaw="fixed", **closed), "held": held(), "open": brush(), "unmeasured": brush(on_jaw=None, gripper_raw=FIXED)}
    # the scenario itself: the three commands of that `next`, each one a command, and each does what it says
    with tool_file(tools[None]):
        arm, bus = rig(TL.active(), pose="rest")
        r = house(cli(["--sim", "status"], arm), ok=True, stage="status")
        assert _commands(r["next"]) == ["fd tool jaw mount moving_outer", "fd tool jaw mount fixed", "fd tool jaw mount between"], r["next"]
        assert "|" not in r["next"] + r["say"] and r["next"].startswith("look at a photo of the gripper first"), r
        for cmd in _commands(r["next"]):
            where = shlex.split(cmd)[-1]
            res = house(cli(["--sim"] + shlex.split(cmd)[3:] + ["--dry"], arm), ok=True, stage="planned")
            assert res["on_jaw"] == where and res["may_move"] == (where == "moving_outer") and TL.active().on_jaw is None, (cmd, res)
    # results of every kind (also when this test runs alone): status with the jaw in place and elsewhere, the lock,
    # the plan, a close that leaves the tool where it was and one that moves it, the refusals
    for label, tool in tools.items():
        for jaw in (None, FIXED + 100):
            with tool_file(tool):
                arm, bus = rig(TL.active(), jaw=jaw, pose="rest" if tool.gripper_raw == FIXED else "standby")
                for words in (["status"], ["status", "--no-robot"], ["close", "--dry"], ["close"], ["open", "--to", "10", "--dry"],
                              ["close", "--to", "5"], ["open", "--to", "10", "--at-ready"], ["mount", "between", "--dry"], ["wiggle"],
                              ["close", "--squeeze", "200"]):
                    cli(["--sim"] + words, arm)
                for fn in (lambda: TL.go_ready(arm), lambda: TL.hold_jaw(arm), lambda: arm.set_gripper(0.5)):
                    try:
                        fn()
                    except (SafetyError, MotionAborted) as e:
                        _RESULTS.append({"reason": str(e)})
                cli(["close", "--at-ready"])                               # (refused before the arm: --at-ready is for --sim)
                _RESULTS.append(tooljaw._too_low(SafetyError("too low"), "close", False, False, (tool, CFG), tool=tool.name,
                                                 jaw={"from_record_ticks": 100}))
    with tool_file(None):
        cli(["--sim", "status"])
        cli(["--sim", "close"])
        with open(TOOL_PATH, "w") as f:
            f.write("{}")
        cli(["--sim", "mount", "fixed"])
        cli(["mount", "fixed"])
    with tool_file(tools["held"]):
        cli(["status"])                                                    # (the real arm: the lock answers, the file is shown)
    _RESULTS.append({"reason": hwlock.message("The servo bus on /dev/ttyACM0")})     # (the text of every refusal by the lock)
    cmds = sorted({c for res in _RESULTS if isinstance(res, dict) for t in _texts(res) for c in _commands(t)})
    assert len(cmds) >= 20 and len(_RESULTS) >= 100, (len(cmds), len(_RESULTS))
    # Round 3 (T12): what a result prints is parsed AS IT IS PRINTED. Nothing is filled in for a placeholder, and the
    # bare name of a command is no command (`fd tool jaw open --to DEG`, `fd tool attach --name N --tip x,y,z`,
    # `fd grip`, `fd --sim tool jaw` were printed in backticks, and each is refused when it is typed).
    bad = [(c, why) for c in cmds for why in [_accepted(c)] if why]
    assert not bad, bad
    # the texts of the three modules themselves (constants, messages, comments): every command in backticks
    spelled = set()
    for mod in (tooljaw, TL, hwlock):
        with open(mod.__file__, encoding="utf-8") as fh:
            spelled.update(c for c in re.findall(r"`(fd [^`\n]+)`", fh.read()) if "{" not in c and "\\" not in c and '"' not in c)
    assert len(spelled) >= 45, len(spelled)
    bad = [(c, why) for c in sorted(spelled) for why in [_accepted(c, fill=True)] if why and not _named_only(c, why)]
    assert not bad, bad
    # the ones the operator meets first are among them, as they are typed
    for want in ("fd tool jaw mount moving_outer", "fd tool jaw mount fixed", "fd tool jaw mount between", "fd tool jaw close",
                 "fd tool jaw close --dry", "fd tool jaw help", "fd tool jaw status", "fd calib-air --tip-only --save",
                 "fd paint status", "fd ready", "fd tool rest", "fd hw unlock", "fd grip 0.5", "fd grasp", "fd tool attach --help"):
        assert want in cmds, (want, [c for c in cmds if c.startswith(want[:12])])
    # and the check itself can fail: the spellings that did not work are told apart
    for wrong in ("fd tool jaw mount moving_outer|fixed|between", "fd tool jaw close|open", "fd tool attach --replace ...",
                  "fd tool attach --name N (--tip x,y,z | --along-fingers LEN | --across-fingers LEN)", "fd --sim tool jaw ...",
                  "fd tool jaw mount outside", "fd tool jaw close --sim", "fd calib-air --tip-only --safe",
                  "fd tool jaw open --to DEGREES", "fd paint statos", "fd tool jaw wiggle"):
        why = _accepted(wrong)
        assert why and not _named_only(wrong, why), (wrong, why)
    # ... and so are the ones round 3 took out of the results: a placeholder, the bare name of a command
    for wrong in ("fd tool jaw close --to DEG", "fd tool jaw open --to DEG", "fd tool attach --name N --tip x,y,z",
                  "fd tool attach --replace --name brush_thin --tip x,y,z", "fd tool set --joint-limit wrist_flex=LO:HI",
                  "fd tool set --joint-limit JOINT=none", "fd calib-air --from DIR --save", "fd tool jaw", "fd --sim tool jaw",
                  "fd grip", "fd tool jaw open", "fd --sim", "fd tool attach", "fd tool set --tip"):
        assert _accepted(wrong), wrong
    assert _accepted("fd --sim tool jaw close --dry --at-ready") is None and _accepted("fd tool jaw help") is None
    assert _accepted("fd tool jaw open --to 0.2") is None and _accepted("fd grip 0.5") is None and _accepted("fd tool attach --help") is None
    # (the lenient reading, for comments and docstrings only)
    assert _accepted("fd tool jaw close --to DEG", fill=True) is None and _accepted("fd tool jaw", fill=True) is None
    assert _named_only("fd grip", _accepted("fd grip", fill=True)) and _named_only("fd tool set --tip", _accepted("fd tool set --tip", fill=True))


def test_zz_the_rig_files_were_never_touched():
    """Runs LAST: the runner sorts the tests by name (it used to be followed by three tests, one of which takes
    $FD_TOOL away and points tooling at other files: a write by one of them would have passed unseen)."""
    assert os.environ["FD_TOOL"] == TOOL_PATH and os.environ["TAZUNA_HW_LOCK"] == HW_LOCK and os.path.exists(HW_LOCK)
    assert TL.TOOL_FILE == os.path.join(tazuna_paths.ROOT, "tool_active.json")
    assert _tree_files() == _TREE_FILES, "a test created, changed or removed a state file of the rig"
    assert not _touched, f"a test reached for hardware: {_touched}"


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
