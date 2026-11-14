"""Tests for what `fd` does with the words of its command line (tools/cli/fd.py), added with the review of
2026-11-13. Run: python tests/test_cli_words.py [name-part ...]      (no hardware: simulated arm; about 10 s)

What they show.
- The words after `paint`, `calib-air` and `tool jaw` reach those commands untouched, also when the first of
  them is an option: `fd --sim calib-air --dry`, `fd --sim calib-air --tip-only --dry`, `fd --sim tool jaw status`
  and `fd --sim paint status` answer with a result; one leading `--` is dropped; `--help` after each of the three
  ends in a usage text or a result, never in fd's own "unrecognized arguments" (that was the answer to every
  spelling the results of those commands print as their `next`). A --sim typed AFTER one of these commands is
  refused with the spelling that works: handed on, it would look like fd's own and simulate behind fd's back.
- `fd --help` names what was added that day: `tool jaw`, `calib-air`, the eleven sub-commands of `paint`.
- `fd tool set --joint-limit` changes the joints it names and keeps the limits of the others (it used to replace
  the whole set); `JOINT=none` removes one, `none` all; the result lists what remains; a range the arm cannot
  apply is refused before anything is stored; every refusal has stage, next and say; the example is 20:97.
- `fd tool status` shows on which jaw a tool is, the goal that holds the jaw, and whether its tip is measured.
- `fd ready` with a tool answers a refusal of the plan (joint limits that leave no stand-by pose, a held jaw that
  stands elsewhere) as a complete result, and passes on what was done about the jaw and how a low tool was lifted.
- `fd tool rest` refuses a fold that has the tool over the canvas of the painting set-up (the brush sank onto the
  paper there on 2026-11-13) and names `fd paint rest`; `--over-canvas` folds all the same; without a painting
  set-up, or without the painting module's verdict, the command is what it was.

Added with round 2 of the review (2026-11-14):
- A PAINTING tool (a measured tip, a jaw record [since round 3: not one on the fixed jaw alone], or any tool beside
  a painting set-up) is never sent to the pen calibration: `fd tool status`, `attach`, `set`, a refused `fd draw`
  and `--contact` name `fd paint status`;
  `fd tool calibrate` is refused before anything is opened unless --pen declares a pen, and with --pen the result
  says what the stage `tip` does to a tip that was measured in the air. A pen on a rig without a painting set-up is
  what it was.
- The next of a refused `fd ready` is the way out the refusal names for THAT tool (it named `fd tool jaw close`
  for every jaw refusal, also where that command is refused).
- fd's own simulated tool commands say which tool file they work on (a copy of the real tool's record, a copy
  that was left behind), a left-behind copy does not count as "already attached", and `fd --sim tool detach` does
  not say "unlocked" while the simulation mirrors a real tool.
- No usage text of `paint`, `calib-air`, `tool jaw` shows --sim after the command; every command fd's own results
  put into backticks is one command that the parsers take.

Added with round 3 (2026-11-14):
- A pen recorded on the FIXED jaw is no painting tool by that record alone (the next of a refused `fd ready` sends
  a pen there): status, a refused draw and the pen calibration treat it as the pen it is (T7).
- `fd --sim tool detach` while the simulation only mirrors the real tool says that the tool is still on: attached
  true, mirrors NAME, the gripper and park stay locked (T8).
- For a painting tool beside a painting set-up the refusal of a locked command (park, grip, ...), of `fd torque off`
  and of `fd ready` names `fd paint rest` in the spelling of the call, not `fd tool rest`, which is refused over the
  canvas (T14).
- What a result prints inside backticks is read AS IT IS PRINTED: no placeholder is filled in, the bare name of a
  command is no command; the strings of fd.py, tooling.py and tooljaw.py are read the same way (T12). What other
  modules still print that way is listed word for word (_LEFT_ELSEWHERE).

Everything runs on the simulated arm, with scratch files ($FD_TOOL, $FD_PAINT_SETUP, $FD_PAINT_STATE, $FD_PAINT_OUT,
$FD_RUNS) and a scratch hardware lock that is set from the first line to the last. A SIMULATED PASS IS NOT A PASS ON
THE REAL ARM: nothing here was run on it.
"""

import os as _os  # the harness modules live in tools/<feature>/ (see tools/tazuna_paths.py)
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "tools"))
import tazuna_paths  # noqa: E402,F401

import contextlib
import glob
import hashlib
import io
import json
import math
import os
import re
import shlex
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
_TMP = tempfile.mkdtemp(prefix="tz_cliwords_")
TOOL_PATH = os.path.join(_TMP, "tool_active.json")
SETUP_PATH = os.path.join(_TMP, "paint_setup.json")
RUNS = os.path.join(_TMP, "runs")
HW_LOCK = os.path.join(_TMP, "hardware.lock")
os.environ["FD_CONFIG"] = os.path.join(HERE, "testdata", "pen_20261111", "config.json")
os.environ["FD_CAMERA_MODEL"] = os.path.join(HERE, "testdata", "pen_20261111", "camera_model.json")
os.environ["FD_TOOL"] = TOOL_PATH
os.environ.update(FD_PAINT_SETUP=SETUP_PATH, FD_PAINT_STATE=os.path.join(_TMP, "paint_state.json"),
                  FD_PAINT_OUT=os.path.join(_TMP, "out"), FD_RUNS=RUNS, TAZUNA_HW_LOCK=HW_LOCK)
with open(HW_LOCK, "w") as _f:        # locked from the first line to the last: no test here can reach the real arm
    json.dump({"reason": "tests/test_cli_words.py", "since": "test"}, _f)

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

RIG_RUNS = os.path.join(tazuna_paths.ROOT, "runs")


def _tree_files() -> dict:
    """The rig's own state files as they are now (path -> md5, None = no file) and the sizes of its run logs:
    the tests must leave the first alone and add nothing of their own to the second."""
    out = {}
    for name in ("tool_active.json", "tool_active.sim.json", "paint_setup.json", "paint_state.json", "paint_state.sim.json",
                 os.path.join("config", "config.json"), os.path.join("config", "camera_model.json")):
        p = os.path.join(tazuna_paths.ROOT, name)
        try:
            with open(p, "rb") as fh:
                out[p] = hashlib.md5(fh.read()).hexdigest()
        except FileNotFoundError:
            out[p] = None
    return out


def _rig_logs() -> dict:
    return {p: os.path.getsize(p) for p in glob.glob(os.path.join(RIG_RUNS, "*", "events.jsonl"))}


_TREE_FILES, _RIG_LOGS = _tree_files(), _rig_logs()  # before any test ran

import numpy as np  # noqa: E402

import aircal_cli  # noqa: E402
import fd  # noqa: E402
import kinematics as K  # noqa: E402
import paintcli  # noqa: E402
import paintskill as PS  # noqa: E402
import tooling as TL  # noqa: E402
import tooljaw  # noqa: E402
from robot import Arm, load_config  # noqa: E402
from sim_bus import SimBus  # noqa: E402

assert os.path.realpath(fd.RUNS) == os.path.realpath(RUNS), "fd does not honour $FD_RUNS: this suite would write the rig's run log"

import camera as _camera  # noqa: E402
import cv2 as _cv2  # noqa: E402
import servo_bus as _servo_bus  # noqa: E402


def _no_capture(*a, **k):
    _touched.append("cv2.VideoCapture")
    raise RuntimeError(f"test touched hardware: cv2.VideoCapture{a!r}")


# The scratch lock is not the only thing between these tests and the hardware (two of them send a command WITHOUT
# --sim, to show that the lock answers it): no VideoCapture (OpenCV opens a camera without Python's open(), which
# the hook above cannot see), and names for the camera and the servo port that do not exist.
_cv2.VideoCapture = _no_capture
_camera.DEV = "/dev/video_none_for_tests"
_servo_bus.Bus.__init__.__defaults__ = ("/dev/ttyACM_none_for_tests",) + tuple(_servo_bus.Bus.__init__.__defaults__[1:])
assert _servo_bus.Bus().port == "/dev/ttyACM_none_for_tests" and not os.path.exists(_camera.DEV)
np.set_printoptions(precision=4, suppress=True)
BRUSH_LIMITS = {"wrist_flex": [20.0, 97.0], "wrist_roll": [-5.0, 5.0]}      # the brushes of 2026-11-13
FIXED = 2041                                 # where the jaw met the fixed jaw that evening; held with goal 2016
# The canvas of 2026-11-13 and the photo pose, with one well, one dish and one paint: a set-up file as a hand
# writes it. With this canvas the tool's default rest pose (pan 20) has the brush tip 4 cm inside the sheet; folded
# at pan -100 (what was used by hand that evening) it is beside the palette.
SETUP = {"photo_q_deg": [-89.2, -61.7, 36.2, 85.6, 0.0],
         "canvas": {"origin_xy": [0.146, -0.118], "size_mm": [224, 153], "pitch_deg": 65.0},
         "contact": {"z0": 0.00542, "x0": 0.2225, "y0": -0.019, "bx": 0.0362, "by": -0.0062},
         "press_mm": 5.0,
         "wells": {"red": {"xy": [0.0716, 0.2651], "z_mm": 2.5}},
         "water": {"rinse": {"xy": [0.346, 0.192], "z_mm": 6.0}},
         "paints": {"red": {"first": [["water", "rinse", 1], ["scrub", "red", 3]], "reload": [["scrub", "red", 2]]}}}


_LOGGED = []            # the command lines fd.main answered (each leaves one record in the run log)
_RESULTS = []           # (command line, result) of every one of them (the commands they name: test_zy_...)


class Skip(Exception):
    pass


def brush(**over):
    """The thin brush of 2026-11-13 as the rig's tool file had it that night: the jaw shut, nothing said of the jaw."""
    kw = dict(name="brush_thin", tip_offset=[0.016, -0.0011, 0.0558], mount="other", pitch_deg=65.0, gripper_raw=FIXED,
              attached_at="2026-11-13T17:17:45", joint_limits_deg={j: list(v) for j, v in BRUSH_LIMITS.items()})
    kw.update(over)
    return TL.Tool(**kw)


def pen(**over):
    kw = dict(name="pen2", tip_offset=[0.0015, -0.0081, 0.0621], mount="along_fingers", pitch_deg=75.0, gripper_raw=2061)
    kw.update(over)
    return TL.Tool(**kw)


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
def setup_file(setup=SETUP):
    """The painting set-up of the test ($FD_PAINT_SETUP): a dict (written as a hand would), text, or None = no file."""
    if os.path.exists(SETUP_PATH):
        os.remove(SETUP_PATH)
    if setup is not None:
        with open(SETUP_PATH, "w", encoding="utf-8") as f:
            f.write(setup if isinstance(setup, str) else json.dumps(setup, indent=1))
    try:
        yield
    finally:
        if os.path.exists(SETUP_PATH):
            os.remove(SETUP_PATH)


def place(arm, q, gripper=None):
    """Put the simulated arm at joint pose q (as if it had been left there)."""
    bus = arm.bus
    raw = arm.q_to_raw_f(q)
    if gripper is not None:
        raw["gripper"] = gripper
    for j, v in raw.items():
        bus.pos[j] = bus.goal[j] = bus.sp[j] = float(v)


def standing(tool, q_deg=None, torque=True, gripper=None):
    """A simulated bus whose arm stands at q_deg (default: the tool's stand-by pose) with `tool`, holding or limp."""
    bus = SimBus(realtime=False)
    arm = Arm(bus=bus, cfg=load_config(), tool=tool).open()
    q = TL.ready_q(arm, tool) if q_deg is None else np.radians(q_deg)
    place(arm, q, gripper=tool.gripper_raw if gripper is None else gripper)
    if torque:
        arm.torque_on()
        bus.sleep(0.6)
    return bus


def writes(bus):
    return bus.io["writes"], len(bus.log)


def run(*argv, bus=None):
    """fd.main(ARGV) as the launcher calls it. bus: the simulated arm of the test in the place of the one fd
    would make (as fd.make_arm(True) builds it, without its start pose).
    -> (the result, None when the last line printed is not one; exit code; what went to stdout; to stderr)"""
    real = fd.make_arm
    if bus is not None:
        def make(sim):
            assert sim, "a test asked for the REAL arm"
            return Arm(bus=bus, cfg=load_config(), tool=TL.active(sim=True))
        fd.make_arm = make
    out, err = io.StringIO(), io.StringIO()
    try:
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = fd.main(list(argv))
                _LOGGED.append(argv)      # (a command that came back wrote its record into the run log)
            except SystemExit as e:       # argparse: a usage text was printed (0) or the words were refused (2)
                rc = 0 if e.code is None else e.code
    finally:
        fd.make_arm = real
    lines = [ln for ln in out.getvalue().splitlines() if ln.strip()]
    try:
        res = json.loads(lines[-1]) if lines else None
    except ValueError:
        res = None
    if isinstance(res, dict):
        _RESULTS.append((argv, res))
    return (res if isinstance(res, dict) else None), rc, out.getvalue(), err.getvalue()


@contextlib.contextmanager
def heard(module, answer=None):
    """The word lists `module.main` is handed while the block runs. answer None: the real command answers;
    a dict: that is the answer (a test of the words alone)."""
    real, got = module.main, []

    def main(argv=None):
        got.append(list(argv))
        return real(argv) if answer is None else dict(answer)

    module.main = main
    try:
        yield got
    finally:
        module.main = real


def house(r, ok=None, stage=None):
    """A result in the house style: ok, stage, next (English, one line), say (Japanese), reason when not ok."""
    assert isinstance(r, dict) and isinstance(r.get("ok"), bool) and r.get("stage"), r
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


def limits_in_file():
    return json.load(open(TOOL_PATH))["joint_limits_deg"]


# ----------------------------------------------------------------------------- the words of paint, calib-air, tool jaw
def test_words_after_paint_calib_air_and_tool_jaw_arrive_untouched():
    """The words alone (the commands' answers are stand-ins): what fd hands on for a command line."""
    stub = {"ok": True, "stage": "stub", "next": "-", "say": "-"}
    plan = "/no/such/plan.json"
    cases = (
        (aircal_cli, ("--sim", "calib-air", "--dry"), ["--sim", "--dry"]),
        (aircal_cli, ("--sim", "calib-air", "--tip-only", "--dry"), ["--sim", "--tip-only", "--dry"]),
        (aircal_cli, ("--sim", "calib-air", "--from", "DIR", "--save", "--force"), ["--sim", "--from", "DIR", "--save", "--force"]),
        (aircal_cli, ("calib-air", "--sets", "air,low", "--end", "stay"), ["--sets", "air,low", "--end", "stay"]),
        (aircal_cli, ("calib-air",), []),
        (aircal_cli, ("--sim", "calib-air", "--", "--dry"), ["--sim", "--dry"]),                 # one leading "--" is dropped
        (aircal_cli, ("--sim", "calib-air", "--", "--", "--dry"), ["--sim", "--", "--dry"]),     # ... one
        (aircal_cli, ("--sim", "calib-air", "--dry", "--"), ["--sim", "--dry", "--"]),           # ... a leading one
        (aircal_cli, ("--sim", "calib-air", "--help"), ["--sim", "--help"]),
        (tooljaw, ("--sim", "tool", "jaw", "status"), ["--sim", "status"]),
        (tooljaw, ("tool", "jaw", "close", "--to", "closed", "--squeeze", "20", "--dry"), ["close", "--to", "closed", "--squeeze", "20", "--dry"]),
        (tooljaw, ("--sim", "tool", "jaw", "--help"), ["--sim", "--help"]),
        (tooljaw, ("--sim", "tool", "jaw", "--", "status"), ["--sim", "status"]),
        (tooljaw, ("--sim", "tool", "jaw"), ["--sim"]),
        (paintcli, ("--sim", "paint", "status"), ["--sim", "status"]),
        (paintcli, ("paint", "run", plan, "--layer", "0", "--dry"), ["run", plan, "--layer", "0", "--dry"]),
        (paintcli, ("--sim", "paint", "test", "--do", "line:0.25,-0.05:0.25,-0.02:0.6", "--dry"),
         ["--sim", "test", "--do", "line:0.25,-0.05:0.25,-0.02:0.6", "--dry"]),
        (paintcli, ("--sim", "paint", "--help"), ["--sim", "--help"]),
        (paintcli, ("--sim", "paint", "rest", "--help"), ["--sim", "rest", "--help"]),
        (paintcli, ("--sim", "paint", "--", "--help"), ["--sim", "--help"]),
        # words that are options of OTHER fd commands, or look like fd's own: they belong to the command they follow
        (paintcli, ("paint", "contact", "--read", "B.jpg", "A.jpg", "--save", "--dry"), ["contact", "--read", "B.jpg", "A.jpg", "--save", "--dry"]),
        (paintcli, ("paint", "rest", "--pan", "-100", "--clearance", "0.04", "--keep-torque"), ["rest", "--pan", "-100", "--clearance", "0.04", "--keep-torque"]),
    )
    for module, argv, want in cases:
        with heard(module, stub) as got:
            res, rc, out, err = run(*argv)
        assert got == [want], (argv, got)
        assert rc == 0 and res is not None and res["stage"] == "stub" and ("sim" in res) == (argv[0] == "--sim"), (argv, res, err)
        assert "unrecognized arguments" not in err and "usage:" not in err, (argv, err)
    # fd's own words are still fd's: `tool` without `jaw` is parsed by fd, and an unknown option of fd is refused by fd
    with heard(tooljaw, stub) as got, tool_file(None):
        res, rc, out, err = run("--sim", "tool", "status")
        assert got == [] and rc == 0 and res["stage"] == "status" and res["attached"] is False, res
        res, rc, out, err = run("--no-such-option", "paint", "status")
        assert rc == 2 and res is None and "unrecognized arguments: --no-such-option" in err, (rc, err)


def test_a_sim_after_the_command_is_refused_not_guessed():
    """fd hands "--sim" in front of the words when fd itself was called with it. A --sim typed after the command
    would arrive looking the same: the command would work on the simulated arm while fd takes the call for a real
    one (seen with `fd tool jaw --sim status`: simulated jaw numbers in a result without "sim"). So what a command
    is handed starts with --sim if, and only if, fd was called with --sim."""
    stub = {"ok": True, "stage": "stub", "next": "-", "say": "-"}
    plan = "/tmp/a plan.json"
    cases = ((tooljaw, ("tool", "jaw", "--sim", "status"), "tool jaw", "`fd --sim tool jaw status`"),
             (tooljaw, ("tool", "jaw", "close", "--sim", "--dry"), "tool jaw", "`fd --sim tool jaw close --dry`"),
             (aircal_cli, ("calib-air", "--sim", "--dry"), "calib-air", "`fd --sim calib-air --dry`"),
             (aircal_cli, ("calib-air", "--", "--sim", "--tip-only", "--dry"), "calib-air", "`fd --sim calib-air --tip-only --dry`"),
             (aircal_cli, ("calib-air", "--from", "/tmp/photos", "--save", "--sim"), "calib-air", "`fd --sim calib-air --from /tmp/photos --save`"),
             (aircal_cli, ("calib-air", "--sim"), "calib-air", "`fd --sim calib-air`"),
             (paintcli, ("paint", "--sim", "status"), "paint", "`fd --sim paint status`"),
             (paintcli, ("paint", "run", plan, "--layer", "0", "--sim"), "paint", "`fd --sim paint run '/tmp/a plan.json' --layer 0`"),
             (paintcli, ("--sim", "paint", "--sim", "status"), "paint", "`fd --sim paint status`"))       # twice: not guessed either
    for module, argv, name, nxt in cases:
        with heard(module, stub) as got:
            res, rc, out, err = run(*argv)
        house(res, ok=False, stage="options")
        assert got == [], f"`fd {' '.join(argv)}` reached the command: {got}"
        assert rc == 1 and res["error"] == "SafetyError" and res["message"] == res["reason"], res
        assert res["reason"] == f"--sim must stand before the command (fd --sim {name}, then its words). Nothing moved", res
        assert res["next"] == nxt and "fd のすぐ後ろ" in res["say"] and ("sim" in res) == (argv[0] == "--sim"), res
    # the real commands, not stand-ins: nothing simulated answers a call fd took for a real one
    with tool_file(brush()), setup_file():
        res, rc, out, err = run("tool", "jaw", "--sim", "status")
        assert rc == 1 and res["stage"] == "options" and "jaw" not in res and "sim" not in res, res
        res, rc, out, err = run("calib-air", "--sim", "--dry")
        assert rc == 1 and res["stage"] == "options" and "sets" not in res and "sim" not in res, res
        res, rc, out, err = run("paint", "--sim", "status")
        assert rc == 1 and res["stage"] == "options" and "setup" not in res and "sim" not in res, res
    assert not _touched


def test_the_four_spellings_the_results_print_answer_with_a_result():
    """`fd --sim calib-air --dry`, `fd --sim calib-air --tip-only --dry`, `fd --sim tool jaw status`,
    `fd --sim paint status`: the commands themselves, through fd.main. Until 2026-11-13 the first two ended in
    "fd: error: unrecognized arguments: --dry", exit code 2, nothing on stdout."""
    with tool_file(brush()), setup_file():
        for argv, mode in ((("--sim", "calib-air", "--dry"), None), (("--sim", "calib-air", "--tip-only", "--dry"), "tip-only"),
                           (("--sim", "calib-air", "--", "--dry"), None)):
            with heard(aircal_cli) as got:
                res, rc, out, err = run(*argv)
            assert "unrecognized arguments" not in err and res is not None, (argv, rc, err[-300:])
            house(res, ok=True, stage="planned")
            assert rc == 0 and res["sim"] is True and res["tool"] == "brush_thin" and got == [[w for w in argv if w not in ("calib-air", "--")]], (argv, res)
            assert mode is None or res["mode"] == mode, res
        with heard(tooljaw) as got:
            res, rc, out, err = run("--sim", "tool", "jaw", "status")
        house(res, ok=True, stage="status")
        assert rc == 0 and res["sim"] is True and res["tool"] == "brush_thin" and res["recorded"]["gripper_raw"] == FIXED and got == [["--sim", "status"]], res
        with heard(paintcli) as got:
            res, rc, out, err = run("--sim", "paint", "status")
        house(res, stage="status")
        assert rc == (0 if res["ok"] else 1) and res["sim"] is True and res["setup"] == SETUP_PATH and got == [["--sim", "status"]], res
        # without --sim the same words go the same way, and it is the hardware lock that answers, not a parser
        with heard(aircal_cli) as got:
            res, rc, out, err = run("calib-air", "--dry")
        house(res, ok=False, stage="hardware")
        assert rc == 1 and "sim" not in res and got == [["--dry"]] and "hardware is locked" in res["reason"], res
    assert not _touched


def test_help_after_each_of_the_three_ends_in_a_usage_text_or_a_result():
    """`fd calib-air --help`, `fd tool jaw --help`, `fd paint --help` (and -h, with --sim, after a sub-command):
    a usage text of THAT command on stdout with exit code 0, or a result; never fd's "unrecognized arguments"."""
    asked = []
    with tool_file(brush()), setup_file():
        for sim in ((), ("--sim",)):
            for cmd, name in ((("calib-air",), "calib-air"), (("tool", "jaw"), "tool jaw"), (("paint",), "paint"),
                              (("paint", "run"), "paint run"), (("paint", "rest"), "paint rest"), (("paint", "--"), "paint"),
                              (("tool", "jaw", "--"), "tool jaw"), (("calib-air", "--"), "calib-air")):
                for word in ("--help", "-h"):
                    argv = (*sim, *cmd, word)
                    res, rc, out, err = run(*argv)
                    asked.append(argv)
                    assert "unrecognized arguments" not in out + err and "invalid choice" not in out + err, (argv, err[-300:])
                    assert not err.strip(), (argv, err[-300:])
                    if res is None:                     # a usage text, printed by the command's own parser
                        usage = " ".join(out.split("\n\n")[0].split())
                        assert rc == 0 and usage.startswith("usage: fd") and name in usage, (argv, rc, out[:200])
                    else:                               # ... or a result that carries the usage
                        assert isinstance(res.get("ok"), bool) and res.get("stage"), (argv, res)
                        assert "usage" in json.dumps(res).lower() or "help" in json.dumps(res).lower(), (argv, res)
    assert len(asked) == 32 and not _touched


def test_no_usage_text_shows_sim_after_the_command():
    """Round 2 (R16). fd refuses a --sim among the words of `paint`, `calib-air` and `tool jaw` (it stands before
    the command), so no usage text of these commands may show it there, and none may list it as an option of its
    own: `fd paint --help` began "usage: fd paint [-h] [--sim]", the spelling that is refused, and the paint
    parser took `--si` and `--s` for it, which got past fd's refusal."""
    sub = next(a for a in paintcli.build_parser()._actions if getattr(a, "choices", None) and "status" in a.choices)
    cases = [(("paint",), "paint"), *[(("paint", name), "paint") for name in sorted(sub.choices)], (("calib-air",), "calib-air")]
    shown = []
    with tool_file(brush()), setup_file():
        for sim in ((), ("--sim",)):
            for words, name in cases:
                for word in ("--help", "-h"):
                    res, rc, out, err = run(*sim, *words, word)
                    assert res is None and rc == 0 and not err.strip(), (words, rc, err[-300:])
                    usage = " ".join(out.split("\n\n")[0].split())             # the usage block: up to the first empty line
                    assert usage.startswith("usage: fd") and f" {name}" in usage, (words, usage[:200])
                    after = usage.partition(f" {name}")[2]
                    if "--sim" in after or any(ln.strip().startswith("--sim") for ln in out.splitlines()):
                        shown.append(f"fd {' '.join((*sim, *words, word))}: {usage[:70]}")
            # `fd tool jaw --help` answers with a result: its usage line and the option texts of its sub-commands
            res, rc, out, err = run(*sim, "tool", "jaw", "--help")
            assert rc == 0 and res["stage"] == "help" and res["usage"] == tooljaw.USAGE, res
            if "--sim" in res["usage"] or any(ln.strip().startswith("--sim") for ln in res["help"]):
                shown.append(f"fd {' '.join((*sim, 'tool', 'jaw', '--help'))}")
    assert not shown, f"{len(shown)} usage text(s) show --sim after the command, where fd refuses it: {shown[:4]}"
    # ... and what they do show is the spelling that works: fd's own option, before the command
    flat = " ".join(run("calib-air", "--help")[2].split())
    assert flat.startswith("usage: fd [--sim] calib-air"), flat[:80]
    assert not _touched


def test_fd_help_names_the_commands_of_2026_11_13():
    res, rc, out, err = run("--help")
    flat = " ".join(out.split())
    assert rc == 0 and res is None and flat.startswith("usage: fd"), out[:200]
    # the module text (the description of `fd --help`)
    for text in ("tool attach / status / set / rest / detach / calibrate / jaw", "fd tool jaw status|mount|close|open",
                 "fd calib-air [--tip-only] [--dry]", "calib-roll, calib-air, table-fit", "hw lock / unlock / status",
                 "paint status / brush / check / look / locate / test / contact / preview / run / rinse / rest", "fd paint rest",
                 "fd --sim calib-air --dry"):
        assert text in flat, text
    doc = " ".join(fd.__doc__.split())
    assert "jaw" in doc and "calib-air" in doc
    # the line of each command in the list
    assert "status / brush / check / look / locate / test / contact / preview / run / rinse / rest (`fd paint --help`" in flat
    assert "attach / status / set / rest / detach / calibrate / jaw" in flat and "`fd calib-air` measures the tip" in flat
    assert "Older copies of the harness do not know the lock" in flat
    assert "sub-commands: status, brush, check, look, locate, test, contact, preview, run, rinse, rest" in " ".join(fd.cmd_paint.__doc__.split())
    res, rc, out, err = run("tool", "--help")
    flat = " ".join(out.split())
    assert rc == 0 and "jaw" in flat and "`fd tool jaw --help`" in flat and "use `fd paint rest`" in flat, flat[:400]
    res, rc, out, err = run("tool", "rest", "--help")
    assert rc == 0 and "--over-canvas" in out and "fd paint rest" in " ".join(out.split()), out
    # every sub-command the paint command has is in fd's list of them (and no other)
    flat = " ".join(run("--help")[2].split())
    listed = [w.strip() for w in flat.split("watercolour with a brush taped to the gripper: ")[1].split(" (`fd paint --help`")[0].split("/")]
    sub = next(a for a in paintcli.build_parser()._actions if getattr(a, "choices", None) and "status" in a.choices)
    assert sorted(listed) == sorted(sub.choices), (listed, sorted(sub.choices))


# ----------------------------------------------------------------------------- fd tool set --joint-limit
def test_joint_limit_of_one_joint_keeps_the_others():
    """2026-11-13: `fd tool set --joint-limit wrist_flex=20:95` on the brush (wrist_flex 20..97, wrist_roll -5..5)
    left {wrist_flex: [20, 95]}: the roll limit, which keeps the handle off the forearm, was gone without a word."""
    with tool_file(brush()):
        res, rc, out, err = run("--sim", "tool", "set", "--joint-limit", "wrist_flex=20:95", "--no-robot")
        house(res, ok=True, stage="set")
        want = {"wrist_flex": [20.0, 95.0], "wrist_roll": [-5.0, 5.0]}
        assert rc == 0 and res["changed"] == ["joint_limits_deg"] and res["saved"] and limits_in_file() == want, res
        assert res["joint_limits_deg"] == want and res["tool"]["joint_limits_deg"] == want and res["before"]["joint_limits_deg"] == BRUSH_LIMITS
        assert "wrist_flex 20..95" in res["say"] and "wrist_roll -5..5" in res["say"], res["say"]
        # the arm of the next command has both
        lim = fd.make_arm(True).open().q_limits()
        assert abs(math.degrees(lim["wrist_roll"][1]) - 5.0) < 1e-6 and abs(math.degrees(lim["wrist_flex"][1]) - 95.0) < 1e-6
        # the same words again change nothing; a dry run stores nothing but shows the set it would leave
        res, rc, out, err = run("--sim", "tool", "set", "--joint-limit", "wrist_flex=20:95", "--no-robot")
        assert rc == 0 and res["changed"] == [] and res["saved"] is False and res["joint_limits_deg"] == want, res
        res, rc, out, err = run("--sim", "tool", "set", "--joint-limit", "shoulder_pan=-95:95", "--dry", "--no-robot")
        assert rc == 0 and res["stage"] == "planned" and res["saved"] is False and limits_in_file() == want, res
        assert res["joint_limits_deg"] == dict(want, shoulder_pan=[-95.0, 95.0]) and "保存していません" in res["say"], res
        # several joints in one command, each named joint replaced, the unnamed one kept
        res, rc, out, err = run("--sim", "tool", "set", "--joint-limit", "wrist_flex=20:97", "--joint-limit", "shoulder_pan=-95:95",
                                "--no-robot")
        assert rc == 0 and limits_in_file() == dict(BRUSH_LIMITS, shoulder_pan=[-95.0, 95.0]) == res["joint_limits_deg"], res
        # JOINT=none removes that joint's limit and nothing else; removing what is not there changes nothing
        res, rc, out, err = run("--sim", "tool", "set", "--joint-limit", "shoulder_pan=none", "--no-robot")
        house(res, ok=True, stage="set")
        assert rc == 0 and res["changed"] == ["joint_limits_deg"] and limits_in_file() == BRUSH_LIMITS == res["joint_limits_deg"], res
        assert "shoulder_pan" not in res["say"] and "wrist_roll -5..5" in res["say"], res["say"]
        res, rc, out, err = run("--sim", "tool", "set", "--joint-limit", "shoulder_pan=none", "--no-robot")
        assert rc == 0 and res["changed"] == [] and res["saved"] is False and limits_in_file() == BRUSH_LIMITS, res
        # the words are read in the order given: `none` first makes the rest the whole set (what the option used to do)
        res, rc, out, err = run("--sim", "tool", "set", "--joint-limit", "none", "--joint-limit", "wrist_flex=25:90", "--no-robot")
        assert rc == 0 and limits_in_file() == {"wrist_flex": [25.0, 90.0]} == res["joint_limits_deg"], res
        res, rc, out, err = run("--sim", "tool", "set", "--joint-limit", "wrist_roll=-5:5", "--joint-limit", "wrist_flex=none",
                                "--joint-limit", "wrist_flex=20:97", "--no-robot")
        assert rc == 0 and limits_in_file() == BRUSH_LIMITS, res
        # `none` alone removes them all, and the result says that none remain
        res, rc, out, err = run("--sim", "tool", "set", "--joint-limit", "none", "--no-robot")
        house(res, ok=True, stage="set")
        assert rc == 0 and res["changed"] == ["joint_limits_deg"] and limits_in_file() is None and res["joint_limits_deg"] == {}, res
        assert "joint_limits_deg" not in res["tool"] and "関節制限はありません" in res["say"], res
        assert TL.active().name == "brush_thin" and TL.active().tip_offset == [0.016, -0.0011, 0.0558]      # nothing else changed
        # a command without the option says nothing about limits and leaves them alone
        TL.save(brush())
        res, rc, out, err = run("--sim", "tool", "set", "--note", "looked at it", "--no-robot")
        assert rc == 0 and res["changed"] == ["notes"] and limits_in_file() == BRUSH_LIMITS and "関節制限" not in res["say"], res
        assert res["joint_limits_deg"] == BRUSH_LIMITS
    # attach: the same words; there is nothing to keep, and removing a limit that is not there is not an error
    with tool_file(None):
        attach = ("--sim", "tool", "attach", "--name", "brush", "--tip", "0.0085,0.0007,0.0611", "--pitch", "65", "--no-robot")
        res, rc, out, err = run(*attach, "--joint-limit", "wrist_flex=20:97", "--joint-limit", "wrist_roll=-5:5", "--joint-limit", "elbow_flex=none")
        assert rc == 0 and res["stage"] == "attached" and limits_in_file() == BRUSH_LIMITS == res["tool"]["joint_limits_deg"], res


def test_joint_limits_the_arm_cannot_apply_are_refused_before_they_are_stored():
    """2026-11-13: `fd --sim tool set --joint-limit wrist_flex=200:300` answered ok true, saved; `fd tool rest
    --dry`, `fd ready` and `fd paint check` then all refused ("no pose would be left") until it was set again."""
    with tool_file(brush()):
        text = open(TOOL_PATH).read()
        for words in (("wrist_flex=200:300",), ("wrist_flex=150:170",), ("shoulder_pan=120:140",),
                      ("wrist_roll=-3:3", "elbow_flex=-200:-150")):
            opts = [w for lim in words for w in ("--joint-limit", lim)]
            for extra in ((), ("--no-robot",), ("--dry",)):
                bus = standing(TL.active(), torque=False)
                before = writes(bus)
                res, rc, out, err = run("--sim", "tool", "set", *opts, *extra, bus=bus)
                house(res, ok=False, stage="options")
                joint = words[-1].split("=")[0]
                assert rc == 1 and res["error"] == "SafetyError" and res["saved"] is False and res["tool"] == "brush_thin", res
                assert "no pose would be left" in res["reason"] and joint in res["reason"] and res["message"] == res["reason"], res
                assert "fd tool set with --joint-limit" in res["next"] and "JOINT=none" in res["next"] and "`fd tool status`" in res["next"], res
                assert "`fd tool set" not in res["next"], res["next"]         # (words: the joint and its range are the operator's to give)
                assert res["joint_limits_deg"] == BRUSH_LIMITS and "wrist_flex 20..97" in res["say"], res
                assert open(TOOL_PATH).read() == text and writes(bus) == before, (words, extra)
        # ... and so the commands that move still have their poses
        res, rc, out, err = run("--sim", "tool", "rest", "--dry", "--over-canvas")
        assert rc == 0 and res["ok"], res
        res, rc, out, err = run("--sim", "ready")
        assert rc == 0 and res["ok"] and res["q_deg"]["wrist_flex"] <= 97.3, res
        # a range the arm can apply but that leaves no stand-by pose is stored, and the result says what is missing
        res, rc, out, err = run("--sim", "tool", "set", "--joint-limit", "wrist_flex=-100:-20")
        house(res, ok=True, stage="set")
        assert rc == 0 and res["saved"] and limits_in_file()["wrist_flex"] == [-100.0, -20.0], res
        assert "error" in res["arm"]["standby"] and any("standby" in n for n in res["notes"]), res
        assert "no stand-by pose" in res["next"] and "fd ready" in res["next"] and "待機姿勢" in res["say"], res
        res, rc, out, err = run("--sim", "tool", "set", "--joint-limit", "wrist_flex=20:97")       # back: nothing is missing
        assert rc == 0 and limits_in_file() == BRUSH_LIMITS and res["notes"] == [] and "no stand-by pose" not in res["next"], res
        assert res["arm"]["rest"]["tip_height_mm"] == 50.0 and "q_deg" in res["arm"]["standby"], res["arm"]
    with tool_file(None):       # attach: refused as well, and no tool file appears
        res, rc, out, err = run("--sim", "tool", "attach", "--name", "brush", "--tip", "0.0085,0.0007,0.0611", "--no-robot",
                                "--joint-limit", "wrist_flex=200:300")
        house(res, ok=False, stage="options")
        assert rc == 1 and "no pose would be left" in res["reason"] and not os.path.exists(TOOL_PATH), res


def test_joint_limit_words_that_cannot_be_read_are_complete_refusals():
    """They used to be {"ok": false, "error": "SafetyError", "message": ...} and nothing else."""
    bad = ("foo", "nojoint=1:2", "wrist_flex=50:20", "wrist_flex=20", "wrist_flex=a:b", "wrist_flex=", "=20:97", "gripper=0:40",
           "wrist_flex=20:97:3", "nojoint=none", "wrist_flex=nan:97", "wrist_flex=20:400", "NONE=1:2")
    with tool_file(brush()):
        text = open(TOOL_PATH).read()
        for word in bad:
            for more in ((), ("--joint-limit", "wrist_roll=-3:3")):          # (a good word beside it is not stored either)
                res, rc, out, err = run("--sim", "tool", "set", "--joint-limit", word, *more, "--no-robot")
                house(res, ok=False, stage="options")
                assert rc == 1 and res["error"] == "SafetyError" and res["saved"] is False and res["message"] == res["reason"], (word, res)
                assert "fd tool set with --joint-limit" in res["next"] and "wrist_flex=20:97" in res["next"] and "`fd tool set" not in res["next"], (word, res)
                assert "記録は変えていません" in res["say"] and res["joint_limits_deg"] == BRUSH_LIMITS, (word, res)
                assert open(TOOL_PATH).read() == text, word
        res, rc, out, err = run("--sim", "tool", "set", "--joint-limit", "foo", "--no-robot")
        assert "JOINT=LO:HI" in res["reason"] and "JOINT=none" in res["reason"] and "wrist_flex=20:97" in res["reason"], res
    with tool_file(None):
        for word in bad:
            res, rc, out, err = run("--sim", "tool", "attach", "--name", "brush", "--tip", "0.0085,0.0007,0.0611", "--no-robot",
                                    "--joint-limit", word)
            house(res, ok=False, stage="options")
            assert rc == 1 and res["error"] == "SafetyError" and not os.path.exists(TOOL_PATH), (word, res)
        res, rc, out, err = run("--sim", "tool", "set", "--joint-limit", "wrist_flex=20:97")     # no tool: said before the words
        assert rc == 1 and res["stage"] == "tool" and "no tool is attached" in res["reason"], res


def test_the_example_of_a_joint_limit_is_the_measured_cap():
    """103 is the wrist's own limit, where a move stopped 3.7 deg short with the handle against the arm on
    2026-11-13; the cap that was measured is 97. The help and the error text showed wrist_flex=20:103."""
    for words in (("tool", "set", "--help"), ("tool", "attach", "--help")):
        res, rc, out, err = run(*words)
        flat = " ".join(out.split())
        assert rc == 0 and "wrist_flex=20:97" in flat and "20:103" not in flat, flat[-600:]
        assert "JOINT=none" in flat and "keeps the limits of the others" in flat, flat[-600:]
    with tool_file(brush()):
        res, rc, out, err = run("--sim", "tool", "set", "--joint-limit", "wrist_flex", "--no-robot")
        assert "wrist_flex=20:97" in res["reason"] and "103" not in res["reason"] + res["next"], res
    with open(os.path.join(tazuna_paths.TOOLS, "cli", "fd.py"), encoding="utf-8") as f:
        assert "20:103" not in f.read(), "tools/cli/fd.py still shows wrist_flex=20:103 somewhere"


# ----------------------------------------------------------------------------- fd tool status, fd ready
def test_tool_status_shows_the_jaw_record_and_whether_the_tip_is_measured():
    with tool_file(brush()):                    # the rig's file of that night: nothing said of the jaw, the tip not recorded as measured
        res, rc, out, err = run("--sim", "tool", "status")
        t = res["tool"]
        assert rc == 0 and t["tip_measured"] is False and "on_jaw" not in t and "gripper_goal_raw" not in t and t["gripper_raw"] == FIXED, t
    held = brush(on_jaw="moving_outer", gripper_goal_raw=FIXED - 25, tip_measured={"at": "2026-11-13T17:25:00", "photos": 34, "rms_px": 6.6})
    with tool_file(held):
        res, rc, out, err = run("--sim", "tool", "status")
        t = res["tool"]
        assert rc == 0 and t["on_jaw"] == "moving_outer" and t["gripper_goal_raw"] == FIXED - 25 and t["tip_measured"] is True, t
        # set shows it before and after: a new tip is not a measured one
        res, rc, out, err = run("--sim", "tool", "set", "--tip", "0.017,-0.001,0.056", "--no-robot")
        assert rc == 0 and res["before"]["tip_measured"] is True and res["tool"]["tip_measured"] is False, res
        assert res["tool"]["on_jaw"] == "moving_outer" and res["tool"]["gripper_goal_raw"] == FIXED - 25, res
    with tool_file(brush(on_jaw="between")):
        t = run("--sim", "tool", "status")[0]["tool"]
        assert t["on_jaw"] == "between" and "gripper_goal_raw" not in t and t["tip_measured"] is False, t
    with tool_file(None):
        res, rc, out, err = run("--sim", "tool", "attach", "--name", "pen", "--along-fingers", "0.055", "--no-robot", "--dry")
        assert rc == 0 and res["tool"]["tip_measured"] is False and "on_jaw" not in res["tool"], res


def test_ready_refused_with_a_tool_is_a_complete_result():
    """It was {"ok": false, "error": "SafetyError", "message": ...}: no stage, no next, no say."""
    # joint limits of the tool that leave no stand-by pose
    tool = brush(joint_limits_deg={"wrist_flex": [-100.0, -20.0]})
    with tool_file(tool):
        for torque in (False, True):
            bus = standing(tool, q_deg=[40.0, -100.0, 85.0, -30.0, 0.0], torque=torque)
            before, pos = writes(bus), dict(bus.pos)
            res, rc, out, err = run("--sim", "ready", bus=bus)
            house(res, ok=False, stage="plan")
            assert rc == 1 and res["error"] == "SafetyError" and res["tool"] == "brush_thin" and res["moved"] is False, res
            assert res["torque"] == ("on" if torque else "off") and res["message"] == res["reason"] and "stand-by" in res["reason"], res
            assert "fd tool set with --joint-limit" in res["next"] and "none (remove them all)" in res["next"] and "`fd tool rest`" in res["next"], res
            assert "`fd tool status`" in res["next"] and "`fd tool set" not in res["next"], res["next"]
            assert "動かしていません" in res["say"] and "q_deg" in res and res["sim"] is True, res
            assert writes(bus) == before and bus.pos == pos, "a refused `ready` wrote to the servos"
    # a jaw that is held shut by a squeeze and stands elsewhere than the tool file says
    held = brush(on_jaw="moving_outer", gripper_goal_raw=FIXED - 25, tip_measured={"at": "2026-11-13T17:25:00"})
    with tool_file(held):
        bus = standing(held, q_deg=np.degrees(TL.rest_q(Arm(bus=SimBus(), cfg=load_config(), tool=held).open(), held)), torque=False,
                       gripper=2182)
        before, pos = writes(bus), dict(bus.pos)
        res, rc, out, err = run("--sim", "ready", bus=bus)
        house(res, ok=False, stage="plan")
        assert rc == 1 and res["moved"] is False and res["torque"] == "off" and "fd tool jaw close" in res["message"], res
        assert "fd tool jaw status" in res["next"] and "fd tool jaw close" in res["next"] and "fd calib-air --tip-only" in res["next"], res
        assert writes(bus) == before and bus.pos == pos and not any(bus.torque.values())
        # with the jaw where the file has it: the arm stands by, and the result says what was done about the squeeze
        res, rc, out, err = run("--sim", "ready")
        assert rc == 0 and res["ok"] and res["tool"] == "brush_thin" and res["jaw"]["held"] is True and res["jaw"]["wrote"] is True, res
        assert "stage" not in res or res["stage"] != "plan"
    # a tool that lies low is lifted first, and the result says how
    tool = pen()
    with tool_file(tool):
        arm = Arm(bus=SimBus(), cfg=load_config(), tool=tool).open()
        q_low = TL.work_q(arm, tool, (0.25, 0.0), 0.015)[0]
        bus = standing(tool, q_deg=np.degrees(q_low), torque=True)
        res, rc, out, err = run("--sim", "ready", bus=bus)
        assert rc == 0 and res["ok"] and res["lifted_first"] is True and isinstance(res["lift_via"], str) and res["lift_via"], res
        assert "jaw" not in res and 0.0 < res["min_tip_height_m"] < TL.LIFT_HEIGHT, res
        # from the rest pose nothing is lifted: no such key
        res, rc, out, err = run("--sim", "ready")
        assert rc == 0 and res["lifted_first"] is False and "lift_via" not in res, res
    # a move that is stopped on the way: the arm holds, the result says so
    tool = pen()
    with tool_file(tool):
        bus = standing(tool, q_deg=np.degrees(TL.rest_q(Arm(bus=SimBus(), cfg=load_config(), tool=tool).open(), tool)), torque=True)
        real = TL.free_move

        def stopped(arm, q_to, speed, **kw):
            q = arm.read_q()
            arm.run_path([q, q + (np.asarray(q_to) - q) * 0.3], 20.0, settle=0.2)
            raise fd.MotionAborted("free move (direct): stopped because the tool tip was 15 mm above the table (measured); the arm holds where it stopped")

        TL.free_move = stopped
        try:
            res, rc, out, err = run("--sim", "ready", bus=bus)
        finally:
            TL.free_move = real
        house(res, ok=False, stage="move")
        assert rc == 1 and res["error"] == "MotionAborted" and res["moved"] is True and res["torque"] == "on", res
        assert "fd state" in res["next"] and "fd ready" in res["next"] and "止まりました" in res["say"], res


def test_the_next_of_a_refused_ready_is_the_way_out_for_that_tool():
    """Round 2 (R17). The rig's own tool file of 2026-11-13 (the jaw shut at raw 2041, nothing said about the jaw
    the brush is on) with the limp jaw 14 ticks off, and a pen registered with `fd tool attach` whose limp jaw
    moved 20 ticks: `fd ready` is refused (the 8 ticks of tooling.jaw_tolerance), and its next said "`fd tool jaw
    close` closes it on the fixed jaw again". That command is refused for a tool whose file does not say the jaw;
    for the pen the word that helps, `fd tool jaw mount fixed`, stood only in the reason. The next is now the
    refusal's own last sentence, which names the way out for THAT tool."""
    def limp(tool, jaw):
        arm = Arm(bus=SimBus(), cfg=load_config(), tool=tool).open()
        return standing(tool, q_deg=np.degrees(TL.rest_q(arm, tool)), torque=False, gripper=jaw)

    def refused_ready(tool, jaw):
        bus = limp(tool, jaw)
        before, pos = writes(bus), dict(bus.pos)
        res, rc, out, err = run("--sim", "ready", bus=bus)
        house(res, ok=False, stage="plan")
        assert rc == 1 and res["moved"] is False and res["torque"] == "off" and f"raw {jaw}" in res["reason"], res
        assert writes(bus) == before and bus.pos == pos and not any(bus.torque.values()), "a refused `ready` wrote to the servos"
        way = res["reason"].partition("Next: ")[2]
        assert way and res["next"] == "nothing moved. " + way[0].upper() + way[1:], res["next"]
        return res

    def typed(cmd, bus=None):
        """One command of a `next`, typed as it is printed, on the simulated arm."""
        words = cmd.split()
        assert words[0] == "fd" and "--sim" not in words, cmd
        return run("--sim", *words[1:], bus=bus)[0]

    cmds = lambda res: [c for c in res["next"].split("`")[1::2] if c.startswith("fd ")]      # noqa: E731
    # the rig's own file, the limp jaw 14 ticks from its record
    with tool_file(brush()):
        res = refused_ready(TL.active(), FIXED + 14)
        assert cmds(res)[:3] == ["fd tool jaw status", "fd tool jaw mount moving_outer", "fd tool jaw close"], res["next"]
        assert "`fd tool jaw mount fixed`" in res["next"] and "photo" in res["next"] and res["next"].index("mount moving_outer") < res["next"].index("jaw close"), res["next"]
        # (what was printed before: `fd tool jaw close` first. As typed, that is refused for this tool)
        r = typed("fd tool jaw close", bus=limp(TL.active(), FIXED + 14))
        assert r["ok"] is False and r["stage"] == "tool" and "does not say on which jaw" in r["reason"], r
        # the commands of the next, in their order: each is taken
        bus = limp(TL.active(), FIXED + 14)
        tooljaw_arm = tooljaw.make_arm
        tooljaw.make_arm = lambda sim: Arm(bus=bus, cfg=load_config(), tool=TL.active(sim=True)).open()
        try:
            r = typed(cmds(res)[0])
            assert r["ok"] and r["stage"] == "status" and r["in_place"] is False and r["jaw"]["from_record_ticks"] == 14, r
            r = typed(cmds(res)[1])
            assert r["ok"] and r["stage"] == "mount" and TL.active().on_jaw == "moving_outer", r
            r = typed(cmds(res)[2])
            assert r["ok"] and r["stage"] == "close" and r["met_fixed_jaw"] and TL.active().gripper_goal_raw is not None, r
        finally:
            tooljaw.make_arm = tooljaw_arm
    # a pen registered with `fd tool attach` (the jaw as it stood then, nothing said about the jaw), moved 20 ticks
    with tool_file(pen()):
        res = refused_ready(TL.active(), 2061 + 20)
        assert "`fd tool jaw mount fixed`" in res["next"] and "does not carry it" in res["next"], res["next"]
        r = typed("fd tool jaw mount fixed")
        assert r["ok"] and TL.active().on_jaw == "fixed", r
        res, rc, out, err = run("--sim", "ready", bus=limp(TL.active(), 2061 + 20))      # the moving jaw no longer matters
        assert rc == 0 and res["ok"] and res["tool"] == "pen2", res
    # a tool between the jaws, one on the moving jaw without a hold, one that is held: each its own way out
    with tool_file(brush(on_jaw="between")):
        res = refused_ready(TL.active(), FIXED + 14)
        assert cmds(res)[0] == "fd tool jaw status" and "fd tool attach --help" in cmds(res), res["next"]
        assert "fd tool attach with --replace, --name brush_thin and the tip" in res["next"] and "x,y,z" not in res["next"], res["next"]
        assert "fd tool jaw close" not in cmds(res), res["next"]
    with tool_file(brush(on_jaw="moving_outer")):
        res = refused_ready(TL.active(), FIXED + 14)
        assert cmds(res)[:2] == ["fd tool jaw status", "fd tool jaw close"] and cmds(res)[-1] == "fd calib-air --tip-only --save", res["next"]
    with tool_file(brush(on_jaw="moving_outer", gripper_goal_raw=FIXED - 25, tip_measured={"at": "2026-11-13T17:25:00"})):
        res = refused_ready(TL.active(), FIXED + 60)
        assert cmds(res) == ["fd tool jaw status", "fd tool jaw close", "fd calib-air --tip-only --save"], res["next"]
    assert not _touched


# ----------------------------------------------------------------------------- fd tool rest and the canvas
def _stand_in_report(arm, tool, setup, pan_deg, clearance_m):
    """What paintskill.rest_report says, for a painting module that does not have it yet: the folded tip, and
    where it sinks to, must lie 30 mm outside the canvas."""
    q = TL.rest_q(arm, tool, clearance_m=clearance_m, pan_deg=pan_deg)
    p = K.fk(q, K.tool_offset(*tool.tip_offset))[:3, 3]
    clear = PS.canvas_distance(setup, float(p[0]), float(p[1]))
    return {"ok": clear >= 0.03, "why": "" if clear >= 0.03 else f"the tip at ({p[0]:.3f}, {p[1]:.3f}) is {clear * 1000:.0f} mm from the canvas"}


@contextlib.contextmanager
def rest_report(fn="real"):
    """paintskill.rest_report for the duration: the module's own (the stand-in when it has none), another
    function, or None = a painting module without it. Yields the list of calls [(pan_deg, clearance_m), ...]."""
    had = hasattr(PS, "rest_report")
    real = getattr(PS, "rest_report", None)
    calls = []
    use = (real or _stand_in_report) if fn == "real" else fn
    if use is None:
        if had:
            del PS.rest_report
    else:
        def report(arm, tool, setup, pan_deg, clearance_m):
            calls.append((round(float(pan_deg), 1), round(float(clearance_m), 3)))
            return use(arm, tool, setup, pan_deg, clearance_m)
        PS.rest_report = report
    try:
        yield calls
    finally:
        if had:
            PS.rest_report = real
        elif hasattr(PS, "rest_report"):
            del PS.rest_report


def test_tool_rest_over_the_canvas_is_refused_and_names_paint_rest():
    """2026-11-13: with the torque off in the tool's rest pose the wrist sank until the brush stood on the paper;
    the rest pose is over the picture. `fd paint rest` folds beside it; `fd tool rest` did not know of a canvas."""
    tool = brush()
    photo = SETUP["photo_q_deg"]
    with tool_file(tool), setup_file(), rest_report() as calls:
        PS.check_setup(json.load(open(SETUP_PATH)), SETUP_PATH)       # (the set-up of this test is one the painting module takes)
        # the arm holds at the photo pose, as after every painting command
        bus = standing(tool, q_deg=photo, torque=True)
        before, pos = writes(bus), dict(bus.pos)
        for extra in (("--dry",), (), ("--keep-torque",), ("--clearance", "0.07"), ("--pan", "35")):
            res, rc, out, err = run("--sim", "tool", "rest", *extra, bus=bus)
            house(res, ok=False, stage="canvas")
            assert rc == 1 and res["error"] == "SafetyError" and res["tool"] == "brush_thin" and res["moved"] is False, (extra, res)
            assert res["torque"] == "on" and res["message"] == res["reason"] and "canvas" in res["reason"] and "Nothing moved" in res["reason"], res
            assert res["canvas"]["ok"] is False and res["canvas"]["judged"] == "the pose asked for" and res["canvas"]["setup"] == SETUP_PATH, res
            assert res["next"].startswith("`fd --sim paint rest` folds the arm"), res
            again = res["next"].split("`")[3].split()                 # the second command of `next`, as it is typed
            assert again[:4] == ["fd", "--sim", "tool", "rest"] and "--over-canvas" in again and all(w in again for w in extra), res["next"]
            assert "fd paint rest" in res["say"] and "キャンバス" in res["say"] and "動かしていません" in res["say"], res
            assert writes(bus) == before and bus.pos == pos, f"a refused `tool rest {extra}` wrote to the servos"
        assert calls and calls[0] == (20.0, 0.05) and calls[-1] == (35.0, 0.05), calls
        # the pose that was used by hand that evening is beside the palette: planned, and the plan names the command
        res, rc, out, err = run("--sim", "tool", "rest", "--pan", "-100", "--clearance", "0.04", "--dry", bus=bus)
        house(res, ok=True, stage="planned")
        assert rc == 0 and res["canvas"]["ok"] is True and res["canvas"]["pan_deg"] == -100.0 and "overridden" not in res["canvas"], res
        assert res["next"] == "`fd --sim tool rest --pan -100 --clearance 0.04` does it" and writes(bus) == before, res
        # --over-canvas: the pose over the sheet is planned all the same, and the result says what was overridden
        res, rc, out, err = run("--sim", "tool", "rest", "--dry", "--over-canvas", bus=bus)
        house(res, ok=True, stage="planned")
        assert rc == 0 and res["canvas"]["ok"] is False and res["canvas"]["overridden"] is True and "--over-canvas" in res["say"], res
        assert res["next"] == "`fd --sim tool rest --over-canvas` does it" and writes(bus) == before and bus.pos == pos, res
        # folded there for real: off the canvas, torque off
        res, rc, out, err = run("--sim", "tool", "rest", "--pan", "-100", "--clearance", "0.04", bus=bus)
        house(res, ok=True, stage="rest")
        assert rc == 0 and res["moved"] is True and res["torque"] == "off" and res["canvas"]["ok"] is True, res
        assert abs(res["q_deg"]["shoulder_pan"] + 100.0) < 1.5 and not any(bus.torque.values()), res["q_deg"]
        # an arm that lies limp in a rest pose is left where it is whatever pan is asked for: what counts is where it LIES.
        # Beside the palette: fine, although the default pan (over the sheet) is what the words say
        n = len(calls)
        res, rc, out, err = run("--sim", "tool", "rest", bus=bus)
        assert rc == 0 and res["ok"] and res["moved"] is False and res["canvas"]["ok"] is True, res
        assert res["canvas"]["judged"] == "where the arm lies folded, torque off" and calls[n:] == [(-100.0, 0.04)], (res["canvas"], calls[n:])
    with tool_file(tool), setup_file(), rest_report() as calls:
        # ... and limp over the sheet (where every simulated arm starts, and where the real one lay that morning): refused,
        # although the words name the good pose; `fd paint rest` is what moves it away
        res, rc, out, err = run("--sim", "tool", "rest", "--pan", "-100", "--clearance", "0.04")
        house(res, ok=False, stage="canvas")
        assert rc == 1 and res["torque"] == "off" and res["canvas"]["pan_deg"] == 20.0 and calls == [(20.0, 0.05)], (res, calls)
        assert "lies folded at pan 20" in res["reason"] and "`fd --sim paint rest`" in res["next"], res
        assert "畳まれています" in res["say"] and "fd paint rest" in res["say"] and "動かしていません" in res["say"], res["say"]
        assert res["next"].split("`")[3] == "fd --sim tool rest --pan -100 --clearance 0.04 --over-canvas", res["next"]
        res, rc, out, err = run("--sim", "tool", "rest", "--over-canvas")
        assert rc == 0 and res["ok"] and res["moved"] is False and res["canvas"]["overridden"] is True, res
    assert not _touched


def test_tool_rest_is_what_it_was_without_a_set_up_or_without_the_verdict():
    tool = brush()
    never = lambda *a: {"ok": False, "why": "the painting module was asked although there is nothing to ask about"}  # noqa: E731

    # no painting set-up: nothing is asked, nothing is added to the result
    with tool_file(tool), setup_file(None), rest_report(never) as calls:
        res, rc, out, err = run("--sim", "tool", "rest", "--dry")
        assert rc == 0 and res["ok"] and "canvas" not in res, res
        bus = standing(tool, q_deg=SETUP["photo_q_deg"], torque=True)
        res, rc, out, err = run("--sim", "tool", "rest", bus=bus)
        assert rc == 0 and res["ok"] and res["moved"] is True and res["torque"] == "off" and "canvas" not in res, res
        assert abs(res["q_deg"]["shoulder_pan"] - 20.0) < 1.5 and calls == [], (res["q_deg"], calls)
    # a painting module without rest_report: as before, with a set-up on record
    with tool_file(tool), setup_file(), rest_report(None):
        assert not hasattr(PS, "rest_report")
        bus = standing(tool, q_deg=SETUP["photo_q_deg"], torque=True)
        res, rc, out, err = run("--sim", "tool", "rest", "--dry", bus=bus)
        assert rc == 0 and res["stage"] == "planned" and "canvas" not in res, res
    # a tool file named by $FD_TOOL with the RIG's set-up (no $FD_PAINT_SETUP): not asked; the result says so when the rig has one
    with tool_file(tool), rest_report(never) as calls:
        saved = os.environ.pop("FD_PAINT_SETUP")
        try:
            res, rc, out, err = run("--sim", "tool", "rest", "--dry")
        finally:
            os.environ["FD_PAINT_SETUP"] = saved
        assert rc == 0 and res["ok"] and calls == [], (res, calls)
        if os.path.isfile(PS.SETUP_FILE):
            assert res["canvas"]["ok"] is None and "$FD_TOOL" in res["canvas"]["why"] and "overridden" not in res["canvas"], res
        else:
            assert "canvas" not in res, res
    # no such fold with this tool: the refusal is the old one, in tooling's words
    with tool_file(tool), setup_file(), rest_report(never) as calls:
        bus = standing(tool, q_deg=SETUP["photo_q_deg"], torque=True)
        res, rc, out, err = run("--sim", "tool", "rest", "--clearance", "0.5", bus=bus)
        assert rc == 1 and res["stage"] == "plan" and "rest clearance" in res["reason"] and "canvas" not in res and calls == [], res
    # a verdict that says the pose is clear changes nothing but the record of it
    with tool_file(tool), setup_file(), rest_report(lambda *a: {"ok": True, "why": ""}) as calls:
        bus = standing(tool, q_deg=SETUP["photo_q_deg"], torque=True)
        res, rc, out, err = run("--sim", "tool", "rest", "--dry", bus=bus)
        assert rc == 0 and res["stage"] == "planned" and res["canvas"]["ok"] is True and calls == [(20.0, 0.05)], res
    # no tool: said before anything else
    with tool_file(None), setup_file(), rest_report(never) as calls:
        res, rc, out, err = run("--sim", "tool", "rest")
        assert rc == 1 and res["stage"] == "tool" and "fd park" in res["next"] and calls == [], res


def test_tool_rest_can_still_fold_when_the_painting_module_fails():
    """A fault in tools/paint must not take away the one command that folds the arm with a tool on: the fault is
    a refusal with the way out (--over-canvas), and with that word the arm is folded."""
    tool = brush()

    def broken(*a):
        raise ZeroDivisionError("a bug in the painting module")

    with tool_file(tool), setup_file(), rest_report(broken):
        bus = standing(tool, q_deg=SETUP["photo_q_deg"], torque=True)
        before, pos = writes(bus), dict(bus.pos)
        res, rc, out, err = run("--sim", "tool", "rest", "--pan", "-100", "--clearance", "0.04", bus=bus)
        house(res, ok=False, stage="internal")
        assert rc == 1 and res["moved"] is False and "ZeroDivisionError" in res["reason"] and "trace" not in res, res
        assert "`fd --sim tool rest --pan -100 --clearance 0.04 --over-canvas`" in res["next"] and "--over-canvas" in res["say"], res
        assert writes(bus) == before and bus.pos == pos
        res, rc, out, err = run("--sim", "tool", "rest", "--pan", "-100", "--clearance", "0.04", "--over-canvas", bus=bus)
        house(res, ok=True, stage="rest")
        assert rc == 0 and res["moved"] is True and res["torque"] == "off" and res["canvas"]["ok"] is None, res
        assert res["canvas"]["overridden"] is True and "ZeroDivisionError" in res["canvas"]["why"] and "--over-canvas" in res["say"], res


def test_tool_rest_with_a_set_up_that_cannot_be_read_does_not_guess():
    """Where the canvas lies is then not known: refused with the way out, not folded over what may be a picture."""
    tool = brush()
    for text in ("{ not json", json.dumps({"canvas": {"origin_xy": [0.146, -0.118]}})):
        with tool_file(tool), setup_file(text), rest_report() as calls:
            bus = standing(tool, q_deg=SETUP["photo_q_deg"], torque=True)
            before, pos = writes(bus), dict(bus.pos)
            res, rc, out, err = run("--sim", "tool", "rest", "--pan", "-100", "--clearance", "0.04", bus=bus)
            house(res, ok=False, stage="setup")
            assert rc == 1 and res["moved"] is False and "cannot be" in res["reason"] and SETUP_PATH in res["reason"], res
            assert "fd paint status" in res["next"] and "`fd --sim tool rest --pan -100 --clearance 0.04 --over-canvas`" in res["next"], res
            assert "--over-canvas" in res["say"] and writes(bus) == before and bus.pos == pos and calls == [], res
            res, rc, out, err = run("--sim", "tool", "rest", "--pan", "-100", "--clearance", "0.04", "--over-canvas", "--dry", bus=bus)
            assert rc == 0 and res["stage"] == "planned" and res["canvas"]["ok"] is None and res["canvas"]["overridden"] is True, res
            assert calls == []


# ----------------------------------------------------------------------------- a painting tool is not a pen (round 2)
MEASURED = {"at": "2026-11-13T17:21:40", "photos": 36, "used": 34, "rms_px": 6.6}      # what `fd calib-air --save` records
PEN_CAL = dict(contact={"x0": 0.24, "y0": 0.0, "a": -0.0015, "bx": 0.0, "by": 0.0, "rms_mm": 0.2, "points": []},
               corr_map={"centre": [0.22, 0.0], "kind": "affine", "coef": [[0, 0], [1, 0], [0, 1]], "rms_mm": 1.0, "max_mm": 2.0, "n": 9},
               calibrated={"tip": {"at": "x"}, "contact": {"at": "x"}, "map": {"at": "x"}})


def _told(res) -> str:
    """What a result tells the operator to do: its next and its say."""
    return res["next"] + " " + res["say"]


def _photos() -> int:
    return len(glob.glob(os.path.join(RUNS, "*", "*.jpg")))


@contextlib.contextmanager
def pen_calibration(store: bool = False):
    """penskill.calibrate for the duration: a stand-in that answers at once with a plan (what is tested is fd's
    gate in front of it, and what fd adds to its answer). store: it stores the tool with a note, as a stage that
    measured something does. Yields the stages of every call it got."""
    import penskill
    real, calls = penskill.calibrate, []

    def calibrate(sess, stages=penskill.STAGES, opts=None, modules=None):
        calls.append(tuple(stages))
        if store:
            sess.arm.tool.notes.append("stored by the stand-in calibration")
            sess.save_tool(sess.arm.tool)
        return {"ok": True, "skill": "calibrate", "stage": "planned", "reason": "", "next": "run it without --dry",
                "say": "校正の計画ができました。", "tool": sess.arm.tool.name, "photos": [], "timings": {},
                "stages": {"map": {"ok": False, "next": "`fd tool calibrate --stage contact` first"}}}

    penskill.calibrate = calibrate
    try:
        yield calls
    finally:
        penskill.calibrate = real


def test_tool_status_leads_a_painting_tool_to_fd_paint_status_never_to_the_pen_calibration():
    """Review of 2026-11-14 (R2), on copies of the rig's files: `fd tool status` answered the brush with "not
    calibrated (tip, contact, map): run `fd tool calibrate` before `fd draw`", and that command, the PEN
    calibration, then pressed the brush 9 to 15 mm below first touch inside the canvas and replaced the tip that
    had been measured in the air. A painting tool: a measured tip, a jaw record, or any tool beside a painting
    set-up."""
    held = brush(on_jaw="moving_outer", gripper_goal_raw=FIXED - 25, tip_measured=dict(MEASURED))
    cases = ((held, True, "tip"),                                     # measured, the jaw held, a set-up on record
             (brush(tip_measured=dict(MEASURED)), False, "tip"),      # a measured tip alone makes it one
             (brush(on_jaw="moving_outer", gripper_goal_raw=FIXED - 25), False, "jaw"),      # a hold makes it one
             (pen(on_jaw="fixed", tip_measured=dict(MEASURED)), False, "tip"),      # on the fixed jaw, and measured in the air
             (brush(), True, "setup"),                                # the rig's files as they stood that night
             (pen(), True, "setup"))                                  # any tool while a painting set-up is on record
    for tool, with_setup, kind in cases:
        with tool_file(tool), setup_file(SETUP if with_setup else None):
            for sim in ((), ("--sim",)):
                res, rc, out, err = run(*sim, "tool", "status")
                house(res, ok=True, stage="status")
                fdw = "fd --sim" if sim else "fd"
                assert rc == 0 and f"`{fdw} paint status`" in res["next"] and "fd paint status" in res["say"], res
                assert "tool calibrate" not in _told(res) and "fd draw" not in _told(res), res
                assert {"tip": "fd calib-air", "jaw": "fd tool jaw", "setup": SETUP_PATH}[kind] in res["painting_tool"], res
                # the fold it names is one that exists: `fd paint rest` needs the set-up
                assert f"`{fdw} {'paint' if with_setup else 'tool'} rest` folds away" in res["next"], res["next"]
                assert ("測定済み" if tool.tip_measured else "未測定") in res["say"] and "絵の具用" in res["say"], res["say"]
    # a pen on a rig without a painting set-up is what it was
    for tool in (pen(), brush()):
        with tool_file(tool), setup_file(None):
            res, rc, out, err = run("--sim", "tool", "status")
            assert rc == 0 and "painting_tool" not in res and "run `fd tool calibrate` before drawing (fd draw)" in res["next"], res
            assert "fd tool calibrate が必要です" in res["say"] and "`fd tool rest` folds away" in res["next"], res
    # a scratch tool file ($FD_TOOL) is not judged by the RIG's set-up, whatever was painted on this desk last (the
    # pen suites run that way). "The rig's set-up" is a stand-in here (paintskill.SETUP_FILE points at a scratch file
    # that exists, $FD_PAINT_SETUP is not set), so that this holds on a desk with a painting set-up and on one without
    with tool_file(pen()):
        saved, rigs = os.environ.pop("FD_PAINT_SETUP"), PS.SETUP_FILE
        PS.SETUP_FILE = os.path.join(_TMP, "as_the_rigs_paint_setup.json")
        with open(PS.SETUP_FILE, "w", encoding="utf-8") as f:
            json.dump(SETUP, f)
        try:
            assert PS.setup_path() == PS.SETUP_FILE and os.path.isfile(PS.SETUP_FILE) and fd._paint_setup_on_record() is None
            res, rc, out, err = run("--sim", "tool", "status")
            assert rc == 0 and "painting_tool" not in res and "fd tool calibrate" in res["next"], res
            with pen_calibration() as calls:
                res, rc, out, err = run("--sim", "tool", "calibrate", "--dry")
            assert rc == 0 and res["ok"] and "painting_tool" not in res and len(calls) == 1, res
            # ... but the rig's own tool file IS judged by it (stand-ins for both tool files, no $FD_TOOL)
            with two_tool_files():
                TL.save(pen())
                assert fd._paint_setup_on_record() == PS.SETUP_FILE
                for sim in ((), ("--sim",)):
                    res, rc, out, err = run(*sim, "tool", "status")
                    assert rc == 0 and PS.SETUP_FILE in res["painting_tool"] and "paint status" in res["next"] and "tool calibrate" not in _told(res), res
                with pen_calibration() as calls:
                    res, rc, out, err = run("--sim", "tool", "calibrate", "--dry")
                assert rc == 1 and res["stage"] == "tool" and calls == [] and PS.SETUP_FILE in res["painting_tool"], res
            # with $FD_PAINT_SETUP pinned too, the scratch tool file is judged by that set-up (the first part of this test)
        finally:
            os.environ["FD_PAINT_SETUP"], PS.SETUP_FILE = saved, rigs
            os.remove(os.path.join(_TMP, "as_the_rigs_paint_setup.json"))
        assert PS.setup_path() == SETUP_PATH
    # a pen that carries the pen calibration draws, also beside a painting set-up: nothing sends it anywhere
    with tool_file(pen(**PEN_CAL)), setup_file():
        res, rc, out, err = run("--sim", "tool", "status")
        assert rc == 0 and res["next"].startswith("calibrated: fd draw with the name or the file of a drawing draws it"), res
        assert "`fd draw cat --dry`" in res["next"] and "FILE" not in res["next"] and SETUP_PATH in res["painting_tool"], res
        assert res["next"].endswith("`fd --sim paint rest` folds away") and "tool rest" not in res["next"], res["next"]
    with tool_file(pen(**PEN_CAL)), setup_file(None):       # ... and without one it is the tool's own fold, as it was
        res, rc, out, err = run("--sim", "tool", "status")
        assert rc == 0 and res["next"].endswith("`fd tool rest` folds away") and "painting_tool" not in res, res
    assert not _touched


def test_a_pen_recorded_on_the_fixed_jaw_is_no_painting_tool():
    """Round 3 (T7). The next of a refused `fd ready` sends a PEN whose limp jaw moved to `fd tool jaw mount fixed`
    (round 2, R17). Any jaw record made a tool a painting tool: after that step `fd tool status` and a refused
    `fd draw` sent the pen to `fd paint status`, which is refused on a rig without a painting set-up ("the set-up
    file is written by hand"), and its own calibration was refused until --pen was added. A record on the FIXED jaw
    alone is no mark of a painting tool; a measured tip, a record on the moving jaw or between the jaws, a jaw
    held shut, and a painting set-up on record are."""
    def limp(tool, jaw):
        arm = Arm(bus=SimBus(), cfg=load_config(), tool=tool).open()
        return standing(tool, q_deg=np.degrees(TL.rest_q(arm, tool)), torque=False, gripper=jaw)

    with tool_file(None), setup_file(None):
        assert fd._painting_tool(pen(on_jaw="fixed")) is None and fd._painting_tool(pen()) is None
        # (round 3b, U6) a jaw record without a hold does not make a painting tool, whichever jaw it names ...
        for mark in (dict(on_jaw="moving_outer"), dict(on_jaw="between")):
            assert fd._painting_tool(pen(**mark)) is None, mark
        # ... a hold does
        # ... a hold does (a hold exists only for a tool on the moving jaw's outer face)
        for mark in (dict(gripper_goal_raw=2036), dict(on_jaw="moving_outer", gripper_goal_raw=2036)):
            assert fd._painting_tool(pen(**mark))[0] == "jaw", mark
        assert fd._painting_tool(pen(on_jaw="fixed", tip_measured=dict(MEASURED)))[0] == "tip"
        # the verifier's walk: a pen is attached, its limp jaw is 20 ticks off, the next of the refused `fd ready`
        res, rc, out, err = run("--sim", "tool", "attach", "--name", "pen", "--along-fingers", "0.055")
        assert rc == 0 and "painting_tool" not in res, res
        raw = TL.active().gripper_raw
        res, rc, out, err = run("--sim", "ready", bus=limp(TL.active(), raw + 20))
        assert rc == 1 and res["stage"] == "plan" and "`fd tool jaw mount fixed`" in res["next"], res
        res, rc, out, err = run("--sim", "tool", "jaw", "mount", "fixed")
        assert rc == 0 and res["ok"] and TL.active().on_jaw == "fixed", res
        res, rc, out, err = run("--sim", "ready", bus=limp(TL.active(), raw + 20))
        assert rc == 0 and res["ok"] and res["tool"] == "pen", res
        # it is still a pen: status and a refused draw name its own calibration, never `fd paint status` ...
        res, rc, out, err = run("--sim", "tool", "status")
        house(res, ok=True, stage="status")
        assert rc == 0 and "painting_tool" not in res and res["tool"]["on_jaw"] == "fixed", res
        assert "run `fd tool calibrate` before drawing (fd draw)" in res["next"] and "`fd tool rest` folds away" in res["next"], res["next"]
        assert "paint status" not in _told(res) and "絵の具用" not in res["say"] and "fd tool calibrate が必要です" in res["say"], res
        res, rc, out, err = run("--sim", "draw", "cat", "--dry")
        assert rc == 1 and res["stage"] == "tool" and res["next"].startswith("`fd tool calibrate` (all stages)"), res
        assert "painting_tool" not in res and "paint status" not in _told(res), res
        # ... which is what that command answers on a rig without a painting set-up
        res, rc, out, err = run("--sim", "paint", "status")
        assert rc == 1 and res["ok"] is False and res["stage"] == "setup", res
        # its calibration needs no --pen (it was refused at stage tool), and the other texts are the pen's
        with pen_calibration() as calls:
            res, rc, out, err = run("--sim", "tool", "calibrate", "--dry")
        assert rc == 0 and res["ok"] and calls == [("tip", "contact", "map", "direction")], (res, calls)
        assert "painting_tool" not in res and "pen_declared" not in res, res
        res, rc, out, err = run("--sim", "tip", "0.25", "0.0", "0.03", "--contact")
        assert rc == 1 and res["stage"] == "tool" and "the pen commands" in res["next"] and "paint status" not in res["next"], res
        res, rc, out, err = run("--sim", "tool", "set", "--note", "looked at it", "--no-robot")
        assert rc == 0 and res["next"] == "`fd tool status` shows the tool" and "painting_tool" not in res, res
    # beside a painting set-up the same pen is a painting tool as any tool is there: by the set-up, not by its jaw
    with tool_file(pen(on_jaw="fixed")), setup_file():
        res, rc, out, err = run("--sim", "tool", "status")
        assert rc == 0 and SETUP_PATH in res["painting_tool"] and "fd tool jaw" not in res["painting_tool"], res
        assert "`fd --sim paint status`" in res["next"] and "絵のセットアップが登録されている" in res["say"], res
    assert not _touched


def test_the_pen_calibration_is_refused_for_a_painting_tool_before_anything_moves():
    """`fd tool calibrate` on a painting tool: stage tool, next `fd paint status`, nothing opened, nothing
    written, no photo. The word --pen is the declaration that a pen is on the gripper and this calibration is
    meant: the refusal spells the call with it, and nothing else lets a painting tool through."""
    held = brush(on_jaw="moving_outer", gripper_goal_raw=FIXED - 25, tip_measured=dict(MEASURED))
    with pen_calibration() as calls:
        for tool, with_setup in ((held, True), (brush(tip_measured=dict(MEASURED)), False), (brush(on_jaw="moving_outer", gripper_goal_raw=FIXED - 25), False),
                                 (brush(), True), (pen(), True)):
            with tool_file(tool), setup_file(SETUP if with_setup else None):
                text, photos = open(TOOL_PATH).read(), _photos()
                for words in ((), ("--dry",), ("--stage", "tip"), ("--stage", "contact", "--marks-on", "scrap", "--dry"),
                              ("--size", "80", "--places", "5", "--new-sheet"), ("--at=-0.01,0.1",)):
                    bus = standing(tool, torque=False)
                    before, pos = writes(bus), dict(bus.pos)
                    res, rc, out, err = run("--sim", "tool", "calibrate", *words, bus=bus)
                    house(res, ok=False, stage="tool")
                    assert rc == 1 and res["error"] == "SafetyError" and res["tool"] == tool.name and res["message"] == res["reason"], res
                    assert "PEN calibration" in res["reason"] and "Nothing moved" in res["reason"] and res["painting_tool"] in res["reason"], res
                    assert res["next"].startswith("`fd --sim paint status` says") and "fd paint status" in res["say"] and "--pen" in res["say"], res
                    # the one other command it names is this call with the word, as it is typed
                    assert res["next"].split("`")[3] == " ".join(("fd --sim tool calibrate --pen", *words)) and res["next"].count("`") == 4, res["next"]
                    assert writes(bus) == before and bus.pos == pos and not any(bus.torque.values()), "a refused `tool calibrate` wrote to the servos"
                    assert open(TOOL_PATH).read() == text and _photos() == photos and "photos" not in res and calls == [], (res, calls)
                # without --sim it is refused the same way BEFORE the arm is asked for: the answer is not the hardware lock's
                res, rc, out, err = run("tool", "calibrate", "--stage", "tip")
                house(res, ok=False, stage="tool")
                assert rc == 1 and "sim" not in res and res["next"].split("`")[1::2] == ["fd paint status", "fd tool calibrate --pen --stage tip"], res
                # ... and with the word the gate is passed: the real arm is asked for, and the lock answers
                res, rc, out, err = run("tool", "calibrate", "--pen", "--stage", "tip")
                house(res, ok=False, stage="hardware")
                assert rc == 1 and "hardware is locked" in res["reason"] and open(TOOL_PATH).read() == text and calls == [], res
                # simulated, with the word: the calibration is reached, and what it names next carries the word too
                res, rc, out, err = run("--sim", "tool", "calibrate", "--pen", "--stage", "map", "--dry")
                assert rc == 0 and res["ok"] and calls == [("map",)] and res["pen_declared"] is True and res["painting_tool"], (res, calls)
                assert res["stages"]["map"]["next"] == "`fd tool calibrate --pen --stage contact` first", res["stages"]
                assert ("measured_tip_replaced" in res) is False, res             # (this stage leaves the tip alone)
                del calls[:]
        # a pen on a rig without a painting set-up needs no word (and may give it)
        with tool_file(pen()), setup_file(None):
            for words in ((), ("--pen",)):
                res, rc, out, err = run("--sim", "tool", "calibrate", "--dry", *words)
                house(res, ok=True, stage="planned")
                assert rc == 0 and "painting_tool" not in res and ("pen_declared" in res) == bool(words) and "measured_tip_replaced" not in res, res
            assert calls == [("tip", "contact", "map", "direction")] * 2, calls
    assert not _touched


def test_with_pen_the_calibration_runs_and_says_what_it_does_to_a_measured_tip():
    """--pen on a tool whose tip was measured in the air: the stage `tip` replaces that tip by the one it reads
    from the ink. The result says so before (--dry) and after, and every command it names carries the word."""
    tool = pen(tip_measured=dict(MEASURED))
    tip = "(1.5, -8.1, 62.1) mm"
    with tool_file(tool), setup_file():
        text = open(TOOL_PATH).read()
        res, rc, out, err = run("--sim", "tool", "calibrate", "--pen", "--dry", "--new-sheet")
        house(res, ok=True, stage="planned")
        assert rc == 0 and res["pen_declared"] is True and res["measured_tip_replaced"] is False and "fd calib-air" in res["painting_tool"], res
        assert any("REPLACES" in n and tip in n and "fd calib-air --save" in n for n in res["notes"]) and "置き換えます" in res["say"], res
        assert open(TOOL_PATH).read() == text
        # a stage that leaves the tip alone says nothing about it
        with pen_calibration() as calls:
            res, rc, out, err = run("--sim", "tool", "calibrate", "--pen", "--stage", "contact", "--dry")
        assert rc == 0 and calls == [("contact",)] and res["pen_declared"] is True and "measured_tip_replaced" not in res, res
        assert not any("REPLACE" in n for n in res.get("notes", [])) and "置き換" not in res["say"], res
        # the tip stage itself, on the simulated sheet
        res, rc, out, err = run("--sim", "tool", "calibrate", "--pen", "--stage", "tip")
        house(res, ok=True, stage="done")
        now = TL.active()
        assert rc == 0 and res["stages"]["tip"]["ok"] and now.tip_measured is None and now.tip_offset != tool.tip_offset, res
        assert res["measured_tip_replaced"] is True and any("has REPLACED" in n and tip in n for n in res["notes"]), res["notes"]
        assert "置き換わりました" in res["say"] and "fd calib-air --save" in res["say"], res["say"]
        # what it names next is typed with the word: the set-up is still on record, the plain command would be refused
        assert "`fd tool calibrate --pen --stage contact`" in res["next"] and "`fd tool calibrate --stage" not in res["next"], res["next"]
        res, rc, out, err = run("--sim", "tool", "calibrate", "--stage", "contact", "--dry")
        assert rc == 1 and res["stage"] == "tool" and SETUP_PATH in res["painting_tool"], res
    assert not _touched


def test_attach_set_draw_and_contact_do_not_send_a_painting_tool_to_the_pen_calibration():
    held = brush(on_jaw="moving_outer", gripper_goal_raw=FIXED - 25, tip_measured=dict(MEASURED))
    attach = ("--sim", "tool", "attach", "--name", "brush", "--tip", "0.0085,0.0007,0.0611", "--pitch", "65", "--no-robot")
    # attach beside a painting set-up
    with tool_file(None), setup_file():
        for extra in (("--dry",), ()):
            res, rc, out, err = run(*attach, *extra)
            house(res, ok=True, stage="planned" if extra else "attached")
            assert rc == 0 and "`fd --sim paint status`" in res["next"] and "fd paint status" in res["say"], res
            assert "tool calibrate" not in _told(res) and SETUP_PATH in res["painting_tool"] and "`fd --sim paint rest`" in res["next"], res
            assert ("保存していません" in res["say"]) == bool(extra), res["say"]
    with tool_file(None), setup_file(None):                 # no set-up: the pen's text, as it was
        res, rc, out, err = run(*attach)
        assert rc == 0 and "painting_tool" not in res and "`fd tool calibrate` measures the tip on paper" in res["next"], res
        assert "fd tool rest を使います" in res["say"] and "`fd draw`" not in res["next"], res
    # set: a new tip takes the record of the measured tip away, and the tool is still not sent to the pen calibration
    with tool_file(brush(tip_measured=dict(MEASURED), **PEN_CAL)), setup_file(None):
        res, rc, out, err = run("--sim", "tool", "set", "--tip", "0.017,-0.001,0.056", "--no-robot")
        house(res, ok=True, stage="set")
        assert rc == 0 and res["calibration_dropped"] and res["before"]["tip_measured"] is True and res["tool"]["tip_measured"] is False, res
        assert "`fd --sim paint status`" in res["next"] and "fd paint status" in res["say"] and "tool calibrate" not in _told(res), res
        assert "fd calib-air" in res["painting_tool"] and "`fd --sim tool status`" in res["next"], res
    with tool_file(brush()), setup_file():                  # nothing dropped: `fd paint status` is named all the same
        res, rc, out, err = run("--sim", "tool", "set", "--note", "looked at it", "--no-robot")
        assert rc == 0 and "`fd --sim paint status`" in res["next"] and "fd paint status" in res["say"] and "dropped" not in res["next"], res
    with tool_file(pen(**PEN_CAL)), setup_file(None):       # a pen: as it was
        res, rc, out, err = run("--sim", "tool", "set", "--tip", "0.002,-0.008,0.060", "--no-robot")
        assert rc == 0 and res["calibration_dropped"] and res["next"].endswith("run `fd tool calibrate` before drawing (fd draw)"), res
        assert "校正はやり直しが必要です" in res["say"] and "painting_tool" not in res, res
        res, rc, out, err = run("--sim", "tool", "set", "--note", "x", "--no-robot")
        assert rc == 0 and res["next"] == "`fd tool status` shows the tool", res
    # fd draw is closed to a brush by the missing pen calibration, and its refusal does not send the brush there
    with tool_file(held), setup_file():
        bus = standing(held, torque=False)
        before, photos = writes(bus), _photos()
        res, rc, out, err = run("--sim", "draw", "cat", "--dry", bus=bus)
        house(res, ok=False, stage="tool")
        assert rc == 1 and res["next"].startswith("`fd --sim paint status` says") and "fd paint status" in res["say"], res
        assert "tool calibrate" not in _told(res) and "PEN" in res["reason"] and "not calibrated" in res["reason"], res
        assert writes(bus) == before and _photos() == photos and res["painting_tool"], res
    with tool_file(pen()), setup_file(None):                # a pen: penskill's own words
        res, rc, out, err = run("--sim", "draw", "cat", "--dry")
        assert rc == 1 and res["stage"] == "tool" and res["next"].startswith("`fd tool calibrate` (all stages)") and "painting_tool" not in res, res
    # pressing a tool onto the paper by hand: refused as before; for a brush the way that is named is the painting one
    with tool_file(held), setup_file():
        bus = standing(held, torque=True)
        before = writes(bus)
        for words in (("tip", "0.25", "0.0", "0.03", "--contact"), ("joints", "wrist_flex=60", "--contact")):
            res, rc, out, err = run("--sim", *words, bus=bus)
            house(res, ok=False, stage="tool")
            assert rc == 1 and "`fd --sim paint status`" in res["next"] and "tool calibrate" not in res["next"] + res["reason"], res
            assert "fd draw" not in res["next"] + res["reason"] and writes(bus) == before, res
    with tool_file(pen()), setup_file(None):
        res, rc, out, err = run("--sim", "tip", "0.25", "0.0", "0.03", "--contact")
        assert rc == 1 and res["stage"] == "tool" and res["next"].endswith(", or use the pen commands (`fd tool status` says what the pen still needs)"), res
        assert "left to the pen commands (fd draw, fd tool calibrate), which stop by what the camera sees" in res["reason"], res
        assert "`fd draw`" not in res["next"] + res["reason"], res            # (the bare name of a command is no command)
    assert not _touched


# ----------------------------------------------------------------------------- fd's own simulated tool commands (round 2)
@contextlib.contextmanager
def two_tool_files():
    """The real and the simulated tool file as `fd --sim` has them without $FD_TOOL, both in the scratch folder
    (stand-ins: tooling's own two paths point there for the duration): (real path, simulated path)."""
    saved = (os.environ.pop("FD_TOOL"), TL.SIM_TOOL_FILE, TL.TOOL_FILE)
    real, simf = os.path.join(_TMP, "two", "tool_active.json"), os.path.join(_TMP, "two", "simstate", "tool_active.sim.json")
    TL.TOOL_FILE, TL.SIM_TOOL_FILE = real, simf
    try:
        yield real, simf
    finally:
        os.environ["FD_TOOL"], TL.SIM_TOOL_FILE, TL.TOOL_FILE = saved
        for p in (real, simf, os.path.join(os.path.dirname(simf), "sim_paper.json")):
            if os.path.exists(p):
                os.remove(p)


def test_simulated_tool_commands_say_which_tool_file_they_work_on():
    """Round 2 (R15), the rest of review finding 32. `fd --sim tool set --pitch 60` on the tool the simulation
    mirrors stores a marked copy of the real tool's record as the simulated tool: the result said nothing of it.
    After the real tool had been registered anew (the copy is then left behind, not read), `fd --sim tool attach`
    answered "tool 'brush_thin' is already attached", and `fd --sim tool detach` said that gripper and park were
    unlocked while the simulation carried the real tool."""
    with two_tool_files() as (real, simf), setup_file(None):
        TL.save(brush())                                                   # the REAL tool; the simulation mirrors it
        real_text = open(real).read()
        res, rc, out, err = run("--sim", "tool", "status")
        assert rc == 0 and res["simulated_tool"] is False and res["file"] == real, res
        # a dry change stores nothing, and says nothing of a simulated file
        res, rc, out, err = run("--sim", "tool", "set", "--pitch", "60", "--no-robot", "--dry")
        assert rc == 0 and res["saved"] is False and res["simulated_tool"] is False and "sim_note" not in res and not os.path.exists(simf), res
        # the change itself: stored as a copy in the simulated file, and the result says so in three places
        res, rc, out, err = run("--sim", "tool", "set", "--pitch", "60", "--no-robot")
        house(res, ok=True, stage="set")
        assert rc == 0 and res["saved"] and res["file"] == simf and res["simulated_tool"] is True and open(real).read() == real_text, res
        assert res["sim_note"].startswith("this command stored the tool in the SIMULATED tool file") and "copy of the real tool" in res["sim_note"], res
        assert "`fd --sim tool detach` returns to the real tool" in res["sim_note"] and "sim_copy_dropped" not in res, res
        assert "シミュレーション用のツール記録に保存しました" in res["say"] and "実機のツール記録は変えていません" in res["say"], res["say"]
        assert TL.SIM_COPY_KEY in json.load(open(simf)) and TL.active(sim=True).pitch_deg == 60.0 and TL.active().pitch_deg == 65.0
        # a later command that stores nothing still says that the simulation reads a file of its own
        res, rc, out, err = run("--sim", "tool", "set", "--pitch", "60", "--no-robot")
        assert rc == 0 and res["saved"] is False and res["simulated_tool"] is True and "copy of the real tool" in res["sim_note"], res
        assert not res["sim_note"].startswith("this command stored") and "保存しました" not in res["say"], res
        # the real tool is registered anew: the copy is left behind (not read); the simulation mirrors the new tool
        TL.save(brush(name="brush_new", attached_at="2026-11-14T09:00:00"))
        real_text = open(real).read()
        res, rc, out, err = run("--sim", "tool", "status")
        assert rc == 0 and res["tool"]["name"] == "brush_new" and res["simulated_tool"] is False and os.path.exists(simf), res
        assert res["sim_copy_dropped"]["file"] == simf and "以前のシミュレーション用のツール記録" in res["say"], res
        # ... so nothing simulated is "already attached": a simulated tool can be attached (it was refused with the
        # name of the copy nobody reads), and the result says what became of that copy
        attach = ("--sim", "tool", "attach", "--name", "simpen", "--along-fingers", "0.06", "--no-robot")
        res, rc, out, err = run(*attach, "--dry")
        assert rc == 0 and res["stage"] == "planned" and res["simulated_tool"] is False and res["sim_copy_dropped"]["file"] == simf, res
        assert json.load(open(simf))["name"] == "brush_thin", "a dry attach wrote the simulated tool file"
        res, rc, out, err = run(*attach)
        house(res, ok=True, stage="attached")
        assert rc == 0 and res["file"] == simf and res["simulated_tool"] is True and TL.active(sim=True).name == "simpen", res
        assert "real tool is now 'brush_new'" in res["sim_copy_dropped"]["why"] and "以前のシミュレーション用のツール記録" in res["say"], res
        assert "tool of its own" in res["sim_note"] and res["sim_note"].startswith("this command stored") and "保存しました" in res["say"], res
        assert TL.SIM_COPY_KEY not in json.load(open(simf)) and open(real).read() == real_text and TL.active().name == "brush_new"
        # a simulated tool that IS read stands in the way of a second attach, as before
        res, rc, out, err = run(*attach)
        house(res, ok=False, stage="tool")
        assert rc == 1 and "tool 'simpen' is already attached" in res["reason"] and res["simulated_tool"] is True and "--replace" in res["next"], res
        # detach of the simulated tool: the simulation mirrors the real tool again, which is still on the gripper
        for dry in (("--dry",), ()):
            res, rc, out, err = run("--sim", "tool", "detach", *dry)
            house(res, ok=True, stage="planned" if dry else "detached")
            assert rc == 0 and res["tool"] == "simpen" and res["mirrors"] == "brush_new" and res["attached"] is True, res
            assert res["removed"] is (not dry) and os.path.exists(simf) is bool(dry), res
            assert "stay locked" in res["next"] and "are unlocked" not in res["next"] and "`fd --sim tool status`" in res["next"], res["next"]
            assert "ロックされたまま" in res["say"] and "ロックは解除" not in res["say"] and "brush_new" in res["say"], res["say"]
            assert res["simulated_tool"] is bool(dry), res
        assert TL.active(sim=True).name == "brush_new" and open(real).read() == real_text
        # a copy that was left behind: detach removes the file, says that it was not read any more, and unlocks nothing
        res, rc, out, err = run("--sim", "tool", "set", "--pitch", "61", "--no-robot")
        assert rc == 0 and TL.active(sim=True).pitch_deg == 61.0, res
        TL.save(brush(name="brush_new", attached_at="2026-11-14T09:00:00", notes=["the tape was renewed"]))
        real_text = open(real).read()
        assert TL.active(sim=True).pitch_deg == 65.0 and os.path.exists(simf)
        res, rc, out, err = run("--sim", "tool", "detach")
        house(res, ok=True, stage="detached")
        assert rc == 0 and res["removed"] and not os.path.exists(simf) and res["mirrors"] == "brush_new" and res["simulated_tool"] is False, res
        assert res["sim_copy_dropped"]["file"] == simf and "stay locked" in res["next"] and "ロックは解除" not in res["say"], res
        res, rc, out, err = run("--sim", "tool", "detach")                # nothing simulated left: the real tool is mirrored
        assert rc == 0 and res["removed"] is False and res["attached"] is True and res["mirrors"] == "brush_new", res
        assert "stay locked" in res["next"] and "ロックされたまま" in res["say"] and "note" not in res, res
        # the pen calibration, declared with --pen: a stage that stores the tool writes the simulated copy, and says so
        with pen_calibration(store=True) as calls:
            res, rc, out, err = run("--sim", "tool", "calibrate", "--pen", "--stage", "contact")
        assert rc == 0 and calls == [("contact",)] and res["simulated_tool"] is True and res["file"] == simf, res
        assert res["sim_note"].startswith("this command stored the tool in the SIMULATED tool file") and "保存しました" in res["say"], res
        assert open(real).read() == real_text and TL.active().notes == ["the tape was renewed"], "the real tool file was written"
        assert TL.active(sim=True).notes == ["the tape was renewed", "stored by the stand-in calibration"]
        with pen_calibration() as calls:                                   # ... one that stores nothing only says which file is read
            res, rc, out, err = run("--sim", "tool", "calibrate", "--pen", "--stage", "map", "--dry")
        assert rc == 0 and res["simulated_tool"] is True and not res["sim_note"].startswith("this command stored"), res
        # no real tool: a simulated tool of its own is detached, and then the simulated gripper IS unlocked
        run("--sim", "tool", "detach")
        TL.clear()
        res, rc, out, err = run(*attach)
        assert rc == 0 and res["simulated_tool"] is True and "sim_copy_dropped" not in res, res
        res, rc, out, err = run("--sim", "tool", "detach")
        assert rc == 0 and res["removed"] and "mirrors" not in res and "are unlocked" in res["next"] and "ロックは解除" in res["say"], res
        assert res["simulated_tool"] is False and TL.active(sim=True) is None
    # with $FD_TOOL there is one tool file for both: nothing is simulated about it, and the results say so
    with tool_file(brush()), setup_file(None):
        res, rc, out, err = run("--sim", "tool", "set", "--pitch", "60", "--no-robot")
        assert rc == 0 and res["saved"] and res["simulated_tool"] is False and "sim_note" not in res and "シミュレーション用" not in res["say"], res
        res, rc, out, err = run("--sim", "tool", "detach")
        assert rc == 0 and res["removed"] and res["simulated_tool"] is False and "mirrors" not in res and "are unlocked" in res["next"], res
        res, rc, out, err = run("tool", "status")                         # without --sim: no such key at all
        assert rc == 0 and "simulated_tool" not in res, res
    assert not _touched


def test_sim_detach_while_the_simulation_mirrors_the_real_tool_says_that_the_tool_is_still_on():
    """Round 3 (T8). Stand-in tool files without $FD_TOOL, the real tool registered, no simulated tool file (the
    rig's own state): `fd --sim tool detach` answered "attached": false, "removed": false and 「ツールは登録されて
    いませんでした」; only a key "note" said that the simulation mirrors the real arm, which carries one. `fd --sim
    tool status` right after showed the tool, and `fd --sim park` was refused because of it."""
    with two_tool_files() as (real, simf), setup_file(None):
        os.makedirs(os.path.dirname(real), exist_ok=True)
        TL.save(brush())                                                   # the REAL tool; no simulated copy
        text = open(real).read()
        for dry in (("--dry",), ()):
            res, rc, out, err = run("--sim", "tool", "detach", *dry)
            house(res, ok=True, stage="detached")
            assert rc == 0 and res["attached"] is True and res["removed"] is False and res["mirrors"] == "brush_thin", res
            assert "note" not in res and res["simulated_tool"] is False and res["file"] == simf, res
            assert "no SIMULATED tool file" in res["next"] and "stay locked" in res["next"] and "`fd --sim tool status`" in res["next"], res["next"]
            assert "without --sim" in res["next"] and "brush_thin" in res["next"], res["next"]
            assert "brush_thin" in res["say"] and "ロックされたまま" in res["say"] and "--sim を付けない fd tool detach" in res["say"], res["say"]
            assert "登録されていませんでした" not in res["say"] and "ロックは解除" not in res["say"], res["say"]
            assert open(real).read() == text and not os.path.exists(simf)
        # what the next commands meet agrees with it
        res, rc, out, err = run("--sim", "tool", "status")
        assert rc == 0 and res["attached"] is True and res["tool"]["name"] == "brush_thin" and res["simulated_tool"] is False, res
        res, rc, out, err = run("--sim", "park")
        assert rc == 1 and res["stage"] == "tool" and res["tool"] == "brush_thin", res
        # no real tool either: nothing is attached, said as before (and with a next, as every result has)
        TL.clear()
        res, rc, out, err = run("--sim", "tool", "detach")
        house(res, ok=True, stage="detached")
        assert rc == 0 and res["attached"] is False and res["removed"] is False and "mirrors" not in res, res
        assert res["say"].startswith("ツールは登録されていませんでした。") and "nothing was attached" in res["next"], res
    with tool_file(None), setup_file(None):                # ($FD_TOOL: one file for both), and the real command
        for sim in (("--sim",), ()):
            res, rc, out, err = run(*sim, "tool", "detach")
            house(res, ok=True, stage="detached")
            assert rc == 0 and res["attached"] is False and res["removed"] is False and "mirrors" not in res, res
    assert not _touched


def test_a_locked_command_names_fd_paint_rest_for_a_painting_tool_beside_a_set_up():
    """Round 3 (T14). On copies of the rig's files `fd --sim park` answered "`fd tool rest` folds the arm away with
    the tool on", and `fd --sim tool rest` was then refused at stage canvas with `fd --sim paint rest` as its next:
    two refusals before the command that folds the arm off the picture, the first without --sim after a simulated
    call. For a painting tool with a painting set-up on record the refusal names `fd paint rest` in the spelling of
    the call (fd._fold_words); a pen, and a brush on a rig without a set-up, are told what they were told."""
    import argparse
    tool = brush()
    with tool_file(tool), setup_file(), rest_report():
        bus = standing(tool, q_deg=SETUP["photo_q_deg"], torque=True)      # where every painting command leaves the arm
        before, pos = writes(bus), dict(bus.pos)
        for words in (("park",), ("grip", "0.5"), ("grasp",), ("probe",), ("bench", "--tier", "slow")):
            res, rc, out, err = run("--sim", *words, bus=bus)
            house(res, ok=False, stage="tool")
            assert rc == 1 and res["tool"] == "brush_thin" and res["message"] == res["reason"], (words, res)
            assert res["next"].startswith("`fd --sim paint rest` folds the arm away with the tool on"), (words, res["next"])
            assert "`fd --sim tool detach`" in res["next"] and "tool rest" not in res["next"] and res["next"] in res["reason"], (words, res)
            assert "fd paint rest" in res["say"] and "キャンバス" in res["say"] and "動かしていません" in res["say"], res["say"]
            assert writes(bus) == before and bus.pos == pos, f"a refused {words} wrote to the servos"
        # switching the torque off away from a rest pose names the same fold
        res, rc, out, err = run("--sim", "torque", "off", bus=bus)
        house(res, ok=False, stage="tool")
        assert rc == 1 and res["next"].startswith("`fd --sim paint rest` (folds the arm away") and "Use `fd --sim paint rest`" in res["reason"], res
        assert "fd paint rest で収納" in res["say"] and "tool rest" not in res["next"] + res["reason"] and writes(bus) == before, res
        # the command that was named until now is refused there, and the one that is named now is taken
        res, rc, out, err = run("--sim", "tool", "rest", "--dry", bus=bus)
        assert rc == 1 and res["stage"] == "canvas" and "`fd --sim paint rest`" in res["next"], res
        res, rc, out, err = run("--sim", "paint", "rest", "--dry")
        assert rc == 0 and res["ok"] is True, res
        # without --sim the spelling is the real command's (asked of the function: the real arm is locked)
        real, sim = argparse.Namespace(sim=False), argparse.Namespace(sim=True)
        assert fd._fold_words(real, tool) == "fd paint rest" and fd._fold_words(sim, tool) == "fd --sim paint rest"
        arm = Arm(bus=SimBus(), cfg=load_config(), tool=tool)
        r = fd._tool_refusal(arm, "park", real)
        assert r["next"].startswith("`fd paint rest` folds the arm away") and "`fd tool detach`" in r["next"] and "--sim" not in r["next"], r
    # a refused `fd ready` names it as well
    lim = brush(joint_limits_deg={"wrist_flex": [-100.0, -20.0]})
    with tool_file(lim), setup_file(), rest_report():
        res, rc, out, err = run("--sim", "ready", bus=standing(lim, q_deg=[40.0, -100.0, 85.0, -30.0, 0.0], torque=False))
        assert rc == 1 and res["stage"] == "plan" and "`fd --sim paint rest` folds the arm away meanwhile" in res["next"], res
    # a pen on a rig without a painting set-up, and a brush there (`fd paint rest` needs the set-up): the tool's own fold
    for other in (pen(), brush(on_jaw="moving_outer")):
        with tool_file(other), setup_file(None):
            res, rc, out, err = run("--sim", "park")
            house(res, ok=False, stage="tool")
            assert rc == 1 and res["next"] == TL.NEXT_LOCKED and "`fd tool rest` folds the arm away" in res["next"], res
            assert "paint rest" not in _told(res) + res["reason"] and "fd tool rest" in res["say"], res
            assert fd._fold_words(argparse.Namespace(sim=True), other) == "fd tool rest"
    assert not _touched


# ----------------------------------------------------------------------------- the commands fd's own results name
# a word that stands for a value the operator has to give: N, DEG, LEN, DIR, FILE, PLAN, BEFORE.jpg, x,y,z, JOINT=LO:HI
_PLACEHOLDER = re.compile(r"[A-Z][A-Z0-9_]*(\.[a-z]{2,4})?|[A-Z]{2,}=.*|.*(x,y|x0,y0|LO:HI).*")
# Texts of modules that are not this suite's, printed by fd's commands as they are, and left by round 3 (reported):
# tools/draw/penskill.py (the pen commands `fd draw` and `fd tool calibrate`; that module was changed for one finding
# only, T15): four placeholders and the bare `fd draw`; tools/motion/robot.py (Arm.q_limits, the refusal of a tool's
# joint limit the arm cannot apply): `fd tool set --joint-limit JOINT=LO:HI` with the joint's name. Listed word for
# word, so that nothing else passes.
_LEFT_ELSEWHERE = {"fd tool attach --name pen --along-fingers LEN", "fd tool set --along-fingers LEN", "fd draw NAME --dry",
                   "fd draw NAME", "fd draw"} | {f"fd tool set --joint-limit {j}=LO:HI" for j in fd.ARM}


def _commands(res) -> list:
    """Every `fd ...` a result puts into backticks: in its next, reason, say, notes and the notes about the arm."""
    texts = [res.get(k) for k in ("next", "reason", "say", "sim_note")] + list(res.get("notes") or [])
    return [c for t in texts if isinstance(t, str) for c in re.findall(r"`(fd [^`]+)`", t)]


def _accepted(cmd: str):
    """Is `fd ...` ONE command the parsers take AS IT IS PRINTED? Why not (None: it is). Nothing is run. A bar, an
    ellipsis or brackets inside the backticks are no command (typed as printed, `a|b` is a shell pipe). Neither is
    a command with a word that only stands for a value (round 3, T12: `fd tool attach --name N --tip x,y,z`, `fd
    tool set --joint-limit JOINT=LO:HI`, `fd draw FILE`, `fd hw lock --reason "..."`: nothing is filled in for such
    a word any more), or the bare name of a command that reads words of its own (`fd tool jaw`)."""
    if "|" in cmd or "..." in cmd or re.search(r"[()\[\]<>{}]", cmd):
        return "not one command: alternatives, an ellipsis or brackets inside the backticks"
    try:
        words = shlex.split(cmd)[1:]
    except ValueError as e:
        return f"cannot be split into words ({e})"
    unknown = [w for w in words for v in [w.partition("=")[2] if w.startswith("--") and "=" in w else w]
               if not v.startswith("-") and _PLACEHOLDER.fullmatch(v)]
    if unknown:
        return f"a placeholder that stands for no value here: {unknown}"
    argv, own, rest = fd._split_own_words(words)
    if own == "jaw_args" and not rest:
        return "only the name of a command (the words after it are missing)"
    sink = io.StringIO()

    def last(code):
        return sink.getvalue().strip().splitlines()[-1] if sink.getvalue().strip() else f"exit code {code}"

    try:
        with contextlib.redirect_stderr(sink), contextlib.redirect_stdout(sink):
            fd.build_parser().parse_args(argv)
            rest = rest[1:] if rest[:1] == ["--"] else rest
            if "--sim" in rest:
                return "--sim after the command"
            if own == "jaw_args" and rest and rest[0] != "help" and not any(w in ("--help", "-h") for w in rest):
                tooljaw.build_parser().parse_args(rest)
            elif own == "calib_air_args" and rest[:1] not in (["--help"], ["-h"]):
                aircal_cli.build_parser().parse_args(rest)
            elif own == "paint_args":
                paintcli.build_parser().parse_args(rest)
    except SystemExit as e:
        return None if e.code == 0 else last(e.code)
    except Exception as e:      # (each module's own "cannot read these words")
        return f"{type(e).__name__}: {e}"
    return None


def _printed_texts(path: str) -> list:
    """[(line, text)] of the string constants of a module that a result can print: not its docstrings, not the help
    texts of its parser (comments are no constants). Pieces joined with + and f-strings are one text, with {..}
    where the program puts a value in; a choice between two sentences gives each of them."""
    import ast
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    skip = set()
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if (isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and body and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str)):
            skip.add(id(body[0].value))
        if isinstance(node, ast.Call):
            for kw in node.keywords:
                if kw.arg in ("help", "description", "epilog", "metavar"):
                    skip.update(id(n) for n in ast.walk(kw.value))

    def text(n) -> str:
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            return n.value
        if isinstance(n, ast.JoinedStr):
            return "".join(v.value if isinstance(v, ast.Constant) and isinstance(v.value, str) else "{..}" for v in n.values)
        if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
            return text(n.left) + text(n.right)
        return "{..}"

    out = []

    def visit(n):
        if id(n) in skip:
            return
        if isinstance(n, (ast.Constant, ast.JoinedStr)) or (isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add)):
            t = text(n)
            if "`" in t:
                out.append((n.lineno, t))
            for part in ast.walk(n):         # what is computed inside it may hold sentences of its own
                if part is not n and isinstance(part, (ast.IfExp, ast.FormattedValue, ast.Call, ast.Subscript, ast.Dict)):
                    for child in ast.iter_child_nodes(part):
                        visit(child)
            return
        for child in ast.iter_child_nodes(n):
            visit(child)

    visit(tree)
    return out


def test_zx_no_text_of_fd_tooling_and_tooljaw_puts_a_placeholder_or_a_bare_name_into_backticks():
    """Round 3 (T12), for the texts that no test brings up as a result (the advice after a failed pick or take, the
    refusals of rare states): in the strings of fd.py, tooling.py and tooljaw.py every `fd ...` inside backticks
    is ONE command that the parsers take as it is printed. Left were `fd take --id N`, `fd pick --id M --into-id
    K`, `fd bench --tier T --accept`, `fd snap --crop x0,y0,x1,y1 --scale 4 --plain`, `fd tip`, `fd draw`, `fd tool
    jaw`, `fd hw lock --reason "..."`: a value the program does not know and the name of a command stand in
    words. Where the program puts a value in ({..}), the results of the test below show the command."""
    seen, bad = 0, []
    for mod in (fd, TL, tooljaw):
        for line, text in _printed_texts(mod.__file__):
            for c in re.findall(r"`(fd[^`]*)`", text):
                seen += 1
                if "{..}" in c:
                    continue
                why = _accepted(c) if c.startswith("fd ") else "no command"
                if why:
                    bad.append((os.path.basename(mod.__file__), line, c, why))
    assert seen >= 200, seen
    assert not bad, bad
    # the reader itself: a docstring and a help text are left out, a choice gives both sentences
    probe = os.path.join(_TMP, "probe_texts.py")
    with open(probe, "w", encoding="utf-8") as fh:
        fh.write('"""`fd draw FILE` in a docstring."""\n'
                 'import argparse\n'
                 'p = argparse.ArgumentParser(description="`fd tool jaw`")\n'
                 'p.add_argument("--to", help="as in `fd tool jaw open --to DEG`")\n'
                 'def f(a, b):\n'
                 '    """`fd grip` here too."""\n'
                 '    # `fd tip` in a comment\n'
                 '    return ("x `fd ready` " + (f"`fd tool jaw close --to {a}`" if b else "`fd hw lock --reason \\"...\\"`")\n'
                 '            + " y `fd take --id N`")\n')
    got = sorted(c for _line, t in _printed_texts(probe) for c in re.findall(r"`(fd[^`]*)`", t))
    assert got == ["fd hw lock --reason \"...\"", "fd ready", "fd take --id N", "fd tool jaw close --to {..}"], got
    os.remove(probe)


def test_zy_every_command_fd_names_in_its_own_results_is_one_command_that_parses():
    """Round 2 (R9), for the results fd.py itself answers (tool attach / status / set / rest / detach / calibrate,
    ready, draw, hw, the refusals of misplaced words): one complete command per pair of backticks. `fd tool
    status` without a tool said "`fd tool attach --name N (--tip x,y,z | --along-fingers LEN | --across-fingers
    LEN)`", the refusal of a second attach "`fd tool set ...`" and "`fd tool attach --replace ...`". The commands
    of `paint`, `calib-air` and `tool jaw` are checked by their own suites.

    Round 3 (T12): what a result prints is parsed AS IT IS PRINTED. Nothing is filled in for a placeholder, and the
    bare name of a command is no command: `fd tool attach --name N --tip x,y,z`, `fd tool set --joint-limit
    JOINT=LO:HI`, `fd draw FILE`, `fd hw lock --reason "..."`, `fd draw`, `fd tool attach`, `fd --sim tool jaw`
    were printed in backticks, and each is refused (or does something else) when it is typed."""
    # results of every kind (also when this test runs alone)
    held = brush(on_jaw="moving_outer", gripper_goal_raw=FIXED - 25, tip_measured=dict(MEASURED))
    for tool, with_setup in ((None, False), (pen(), False), (pen(**PEN_CAL), False), (brush(), True), (held, True), (brush(on_jaw="between"), False)):
        with tool_file(tool), setup_file(SETUP if with_setup else None), pen_calibration():
            for words in (("tool", "status"), ("tool", "attach", "--name", "x", "--along-fingers", "0.06", "--no-robot"),
                          ("tool", "set", "--tip", "0.01,0.0,0.06", "--no-robot", "--dry"), ("tool", "set", "--joint-limit", "foo"),
                          ("tool", "set", "--joint-limit", "wrist_flex=200:300", "--no-robot"), ("tool", "rest", "--dry"), ("tool", "rest"),
                          ("tool", "rest", "--pan", "-100", "--clearance", "0.04", "--dry"), ("tool", "calibrate", "--dry"), ("draw", "cat", "--dry"),
                          ("ready",), ("park",), ("grip", "0.5"), ("tip", "0.25", "0.0", "0.03", "--contact"), ("tool", "detach", "--dry"),
                          ("hw", "status"), ("hw", "lock"), ("hw", "unlock"), ("paint", "--sim", "status"), ("tool", "jaw", "--sim", "status")):
                run("--sim", *words)
            if tool is not None:
                arm = Arm(bus=SimBus(), cfg=load_config(), tool=tool).open()
                run("--sim", "ready", bus=standing(tool, q_deg=np.degrees(TL.rest_q(arm, tool)), torque=False, gripper=(tool.gripper_raw or 2061) + 60))
            run("tool", "status")
            run("hw", "status")
            run("ready")                                                   # (the real arm: the scratch lock answers)
    own = [(argv, res) for argv, res in _RESULTS
           if next((w for w in argv if not w.startswith("-")), "") not in ("paint", "calib-air")
           and [w for w in argv if not w.startswith("-")][:2] != ["tool", "jaw"]]
    cmds = sorted({c for _argv, res in own for c in _commands(res)})
    assert len(cmds) >= 30 and len(own) >= 120, (len(cmds), len(own))
    bad = [(c, why) for c in cmds for why in [_accepted(c)] if why and c not in _LEFT_ELSEWHERE]
    assert not bad, bad
    # (every one on that list is refused as it is printed: none of them hides a command that works)
    assert all(_accepted(c) for c in _LEFT_ELSEWHERE), [c for c in _LEFT_ELSEWHERE if not _accepted(c)]
    # a misplaced --sim is answered by fd itself: the corrected call it prints is checked as well
    fixes = sorted({res["next"].strip("`") for argv, res in _RESULTS
                    if res.get("stage") == "options" and "--sim must stand" in (res.get("reason") or "")})
    assert fixes and all(c.startswith("fd --sim ") for c in fixes) and not [(c, _accepted(c)) for c in fixes if _accepted(c)], fixes
    # the ones an operator meets first are among them, as they are typed
    for want in ("fd tool attach --help", "fd paint status", "fd --sim paint status", "fd tool status", "fd --sim tool status",
                 "fd tool rest", "fd --sim paint rest", "fd tool detach", "fd --sim tool detach", "fd hw status", "fd hw unlock",
                 "fd hw lock", "fd draw cat --dry", "fd tool jaw status", "fd tool jaw mount fixed", "fd tool calibrate",
                 "fd --sim tool calibrate --pen --dry", "fd --sim tool set --help"):
        assert want in cmds, (want, [c for c in cmds if c.startswith(want[:11])])
    # and the check itself can fail: the spellings that did not work are told apart ...
    for wrong in ("fd tool attach --name N (--tip x,y,z | --along-fingers LEN | --across-fingers LEN)", "fd tool set ...",
                  "fd tool attach --replace ...", "fd tool jaw mount moving_outer|fixed|between", "fd tool calibrate --pencil",
                  "fd paint status --sim", "fd --sim ...", "fd tool calibrate --stage STAGE", "fd tool rest --over-the-canvas",
                  # ... and so are the ones round 3 took out of the results: a placeholder, the bare name of a command
                  "fd tool attach --name N --tip x,y,z", "fd tool set --joint-limit JOINT=LO:HI", "fd tool set --joint-limit wrist_flex=LO:HI",
                  "fd tool set --joint-limit JOINT=none", "fd draw FILE", "fd hw lock --reason \"...\"", "fd take --id N",
                  "fd pick --id M --into-id K", "fd bench --tier T --accept", "fd snap --crop x0,y0,x1,y1 --scale 4 --plain",
                  "fd calib-air --from DIR --save", "fd paint run PLAN --layer 0 --dry", "fd paint contact --read BEFORE.jpg AFTER.jpg",
                  "fd tool rest --pan P", "fd tool rest --pan", "fd tool attach", "fd tool jaw", "fd --sim tool jaw", "fd draw",
                  "fd grip", "fd tip", "fd --sim", "fd paint"):
        assert _accepted(wrong), wrong
    assert _accepted("fd hw lock") is None and _accepted("fd tool calibrate --pen --stage contact") is None
    assert _accepted("fd tool rest --pan -100 --clearance 0.04 --over-canvas") is None and _accepted("fd --sim paint rest") is None
    assert _accepted("fd tool set --joint-limit wrist_flex=20:97") is None and _accepted("fd tool set --tip=0.0008,0.0212,0.0777") is None
    assert _accepted("fd draw cat --dry") is None and _accepted("fd grip 0.5") is None and _accepted("fd take --id 3 --hollow") is None
    assert _accepted("fd calib-air --from /tmp/photos --save") is None and _accepted("fd paint run /tmp/plan.json --layer 0 --dry") is None
    assert not _touched


# ----------------------------------------------------------------------------- the rig
def test_zz_the_rig_files_and_its_run_log_were_never_touched():
    """Runs last (by its name)."""
    assert os.environ["FD_TOOL"] == TOOL_PATH and os.environ["FD_PAINT_SETUP"] == SETUP_PATH and os.environ["FD_RUNS"] == RUNS
    assert os.environ["TAZUNA_HW_LOCK"] == HW_LOCK and os.path.exists(HW_LOCK), "the scratch lock was taken off by a test"
    assert TL.TOOL_FILE == os.path.join(tazuna_paths.ROOT, "tool_active.json")
    assert _tree_files() == _TREE_FILES, "a test created, changed or removed a state file of the rig"
    assert not _touched, f"a test reached for hardware: {_touched}"
    # the commands of this suite logged into the scratch folder, every one of them ...
    records = [json.loads(ln) for p in sorted(glob.glob(os.path.join(RUNS, "*", "events.jsonl"))) for ln in open(p, encoding="utf-8")]
    assert [r["cmd"] for r in records] == [next(w for w in argv if not w.startswith("-")) for argv in _LOGGED], (len(records), len(_LOGGED))
    # ... and none of their records is in the rig's run log (another process may write it meanwhile: only records
    # that name this suite's scratch folder count)
    token, own = os.path.basename(_TMP), []
    for p, size in _rig_logs().items():
        was = _RIG_LOGS.get(p, 0)
        if size > was:
            with open(p, "rb") as f:
                f.seek(was)
                own += [ln[:200] for ln in f.read().decode("utf-8", "replace").splitlines() if token in ln]
    assert not own, f"{len(own)} record(s) of this suite were written into the rig's run log: {own[:3]}"


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
