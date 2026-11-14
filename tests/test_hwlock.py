"""Tests for the hardware lock (tools/motion/hwlock.py and its checks in servo_bus.py, camera.py, fastcam.py, fd.py).
Run: python tests/test_hwlock.py [name-part ...]      (no hardware; about 15 s)

What they show: with the lock file present the real servo bus and the real camera are refused BEFORE the device
is touched (an audit hook fails the test at the first open of /dev/ttyACM*, /dev/ttyUSB*, /dev/video* or call of
v4l2-ctl), the refusal says how to unlock, the simulated arm keeps working, and `fd hw lock | status | unlock`
write, show and remove the file. A lock that is set keeps its first reason and time. `fd --sim hw lock|unlock`
change nothing (--sim does not make the lock a simulated one). `fd hw lock` / `status` list the copies of the
harness next to the folder whose servo bus does not look at the lock, and say so.

Every test uses $TAZUNA_HW_LOCK (a scratch file): the user's real lock file is never read or written.

How `fd` is started here (review of 2026-11-13: the suite ran fd.py as plain processes; every run appended nine
events to the rig's run log, among them "hw unlock ... removed: true", and between its two real-arm commands
and /dev/ttyACM0 stood nothing but the check under test). Each `fd` of this suite is a process that
- carries the no-hardware audit hook itself (the hook of this file does not reach a child process),
- has names for the servo port and the camera that do not exist (they are constants of servo_bus.py and
  camera.py, not keys of the configuration, so the child replaces them), and no cv2.VideoCapture,
- writes its run log to a scratch folder ($FD_RUNS) and cannot open a file under the rig's runs/ for writing,
- reads the configuration of the test data ($FD_CONFIG), not the rig's.
A command for the real arm is only started while the scratch lock is set. The last test shows that the scratch
run log holds every command of the suite and the rig's run log none of them.
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "tools"))
import tazuna_paths  # noqa: E402,F401

import contextlib
import glob
import io
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
_TMP = tempfile.mkdtemp(prefix="tz_hwlock_")
LOCK = os.path.join(_TMP, "hardware.lock")
RUNS = os.path.join(_TMP, "runs")                                   # the run log of every fd of this suite
RIG_RUNS = os.path.join(tazuna_paths.ROOT, "runs")                  # ... and where none of it may go
TOOLS_DIR = os.path.join(tazuna_paths.ROOT, "tools")
os.environ["TAZUNA_HW_LOCK"] = LOCK
os.environ["FD_TOOL"] = os.path.join(_TMP, "tool_active.json")      # no tool, whatever the tree's arm carries
os.environ["FD_RUNS"] = RUNS
os.environ["FD_CONFIG"] = os.path.join(HERE, "testdata", "pen_20261111", "config.json")
os.environ["FD_CAMERA_MODEL"] = os.path.join(HERE, "testdata", "pen_20261111", "camera_model.json")


def _rig_logs() -> dict:
    """{run log of the rig: its size} for every run folder under the rig's runs/."""
    out = {}
    for p in glob.glob(os.path.join(RIG_RUNS, "*", "events.jsonl")):
        try:
            out[p] = os.path.getsize(p)
        except OSError:
            pass
    return out


_RIG_LOGS = _rig_logs()         # before the first fd of this suite

import hwlock  # noqa: E402
from servo_bus import Bus, BusError  # noqa: E402

_touched = []


def _no_hardware(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        p = os.fsdecode(args[0])
        if p.startswith("/dev/ttyACM") or p.startswith("/dev/ttyUSB") or p.startswith("/dev/video"):
            _touched.append(p)
            raise RuntimeError(f"test touched hardware: open({p})")
    if event == "subprocess.Popen" and args and "v4l2" in str(args[0]) + str(args[1] if len(args) > 1 else ""):
        _touched.append("v4l2-ctl")
        raise RuntimeError("test touched hardware: v4l2-ctl")


sys.addaudithook(_no_hardware)

import camera as _camera  # noqa: E402
import cv2 as _cv2  # noqa: E402
import servo_bus as _servo_bus  # noqa: E402


def _no_capture(*a, **k):
    _touched.append("cv2.VideoCapture")
    raise RuntimeError(f"test touched hardware: cv2.VideoCapture{a!r}")


# In this process too, the checks under test are not the only thing before the hardware: OpenCV opens a camera
# without Python's open() (the hook above does not see it), so there is no VideoCapture here, and the names the
# harness would use for the camera and the servo port are ones that do not exist.
_cv2.VideoCapture = _no_capture
_camera.DEV = "/dev/video_none_for_tests"
_servo_bus.Bus.__init__.__defaults__ = ("/dev/ttyACM_none_for_tests",) + tuple(_servo_bus.Bus.__init__.__defaults__[1:])
assert Bus().port == "/dev/ttyACM_none_for_tests" and not os.path.exists(Bus().port) and not os.path.exists(_camera.DEV)

import fd  # noqa: E402      (after $FD_RUNS: the commands this process runs itself log to the scratch folder too)

assert os.path.realpath(fd.RUNS) == os.path.realpath(RUNS), "fd does not honour $FD_RUNS: this suite would write the rig's run log"


class Skip(Exception):
    pass


# What every `fd` child of this suite runs before fd.main: argv[1] = tools/, argv[2] = the rig's runs/ folder.
# (The name of the camera's settings program is spelled in two pieces: this text is on the child's command line,
# where the hook of THIS process would take it for a call of that program.)
_CHILD_HEAD = r'''
import os, sys
RIG_RUNS = os.path.realpath(sys.argv[2]) + os.sep
WRITES = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND
CAMERA_TOOL = "v4" "l2"
def _under_rig_runs(p):
    return isinstance(p, (str, bytes)) and (os.path.realpath(os.fsdecode(p)) + os.sep).startswith(RIG_RUNS)
def _no_hardware(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        p = os.fsdecode(args[0])
        if p.startswith(("/dev/ttyACM", "/dev/ttyUSB", "/dev/video")):
            raise RuntimeError("test touched hardware: open(%s)" % p)
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        if flags & WRITES and _under_rig_runs(p):
            raise RuntimeError("test wrote into the rig's runs/: open(%s)" % p)
    if event in ("os.mkdir", "os.remove", "os.rmdir") and args and _under_rig_runs(args[0]):
        raise RuntimeError("test changed the rig's runs/: %s(%s)" % (event, os.fsdecode(args[0])))
    if event == "os.rename" and len(args) > 1 and (_under_rig_runs(args[0]) or _under_rig_runs(args[1])):
        raise RuntimeError("test changed the rig's runs/: os.rename")
    if event == "subprocess.Popen" and args and CAMERA_TOOL in str(args[0]) + str(args[1] if len(args) > 1 else ""):
        raise RuntimeError("test touched hardware: the camera's settings program")
sys.addaudithook(_no_hardware)
sys.path.insert(0, sys.argv[1])
import tazuna_paths
import camera, cv2, servo_bus
NO_PORT, NO_CAMERA = "/dev/ttyACM_none_for_tests", "/dev/video_none_for_tests"
servo_bus.Bus.__init__.__defaults__ = (NO_PORT,) + tuple(servo_bus.Bus.__init__.__defaults__[1:])
servo_bus.LOCK_PATH = os.path.join(os.path.dirname(os.environ["TAZUNA_HW_LOCK"]), "bus.lock")
camera.DEV = NO_CAMERA
def _no_capture(*a, **k):
    raise RuntimeError("test touched hardware: cv2.VideoCapture%r" % (a,))
cv2.VideoCapture = _no_capture
assert servo_bus.Bus().port == NO_PORT and not os.path.exists(NO_PORT) and not os.path.exists(NO_CAMERA)
import fd
assert os.path.realpath(fd.RUNS) == os.path.realpath(os.environ["FD_RUNS"]) and not _under_rig_runs(fd.RUNS), fd.RUNS
'''
_FD_CODE = _CHILD_HEAD + "sys.exit(fd.main(sys.argv[3:]))\n"
_COMMANDS = []          # every fd command of this suite that wrote a run log record, in order


def _child(code, *argv):
    return subprocess.run([sys.executable, "-c", code, TOOLS_DIR, RIG_RUNS, *argv], capture_output=True, text=True,
                          timeout=120, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))


def _fd(*argv):
    """`fd ARGV` in a process of its own (see the top of this file). Returns (result, exit code)."""
    assert os.environ["TAZUNA_HW_LOCK"] == LOCK and os.environ["FD_RUNS"] == RUNS
    if "--sim" not in argv and argv[0] != "hw":
        # a command for the REAL arm or camera: only the lock keeps it from them, so the lock must be there
        assert os.path.exists(LOCK), f"`fd {' '.join(argv)}` would be started without the scratch lock set"
    p = _child(_FD_CODE, *argv)
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    assert lines, (argv, p.returncode, p.stderr[-800:])
    assert "test touched hardware" not in p.stdout + p.stderr and "the rig's runs/" not in p.stderr, p.stderr[-800:]
    _COMMANDS.append(next(w for w in argv if not w.startswith("-")))
    return json.loads(lines[-1]), p.returncode


def _fd_here(*argv):
    """fd.main(ARGV) in this process (for the tests that put something of their own into fd)."""
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fd.main(list(argv))
    _COMMANDS.append(next(w for w in argv if not w.startswith("-")))
    return json.loads(buf.getvalue().strip().splitlines()[-1]), rc


def _house(r):
    """A result in the house style: ok, stage, next (one line), say (Japanese), a reason when it is not ok."""
    assert isinstance(r.get("ok"), bool) and r.get("stage"), r
    assert isinstance(r.get("next"), str) and r["next"] and "\n" not in r["next"], r
    assert isinstance(r.get("say"), str) and any(0x3040 <= ord(ch) <= 0x30FF or 0x4E00 <= ord(ch) <= 0x9FFF for ch in r["say"]), r
    assert r["ok"] or r.get("reason"), r
    return r


def _clean():
    if os.path.exists(LOCK):
        os.remove(LOCK)


def test_no_file_means_not_locked():
    _clean()
    assert hwlock.path() == LOCK
    assert hwlock.info() is None and hwlock.message("The servo bus") is None
    assert hwlock.unlock() is False


def test_lock_writes_reason_and_time_and_unlock_removes_it():
    _clean()
    d = hwlock.lock("agents are editing the paint feature")
    assert d["reason"] == "agents are editing the paint feature" and d["since"] and d["file"] == LOCK and d["already"] is False
    msg = hwlock.message("The servo bus on /dev/ttyACM0")
    assert "locked" in msg and "agents are editing" in msg and "fd hw unlock" in msg and "nothing moved" in msg
    assert hwlock.unlock() is True and hwlock.info() is None


def test_a_second_lock_keeps_the_first_reason_and_time():
    """Until 2026-11-13 `fd hw lock` on a locked rig replaced the reason and the time without a word."""
    _clean()
    first = hwlock.lock("the user: the brush is taped on, the arm is folded")
    with open(LOCK, "w", encoding="utf-8") as f:            # as if it had been set an hour ago
        json.dump({"reason": first["reason"], "since": "2026-11-13T19:26:43", "by": "user"}, f, ensure_ascii=False)
    with open(LOCK, "rb") as f:
        text = f.read()
    again = hwlock.lock("an agent, later, with another reason")
    assert again["already"] is True and again["reason"] == first["reason"] and again["since"] == "2026-11-13T19:26:43", again
    assert again["by"] == "user" and open(LOCK, "rb").read() == text, "a second lock rewrote the lock file"
    assert "the brush is taped on" in hwlock.message("The servo bus") and "2026-11-13T19:26:43" in hwlock.message("The servo bus")
    # the command: ok (the rig IS locked), says that it was locked already, shows the first reason, writes nothing
    r, code = _fd("hw", "lock", "--reason", "二度目")
    _house(r)
    assert code == 0 and r["ok"] and r["locked"] is True and r["already"] is True and r["stage"] == "locked", r
    assert r["reason"] == first["reason"] and r["since"] == "2026-11-13T19:26:43" and r["asked_reason"] == "二度目", r
    assert "already" in r["next"] and "NOT written" in r["next"] and "すでにロック" in r["say"] and "二度目" not in r["say"], r
    assert open(LOCK, "rb").read() == text
    r, code = _fd("hw", "lock")                               # no reason given: nothing "was not written"
    assert code == 0 and r["already"] is True and "asked_reason" not in r and "NOT written" not in r["next"], r
    r, code = _fd("hw", "status")
    assert r["locked"] is True and r["reason"] == first["reason"] and r["since"] == "2026-11-13T19:26:43", r
    _clean()
    r, code = _fd("hw", "lock", "--reason", "二度目")          # after an unlock it is a first lock again
    assert code == 0 and r["already"] is False and r["reason"] == "二度目" and "asked_reason" not in r, r
    _clean()


def test_a_damaged_lock_file_still_locks():
    _clean()
    with open(LOCK, "w") as f:
        f.write("{ not json")
    d = hwlock.info()
    assert d is not None and "cannot be read" in d["reason"] and d["damaged"] is True
    assert "hardware is locked" in hwlock.message("The servo bus")
    # it has no reason or time to keep: `lock` writes it anew (and it is a lock all the while)
    d = hwlock.lock("written again")
    assert d["already"] is False and d["reason"] == "written again" and d["since"] and "damaged" not in d, d
    assert json.load(open(LOCK))["reason"] == "written again" and not os.path.exists(LOCK + ".tmp")
    _clean()


def test_real_ports_and_test_ports():
    assert hwlock.real_port("/dev/ttyACM0") and hwlock.real_port("/dev/ttyUSB1")
    assert not hwlock.real_port("/dev/pts/7") and not hwlock.real_port(os.path.join(_TMP, "pty"))


def test_bus_open_is_refused_before_the_port_is_touched():
    _clean()
    hwlock.lock("test")
    del _touched[:]
    bus = Bus(port="/dev/ttyACM0", lock_path=os.path.join(_TMP, "bus.lock"))
    try:
        bus.open(wait_s=0.2)
    except BusError as e:
        assert "hardware is locked" in str(e) and "/dev/ttyACM0" in str(e)
    else:
        raise AssertionError("the bus was opened although the hardware is locked")
    assert not _touched and not bus.is_open
    assert not os.path.exists(os.path.join(_TMP, "bus.lock")), "the refusal must come before the flock file is made"
    _clean()


def test_camera_is_refused_before_the_device_is_touched():
    import camera as C
    _clean()
    hwlock.lock("test")
    del _touched[:]
    for call in (lambda: C.grab(), lambda: C.grab(exposure="paper"), C.lock_settings, C.restore_auto,
                 lambda: C.set_exposure("paper")):
        try:
            call()
        except hwlock.HardwareLocked as e:
            assert "hardware is locked" in str(e)
        else:
            raise AssertionError("a camera call went through although the hardware is locked")
    assert C.read_settings() == {} and C.settings_locked() is None
    assert not _touched
    _clean()


def test_fast_camera_is_refused():
    import fastcam
    _clean()
    hwlock.lock("test")
    del _touched[:]
    try:
        fastcam.FastCamera(start=True)
    except hwlock.HardwareLocked:
        pass
    else:
        raise AssertionError("FastCamera started although the hardware is locked")
    assert not _touched
    _clean()


def test_fd_hw_commands_and_the_refusal_of_a_real_command():
    _clean()
    r, code = _fd("hw", "status")
    _house(r)
    assert code == 0 and r["ok"] and r["locked"] is False and r["file"] == LOCK and "fd hw lock" in r["next"], r
    r, code = _fd("hw", "lock", "--reason", "演習")
    _house(r)
    assert code == 0 and r["locked"] is True and r["reason"] == "演習" and r["already"] is False and os.path.exists(LOCK), r
    assert "fd hw unlock" in r["next"] and "ロックしました" in r["say"], r
    r, code = _fd("hw", "status")
    _house(r)
    assert r["locked"] is True and r["reason"] == "演習" and "演習" in r["say"] and "fd hw unlock" in r["next"], r
    r, code = _fd("state")                      # the real arm: refused, with the way out
    _house(r)
    assert code == 1 and r["ok"] is False and r["stage"] == "hardware" and "fd hw unlock" in r["next"], r
    assert "hardware is locked" in r["message"] and "演習" in r["message"] and "sim" not in r, r
    r, code = _fd("snap", "--no-robot")         # the real camera: refused
    assert code == 1 and r["ok"] is False and r["stage"] == "hardware" and "hardware is locked" in r["message"], r
    r, code = _fd("hw", "unlock")
    _house(r)
    assert code == 0 and r["removed"] is True and r["locked"] is False and not os.path.exists(LOCK), r
    assert r["was"]["reason"] == "演習" and "fd hw lock" in r["next"] and "外しました" in r["say"], r
    r, code = _fd("hw", "unlock")
    _house(r)
    assert code == 0 and r["removed"] is False and "was" not in r and "ロックされていませんでした" in r["say"], r


def test_fd_sim_hw_lock_and_unlock_change_nothing():
    """2026-11-13: `fd --sim hw unlock` removed the lock (ok true, 「実機のロックを外しました」). "--sim only" is what
    a program is allowed while the arm is locked: with it the lock can be looked at, not set and not taken off."""
    _clean()
    r, code = _fd("--sim", "hw", "lock", "--reason", "from a simulation")
    _house(r)
    assert code == 1 and r["ok"] is False and r["stage"] == "options" and r["error"] == "SafetyError" and r["sim"] is True, r
    assert not os.path.exists(LOCK) and r["locked"] is False and "--sim" in r["reason"] and "REAL" in r["reason"], r
    assert "fd hw lock" in r["next"] and "without --sim" in r["next"] and "実行しませんでした" in r["say"], r
    hwlock.lock("the user's lock")
    with open(LOCK, "rb") as f:
        text = f.read()
    r, code = _fd("--sim", "hw", "unlock")
    _house(r)
    assert code == 1 and r["ok"] is False and r["stage"] == "options" and r["locked"] is True and "removed" not in r, r
    assert os.path.exists(LOCK) and open(LOCK, "rb").read() == text, "`fd --sim hw unlock` touched the lock"
    assert "fd hw unlock" in r["next"] and "without --sim" in r["next"] and "ロック中" in r["say"], r
    r, code = _fd("--sim", "hw", "lock", "--reason", "x")     # ... nor "locked already" for a simulated lock
    assert code == 1 and r["stage"] == "options" and r["locked"] is True and open(LOCK, "rb").read() == text, r
    r, code = _fd("--sim", "hw", "status")                    # looking is fine
    _house(r)
    assert code == 0 and r["ok"] and r["locked"] is True and r["reason"] == "the user's lock" and r["sim"] is True, r
    r, code = _fd("state")                                    # and the real arm is as locked as before
    assert code == 1 and r["stage"] == "hardware" and "the user's lock" in r["message"], r
    _clean()


def test_a_lock_file_that_cannot_be_written_is_a_refusal_not_a_traceback():
    blocker = os.path.join(_TMP, "a_file_where_a_folder_should_be")
    with open(blocker, "w") as f:
        f.write("x")
    os.environ["TAZUNA_HW_LOCK"] = os.path.join(blocker, "hardware.lock")
    try:
        r, code = _fd_here("hw", "lock", "--reason", "cannot be written")
        _house(r)
        assert code == 1 and r["ok"] is False and r["stage"] == "lock" and r["locked"] is False and "trace" not in r, r
        assert "could not be written" in r["reason"] and "fd hw status" in r["next"] and "ロックされていません" in r["say"], r
        r, code = _fd_here("hw", "status")
        assert code == 0 and r["locked"] is False, r
    finally:
        os.environ["TAZUNA_HW_LOCK"] = LOCK
    assert hwlock.path() == LOCK and open(blocker).read() == "x"


def test_the_simulated_arm_works_while_locked():
    _clean()
    hwlock.lock("test")
    try:
        r, code = _fd("--sim", "state")
        assert code == 0 and r["ok"] and r.get("sim") is True, r
        r, code = _fd("--sim", "tool", "status")
        assert code == 0 and r["ok"], r
    finally:
        _clean()


def _harness_copy(folder, layout="flat", lock=False, launcher=True):
    """A stand-in for a copy of the harness: a servo_bus.py where that layout has it, with or without the lock."""
    bus = os.path.join(folder, "servo_bus.py" if layout == "flat" else os.path.join("tools", "motion", "servo_bus.py"))
    os.makedirs(os.path.dirname(bus), exist_ok=True)
    with open(bus, "w") as f:
        f.write('PORT = "/dev/ttyACM0"\n' + ("import hwlock\n" if lock else "") + "class Bus:\n    pass\n")
    if launcher:
        with open(os.path.join(folder, "fd.py"), "w") as f:
            f.write("# fd\n")
    return folder


def test_copies_of_the_harness_that_do_not_know_the_lock_are_listed():
    """2026-11-13: five older copies beside the live folder (and six one folder down) had an fd of their own and
    a servo_bus.py without the lock; `fd hw lock` said that servos and camera would not be opened."""
    base = os.path.join(_TMP, "copies")
    live = _harness_copy(os.path.join(base, "live"), "feature", lock=True)
    os.symlink(live, os.path.join(base, "live_link"))                                  # the folder under another name
    old = _harness_copy(os.path.join(base, "old_flat"))                                # beside it, before the lock
    _harness_copy(os.path.join(old, "refs", "snap1"), launcher=False)                  # (a snapshot inside a copy: part of it)
    _harness_copy(os.path.join(base, "new_clone"), "feature", lock=True)               # a clone that has the lock
    a = _harness_copy(os.path.join(base, "work", "A"))                                 # one folder down
    b = _harness_copy(os.path.join(base, "work", "B"), "feature")                      # ... a checkout by feature, before the lock
    _harness_copy(os.path.join(base, "work", "C"), lock=True)
    _harness_copy(os.path.join(base, "work", "deep", "x"))                             # two folders down: not looked for
    _harness_copy(os.path.join(base, ".hidden"))
    os.makedirs(os.path.join(base, "notes"))
    with open(os.path.join(base, "notes", "servo_bus.py.txt"), "w") as f:
        f.write("not a module\n")
    want = sorted([old, a, b])
    assert hwlock.copies_without_lock(live) == want, hwlock.copies_without_lock(live)
    assert hwlock.copies_without_lock(os.path.join(base, "live_link")) == want          # asked through the link: the same
    alone = _harness_copy(os.path.join(_TMP, "alone", "live"), "feature", lock=True)
    assert hwlock.copies_without_lock(alone) == []
    one = _harness_copy(os.path.join(_TMP, "one", "live"), "feature", lock=True)
    one_old = _harness_copy(os.path.join(_TMP, "one", "before_the_lock"))
    assert hwlock.copies_without_lock(one) == [one_old]
    assert hwlock.copies_without_lock(os.path.join(_TMP, "no", "such", "folder")) == []
    # the folder this module really lies in is the default, and it is never one of its own "copies"
    assert hwlock.ROOT == tazuna_paths.ROOT and tazuna_paths.ROOT not in hwlock.copies_without_lock()
    # nothing was written or run in those folders
    assert sorted(os.listdir(old)) == ["fd.py", "refs", "servo_bus.py"] and sorted(os.listdir(a)) == ["fd.py", "servo_bus.py"]
    # the commands list them and say what the lock does not cover
    _clean()
    real_root = fd.ROOT
    try:
        fd.ROOT = live
        r, code = _fd_here("hw", "lock", "--reason", "copies")
        _house(r)
        assert code == 0 and r["locked"] is True and r["copies_without_lock"] == want, r
        assert "3 older copies" in r["next"] and "copies_without_lock" in r["next"] and "NOT" in r["next"], r
        assert "3 個" in r["say"] and "古いコピー" in r["say"] and "このフォルダのハーネス" in r["say"], r
        r, code = _fd_here("hw", "status")
        _house(r)
        assert r["locked"] is True and r["copies_without_lock"] == want and "3 older copies" in r["next"] and "3 個" in r["say"], r
        r, code = _fd_here("hw", "lock")                          # locked already: the copies are told again
        assert r["already"] is True and r["copies_without_lock"] == want and "3 older copies" in r["next"], r
        _clean()
        r, code = _fd_here("hw", "status")                        # not locked: listed, nothing to warn of
        assert r["locked"] is False and r["copies_without_lock"] == want and "older cop" not in r["next"], r
        fd.ROOT = one                                             # one such copy
        r, code = _fd_here("hw", "lock")
        assert r["copies_without_lock"] == [one_old] and "1 older copy" in r["next"] and "does NOT" in r["next"], r
        assert "1 個" in r["say"], r
        _clean()
        fd.ROOT = alone                                           # no such copy: no such sentence
        r, code = _fd_here("hw", "lock", "--reason", "alone")
        _house(r)
        assert r["copies_without_lock"] == [] and "older cop" not in r["next"] and "古いコピー" not in r["say"], r
        r, code = _fd_here("hw", "status")
        assert r["copies_without_lock"] == [] and "older cop" not in r["next"] and "古いコピー" not in r["say"], r
    finally:
        fd.ROOT = real_root
        _clean()
    # and the texts no longer promise that every copy on the computer honours the lock
    for text in (hwlock.__doc__, fd.cmd_hw.__doc__):
        assert "every clone and working copy honours" not in text and "any other copy on this computer" not in text, text[:200]
    assert "copies_without_lock" in hwlock.__doc__ and "not look" in fd.cmd_hw.__doc__


def test_lock_ignorant_harness_code_inside_the_project_folder_is_listed_too():
    """Round 2 (R18). tests/refs/patches/motion-safety/snap1 and snap3 each hold an fd.py and a servo_bus.py with
    PORT /dev/ttyACM0 and no lock check, backups/v1_code another servo_bus.py: `python tests/refs/patches/
    motion-safety/snap1/fd.py ready` would open the servo bus while `fd hw status` says "locked". Only the
    folders NEXT TO the project folder were looked at."""
    base = os.path.join(_TMP, "inside")
    live = _harness_copy(os.path.join(base, "live"), "feature", lock=True)
    snap1 = _harness_copy(os.path.join(live, "tests", "refs", "patches", "motion-safety", "snap1"))
    snap3 = _harness_copy(os.path.join(live, "tests", "refs", "patches", "motion-safety", "snap3"))
    v1 = _harness_copy(os.path.join(live, "backups", "v1_code"), launcher=False)        # a bus module with no fd beside it
    nested = _harness_copy(os.path.join(live, "work", "old_checkout"), "feature")       # a whole copy by feature, inside
    _harness_copy(os.path.join(live, "work", "new_checkout"), "feature", lock=True)     # ... and one that has the lock
    _harness_copy(os.path.join(live, "tests", "refs", "snap_with_lock"), lock=True)
    _harness_copy(os.path.join(live, ".git", "modules", "x"))                           # hidden: not entered
    _harness_copy(os.path.join(live, "tools", "__pycache__"))
    with open(os.path.join(live, "tests", "refs", "servo_bus.py.txt"), "w") as f:       # (the name must be the module's)
        f.write("not a module\n")
    beside = _harness_copy(os.path.join(base, "old_flat"))
    os.symlink(beside, os.path.join(live, "link_to_old"))                               # a link is not followed
    want = sorted([beside, snap1, snap3, v1, nested])
    assert hwlock.copies_without_lock(live) == want, hwlock.copies_without_lock(live)
    assert hwlock._inside_without_lock(live) == [v1, snap1, snap3, nested], hwlock._inside_without_lock(live)
    # the project folder's own bus module is not a copy of itself, whatever it says and in either layout
    for layout in ("flat", "feature"):
        bare = _harness_copy(os.path.join(_TMP, "inside_own_" + layout, "live"), layout)        # (no lock in its own module)
        deep = _harness_copy(os.path.join(bare, "refs", "snap"))
        assert hwlock.copies_without_lock(bare) == [deep], (layout, hwlock.copies_without_lock(bare))
    # nothing was written or run in those folders
    assert sorted(os.listdir(snap1)) == ["fd.py", "servo_bus.py"] and sorted(os.listdir(v1)) == ["servo_bus.py"]
    # the commands list them and say that some lie inside the folder
    _clean()
    real_root = fd.ROOT
    try:
        fd.ROOT = live
        r, code = _fd_here("hw", "lock", "--reason", "inside")
        _house(r)
        assert code == 0 and r["locked"] is True and r["copies_without_lock"] == want, r
        assert "5 older copies of the harness next to this folder or inside it (4 inside" in r["next"] and "do NOT look" in r["next"], r["next"]
        assert "5 個" in r["say"] and "このフォルダの近くや中に" in r["say"], r["say"]
        r, code = _fd_here("hw", "status")
        assert r["copies_without_lock"] == want and "4 inside" in r["next"] and "近くや中" in r["say"], r
        fd.ROOT = os.path.join(_TMP, "inside_own_feature", "live")                      # only inside
        r, code = _fd_here("hw", "status")
        assert len(r["copies_without_lock"]) == 1 and "1 older copy of the harness inside this folder does NOT look" in r["next"], r
    finally:
        fd.ROOT = real_root
        _clean()
    # this project folder: its own snapshots and old code are in the list (where a clone has them), itself is not
    listed = hwlock.copies_without_lock()
    for rel in (os.path.join("tests", "refs", "patches", "motion-safety", "snap1"),
                os.path.join("tests", "refs", "patches", "motion-safety", "snap3"), os.path.join("backups", "v1_code")):
        folder = os.path.join(tazuna_paths.ROOT, rel)
        bus = os.path.join(folder, "servo_bus.py")
        if os.path.isfile(bus) and "hwlock" not in open(bus, encoding="utf-8", errors="replace").read():
            assert folder in listed, (folder, listed)
    assert tazuna_paths.ROOT not in listed and os.path.join(tazuna_paths.ROOT, "tools", "motion") not in listed
    assert "INSIDE the project folder" in hwlock.__doc__ and "inside it" in fd.cmd_hw.__doc__


def test_the_fd_processes_of_this_suite_cannot_reach_the_hardware_or_the_rig_log():
    """What stands between an `fd` child of this suite and the real arm when the check under test fails: the hook
    inside the child, device names that do not exist, a run log folder of its own."""
    probe = _CHILD_HEAD + r'''
import json
out = {"runs": fd.RUNS, "port": servo_bus.Bus().port, "camera": camera.DEV, "bus_lock": servo_bus.LOCK_PATH}
for key, what in (("serial", lambda: open(NO_PORT, "rb")), ("video", lambda: open(NO_CAMERA, "rb")),
                  ("rig_log", lambda: open(os.path.join(sys.argv[2], "no_such_run_for_tests", "events.jsonl"), "a")),
                  ("rig_dir", lambda: os.mkdir(os.path.join(sys.argv[2], "no_such_run_for_tests", "sub"))),
                  ("capture", lambda: cv2.VideoCapture(camera.DEV)),
                  ("settings", lambda: __import__("subprocess").run([CAMERA_TOOL + "-ctl", "--version"], capture_output=True))):
    try:
        what()
        out[key] = "went through"
    except Exception as e:
        out[key] = "%s: %s" % (type(e).__name__, e)
print(json.dumps(out))
'''
    before = sorted(os.listdir(RIG_RUNS)) if os.path.isdir(RIG_RUNS) else None
    p = _child(probe)
    assert p.returncode == 0, p.stderr[-800:]
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert os.path.realpath(out["runs"]) == os.path.realpath(RUNS) and out["bus_lock"].startswith(_TMP), out
    assert out["port"] == "/dev/ttyACM_none_for_tests" and out["camera"] == "/dev/video_none_for_tests", out
    assert not os.path.exists(out["port"]) and not os.path.exists(out["camera"])
    for key in ("serial", "video", "capture", "settings"):
        assert out[key].startswith("RuntimeError: test touched hardware"), (key, out[key])
    for key in ("rig_log", "rig_dir"):
        assert out[key].startswith("RuntimeError: test") and "the rig's runs/" in out[key], (key, out[key])
    assert (sorted(os.listdir(RIG_RUNS)) if os.path.isdir(RIG_RUNS) else None) == before
    # a real-arm command of this suite is not even started without the scratch lock
    _clean()
    n = len(_COMMANDS)
    try:
        _fd("state")
    except AssertionError as e:
        assert "without the scratch lock" in str(e)
    else:
        raise AssertionError("`fd state` was started although the scratch lock was not set")
    assert len(_COMMANDS) == n


def test_zz_the_run_log_of_this_suite_is_in_the_scratch_folder_and_not_in_the_rig():
    """Runs last (by its name). Every fd command of the suite left its record in $FD_RUNS; the rig's runs/ got
    none of them. (Another process may write the rig's log while this suite runs: only records of THIS suite
    count, and they are told by the scratch folder they name.)"""
    if not _COMMANDS:
        raise Skip("no fd command was run (a selection of tests)")
    records = []
    for p in sorted(glob.glob(os.path.join(RUNS, "*", "events.jsonl"))):
        with open(p, encoding="utf-8") as f:
            records += [json.loads(ln) for ln in f if ln.strip()]
    assert [r["cmd"] for r in records] == _COMMANDS, ([r["cmd"] for r in records], _COMMANDS)
    token = os.path.basename(_TMP)
    grown, own = [], []
    for p, size in _rig_logs().items():
        was = _RIG_LOGS.get(p, 0)
        if size != was:
            grown.append(p)
        if size > was:
            with open(p, "rb") as f:
                f.seek(was)
                own += [ln[:200] for ln in f.read().decode("utf-8", "replace").splitlines() if token in ln]
    assert not own, f"{len(own)} record(s) of this suite were written into the rig's run log: {own[:3]}"
    if grown:
        print(f"note: the rig's run log changed while the suite ran, through another process (no record of this suite in it): {grown}")
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
