"""The hardware lock: while the lock file exists, this harness opens neither the real servo bus nor the real camera.

Why this exists (2026-11-13): the harness is also worked on by programs and by AI agents that are told "simulation
and tests only". Being told is not a guard: one `fd ready` typed in the wrong folder moves the real arm, and a copy
of the repository does not even know which tool is taped to the gripper. With the lock set, every way to the real
hardware answers with a refusal that says how to unlock; the simulated arm (--sim right after fd), saved photos and
the test suites are not affected.

    fd hw lock --reason "agents are editing the paint feature"     # writes the file
    fd hw status                                                   # locked? since when, why
    fd hw unlock                                                   # removes it

The file is ONE per user on this computer (default ~/.config/tazuna/hardware.lock, or $TAZUNA_HW_LOCK), not one per
copy of the repository: every copy that CONTAINS THIS MODULE looks at the same file. That is not every copy. A
working copy from before 2026-11-13 has a servo_bus.py and a camera.py that know nothing of the lock and open
/dev/ttyACM0 and the camera as they always did: on the day the lock was written, eleven such copies with an `fd`
of their own lay next to the live folder (five beside it, six one folder further down) while `fd hw status` said
"locked". Nothing in this module can stop them. copies_without_lock() finds the ones next to this folder, and
`fd hw lock` / `fd hw status` list them, so that whoever locks the arm knows what the lock does not cover; copies
elsewhere on the computer are not looked for. The same holds for harness code INSIDE the project folder that is
older than the lock: tests/refs/patches/motion-safety/snap1 and snap3 each hold an fd.py and a servo_bus.py that
opens /dev/ttyACM0 (`python tests/refs/patches/motion-safety/snap1/fd.py ready` moves the arm of a locked rig), and
backups/v1_code holds another. Those folders are listed too (found by their servo_bus.py; read and listed, never
run).

A lock that is set is kept: `lock` on a locked rig changes neither the reason nor the time (they are what whoever
set it relies on, and what every refusal quotes) and says that it was locked already.

The lock does not move anything and does not change the torque: a locked arm stays as it is (lock it in a pose in
which it may stay).

Checked in servo_bus.Bus.open() for real serial ports (a simulated or test bus on a pty is not a real port) and
in camera.py / fastcam.py before the camera device or v4l2-ctl is touched.
Tests: tests/test_hwlock.py.
"""

from __future__ import annotations

import json
import os
import time

DEFAULT = os.path.join(os.path.expanduser("~"), ".config", "tazuna", "hardware.lock")
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))      # the project folder (tools/motion/ -> up)
# Where a copy of the harness keeps the module that opens the serial port: in the folder itself (every copy until
# the reorganisation of 2026-11-13) or under tools/motion/ (since then).
BUS_FILES = ("servo_bus.py", os.path.join("tools", "motion", "servo_bus.py"))


class HardwareLocked(RuntimeError):
    """The real camera was asked for while the hardware lock is set."""


def path() -> str:
    return os.environ.get("TAZUNA_HW_LOCK") or DEFAULT


def info() -> dict | None:
    """What the lock file says ({"reason", "since", "by"}), or None when the hardware is not locked. A file
    that cannot be read still locks: the reason then says so, and "damaged" is True."""
    p = path()
    if not os.path.exists(p):
        return None
    damaged = False
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            raise ValueError("not an object")
    except (OSError, ValueError) as e:
        d, damaged = {"reason": f"(the lock file cannot be read: {e})"}, True
    out = {"reason": str(d.get("reason") or ""), "since": d.get("since"), "by": d.get("by"), "file": p}
    if damaged:
        out["damaged"] = True
    return out


def message(what: str) -> str | None:
    """The refusal text for `what` (e.g. "the servo bus on /dev/ttyACM0"), or None when nothing is locked."""
    d = info()
    if d is None:
        return None
    why = f": {d['reason']}" if d["reason"] else ""
    since = f" since {d['since']}" if d.get("since") else ""
    # (one complete command per pair of backticks: the pattern "fd --sim ..." stood in them, which is no command)
    return (f"the hardware is locked{since}{why}. {what} was not opened and nothing moved. The simulated arm "
            f"(--sim right after fd, before the command) and the tests still work; `fd hw unlock` (or deleting {d['file']}) "
            "unlocks")


def real_port(port: str) -> bool:
    """A serial port of real hardware (as against a pty of the test doubles)."""
    p = os.path.realpath(str(port))
    return p.startswith("/dev/ttyACM") or p.startswith("/dev/ttyUSB") or p.startswith("/dev/ttyTHS") or p.startswith("/dev/ttyS")


def lock(reason: str = "") -> dict:
    """Set the lock. Returns info() plus "already": was it locked before this call?

    A lock that is already set is KEPT as it is: the reason and the time in the file are the first ones (until
    2026-11-13 a second `fd hw lock` replaced both without saying so, and the record of since when and why the
    arm had been locked was gone). The reason asked for now is then not written. A lock file that cannot be read
    has no reason or time to keep: it is written anew (the file is replaced in one step, never absent)."""
    old = info()
    if old is not None and not old.get("damaged"):
        return {**old, "already": True}
    p = path()
    os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    d = {"reason": str(reason or ""), "since": time.strftime("%Y-%m-%dT%H:%M:%S"), "by": os.environ.get("USER", "")}
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False)
    os.replace(tmp, p)
    return {**info(), "already": False}


def unlock() -> bool:
    """True when a lock was removed."""
    p = path()
    if os.path.exists(p):
        os.remove(p)
        return True
    return False


def _bus_file(folder: str) -> str | None:
    for rel in BUS_FILES:
        p = os.path.join(folder, rel)
        if os.path.isfile(p):
            return p
    return None


def _knows_lock(bus_file: str) -> bool:
    """Does this servo bus module look at the lock? (It imports this module by name. A file that cannot be read
    is taken as one that does not.)"""
    try:
        with open(bus_file, encoding="utf-8", errors="replace") as f:
            return "hwlock" in f.read()
    except OSError:
        return False


def _inside_without_lock(root: str) -> list:
    """Folders BELOW the project folder `root` with harness code that does not look at the lock: every folder that
    holds a servo_bus.py without it, at any depth (snapshots kept as references, old code in backups/). For a copy
    in the layout by feature (its bus module under tools/motion/) the folder is the copy itself. The project
    folder's own bus module is not a copy; hidden folders and __pycache__ are not entered, links are not followed.
    Only names are listed and those files read: nothing is run or changed."""
    own = {os.path.realpath(os.path.join(root, rel)) for rel in BUS_FILES}
    by_feature = os.sep + os.path.join("tools", "motion")
    found = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__")
        bus = os.path.join(folder, "servo_bus.py")
        if "servo_bus.py" not in files or os.path.realpath(bus) in own or _knows_lock(bus):
            continue
        found.append(folder[:-len(by_feature)] if folder.endswith(by_feature) else folder)
    return found


def copies_without_lock(root: str | None = None) -> list:
    """Folders next to `root` (default: the project folder this module lies in), and folders inside it, that hold
    harness code whose servo bus module does NOT look at the lock: code started from there opens the real servo
    bus (and camera) while `fd hw status` says "locked". Sorted paths; [] when there is none.

    "Next to": the folders beside root, and the folders one level below those (working copies are kept that way:
    fable_direct_dev/ beside the live folder, jetharness_work/A/ one level down). A copy is a folder with a
    servo_bus.py at one of the two places a harness has it (BUS_FILES). root itself and links to it are left out;
    hidden folders are not entered. "Inside": _inside_without_lock (since round 2 of the review: the snapshots under
    tests/refs/ and the old code in backups/ were not listed, and each has a servo_bus.py for /dev/ttyACM0).
    Files are only listed and read, nothing is run or changed."""
    root = os.path.realpath(root or ROOT)
    parent = os.path.dirname(root)
    seen, found = {root}, []

    def subfolders(folder):
        try:
            with os.scandir(folder) as it:
                return sorted(e.path for e in it if not e.name.startswith(".") and e.is_dir())
        except OSError:
            return []

    def look(folder) -> bool:
        """True when `folder` is a copy of the harness (so that it is not searched for copies inside it)."""
        real = os.path.realpath(folder)
        if real in seen:
            return True
        seen.add(real)
        bus = _bus_file(folder)
        if bus is None:
            return False
        if not _knows_lock(bus):
            found.append(folder)
        return True

    for beside in subfolders(parent):
        if not look(beside):
            for below in subfolders(beside):
                look(below)
    return sorted(set(found) | set(_inside_without_lock(root)))
