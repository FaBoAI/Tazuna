"""Tests of the servo protocol layer: the REAL servo_bus.Bus against six simulated STS3215
servos on a pseudo-terminal (fakebus.py). NO HARDWARE: the only tty this file can open is a
pty, and every Bus gets a scratch lock file, never /tmp/fable_direct_bus.lock.

Run:  cd /home/jetson/lerobot/fable_direct && /home/jetson/miniconda/envs/lerobot/bin/python test_bus_fake.py
      -k TEXT     only the tests whose name contains TEXT
      --fuzz N    N operations in the random-fault test (default 250)
      --with-arm  also run robot.Arm on the real Bus through the fake (robot.py is owned by MOTION)

Two packet timeouts are used. Tests of logic (packets, guards, counters, torque sequences)
open the Bus with RELAXED_MS, so that a scheduling hiccup of this (shared, loaded) machine
cannot add a retry and blur an exact packet count. Tests of timing (late replies, host
stalls, costs, fuzz) use the default 12 ms of the real bus and assert only what must hold
however the scheduler behaves.
"""

from __future__ import annotations

import contextlib
import errno
import fcntl
import inspect
import json
import os
import random
import subprocess
import sys
import termios
import threading
import time
import traceback

import serial

# ---- hard guard: nothing in this process (or in the client processes it starts) can open a real tty
_serial_open = serial.Serial.open


def _guarded_open(self):
    if not str(self.port).startswith("/dev/pts/"):
        raise RuntimeError(f"TEST GUARD: refusing to open {self.port!r} (only pty ports are allowed in tests)")
    return _serial_open(self)


serial.Serial.open = _guarded_open
# The gripper in these tests is bare: they never read the tool file of the tree (tooling.py), so they pass
# whether or not a tool is attached to the real arm, and nobody has to `fd tool detach` to run them.
os.environ["FD_TOOL"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "testdata", "no_tool_attached.json")

import scservo_sdk as scs  # noqa: E402

import fakebus  # noqa: E402
import servo_bus as SB  # noqa: E402
from fakebus import (INST_PING, INST_READ, INST_SYNC_READ, INST_SYNC_WRITE, INST_WRITE, LIMITS, REST_RAW,  # noqa: E402
                     FakeBus, hexs, inject_host_stall, pkt)
from servo_bus import IDS, JOINTS, REG, Bus, BusError  # noqa: E402

RELAXED_MS = 150.0
FOREVER = 10 ** 9
EEPROM_READ = lambda sid: pkt(sid, [INST_READ, 0, 40])  # noqa: E731
RAM_READ = lambda sid: pkt(sid, [INST_READ, 40, 31])  # noqa: E731
FUZZ_OPS = 250
RESULTS = {}  # numbers worth reporting, printed at the end


# ------------------------------------------------------------------------------ helpers
@contextlib.contextmanager
def rig(relaxed: bool = True, opened: bool = True, retries: int = 4, **fake_kw):
    """A fresh fake bus and the real Bus on it (opened unless opened=False)."""
    fake = FakeBus(**fake_kw)
    bus = fake.bus(retries=retries, timeout_ms=RELAXED_MS if relaxed else None)
    try:
        if opened:
            bus.open(wait_s=2.0)
        yield fake, bus
        if fake.thread_error is not None:
            raise AssertionError(f"the fake bus thread died: {fake.thread_error!r}")
    finally:
        bus.close()
        fake.close()


class Skip(Exception):
    """Raised by a test that is not run (counted separately, never as a pass)."""


class raises:
    """with raises(BusError, "text", ...) as r: ...   -> r.msg is the message; every text must be in it."""

    def __init__(self, exc, *texts):
        self.exc, self.texts, self.msg, self.err = exc, texts, None, None

    def __enter__(self):
        return self

    def __exit__(self, et, ev, tb):
        if et is None:
            raise AssertionError(f"expected {self.exc.__name__}, nothing was raised")
        if not issubclass(et, self.exc):
            return False
        self.err, self.msg = ev, str(ev)
        for t in self.texts:
            assert t in self.msg, f"{t!r} not in the error message {self.msg!r}"
        return True


def wr(fake, since_t=0.0):
    """[(joint, register, value)] of the writes the servos executed after since_t."""
    return [(j, reg, v) for j, reg, v, _ in fake.written(since_t)]


def lock_is_free(path) -> bool:
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o666)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    except BlockingIOError:
        return False
    finally:
        os.close(fd)


def port_openable(port) -> bool:
    try:
        os.close(os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK))
        return True
    except OSError as e:
        assert e.errno == errno.EBUSY, e
        return False


def hold(fake, offsets: dict, velocity=1100, acceleration=40):
    """Put the simulated arm into a HOLDING state behind the Bus's back: torque on, caps set,
    Lock 1, goal = position + offset on every joint (gravity sag compensation / grasp squeeze),
    in static equilibrium (the standing error is modelled as sag)."""
    for j in JOINTS:
        sid = IDS[j]
        off = offsets.get(j, 0)
        fake.sag[sid] = float(-off)
        fake.poke(j, "Goal_Position", int(round(fake.pos[sid])) + off)
        fake.poke(j, "Acceleration", acceleration)
        fake.poke(j, "Goal_Velocity", velocity)
        fake.poke(j, "Lock", 1)
        fake.poke(j, "Torque_Enable", 1)


def goal_minus_pos(fake) -> dict:
    return {j: fake.reg(j, "Goal_Position") - int(round(fake.pos[IDS[j]])) for j in JOINTS}


def spawn_holder(port):
    """Another process that opens `port` and keeps it open until killed."""
    code = ("import os, sys, time\n"
            f"fd = os.open({port!r}, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)\n"
            "print('holding', flush=True)\n"
            "time.sleep(60)\n")
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True)
    assert p.stdout.readline().strip() == "holding"
    return p


# ============================================================================== A. open / close
def test_a00_this_test_process_cannot_open_anything_but_a_pty():
    with raises(RuntimeError, "TEST GUARD"):
        serial.Serial("/dev/ttyNOT_A_PTY_guard_check")
    with FakeBus() as fake:
        with raises(BusError, "cannot open /dev/ttyNOT_A_PTY_guard_check", "TEST GUARD"):
            Bus(port="/dev/ttyNOT_A_PTY_guard_check", lock_path=fake.lock_path).open(wait_s=0.5)
        assert fake.port.startswith("/dev/pts/") and os.path.dirname(fake.lock_path) != "/tmp"
        assert fake.lock_path != SB.LOCK_PATH and lock_is_free(fake.lock_path)


def test_a01_open_checks_the_six_models_and_claims_the_port():
    with rig() as (fake, bus):
        assert fake.sent() == [EEPROM_READ(sid) for sid in range(1, 7)], "open() = one EEPROM-block read per servo"
        assert bus.is_open and bus.exclusive
        assert bus.stats == {"retries": 0, "fallbacks": 0, "failures": 0, "requests": 6}
        assert bus.status == dict.fromkeys(JOINTS, 0) and bus.faults() == {}
        assert not port_openable(fake.port), "TIOCEXCL: nobody else can open the port while the Bus has it"
        assert os.readlink(f"/proc/self/fd/{bus._lock_fd}") == fake.lock_path, "flock on the scratch file"
        assert not lock_is_free(fake.lock_path)
        bus.close()
        assert not bus.is_open and not bus.exclusive
        assert port_openable(fake.port) and lock_is_free(fake.lock_path)
        bus.close()  # twice is fine
        bus.open(wait_s=1.0)  # and the same object can be opened again
        assert bus.positions() == REST_RAW


def test_a02_context_manager():
    with FakeBus() as fake:
        with fake.bus() as bus:
            assert bus.is_open and bus.positions() == REST_RAW
        assert not bus.is_open and port_openable(fake.port) and lock_is_free(fake.lock_path)


def test_a03_defaults_for_the_real_port_are_unchanged():
    b = Bus()  # constructed only, never opened
    assert (b.port, b.baud, b.retries) == ("/dev/ttyACM0", 1_000_000, 4) == (SB.PORT, SB.BAUD, 4)
    assert b.lock_path is None and SB.LOCK_PATH == "/tmp/fable_direct_bus.lock"
    assert b.timeout_ms == 12.0 == SB.PKT_TIMEOUT_MS and SB.QUARANTINE_S == 0.020
    assert list(inspect.signature(Bus.__init__).parameters) == ["self", "port", "baud", "retries", "lock_path",
                                                                "timeout_ms"]
    assert inspect.signature(Bus.open).parameters["wait_s"].default == 20.0
    # lock_path=None means "LOCK_PATH at open() time" (tools that redirect the module constant keep working)
    with FakeBus() as fake:
        old = SB.LOCK_PATH
        SB.LOCK_PATH = fake.lock_path + ".module_default"
        try:
            bus = Bus(port=fake.port).open(wait_s=1.0)
            try:
                assert os.readlink(f"/proc/self/fd/{bus._lock_fd}") == SB.LOCK_PATH
                assert bus.ph.ser.timeout == 0.012 and bus.exclusive
            finally:
                bus.close()
        finally:
            SB.LOCK_PATH = old


def test_a04_open_refuses_a_wrong_model_number():
    with rig(opened=False) as (fake, bus):
        fake.poke("wrist_flex", "Model", 1234)
        with raises(BusError, "wrist_flex", "1234", "777"):
            bus.open(wait_s=1.0)
        assert not bus.is_open and bus.ph is None
        assert port_openable(fake.port) and lock_is_free(fake.lock_path), "a refused open() leaves nothing claimed"
        assert all(p[4] == INST_READ for p in fake.sent()), "only reads were sent"
        fake.poke("wrist_flex", "Model", 777)
        bus.open(wait_s=1.0)
        assert bus.limits() == LIMITS


def test_a05_open_refuses_a_servo_that_does_not_answer():
    with rig(relaxed=False, opened=False) as (fake, bus):
        fake.rule(lambda r: r["id"] == 3, count=FOREVER, dead=True)
        t0 = time.perf_counter()
        with raises(BusError, "elbow_flex", "servo id 3", "does not answer", "no reply") as r:
            bus.open(wait_s=1.0)
        dt = time.perf_counter() - t0
        RESULTS["open() refusing a dead servo (incl. the port-user scan)"] = f"BusError after {dt * 1e3:.0f} ms"
        assert not bus.is_open and port_openable(fake.port) and lock_is_free(fake.lock_path)
        assert bus.stats["retries"] == 4 and bus.stats["failures"] == 1 and bus.stats["requests"] == 6, bus.stats
        assert fake.sent() == [EEPROM_READ(1), EEPROM_READ(2)] + [EEPROM_READ(3)] * 4
        assert 4 * 0.030 < dt < 3.0, dt


def test_a06_open_refuses_a_port_another_process_has_open():
    with rig(opened=False) as (fake, bus):
        holder = spawn_holder(fake.port)
        try:
            with raises(BusError, "in use by pid(s)", str(holder.pid), "refusing to touch it"):
                bus.open(wait_s=1.0)
            assert fake.sent() == [] and fake.replies == [], "nothing was sent to a port in use"
            assert not bus.is_open and lock_is_free(fake.lock_path)
        finally:
            holder.kill()
            holder.wait()
        bus.open(wait_s=1.0)
        assert bus.positions() == REST_RAW


def test_a07_open_waits_for_the_flock_and_gives_up_with_a_reason():
    with rig(opened=False) as (fake, bus):
        fd = os.open(fake.lock_path, os.O_CREAT | os.O_RDWR, 0o666)  # "another fable_direct process"
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            t0 = time.perf_counter()
            with raises(BusError, "holds the bus lock", fake.lock_path):
                bus.open(wait_s=0.3)
            assert 0.25 < time.perf_counter() - t0 < 1.5
            assert fake.sent() == [] and not bus.is_open and bus._lock_fd is None
            threading.Timer(0.25, lambda: fcntl.flock(fd, fcntl.LOCK_UN)).start()  # released while we wait
            t0 = time.perf_counter()
            bus.open(wait_s=5.0)
            assert 0.15 < time.perf_counter() - t0 < 5.0
            assert bus.positions() == REST_RAW
        finally:
            os.close(fd)
        with FakeBus() as fake2:  # the holder is a Bus: the waiting side is told which process it is
            with raises(BusError, f"another fable_direct process (pid {os.getpid()}) holds the bus lock"):
                Bus(port=fake2.port, lock_path=fake.lock_path).open(wait_s=0.0)
            assert fake2.sent() == [] and bus.positions() == REST_RAW


def test_a08_open_failures_are_bus_errors():
    with FakeBus() as fake:
        with raises(BusError, "cannot open /dev/pts/999999"):
            Bus(port="/dev/pts/999999", lock_path=fake.lock_path).open(wait_s=1.0)
        with raises(BusError, "unsupported baud rate 123456"):
            fake.bus(baud=123456).open(wait_s=1.0)
        with raises(BusError, "cannot open the bus lock file"):
            Bus(port=fake.port, lock_path="/nonexistent_dir_for_test/bus.lock").open(wait_s=1.0)
        assert fake.sent() == [] and lock_is_free(fake.lock_path) and port_openable(fake.port)
        with raises(BusError, "retries"):
            fake.bus(retries=0)
        with raises(BusError, "timeout_ms"):
            fake.bus(timeout_ms=0)
        bus = fake.bus()
        real_harden = Bus._harden

        def broken_harden(self):  # the port opens but cannot be configured
            raise termios.error(25, "Inappropriate ioctl for device")

        Bus._harden = broken_harden
        try:
            with raises(BusError, "cannot configure", fake.port, "Inappropriate ioctl"):
                bus.open(wait_s=1.0)
        finally:
            Bus._harden = real_harden
        assert not bus.is_open and fake.sent() == [] and lock_is_free(fake.lock_path) and port_openable(fake.port)
        bus.open(wait_s=1.0)
        bus.close()


def test_a09_open_twice_and_use_after_close():
    with rig() as (fake, bus):
        with raises(BusError, "already open"):
            bus.open(wait_s=0.2)
        assert bus.positions() == REST_RAW, "the first session is untouched by the refused second open()"
        bus.close()
        n = len(fake.requests)
        calls = [lambda: bus.read("gripper", "Present_Position"), lambda: bus.write("gripper", "Goal_Position", 2200),
                 lambda: bus.sync_read("Present_Position"), lambda: bus.sync_write("Goal_Position", {"gripper": 2200}),
                 bus.positions, bus.feedback, bus.snapshot, bus.limits, bus.torque_on, bus.torque_off,
                 lambda: bus.set_p_gain("shoulder_pan", 32), lambda: bus.read_block("gripper", 40, 31)]
        for fn in calls:
            with raises(BusError, "not open"):
                fn()
        fake.drain()
        assert len(fake.requests) == n


def test_a11_a_second_bus_on_the_same_port_is_refused_by_the_kernel():
    with rig() as (fake, bus):  # same process, another lock file: only the exclusive open stands in the way
        other = Bus(port=fake.port, lock_path=fake.lock_path + ".other")
        with raises(BusError, "cannot open", fake.port):
            other.open(wait_s=0.5)
        assert not other.is_open and lock_is_free(fake.lock_path + ".other")
        assert bus.positions() == REST_RAW and bus.exclusive


def test_a10_port_users_sees_other_holders_and_fails_closed():
    with FakeBus() as fake:
        assert SB.port_users(fake.port) == [], "this process (the fake keeps the slave side open) is not counted"
        holder = spawn_holder(fake.port)
        old_path, old_listdir = os.environ["PATH"], os.listdir
        try:
            assert SB.port_users(fake.port) == [holder.pid]
            os.environ["PATH"] = "/nonexistent"  # no fuser: /proc alone still finds the holder
            assert SB.port_users(fake.port) == [holder.pid]

            def no_proc(path="."):
                if str(path).startswith("/proc"):
                    raise PermissionError("test: /proc unreadable")
                return old_listdir(path)

            os.listdir = no_proc  # neither source usable: the answer must not be "free"
            with raises(BusError, "cannot determine whether the serial port is in use"):
                SB.port_users(fake.port)
            with raises(BusError, "cannot determine"):
                fake.bus().open(wait_s=0.5)
            os.environ["PATH"] = old_path  # fuser alone
            assert SB.port_users(fake.port) == [holder.pid]
        finally:
            os.listdir = old_listdir
            os.environ["PATH"] = old_path
            holder.kill()
            holder.wait()
        assert lock_is_free(fake.lock_path)


# ============================================================================== B. limits()
def test_b01_limits_are_the_servo_stored_ones():
    with rig() as (fake, bus):
        m = fake.mark()
        assert bus.limits() == LIMITS
        assert fake.sent(m) == [EEPROM_READ(sid) for sid in range(1, 7)], "six 40-byte block reads, nothing else"


def test_b02_limits_refuses_a_servo_that_is_not_in_position_mode():
    for mode in (1, 2, 3):
        with rig() as (fake, bus):
            fake.poke("elbow_flex", "Operating_Mode", mode)
            with raises(BusError, "elbow_flex", f"Operating_Mode is {mode}", "position mode"):
                bus.limits()


def test_b03_limits_refuses_implausible_limits():
    for lo, hi in ((3000, 1000), (2000, 2100), (0, 0), (2047, 2047), (100, 5000), (3900, 4095)):
        with rig() as (fake, bus):
            fake.poke("shoulder_lift", "Min_Position_Limit", lo)
            fake.poke("shoulder_lift", "Max_Position_Limit", hi)
            with raises(BusError, "shoulder_lift", "implausible position limits", str(lo), str(hi)):
                bus.limits()
    with rig() as (fake, bus):  # the narrowest range that is still accepted
        fake.poke("shoulder_lift", "Min_Position_Limit", 1000)
        fake.poke("shoulder_lift", "Max_Position_Limit", 1300)
        assert bus.limits()["shoulder_lift"] == (1000, 1300)


# ============================================================================== C. read / write / sync
def test_c01_register_table_matches_the_independent_one():
    for name, entry in REG.items():
        assert fakebus.TABLE[name] == entry, (name, entry, fakebus.TABLE[name])
    assert JOINTS == fakebus.JOINTS and IDS == fakebus.IDS == {j: i + 1 for i, j in enumerate(JOINTS)}
    path = "/home/jetson/src/lerobot-0.6.0-cu130/src/lerobot/motors/feetech/tables.py"
    if os.path.exists(path):  # read-only cross-check against LeRobot 0.6.0 (skipped if the source is not there)
        import importlib.util
        spec = importlib.util.spec_from_file_location("_lerobot_feetech_tables", path)
        t = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(t)
        for name, (addr, size, sign) in REG.items():
            key = {"Model": "Model_Number", "Firmware_Major": "Firmware_Major_Version",
                   "Firmware_Minor": "Firmware_Minor_Version"}.get(name, name)
            assert tuple(t.STS_SMS_SERIES_CONTROL_TABLE[key]) == (addr, size), name
            if name != "Present_Current":  # LeRobot decodes no sign there; the vendor SDK uses bit 15
                assert t.STS_SMS_SERIES_ENCODINGS_TABLE.get(key) == sign, name
        assert t.MODEL_NUMBER_TABLE["sts3215"] == 777
        RESULTS["REG vs LeRobot 0.6.0 table"] = f"{len(REG)} entries identical (address, size, sign bit)"
    else:
        RESULTS["REG vs LeRobot 0.6.0 table"] = "not checked (source not found)"


def test_c02_sign_magnitude_codec():
    for bit in (10, 11, 15):
        lim = (1 << bit) - 1
        for v in list(range(-lim, lim + 1, 1 if bit < 15 else 13)) + [-lim, -1, 0, 1, lim]:
            raw = SB._encode(v, bit)
            assert raw == fakebus.enc(v, bit) and 0 <= raw < (1 << (bit + 1))
            assert SB._decode(raw, bit) == v == fakebus.dec(raw, bit)
        for v in (lim + 1, -(lim + 1)):
            with raises(ValueError):
                SB._encode(v, bit)
    assert SB._encode(-5, 15) == 0x8005 and SB._decode(0x8005, 15) == -5
    assert SB._decode(0x400 | 123, 10) == -123 and SB._decode(0x800 | 854, 11) == -854
    assert SB._encode(4095, None) == 4095 and SB._decode(65535, None) == 65535
    with raises(ValueError):
        SB._encode(-1, None)


def test_c03_read_every_register_of_every_servo():
    with rig() as (fake, bus):
        fake.frozen = True  # Torque_Enable is poked too: nothing may move
        truth = {}
        for j in JOINTS:
            sid = IDS[j]
            for k, (name, (_, size, sign)) in enumerate(REG.items()):
                if name == "Model":
                    v = 777
                elif sign is not None:
                    v = (-1) ** (k + sid) * (100 * sid + 3 * k)  # both signs, well inside every magnitude
                else:
                    v = (17 * sid + 5 * k) % 200 + 1 if size == 1 else 1000 + 100 * sid + k
                fake.poke(j, name, v)
                truth[j, name] = v
        m = fake.mark()
        for (j, name), v in truth.items():
            assert bus.read(j, name) == v, (j, name, bus.read(j, name), v)
        for p in fake.sent(m):
            assert p in (EEPROM_READ(p[2]), RAM_READ(p[2])), hexs(p)
        assert fake.sent(m, sid=3)[0] == bytes.fromhex("FFFF0304020028CE"), "EEPROM block request, byte-exact"
        ram3 = [p for p in fake.sent(m, sid=3) if p[5] == 40]
        assert ram3[0] == bytes.fromhex("FFFF030402281FAF"), "RAM block request, byte-exact"
        assert bus.stats["retries"] == 0


def test_c04_write_packets_are_byte_exact_and_acknowledged():
    with rig() as (fake, bus):
        cases = [("gripper", "Goal_Velocity", 1100, "FFFF0605032E4C0473"),
                 ("gripper", "Acceleration", 40, "FFFF0604032928A1"),
                 ("gripper", "Torque_Enable", 1, "FFFF0604032801C9"),
                 ("shoulder_pan", "Goal_Position", 2047, "FFFF0105032AFF07C6"),
                 ("elbow_flex", "Goal_Position", 0x0A0B, "FFFF0305032A0B0AB5"),
                 ("wrist_roll", "Torque_Limit", 1000, "FFFF05050330E803D7"),
                 ("shoulder_lift", "Lock", 1, "FFFF0204033701BE")]
        for j, reg, v, hexpkt in cases:
            m, n_rep = fake.mark(), len(fake.replies)
            bus.write(j, reg, v)
            sent = fake.sent(m)
            assert sent == [bytes.fromhex(hexpkt)], (reg, [hexs(p) for p in sent])
            addr, size, _ = REG[reg]
            assert sent[0] == pkt(IDS[j], [INST_WRITE, addr] + list(v.to_bytes(size, "little"))), "low byte first"
            assert fake.reg(j, reg) == v
            assert fake.replies[n_rep:] == [(fake.replies[-1][0], pkt(IDS[j], [0]))], "one 6-byte status packet"
        assert bus.stats["retries"] == 0 and fake.ee_writes == []


def test_c05_sync_read_packets_and_values():
    with rig() as (fake, bus):
        for j in JOINTS:
            sid = IDS[j]
            fake.poke(j, "Present_Load", (-1) ** sid * (40 + sid))
            fake.poke(j, "Present_Current", 20 + sid)
            fake.poke(j, "Present_Velocity", (-1) ** sid * 7 * sid)
            fake.poke(j, "Present_Voltage", 47 + sid)
            fake.poke(j, "Present_Temperature", 30 + sid)
            fake.poke(j, "Status", 0)
            fake.poke(j, "Moving", sid % 2)
            fake.poke(j, "Goal_Position", REST_RAW[j] - 10 * sid)
        m = fake.mark()
        assert bus.sync_read("Present_Position") == REST_RAW == bus.positions()
        assert fake.sent(m) == [bytes.fromhex("FFFFFE0A82380201020304050626")] * 2, "position sync read, byte-exact"
        m = fake.mark()
        assert bus.sync_read("Present_Load") == {j: (-1) ** IDS[j] * (40 + IDS[j]) for j in JOINTS}
        assert bus.sync_read("Present_Current") == {j: 20 + IDS[j] for j in JOINTS}
        assert bus.sync_read("Present_Velocity") == {j: (-1) ** IDS[j] * 7 * IDS[j] for j in JOINTS}
        assert bus.sync_read("Present_Temperature") == {j: 30 + IDS[j] for j in JOINTS}
        assert bus.sync_read("Moving") == {j: IDS[j] % 2 for j in JOINTS}
        assert fake.sent(m) == [pkt(0xFE, [INST_SYNC_READ, 56, 15, 1, 2, 3, 4, 5, 6])] * 5, "the 15-byte feedback block"
        m = fake.mark()  # a subset keeps the order that was asked for
        assert bus.sync_read("Present_Position", ("gripper", "elbow_flex")) == {"gripper": 2184, "elbow_flex": 3025}
        assert list(bus.sync_read("Present_Position", ["wrist_roll", "shoulder_pan"])) == ["wrist_roll", "shoulder_pan"]
        assert fake.sent(m) == [pkt(0xFE, [INST_SYNC_READ, 56, 2, 6, 3]), pkt(0xFE, [INST_SYNC_READ, 56, 2, 5, 1])]
        m = fake.mark()  # a register outside the feedback block: one RAM-block read per servo, no sync read
        assert bus.sync_read("Goal_Position") == {j: REST_RAW[j] - 10 * IDS[j] for j in JOINTS}
        assert fake.sent(m) == [RAM_READ(sid) for sid in range(1, 7)]
        assert bus.sync_read("Present_Position", (j for j in JOINTS)) == REST_RAW, "any iterable of joints"
        assert bus.feedback(j for j in JOINTS[:2]).keys() == set(JOINTS[:2])
        assert bus.sync_read("Present_Position", ()) == {} and bus.stats["retries"] == 0


def test_c06_sync_write_packet_is_byte_exact_and_not_acknowledged():
    with rig() as (fake, bus):
        vals = {"shoulder_pan": 2047, "shoulder_lift": 955, "elbow_flex": 3025, "wrist_flex": 0x0A0B, "wrist_roll": 4095}
        m, n_rep = fake.mark(), len(fake.replies)
        bus.sync_write("Goal_Position", vals)
        fake.drain()
        body = [INST_SYNC_WRITE, 42, 2]
        for j, v in vals.items():
            body += [IDS[j], v & 0xFF, v >> 8]
        assert fake.sent(m) == [pkt(0xFE, body)], hexs(fake.sent(m)[0])
        assert fake.sent(m)[0][:10] == bytes.fromhex("FFFFFE13832A0201FF07")
        assert {j: fake.reg(j, "Goal_Position") for j in vals} == vals
        assert fake.reg("gripper", "Goal_Position") == REST_RAW["gripper"], "a servo that is not addressed is untouched"
        assert len(fake.replies) == n_rep, "a broadcast gets no status packet"
        m = fake.mark()
        bus.sync_write("Acceleration", {"gripper": 40, "shoulder_pan": 20})  # one-byte register
        fake.drain()
        assert fake.sent(m) == [pkt(0xFE, [INST_SYNC_WRITE, 41, 1, 6, 40, 1, 20])]
        assert fake.reg("gripper", "Acceleration") == 40 and fake.reg("shoulder_pan", "Acceleration") == 20
        assert bus.positions() == REST_RAW, "the next request right after a broadcast works (torque is off: nothing moved)"


def test_c07_byte_order_does_not_depend_on_the_sdk_global():
    with rig() as (fake, bus):
        try:
            scs.PacketHandler(1)  # what a second, big-endian packet handler elsewhere in the process would do
            assert scs.SCS_GETEND() == 1
            bus.write("shoulder_pan", "Goal_Position", 2047)
            assert fake.reg("shoulder_pan", "Goal_Position") == 2047
            bus.sync_write("Goal_Position", {"shoulder_lift": 1000})
            fake.drain()
            assert fake.reg("shoulder_lift", "Goal_Position") == 1000
            assert bus.read("shoulder_pan", "Present_Position") == 1999 and bus.positions() == REST_RAW
            assert bus.limits() == LIMITS and bus.snapshot()["shoulder_pan"]["goal"] == 2047
        finally:
            scs.PacketHandler(0)


def test_c08_feedback_and_snapshot():
    with rig() as (fake, bus):
        truth = {}
        for j in JOINTS:
            sid = IDS[j]
            truth[j] = {"pos": REST_RAW[j], "load": (-1) ** sid * 11 * sid, "current": 3 * sid, "volt": 4.5 + sid / 10,
                        "temp": 30 + sid, "status": 0, "torque": sid % 2, "goal": REST_RAW[j] - sid,
                        "acc": 10 * sid, "vel_cap": 100 * sid, "lock": 1}
            for key, reg in (("load", "Present_Load"), ("current", "Present_Current"), ("temp", "Present_Temperature"),
                             ("goal", "Goal_Position"), ("acc", "Acceleration"), ("vel_cap", "Goal_Velocity")):
                fake.poke(j, reg, truth[j][key])
            fake.poke(j, "Present_Voltage", 45 + sid)
        fake.frozen = True  # nothing may move although some joints get torque below
        for j in JOINTS:
            fake.poke(j, "Torque_Enable", IDS[j] % 2)
        m = fake.mark()
        fb = bus.feedback()
        assert fake.sent(m) == [pkt(0xFE, [INST_SYNC_READ, 56, 15, 1, 2, 3, 4, 5, 6])], "ONE sync read"
        keys = ("pos", "load", "current", "volt", "temp", "status")
        assert fb == {j: {k: truth[j][k] for k in keys} for j in JOINTS}, fb
        assert bus.feedback(["gripper"]) == {"gripper": {k: truth["gripper"][k] for k in keys}}
        m = fake.mark()
        snap = bus.snapshot()
        assert fake.sent(m) == [RAM_READ(sid) for sid in range(1, 7)], "one RAM block per servo"
        assert snap == truth, snap
        assert list(snap) == list(JOINTS)


def test_c09_streaming_goals_like_the_motion_loop():
    """50 Hz for one second: one sync_write of goals, followed at once by one feedback sync read."""
    with rig() as (fake, bus):
        bus.torque_on(velocity=1100, acceleration=40)
        m, base = fake.mark(), dict(bus.stats)
        seen, goal = [], dict(REST_RAW)
        t_next = time.perf_counter()
        for k in range(1, 51):
            goal["shoulder_pan"] = REST_RAW["shoulder_pan"] + 4 * k  # 200 ticks/s, well under the 1100 cap
            goal["gripper"] = REST_RAW["gripper"] + 6 * k
            bus.sync_write("Goal_Position", goal)
            fb = bus.feedback()
            seen.append(fb["shoulder_pan"]["pos"])
            assert fb["elbow_flex"]["pos"] == REST_RAW["elbow_flex"] and fb["gripper"]["status"] == 0
            t_next += 0.02
            time.sleep(max(0.0, t_next - time.perf_counter()))
        time.sleep(0.15)
        assert bus.positions() == {**REST_RAW, "shoulder_pan": REST_RAW["shoulder_pan"] + 200,
                                   "gripper": REST_RAW["gripper"] + 300}
        assert seen == sorted(seen) and seen[-1] > REST_RAW["shoulder_pan"] + 150, "the servo follows the stream"
        sent = fake.sent(m)
        assert [p[4] for p in sent] == [INST_SYNC_WRITE, INST_SYNC_READ] * 50 + [INST_SYNC_READ]
        assert bus.stats == {"retries": 0, "fallbacks": 0, "failures": 0, "requests": base["requests"] + 101}
        assert {j: fake.reg(j, "Goal_Position") for j in JOINTS} == goal


# ============================================================================== D. write guards
def test_d01_only_the_six_ram_registers_can_be_written():
    assert SB.WRITABLE == {"Torque_Enable", "Acceleration", "Goal_Position", "Goal_Velocity", "Torque_Limit", "Lock"}
    assert SB.WRITABLE == set(SB.RANGE) and all(REG[r][0] >= 40 for r in SB.WRITABLE)
    with rig() as (fake, bus):
        calib = fake.calib_bytes()
        for reg in REG:
            m = fake.mark()
            if reg in SB.WRITABLE:
                bus.write("gripper", reg, 1)
                bus.sync_write(reg, {"gripper": 1})
                fake.drain()
                assert len(fake.sent(m)) == 2
            else:
                with raises(BusError, "refusing to write", reg, "only RAM registers"):
                    bus.write("gripper", reg, 1)
                with raises(BusError, "refusing to write", reg):
                    bus.sync_write(reg, {"gripper": 1})
                fake.drain()
                assert fake.sent(m) == [], f"{reg}: a refused write must not reach the wire"
        assert fake.ee_writes == [] and fake.calib_bytes() == calib, "nothing below address 40 was written"
        assert all(addr >= 40 for _, _, addr, _, _ in fake.writes)


def test_d02_torque_enable_128_is_refused():
    with rig() as (fake, bus):
        m = fake.mark()
        with raises(BusError, "Torque_Enable=128", "outside [0, 1]"):
            bus.write("gripper", "Torque_Enable", 128)
        with raises(BusError, "Torque_Enable=128"):
            bus.sync_write("Torque_Enable", {j: 128 for j in JOINTS})
        with raises(BusError, "Torque_Enable=2"):
            bus.write("gripper", "Torque_Enable", 2)
        fake.drain()
        assert fake.sent(m) == [] and fake.rezero_events == []
        # what the guard prevents (straight through the SDK, on the simulated servo only):
        offset = fake.stored("gripper", "Homing_Offset")
        fake.poke("gripper", "Lock", 0)
        bus.pk.write1ByteTxRx(bus.ph, 6, 40, 128)
        assert len(fake.rezero_events) == 1 and fake.stored("gripper", "Homing_Offset") != offset
        assert bus.read("gripper", "Present_Position") == 2048


def test_d03_out_of_range_values_are_refused_before_anything_is_sent():
    with rig() as (fake, bus):
        m = fake.mark()
        bad = [("Goal_Position", -5), ("Goal_Position", 4096), ("Goal_Position", 30000), ("Acceleration", 255),
               ("Acceleration", 300), ("Acceleration", -1), ("Goal_Velocity", 4001), ("Goal_Velocity", -400),
               ("Torque_Limit", 1001), ("Torque_Limit", 5000), ("Torque_Enable", -1), ("Lock", 2)]
        for reg, v in bad:
            with raises(BusError, f"refusing {reg}={v}", "outside"):
                bus.write("gripper", reg, v)
            with raises(BusError, f"refusing {reg}={v}"):
                bus.sync_write(reg, {"gripper": v})
        fake.drain()
        assert fake.sent(m) == []
        assert bus.ph.is_using is False and bus.read("gripper", "Present_Position") == 2184, "the port is not left busy"
        for reg, (lo, hi) in SB.RANGE.items():  # both ends of every range are accepted
            bus.write("wrist_roll", reg, lo)
            bus.write("wrist_roll", reg, hi)
            assert fake.reg("wrist_roll", reg) == hi


def test_d04_lock_can_be_set_but_never_cleared():
    with rig() as (fake, bus):
        fake.poke("elbow_flex", "Lock", 0)
        m = fake.mark()
        with raises(BusError, "refusing Lock=0", "outside [1, 1]"):
            bus.write("elbow_flex", "Lock", 0)
        with raises(BusError, "refusing Lock=0"):
            bus.sync_write("Lock", {"elbow_flex": 0})
        fake.drain()
        assert fake.sent(m) == [] and fake.reg("elbow_flex", "Lock") == 0
        bus.write("elbow_flex", "Lock", 1)
        assert fake.reg("elbow_flex", "Lock") == 1


def test_d05_values_that_are_not_numbers_are_refused():
    with rig() as (fake, bus):
        m = fake.mark()
        for v in (float("nan"), float("inf"), -float("inf"), None, "2047", b"1", [2047], {"a": 1}):
            with raises(BusError, "refusing Goal_Position=", "not a number"):
                bus.write("gripper", "Goal_Position", v)
            with raises(BusError, "not a number"):
                bus.sync_write("Goal_Position", {"gripper": v})
        fake.drain()
        assert fake.sent(m) == []
        import numpy as np
        bus.write("gripper", "Goal_Position", np.int64(2200))  # numpy scalars and whole floats are fine
        bus.write("gripper", "Goal_Velocity", 400.0)
        bus.sync_write("Goal_Position", {"gripper": np.float64(2210.0)})
        fake.drain()
        assert fake.reg("gripper", "Goal_Position") == 2210 and fake.reg("gripper", "Goal_Velocity") == 400


def test_d06_unknown_joints_and_registers_are_bus_errors():
    with rig() as (fake, bus):
        m = fake.mark()
        for fn in (lambda: bus.read("elbow", "Present_Position"), lambda: bus.write("elbow", "Goal_Position", 2000),
                   lambda: bus.sync_read("Present_Position", ["gripper", "thumb"]), lambda: bus.read_block(7, 0, 40),
                   lambda: bus.sync_write("Goal_Position", {"thumb": 2000}), lambda: bus.feedback(["thumb"]),
                   lambda: bus.torque_on(["thumb"]), lambda: bus.torque_off(["thumb"]),
                   lambda: bus.set_p_gain("thumb", 32), lambda: bus.torque_on(["gripper", "gripper"])):
            with raises(BusError, "joint"):
                fn()
        for fn in (lambda: bus.read("gripper", "Present_Positon"), lambda: bus.sync_read("Goal_Speed"),
                   lambda: bus.read("gripper", None)):
            with raises(BusError, "unknown register"):
                fn()
        with raises(BusError, "refusing to write"):
            bus.write("gripper", "Goal_Speed", 1)
        with raises(BusError, "outside the control table"):
            bus.read_block("gripper", 40, 0)
        fake.drain()
        assert fake.sent(m) == []


def test_d07_sync_write_checks_every_value_before_sending_anything():
    with rig() as (fake, bus):
        m = fake.mark()
        with raises(BusError, "refusing Goal_Position=5000"):
            bus.sync_write("Goal_Position", {"shoulder_pan": 2000, "shoulder_lift": 1000, "gripper": 5000})
        with raises(BusError, "no joints"):
            bus.sync_write("Goal_Position", {})
        fake.drain()
        assert fake.sent(m) == []
        assert {j: fake.reg(j, "Goal_Position") for j in JOINTS} == REST_RAW


# ------------------------------------------------------------------------------ truth helpers
def distinct(fake):
    """Motion model off and distinct values in everything the Bus reports, so that a reply
    attributed to the wrong register or the wrong servo cannot go unnoticed."""
    fake.frozen = True
    for j in JOINTS:
        sid = IDS[j]
        fake.poke(j, "Present_Load", (-1) ** sid * (40 + 7 * sid))
        fake.poke(j, "Present_Current", 10 + sid)
        fake.poke(j, "Present_Voltage", 44 + sid)
        fake.poke(j, "Present_Temperature", 30 + sid)
        fake.poke(j, "Goal_Position", REST_RAW[j] - 17 - sid)
        fake.poke(j, "Acceleration", 20 + sid)
        fake.poke(j, "Goal_Velocity", 400 + 10 * sid)
        fake.poke(j, "Torque_Limit", 900 + sid)
        fake.poke(j, "Torque_Enable", sid % 2)
        fake.poke(j, "P_Coefficient", 16 + sid)


def truth_snapshot(fake) -> dict:
    g = fake.reg
    return {j: {"pos": g(j, "Present_Position"), "load": g(j, "Present_Load"), "current": g(j, "Present_Current"),
                "volt": g(j, "Present_Voltage") / 10.0, "temp": g(j, "Present_Temperature"),
                "torque": g(j, "Torque_Enable"), "goal": g(j, "Goal_Position"), "status": g(j, "Status"),
                "acc": g(j, "Acceleration"), "vel_cap": g(j, "Goal_Velocity"), "lock": g(j, "Lock")} for j in JOINTS}


def check_everything(fake, bus):
    """Every value the Bus can report equals the simulated servos' registers (needs distinct())."""
    t = truth_snapshot(fake)
    assert bus.limits() == LIMITS
    snap = bus.snapshot()
    assert snap == t, {j: (snap[j], t[j]) for j in JOINTS if snap[j] != t[j]}
    assert bus.positions() == {j: t[j]["pos"] for j in JOINTS}
    assert bus.feedback() == {j: {k: t[j][k] for k in ("pos", "load", "current", "volt", "temp", "status")}
                              for j in JOINTS}
    for j in JOINTS:
        assert bus.read(j, "P_Coefficient") == fake.reg(j, "P_Coefficient")
        assert bus.read(j, "Goal_Position") == t[j]["goal"]
        assert bus.read(j, "Torque_Limit") == fake.reg(j, "Torque_Limit")


# ============================================================================== E. late and stale replies (finding SP-0)
def test_e01_a_late_reply_never_corrupts_limits_or_what_follows():
    """The SP-0 scenario: the elbow's limit read is answered late while Arm.open() reads the limits."""
    rows = []
    for late_ms in (5, 11, 13, 20, 31.5, 32.5, 33.5, 36, 52, 53.5, 60, 90):
        with rig(relaxed=False) as (fake, bus):
            distinct(fake)
            fake.rule(lambda r: r["id"] == 3 and r["inst"] == INST_READ and r["addr"] == 0, delay=late_ms / 1000.0)
            t0 = time.perf_counter()
            lim = bus.limits()
            dt = (time.perf_counter() - t0) * 1e3
            assert lim == LIMITS, (late_ms, lim)
            check_everything(fake, bus)  # whatever is still in flight must not leak into what follows
            time.sleep(0.11)
            check_everything(fake, bus)
            if late_ms >= 20:  # (13 ms is within the scheduling jitter of "later than the 12 ms timeout")
                assert bus.stats["retries"] >= 1, "a reply later than the timeout is a counted retry"
            assert bus.stats["failures"] == 0
            rows.append(f"{late_ms:g} ms late: {dt:.0f} ms, {bus.stats['retries']} retr.")
    RESULTS["limits() with ONE late elbow reply (always correct)"] = "; ".join(rows)


def test_e02_a_late_write_ack_is_never_taken_for_data():
    """The old code returned the ack's checksum byte (247) as Torque_Enable after a stalled write."""
    for late_ms in (13, 20, 32.5, 36, 53, 60, 75):
        with rig(relaxed=False) as (fake, bus):
            distinct(fake)
            fake.rule(lambda r: r["id"] == 6 and r["inst"] == INST_WRITE, delay=late_ms / 1000.0)
            bus.write("gripper", "Goal_Position", 2500)
            assert bus.read("gripper", "Torque_Enable") == fake.reg("gripper", "Torque_Enable") == 0
            assert bus.read("gripper", "Present_Position") == 2184
            assert bus.read("gripper", "Goal_Position") == 2500 == fake.reg("gripper", "Goal_Position")
            assert bus.read("gripper", "Model") == 777
            assert all(g == 2500 for _, sid, g in fake.goal_log if sid == 6)
            check_everything(fake, bus)
            assert bus.stats["retries"] >= (late_ms >= 20) and bus.stats["failures"] == 0


def test_e03_late_sync_replies_are_never_attributed_to_another_servo():
    """Every servo stamps Present_Position with the request counter. The six replies of one
    position sync read arrive late (spread out, or all at once). A value returned by
    positions() must be the answer of the servo it is attributed to, and to a request of that
    very call - with the one exception the protocol cannot rule out (see test_e09): the late
    burst as a whole landing exactly between a later request and its answers, taken once."""
    stale_cases = []
    for burst in (False, True):
        for late_ms in (13, 25, 31.5, 32, 32.25, 32.5, 32.75, 33, 33.5, 34, 34.5, 36, 40, 52, 60):
            with rig(relaxed=False, stamp=True) as (fake, bus):
                def delay(late=late_ms / 1000.0, state={"slot": 0}):
                    k = state["slot"]
                    state["slot"] += 1
                    return late - (k * fake.reply_delay if burst else 0.0)  # burst: all six arrive together
                fake.rule(lambda r: r["inst"] == INST_SYNC_READ, count=6, delay=delay)
                late_request = fake.seq + 1  # the request whose six answers are late
                stale_calls = []
                for k in range(6):
                    s0 = fake.seq
                    pos = bus.positions()
                    s1 = fake.seq
                    assert list(pos) == list(JOINTS)
                    if not all(s0 < v <= s1 for v in pos.values()):
                        stale_calls.append(k)
                        assert set(pos.values()) == {late_request}, \
                            f"late {late_ms} ms, call {k}: {pos} is neither fresh ({s0 + 1}..{s1}) nor the late burst"
                    if k == 3:
                        time.sleep(0.06)  # by now the late burst has arrived, whatever the scheduler did
                assert len(stale_calls) <= 1 and not set(stale_calls) & {0, 4, 5}, (late_ms, stale_calls)
                assert bus.stats["retries"] >= (late_ms >= 20) and bus.stats["failures"] == 0
                if stale_calls:
                    stale_cases.append(f"{late_ms:g} ms{' burst' if burst else ''}")
    RESULTS["position sync read with six late replies, 30 cases"] = (
        "always fresh" if not stale_cases else
        f"the late burst was taken once as the answer of a later call for: {', '.join(stale_cases)}; fresh otherwise")


def test_e04_a_host_stall_never_turns_a_delivered_reply_into_wrong_data():
    """The process is descheduled (in-process model) at the worst moments of the elbow's limit
    read, 8 times in a row; the servos answer on time. With blocking reads a delivered reply
    is read, not timed out: correct limits and (apart from one tolerated scheduler hiccup)
    no retry at all."""
    is_elbow_eeprom = lambda tx: tx[2] == 3 and tx[4] == INST_READ and tx[5] == 0  # noqa: E731
    rows = []
    for where in ("before_read", "mid_reply", "timeout_check"):
        for stall_ms in (40, 60):
            with rig(relaxed=False) as (fake, bus):
                distinct(fake)
                st = inject_host_stall(bus, is_elbow_eeprom, stall_ms / 1000.0, where, count=8)
                for _ in range(8):
                    assert bus.limits() == LIMITS
                check_everything(fake, bus)
                if where == "timeout_check":
                    assert st["fired"] <= 1, "the timeout clock is not even consulted when the reply is there"
                else:
                    assert st["fired"] == 8
                assert bus.stats["retries"] <= 1 and bus.stats["failures"] == 0, bus.stats
                rows.append(f"{where} {stall_ms} ms x{st['fired']}: {bus.stats['retries']} retr.")
    with rig(relaxed=False, stamp=True) as (fake, bus):  # the same on the position sync read
        st = inject_host_stall(bus, lambda tx: tx[4] == INST_SYNC_READ, 0.040, "before_read", count=8)
        for k in range(10):
            s0 = fake.seq
            pos = bus.positions()
            assert all(s0 < v <= fake.seq for v in pos.values()), (k, pos, s0)
        assert st["fired"] == 8 and bus.stats["retries"] <= 1
    RESULTS["8 host stalls per case during limits() (always correct)"] = "; ".join(rows)


def test_e05_a_real_process_stall_during_open_and_limits():
    """The client PROCESS (real Bus.open() + limits(), as Arm.open() does) is frozen with SIGSTOP
    for 60 ms at a random instant of every elbow limit read; the servos answer on time."""
    n = 12
    rnd = random.Random(7)
    with FakeBus(keep_slave_open=False) as fake:
        fake.rule(lambda r: r["id"] == 3 and r["inst"] == INST_READ and r["addr"] == 0, count=FOREVER,
                  stall=0.060, stall_after=lambda: rnd.uniform(0.0, 0.00035))
        p = fake.spawn_client([sys.executable, os.path.abspath(__file__), "--client", "limits", fake.port,
                               fake.lock_path, str(n)])
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, f"client failed:\n{err}"
        res = json.loads(out)
        truth = {j: list(v) for j, v in LIMITS.items()}
        wrong = [r for r in res["results"] if r != ["ok", truth]]
        assert not wrong, wrong[:3]
        assert len(res["results"]) == n and len(fake.stall_log) >= n, (len(res["results"]), len(fake.stall_log))
        assert all(0.055 < b - a < 0.5 for a, b in fake.stall_log), "each stall really lasted about 60 ms"
        assert res["stats"]["failures"] == 0 and res["stats"]["retries"] <= 2, res["stats"]
        assert fake.thread_error is None
        RESULTS["real SIGSTOP stalls (60 ms) during open()+limits() in a client process"] = (
            f"{len(fake.stall_log)} stalls, {n}/{n} limits() correct, client stats {res['stats']}")


def test_e06_misaddressed_duplicated_and_noisy_replies():
    is_ram3 = lambda r: r["id"] == 3 and r["inst"] == INST_READ and r["addr"] == 40  # noqa: E731
    for action in (dict(wrong_id=4), dict(wrong_id=6), dict(dup=0.0002), dict(dup=0.004), dict(dup=0.034),
                   dict(garbage=b"\x00\x13\x37"), dict(garbage=b"\xff"), dict(garbage=b"\xff\xff"),
                   dict(garbage=b"\xff\xff\x03"), dict(garbage=b"\xff\xff\x03\x21\x00"), dict(garbage=bytes(40)),
                   dict(garbage=pkt(3, [0])), dict(garbage=pkt(3, [0, 1, 2])), dict(garbage=pkt(4, [0] * 32))):
        with rig(relaxed=False) as (fake, bus):
            distinct(fake)
            fake.rule(is_ram3, **action)
            assert bus.read("elbow_flex", "Goal_Position") == fake.reg("elbow_flex", "Goal_Position"), action
            assert bus.read("elbow_flex", "Min_Position_Limit") == 775, action
            assert bus.read("wrist_flex", "Goal_Position") == fake.reg("wrist_flex", "Goal_Position"), action
            check_everything(fake, bus)
            time.sleep(0.05)
            check_everything(fake, bus)
            assert bus.stats["failures"] == 0
    rnd = random.Random(11)  # random line noise in front of 150 replies: correct value or BusError, never a wrong value
    with rig(relaxed=False) as (fake, bus):
        distinct(fake)
        fake.fault_hook = lambda req: {"garbage": bytes(rnd.choice((0xFF, 0xFF, 0x03, 0x21, 0x00, rnd.randrange(256)))
                                                        for _ in range(rnd.randrange(1, 7)))}
        ok = err = 0
        for i in range(150):
            j = JOINTS[i % 6]
            try:
                assert bus.read(j, "Goal_Position") == fake.reg(j, "Goal_Position")
                ok += 1
            except BusError:
                err += 1
        fake.fault_hook = None
        time.sleep(0.03)
        check_everything(fake, bus)
        RESULTS["150 reads with random line noise in front of every reply"] = (
            f"{ok} correct, {err} BusError, 0 wrong; {bus.stats['retries']} retries")


def test_e07_a_reply_to_another_request_cannot_satisfy_the_lock_check_of_set_p_gain():
    """Lock really is 0 (LeRobot's disconnect). Replies of the same servo to OTHER requests
    (limit block, write ack, sync reads) are late by various amounts; the old code took a
    stale 1-byte reply for the Lock value. set_p_gain must refuse and never write address 21."""
    late_of = {
        "limit block read": (lambda r: r["inst"] == INST_READ and r["addr"] == 0,
                             lambda bus: bus.read("shoulder_lift", "Min_Position_Limit")),
        "write ack": (lambda r: r["inst"] == INST_WRITE, lambda bus: bus.write("shoulder_lift", "Goal_Velocity", 400)),
        "position sync read": (lambda r: r["inst"] == INST_SYNC_READ and r["n"] == 2, lambda bus: bus.positions()),
        "feedback sync read": (lambda r: r["inst"] == INST_SYNC_READ and r["n"] == 15, lambda bus: bus.feedback()),
    }
    for name, (match, first) in late_of.items():
        for late_ms in (13, 32.5, 40, 53.5, 60):
            with rig(relaxed=False) as (fake, bus):
                fake.lerobot_disconnect()
                fake.rule(lambda r, match=match: r["id"] == 2 and match(r), delay=late_ms / 1000.0)
                first(bus)  # retried; its first answer is still on the way
                with raises(BusError, "Lock is not 1"):
                    bus.set_p_gain("shoulder_lift", 32)
                assert fake.ee_writes == [] and fake.reg("shoulder_lift", "P_Coefficient") == 16, (name, late_ms)
                assert bus.stats["retries"] >= (late_ms >= 20)


def test_e08_the_worst_case_straggler_is_an_older_answer_to_the_same_question():
    """What the protocol cannot rule out (it has no sequence numbers): a reply that arrives in
    the fraction of a millisecond between the flush of a LATER identical request to the same
    servo and that request's own answer. Pinned here: it is an older sample of the same
    register block of the same servo, and the next request is fresh again (no lasting lag)."""
    is_ram6 = lambda p: p[2] == 6 and p[4] == INST_READ and p[5] == 40  # noqa: E731
    with rig(relaxed=False, stamp=True) as (fake, bus):
        fake.rule(lambda r: r["id"] == 6 and r["inst"] == INST_READ and r["addr"] == 40,
                  hold_until=is_ram6, hold_skip=1)
        s0 = fake.seq
        a = bus.read("gripper", "Present_Position")  # first attempt unanswered (held back), second answered
        s1 = fake.seq
        assert s0 < a <= s1 and bus.stats["retries"] >= 1
        b = bus.read("gripper", "Present_Position")  # the held reply lands right behind this request
        s2 = fake.seq
        assert b == s0 + 1, (b, s0, s1, s2)  # ... and is taken: the answer to the first attempt of the call before
        assert bus.read("gripper", "Goal_Position") == 2184 and bus.read("gripper", "Lock") == 1  # still the right fields
        time.sleep(0.005)
        s3 = fake.seq
        c = bus.read("gripper", "Present_Position")
        assert s3 < c <= fake.seq, "fresh again"
        assert bus.read("gripper", "Model") == 777 and bus.limits() == LIMITS
        assert bus.stats["failures"] == 0


def test_e09_the_worst_case_straggler_burst_of_a_sync_read():
    """Same as test_e08 for the position sync read: the six answers to a request that timed
    out arrive right behind the request of the NEXT call. They are taken (each for its own
    servo: the id order is checked), the call after that is fresh again."""
    is_pos_sync = lambda p: p[2] == 0xFE and p[4] == INST_SYNC_READ and p[6] == 2  # noqa: E731
    with rig(relaxed=False, stamp=True) as (fake, bus):
        fake.rule(lambda r: r["inst"] == INST_SYNC_READ and r["n"] == 2, count=6, hold_until=is_pos_sync, hold_skip=1)
        s0 = fake.seq
        a = bus.positions()  # first attempt unanswered (held back), second answered
        assert all(s0 < v <= fake.seq for v in a.values()) and bus.stats["retries"] >= 1
        b = bus.positions()  # the held burst lands right behind this request
        assert b == dict.fromkeys(JOINTS, s0 + 1), b
        time.sleep(0.005)
        s2 = fake.seq
        c = bus.positions()
        assert all(s2 < v <= fake.seq for v in c.values()), "fresh again"
        assert bus.stats["failures"] == 0 and bus.stats["fallbacks"] == 0


# ============================================================================== F. retries, fallback, counters
def test_f01_a_lost_corrupt_or_misaddressed_reply_is_retried_once_and_counted():
    actions = {"drop": dict(drop=True), "corrupt": dict(corrupt=True), "truncate to 5": dict(truncate=5),
               "truncate to 1": dict(truncate=1), "wrong id": dict(wrong_id=4), "request lost": dict(dead=True)}
    for name, action in actions.items():
        with rig() as (fake, bus):  # a read
            distinct(fake)
            r = fake.rule(lambda q: q["id"] == 3 and q["inst"] == INST_READ and q["addr"] == 40, **action)
            m, base = fake.mark(), dict(bus.stats)
            assert bus.read("elbow_flex", "Goal_Position") == fake.reg("elbow_flex", "Goal_Position"), name
            assert r["hits"] == 1 and fake.sent(m) == [RAM_READ(3)] * 2, name
            assert bus.stats == {"retries": base["retries"] + 1, "fallbacks": 0, "failures": 0,
                                 "requests": base["requests"] + 2}, (name, bus.stats)
            check_everything(fake, bus)
            assert bus.stats["retries"] == base["retries"] + 1
        with rig() as (fake, bus):  # a write: sent again, and absolute values make the repeat harmless
            r = fake.rule(lambda q: q["id"] == 3 and q["inst"] == INST_WRITE, **action)
            m = fake.mark()
            bus.write("elbow_flex", "Goal_Position", 2000)
            assert fake.sent(m) == [pkt(3, [INST_WRITE, 42, 0xD0, 0x07])] * 2 and r["hits"] == 1, name
            assert fake.reg("elbow_flex", "Goal_Position") == 2000
            assert [g for _, sid, g in fake.goal_log if sid == 3] == [2000] * (1 if name == "request lost" else 2)
            assert bus.stats["retries"] == 1 and bus.stats["failures"] == 0


def test_f02_a_dead_servo_fails_after_all_attempts_with_a_clear_error():
    for retries in (4, 2, 1):
        with rig(retries=retries) as (fake, bus):
            fake.rule(lambda q: q["id"] == 3, count=FOREVER, dead=True)
            m, base = fake.mark(), dict(bus.stats)
            with raises(BusError, "read [40..70] from elbow_flex", f"failed after {retries} attempt", "no reply",
                        "comm result -6"):
                bus.read("elbow_flex", "Goal_Position")
            assert fake.sent(m) == [RAM_READ(3)] * retries
            assert bus.stats == {"retries": base["retries"] + retries, "fallbacks": 0, "failures": 1,
                                 "requests": base["requests"] + retries}, bus.stats
            with raises(BusError, "write Goal_Position=2000 to elbow_flex", "no reply"):
                bus.write("elbow_flex", "Goal_Position", 2000)
            assert bus.stats["failures"] == 2
            assert bus.read("wrist_flex", "Present_Position") == 2820, "the other servos are still reachable"
            assert bus.ph.is_using is False


def test_f03_what_a_lost_reply_costs_at_the_real_timeout():
    def ms(fn, expect_error=False):
        t0, c0 = time.perf_counter(), time.thread_time()
        try:
            fn()
            assert not expect_error
        except BusError:
            assert expect_error
        return (time.perf_counter() - t0) * 1e3, (time.thread_time() - c0) * 1e3

    with rig(relaxed=False) as (fake, bus):
        distinct(fake)
        fake.rule(lambda q: q["id"] == 3 and q["inst"] == INST_READ, drop=True)
        one, _ = ms(lambda: bus.read("elbow_flex", "Goal_Position"))
        fake.rule(lambda q: q["inst"] == INST_SYNC_READ and q["id"] == 4, drop=True)
        mid, _ = ms(bus.positions)
        fake.rule(lambda q: q["inst"] == INST_SYNC_READ and q["id"] == 6, drop=True)
        last, _ = ms(bus.positions)
        fake.rule(lambda q: q["inst"] == INST_SYNC_READ and q["id"] == 2, corrupt=True)
        corrupt, _ = ms(bus.feedback)
        fake.rule(lambda q: q["id"] == 3, count=FOREVER, dead=True)
        dead_read, dead_cpu = ms(lambda: bus.read("elbow_flex", "Goal_Position"), expect_error=True)
        dead_sync, _ = ms(bus.positions, expect_error=True)
        assert bus.stats["fallbacks"] == 1
        # lower bounds are fixed by construction (12 ms timeout, 20 ms quarantine); upper bounds are generous
        assert 12 + 20 + 20 - 3 < one < 400, one
        assert 20 - 2 < mid < 400 and 12 + 20 - 3 < last < 400 and 20 - 2 < corrupt < 400, (mid, last, corrupt)
        assert 4 * 32 - 6 < dead_read < 1000 and 4 * 20 + 4 * 32 - 8 < dead_sync < 1500, (dead_read, dead_sync)
        RESULTS["cost of comm faults at the real 12 ms timeout"] = (
            f"single read, reply lost once {one:.0f} ms; sync read, a middle reply lost {mid:.0f} ms, the last reply "
            f"lost {last:.0f} ms, a corrupt reply {corrupt:.0f} ms; dead servo: read() BusError after {dead_read:.0f} ms "
            f"({dead_cpu:.1f} ms CPU), positions() BusError after {dead_sync:.0f} ms")


def test_f04_sync_read_retry_and_fallback_are_counted():
    with rig() as (fake, bus):
        distinct(fake)
        t = truth_snapshot(fake)
        pos = {j: t[j]["pos"] for j in JOINTS}
        fake.rule(lambda q: q["inst"] == INST_SYNC_READ and q["id"] == 4, drop=True)  # (a) lost once
        base = dict(bus.stats)
        assert bus.positions() == pos
        assert bus.stats == {"retries": base["retries"] + 1, "fallbacks": 0, "failures": 0,
                             "requests": base["requests"] + 2}, bus.stats
        # (b) the sync replies of one servo never arrive although it answers single reads: fallback
        r = fake.rule(lambda q: q["inst"] == INST_SYNC_READ and q["id"] == 4, count=FOREVER, drop=True)
        m, base = fake.mark(), dict(bus.stats)
        assert bus.positions() == pos
        sync = pkt(0xFE, [INST_SYNC_READ, 56, 2, 1, 2, 3, 4, 5, 6])
        assert fake.sent(m) == [sync] * 4 + [pkt(sid, [INST_READ, 56, 2]) for sid in range(1, 7)]
        assert bus.stats == {"retries": base["retries"] + 4, "fallbacks": 1, "failures": 0,
                             "requests": base["requests"] + 10}, bus.stats
        fb = bus.feedback()
        assert fb == {j: {k: t[j][k] for k in ("pos", "load", "current", "volt", "temp", "status")} for j in JOINTS}
        assert bus.stats["fallbacks"] == 2 and bus.stats["failures"] == 0
        fake.rules.remove(r)  # (c) healthy again: no further retries
        base = dict(bus.stats)
        assert bus.positions() == pos
        assert bus.stats == {**base, "requests": base["requests"] + 1}


def test_f05_sync_read_with_a_dead_servo_fails_with_both_reasons():
    with rig(retries=2) as (fake, bus):
        fake.rule(lambda q: q["id"] == 4, count=FOREVER, dead=True)
        base = dict(bus.stats)
        with raises(BusError, "sync read [56..57] failed after 2 attempt", "fallback to single reads",
                    "read [56..57] from wrist_flex", "no reply"):
            bus.positions()
        assert bus.stats == {"retries": base["retries"] + 4, "fallbacks": 1, "failures": 1,
                             "requests": base["requests"] + 2 + 3 + 2}, bus.stats
        with raises(BusError, "sync read [56..70]", "wrist_flex"):
            bus.feedback()
        assert bus.sync_read("Present_Position", ["shoulder_pan", "gripper"]) == {"shoulder_pan": 1999, "gripper": 2184}
        assert bus.feedback(["elbow_flex"])["elbow_flex"]["pos"] == 3025


def test_f06_sync_write_transmit_failures_are_retried_and_counted():
    with rig() as (fake, bus):
        real = bus.ph.writePort
        calls = {"n": 0}

        def short_once(packet):
            calls["n"] += 1
            return 0 if calls["n"] == 1 else real(packet)

        bus.ph.writePort = short_once
        base = dict(bus.stats)
        bus.sync_write("Goal_Position", {"gripper": 2300})
        fake.drain()
        assert fake.reg("gripper", "Goal_Position") == 2300
        assert bus.stats == {"retries": base["retries"] + 1, "fallbacks": 0, "failures": 0,
                             "requests": base["requests"] + 2}
        bus.ph.writePort = lambda packet: 0
        with raises(BusError, "sync write Goal_Position", "could not be transmitted in 4 attempt", "comm result -2"):
            bus.sync_write("Goal_Position", {"gripper": 2400})
        bus.ph.writePort = real
        fake.drain()
        assert fake.reg("gripper", "Goal_Position") == 2300 and bus.stats["failures"] == 1
        bus.sync_write("Goal_Position", {"gripper": 2400})
        fake.drain()
        assert fake.reg("gripper", "Goal_Position") == 2400 and bus.positions() == REST_RAW


# ============================================================================== G. torque_on
CARRY_RAW = {"shoulder_pan": 2044, "shoulder_lift": 2173, "elbow_flex": 2160, "wrist_flex": 2812,
             "wrist_roll": 2212, "gripper": 2194}  # a pose of the real run log: carrying a block
OFFSETS = {"shoulder_pan": -5, "shoulder_lift": -32, "elbow_flex": -23, "wrist_flex": 1, "wrist_roll": -4,
           "gripper": -40}  # goal - position while carrying (real log); gripper: the grasp squeeze


def holding_rig(fake):
    """The arm holds a block in the air: sag offsets on the arm joints, the gripper squeezing
    40 ticks into an object it cannot close through."""
    for j, raw in CARRY_RAW.items():
        fake.set_pos(j, raw)
    hold(fake, OFFSETS)
    grip = fake.pos[IDS["gripper"]]
    fake.blocker = lambda sid, p: sid == IDS["gripper"] and p < grip
    fake.sag[IDS["gripper"]] = 0.0
    time.sleep(0.03)
    assert goal_minus_pos(fake) == OFFSETS


def test_g01_torque_on_from_all_off_seats_goal_and_caps_before_torque():
    for auto in (False, True):  # with and without "a Goal_Position write switches torque on"
        with rig(auto_torque_on_goal=auto) as (fake, bus):
            pos0 = dict(fake.pos)
            t_mark, m = fake.now(), fake.mark()
            assert bus.torque_on(velocity=1100, acceleration=40) == {"newly_enabled": list(JOINTS)}
            w = wr(fake, t_mark)
            for j in JOINTS:
                assert [(reg, v) for jj, reg, v in w if jj == j] == [
                    ("Lock", 1), ("Acceleration", 40), ("Goal_Velocity", 1100), ("Goal_Position", REST_RAW[j]),
                    ("Torque_Enable", 1)], (j, w)
            regs = [reg for _, reg, _ in w]
            g0, t0 = regs.index("Goal_Position"), regs.index("Torque_Enable")
            assert set(regs[:g0]) == {"Lock", "Acceleration", "Goal_Velocity"}, "every cap is in before the first goal"
            assert set(regs[g0:t0]) == {"Goal_Position"} and set(regs[t0:]) == {"Torque_Enable"}
            pk = fake.sent(m)
            last_goal = max(i for i, p in enumerate(pk) if p[4] == INST_WRITE and p[5] == 42)
            first_torque = min(i for i, p in enumerate(pk) if p[4] == INST_WRITE and p[5] == 40)
            assert pk[last_goal + 1:first_torque] == [RAM_READ(sid) for sid in range(1, 7)], "read back before enabling"
            assert not [p for p in pk if p[4] == INST_SYNC_WRITE] and fake.ee_writes == []
            v = fake.view()
            assert v["torque"] == [1] * 6 and v["lock"] == [1] * 6 and v["acc"] == [40] * 6 and v["vel"] == [1100] * 6
            assert v["goal"] == v["pos"] == [REST_RAW[j] for j in JOINTS]
            time.sleep(0.3)
            assert fake.pos == pos0, "nothing moved"
            assert bus.stats["retries"] == 0 and bus.stats["failures"] == 0


def test_g02_redundant_torque_on_sends_no_write_and_keeps_every_goal():
    with rig() as (fake, bus):
        holding_rig(fake)
        pos0 = dict(fake.pos)
        m = fake.mark()
        for _ in range(3):
            assert bus.torque_on(velocity=1100, acceleration=40) == {"newly_enabled": []}
        fake.drain()
        sent = fake.sent(m)
        assert not [p for p in sent if p[4] in (INST_WRITE, INST_SYNC_WRITE)], "not a single write"
        assert sent == [RAM_READ(sid) for sid in range(1, 7)] * 3, "six block reads per call and nothing else"
        time.sleep(0.3)
        assert goal_minus_pos(fake) == OFFSETS and fake.pos == pos0, "no droop, no release"
        assert fake.pos[6] - fake.reg("gripper", "Goal_Position") == 40, "the grasp squeeze survived"
        assert fake.view()["torque"] == [1] * 6 and bus.stats["retries"] == 0


def test_g03_torque_on_with_one_joint_off_touches_only_that_joint():
    for auto in (False, True):
        with rig(auto_torque_on_goal=auto) as (fake, bus):
            holding_rig(fake)
            fake.power_cycle("elbow_flex")  # the elbow servo browned out: torque off, caps and goal gone
            fake.sag[IDS["elbow_flex"]] = 0.0
            goals0 = {j: fake.reg(j, "Goal_Position") for j in JOINTS}
            t_mark = fake.now()
            assert bus.torque_on(velocity=1100, acceleration=40) == {"newly_enabled": ["elbow_flex"]}
            assert wr(fake, t_mark) == [("elbow_flex", "Lock", 1), ("elbow_flex", "Acceleration", 40),
                                        ("elbow_flex", "Goal_Velocity", 1100), ("elbow_flex", "Goal_Position", 2160),
                                        ("elbow_flex", "Torque_Enable", 1)]
            assert fake.view()["torque"] == [1] * 6
            assert {j: fake.reg(j, "Goal_Position") for j in JOINTS if j != "elbow_flex"} == {
                j: g for j, g in goals0.items() if j != "elbow_flex"}
            time.sleep(0.2)
            assert goal_minus_pos(fake) == {**OFFSETS, "elbow_flex": 0}


def test_g04_torque_on_after_lerobot_left_lock_0():
    with rig() as (fake, bus):
        fake.lerobot_disconnect()
        assert fake.view()["lock"] == [0] * 6 and fake.view()["torque"] == [0] * 6
        t_mark = fake.now()
        assert bus.torque_on(velocity=1100, acceleration=40) == {"newly_enabled": list(JOINTS)}
        w = wr(fake, t_mark)
        for j in JOINTS:
            assert [x for x in w if x[0] == j][0] == (j, "Lock", 1), "Lock := 1 is the first thing a servo is told"
        v = fake.view()
        assert v["lock"] == [1] * 6 and v["torque"] == [1] * 6 and fake.ee_writes == []
        for j in JOINTS[:5]:  # the gain can now be raised, and nothing of it is stored
            assert bus.set_p_gain(j, 32) is True
        v = fake.view()
        assert v["P"] == [32] * 5 + [16] and v["P_stored"] == [16] * 6
        assert [(lock, stored) for *_, lock, stored in fake.ee_writes] == [(1, False)] * 5
        fake.power_cycle()
        assert fake.view()["P"] == [16] * 6


def test_g05_torque_on_rewrites_only_what_differs_on_a_holding_joint():
    with rig() as (fake, bus):
        holding_rig(fake)
        fake.poke("shoulder_lift", "Goal_Velocity", 0)  # cap lost
        fake.poke("wrist_flex", "Acceleration", 254)    # what LeRobot's configure() leaves
        fake.poke("wrist_roll", "Lock", 0)
        t_mark = fake.now()
        assert bus.torque_on(velocity=1100, acceleration=40) == {"newly_enabled": []}
        assert sorted(wr(fake, t_mark)) == [("shoulder_lift", "Goal_Velocity", 1100),
                                            ("wrist_flex", "Acceleration", 40), ("wrist_roll", "Lock", 1)]
        time.sleep(0.2)
        assert goal_minus_pos(fake) == OFFSETS
        v = fake.view()
        assert v["acc"] == [40] * 6 and v["vel"] == [1100] * 6 and v["lock"] == [1] * 6


def test_g06_torque_on_refuses_missing_caps_and_bad_arguments_with_nothing_sent():
    with rig() as (fake, bus):
        m = fake.mark()
        for kw, text in ((dict(velocity=0), "unlimited"), (dict(acceleration=0), "unlimited"),
                         (dict(velocity=0, acceleration=0), "nothing was changed"),
                         (dict(velocity=4001), "Goal_Velocity=4001"), (dict(acceleration=255), "Acceleration=255"),
                         (dict(velocity=-5), "Goal_Velocity=-5"), (dict(velocity=float("nan")), "not a number"),
                         (dict(acceleration=None), "not a number"), (dict(max_goal_gap=None), "max_goal_gap")):
            with raises(BusError, text):
                bus.torque_on(**kw)
        fake.drain()
        assert fake.sent(m) == [] and fake.view()["torque"] == [0] * 6


def test_g07_torque_on_refuses_a_joint_that_is_not_in_position_mode():
    with rig() as (fake, bus):
        fake.poke("gripper", "Operating_Mode", 1)  # velocity mode: Goal_Velocity would be a spin command
        m = fake.mark()
        with raises(BusError, "torque_on refused, nothing was changed", "gripper", "Operating_Mode is 1"):
            bus.torque_on(velocity=1100, acceleration=40)
        fake.drain()
        assert all(p[4] == INST_READ for p in fake.sent(m)) and fake.view()["torque"] == [0] * 6
        assert fake.view()["vel"] == [0] * 6 and fake.view()["acc"] == [0] * 6
        assert bus.torque_on(JOINTS[:5], velocity=1100, acceleration=40) == {"newly_enabled": list(JOINTS[:5])}
        assert fake.view()["torque"] == [1, 1, 1, 1, 1, 0]
    with rig() as (fake, bus):
        fake.poke("wrist_flex", "Min_Position_Limit", 0)
        fake.poke("wrist_flex", "Max_Position_Limit", 0)  # multi-turn configuration
        with raises(BusError, "nothing was changed", "wrist_flex", "implausible position limits"):
            bus.torque_on(velocity=1100, acceleration=40)
        assert fake.view()["torque"] == [0] * 6


def test_g08_torque_on_refuses_a_joint_standing_outside_its_own_limits():
    with rig() as (fake, bus):
        fake.set_pos("elbow_flex", 3100)  # 36 ticks beyond its limit 3064
        fake.poke("elbow_flex", "Goal_Position", 3100)
        m = fake.mark()
        with raises(BusError, "torque_on refused, nothing was changed", "elbow_flex", "position 3100",
                    "775..3064", "by hand"):
            bus.torque_on(velocity=1100, acceleration=40)
        fake.drain()
        assert all(p[4] == INST_READ for p in fake.sent(m)) and fake.view()["torque"] == [0] * 6
        # why it is refused: the servo clamps its goal to its limits, "goal := present" would not hold it still
        fake.poke("elbow_flex", "Goal_Velocity", 1100)
        fake.poke("elbow_flex", "Torque_Enable", 1)
        time.sleep(0.3)
        assert round(fake.pos[3]) == 3064, "(simulated servo) it walks to the limit by itself"
    with rig() as (fake, bus):  # inside max_goal_gap: accepted, and the goal is seated ON the limit
        fake.set_pos("elbow_flex", 3070)
        fake.poke("elbow_flex", "Goal_Position", 3070)
        assert bus.torque_on(velocity=1100, acceleration=40) == {"newly_enabled": list(JOINTS)}
        assert fake.reg("elbow_flex", "Goal_Position") == 3064 and fake.view()["torque"] == [1] * 6
        time.sleep(0.2)
        assert round(fake.pos[3]) == 3064
    with rig() as (fake, bus):  # a tighter gap
        fake.set_pos("elbow_flex", 3070)
        with raises(BusError, "elbow_flex", "more than 3 ticks outside"):
            bus.torque_on(velocity=1100, acceleration=40, max_goal_gap=3)
        assert fake.view()["torque"] == [0] * 6


def test_g09_torque_on_aborts_and_switches_back_off_when_a_joint_moves_meanwhile():
    for auto in (False, True):
        with rig(auto_torque_on_goal=auto) as (fake, bus):
            seen = {"n": 0}

            def shove(req):  # the lift is pushed 50 ticks between "goal := present" and the read-back
                seen["n"] += 1
                if seen["n"] == 2:
                    fake.set_pos("shoulder_lift", fake.pos[2] + 50)

            fake.rule(lambda q: q["id"] == 2 and q["inst"] == INST_READ and q["addr"] == 40, count=FOREVER, call=shove)
            t_mark = fake.now()
            with raises(BusError, "torque_on aborted", "shoulder_lift: goal 955 is -50 ticks from position 1005",
                        "switched back OFF"):
                bus.torque_on(velocity=1100, acceleration=40)
            assert fake.view()["torque"] == [0] * 6, "nothing is left torqued behind the error"
            assert ("Torque_Enable", 1) not in [(reg, v) for _, reg, v in wr(fake, t_mark)]


def test_g10_torque_on_with_a_servo_dying_mid_sequence_leaves_no_new_joint_torqued():
    for auto in (False, True):
        with rig(retries=2, auto_torque_on_goal=auto) as (fake, bus):  # all off; wrist_flex ignores "torque on"
            fake.rule(lambda q: q["id"] == 4 and q["inst"] == INST_WRITE and q["addr"] == 40, count=FOREVER, dead=True)
            with raises(BusError, "torque_on aborted", "write Torque_Enable=1 to wrist_flex", "switched back OFF",
                        "NOT confirmed for wrist_flex"):
                bus.torque_on(velocity=1100, acceleration=40)
            assert fake.view()["torque"] == [0, 0, 0, 1 if auto else 0, 0, 0], "only the unreachable servo can differ"
    with rig(retries=2) as (fake, bus):  # five joints holding, the elbow is off and cannot be enabled
        holding_rig(fake)
        fake.power_cycle("elbow_flex")
        fake.sag[IDS["elbow_flex"]] = 0.0
        goals0 = {j: fake.reg(j, "Goal_Position") for j in JOINTS}
        fake.rule(lambda q: q["id"] == 3 and q["inst"] == INST_WRITE and q["addr"] == 40, count=FOREVER, dead=True)
        with raises(BusError, "torque_on aborted", "elbow_flex", "keep holding as before"):
            bus.torque_on(velocity=1100, acceleration=40)
        assert fake.view()["torque"] == [1, 1, 0, 1, 1, 1]
        assert {j: fake.reg(j, "Goal_Position") for j in JOINTS} == goals0
        time.sleep(0.2)
        assert {j: v for j, v in goal_minus_pos(fake).items() if j != "elbow_flex"} == {
            j: v for j, v in OFFSETS.items() if j != "elbow_flex"}


def test_g11_torque_on_failing_while_writing_the_caps_changes_no_torque():
    with rig(retries=2) as (fake, bus):
        fake.rule(lambda q: q["id"] == 2 and q["inst"] == INST_WRITE and q["addr"] == 46, count=FOREVER, dead=True)
        t_mark = fake.now()
        with raises(BusError, "torque_on aborted", "write Goal_Velocity=1100 to shoulder_lift",
                    "Torque state not changed"):
            bus.torque_on(velocity=1100, acceleration=40)
        assert {reg for _, reg, _ in wr(fake, t_mark)} <= {"Lock", "Acceleration", "Goal_Velocity"}
        assert fake.view()["torque"] == [0] * 6


def test_g12_torque_on_verifies_the_caps_before_enabling():
    for auto in (False, True):
        with rig(auto_torque_on_goal=auto) as (fake, bus):
            fake.rule(lambda q: q["id"] == 5 and q["inst"] == INST_WRITE and q["addr"] == 46, ignore=True)
            with raises(BusError, "torque_on aborted", "wrist_roll: Goal_Velocity reads back 0 instead of 1100",
                        "switched back OFF"):
                bus.torque_on(velocity=1100, acceleration=40)  # the cap write was acknowledged but did not take
            assert fake.view()["torque"] == [0] * 6


def test_g13_torque_on_and_off_for_a_subset_of_joints():
    with rig() as (fake, bus):
        assert bus.torque_on(["gripper"], velocity=400, acceleration=20) == {"newly_enabled": ["gripper"]}
        assert fake.view()["torque"] == [0, 0, 0, 0, 0, 1]
        assert bus.torque_on("wrist_roll") == {"newly_enabled": ["wrist_roll"]}  # one name; default caps 400 / 20
        assert fake.reg("wrist_roll", "Goal_Velocity") == 400 and fake.reg("wrist_roll", "Acceleration") == 20
        assert bus.torque_on(("gripper", "wrist_roll", "shoulder_pan")) == {"newly_enabled": ["shoulder_pan"]}
        assert fake.view()["torque"] == [1, 0, 0, 0, 1, 1]
        bus.torque_off(["gripper"])
        assert fake.view()["torque"] == [1, 0, 0, 0, 1, 0] and fake.view()["vel"] == [400, 0, 0, 0, 400, 0]
        bus.torque_off("wrist_roll", restore_profile=False)
        assert fake.view()["torque"] == [1, 0, 0, 0, 0, 0] and fake.view()["vel"] == [400, 0, 0, 0, 400, 0]
        assert bus.torque_on(()) == {"newly_enabled": []}


def test_g14_torque_on_while_the_arm_is_still_sagging():
    with rig() as (fake, bus):  # limp arm settling: the lift creeps 300 ticks/s while torque is off
        fake.set_pos("shoulder_lift", 1500)
        fake.drift[IDS["shoulder_lift"]] = 300.0
        assert bus.torque_on(velocity=1100, acceleration=40) == {"newly_enabled": list(JOINTS)}
        pos = fake.pos[IDS["shoulder_lift"]]
        assert abs(fake.reg("shoulder_lift", "Goal_Position") - pos) <= 12, "goal seated where the joint was"
        time.sleep(0.2)
        assert abs(fake.pos[IDS["shoulder_lift"]] - pos) <= 12 and fake.view()["torque"] == [1] * 6, "and it holds"


# ============================================================================== H. torque_off
def test_h01_torque_off_restores_the_profile_registers():
    with rig() as (fake, bus):
        bus.torque_on(velocity=1100, acceleration=40)
        goals = fake.view()["goal"]
        t_mark, m = fake.now(), fake.mark()
        assert bus.torque_off() is None
        assert wr(fake, t_mark) == [x for j in JOINTS for x in ((j, "Goal_Velocity", 0), (j, "Acceleration", 0),
                                                                (j, "Torque_Enable", 0))]
        assert fake.sent(m)[:3] == [bytes.fromhex("FFFF0105032E0000C8"), bytes.fromhex("FFFF0104032900CE"),
                                    bytes.fromhex("FFFF0104032800CF")]
        v = fake.view()
        assert v["torque"] == [0] * 6 and v["acc"] == [0] * 6 and v["vel"] == [0] * 6, "as after a power cycle"
        assert v["lock"] == [1] * 6 and v["goal"] == goals and fake.ee_writes == []
    with rig() as (fake, bus):
        bus.torque_on(velocity=1100, acceleration=40)
        t_mark = fake.now()
        bus.torque_off(restore_profile=False)
        assert wr(fake, t_mark) == [(j, "Torque_Enable", 0) for j in JOINTS]
        v = fake.view()
        assert v["torque"] == [0] * 6 and v["acc"] == [40] * 6 and v["vel"] == [1100] * 6


def test_h02_torque_off_switches_off_every_joint_it_can_reach():
    with rig(retries=2) as (fake, bus):
        bus.torque_on(velocity=1100, acceleration=40)
        fake.rule(lambda q: q["id"] == 3, count=FOREVER, dead=True)
        with raises(BusError, "torque_off incomplete", "write Goal_Velocity=0 to elbow_flex", "no reply",
                    "Torque is off on shoulder_pan, shoulder_lift, wrist_flex, wrist_roll, gripper",
                    "NOT confirmed off on elbow_flex"):
            bus.torque_off()
        v = fake.view()
        assert v["torque"] == [0, 0, 1, 0, 0, 0] and v["acc"] == [0, 0, 40, 0, 0, 0] and v["vel"] == [0, 0, 1100, 0, 0, 0]
    with rig(retries=2) as (fake, bus):  # only a cap write is lost: torque still goes off everywhere
        bus.torque_on(velocity=1100, acceleration=40)
        fake.rule(lambda q: q["id"] == 2 and q["inst"] == INST_WRITE and q["addr"] == 46, count=FOREVER, dead=True)
        with raises(BusError, "torque_off incomplete", "shoulder_lift", "Torque is off on " + ", ".join(JOINTS)) as r:
            bus.torque_off()
        assert "NOT confirmed" not in r.msg and fake.view()["torque"] == [0] * 6


def test_h03_torque_off_works_on_a_servo_that_reports_a_fault():
    with rig() as (fake, bus):
        bus.torque_on(velocity=1100, acceleration=40)
        fake.set_fault("gripper", 0x20)  # overload: LeRobot's disable_torque() raises here and leaves the rest torqued
        bus.torque_off()
        assert fake.view()["torque"] == [0] * 6 and bus.faults() == {"gripper": ["overload"]}


# ============================================================================== I. set_p_gain
def test_i01_set_p_gain_changes_ram_only_and_a_power_cycle_restores_the_stored_value():
    with rig() as (fake, bus):
        calib = fake.calib_bytes()
        m = fake.mark()
        assert bus.set_p_gain("shoulder_pan", 32) is True
        assert fake.sent(m) == [EEPROM_READ(1), RAM_READ(1), bytes.fromhex("FFFF0104031520C2"), EEPROM_READ(1)]
        assert fake.reg("shoulder_pan", "P_Coefficient") == 32 == bus.read("shoulder_pan", "P_Coefficient")
        assert fake.stored("shoulder_pan", "P_Coefficient") == 16, "the stored configuration is untouched"
        assert [e[1:] for e in fake.ee_writes] == [(1, 21, 32, 1, False)]
        assert fake.calib_bytes() == calib and fake.view()["P"] == [32, 16, 16, 16, 16, 16]
        fake.power_cycle()
        assert bus.read("shoulder_pan", "P_Coefficient") == 16


def test_i02_set_p_gain_sends_no_write_when_the_gain_is_already_there():
    with rig() as (fake, bus):
        m = fake.mark()
        assert bus.set_p_gain("elbow_flex", 16) is False  # the value the servo booted with
        assert fake.sent(m) == [EEPROM_READ(3)]
        assert bus.set_p_gain("elbow_flex", 32) is True
        m = fake.mark()
        for _ in range(3):
            assert bus.set_p_gain("elbow_flex", 32) is False
        assert fake.sent(m) == [EEPROM_READ(3)] * 3, "one block read to learn the present gain, and no write"
        assert len(fake.ee_writes) == 1
        fake.poke("elbow_flex", "Lock", 0)  # nothing to write, so nothing to refuse either
        assert bus.set_p_gain("elbow_flex", 32) is False
        fake.power_cycle("elbow_flex")  # after a servo reset the gain is gone: the same call writes again
        assert fake.reg("elbow_flex", "P_Coefficient") == 16
        assert bus.set_p_gain("elbow_flex", 32) is True and fake.reg("elbow_flex", "P_Coefficient") == 32
        assert bus.set_p_gain("elbow_flex", 16) is True and bus.set_p_gain("elbow_flex", 16) is False  # and back


def test_i03_set_p_gain_refuses_while_lock_is_0():
    with rig() as (fake, bus):
        fake.lerobot_disconnect()
        m = fake.mark()
        with raises(BusError, "shoulder_lift: Lock is not 1", "persist", "stays 16"):
            bus.set_p_gain("shoulder_lift", 32)
        assert [p for p in fake.sent(m) if p[4] != INST_READ] == [] and fake.ee_writes == []
        assert fake.reg("shoulder_lift", "P_Coefficient") == 16 == fake.stored("shoulder_lift", "P_Coefficient")
        bus.write("shoulder_lift", "Lock", 1)  # what torque_on does first
        assert bus.set_p_gain("shoulder_lift", 32) is True
        assert fake.reg("shoulder_lift", "P_Coefficient") == 32 and fake.stored("shoulder_lift", "P_Coefficient") == 16


def test_i04_set_p_gain_reports_a_gain_that_does_not_take():
    with rig() as (fake, bus):
        fake.rule(lambda q: q["inst"] == INST_WRITE and q["addr"] == 21, ignore=True)  # acknowledged, not applied
        with raises(BusError, "wrist_flex", "did not read back as 32", "reports 16"):
            bus.set_p_gain("wrist_flex", 32)
    with rig(retries=2) as (fake, bus):  # never acknowledged
        fake.rule(lambda q: q["inst"] == INST_WRITE and q["addr"] == 21, count=FOREVER, dead=True)
        with raises(BusError, "write P_Coefficient=32 to wrist_flex", "no reply"):
            bus.set_p_gain("wrist_flex", 32)
        assert fake.reg("wrist_flex", "P_Coefficient") == 16 and bus.stats["failures"] == 1


def test_i05_set_p_gain_refuses_values_outside_8_to_40():
    with rig() as (fake, bus):
        m = fake.mark()
        for p in (7, 41, 64, 0, -16, 255):
            with raises(BusError, f"P gain {p} outside the allowed 8..40"):
                bus.set_p_gain("shoulder_pan", p)
        for p in (32.5, float("nan"), None, "32"):
            with raises(BusError, "P gain"):
                bus.set_p_gain("shoulder_pan", p)
        fake.drain()
        assert fake.sent(m) == [] and fake.view()["P"] == [16] * 6
        assert bus.set_p_gain("gripper", 8) and bus.set_p_gain("gripper", 40) and bus.set_p_gain("gripper", 16.0)
        assert fake.view()["P"] == [16] * 6 and fake.view()["P_stored"] == [16] * 6


def test_i06_whether_the_gain_survives_a_power_cycle_depends_on_the_firmware_honouring_lock():
    """Both firmware models of the fake give the Bus the same answers until the power is cycled:
    non-persistence cannot be proven offline (hardware check: power-cycle, read P, expect 16)."""
    after = {}
    for model in ("documented", "ignored"):
        with rig(lock_model=model) as (fake, bus):
            bus.torque_on(velocity=1100, acceleration=40)
            calib = fake.calib_bytes()
            for j in JOINTS[:5]:
                assert bus.set_p_gain(j, 32) is True
            assert [bus.read(j, "P_Coefficient") for j in JOINTS] == [32] * 5 + [16]
            fake.power_cycle()
            after[model] = [bus.read(j, "P_Coefficient") for j in JOINTS]
            assert fake.calib_bytes() == calib, "limits, homing offset and mode are never written in either case"
    assert after == {"documented": [16] * 6, "ignored": [32] * 5 + [16]}, after


# ============================================================================== J. faults()
def test_j01_the_status_byte_of_every_reply_is_kept_and_decoded():
    with rig() as (fake, bus):
        assert bus.faults() == {}
        fake.set_fault("gripper", 0x20)
        assert bus.positions() == REST_RAW, "an error bit is not an exception at this layer"
        assert bus.status["gripper"] == 32 and bus.faults() == {"gripper": ["overload"]}
        fake.set_fault("shoulder_lift", 0x2D)
        bus.read("shoulder_lift", "Present_Load")
        assert bus.faults() == {"shoulder_lift": ["voltage", "overheat", "overcurrent", "overload"],
                                "gripper": ["overload"]}
        assert bus.feedback()["gripper"]["status"] == 32 and bus.snapshot()["shoulder_lift"]["status"] == 0x2D
        fake.set_fault("elbow_flex", 0x12)  # sensor + a bit the table does not name
        bus.write("elbow_flex", "Goal_Velocity", 400)  # the acknowledgement of a write carries it too
        assert bus.status["elbow_flex"] == 0x12 and bus.faults()["elbow_flex"] == ["sensor", "unknown_0x10"]
        for j in ("gripper", "shoulder_lift", "elbow_flex"):
            fake.set_fault(j, 0)
        bus.feedback()
        assert bus.faults() == {} and bus.status == dict.fromkeys(JOINTS, 0)
        fake.rule(lambda q: q["id"] == 1 and q["inst"] == INST_SYNC_READ, err=0x04)  # in one reply only
        bus.positions()
        assert bus.faults() == {"shoulder_pan": ["overheat"]}
        bus.positions()
        assert bus.faults() == {} and bus.stats["retries"] == 0


# ============================================================================== K. serial exceptions
def test_k01_an_unplugged_adapter_surfaces_as_bus_errors_only():
    with rig() as (fake, bus):
        bus.torque_on(velocity=1100, acceleration=40)
        fake.hang_up()  # reads and writes on the port now fail with EIO, flushes with termios.error
        base = bus.stats["failures"]
        calls = {"read": lambda: bus.read("gripper", "Present_Position"),
                 "write": lambda: bus.write("gripper", "Goal_Position", 2200),
                 "sync_read": lambda: bus.sync_read("Present_Load"), "positions": bus.positions,
                 "sync_write": lambda: bus.sync_write("Goal_Position", {"gripper": 2200}),
                 "feedback": bus.feedback, "snapshot": bus.snapshot, "limits": bus.limits,
                 "torque_on": lambda: bus.torque_on(velocity=1100, acceleration=40), "torque_off": bus.torque_off,
                 "set_p_gain": lambda: bus.set_p_gain("shoulder_pan", 32)}
        for fn in calls.values():
            with raises(BusError, "serial I/O failed"):  # anything but a BusError propagates and fails the test
                fn()
        assert bus.stats["failures"] >= base + len(calls)
        bus.close()
        assert not bus.is_open


def test_k02_an_exception_in_the_middle_of_a_transaction_does_not_wedge_the_port():
    with rig() as (fake, bus):
        ser = bus.ph.ser
        real_flush = ser.reset_input_buffer

        def boom():
            raise termios.error(5, "Input/output error")

        ser.reset_input_buffer = boom
        for fn in (lambda: bus.read("gripper", "Present_Position"), lambda: bus.write("gripper", "Goal_Position", 2200),
                   bus.positions, lambda: bus.sync_write("Goal_Position", {"gripper": 2200})):
            with raises(BusError, "serial I/O failed", "error", "Input/output error"):
                fn()
        ser.reset_input_buffer = real_flush
        assert bus.positions() == REST_RAW
        real_read = bus.ph.readPort
        for exc in (serial.SerialException("device reports readiness to read but returned no data"),
                    OSError(5, "Input/output error"), ValueError("unexpected")):
            def boom_read(n, exc=exc):
                raise exc
            bus.ph.readPort = boom_read  # raised between request and reply: the SDK has set its busy flag
            with raises(BusError, "read [40..70] from gripper", "serial I/O failed", type(exc).__name__):
                bus.read("gripper", "Present_Position")
            with raises(BusError, "sync read [56..57]", "serial I/O failed", type(exc).__name__):
                bus.positions()
            assert bus.ph.is_using is False
            bus.ph.readPort = real_read
            assert bus.read("gripper", "Present_Position") == 2184 and bus.positions() == REST_RAW


# ============================================================================== L. random faults
def test_l01_random_wire_faults_never_yield_a_wrong_value():
    """Late (5..90 ms), lost, corrupt and truncated replies at random while the Bus is used the
    way the harness uses it. Every value handed back is compared with the simulated servos:
    registers must be exact, positions must stem from a request of that very call (stamps)."""
    rnd, frnd = random.Random(3), random.Random(4)
    kinds = {"late": 0, "drop": 0, "corrupt": 0, "truncate": 0}

    def faults(req):
        x = frnd.random()  # per reply: 1 % late, 1 % lost, 1 % corrupt, 0.5 % truncated
        if x < 0.010:
            kinds["late"] += 1
            return {"delay": frnd.uniform(0.005, 0.090)}
        if x < 0.020:
            kinds["drop"] += 1
            return {"drop": True}
        if x < 0.030:
            kinds["corrupt"] += 1
            return {"corrupt": True}
        if x < 0.035:
            kinds["truncate"] += 1
            return {"truncate": frnd.randint(1, 5)}
        return None

    tally = {"ops": 0, "ok": 0, "BusError": 0, "values": 0, "wrong": 0, "stale": 0}
    examples = []
    with rig(relaxed=False, stamp=True) as (fake, bus):
        distinct(fake)
        static = ("Min_Position_Limit", "Max_Position_Limit", "P_Coefficient", "Torque_Enable", "Acceleration",
                  "Lock", "Present_Load", "Present_Voltage", "Present_Temperature", "Status", "Present_Current",
                  "Torque_Limit", "Operating_Mode", "Model", "Goal_Position", "Goal_Velocity", "Homing_Offset")

        def chk(what, got, want):
            tally["values"] += 1
            if got != want:
                tally["wrong"] += 1
                examples.append(f"{what}: got {got!r}, the servo has {want!r}")

        def chk_stamp(what, got, s0):
            tally["values"] += 1
            if not s0 < got <= fake.seq:
                tally["stale"] += 1
                examples.append(f"{what}: answer to request {got}, this call sent {s0 + 1}..{fake.seq}")

        fake.fault_hook = faults
        for _ in range(FUZZ_OPS):
            op = rnd.choice(("limits", "snapshot", "positions", "feedback", "read", "write_rb", "offset", "grip",
                             "p_gain"))
            j = rnd.choice(JOINTS)
            tally["ops"] += 1
            s0 = fake.seq
            try:
                if op == "limits":
                    lim = bus.limits()
                    for k in JOINTS:
                        chk(f"limits {k}", lim[k], LIMITS[k])
                elif op == "snapshot":
                    snap = bus.snapshot()
                    t = truth_snapshot(fake)
                    for k in JOINTS:
                        chk_stamp(f"snapshot {k}.pos", snap[k].pop("pos"), s0)
                        t[k].pop("pos")
                        chk(f"snapshot {k}", snap[k], t[k])
                elif op == "positions":
                    for k, v in bus.positions().items():
                        chk_stamp(f"positions {k}", v, s0)
                elif op == "feedback":
                    fb = bus.feedback()
                    t = truth_snapshot(fake)
                    for k in JOINTS:
                        chk_stamp(f"feedback {k}.pos", fb[k].pop("pos"), s0)
                        chk(f"feedback {k}", fb[k], {x: t[k][x] for x in ("load", "current", "volt", "temp", "status")})
                elif op == "read":
                    reg = rnd.choice(static)
                    chk(f"read {j}.{reg}", bus.read(j, reg), fake.reg(j, reg))
                elif op == "write_rb":
                    v = rnd.choice((0, 400, 1100, 2000))
                    bus.write(j, "Goal_Velocity", v)
                    chk(f"servo register after write {j}.Goal_Velocity", fake.reg(j, "Goal_Velocity"), v)
                    chk(f"read back {j}.Goal_Velocity", bus.read(j, "Goal_Velocity"), v)
                elif op == "offset":  # what Arm._current_offset() does
                    for k, v in bus.positions().items():
                        chk_stamp(f"offset {k}.pos", v, s0)
                    for k in JOINTS[:5]:
                        chk(f"offset {k}.Goal_Position", bus.read(k, "Goal_Position"), fake.reg(k, "Goal_Position"))
                elif op == "grip":  # what Arm.set_gripper() / grasp() do on servo 6
                    chk("grip Torque_Enable", bus.read("gripper", "Torque_Enable"), fake.reg("gripper", "Torque_Enable"))
                    g = rnd.randrange(2100, 3000)
                    bus.write("gripper", "Goal_Position", g)
                    chk("servo register after write gripper.Goal_Position", fake.reg("gripper", "Goal_Position"), g)
                    s0 = fake.seq
                    chk_stamp("grip pos", bus.read("gripper", "Present_Position"), s0)
                    chk("grip load", bus.read("gripper", "Present_Load"), fake.reg("gripper", "Present_Load"))
                    chk("grip current", bus.read("gripper", "Present_Current"), fake.reg("gripper", "Present_Current"))
                else:  # p_gain
                    p = rnd.choice((16, 32))
                    bus.set_p_gain(j, p)
                    chk(f"servo register after set_p_gain {j}", fake.reg(j, "P_Coefficient"), p)
                tally["ok"] += 1
            except BusError:
                tally["BusError"] += 1  # reported: acceptable
        fake.fault_hook = None
        time.sleep(0.12)
        t = truth_snapshot(fake)
        snap = bus.snapshot()
        for k in JOINTS:
            snap[k].pop("pos"), t[k].pop("pos")
        assert snap == t and bus.limits() == LIMITS
        stats = dict(bus.stats)
    n_faults = sum(kinds.values())
    RESULTS["random wire faults"] = (f"{tally['ops']} operations, {tally['values']} values checked, faults injected "
                                     f"{kinds}: {tally['wrong']} wrong register values, {tally['stale']} stale "
                                     f"positions, {tally['BusError']} BusError; bus stats {stats}")
    assert tally["wrong"] == 0, examples[:5]
    assert tally["stale"] <= kinds["late"], examples[:5]  # a straggler can be taken at most once (see test_e08)
    assert tally["ok"] + tally["BusError"] == tally["ops"] and n_faults > 0
    assert stats["retries"] >= n_faults // 3
    if tally["stale"]:
        RESULTS["random wire faults: stale positions"] = "; ".join(examples[:4])


# ============================================================================== M. public API
def test_m01_public_api_used_by_robot_and_sim_bus_is_intact():
    assert SB.JOINTS == ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
    assert SB.IDS == {j: i + 1 for i, j in enumerate(SB.JOINTS)}
    for name in ("JOINTS", "IDS", "REG", "RANGE", "WRITABLE", "BusError", "Bus", "port_users", "PORT", "BAUD",
                 "LOCK_PATH", "FIRST_RAM_ADDR", "EEPROM_BLOCK", "RAM_BLOCK", "FEEDBACK_BLOCK", "PKT_TIMEOUT_MS",
                 "QUARANTINE_S", "ERR_NAMES"):
        assert hasattr(SB, name), name
    assert issubclass(BusError, RuntimeError)
    assert (SB.EEPROM_BLOCK, SB.RAM_BLOCK, SB.FEEDBACK_BLOCK, SB.FIRST_RAM_ADDR) == ((0, 40), (40, 31), (56, 15), 40)
    assert SB.ERR_NAMES == {1: "voltage", 2: "sensor", 4: "overheat", 8: "overcurrent", 32: "overload"}
    assert SB.RANGE == {"Torque_Enable": (0, 1), "Acceleration": (0, 254), "Goal_Position": (0, 4095),
                        "Goal_Velocity": (0, 4000), "Torque_Limit": (0, 1000), "Lock": (1, 1)}
    params = {"open": ["wait_s"], "close": [], "read": ["joint", "reg"], "write": ["joint", "reg", "value"],
              "sync_read": ["reg", "joints"], "sync_write": ["reg", "values"], "positions": [],
              "feedback": ["joints"], "snapshot": [], "limits": [], "faults": [],
              "torque_on": ["joints", "velocity", "acceleration", "max_goal_gap"],
              "torque_off": ["joints", "restore_profile"], "set_p_gain": ["joint", "p"],
              "read_block": ["joint", "base", "n"], "__enter__": [], "__exit__": ["exc"]}
    for name, want in params.items():
        assert list(inspect.signature(getattr(Bus, name)).parameters)[1:] == want, name
    d = lambda fn, p: inspect.signature(fn).parameters[p].default  # noqa: E731
    assert d(Bus.torque_on, "joints") == JOINTS and d(Bus.torque_on, "velocity") == 400
    assert d(Bus.torque_on, "acceleration") == 20 and d(Bus.torque_on, "max_goal_gap") == 12
    assert d(Bus.torque_off, "joints") == JOINTS and d(Bus.torque_off, "restore_profile") is True
    assert d(Bus.sync_read, "joints") == JOINTS and d(Bus.feedback, "joints") == JOINTS
    assert d(SB.port_users, "port") == SB.PORT
    b = Bus(port="/dev/pts/none", baud=1_000_000, retries=3)
    assert (b.port, b.baud, b.retries) == ("/dev/pts/none", 1_000_000, 3) and not b.is_open
    assert b.status == dict.fromkeys(JOINTS, 0) and b.faults() == {}
    assert b.stats["retries"] == 0 and b.stats["fallbacks"] == 0
    with rig() as (fake, bus):  # the keys robot.py / sim_bus.py rely on
        assert set(bus.snapshot()["gripper"]) == {"pos", "load", "current", "volt", "temp", "torque", "goal",
                                                  "status", "acc", "vel_cap", "lock"}
        assert set(bus.feedback()["gripper"]) == {"pos", "load", "current", "volt", "temp", "status"}
        assert bus.torque_on(JOINTS, velocity=1100, acceleration=40) == {"newly_enabled": list(JOINTS)}
        assert bus.torque_off(JOINTS) is None


# ============================================================================== N. cost of the calls
def test_n01_cost_of_the_calls_on_the_fake_bus():
    def timed(fn, n=40):
        wall, cpu = [], []
        for _ in range(n):
            w0, c0 = time.perf_counter(), time.thread_time()
            fn()
            wall.append((time.perf_counter() - w0) * 1e3)
            cpu.append((time.thread_time() - c0) * 1e3)
        wall.sort()
        cpu.sort()
        return wall[n // 2], cpu[n // 2]

    table = {}
    for label, delay in (("0.3 ms", 0.0003), ("0 ms", 0.0)):
        with rig(relaxed=False, reply_delay=delay) as (fake, bus):
            goal = {j: REST_RAW[j] for j in JOINTS[:5]}
            flip = {"p": 16}

            def p_change():
                flip["p"] = 48 - flip["p"]
                bus.set_p_gain("shoulder_pan", flip["p"])

            def on_off():
                bus.torque_on(velocity=1100, acceleration=40)
                bus.torque_off()

            rows = {
                "read": timed(lambda: bus.read("elbow_flex", "Goal_Position")),
                "write": timed(lambda: bus.write("gripper", "Goal_Velocity", 1100)),
                "positions": timed(bus.positions),
                "feedback": timed(bus.feedback),
                "sync_write": timed(lambda: bus.sync_write("Goal_Position", goal)),
                "stream cycle (sync_write + feedback)": timed(lambda: (bus.sync_write("Goal_Position", goal),
                                                                       bus.feedback())),
                "snapshot": timed(bus.snapshot, 20),
                "limits": timed(bus.limits, 20),
                "set_p_gain (already there)": timed(lambda: bus.set_p_gain("shoulder_pan", 16)),
                "set_p_gain (change)": timed(p_change, 20),
                "torque_on + torque_off (from all off)": timed(on_off, 8),
            }
            bus.torque_on(velocity=1100, acceleration=40)
            rows["torque_on (redundant)"] = timed(lambda: bus.torque_on(velocity=1100, acceleration=40), 20)
            rows["torque_off"] = timed(bus.torque_off, 8)
            assert bus.stats["failures"] == 0
            table[label] = rows
            if delay:
                assert rows["positions"][0] < 25 and rows["read"][0] < 15 and rows["sync_write"][0] < 10, rows
    with FakeBus() as fake:
        b = fake.bus()

        def open_close():
            b.open(wait_s=2.0)
            b.close()

        t0 = time.perf_counter()
        users = SB.port_users(fake.port)
        scan = (time.perf_counter() - t0) * 1e3
        assert users == []
        oc = timed(open_close, 5)
    lines = [f"{'call':40s} {'wall ms':>8s} {'cpu ms':>7s}   (servo reply delay 0.3 ms per reply | 0 ms)"]
    for name in table["0.3 ms"]:
        a, z = table["0.3 ms"][name], table["0 ms"][name]
        lines.append(f"{name:40s} {a[0]:8.2f} {a[1]:7.2f}   | {z[0]:6.2f} {z[1]:6.2f}")
    lines.append(f"{'open() + close()':40s} {oc[0]:8.1f} {oc[1]:7.1f}   (of which the port-user scan: {scan:.0f} ms)")
    RESULTS["cost of the calls (median; cpu = this thread only, the simulated servos run in another)"] = (
        "\n  " + "\n  ".join(lines))


# ============================================================================== O. optional: robot.Arm on the real Bus
def test_o01_arm_on_the_real_bus_through_the_fake():
    """Only with --with-arm: robot.py belongs to MOTION and is being rewritten; this checks that
    Arm.open() / torque_on() / state() / torque_off() run on the real Bus through the pty."""
    if not os.environ.get("FD_TEST_WITH_ARM"):
        raise Skip("optional, run with --with-arm")
    import robot
    with rig(relaxed=False, opened=False) as (fake, bus):
        arm = robot.Arm(bus=bus, cfg=json.loads(json.dumps(robot.DEFAULT_CONFIG)))
        arm.open()
        assert arm.raw_limits == LIMITS
        arm.torque_on()
        v = fake.view()
        assert v["torque"] == [1] * 6 and v["lock"] == [1] * 6 and v["P_stored"] == [16] * 6
        st = arm.state()
        assert st["raw"] == REST_RAW
        m = fake.mark()
        arm.torque_on()
        writes = [p for p in fake.sent(m) if p[4] in (INST_WRITE, INST_SYNC_WRITE)]
        arm.torque_off()
        assert fake.view()["torque"] == [0] * 6
        RESULTS["robot.Arm on the real Bus"] = (f"open/torque_on/state/torque_off ok; a second Arm.torque_on() sent "
                                                f"{len(writes)} write packets; bus stats {bus.stats}")


# ============================================================================== P. the fake itself
def test_p01_the_fake_speaks_the_protocol_the_vendor_sdk_expects():
    """The simulated servos driven by the vendor SDK's own functions (ping, 1/2-byte reads and
    writes, GroupSyncRead, GroupSyncWrite), independently of servo_bus's packet building."""
    with rig() as (fake, bus):
        ph, pk = bus.ph, bus.pk
        ph.setPacketTimeout = lambda n: ph.setPacketTimeoutMillis(RELAXED_MS)
        fake.frozen = True
        for sid in range(1, 7):
            assert pk.ping(ph, sid) == (777, scs.COMM_SUCCESS, 0)  # PING, then the SDK reads the model number
        assert fake.sent(inst=INST_PING)[-1] == bytes.fromhex("FFFF060201F6")
        assert pk.read2ByteTxRx(ph, 3, 56) == (3025, scs.COMM_SUCCESS, 0)
        assert pk.read1ByteTxRx(ph, 6, 55) == (1, scs.COMM_SUCCESS, 0)
        assert pk.write1ByteTxRx(ph, 6, 41, 30) == (scs.COMM_SUCCESS, 0) and fake.reg("gripper", "Acceleration") == 30
        assert pk.write2ByteTxRx(ph, 2, 46, 777) == (scs.COMM_SUCCESS, 0) and fake.reg(2, "Goal_Velocity") == 777
        gr = scs.GroupSyncRead(ph, pk, 56, 2)
        for sid in range(1, 7):
            gr.addParam(sid)
        assert gr.txRxPacket() == scs.COMM_SUCCESS
        assert {j: gr.getData(IDS[j], 56, 2) for j in JOINTS} == REST_RAW
        gw = scs.GroupSyncWrite(ph, pk, 42, 2)
        gw.addParam(1, [0x34, 0x08])
        gw.addParam(5, [0x00, 0x04])
        assert gw.txPacket() == scs.COMM_SUCCESS
        fake.drain()
        assert fake.reg(1, "Goal_Position") == 0x0834 and fake.reg(5, "Goal_Position") == 0x0400
        n = len(fake.replies)
        ph.setPacketTimeout = lambda n: ph.setPacketTimeoutMillis(20)
        ph.ser.timeout = 0.02
        assert pk.read2ByteTxRx(ph, 9, 56)[1] == scs.COMM_RX_TIMEOUT, "no servo 9 on the bus: silence"
        ph.is_using = False
        ph.ser.write(bytes.fromhex("FFFF0304020028FF"))  # wrong checksum: ignored by the servos
        fake.drain()
        assert len(fake.replies) == n and len(fake.bad_requests) == 1
        fake.set_fault(4, 0x21)  # the error byte travels in the SDK's third return value
        assert pk.read1ByteTxRx(ph, 4, 65) == (0x21, scs.COMM_SUCCESS, 0x21)
        # power cycle model: RAM back to power-up values, EEPROM area back to the stored image
        fake.poke(2, "P_Coefficient", 32, store=False)
        fake.poke(2, "Torque_Enable", 1)
        fake.poke(2, "Lock", 0)
        fake.power_cycle()
        assert (fake.reg(2, "P_Coefficient"), fake.reg(2, "Torque_Enable"), fake.reg(2, "Lock"),
                fake.reg(2, "Goal_Velocity"), fake.reg(6, "Acceleration"), fake.reg(4, "Status")) == (16, 0, 1, 0, 0, 0)
        assert fake.reg(2, "Goal_Position") == fake.reg(2, "Present_Position") == 955


# ------------------------------------------------------------------------------ client process (test_e05)
def _client(argv):
    scenario, port, lock_path = argv[0], argv[1], argv[2]
    if scenario != "limits":
        raise SystemExit(f"unknown client scenario {scenario!r}")
    bus = Bus(port=port, lock_path=lock_path).open(wait_s=5.0)  # the real open(), default 12 ms timeout
    results = []
    try:
        for _ in range(int(argv[3])):
            try:
                results.append(["ok", {j: list(v) for j, v in bus.limits().items()}])
            except BusError as e:
                results.append(["BusError", str(e)])
        stats = dict(bus.stats)
    finally:
        bus.close()
    print(json.dumps({"results": results, "stats": stats}))
    return 0


# ------------------------------------------------------------------------------ runner
def _main(argv):
    global FUZZ_OPS
    only = None
    args = list(argv)
    while args:
        a = args.pop(0)
        if a == "-k":
            only = args.pop(0)
        elif a == "--fuzz":
            FUZZ_OPS = int(args.pop(0))
        elif a == "--with-arm":
            os.environ["FD_TEST_WITH_ARM"] = "1"
        else:
            raise SystemExit(f"unknown argument {a!r}\n{__doc__}")
    tests = sorted((n, f) for n, f in globals().items() if n.startswith("test_") and callable(f))
    if only:
        tests = [(n, f) for n, f in tests if only in n]
    passed = failed = skipped = 0
    t_all = time.perf_counter()
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            passed += 1
            print(f"PASS {name}  ({time.perf_counter() - t0:.2f} s)", flush=True)
        except Skip as e:
            skipped += 1
            print(f"SKIP {name}  ({e})", flush=True)
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"FAIL {name}  ({time.perf_counter() - t0:.2f} s)", flush=True)
            traceback.print_exc()
    if RESULTS:
        print("\n--- numbers ---")
        for k, v in RESULTS.items():
            print(f"{k}: {v}")
    print(f"\npassed: {passed}  failed: {failed}  skipped: {skipped}  total: {len(tests)}  "
          f"({time.perf_counter() - t_all:.1f} s)")
    print("failures:", failed)
    return 1 if failed else 0


if __name__ == "__main__":
    if sys.argv[1:2] == ["--client"]:
        sys.exit(_client(sys.argv[2:]))
    sys.exit(_main(sys.argv[1:]))
