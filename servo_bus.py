"""Raw Feetech STS3215 bus access for the SO-101 follower (no LeRobot robot class).

Design rules
- Works in RAW encoder ticks (0..4095). No normalization, no calibration writes.
- The servo-stored calibration (homing offsets, position limits) is the source of truth
  and must stay intact. Only RAM registers (address >= 40) are written, with ONE guarded
  exception: `set_p_gain` writes P_Coefficient (address 21) while Lock == 1, which makes
  the servo apply the value for the current power cycle only (see its docstring).
- One process at a time: an flock on LOCK_PATH serializes our own tools, the port is
  refused if another process already has it open, and it is opened exclusively.
- Every reply is validated against its request (servo id and payload length) and the
  receive buffer is flushed before every request, so a late reply can never be taken for
  the answer to a different request.
- Nothing fails silently: every refusal and every communication failure is a BusError
  that says what was refused or what failed (pyserial / termios exceptions never escape).
  Failed attempts, sync-read fallbacks and failed transactions are counted in `Bus.stats`;
  the error byte of the latest reply of every servo is kept in `Bus.status` and decoded by
  `Bus.faults()`.
- Not thread-safe: one thread talks to the bus.

Tests: test_bus_fake.py runs this class against six simulated servos on a pty (fakebus.py).
"""

from __future__ import annotations

import fcntl
import os
import subprocess
import termios
import time

import scservo_sdk as scs

PORT = "/dev/ttyACM0"
BAUD = 1_000_000
LOCK_PATH = "/tmp/fable_direct_bus.lock"

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
IDS = {name: i + 1 for i, name in enumerate(JOINTS)}

# name -> (address, size, sign_bit or None)
REG = {
    "Firmware_Major": (0, 1, None),
    "Firmware_Minor": (1, 1, None),
    "Model": (3, 2, None),
    "Min_Position_Limit": (9, 2, None),
    "Max_Position_Limit": (11, 2, None),
    "Max_Torque_Limit": (16, 2, None),
    "Phase": (18, 1, None),  # bit 2 set (factory value 12): Goal_Velocity is in steps/s
    "P_Coefficient": (21, 1, None),
    "D_Coefficient": (22, 1, None),
    "I_Coefficient": (23, 1, None),
    "Protection_Current": (28, 2, None),
    "Homing_Offset": (31, 2, 11),
    "Operating_Mode": (33, 1, None),
    "Protective_Torque": (34, 1, None),  # % output after the overload protection tripped
    "Protection_Time": (35, 1, None),  # x 10 ms above Overload_Torque before it trips
    "Overload_Torque": (36, 1, None),
    "Torque_Enable": (40, 1, None),
    "Acceleration": (41, 1, None),
    "Goal_Position": (42, 2, 15),
    "Goal_Velocity": (46, 2, 15),
    "Torque_Limit": (48, 2, None),
    "Lock": (55, 1, None),
    "Present_Position": (56, 2, 15),
    "Present_Velocity": (58, 2, 15),
    "Present_Load": (60, 2, 10),
    "Present_Voltage": (62, 1, None),
    "Present_Temperature": (63, 1, None),
    "Status": (65, 1, None),
    "Moving": (66, 1, None),
    "Present_Current": (69, 2, 15),
}
FIRST_RAM_ADDR = 40
# RAM registers that may be written, with their allowed value range.
# Torque_Enable is limited to 0/1 (128 would re-zero the encoder offset).
# Lock may only be SET (1): that can only make later writes less persistent.
RANGE = {
    "Torque_Enable": (0, 1),
    "Acceleration": (0, 254),
    "Goal_Position": (0, 4095),
    "Goal_Velocity": (0, 4000),
    "Torque_Limit": (0, 1000),
    "Lock": (1, 1),
}
WRITABLE = set(RANGE)

EEPROM_BLOCK = (0, 40)  # addresses 0..39
RAM_BLOCK = (40, 31)  # addresses 40..70
FEEDBACK_BLOCK = (56, 15)  # addresses 56..70
PKT_TIMEOUT_MS = 12.0  # a status packet at 1 Mbps takes < 1 ms; a 6-servo sync read 2-4 ms
QUARANTINE_S = 0.020  # after a failed attempt: let stragglers land, then flush
ERR_NAMES = {1: "voltage", 2: "sensor", 4: "overheat", 8: "overcurrent", 32: "overload"}
P_GAIN_RANGE = (8, 40)

_COMM_TEXT = {
    scs.COMM_PORT_BUSY: "port busy",
    scs.COMM_TX_FAIL: "could not transmit",
    scs.COMM_RX_FAIL: "no status packet",
    scs.COMM_TX_ERROR: "bad instruction packet",
    scs.COMM_RX_TIMEOUT: "no reply",
    scs.COMM_RX_CORRUPT: "corrupt or unexpected reply",
    scs.COMM_NOT_AVAILABLE: "not available",
}


class BusError(RuntimeError):
    pass


def _comm(res) -> str:
    return f"{_COMM_TEXT.get(res, 'error')}, comm result {res}"


def _decode(value: int, sign_bit):
    if sign_bit is None:
        return value
    mag = value & ((1 << sign_bit) - 1)
    return -mag if value & (1 << sign_bit) else mag


def _encode(value: int, sign_bit):
    if sign_bit is None:
        if value < 0:
            raise ValueError("negative value for unsigned register")
        return value
    mag = abs(int(value))
    if mag >= (1 << sign_bit):
        raise ValueError(f"magnitude {mag} too large")
    return mag | (1 << sign_bit) if value < 0 else mag


def _lock_holder(lock_path: str) -> str:
    """' (pid N)' when the holder of the bus lock wrote its pid into the lock file, else ''."""
    try:
        with open(lock_path) as f:
            pid = f.read(32).strip()
    except OSError:
        return ""
    return f" (pid {pid})" if pid.isdigit() else ""


def port_users(port: str = PORT) -> list[int]:
    """PIDs of other processes that have the serial port open.

    Scans /proc (sees every process whose fd table is readable) and also asks fuser.
    Raises BusError if neither source is usable, so a failed check never reads as "free"."""
    me = os.getpid()
    pids = set()
    usable = False
    try:
        real = os.path.realpath(port)
        for name in os.listdir("/proc"):
            if not name.isdigit() or int(name) == me:
                continue
            try:
                for fd in os.listdir(f"/proc/{name}/fd"):
                    try:
                        if os.path.realpath(os.readlink(f"/proc/{name}/fd/{fd}")) == real:
                            pids.add(int(name))
                    except OSError:
                        continue
            except OSError:
                continue  # not our process: unreadable
        usable = True
    except OSError:
        pass
    try:
        out = subprocess.run(["fuser", port], capture_output=True, text=True, timeout=5)
        pids.update(int(p) for p in out.stdout.split() if p.isdigit() and int(p) != me)
        usable = True
    except (OSError, subprocess.SubprocessError):
        pass
    if not usable:
        raise BusError("cannot determine whether the serial port is in use (no /proc, no fuser)")
    return sorted(pids)


class Bus:
    def __init__(self, port: str = PORT, baud: int = BAUD, retries: int = 4, lock_path: str | None = None,
                 timeout_ms: float | None = None):
        """port, baud: the serial port of the servo bus.
        retries: attempts per transaction (>= 1).
        lock_path: flock file that serializes the processes using the bus. None = LOCK_PATH,
            the file every tool on the real port must share; tests pass a scratch file so they
            can neither block nor be blocked by the real harness.
        timeout_ms: how long to wait for a status packet. None = PKT_TIMEOUT_MS, the value
            used on the real bus; only tests on a loaded machine have a reason to raise it."""
        try:
            self.retries = int(retries)
            self.timeout_ms = float(PKT_TIMEOUT_MS if timeout_ms is None else timeout_ms)
        except (TypeError, ValueError):
            raise BusError(f"Bus(retries={retries!r}, timeout_ms={timeout_ms!r}): not numbers") from None
        if self.retries < 1:
            raise BusError(f"Bus(retries={retries!r}): at least one attempt per transaction is needed")
        if not 1.0 <= self.timeout_ms <= 1000.0:
            raise BusError(f"Bus(timeout_ms={timeout_ms!r}): outside 1..1000 ms")
        self.port = port
        self.baud = baud
        self.lock_path = lock_path
        self._lock_fd = None
        self.ph = None
        self.pk = None
        self.exclusive = False  # True once the port is held with TIOCEXCL
        self.status = dict.fromkeys(JOINTS, 0)  # error byte of the most recent reply per servo
        # requests: instruction packets sent; retries: failed attempts (each is followed by a
        # quarantine and, while attempts are left, a re-send); fallbacks: sync reads that had to
        # be replaced by single reads; failures: transactions that gave up and raised BusError
        self.stats = {"retries": 0, "fallbacks": 0, "failures": 0, "requests": 0}

    # ---------------------------------------------------------------- open
    @property
    def is_open(self) -> bool:
        return self.ph is not None and self.pk is not None

    def open(self, wait_s: float = 20.0) -> "Bus":
        if self.ph is not None or self._lock_fd is not None:
            raise BusError(f"the bus on {self.port} is already open")
        lock_path = self.lock_path or LOCK_PATH
        try:
            self._lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o666)
        except OSError as e:
            raise BusError(f"cannot open the bus lock file {lock_path}: {e}") from None
        t0 = time.monotonic()
        while True:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - t0 > wait_s:
                    self._release_lock()
                    raise BusError(f"another fable_direct process{_lock_holder(lock_path)} holds the bus lock "
                                   f"{lock_path} (waited {wait_s:g} s)") from None
                time.sleep(0.1)
            except OSError as e:
                self._release_lock()
                raise BusError(f"cannot lock {lock_path}: {e}") from None
        try:  # our pid, for the error message of a process that waits for the lock in vain
            os.ftruncate(self._lock_fd, 0)
            os.pwrite(self._lock_fd, f"{os.getpid()}\n".encode(), 0)
        except OSError:
            pass
        try:
            others = port_users(self.port)
            if others:
                raise BusError(f"{self.port} is in use by pid(s) {others}; refusing to touch it")
            self.ph = scs.PortHandler(self.port)
            try:
                opened = self.ph.setBaudRate(self.baud)  # opens the port at that rate
            except Exception as e:  # serial.SerialException / OSError / ValueError
                raise BusError(f"cannot open {self.port}: {type(e).__name__}: {e}") from None
            if not opened:
                raise BusError(f"cannot open {self.port}: unsupported baud rate {self.baud!r}")
            try:
                self._harden()
            except Exception as e:
                raise BusError(f"cannot configure {self.port}: {type(e).__name__}: {e}") from None
            self.pk = scs.PacketHandler(0)
            for name in JOINTS:
                try:
                    model = self.read(name, "Model")
                except BusError as e:
                    raise BusError(f"{name} (servo id {IDS[name]}) does not answer on {self.port} "
                                   f"(servo power on? cable?): {e}") from None
                if model != 777:
                    raise BusError(f"{name}: unexpected model number {model} (expected STS3215=777)")
        except BaseException:
            self.close()
            raise
        return self

    def _harden(self):
        """Exclusive open, blocking reads with a short timeout (the kernel waits, so a
        descheduled process cannot turn a delivered reply into a 'timeout')."""
        ph = self.ph
        self.exclusive = False
        try:
            fcntl.ioctl(ph.ser.fd, termios.TIOCEXCL)
            self.exclusive = True
        except (OSError, AttributeError, TypeError):
            pass  # not a tty (in-process test doubles); see `exclusive`
        timeout_ms = self.timeout_ms
        ph.ser.timeout = timeout_ms / 1000.0
        ph.setPacketTimeout = lambda n: ph.setPacketTimeoutMillis(timeout_ms * 0.9)

    def _release_lock(self):
        if self._lock_fd is not None:
            try:
                fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(self._lock_fd)
                self._lock_fd = None

    def close(self):
        """Closes the port. Does NOT change torque state (servos keep holding).
        Safe to call more than once."""
        try:
            ph = self.ph
            if ph is not None:
                try:  # give the exclusive claim back explicitly (a tty keeps it while any fd is left)
                    fcntl.ioctl(ph.ser.fd, termios.TIOCNXCL)
                except Exception:
                    pass
                try:
                    ph.closePort()
                except Exception:
                    pass
        finally:
            self.ph = None
            self.pk = None
            self.exclusive = False
            self._release_lock()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # ---------------------------------------------------------------- plumbing
    def _require_open(self):
        if self.ph is None or self.pk is None:
            raise BusError(f"the bus on {self.port} is not open")

    def _flush(self):
        self.ph.is_using = False
        self.ph.ser.reset_input_buffer()

    def _quarantine(self):
        """Let the stragglers of a failed attempt land, then discard them."""
        time.sleep(QUARANTINE_S)
        self._flush()

    def _retry_cleanup(self):
        """Bookkeeping and clean-up after ONE failed attempt."""
        self.stats["retries"] += 1
        self._quarantine()

    def _io_error(self, what: str, e: Exception) -> "BusError":
        """An exception out of pyserial / termios / the SDK in the middle of a transaction:
        counted, the SDK's busy flag cleared, and turned into a BusError."""
        try:
            self.ph.is_using = False
        except Exception:
            pass
        self.stats["failures"] += 1
        return BusError(f"{what}: serial I/O failed: {type(e).__name__}: {e}")

    @staticmethod
    def _sid(joint) -> int:
        try:
            return IDS[joint]
        except (KeyError, TypeError):
            raise BusError(f"unknown joint {joint!r} (known: {', '.join(JOINTS)})") from None

    @staticmethod
    def _reg(reg) -> tuple:
        try:
            return REG[reg]
        except (KeyError, TypeError):
            raise BusError(f"unknown register {reg!r}") from None

    @classmethod
    def _joint_list(cls, joints) -> tuple:
        joints = (joints,) if isinstance(joints, str) else tuple(joints)
        for j in joints:
            cls._sid(j)
        if len(set(joints)) != len(joints):
            raise BusError(f"joint named twice in {joints}")
        return joints

    @staticmethod
    def _as_int(value, what: str) -> int:
        """int(value) for numbers; None, text, NaN and inf are refused with a BusError."""
        try:
            if isinstance(value, (str, bytes, bytearray)):
                raise TypeError
            return int(value)
        except (TypeError, ValueError, OverflowError):
            raise BusError(f"refusing {what}={value!r}: not a number") from None

    @classmethod
    def _check_value(cls, reg: str, value) -> int:
        if not isinstance(reg, str) or reg not in RANGE or REG[reg][0] < FIRST_RAM_ADDR:
            raise BusError(f"refusing to write {reg}: only RAM registers {sorted(WRITABLE)} are allowed")
        lo, hi = RANGE[reg]
        v = cls._as_int(value, reg)
        if not lo <= v <= hi:
            raise BusError(f"refusing {reg}={value}: outside [{lo}, {hi}]")
        return v

    @staticmethod
    def _field(block, base: int, reg: str) -> int:
        addr, size, sign = REG[reg]
        d = block[addr - base:addr - base + size]
        return _decode(d[0] if size == 1 else d[0] | (d[1] << 8), sign)

    def _transact(self, joint: str, tx: list, data_len: int, what: str):
        """One request -> one validated reply (retries included). Returns the data bytes."""
        self._require_open()
        last = None
        try:
            for attempt in range(self.retries):
                self._flush()
                self.stats["requests"] += 1
                rx, res, err = self.pk.txRxPacket(self.ph, list(tx))
                if res == scs.COMM_SUCCESS and (
                    len(rx) < 6 + data_len or rx[scs.PKT_ID] != tx[scs.PKT_ID] or rx[scs.PKT_LENGTH] != data_len + 2
                ):
                    res = scs.COMM_RX_CORRUPT
                if res == scs.COMM_SUCCESS:
                    self.status[joint] = err
                    if attempt:
                        self._quarantine()  # the answer to a failed attempt may still be on its way
                    return rx[5:5 + data_len]
                last = res
                self._retry_cleanup()
        except Exception as e:
            raise self._io_error(what, e) from None
        self.stats["failures"] += 1
        raise BusError(f"{what} failed after {self.retries} attempt(s) ({_comm(last)})")

    def read_block(self, joint: str, base: int, n: int):
        sid = self._sid(joint)
        if not (0 <= base <= 255 and 1 <= n <= 240):
            raise BusError(f"read_block({joint}, {base}, {n}): outside the control table")
        tx = [0, 0, sid, 4, scs.INST_READ, base, n, 0]
        return self._transact(joint, tx, n, f"read [{base}..{base + n - 1}] from {joint}")

    # ---------------------------------------------------------------- single
    def read(self, joint: str, reg: str) -> int:
        """Every single-servo read fetches the whole EEPROM block (0..39) or the whole RAM
        block (40..70): request shapes are unique, so a stale reply can only be an older
        answer to the same question, never another register."""
        base, n = EEPROM_BLOCK if self._reg(reg)[0] < FIRST_RAM_ADDR else RAM_BLOCK
        return self._field(self.read_block(joint, base, n), base, reg)

    def write(self, joint: str, reg: str, value: int):
        value = self._check_value(reg, value)  # refused before anything is put on the wire
        sid = self._sid(joint)
        addr, size, sign = REG[reg]
        raw = _encode(value, sign)
        data = [raw] if size == 1 else [raw & 0xFF, (raw >> 8) & 0xFF]
        tx = [0, 0, sid, size + 3, scs.INST_WRITE, addr] + data + [0]
        self._transact(joint, tx, 0, f"write {reg}={value} to {joint}")

    # ---------------------------------------------------------------- group
    def _sync_block(self, base: int, n: int, joints):
        self._require_open()
        joints = self._joint_list(joints)
        if not joints:
            return {}
        ids = [IDS[j] for j in joints]
        what = f"sync read [{base}..{base + n - 1}]"
        last = None
        try:
            for _ in range(self.retries):
                self._flush()
                self.stats["requests"] += 1
                out = {}
                res = self.pk.syncReadTx(self.ph, base, n, ids, len(ids))
                if res == scs.COMM_SUCCESS:
                    for j, sid in zip(joints, ids):
                        rx, res = self.pk.rxPacket(self.ph)
                        if res == scs.COMM_SUCCESS and (
                            len(rx) < 6 + n or rx[scs.PKT_ID] != sid or rx[scs.PKT_LENGTH] != n + 2
                        ):
                            res = scs.COMM_RX_CORRUPT
                        if res != scs.COMM_SUCCESS:
                            break
                        self.status[j] = rx[scs.PKT_ERROR]
                        out[j] = rx[5:5 + n]
                self.ph.is_using = False
                if res == scs.COMM_SUCCESS and len(out) == len(ids):
                    return out
                last = res
                self._retry_cleanup()
        except Exception as e:
            raise self._io_error(what, e) from None
        self.stats["fallbacks"] += 1
        try:
            return {j: self.read_block(j, base, n) for j in joints}
        except BusError as e:
            raise BusError(f"{what} failed after {self.retries} attempt(s) ({_comm(last)}); "
                           f"fallback to single reads: {e}") from None

    def sync_read(self, reg: str, joints=JOINTS) -> dict:
        addr, size, _ = self._reg(reg)
        joints = self._joint_list(joints)
        base, n = (addr, size) if reg == "Present_Position" else FEEDBACK_BLOCK
        if not (base <= addr and addr + size <= base + n):
            return {j: self.read(j, reg) for j in joints}
        return {j: self._field(b, base, reg) for j, b in self._sync_block(base, n, joints).items()}

    def sync_write(self, reg: str, values: dict):
        """One broadcast packet, NOT acknowledged by the servos. Every value is checked
        before anything is sent."""
        self._require_open()
        if not values:
            raise BusError(f"sync_write {reg}: no joints given")
        raws = {}
        for j, v in values.items():
            v = self._check_value(reg, v)
            raws[self._sid(j)] = _encode(v, REG[reg][2])
        addr, size, _ = REG[reg]
        g = scs.GroupSyncWrite(self.ph, self.pk, addr, size)
        for sid, raw in raws.items():
            g.addParam(sid, [raw & 0xFF] if size == 1 else [raw & 0xFF, (raw >> 8) & 0xFF])
        what = f"sync write {reg}"
        last = None
        try:
            self._flush()
            for _ in range(self.retries):
                self.stats["requests"] += 1
                res = g.txPacket()
                if res == scs.COMM_SUCCESS:
                    return
                last = res
                self.stats["retries"] += 1
                self.ph.is_using = False
                time.sleep(0.002)
        except Exception as e:
            raise self._io_error(what, e) from None
        self.stats["failures"] += 1
        raise BusError(f"{what} could not be transmitted in {self.retries} attempt(s) ({_comm(last)})")

    # ---------------------------------------------------------------- helpers
    def positions(self) -> dict:
        return self.sync_read("Present_Position")

    def feedback(self, joints=JOINTS) -> dict:
        """Position, load, current, status of all joints from ONE sync read (56..70)."""
        out = {}
        for j, b in self._sync_block(*FEEDBACK_BLOCK, joints).items():
            f = lambda reg: self._field(b, FEEDBACK_BLOCK[0], reg)  # noqa: E731
            out[j] = {"pos": f("Present_Position"), "load": f("Present_Load"), "current": f("Present_Current"),
                      "volt": f("Present_Voltage") / 10.0, "temp": f("Present_Temperature"), "status": f("Status")}
        return out

    def snapshot(self) -> dict:
        """State of every joint; all RAM fields of a joint come from the same instant."""
        out = {}
        for j in JOINTS:
            b = self.read_block(j, *RAM_BLOCK)
            f = lambda reg: self._field(b, RAM_BLOCK[0], reg)  # noqa: E731
            out[j] = {
                "pos": f("Present_Position"), "load": f("Present_Load"), "current": f("Present_Current"),
                "volt": f("Present_Voltage") / 10.0, "temp": f("Present_Temperature"),
                "torque": f("Torque_Enable"), "goal": f("Goal_Position"), "status": f("Status"),
                "acc": f("Acceleration"), "vel_cap": f("Goal_Velocity"), "lock": f("Lock"),
            }
        return out

    def _checked_limits(self, joint: str, eeprom) -> tuple:
        """(min, max) position limit out of an EEPROM block, after the plausibility checks."""
        lo, hi = self._field(eeprom, 0, "Min_Position_Limit"), self._field(eeprom, 0, "Max_Position_Limit")
        mode = self._field(eeprom, 0, "Operating_Mode")
        if mode != 0:
            raise BusError(f"{joint}: Operating_Mode is {mode}, not 0 (position mode)")
        if not (0 <= lo < hi <= 4095) or hi - lo < 300:
            raise BusError(f"{joint}: implausible position limits ({lo}, {hi})")
        return lo, hi

    def limits(self) -> dict:
        """Servo-stored position limits (the calibration actually in effect), validated."""
        return {j: self._checked_limits(j, self.read_block(j, *EEPROM_BLOCK)) for j in JOINTS}

    def faults(self) -> dict:
        """Decoded error bytes of the most recent replies (free: no extra bus traffic).
        Only joints whose latest reply carried an error bit are listed."""
        known = sum(ERR_NAMES)
        out = {}
        for j, s in self.status.items():
            if s:
                out[j] = [n for b, n in ERR_NAMES.items() if s & b]
                if s & ~known:
                    out[j].append(f"unknown_0x{s & ~known:02x}")
        return out

    def _torque_off_quietly(self, joints) -> list:
        """Torque_Enable := 0 on each joint, never raising. Returns the joints that did not confirm."""
        unconfirmed = []
        for j in joints:
            try:
                self.write(j, "Torque_Enable", 0)
            except BusError:
                unconfirmed.append(j)
        return unconfirmed

    def torque_on(self, joints=JOINTS, velocity: int = 400, acceleration: int = 20, max_goal_gap: int = 12) -> dict:
        """Enable torque without motion. Returns {"newly_enabled": [joints that were off]}.

        Idempotent: a joint that is already holding keeps its goal (gravity compensation,
        grasp squeeze) and is only written to where Lock / Acceleration / Goal_Velocity differ
        from what is asked for, so a redundant call sends no write at all.
        A joint whose torque is off gets Lock := 1 first (nothing written afterwards can be
        persisted), then the profile caps, then goal := present; goal, position and caps are
        read back before Torque_Enable := 1.

        Refused with nothing changed: velocity / acceleration 0 (= unlimited in the servo), a
        joint to be enabled that is not in position mode, or that stands more than max_goal_gap
        ticks outside its own position limits (the servo clamps the goal to them and would move
        there by itself). If anything fails after the first goal write, torque is switched back
        off on the joints this call was enabling; joints that were holding are never touched."""
        joints = self._joint_list(joints)
        gap = self._as_int(max_goal_gap, "max_goal_gap")
        want = {"Lock": 1, "Acceleration": self._check_value("Acceleration", acceleration),
                "Goal_Velocity": self._check_value("Goal_Velocity", velocity)}
        if want["Acceleration"] < 1 or want["Goal_Velocity"] < 1:
            raise BusError(f"torque_on(velocity={velocity}, acceleration={acceleration}): 0 means 'unlimited' "
                           f"in the servo; refusing to enable torque without the in-servo caps (nothing was changed)")
        lim = {}
        try:
            ram = {j: self.read_block(j, *RAM_BLOCK) for j in joints}
            new = [j for j in joints if self._field(ram[j], RAM_BLOCK[0], "Torque_Enable") != 1]
            for j in new:
                lo, hi = lim[j] = self._checked_limits(j, self.read_block(j, *EEPROM_BLOCK))
                pos = self._field(ram[j], RAM_BLOCK[0], "Present_Position")
                if not lo - gap <= pos <= hi + gap:
                    raise BusError(f"{j}: position {pos} is more than {gap} ticks outside the servo's own limits "
                                   f"{lo}..{hi}; with torque on it would move to the limit by itself. Move the "
                                   f"joint inside its limits by hand")
        except BusError as e:
            raise BusError(f"torque_on refused, nothing was changed: {e}") from None
        try:
            for j in joints:
                for reg, val in want.items():
                    if j in new or self._field(ram[j], RAM_BLOCK[0], reg) != val:
                        self.write(j, reg, val)
        except BusError as e:
            raise BusError(f"torque_on aborted: {e}. Torque state not changed (Lock / Acceleration / "
                           f"Goal_Velocity may already be written on some joints)") from None
        if not new:
            return {"newly_enabled": []}
        try:
            pos = self.sync_read("Present_Position", new)
            for j in new:
                self.write(j, "Goal_Position", max(lim[j][0], min(lim[j][1], pos[j])))
            for j in new:
                b = self.read_block(j, *RAM_BLOCK)
                goal, now = self._field(b, RAM_BLOCK[0], "Goal_Position"), self._field(b, RAM_BLOCK[0], "Present_Position")
                if abs(goal - now) > gap:
                    raise BusError(f"{j}: goal {goal} is {goal - now:+d} ticks from position {now} after goal:=present")
                for reg, val in want.items():
                    if self._field(b, RAM_BLOCK[0], reg) != val:
                        raise BusError(f"{j}: {reg} reads back {self._field(b, RAM_BLOCK[0], reg)} instead of {val}")
            for j in new:
                self.write(j, "Torque_Enable", 1)
        except BusError as e:
            unconfirmed = self._torque_off_quietly(new)
            msg = f"torque_on aborted: {e}. Torque switched back OFF on {', '.join(new)}"
            if unconfirmed:
                msg += f" (NOT confirmed for {', '.join(unconfirmed)})"
            held = [j for j in joints if j not in new]
            if held:
                msg += f"; {', '.join(held)} keep holding as before"
            raise BusError(msg) from None
        return {"newly_enabled": new}

    def torque_off(self, joints=JOINTS, restore_profile: bool = True):
        """Torque off. Also puts the profile registers back to their power-up values so a
        later LeRobot session is not silently speed-capped by what this harness wrote.

        Best effort per joint: a servo that does not answer never keeps the others from
        being switched off. Raises BusError afterwards, naming what failed."""
        joints = self._joint_list(joints)
        problems, retry = [], []
        for j in joints:
            try:
                if restore_profile:
                    self.write(j, "Goal_Velocity", 0)
                    self.write(j, "Acceleration", 0)
                self.write(j, "Torque_Enable", 0)
            except BusError as e:
                problems.append(str(e))
                retry.append(j)
        still_on = self._torque_off_quietly(retry)  # second chance, torque only
        if problems:
            off = [j for j in joints if j not in still_on]
            raise BusError(f"torque_off incomplete: {'; '.join(problems)}. Torque is off on "
                           f"{', '.join(off) or 'no joint'}"
                           + (f"; NOT confirmed off on {', '.join(still_on)}" if still_on else "")
                           + (" (profile registers of the failed joints may not be restored)" if restore_profile else ""))

    def set_p_gain(self, joint: str, p: int) -> bool:
        """Set the position-loop P gain for this power cycle only. Returns True if the gain
        was written, False if the servo already had it (then no write is sent at all).

        P_Coefficient lives in the EEPROM area, but with Lock == 1 (checked here) the servo
        applies the value in RAM and does not save it, so the stored configuration is not
        changed and a power cycle restores it. This is the single exception to the "RAM
        registers only" rule; the range is limited to sane values. (LeRobot itself rewrites
        P=16 persistently on every connect.)"""
        lo, hi = P_GAIN_RANGE
        value = self._as_int(p, "P gain")
        if value != p:
            raise BusError(f"refusing P gain={p!r}: not a whole number")
        if not lo <= value <= hi:
            raise BusError(f"P gain {p} outside the allowed {lo}..{hi}")
        sid = self._sid(joint)
        current = self.read(joint, "P_Coefficient")
        if current == value:
            return False
        if self.read(joint, "Lock") != 1:
            raise BusError(f"{joint}: Lock is not 1; refusing a write that would persist to EEPROM "
                           f"(P gain stays {current})")
        tx = [0, 0, sid, 4, scs.INST_WRITE, REG["P_Coefficient"][0], value, 0]
        self._transact(joint, tx, 0, f"write P_Coefficient={value} to {joint}")
        back = self.read(joint, "P_Coefficient")
        if back != value:
            raise BusError(f"{joint}: P gain did not read back as {value} (the servo reports {back})")
        return True
