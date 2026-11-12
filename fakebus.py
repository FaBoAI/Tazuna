"""Six simulated Feetech STS3215 servos behind a pseudo-terminal: the test rig for servo_bus.

NO HARDWARE. The only device this module ever creates or opens is its own pty pair
(/dev/pts/N); it never touches /dev/ttyACM*, the camera or the real bus lock file.

The master end of the pty is served by a thread that parses Feetech protocol-0 instruction
packets (PING / READ / WRITE / SYNC_READ / SYNC_WRITE) and answers like the six servos of
the SO-101 follower. The slave end is an ordinary tty path that pyserial / scservo_sdk can
open, so the REAL servo_bus.Bus - open() with its flock, port-user scan and exclusive open
included - runs unmodified against it:

    fake = FakeBus()
    bus = fake.bus().open()          # servo_bus.Bus(port=fake.port, lock_path=fake.lock_path)
    ...
    bus.close(); fake.close()

What is modelled
- The control table of each servo as a RAM shadow (`mem`, what READ returns) plus a separate
  stored image of the EEPROM area (`eeprom`, addresses 0..39, what a power cycle restores).
  `power_cycle()` reloads the shadow from the stored image and resets the RAM registers.
- Lock (55). lock_model="documented": a write below address 40 always changes the RAM shadow
  (seen on the real arm: P read back 32 and the load doubled with Lock = 1) and is copied to
  the stored image only while Lock == 0. lock_model="ignored" is the pessimistic firmware in
  which every such write is stored (what the hardware check of set_p_gain has to exclude).
- Torque_Enable = 128 re-zeroes the encoder offset (recorded in `rezero_events`).
- auto_torque_on_goal=True: a Goal_Position write switches torque on (reported for real
  STS3215 firmware by third parties; not verified on this arm, so tests run both ways).
- Simple motion: a torqued servo moves towards its goal (clamped to its own position limits,
  plus an optional standing error `sag`) at the Goal_Velocity cap; `blocker` stops a joint
  (object between the jaws); `drift` moves a joint whose torque is off (arm sagging);
  `frozen = True` switches the motion model off.
- Fault injection, see FakeBus.rule(): dead servo, dropped / corrupt / truncated / late /
  duplicated / misaddressed replies, line noise, status (error) byte, servo reset, and a REAL
  host stall (SIGSTOP of a client process started with spawn_client()).
  inject_host_stall() models a host stall for a Bus living in this process.

The register table below is typed in independently of servo_bus.REG on purpose (vendor
memory map), so a wrong address in servo_bus shows up as a test failure.
"""

from __future__ import annotations

import heapq
import os
import pty
import select
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import tty

JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")
IDS = {name: i + 1 for i, name in enumerate(JOINTS)}
# the real arm (DESIGN.md section 3): parked pose and servo-stored limits, raw ticks
REST_RAW = {"shoulder_pan": 1999, "shoulder_lift": 955, "elbow_flex": 3025, "wrist_flex": 2820,
            "wrist_roll": 1983, "gripper": 2184}
LIMITS = {"shoulder_pan": (864, 3418), "shoulder_lift": (931, 3356), "elbow_flex": (775, 3064),
          "wrist_flex": (835, 3266), "wrist_roll": (0, 4095), "gripper": (1998, 3648)}
HOMING = {"shoulder_pan": 1411, "shoulder_lift": 950, "elbow_flex": -854, "wrist_flex": -1974,
          "wrist_roll": 1783, "gripper": 1857}

# name -> (address, size, sign bit or None); Feetech STS/SMS memory map
TABLE = {
    "Firmware_Major": (0, 1, None),
    "Firmware_Minor": (1, 1, None),
    "Model": (3, 2, None),
    "ID": (5, 1, None),
    "Baud_Rate": (6, 1, None),
    "Return_Delay_Time": (7, 1, None),
    "Min_Position_Limit": (9, 2, None),
    "Max_Position_Limit": (11, 2, None),
    "Max_Temperature_Limit": (13, 1, None),
    "Max_Voltage_Limit": (14, 1, None),
    "Min_Voltage_Limit": (15, 1, None),
    "Max_Torque_Limit": (16, 2, None),
    "Phase": (18, 1, None),
    "P_Coefficient": (21, 1, None),
    "D_Coefficient": (22, 1, None),
    "I_Coefficient": (23, 1, None),
    "Protection_Current": (28, 2, None),
    "Homing_Offset": (31, 2, 11),
    "Operating_Mode": (33, 1, None),
    "Protective_Torque": (34, 1, None),
    "Protection_Time": (35, 1, None),
    "Overload_Torque": (36, 1, None),
    "Torque_Enable": (40, 1, None),
    "Acceleration": (41, 1, None),
    "Goal_Position": (42, 2, 15),
    "Goal_Time": (44, 2, None),
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
EEPROM_SIZE = 40  # addresses 0..39 are the EEPROM area
READ_ONLY = range(56, 71)  # feedback registers: a write is acknowledged and ignored
MEM_SIZE = 128
NO_LOAD_SPEED = 3400.0  # ticks/s when Goal_Velocity is 0 ("maximum")

INST_PING, INST_READ, INST_WRITE, INST_SYNC_READ, INST_SYNC_WRITE = 1, 2, 3, 0x82, 0x83
BROADCAST = 0xFE
STAMP_MOD = 30000


def pkt(sid: int, body) -> bytes:
    """body = [instruction or error byte, parameters...] -> complete packet (header, length, checksum)."""
    p = [0xFF, 0xFF, sid, len(body) + 1] + [int(b) & 0xFF for b in body]
    p.append(~sum(p[2:]) & 0xFF)
    return bytes(p)


def hexs(b) -> str:
    return " ".join(f"{x:02X}" for x in b)


def enc(value: int, sign_bit) -> int:
    """Sign-magnitude encoding as the servo stores it."""
    if sign_bit is None:
        return int(value)
    return (abs(int(value)) | (1 << sign_bit)) if value < 0 else int(value)


def dec(raw: int, sign_bit) -> int:
    if sign_bit is None:
        return raw
    mag = raw & ((1 << sign_bit) - 1)
    return -mag if raw & (1 << sign_bit) else mag


def _sid(joint_or_id) -> int:
    return joint_or_id if isinstance(joint_or_id, int) else IDS[joint_or_id]


class FakeBus:
    def __init__(self, reply_delay: float = 0.0003, lock_model: str = "documented",
                 auto_torque_on_goal: bool = False, keep_slave_open: bool = True, stamp: bool = False,
                 start: bool = True):
        """reply_delay: seconds between a request and its reply (sync read: per reply slot).
        keep_slave_open: keep our own descriptor of the slave side (the pty then survives the
            client closing it). Must be False when the client is ANOTHER process: Bus.open()
            refuses a port that some other process has open.
        stamp: every reply that covers Present_Position carries the request counter `seq`
            (mod STAMP_MOD) in that field, so a test can tell a fresh reply from a stale one."""
        if lock_model not in ("documented", "ignored"):
            raise ValueError("lock_model must be 'documented' or 'ignored'")
        self.m, s = pty.openpty()
        tty.setraw(s)
        self.port = os.ttyname(s)
        if not self.port.startswith("/dev/pts/"):
            raise RuntimeError(f"fake bus got an unexpected tty name {self.port}")
        self.s = s
        if not keep_slave_open:
            os.close(s)
            self.s = None
        self._tmp = tempfile.mkdtemp(prefix="fakebus_")
        self.lock_path = os.path.join(self._tmp, "bus.lock")  # scratch flock file, never the real one
        self.reply_delay = reply_delay
        self.lock_model = lock_model
        self.auto_torque_on_goal = auto_torque_on_goal
        self.stamp = stamp
        self.mem, self.eeprom, self.pos, self.fault = {}, {}, {}, {}
        self.sag = {}        # sid -> standing error (ticks, + = towards higher ticks) while torqued
        self.drift = {}      # sid -> ticks/s while torque is OFF (arm sagging / pushed by hand)
        self.blocker = None  # fn(sid, candidate_pos) -> True if that position cannot be reached
        self.frozen = False  # True: no motion model at all (register-level tests)
        for j in JOINTS:
            sid = IDS[j]
            self.eeprom[sid] = self._factory_eeprom(j)
            self.mem[sid] = bytearray(MEM_SIZE)
            self.mem[sid][0:EEPROM_SIZE] = self.eeprom[sid]
            self.pos[sid] = float(REST_RAW[j])
            self.fault[sid] = 0  # error bits the servo reports in every reply and in Status (65)
            self._power_up(sid)
        self.rules = []          # fault injection, see rule()
        self.fault_hook = None   # fn(req) -> action dict or None, asked before the rules (random faults)
        self.requests = []       # (t, raw instruction packet) for every well-formed packet received
        self.bad_requests = []   # (t, bytes) with a wrong checksum
        self.replies = []        # (t, raw bytes put on the wire)
        self.writes = []         # (t, sid, addr, [bytes], "WRITE" | "SYNC_WRITE") executed by a servo
        self.ee_writes = []      # (t, sid, addr, value, lock_at_write, stored) for addresses < 40
        self.goal_log = []       # (t, sid, goal) for every executed Goal_Position write
        self.rezero_events = []  # (t, sid) Torque_Enable = 128
        self.stall_log = []      # (t_stop, t_cont) of client-process stalls
        self.seq = 0             # well-formed requests parsed so far
        self.client_pid = None
        self.thread_error = None
        self._held = []          # [release predicate, requests still to skip, bytes] (hold_until rules)
        self._pending = []       # heap of (due, n, bytes)
        self._timers = []        # heap of (due, n, fn)
        self._n = 0
        self._idle = 0
        self._stop = False
        self._closed = False
        self.t0 = time.perf_counter()
        self.th = threading.Thread(target=self._run, name="fakebus", daemon=True)
        if start:
            self.th.start()

    # ------------------------------------------------------------------ servo state
    @staticmethod
    def _factory_eeprom(joint: str) -> bytearray:
        sid = IDS[joint]
        grip = joint == "gripper"
        m = bytearray(EEPROM_SIZE)
        m[0], m[1] = 3, 10                                   # firmware 3.10
        m[3:5] = (777).to_bytes(2, "little")                 # STS3215
        m[5] = sid
        m[6], m[7], m[8] = 0, 0, 1                           # 1 Mbps, return delay 0, status level 1
        m[9:11] = LIMITS[joint][0].to_bytes(2, "little")
        m[11:13] = LIMITS[joint][1].to_bytes(2, "little")
        m[13], m[14], m[15] = 70, 80, 40                     # max temp, max / min voltage
        m[16:18] = (500 if grip else 1000).to_bytes(2, "little")
        m[18], m[19], m[20] = 12, 44, 47                     # phase, unloading / LED alarm conditions
        m[21], m[22], m[23] = 16, 32, 0                      # P, D, I as LeRobot's configure() stores them
        m[24], m[26], m[27] = 16, 1, 1                       # startup force, dead zones
        m[28:30] = (250 if grip else 500).to_bytes(2, "little")
        m[30] = 1
        m[31:33] = enc(HOMING[joint], 11).to_bytes(2, "little")
        m[33] = 0                                            # position mode
        m[34], m[35], m[36] = 20, 200, (25 if grip else 80)  # protective torque, time, overload torque
        m[38] = 200
        return m

    def _store_pos(self, sid: int):
        v = int(round(self.pos[sid]))
        self.mem[sid][56:58] = enc(max(-32767, min(32767, v)), 15).to_bytes(2, "little")

    def _power_up(self, sid: int):
        """RAM registers as after power-on. The shaft stays where it is."""
        m = self.mem[sid]
        m[40] = 0                     # torque off
        m[41] = 0                     # acceleration: 0 = no ramp
        self._store_pos(sid)
        m[42:44] = m[56:58]           # goal = present
        m[44:48] = b"\x00\x00\x00\x00"  # goal time, goal velocity (0 = maximum)
        m[48:50] = m[16:18]           # torque limit <- max torque limit
        m[55] = 1                     # Lock = 1 at power-up (measured on the real arm)
        m[58:62] = b"\x00\x00\x00\x00"  # velocity, load
        m[62], m[63] = 49, 25 + sid   # 4.9 V, temperature
        m[65], m[66] = self.fault[sid], 0
        m[69:71] = b"\x00\x00"

    def power_cycle(self, joint_or_id=None):
        """Servo power off / on (all servos, or one): the EEPROM area is reloaded from the
        stored image, the RAM registers return to their power-up values, faults clear."""
        for sid in ([_sid(joint_or_id)] if joint_or_id is not None else list(self.mem)):
            self.mem[sid][0:EEPROM_SIZE] = self.eeprom[sid]
            self.fault[sid] = 0
            self._power_up(sid)

    reset_servo = power_cycle  # a brown-out reset of one servo looks the same

    def lerobot_disconnect(self):
        """What LeRobot's default disconnect() -> disable_torque() leaves: Torque_Enable=0, Lock=0."""
        for m in self.mem.values():
            m[40] = 0
            m[55] = 0

    def reg(self, joint_or_id, name: str) -> int:
        """Decoded value of a register in the RAM shadow (the truth the servo would report)."""
        addr, size, sign = TABLE[name]
        return dec(int.from_bytes(self.mem[_sid(joint_or_id)][addr:addr + size], "little"), sign)

    def stored(self, joint_or_id, name: str) -> int:
        """Decoded value in the stored EEPROM image (what a power cycle brings back)."""
        addr, size, sign = TABLE[name]
        if addr >= EEPROM_SIZE:
            raise KeyError(f"{name} is not in the EEPROM area")
        return dec(int.from_bytes(self.eeprom[_sid(joint_or_id)][addr:addr + size], "little"), sign)

    def poke(self, joint_or_id, name: str, value: int, store: bool = True):
        """Set a register behind the servo's back (test setup). EEPROM-area registers are also
        put into the stored image unless store=False."""
        addr, size, sign = TABLE[name]
        sid = _sid(joint_or_id)
        raw = enc(value, sign).to_bytes(size, "little")
        self.mem[sid][addr:addr + size] = raw
        if addr < EEPROM_SIZE and store:
            self.eeprom[sid][addr:addr + size] = raw
        if name == "Present_Position":
            self.pos[sid] = float(value)

    def set_pos(self, joint_or_id, ticks: float):
        sid = _sid(joint_or_id)
        self.pos[sid] = float(ticks)
        self._store_pos(sid)

    def set_fault(self, joint_or_id, bits: int):
        """Error bits (1 voltage, 2 sensor, 4 overheat, 8 overcurrent, 32 overload) the servo
        reports from now on in the error byte of every reply and in Status (65)."""
        sid = _sid(joint_or_id)
        self.fault[sid] = bits & 0x7F
        self.mem[sid][65] = bits & 0x7F

    def view(self) -> dict:
        ids = sorted(self.mem)
        g = lambda name: [self.reg(i, name) for i in ids]  # noqa: E731
        return {"torque": g("Torque_Enable"), "lock": g("Lock"), "acc": g("Acceleration"),
                "vel": g("Goal_Velocity"), "goal": g("Goal_Position"),
                "pos": [int(round(self.pos[i])) for i in ids], "P": g("P_Coefficient"),
                "P_stored": [self.stored(i, "P_Coefficient") for i in ids]}

    def calib_bytes(self) -> dict:
        """Stored bytes that make up the calibration: limits (9..12), homing offset (31..32), mode (33)."""
        return {i: bytes(self.eeprom[i][9:13] + self.eeprom[i][31:34]).hex() for i in sorted(self.eeprom)}

    # ------------------------------------------------------------------ test-side helpers
    def now(self) -> float:
        return time.perf_counter() - self.t0

    def bus(self, **kw):
        """The REAL servo_bus.Bus bound to this fake (not opened yet): pty port + scratch lock file."""
        import servo_bus
        kw.setdefault("lock_path", self.lock_path)
        return servo_bus.Bus(port=self.port, **kw)

    def rule(self, match, count: int = 1, **action) -> dict:
        """Add a fault-injection rule and return it (its "hits" counter tells how often it fired).

        match(req) -> bool with req = dict(id, inst, addr, n[, slot]); the first rule that
        matches and still has count left decides what happens to that request:
          dead=True      the servo neither executes nor answers (servo off the bus / request lost)
          ignore=True    acknowledged but NOT executed (a write that does not take)
          drop=True      executed, but the reply is lost
          corrupt=True   reply with a wrong checksum
          truncate=k     only the first k bytes of the reply are sent
          delay=s        reply s seconds late (number, or a function returning one)
          err=byte       error byte of this reply
          wrong_id=k     the reply carries servo id k (well-formed packet)
          dup=s          the reply is sent a second time s seconds after the first
          garbage=bytes  line noise put on the wire right before the reply
          reset=True     the servo resets (RAM back to power-up values) right after answering
          call=fn        fn(req) is called in the servo thread before the request is executed
          hold_until=fn  the reply is kept back and put on the wire at the very moment a LATER
                         request with fn(raw_packet) true arrives, i.e. right after the client
                         flushed its input for that request (the worst-case straggler);
                         hold_skip=k lets k such requests pass first
          stall=s        freeze the client PROCESS (SIGSTOP) for s seconds, starting stall_after
                         seconds from now (needs spawn_client(); the servos keep answering on time)"""
        r = dict(match=match, count=count, hits=0, **action)
        self.rules.append(r)
        return r

    def mark(self) -> int:
        """Index into `requests` for "everything sent from now on"."""
        self.drain()
        return len(self.requests)

    def sent(self, since: int = 0, inst=None, sid=None):
        """Request packets received from index `since` on, optionally filtered."""
        self.drain()
        return [p for _, p in self.requests[since:]
                if (inst is None or p[4] == inst) and (sid is None or p[2] == sid)]

    def written(self, since_t: float = 0.0) -> list:
        """(joint, register name, value, via) for every write executed after time since_t."""
        by_addr = {(a, n): (name, sg) for name, (a, n, sg) in TABLE.items()}
        out = []
        for t, sid, addr, data, via in self.writes:
            if t < since_t:
                continue
            name, sg = by_addr.get((addr, len(data)), (f"addr{addr}", None))
            out.append((JOINTS[sid - 1], name, dec(int.from_bytes(bytes(data), "little"), sg), via))
        return out

    def drain(self, timeout: float = 2.0):
        """Wait until the servo thread has processed everything written to the port so far
        (needed after a broadcast, which no reply confirms)."""
        target = self._idle + 2
        t_end = time.perf_counter() + timeout
        while self._idle < target and self.th.is_alive() and time.perf_counter() < t_end:
            time.sleep(0.0005)
        if self.thread_error is not None:
            raise RuntimeError(f"fake bus thread died: {self.thread_error!r}")

    def spawn_client(self, argv, **kw) -> subprocess.Popen:
        """Start a client PROCESS that will open self.port; `stall` rules freeze it with SIGSTOP."""
        if self.s is not None:
            raise RuntimeError("use FakeBus(keep_slave_open=False) for a client in another process")
        kw.setdefault("stdout", subprocess.PIPE)
        kw.setdefault("stderr", subprocess.PIPE)
        kw.setdefault("text", True)
        p = subprocess.Popen(argv, **kw)
        self.client_pid = p.pid
        return p

    def stall_client(self, dur: float):
        """Freeze the client process for dur seconds (a real host stall)."""
        pid = self.client_pid
        if pid is None:
            self.thread_error = RuntimeError("stall requested but no client process (spawn_client)")
            return
        t = self.now()
        try:
            os.kill(pid, signal.SIGSTOP)
        except ProcessLookupError:
            return

        def cont():
            try:
                os.kill(pid, signal.SIGCONT)
            except ProcessLookupError:
                pass
            self.stall_log.append((t, self.now()))

        self._n += 1
        heapq.heappush(self._timers, (time.perf_counter() + dur, self._n, cont))

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop = True
        if self.th.is_alive():
            self.th.join(2.0)
        if self.client_pid is not None:
            try:
                os.kill(self.client_pid, signal.SIGCONT)  # never leave a client frozen
            except (ProcessLookupError, PermissionError):
                pass
        for fd in (self.m, self.s):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        shutil.rmtree(self._tmp, ignore_errors=True)

    def hang_up(self):
        """Close the master side while a client is connected: what unplugging the USB adapter
        looks like to the client (reads and writes fail with EIO)."""
        self._stop = True
        if self.th.is_alive():
            self.th.join(2.0)
        os.close(self.m)
        self.m = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------ servo thread
    def _run(self):
        buf = bytearray()
        last = time.perf_counter()
        try:
            while not self._stop:
                now = time.perf_counter()
                self._step(now - last)
                last = now
                while self._timers and self._timers[0][0] <= now:
                    heapq.heappop(self._timers)[2]()
                while self._pending and self._pending[0][0] <= now:
                    data = heapq.heappop(self._pending)[2]
                    try:
                        os.write(self.m, data)
                    except OSError:
                        pass
                    self.replies.append((self.now(), data))
                wait = 0.002
                for h in (self._pending, self._timers):
                    if h:
                        wait = max(0.0, min(wait, h[0][0] - time.perf_counter()))
                r, _, _ = select.select([self.m], [], [], wait)
                if not r:
                    self._idle += 1
                    continue
                try:
                    chunk = os.read(self.m, 4096)
                except OSError:  # EIO while no client has the slave side open
                    time.sleep(0.001)
                    self._idle += 1
                    continue
                buf += chunk
                self._parse(buf)
        except Exception as e:  # noqa: BLE001 - reported through thread_error / drain()
            self.thread_error = e

    def _step(self, dt: float):
        if dt <= 0 or self.frozen:
            return
        for sid, m in self.mem.items():
            if not m[40]:
                v = self.drift.get(sid, 0.0)
                if v:
                    self.pos[sid] += v * dt
                    self._store_pos(sid)
                continue
            goal = dec(int.from_bytes(m[42:44], "little"), 15)
            lo, hi = int.from_bytes(m[9:11], "little"), int.from_bytes(m[11:13], "little")
            if lo or hi:  # the servo clamps the goal to its own limits (0 / 0 = multi-turn: no clamp)
                goal = max(lo, min(hi, goal))
            target = goal + self.sag.get(sid, 0.0)
            cap = float(int.from_bytes(m[46:48], "little") & 0x7FFF) or NO_LOAD_SPEED
            step = max(-cap * dt, min(cap * dt, target - self.pos[sid]))
            cand = self.pos[sid] + step
            if self.blocker is not None and self.blocker(sid, cand) and not self.blocker(sid, self.pos[sid]):
                continue
            self.pos[sid] = cand
            self._store_pos(sid)

    def _parse(self, buf: bytearray):
        while True:
            i = buf.find(b"\xff\xff")
            if i < 0:
                del buf[:max(0, len(buf) - 1)]
                return
            if i:
                del buf[:i]
            if len(buf) < 4:
                return
            total = buf[3] + 4
            if len(buf) < total:
                return
            p = bytes(buf[:total])
            del buf[:total]
            if (~sum(p[2:-1]) & 0xFF) != p[-1]:
                self.bad_requests.append((self.now(), p))
                continue
            self.seq += 1
            self.requests.append((self.now(), p))
            for h in list(self._held):  # stragglers that were waiting for this request
                if h[0](p):
                    if h[1] > 0:
                        h[1] -= 1
                        continue
                    self._held.remove(h)
                    try:
                        os.write(self.m, h[2])
                    except OSError:
                        pass
                    self.replies.append((self.now(), h[2]))
            self._dispatch(p)

    def _action(self, req: dict) -> dict:
        if self.fault_hook is not None:
            act = self.fault_hook(req)
            if act:
                return act
        for r in self.rules:
            if r["count"] > 0 and r["match"](req):
                r["count"] -= 1
                r["hits"] += 1
                if "call" in r:
                    r["call"](req)
                if "stall" in r:
                    dur = r["stall"]
                    after = r.get("stall_after", 0.0)
                    after = after() if callable(after) else after
                    self._n += 1
                    heapq.heappush(self._timers, (time.perf_counter() + after, self._n,
                                                  lambda: self.stall_client(dur)))
                return r
        return {}

    def _reply(self, sid: int, data, req: dict, act: dict, slot: int = 0):
        if act.get("drop"):
            return
        data = bytearray(data)
        if self.stamp and req["inst"] in (INST_READ, INST_SYNC_READ):
            a, n = req["addr"], req["n"]
            if a <= 56 and a + n >= 58:
                data[56 - a:58 - a] = (self.seq % STAMP_MOD).to_bytes(2, "little")
        p = bytearray(pkt(act.get("wrong_id", sid), [act.get("err", self.fault[sid])] + list(data)))
        if act.get("corrupt"):
            p[-1] ^= 0x55
        if "truncate" in act:
            p = p[:act["truncate"]]
        if "hold_until" in act:
            self._held.append([act["hold_until"], act.get("hold_skip", 0), bytes(p)])
            return
        delay = act.get("delay", 0.0)
        delay = delay() if callable(delay) else delay
        due = time.perf_counter() + self.reply_delay * (slot + 1) + delay
        out = bytes(act.get("garbage", b"")) + bytes(p)
        self._n += 1
        heapq.heappush(self._pending, (due, self._n, out))
        if "dup" in act:
            self._n += 1
            heapq.heappush(self._pending, (due + act["dup"], self._n, bytes(p)))
        if act.get("reset"):
            self._n += 1
            heapq.heappush(self._timers, (due, self._n, lambda: self.power_cycle(sid)))

    def _write_mem(self, sid: int, addr: int, data, via: str):
        m = self.mem[sid]
        data = list(data)
        self.writes.append((self.now(), sid, addr, data, via))
        for k, b in enumerate(data):
            a = addr + k
            if a >= MEM_SIZE or a in READ_ONLY:
                continue
            if a < EEPROM_SIZE:
                stored = m[55] == 0 or self.lock_model == "ignored"
                self.ee_writes.append((self.now(), sid, a, b, m[55], stored))
                if stored:
                    self.eeprom[sid][a] = b
                m[a] = b
            elif a == 40 and b == 128:
                # "set the current position as 2048": rewrites the encoder offset (calibration lost)
                self.rezero_events.append((self.now(), sid))
                self.pos[sid] = 2048.0
                self._store_pos(sid)
                m[31:33] = b"\x00\x00"
                if m[55] == 0 or self.lock_model == "ignored":
                    self.eeprom[sid][31:33] = b"\x00\x00"
            else:
                m[a] = b
        if addr <= 42 < addr + len(data):
            self.goal_log.append((self.now(), sid, dec(int.from_bytes(m[42:44], "little"), 15)))
            if self.auto_torque_on_goal:
                m[40] = 1

    def _dispatch(self, p: bytes):
        sid, inst, par = p[2], p[4], p[5:-1]
        if inst == INST_PING and sid in self.mem:
            req = dict(id=sid, inst=inst, addr=None, n=0)
            act = self._action(req)
            if not act.get("dead"):
                self._reply(sid, [], req, act)
        elif inst == INST_READ and sid in self.mem and len(par) >= 2:
            addr, n = par[0], par[1]
            req = dict(id=sid, inst=inst, addr=addr, n=n)
            act = self._action(req)
            if not act.get("dead"):
                self._reply(sid, self.mem[sid][addr:addr + n], req, act)
        elif inst == INST_WRITE and sid in self.mem and len(par) >= 2:
            addr, data = par[0], par[1:]
            req = dict(id=sid, inst=inst, addr=addr, n=len(data))
            act = self._action(req)
            if act.get("dead"):
                return
            if not act.get("ignore"):
                self._write_mem(sid, addr, data, "WRITE")
            self._reply(sid, [], req, act)
        elif inst == INST_SYNC_READ and sid == BROADCAST and len(par) >= 2:
            addr, n, ids = par[0], par[1], par[2:]
            slot = 0
            for k in ids:
                if k in self.mem:
                    req = dict(id=k, inst=inst, addr=addr, n=n, slot=slot)
                    act = self._action(req)
                    if not act.get("dead"):
                        self._reply(k, self.mem[k][addr:addr + n], req, act, slot)
                    slot += 1  # a silent servo leaves its reply slot empty
        elif inst == INST_SYNC_WRITE and sid == BROADCAST and len(par) >= 2:
            addr, n, rest = par[0], par[1], par[2:]
            for k in range(0, len(rest), n + 1):
                if rest[k] in self.mem:
                    act = self._action(dict(id=rest[k], inst=inst, addr=addr, n=n))
                    if not (act.get("dead") or act.get("ignore")):
                        self._write_mem(rest[k], addr, rest[k + 1:k + 1 + n], "SYNC_WRITE")


def inject_host_stall(bus, when, stall_s: float = 0.060, where: str = "before_read", count: int = 1) -> dict:
    """Model the host process being descheduled for stall_s during the first `count`
    transactions of `bus` (a servo_bus.Bus living in THIS process) whose instruction packet
    satisfies when(txpacket). The simulated servos keep answering on time meanwhile.

    where="before_read":   right after the request went out, before the reply is read.
    where="mid_reply":     after the first read of the reply (its 6-byte head) returned and
                           before the rest is read.
    where="timeout_check": just before the SDK looks at its timeout clock (the weak spot of
                           the SDK's original polling loop; with blocking reads the clock is
                           only consulted when a read came back short).
    Returns a dict whose "fired" entry counts the stalls that happened."""
    if where not in ("before_read", "mid_reply", "timeout_check"):
        raise ValueError(where)
    ph, pk = bus.ph, bus.pk
    orig_tx, orig_read, orig_to = pk.txPacket, ph.readPort, ph.isPacketTimeout
    state = {"armed": False, "fired": 0, "reads": 0}

    def stall():
        state["armed"] = False
        state["fired"] += 1
        time.sleep(stall_s)

    def tx(port, txpacket):
        res = orig_tx(port, txpacket)
        state["armed"] = state["fired"] < count and bool(when(txpacket))
        state["reads"] = 0
        return res

    def read(n):
        if state["armed"]:
            state["reads"] += 1
            if (where == "before_read" and state["reads"] == 1) or (where == "mid_reply" and state["reads"] == 2):
                stall()
        return orig_read(n)

    def is_timeout():
        if where == "timeout_check" and state["armed"]:
            stall()
        return orig_to()

    pk.txPacket = tx
    ph.readPort = read
    ph.isPacketTimeout = is_timeout
    return state
