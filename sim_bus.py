"""A simulated servo bus with the same interface as servo_bus.Bus (no hardware, no serial port).

Used to exercise the motion / safety logic without touching the robot. The model is
deliberately simple: each torqued servo moves towards its goal with the speed cap of its
Goal_Velocity register and a small constant gravity-like droop on shoulder_lift / elbow_flex,
plus an optional "blocker" (a rigid obstacle: a joint cannot advance into it).
`phys_bus.PhysBus` derives a torque-level model (gravity, friction, compliant contact) from
this class.

What is mirrored from servo_bus.Bus (as of its 02:32 version; keep in sync when that file changes):
- read / write / sync_read / sync_write with the same register names; writes are refused
  outside the RAM registers and value ranges the real bus accepts, for values that are not
  numbers, for unknown joints / registers and for an empty sync write (BusError, same wording)
- positions(), feedback() (ONE transaction: position, load, current, voltage, temperature,
  status of every joint), snapshot() (incl. acc, vel_cap, lock), limits(), faults()
- status (error byte of the most recent reply per servo), stats (the same counter keys) and
  is_open
- torque_on(): idempotent, returns {"newly_enabled": [...]}; a joint that is already holding
  keeps its goal and is only written to where Lock / Acceleration / Goal_Velocity differ; a
  joint to be enabled gets goal := present (clamped into its limits); refused with nothing
  changed for velocity / acceleration 0 or a joint standing more than max_goal_gap ticks
  outside its limits
- torque_off(): puts Goal_Velocity / Acceleration back to their power-up value 0
- set_p_gain() -> bool: 8..40 whole numbers only, False (no write) when already equal,
  refused unless Lock == 1
- Present_Load follows the law fitted on the real arm (refs/research/sts3215-behavior.md):
  |load| = (P/4) * (|goal - present| - 1) + 16, capped at Torque_Limit, positive when
  present > goal

Test hooks that the real bus does not have: `now` / `sleep()` (simulated time), `log` (every
goal write), `io` (transaction counters), `fault` (inject servo status bits), `volt`, `temp`,
`power_cycle()`, `blocker`, `io_s` (extra time per control cycle to emulate bus I/O) and
`servo_profile` (model the servo-internal trapezoid setpoint generator, see profile_step).
"""

from __future__ import annotations

import math
import time

from servo_bus import JOINTS, BusError

try:  # the write guard of the real bus (servo_bus.py is owned by BUS; keep a local fallback)
    from servo_bus import RANGE as _BUS_RANGE
except ImportError:  # pragma: no cover
    _BUS_RANGE = None
try:
    from servo_bus import ERR_NAMES
except ImportError:  # pragma: no cover
    ERR_NAMES = {1: "voltage", 2: "sensor", 4: "overheat", 8: "overcurrent", 32: "overload"}

RANGE = dict(_BUS_RANGE) if _BUS_RANGE else {
    "Torque_Enable": (0, 1),
    "Acceleration": (0, 254),
    "Goal_Position": (0, 4095),
    "Goal_Velocity": (0, 4000),
    "Torque_Limit": (0, 1000),
    "Lock": (1, 1),
}

REST_RAW = {
    "shoulder_pan": 1999,
    "shoulder_lift": 955,
    "elbow_flex": 3025,
    "wrist_flex": 2820,
    "wrist_roll": 1983,
    "gripper": 2184,
}
LIMITS = {
    "shoulder_pan": (864, 3418),
    "shoulder_lift": (931, 3356),
    "elbow_flex": (775, 3064),
    "wrist_flex": (835, 3266),
    "wrist_roll": (0, 4095),
    "gripper": (1998, 3648),
}
P_BOOT = 16  # P_Coefficient stored in the servos' EEPROM (LeRobot's value)
UNCAPPED = 3000.0  # ticks/s used when Goal_Velocity is 0 (= "maximum speed" on the real servo)


def profile_step(sp: float, v: float, goal: float, vcap: float, acc: float, dt: float):
    """One step of the servo-internal setpoint generator: move the setpoint `sp` (ticks) towards
    `goal` with |speed| <= vcap (ticks/s) and |acceleration| <= acc (ticks/s^2), always able to
    stop at the goal (speed <= sqrt(2 * acc * distance)). acc <= 0 means no ramp.

    This is the INFERRED behaviour of the STS3215 position mode with Goal_Velocity / Acceleration
    set (Feetech documents T = D/V + V/A for a single goal); it has not been measured on this arm.
    A goal that moves at constant speed v is followed with a lag of v^2 / (2 * acc) ticks.
    Returns (sp, v)."""
    e = goal - sp
    if acc <= 0.0:
        step = max(-vcap * dt, min(vcap * dt, e))
        return sp + step, (step / dt if dt > 0 else 0.0)
    v_des = math.copysign(min(vcap, math.sqrt(2.0 * acc * abs(e))), e) if e != 0.0 else 0.0
    v += max(-acc * dt, min(acc * dt, v_des - v))
    step = v * dt
    if abs(e) < 0.5 and abs(v) <= 2.0 * acc * dt:
        return goal, 0.0  # arrived
    if step * e > 0 and abs(step) > abs(e):
        step = e  # never run past the goal within one step
    return sp + step, v


class SimBus:
    def __init__(self, sag_ticks: float = 12.0, blocker=None, realtime: bool = False, io_s: float = 0.0,
                 servo_profile: bool = False, lock: int = 1):
        self.pos = {j: float(v) for j, v in REST_RAW.items()}
        self.goal = dict(self.pos)  # Goal_Position register (reads back as written)
        self.torque = dict.fromkeys(JOINTS, 0)
        self.vel_cap = dict.fromkeys(JOINTS, 0)  # Goal_Velocity register (0 = no cap)
        self.acc = dict.fromkeys(JOINTS, 0)  # Acceleration register (0 = no ramp)
        self.torque_limit = dict.fromkeys(JOINTS, 1000)
        self.torque_limit["gripper"] = 500  # LeRobot stores Max_Torque_Limit 500 on the gripper
        self.lock = dict.fromkeys(JOINTS, int(lock))  # 1 at power-up; LeRobot leaves 0 on disconnect
        self.p_gain = dict.fromkeys(JOINTS, P_BOOT)
        self.fault = dict.fromkeys(JOINTS, 0)  # servo-side Status register (tests inject bits here)
        self.status = dict.fromkeys(JOINTS, 0)  # error byte of the most recent reply per servo
        self.stats = {"retries": 0, "fallbacks": 0, "failures": 0, "requests": 0}  # keys as in Bus
        self.io = {"reads": 0, "writes": 0}  # bus transactions (a sync read / write counts once)
        self.volt = 4.9
        self.temp = dict.fromkeys(JOINTS, 25)
        self.limits_raw = dict(LIMITS)
        self.sag = sag_ticks
        self.blocker = blocker  # callable(pos_dict_candidate) -> bool (True = blocked)
        self.realtime = realtime
        self.io_s = float(io_s)
        self.servo_profile = bool(servo_profile)
        self.sp = dict(self.pos)  # servo-internal profiled setpoint (only used with servo_profile)
        self.sp_v = dict.fromkeys(JOINTS, 0.0)
        self.now = 0.0
        self._last = 0.0
        self.log = []  # (t, joint, value) for every goal write
        self.opened = False

    # --- time handling: tests advance simulated time explicitly through sleep()
    def sleep(self, dt: float):
        if self.realtime:
            time.sleep(dt)
        self.now += dt + self.io_s
        self._step()

    def _step(self):
        dt = self.now - self._last
        self._last = self.now
        if dt <= 0:
            return
        self._advance(dt)

    def _eff_goal(self, j: str) -> float:
        """The servo clamps a goal outside its stored limits silently (the register keeps the
        written value)."""
        lo, hi = self.limits_raw[j]
        return max(lo, min(hi, self.goal[j]))

    def _advance_profile(self, j: str, dt: float):
        n = max(1, int(math.ceil(dt / 0.002)))
        h = dt / n
        vcap = float(self.vel_cap[j]) if self.vel_cap[j] > 0 else UNCAPPED
        acc = 100.0 * self.acc[j]
        goal = self._eff_goal(j)
        sp, v = self.sp[j], self.sp_v[j]
        for _ in range(n):
            sp, v = profile_step(sp, v, goal, vcap, acc, h)
        self.sp[j], self.sp_v[j] = sp, v

    def _drive_ref(self, j: str) -> float:
        """What the position loop of servo j is regulating on."""
        return self.sp[j] if self.servo_profile else self._eff_goal(j)

    def _blocked_step(self, j: str, step: float) -> float:
        """Largest part of `step` joint j can take without entering the obstacle."""
        if self.blocker is None or step == 0.0:
            return step
        cand = dict(self.pos)
        cand[j] = self.pos[j] + step
        if not self.blocker(cand) or self.blocker(self.pos):
            return step
        lo, hi = 0.0, 1.0  # bisect: stop right at the surface
        for _ in range(10):
            mid = 0.5 * (lo + hi)
            cand[j] = self.pos[j] + step * mid
            if self.blocker(cand):
                hi = mid
            else:
                lo = mid
        return step * lo

    def _advance_joint(self, j: str, dt: float):
        """Kinematic servo model: follow the (profiled) goal with the speed cap, droop, blocker."""
        if not self.torque[j]:
            return
        if self.servo_profile:
            self._advance_profile(j, dt)
        target = self._drive_ref(j)
        if j in ("shoulder_lift", "elbow_flex"):
            target = target + self.sag  # droop towards "down" (positive ticks)
        cap = float(self.vel_cap[j]) if self.vel_cap[j] > 0 else UNCAPPED
        step = max(-cap * dt, min(cap * dt, target - self.pos[j]))
        self.pos[j] += self._blocked_step(j, step)

    def _advance(self, dt: float):
        for j in JOINTS:
            self._advance_joint(j, dt)

    # --- registers
    def _load(self, j: str) -> int:
        if not self.torque[j]:
            return 0
        e = self._drive_ref(j) - self.pos[j]
        if abs(e) <= 1.0:
            return 0
        mag = min(float(self.torque_limit[j]), self.p_gain[j] / 4.0 * (abs(e) - 1.0) + 16.0)
        return int(round(-mag if e > 0 else mag))  # load > 0 <=> present > goal

    def _reg(self, j: str, reg: str) -> int:
        if reg == "Present_Position":
            return int(round(self.pos[j]))
        if reg == "Goal_Position":
            return int(round(self.goal[j]))
        if reg == "Torque_Enable":
            return self.torque[j]
        if reg == "Min_Position_Limit":
            return self.limits_raw[j][0]
        if reg == "Max_Position_Limit":
            return self.limits_raw[j][1]
        if reg == "Present_Voltage":
            return int(round(self.volt * 10))
        if reg == "Present_Temperature":
            return int(self.temp[j])
        if reg == "Present_Load":
            return self._load(j)
        if reg == "Present_Current":
            return abs(self._load(j)) // 10
        if reg == "Status":
            return self.fault[j]
        if reg in ("Moving", "Present_Velocity", "Homing_Offset", "Operating_Mode", "I_Coefficient"):
            return 0
        if reg == "Model":
            return 777
        if reg == "Acceleration":
            return self.acc[j]
        if reg == "Goal_Velocity":
            return self.vel_cap[j]
        if reg == "Torque_Limit":
            return self.torque_limit[j]
        if reg == "Max_Torque_Limit":
            return 500 if j == "gripper" else 1000
        if reg == "Lock":
            return self.lock[j]
        if reg == "P_Coefficient":
            return self.p_gain[j]
        if reg == "D_Coefficient":
            return 32
        if reg == "Protection_Current":
            return 250 if j == "gripper" else 500
        if reg == "Overload_Torque":
            return 25 if j == "gripper" else 80
        raise BusError(f"unknown register {reg!r}")

    @staticmethod
    def _as_int(value, what: str) -> int:
        try:
            if isinstance(value, (str, bytes, bytearray)):
                raise TypeError
            return int(value)
        except (TypeError, ValueError, OverflowError):
            raise BusError(f"refusing {what}={value!r}: not a number") from None

    @classmethod
    def _check_value(cls, reg: str, value) -> int:
        if not isinstance(reg, str) or reg not in RANGE:
            raise BusError(f"refusing to write {reg}: only RAM registers {sorted(RANGE)} are allowed")
        lo, hi = RANGE[reg]
        v = cls._as_int(value, reg)
        if not lo <= v <= hi:
            raise BusError(f"refusing {reg}={value}: outside [{lo}, {hi}]")
        return v

    @staticmethod
    def _joint_list(joints) -> tuple:
        joints = (joints,) if isinstance(joints, str) else tuple(joints)
        for j in joints:
            if j not in JOINTS:
                raise BusError(f"unknown joint {j!r} (known: {', '.join(JOINTS)})")
        if len(set(joints)) != len(joints):
            raise BusError(f"joint named twice in {joints}")
        return joints

    def _set(self, j: str, reg: str, v: int):
        if reg == "Goal_Position":
            self.goal[j] = float(v)
            self.log.append((self.now, j, v))
        elif reg == "Torque_Enable":
            if v and not self.torque[j]:
                self.sp[j], self.sp_v[j] = self.pos[j], 0.0  # the profile starts where the horn is
            self.torque[j] = v
        elif reg == "Goal_Velocity":
            self.vel_cap[j] = v
        elif reg == "Acceleration":
            self.acc[j] = v
        elif reg == "Torque_Limit":
            self.torque_limit[j] = v
        elif reg == "Lock":
            self.lock[j] = v

    # --- Bus interface
    def open(self, wait_s: float = 0.0):
        self.opened = True
        return self

    def close(self):
        self.opened = False

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    @property
    def is_open(self) -> bool:
        return self.opened

    def _count(self, kind: str, n: int = 1):
        self.io[kind] += n
        self.stats["requests"] += n

    def read(self, joint, reg):
        (joint,) = self._joint_list(joint)
        self._step()
        self._count("reads")
        self.status[joint] = self.fault[joint]
        return self._reg(joint, reg)

    def write(self, joint, reg, value):
        v = self._check_value(reg, value)  # refused before anything is "sent"
        (joint,) = self._joint_list(joint)
        self._step()
        self._count("writes")
        self.status[joint] = self.fault[joint]
        self._set(joint, reg, v)

    def sync_read(self, reg, joints=JOINTS):
        joints = self._joint_list(joints)
        self._step()
        self._count("reads")
        out = {}
        for j in joints:
            self.status[j] = self.fault[j]
            out[j] = self._reg(j, reg)
        return out

    def sync_write(self, reg, values):
        if not values:
            raise BusError(f"sync_write {reg}: no joints given")
        checked = {j: self._check_value(reg, v) for j, v in values.items()}  # nothing is sent if one is bad
        self._joint_list(tuple(checked))
        self._step()
        self._count("writes")
        for j, v in checked.items():
            self._set(j, reg, v)

    def positions(self):
        return self.sync_read("Present_Position")

    def feedback(self, joints=JOINTS) -> dict:
        """Position, load, current, voltage, temperature, status of all joints from ONE read."""
        joints = self._joint_list(joints)
        self._step()
        self._count("reads")
        out = {}
        for j in joints:
            self.status[j] = self.fault[j]
            load = self._load(j)
            out[j] = {"pos": int(round(self.pos[j])), "load": load, "current": abs(load) // 10,
                      "volt": int(round(self.volt * 10)) / 10.0, "temp": int(self.temp[j]), "status": self.fault[j]}
        return out

    def snapshot(self) -> dict:
        """State of every joint (one RAM-block read per joint on the real bus)."""
        self._step()
        self._count("reads", len(JOINTS))
        out = {}
        for j in JOINTS:
            self.status[j] = self.fault[j]
            load = self._load(j)
            out[j] = {
                "pos": int(round(self.pos[j])), "load": load, "current": abs(load) // 10,
                "volt": int(round(self.volt * 10)) / 10.0, "temp": int(self.temp[j]),
                "torque": self.torque[j], "goal": int(round(self.goal[j])), "status": self.fault[j],
                "acc": self.acc[j], "vel_cap": self.vel_cap[j], "lock": self.lock[j],
            }
        return out

    def limits(self):
        self._count("reads", len(JOINTS))
        return dict(self.limits_raw)

    def faults(self) -> dict:
        """Decoded error bytes of the most recent replies (free: no extra bus traffic)."""
        known = sum(ERR_NAMES)
        out = {}
        for j, s in self.status.items():
            if s:
                out[j] = [n for b, n in ERR_NAMES.items() if s & b]
                if s & ~known:
                    out[j].append(f"unknown_0x{s & ~known:02x}")
        return out

    def torque_on(self, joints=JOINTS, velocity: int = 400, acceleration: int = 20, max_goal_gap: int = 12) -> dict:
        """Enable torque without motion, like Bus.torque_on: a joint that is already holding
        keeps its goal and is only written to where Lock / Acceleration / Goal_Velocity differ;
        a joint whose torque is off gets Lock := 1, the caps and goal := present (clamped into
        its limits) before Torque_Enable := 1. Refused with nothing changed: caps of 0, or a
        joint to be enabled that stands more than max_goal_gap ticks outside its limits."""
        joints = self._joint_list(joints)
        gap = self._as_int(max_goal_gap, "max_goal_gap")
        want = {"Lock": 1, "Acceleration": self._check_value("Acceleration", acceleration),
                "Goal_Velocity": self._check_value("Goal_Velocity", velocity)}
        if want["Acceleration"] < 1 or want["Goal_Velocity"] < 1:
            raise BusError(f"torque_on(velocity={velocity}, acceleration={acceleration}): 0 means 'unlimited' "
                           f"in the servo; refusing to enable torque without the in-servo caps (nothing was changed)")
        self._step()
        new = [j for j in joints if self.torque[j] != 1]
        for j in new:
            lo, hi = self.limits_raw[j]
            pos = int(round(self.pos[j]))
            if not lo - gap <= pos <= hi + gap:
                raise BusError(f"torque_on refused, nothing was changed: {j}: position {pos} is more than {gap} "
                               f"ticks outside the servo's own limits {lo}..{hi}; with torque on it would move to "
                               f"the limit by itself. Move the joint inside its limits by hand")
        have = {"Lock": self.lock, "Acceleration": self.acc, "Goal_Velocity": self.vel_cap}
        for j in joints:
            for reg, val in want.items():
                if j in new or have[reg][j] != val:
                    self.write(j, reg, val)
        for j in new:
            lo, hi = self.limits_raw[j]
            self.write(j, "Goal_Position", max(lo, min(hi, int(round(self.pos[j])))))
        for j in new:
            self.write(j, "Torque_Enable", 1)
        return {"newly_enabled": new}

    def torque_off(self, joints=JOINTS, restore_profile: bool = True):
        """Torque off; also puts the profile registers back to their power-up values."""
        joints = self._joint_list(joints)
        for j in joints:
            if restore_profile:
                self.write(j, "Goal_Velocity", 0)
                self.write(j, "Acceleration", 0)
            self.write(j, "Torque_Enable", 0)

    def set_p_gain(self, joint, p) -> bool:
        """Position-loop P gain for this power cycle only, like Bus.set_p_gain: True if written,
        False if the servo already had it (no write); refused unless Lock == 1."""
        value = self._as_int(p, "P gain")
        if value != p:
            raise BusError(f"refusing P gain={p!r}: not a whole number")
        if not 8 <= value <= 40:
            raise BusError(f"P gain {p} outside the allowed 8..40")
        current = self.read(joint, "P_Coefficient")
        if current == value:
            return False
        if self.read(joint, "Lock") != 1:
            raise BusError(f"{joint}: Lock is not 1; refusing a write that would persist to EEPROM "
                           f"(P gain stays {current})")
        self._count("writes")
        self.p_gain[joint] = value
        return True

    # --- test hooks
    def power_cycle(self, joints=JOINTS):
        """What a servo looks like after a reset (supply dip): limp, uncapped, boot gain."""
        for j in joints:
            self.torque[j] = 0
            self.vel_cap[j] = 0
            self.acc[j] = 0
            self.p_gain[j] = P_BOOT
            self.lock[j] = 1
            self.fault[j] = 0
            self.goal[j] = float(int(round(self.pos[j])))
            self.sp[j], self.sp_v[j] = self.pos[j], 0.0
