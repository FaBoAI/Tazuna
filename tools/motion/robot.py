"""Arm: joint mapping, safety checks and smooth motion for the SO-101 follower.

Everything is expressed in URDF joint radians internally; the servo side is raw ticks.
The mapping raw <-> URDF is  q = sign * (raw - zero_tick) * 2*pi/4096  per joint, with the
zero ticks stored in config.json (they are refined by self-calibration, never written to
the servos).

Motion model (see DESIGN.md and refs/reviews for the findings each rule answers):
- A motion is a stream of position setpoints at about 50 Hz along a pre-checked joint path
  with ONE time law over the whole path (no stop at intermediate waypoints; corners are
  rounded by a short moving average of the reference, and the rounded path is what is
  checked). The time law runs on a plan clock that can never jump ahead (MS-7).
- The servos run a soft P-only loop, so a software integral term removes the standing error.
  It may correct freely only in the direction that holds the links up against gravity; in
  the other direction (and on pan / roll) it gets a small friction allowance, so it cannot
  wind up against an obstacle (MS-0), and a contact push is never inherited by the next
  motion as "gravity compensation" (MS-1).
- A motion that does not reach its target releases the push (goal := measured pose + the
  compensation the motion started with) and raises MotionAborted. Nothing fails silently.
- One bus read per control cycle gives position, load, voltage, temperature and the servo
  fault bits of all six joints; overload / overheat / over-current / sensor bits abort.
- Speeds come from named tiers (cfg["speed_tiers"]: joint speeds, servo-side profile, time
  law, per-joint speed scale); a tier that has not been validated on the hardware is refused
  unless explicitly allowed (bench_motion, the free-space benchmark that validates one).

Units: joint angles in radians inside, degrees in results and speeds (deg/s of the fastest
joint); positions in metres in the robot base frame; servo values in raw ticks.
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np

import gripper_model as GM
import kinematics as K
import tazuna_paths
from servo_bus import JOINTS, Bus, BusError

CONFIG_PATH = os.path.join(tazuna_paths.CONFIG_DIR, "config.json")     # the measurements of this rig
TICKS_PER_RAD = 4096.0 / (2.0 * math.pi)
TICKS_PER_DEG = 4096.0 / 360.0
ARM = K.ARM_JOINTS
JAW_Q_TOUCH_DEG = -11.41  # URDF gripper angle at which the fingertips touch (gripper_model.Q_CLOSED)
FAULT_NAMES = {1: "voltage", 2: "sensor", 4: "overheat", 8: "overcurrent", 32: "overload"}
# Points of the moving jaw that can be its lowest part, given at jaw angle 0 in gripper_frame_link
# axes (mm): (x, z, half width along y). Pad front edge, nose, outer nose corner, mid finger back.
_JAW_PTS_Q0 = ((-15.80, 6.31, 5.0), (-16.80, 7.27, 5.0), (-21.38, 6.6, 5.0), (-31.93, -15.0, 11.5))

DEFAULT_CONFIG = {
    # raw tick that corresponds to URDF angle 0 for each joint
    "zero_ticks": {"shoulder_pan": 2047.0, "shoulder_lift": 2047.0, "elbow_flex": 2047.0,
                   "wrist_flex": 2047.0, "wrist_roll": 2047.0},
    "sign": {"shoulder_pan": 1, "shoulder_lift": 1, "elbow_flex": 1, "wrist_flex": 1, "wrist_roll": 1},
    "limit_margin_ticks": 45,
    # extra software limits in URDF degrees (intersected with the servo limits)
    "soft_limits_deg": {"shoulder_pan": [-100, 100], "wrist_roll": [-160, 160]},
    "gripper": {"closed_raw": 1998, "open_raw": 3648, "margin": 25},
    # raw gripper position at which the fingertips touch (lower clamp of every gripper goal)
    "gripper_touch_raw": 2050,
    # physical fingertip relative to gripper_frame_link, in ITS axes (metres)
    "tip_offset": [0.0, 0.0, 0.0],
    # lowest z (base frame) the fingertip may be commanded to without allow_contact
    "floor_z": -0.02,
    "floor_clearance": 0.004,
    # a motion that STARTS below a height limit may go at most 1 mm lower per command (encoder
    # noise) and never more than this far below the limit in total (MS-5)
    "floor_relax_max_m": 0.003,
    # allow_contact lets the fingertip go this far below the floor limit, not further (MS-5)
    "contact_depth_m": 0.006,
    # extra slack of the moving-jaw floor check on top of what a vertical approach needs
    "jaw_floor_slack_m": 0.0003,
    # jaw opening (URDF deg) assumed for the floor check while the gripper position is unknown
    "jaw_unknown_q_deg": 40.0,
    # axis-aligned keep-out boxes for the fingertip: name -> [xmin,xmax,ymin,ymax,zmin,zmax]
    "keepouts": {"base_column": [-0.12, 0.085, -0.065, 0.065, -0.05, 0.13]},
    "servo_velocity_cap": 1100,   # steps/s, in-servo profile cap (safety net)
    "servo_acceleration": 40,     # x100 steps/s^2
    "max_joint_speed_deg_s": 70.0,
    "track_err_abort_deg": 10.0,  # P=32 saturates at 10.7 deg of error
    "track_err_abort_s": 0.6,
    "max_temp_c": 62,
    "min_volt": 4.2,              # at torque_on (unloaded rail)
    "min_volt_motion": 3.9,       # before each motion (the 5 V rail read 4.1 V in a normal carry)
    "sag_corr_max_deg": 7.0,      # integral authority in the direction that holds the links up
    "friction_corr_max_deg": 1.5,  # ... against that direction and on pan / roll, while tracking
    "inherit_corr_max_deg": 0.5,  # ... of a standing offset taken over from the previous motion
    "blocked_err_deg": 1.0,       # residual after a full settle (>= 2 s) that means "target not reached"
    "arrive_tol_deg": 3.0,        # coarse arrival test of motions without (or with a short) settle
    "arrive_grace_s": 0.6,
    "pinned_abort_s": 0.5,        # pushing beyond the correction limits with error left for this long: give up
    "settle_s": 2.5,              # settle=True: at most this long (the loop leaves as soon as it is in tolerance)
    "settle_dwell_s": 0.2,        # ... for this long
    "min_move_s": 0.35,
    "corner_blend_s": 0.2,        # reference smoothing of paths with corners (0 = off)
    "corner_kink_min": 0.25,
    "path_start_tol_deg": 1.0,    # a path may start this far from the measured pose (it is re-anchored)
    "limit_slack_ticks": 2,       # encoder flicker allowance of the start-pose limit relaxation
    "fault_abort_mask": 46,       # sensor 2 | overheat 4 | overcurrent 8 | overload 32 (not voltage 1)
    # True: one 15-byte sync read per cycle (position, load, voltage, temperature, status).
    # False: the 2-byte position sync read v1 used (fault bits then come from the reply error
    # byte only; no load / voltage / temperature figures in the results).
    "stream_feedback": True,
    # position-loop P gain for the arm joints while this harness drives them (the servos
    # boot with LeRobot's soft value 16; 32 is the STS3215 factory default). RAM-only write.
    "p_gain": 32,
    "p_gain_restore": 16,
    "gain_step_max_err_ticks": 10,  # a joint holding more standing error than this is "loaded"
    "ki_per_s": 2.0,
    "integ_deadzone_deg": 0.12,
    "integ_speed_gate_deg_s": 5.0,
    "squeeze_ticks": 40,          # holding offset after a grasp (60 sits on the overload threshold)
    "gripper_blocked_ticks": 40,
    # --- speed tiers (deg/s of the fastest joint). A tier may carry its own servo-side profile:
    # "servo_velocity_cap" (about 1.5 x the transit speed, in steps/s) and "servo_acceleration"
    # (scaled with the square of the transit speed, so the in-servo trapezoid profile trails the
    # setpoint stream by the same ~10 ticks as in the validated tier; 254 is the register maximum
    # and what LeRobot itself always writes). Without them the two top-level values above apply.
    # "slow" is exactly what ran on the real arm on 2026-11-10 (Goal_Velocity 1100, Acceleration 40).
    # A tier may also name its time law: "minjerk" (default; one min-jerk law over the path, the
    # speed is reached only at mid-path: what ran on the real arm) or "cruise" (ramp up within
    # cruise_ramp_s, hold the speed, ramp down: same peak speed, about 35-45 % shorter long moves).
    "speed_tier": "slow",
    "speed_tiers": {
        "slow": {"transit": 25.0, "approach": 12.0, "precise": 8.0},
        "normal": {"transit": 45.0, "approach": 20.0, "precise": 10.0,
                   "servo_velocity_cap": 770, "servo_acceleration": 130},
        "fast": {"transit": 70.0, "approach": 30.0, "precise": 12.0,
                 "servo_velocity_cap": 1200, "servo_acceleration": 254},
        # the validated speeds and servo profile with the cruise time law (NOT yet validated)
        "slow_cruise": {"transit": 25.0, "approach": 12.0, "precise": 8.0, "time_law": "cruise"},
    },
    "speed_tiers_validated": ["slow"],
    "cruise_ramp_s": 0.5,
    # Per-joint factor on the tier speed (joints not named: 1.0; a tier may carry its own
    # "joint_speed_scale"). wrist_roll holds no gravity load and ran at 40 deg/s on the real arm
    # (the roll flips of calib-at) while the other joints were validated at 25 deg/s. The scaled
    # joint still stays under max_joint_speed_deg_s and under the servo-side cap / 1.25.
    "joint_speed_scale": {"wrist_roll": 1.6},
    # stand-by pose (fingertip) and the folded pose the arm is parked in (servo ticks)
    "ready": {"xyz": [0.20, 0.0, 0.12], "pitch": 70.0, "roll": 0.0},
    "rest_raw": {"shoulder_pan": 1999, "shoulder_lift": 955, "elbow_flex": 3025, "wrist_flex": 2820,
                 "wrist_roll": 1983},
    # pass / fail thresholds of bench_motion (per move) and its early-abort limits
    # (settle_1deg_s: time after the end of the plan until every joint stays within 1 deg; the time
    # to the 0.25 deg tolerance is reported as settle_s but not judged: it is set by friction, not speed)
    # peak_err_deg: 70 % of the tracking-abort threshold (P = 32 saturates at 10.7 deg).
    "bench": {"start_tol_deg": 5.0, "peak_err_deg": 7.0, "overshoot_deg": 1.5, "settle_1deg_s": 0.8,
              "final_err_deg": 0.8, "peak_load": 700, "temp_rise_c": 6.0, "min_volt": 4.0, "min_loop_hz": 30.0,
              "abort_load": 950, "abort_load_s": 0.4, "abort_volt": 3.8},
}


def _merge(base: dict, over: dict) -> dict:
    """Recursive dict update (lists and scalars are replaced)."""
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _merge(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path: str | None = None) -> dict:
    """DEFAULT_CONFIG overlaid with the file (keys the file lacks keep their defaults).
    The file is config.json, or $FD_CONFIG when that is set: the regression tests pin the calibration
    their recorded frames and scenes were measured with (tests/testdata/calib_*/), so that re-calibrating
    the real arm does not change them."""
    path = path or os.environ.get("FD_CONFIG") or CONFIG_PATH
    cfg = json.loads(json.dumps(DEFAULT_CONFIG))
    if os.path.exists(path):
        with open(path) as f:
            user = json.load(f)
        _merge(cfg, user)
    return cfg


def save_config(cfg: dict, path: str = CONFIG_PATH):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cfg, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


class SafetyError(RuntimeError):
    pass


class MotionAborted(RuntimeError):
    """A motion was stopped (tracking error, servo fault, target not reached, monitor). The arm
    holds the measured pose with the gravity compensation the motion started with. `info`
    carries what is known about the abort ("aborted": True, "reason", joint errors ...)."""

    def __init__(self, *args, info: dict | None = None):
        super().__init__(*args)
        self.info = dict(info or {})
        self.info.setdefault("aborted", True)


RIG_KEYS = ("zero_ticks", "tip_offset", "floor_z", "table_z")      # what only a measurement of the rig can give
OLD_LAYOUT_FILES = ("config.json", "camera_model.json", "calib_points.json", "probes.json")      # measurements


def require_rig_config() -> str:
    """The configuration file a REAL arm is driven by, or SafetyError. Refused:
    - the file load_config() would read ($FD_CONFIG when set, else config/config.json) does not exist or lacks one of
      RIG_KEYS: load_config() would fill the gaps with the built-in numbers, which were measured on another arm;
    - a configuration or calibration file lies in the project folder itself, where these files were kept until the
      reorganisation of 2026-11-13: it would be this rig's data, silently ignored in favour of config/."""
    path = os.environ.get("FD_CONFIG") or CONFIG_PATH
    if not os.path.isfile(path):
        raise SafetyError(f"the rig's configuration file is missing ({path}): the real arm is not opened "
                          "without the measurements of this rig (zero offsets, table height, limits)")
    try:
        with open(path) as f:
            data = json.load(f)
    except ValueError as e:
        raise SafetyError(f"the rig's configuration file cannot be read ({path}: {e}): the real arm is not opened")
    missing = [k for k in RIG_KEYS if not isinstance(data, dict) or data.get(k) is None]
    if missing:
        raise SafetyError(f"the rig's configuration file ({path}) has no {', '.join(missing)}: the real arm is not "
                          "opened with built-in numbers in their place (they were measured on another arm)")
    stale = [n for n in OLD_LAYOUT_FILES if os.path.isfile(os.path.join(tazuna_paths.ROOT, n))]
    if stale:
        raise SafetyError(f"{', '.join(stale)} in the project folder ({tazuna_paths.ROOT}): these files are read from "
                          "config/ since 2026-11-13, so one left at the old place is ignored. Move it into config/ (or "
                          "delete it if config/ already holds this rig's data): the real arm is not opened while it is "
                          "unclear which measurements are meant")
    return path


def min_jerk(tau: float) -> float:
    tau = max(0.0, min(1.0, tau))
    return tau * tau * tau * (10.0 - 15.0 * tau + 6.0 * tau * tau)


def _deg(v) -> list:
    return [round(math.degrees(float(x)), 2) for x in v]


class Arm:
    def __init__(self, bus=None, cfg: dict | None = None, sleep=None, log=None, tool="active"):
        """tool: what is taped to the gripper. "active" (default) = whatever the tool file says
        (tooling.active(): tool_active.json in the project folder, or $FD_TOOL; no file = no tool = the arm
        behaves exactly as it always did); None = nothing; a tooling.Tool = that tool (simulation, tests)."""
        if bus is None and cfg is None:
            require_rig_config()     # the real servo bus: only with the measurements of this rig (else SafetyError)
        self.cfg = cfg if cfg is not None else load_config()
        self.bus = bus if bus is not None else Bus()
        self._sleep = sleep if sleep is not None else getattr(self.bus, "sleep", time.sleep)
        self._clock = (lambda: self.bus.now) if hasattr(self.bus, "now") else time.perf_counter
        self.log = log or (lambda *a, **k: None)
        self.raw_limits = None
        self._active = None      # speeds() dict of the tier whose servo profile is applied (None: cfg["speed_tier"])
        self._jaw_raw = None     # last gripper position seen on the bus (for the moving-jaw floor check)
        self._comp_hint = None   # compensation the previous motion of this Arm started with (see _current_offset)
        self._last_fb = None     # feedback of the most recent control cycle
        self._tool_gf = K.tool_offset(0.0, 0.0, 0.0)
        # A tool on the gripper (tooling.Tool) or None. With one, tip_tool() is the TOOL tip (so planning,
        # the floor check and the keep-outs are about the tool tip), the gripper never receives a goal,
        # ready_q() / rest_q() are the tool's poses and torque_off() wants the tool's rest pose.
        # cfg["tip_offset"] stays the fingertip: the moving-jaw line of the floor check depends on it.
        if isinstance(tool, str):
            if tool != "active":
                raise SafetyError(f"Arm(tool={tool!r}): pass \"active\", None or a tooling.Tool")
            import tooling  # (tooling imports this module: not at the top)
            tool = tooling.active()
        self.tool = tool

    def _c(self, key: str):
        """Config value, falling back to DEFAULT_CONFIG for keys an older config dict lacks."""
        return self.cfg[key] if key in self.cfg else DEFAULT_CONFIG[key]

    # ------------------------------------------------------------ lifecycle
    def open(self):
        self.bus.open()
        self.raw_limits = self.bus.limits()
        return self

    def close(self):
        self.bus.close()

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------ mapping
    def raw_to_q(self, raw: dict) -> np.ndarray:
        z, s = self.cfg["zero_ticks"], self.cfg["sign"]
        return np.array([s[j] * (raw[j] - z[j]) / TICKS_PER_RAD for j in ARM])

    def q_to_raw_f(self, q) -> dict:
        z, s = self.cfg["zero_ticks"], self.cfg["sign"]
        return {j: z[j] + s[j] * float(q[i]) * TICKS_PER_RAD for i, j in enumerate(ARM)}

    def q_to_raw(self, q) -> dict:
        return {j: int(round(v)) for j, v in self.q_to_raw_f(q).items()}

    def q_limits(self) -> dict:
        """Joint limits in URDF radians: servo-stored limits minus margin (mapped through the
        configured zero tick and sign of EVERY joint, wrist_roll included: KI-3), intersected
        with the optional soft limits."""
        m = self.cfg["limit_margin_ticks"]
        z, s = self.cfg["zero_ticks"], self.cfg["sign"]
        out = {}
        for j in ARM:
            lo_raw, hi_raw = self.raw_limits[j]
            a = s[j] * (lo_raw + m - z[j]) / TICKS_PER_RAD
            b = s[j] * (hi_raw - m - z[j]) / TICKS_PER_RAD
            a, b = min(a, b), max(a, b)
            soft = self.cfg.get("soft_limits_deg", {}).get(j)
            if soft:
                a, b = max(a, math.radians(soft[0])), min(b, math.radians(soft[1]))
            out[j] = (a, b)
        return out

    def tip_tool(self) -> np.ndarray:
        """The frame every "tip" of this class refers to: the physical fingertip (cfg["tip_offset"]), or
        the tip of the tool that is taped to the gripper (self.tool)."""
        if self.tool is not None:
            return K.tool_offset(*self.tool.tip_offset)
        return K.tool_offset(*self.cfg["tip_offset"])

    # ------------------------------------------------------------ reading
    def read_raw(self) -> dict:
        raw = self.bus.positions()
        self._jaw_raw = raw.get("gripper", self._jaw_raw)
        return raw

    def read_q(self) -> np.ndarray:
        return self.raw_to_q(self.read_raw())

    def gripper_frac(self, raw: int) -> float:
        g = self.cfg["gripper"]
        return (raw - g["closed_raw"]) / float(g["open_raw"] - g["closed_raw"])

    def state(self) -> dict:
        snap = self.bus.snapshot()
        raw = {j: snap[j]["pos"] for j in JOINTS}
        self._jaw_raw = raw["gripper"]
        q = self.raw_to_q(raw)
        ps = K.pose_summary(q)
        tip = K.fk(q, self.tip_tool())[:3, 3]
        jaw = K.jaw_open_axis(ps["T"])
        horizontal = math.hypot(jaw[0], jaw[1])
        try:
            p_gain = {j: self.bus.read(j, "P_Coefficient") for j in JOINTS}
        except BusError:
            p_gain = None
        return {
            "raw": raw,
            "q_deg": {j: round(math.degrees(q[i]), 2) for i, j in enumerate(ARM)},
            "tool_xyz": [round(float(v), 4) for v in ps["xyz"]],
            "tip_xyz": [round(float(v), 4) for v in tip],
            "pitch_down_deg": round(ps["pitch_down_deg"], 2),
            # direction fixed jaw -> moving jaw; meaningless when that axis is nearly vertical (KI-2)
            "jaw_yaw_deg": round(math.degrees(math.atan2(jaw[1], jaw[0])), 1) if horizontal >= 0.5 else None,
            "jaw_tilt_deg": round(math.degrees(math.asin(max(-1.0, min(1.0, float(jaw[2]))))), 1),
            "gripper_raw": raw["gripper"],
            "gripper_open_frac": round(self.gripper_frac(raw["gripper"]), 3),
            "jaw_q_deg": round(self.gripper_q_deg(raw["gripper"]), 2),
            "torque": {j: snap[j]["torque"] for j in JOINTS},
            "load": {j: snap[j]["load"] for j in JOINTS},
            "current": {j: snap[j]["current"] for j in JOINTS},
            "temp_c": {j: snap[j]["temp"] for j in JOINTS},
            "volt": min(snap[j]["volt"] for j in JOINTS),
            "goal_minus_pos_ticks": {j: snap[j]["goal"] - snap[j]["pos"] for j in JOINTS} if all(
                snap[j]["torque"] for j in JOINTS) else None,
            "status": {j: snap[j]["status"] for j in JOINTS},
            "faults": {j: self._fault_names(snap[j]["status"]) for j in JOINTS if snap[j]["status"]},
            "p_gain": p_gain,
            "profile": {j: [snap[j].get("vel_cap"), snap[j].get("acc")] for j in JOINTS},
            "speed_tier": self._active["tier"] if self._active else self._c("speed_tier"),
        }

    # ------------------------------------------------------------ safety
    def _relax(self, limit: float, z0) -> float:
        """Height limit for a motion that starts at height z0 (None: no start pose).

        A start above the limit changes nothing. A start below it may go 1 mm lower (encoder
        noise between commands) but in total never more than floor_relax_max_m below the limit;
        deeper than that only 0.2 mm of noise slack is left, so the arm can always move up and
        sideways but cannot be walked into the table millimetre by millimetre (MS-5)."""
        if z0 is None:
            return limit
        d = float(self._c("floor_relax_max_m"))
        return max(min(limit, z0 - 0.001), min(limit - d, z0 - 2e-4))

    def _jaw_samples(self, jaw_q_deg=None, goal_raw=None) -> list:
        """Jaw opening angles (URDF deg) to assume for the moving-jaw floor check."""
        if jaw_q_deg is not None:
            return [float(v) for v in np.atleast_1d(jaw_q_deg)]
        if self._jaw_raw is None:
            try:
                self.read_raw()
            except Exception:  # no usable bus yet: fall back to a wide-open jaw (conservative)
                pass
        if self._jaw_raw is None:
            return [float(self._c("jaw_unknown_q_deg"))]
        q0 = self.gripper_q_deg(self._jaw_raw)
        if goal_raw is None:
            return [q0]
        return [float(v) for v in np.linspace(q0, self.gripper_q_deg(goal_raw), 4)]

    @staticmethod
    def _jaw_low_z(t_gf: np.ndarray, jaws) -> float:
        """Lowest base-frame z of the distal part of the moving jaw (t_gf: gripper_frame pose)."""
        low = math.inf
        for qj in jaws:
            for x0, z0, hw in _JAW_PTS_Q0:
                x, z = GM.jaw_point(qj, (x0, z0))
                base = t_gf[2, 0] * x * 1e-3 + t_gf[2, 2] * z * 1e-3 + t_gf[2, 3]
                low = min(low, base - abs(t_gf[2, 1]) * hw * 1e-3)
        return low

    def _check_ctx(self, q_start=None, jaw_q_deg=None, gripper_goal_raw=None) -> dict:
        """Everything check_q needs that does not depend on the pose under test."""
        lim = self.q_limits()
        flicker = float(self._c("limit_slack_ticks")) / TICKS_PER_RAD
        tool = self.tip_tool()
        jaws = self._jaw_samples(jaw_q_deg, gripper_goal_raw)
        # In a vertical approach the nose of the moving jaw reaches up to 8.06 mm beyond the
        # gripper_frame origin (gripper_model.moving_tip_depth), i.e. a little below the configured
        # fingertip point; the floor clearance already absorbs that. Only what a tilt adds is refused.
        jaw_slack = max(0.0, 0.00806 - float(self.cfg["tip_offset"][2])) + float(self._c("jaw_floor_slack_m"))
        ctx = {"lim": [], "tool": tool, "jaws": jaws, "tip0": None, "frame0_z": None, "jaw0_z": None,
               "jaw_slack": jaw_slack}
        for i, j in enumerate(ARM):
            lo, hi = lim[j]
            if q_start is not None:
                # a start pose outside the limits (the folded rest pose sits on the stops) may
                # move back in, never further out (2 ticks of encoder flicker allowed)
                if q_start[i] < lo:
                    lo = q_start[i] - flicker
                if q_start[i] > hi:
                    hi = q_start[i] + flicker
            ctx["lim"].append((lo, hi))
        if q_start is not None:
            fr0 = K.link_frames(q_start)
            ctx["tip0"] = (fr0[4] @ tool)[:3, 3]
            ctx["frame0_z"] = [f[2, 3] for f in fr0]
            ctx["jaw0_z"] = self._jaw_low_z(fr0[4] @ self._tool_gf, jaws)
        return ctx

    def _check_pose(self, q, ctx: dict, allow_contact: bool, ignore_keepouts, what: str, contact_depth=None):
        for i, j in enumerate(ARM):
            lo, hi = ctx["lim"][i]
            if not (lo - 1e-9 <= q[i] <= hi + 1e-9):
                raise SafetyError(
                    f"{what}: {j}={math.degrees(q[i]):.1f} deg outside limits "
                    f"[{math.degrees(lo):.1f}, {math.degrees(hi):.1f}]"
                )
        frames = K.link_frames(q)
        tip = (frames[4] @ ctx["tool"])[:3, 3]
        jaw_z = self._jaw_low_z(frames[4] @ self._tool_gf, ctx["jaws"])
        tip0 = ctx["tip0"]
        tip0_z = None if tip0 is None else tip0[2]
        slack = ctx["jaw_slack"]
        base_floor = self.cfg["floor_z"] + self.cfg["floor_clearance"]
        if not allow_contact:
            floor = self._relax(base_floor, tip0_z)
            if tip[2] < floor:
                raise SafetyError(f"{what}: fingertip z={tip[2]:.4f} below floor limit {floor:.4f}")
            jfloor = self._relax(base_floor - slack, ctx["jaw0_z"])
            if jaw_z < jfloor:
                raise SafetyError(
                    f"{what}: moving jaw z={jaw_z:.4f} below floor limit {jfloor:.4f} (fingertip z={tip[2]:.4f}; "
                    f"with this tilt / wrist roll the opened jaw, {max(ctx['jaws']):.0f} deg, hangs below the fingertip)")
            # wrist-flex axis and wrist-roll origin carry bulky parts: keep them well above the table
            for name, idx, clr in (("wrist", 3, 0.035), ("gripper body", 4, 0.03)):
                z = frames[idx][2, 3]
                zmin = self._relax(self.cfg["floor_z"] + clr, None if ctx["frame0_z"] is None else ctx["frame0_z"][idx])
                if z < zmin:
                    raise SafetyError(f"{what}: {name} z={z:.4f} too close to the table (min {zmin:.4f})")
        else:
            depth = float(self._c("contact_depth_m")) if contact_depth is None else float(contact_depth)
            if math.isfinite(depth):  # math.inf: probe_down, which bounds its own descent
                floor = self._relax(base_floor - depth, tip0_z)
                if tip[2] < floor:
                    raise SafetyError(f"{what}: fingertip z={tip[2]:.4f} below the contact depth limit {floor:.4f} "
                                      f"(floor limit {base_floor:.4f} minus {depth * 1000:.0f} mm)")
                jfloor = self._relax(base_floor - depth - slack, ctx["jaw0_z"])
                if jaw_z < jfloor:
                    raise SafetyError(f"{what}: moving jaw z={jaw_z:.4f} below the contact depth limit {jfloor:.4f}")
        for name, box in self.cfg.get("keepouts", {}).items():
            if name in ignore_keepouts:
                continue
            x0, x1, y0, y1, z0, z1 = box

            def inside(p):
                return x0 <= p[0] <= x1 and y0 <= p[1] <= y1 and z0 <= p[2] <= z1

            def depth_in(p):
                return min(p[0] - x0, x1 - p[0], p[1] - y0, y1 - p[1], p[2] - z0, z1 - p[2])

            # a motion that starts inside a keep-out may leave it, not go deeper into it (MS-5)
            if inside(tip) and not (tip0 is not None and inside(tip0) and depth_in(tip) <= depth_in(tip0) + 5e-4):
                raise SafetyError(f"{what}: fingertip {np.round(tip, 3)} inside keep-out '{name}'")

    def check_q(self, q, allow_contact: bool = False, ignore_keepouts=(), what: str = "target", q_start=None, *,
                jaw_q_deg=None, contact_depth=None):
        """Raise SafetyError if pose q violates joint limits, the floor, or a keep-out.

        q_start (the pose a motion starts from) relaxes each constraint only as far as the
        start pose already violates it, so the arm can always move back INTO the safe set
        (e.g. from the folded rest pose, which sits on the mechanical stops) but not further
        out: see _relax for the height limits; a start inside a keep-out may not go deeper.

        The floor is tested for the fingertip AND for the moving jaw (KI-2): with a tilted
        approach and |roll| > 90 deg the opened jaw hangs 10-20 mm below the fingertip.
        jaw_q_deg: jaw opening (URDF deg, or several) to assume; default: the gripper position
        last seen on the bus. allow_contact lets the fingertip go contact_depth (default
        cfg["contact_depth_m"]) below the floor limit and skips the wrist / gripper-body lines."""
        q = np.asarray(q, dtype=float)
        if q.shape != (5,) or not np.all(np.isfinite(q)):
            raise SafetyError(f"{what}: not a finite 5-joint pose: {q}")
        q0 = None if q_start is None else np.asarray(q_start, dtype=float)
        self._check_pose(q, self._check_ctx(q0, jaw_q_deg), allow_contact, ignore_keepouts, what, contact_depth)

    @staticmethod
    def _fault_names(status: int) -> list:
        return [n for b, n in FAULT_NAMES.items() if status & b] or ([f"status {status}"] if status else [])

    def health_check(self, snap: dict | None = None, motion: bool = False):
        """Temperature, supply voltage and servo fault bits of all six joints (one snapshot).

        At torque_on (motion=False) any fault bit and a rail below min_volt refuse. Before a
        motion the unloaded-rail threshold does not apply (the 5 V rail dips to 4.1 V in a
        normal carry): the limit is min_volt_motion and the voltage bit alone is tolerated."""
        snap = self.bus.snapshot() if snap is None else snap
        vmin = float(self._c("min_volt_motion")) if motion else float(self.cfg["min_volt"])
        for j in JOINTS:
            if snap[j]["temp"] > self.cfg["max_temp_c"]:
                raise SafetyError(f"{j} temperature {snap[j]['temp']} C too high (max {self.cfg['max_temp_c']})")
            if snap[j]["volt"] < vmin:
                raise SafetyError(f"{j} voltage {snap[j]['volt']} V too low (min {vmin})")
            st = snap[j]["status"]
            if (st & ~1) if motion else st:
                raise SafetyError(f"{j} reports error status {st} ({', '.join(self._fault_names(st))})")
        return snap

    # ------------------------------------------------------------ speed tiers
    def speeds(self, tier: str | None = None, allow_unvalidated: bool = False) -> dict:
        """Joint speeds (deg/s of the fastest joint) of a speed tier:
        {"tier", "transit", "approach", "precise", "validated", "servo_velocity_cap", "servo_acceleration",
        "time_law", "joint_speed_scale"}. A joint may move at up to joint_speed_scale x the tier speed
        (run_path clamps that product to max_joint_speed_deg_s and to the servo-side cap / 1.25).

        tier None -> cfg["speed_tier"]. A tier that is not listed in cfg["speed_tiers_validated"]
        raises SafetyError unless allow_unvalidated=True (used only by bench_motion)."""
        name = tier if tier is not None else self._c("speed_tier")
        tiers = self._c("speed_tiers")
        if name not in tiers:
            raise SafetyError(f"unknown speed tier {name!r} (known: {sorted(tiers)})")
        validated = name in self._c("speed_tiers_validated")
        if not validated and not allow_unvalidated:
            raise SafetyError(
                f"speed tier {name!r} has not been validated on this arm (validated: "
                f"{list(self._c('speed_tiers_validated'))}); run bench_motion({name!r}) on the hardware and add "
                f"it to speed_tiers_validated first")
        t = tiers[name]
        out = {"tier": name, "validated": validated}
        vmax = float(self.cfg["max_joint_speed_deg_s"])
        for k in ("transit", "approach", "precise"):
            try:
                v = float(t[k])
            except (KeyError, TypeError, ValueError):
                raise SafetyError(f"speed tier {name!r}: {k!r} is missing or not a number")
            if not (math.isfinite(v) and 0.0 < v <= vmax):
                raise SafetyError(f"speed tier {name!r}: {k}={v} deg/s outside (0, {vmax}]")
            out[k] = v
        vel = int(t.get("servo_velocity_cap", self.cfg["servo_velocity_cap"]))
        acc = int(t.get("servo_acceleration", self.cfg["servo_acceleration"]))
        if not 1 <= acc <= 254:
            raise SafetyError(f"speed tier {name!r}: servo_acceleration {acc} outside 1..254")
        if not (vel <= 4000 and vel / TICKS_PER_DEG >= 1.25 * out["transit"]):
            raise SafetyError(f"speed tier {name!r}: servo_velocity_cap {vel} steps/s must be at least 1.25 x the "
                              f"transit speed ({out['transit']} deg/s = {out['transit'] * TICKS_PER_DEG:.0f} steps/s) "
                              f"and at most 4000")
        out["servo_velocity_cap"], out["servo_acceleration"] = vel, acc
        out["time_law"] = t.get("time_law", "minjerk")
        if out["time_law"] not in ("minjerk", "cruise"):
            raise SafetyError(f"speed tier {name!r}: time_law {out['time_law']!r} is not 'minjerk' or 'cruise'")
        scale = dict.fromkeys(ARM, 1.0)
        for src_cfg in (self._c("joint_speed_scale"), t.get("joint_speed_scale", {})):
            for j, v in src_cfg.items():
                if j not in scale:
                    raise SafetyError(f"joint_speed_scale: unknown joint {j!r} (arm joints: {list(ARM)})")
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    v = float("nan")
                if not (math.isfinite(v) and 0.2 <= v <= 3.0):
                    raise SafetyError(f"joint_speed_scale[{j!r}] must be a number in 0.2 .. 3.0")
                scale[j] = v
        out["joint_speed_scale"] = scale
        return out

    def _profile(self) -> dict:
        """Servo-side profile registers (Goal_Velocity cap, Acceleration) this Arm applies and
        expects: the arm joints follow the active speed tier, the gripper keeps the top-level
        values in every tier."""
        sp = self._active if self._active is not None else self.speeds(None)
        g = self.cfg["gripper"]
        grip = (int(g.get("velocity_cap", self.cfg["servo_velocity_cap"])),
                int(g.get("acceleration", self.cfg["servo_acceleration"])))
        arm = (sp["servo_velocity_cap"], sp["servo_acceleration"])
        return {"tier": sp["tier"], "arm": arm, "gripper": grip, "time_law": sp["time_law"],
                "joint_speed_scale": sp["joint_speed_scale"],
                "joints": {j: (grip if j == "gripper" else arm) for j in JOINTS}}

    def apply_profile(self, tier: str | None = None, allow_unvalidated: bool = False) -> dict:
        """Make `tier` the active speed tier of this Arm and write its servo-side profile
        registers (only where they differ). Goals are not touched. Returns speeds(tier)."""
        sp = self.speeds(tier, allow_unvalidated=allow_unvalidated)
        before, self._active = self._active, sp
        try:
            prof = self._profile()
            snap = self.bus.snapshot()
            for j in JOINTS:
                vel, acc = prof["joints"][j]
                if snap[j].get("acc") != acc:
                    self.bus.write(j, "Acceleration", acc)
                if snap[j].get("vel_cap") != vel:
                    self.bus.write(j, "Goal_Velocity", vel)
        except BaseException:
            self._active = before  # registers may be mixed now: _require_torque will say so
            raise
        return sp

    # ------------------------------------------------------------ torque
    def torque_on(self) -> dict:
        """Enable torque on all six joints without motion. Idempotent: joints that are already
        holding keep their goals (gravity compensation, grasp squeeze).

        Order (MS-6): health check, validate the configured gain and profile, Lock := 1 and
        the P gain BEFORE Torque_Enable, so a refused or failed gain write leaves the arm
        exactly as limp as it was. Returns {"newly_enabled": [...], "p_gain", "tier", ...}."""
        snap = self.health_check()
        prof = self._profile()  # raises for an unknown / unvalidated tier before anything changes
        gains = self.apply_gains(snap=snap)
        newly = []
        groups = [(JOINTS, prof["arm"])] if prof["arm"] == prof["gripper"] else [
            (tuple(ARM), prof["arm"]), (("gripper",), prof["gripper"])]
        try:
            for joints, (vel, acc) in groups:
                res = self.bus.torque_on(joints, velocity=vel, acceleration=acc)
                if isinstance(res, dict):
                    newly += list(res.get("newly_enabled", []))
        except BusError:
            # the bus refused (e.g. a joint beyond its stored limits): if the arm was limp before,
            # do not leave it limp with the raised gain
            if gains["changed"] and not any(snap[j]["torque"] for j in JOINTS) and not newly:
                try:
                    self.restore_gains()
                except (BusError, SafetyError):
                    pass
            raise
        return {"newly_enabled": newly, "p_gain": gains["p_gain"], "gain_changed": gains["changed"],
                "tier": prof["tier"], "servo_velocity_cap": prof["arm"][0], "servo_acceleration": prof["arm"][1]}

    def _set_gains(self, p: int, force: bool, snap: dict | None = None) -> dict:
        """Write P_Coefficient = p on the arm joints (RAM only, Lock := 1 first).

        Nothing is written when the value is out of range. A joint that is torqued and holds
        a standing error is "loaded": stepping its gain changes the holding torque at once.
        Raising the gain there is ramped in steps of 2 (the arm creeps towards its goal);
        lowering it would let the arm sag by up to the standing error and is refused unless
        force=True (then ramped)."""
        if not 8 <= int(p) <= 40:
            raise SafetyError(f"P gain {p} is outside the allowed 8..40; nothing was changed")
        p = int(p)
        cur = {j: self.bus.read(j, "P_Coefficient") for j in ARM}
        todo = [j for j in ARM if cur[j] != p]
        if not todo:
            return {"p_gain": p, "changed": [], "ramped": []}
        snap = self.bus.snapshot() if snap is None else snap
        lim = int(self._c("gain_step_max_err_ticks"))
        loaded = {j: snap[j]["goal"] - snap[j]["pos"] for j in todo
                  if snap[j]["torque"] == 1 and abs(snap[j]["goal"] - snap[j]["pos"]) > lim}
        lowering = {j: v for j, v in loaded.items() if p < cur[j]}
        if lowering and not force:
            raise SafetyError(
                f"refusing to lower the P gain to {p} while joints hold a standing error (goal - pos ticks: "
                f"{lowering}): the arm would sag by up to that much. Park and switch torque off (that restores "
                f"the gain), or pass force=True to ramp it down. Nothing was changed.")
        for j in todo:
            self.bus.write(j, "Lock", 1)  # with Lock = 1 the gain write can never be persisted
        for j in todo:
            if j not in loaded:
                self.bus.set_p_gain(j, p)
        if loaded:
            level = dict(cur)
            while any(level[j] != p for j in loaded):
                for j in loaded:
                    if level[j] != p:
                        level[j] += max(-2, min(2, p - level[j]))
                        self.bus.set_p_gain(j, level[j])
                self._sleep(0.05)
        return {"p_gain": p, "changed": todo, "ramped": sorted(loaded)}

    def apply_gains(self, force: bool = False, snap: dict | None = None) -> dict:
        """Raise the position-loop P gain of the arm joints (RAM only, see Bus.set_p_gain)."""
        p = self.cfg.get("p_gain")
        if not p or not hasattr(self.bus, "set_p_gain"):
            return {"p_gain": None, "changed": [], "ramped": []}
        return self._set_gains(int(p), force, snap)

    def restore_gains(self, force: bool = False) -> dict:
        p = self.cfg.get("p_gain_restore", 16)
        if not hasattr(self.bus, "set_p_gain"):
            return {"p_gain": None, "changed": [], "ramped": []}
        return self._set_gains(int(p), force)

    def torque_off(self, force: bool = False) -> dict:
        """Torque off on all six joints (the arm goes limp where it is), then the boot gain.

        With a tool attached this is refused (SafetyError, nothing written) unless the arm is in the
        tool's rest pose or holds nothing up anyway: the plain rest pose would put the tool into the
        table, and anywhere else the arm falls with the tool first. force=True is the emergency stop."""
        if self.tool is not None and not force:
            import tooling
            why = tooling.torque_off_block(self)
            if why:
                raise SafetyError(why)
        self.bus.torque_off(JOINTS)
        self._active = None
        try:
            g = self.restore_gains()
        except (BusError, SafetyError) as e:
            raise type(e)(f"torque is OFF on all joints, but the P gain could not be restored: {e}") from e
        return {"torque": "off", "p_gain": g["p_gain"]}

    def is_torqued(self) -> bool:
        """True when all five arm joints are torqued (the gripper is not looked at)."""
        return all(self.bus.read(j, "Torque_Enable") == 1 for j in ARM)

    def _require_torque(self) -> dict:
        """Gate of every motion (MS-8). All SIX servos must be torqued and still carry the
        profile registers and the gain this harness wrote: a servo that reset on a supply dip
        comes back limp, without speed / acceleration caps and with the boot gain. Also runs
        the per-motion health check. One snapshot; returns it."""
        snap = self.bus.snapshot()
        self._jaw_raw = snap["gripper"]["pos"]
        off = [j for j in JOINTS if snap[j]["torque"] != 1]
        if off:
            raise SafetyError(f"torque is off on {', '.join(off)}; run torque_on first")
        prof = self._profile()
        lost = []
        for j in JOINTS:
            vel, acc = prof["joints"][j]
            if "acc" in snap[j] and (snap[j]["acc"] != acc or snap[j]["vel_cap"] != vel):
                lost.append(f"{j}: Goal_Velocity {snap[j]['vel_cap']} / Acceleration {snap[j]['acc']} "
                            f"(expected {vel} / {acc})")
        p = self.cfg.get("p_gain")
        if p and hasattr(self.bus, "set_p_gain"):
            for j in ARM:
                got = self.bus.read(j, "P_Coefficient")
                if got != int(p):
                    lost.append(f"{j}: P gain {got} (expected {int(p)})")
        if lost:
            raise SafetyError("servo state lost or changed (" + "; ".join(lost) + "); run torque_on to re-apply "
                              "(it is idempotent: holding joints keep their goals)")
        self.health_check(snap=snap, motion=True)
        return snap

    # ------------------------------------------------------------ motion core
    def _hold_sign(self, q) -> np.ndarray:
        """+1 where +q raises the links carried by the joint (holding them up against gravity
        then needs a POSITIVE goal - present), -1 where it lowers them, 0 where gravity has
        almost no lever arm (and for pan / roll). FK only: the sign of d(mean height of the
        distal joint frames and the fingertip)/dq."""
        tool = self.tip_tool()

        def height(qq, i):
            fr = K.link_frames(qq)
            return float(np.mean([f[2, 3] for f in fr[i + 1:]] + [(fr[4] @ tool)[2, 3]]))

        q = np.asarray(q, dtype=float)
        out = np.zeros(5)
        for i in (1, 2, 3):
            d = np.zeros(5)
            d[i] = 1e-3
            g = (height(q + d, i) - height(q - d, i)) / 2e-3  # metres per radian
            if abs(g) > 0.02:
                out[i] = math.copysign(1.0, g)
        return out

    def _offset_bounds(self, q, allow_deg: float):
        """Bounds (lo, hi) of the software correction at pose q: the full sag_corr_max_deg in
        the direction that holds the links up, only `allow_deg` (friction) in the other
        direction and on pan / roll."""
        mx = math.radians(self.cfg["sag_corr_max_deg"])
        al = math.radians(allow_deg)
        sgn = self._hold_sign(q)
        return np.where(sgn < 0, -mx, -al), np.where(sgn > 0, mx, al)

    def _path_bounds(self, wps, allow_deg: float):
        """Union of _offset_bounds over a path (at most 12 poses incl. both ends)."""
        idx = sorted(set(int(round(v)) for v in np.linspace(0, len(wps) - 1, min(len(wps), 12))))
        lo, hi = np.full(5, np.inf), np.full(5, -np.inf)
        for i in idx:
            a, b = self._offset_bounds(wps[i], allow_deg)
            lo, hi = np.minimum(lo, a), np.maximum(hi, b)
        return lo, hi

    def _current_offset(self, snap: dict | None = None) -> np.ndarray:
        """Standing (goal - present) offset per arm joint in radians: the gravity-sag
        compensation currently in effect. Only what gravity can explain is taken over: the
        holding direction up to sag_corr_max_deg, anything else up to inherit_corr_max_deg,
        so the push left by a contact is never carried into the next motion (MS-1).

        A gravity-loaded joint whose standing offset is outside those bounds is pressing on
        something; its true gravity compensation is then better estimated by what the previous
        motion of this Arm STARTED with (before the contact) than by the bound, so that value
        is used when there is one."""
        snap = self.bus.snapshot() if snap is None else snap
        s = self.cfg["sign"]
        off = np.array([s[j] * (snap[j]["goal"] - snap[j]["pos"]) / TICKS_PER_RAD for j in ARM])
        q = self.raw_to_q({j: snap[j]["pos"] for j in ARM})
        lo, hi = self._offset_bounds(q, float(self._c("inherit_corr_max_deg")))
        out = np.clip(off, lo, hi)
        if self._comp_hint is not None:
            loaded = (lo < -1.5 * math.radians(float(self._c("inherit_corr_max_deg")))) | (
                hi > 1.5 * math.radians(float(self._c("inherit_corr_max_deg"))))  # joints with a holding side
            pressing = loaded & (np.abs(off - out) > math.radians(0.05))
            out = np.where(pressing, np.clip(self._comp_hint, lo, hi), out)
        return out

    def _goal_raw(self, q_plan, integ=None, strict: bool = True, start=None) -> dict:
        """Servo goals for setpoint q_plan (+ correction integ). A planned setpoint outside the
        servo's own limits is an error, never clamped silently (KI-3); only the correction (and
        a hold, strict=False) is limited to the servo range. `start` (raw ticks of the pose the
        motion started from) widens that test as far as the start pose itself was outside, so
        an arm resting beyond a stored limit can still be driven back in."""
        plan = self.q_to_raw_f(q_plan)
        cmd = plan if integ is None else self.q_to_raw_f(np.asarray(q_plan, dtype=float) + integ)
        out = {}
        for j in ARM:
            lo, hi = self.raw_limits[j]
            ok_lo, ok_hi = lo - 3.0, hi + 3.0
            if start is not None:
                ok_lo, ok_hi = min(ok_lo, start[j] - 3.0), max(ok_hi, start[j] + 3.0)
            if strict and not (ok_lo <= plan[j] <= ok_hi):
                raise SafetyError(f"setpoint {j}={plan[j]:.0f} ticks is outside the servo limits [{lo}, {hi}]; "
                                  f"refusing to clamp it silently")
            out[j] = int(round(max(lo + 5, min(hi - 5, cmd[j]))))
        return out

    def _write_q(self, q_plan, integ=None, start=None):
        self.bus.sync_write("Goal_Position", self._goal_raw(q_plan, integ, start=start))

    def _hold(self, q_meas, integ=None):
        """Stop where the arm is: goal := measured pose (+ the compensation to keep). Never raises
        for a pose near the limits. The sync write is a broadcast without a reply, and this one
        is the write that takes a push off the servos, so the goals are read back and any joint
        that missed it is written again individually (acknowledged)."""
        goals = self._goal_raw(q_meas, integ, strict=False)
        self.bus.sync_write("Goal_Position", goals)
        snap = self.bus.snapshot()
        for j in ARM:
            if snap[j]["goal"] != goals[j]:
                self.bus.write(j, "Goal_Position", goals[j])

    def _hold_here(self):
        """Hold the measured pose, keeping only the part of the standing offset that gravity
        explains (one snapshot). Releases any push against an obstacle."""
        snap = self.bus.snapshot()
        q = self.raw_to_q({j: snap[j]["pos"] for j in ARM})
        self._hold(q, self._current_offset(snap))

    def _feedback(self):
        """ONE bus read per control cycle: (q_arm, feedback dict or None)."""
        if self._c("stream_feedback") and hasattr(self.bus, "feedback"):
            fb = self.bus.feedback()
            raw = {j: fb[j]["pos"] for j in JOINTS}
        else:
            fb = None
            raw = self.bus.positions()
        self._jaw_raw = raw.get("gripper", self._jaw_raw)
        self._last_fb = fb
        return self.raw_to_q(raw), fb

    def _faults(self, fb, mask: int) -> dict:
        """Joints whose status (Status register of this cycle or the reply error byte) has a bit of mask."""
        reply = getattr(self.bus, "status", None) or {}
        out = {}
        for j in JOINTS:
            st = (fb[j]["status"] if fb else 0) | int(reply.get(j, 0))
            if st & mask:
                out[j] = st
        return out

    @staticmethod
    def _pushing(integ, err, lo, hi, tol: float):
        """Joints that push on something: the standing command (goal - present = correction +
        error) is outside what the correction may ever hold, with error left. In free air at
        rest that sum cannot leave the friction band on the pressing side (statics), so this
        test does not depend on how long the integrator has had to wind up."""
        total = integ + err
        eps = math.radians(0.05)
        return ((total > hi + eps) & (err > tol)) | ((total < lo - eps) & (err < -tol))

    # The opt-in slow-motion mode ("fine", run_path(..., fine=True)). NOT validated on the hardware:
    # `fd track-test --fine --repeat 3 --accept` on the real arm is what may record it as passed.
    # ff_deg: Coulomb-friction feed-forward per joint (deg of goal offset in the direction the joint is
    # planned to move). Sized from the traces of 2026-11-11 (frozen correction): a moving joint ran
    # 11 ticks (lift, elbow) / 1.5 - 2 ticks (pan, wrist_flex) behind its goal when it stopped; the
    # defaults are 60 % (lift, elbow) to 80 % (pan, wrist_flex) of that at 0.088 deg per tick, so a
    # reversal is helped, never over-driven.
    # ff_v0_deg_s: planned joint speed at which the feed-forward is fully on (linear below it).
    FINE_DEFAULTS = {"ff_deg": {"shoulder_pan": 0.12, "shoulder_lift": 0.6, "elbow_flex": 0.6, "wrist_flex": 0.12,
                                "wrist_roll": 0.0},
                     "ff_v0_deg_s": 0.5}

    def _fine_params(self) -> dict:
        """Parameters of the fine mode: FINE_DEFAULTS overlaid with cfg["fine_motion"], validated
        ({"ff": rad per joint, "v0": rad/s}). The feed-forward may not be larger than the friction
        allowance of the correction (friction_corr_max_deg): a typing error in the config must not
        become a push."""
        user = self.cfg.get("fine_motion") or {}
        ff = dict(self.FINE_DEFAULTS["ff_deg"])
        ff.update(user.get("ff_deg") or {})
        cap = float(self._c("friction_corr_max_deg"))
        out = np.zeros(5)
        for j, v in ff.items():
            if j not in ARM:
                raise SafetyError(f"fine_motion.ff_deg: unknown joint {j!r} (arm joints: {list(ARM)})")
            try:
                v = float(v)
            except (TypeError, ValueError):
                v = float("nan")
            if not (math.isfinite(v) and 0.0 <= v <= cap):
                raise SafetyError(f"fine_motion.ff_deg[{j!r}] must be a number in 0 .. {cap} deg (friction_corr_max_deg)")
            out[ARM.index(j)] = math.radians(v)
        try:
            v0 = float(user.get("ff_v0_deg_s", self.FINE_DEFAULTS["ff_v0_deg_s"]))
        except (TypeError, ValueError):
            v0 = float("nan")
        if not (math.isfinite(v0) and 0.05 <= v0 <= 5.0):
            raise SafetyError("fine_motion.ff_v0_deg_s must be a number in 0.05 .. 5 deg/s")
        return {"ff": out, "v0": math.radians(v0)}

    @staticmethod
    def _fine_corr(fine: dict, integ, qd, lo, hi):
        """Correction written in the fine mode: the integral term plus a Coulomb-friction
        feed-forward in the direction each joint is PLANNED to move (zero for a joint whose plan
        rests, so the settle phase and its pinned test see exactly the integral term). The sum stays
        inside the same bounds [lo, hi] the integral term has, so the fine mode cannot push harder
        on an obstacle than the default mode.

        Why: a joint that reverses needs its standing command swung through the whole friction
        band before it moves again. The integral term alone takes 1 - 3 s for that at drawing speed
        (measured 2026-11-11 at 6 mm/s with the integral on: at a corner of the square shoulder_lift
        stood 2.8 s while its plan moved 12 ticks and let go 12.7 ticks behind, 4 mm of height; on
        the circle the elbow stood 1.2 s and let go 17 ticks behind). The feed-forward makes part of
        the swing at the moment the plan reverses. It does NOT make the slips of a steadily moving
        joint smaller (their size is set inside the servo)."""
        return np.clip(integ + fine["ff"] * np.clip(qd / fine["v0"], -1.0, 1.0), lo, hi)

    def _track(self, plan, duration: float, integ0, hz: float = 50.0, monitor=None,
               integ_lo=None, integ_hi=None, settle_s: float = 0.0, tol_deg: float = 0.25,
               arrive_tol_deg: float | None = None, arrive_grace_s: float = 0.0, fine: dict | None = None):
        """Stream setpoints plan(t) -> q (rad, 5) for `duration` seconds, then optionally keep
        regulating on the final setpoint for up to `settle_s` (leaving as soon as every joint
        has been inside tol_deg for settle_dwell_s).

        The servos run a soft P-only loop with a wide friction band, so an outer integral
        term (in software, on the encoder readings) removes the standing error:
            command = plan + integ,   d(integ)/dt = Ki * (plan - measured)
        Integration is suspended for a joint while its planned speed is high (velocity lag
        is not something to integrate) and inside a small dead zone (no hunting). integ is
        clamped to [integ_lo, integ_hi], which bounds how hard the arm can push on an obstacle.

        The plan is indexed by a plan clock that advances by at most two periods per cycle, so
        a stalled loop (bus retries, descheduled process) resumes where it stopped instead of
        skipping the checked path (MS-7). Every cycle does one write and one read.

        Aborts (hold the measured pose with the compensation the motion started with, then
        MotionAborted): tracking error above track_err_abort_deg for track_err_abort_s; a
        servo fault bit of cfg["fault_abort_mask"].

        fine: None (default; every write is what it always was) or the dict of _fine_params():
        the goal then carries _fine_corr (integral term + friction feed-forward, same bounds)
        instead of the integral term alone. Aborts, the settle phase and its pinned test, the
        monitor and the returned "integ" are the same in both modes."""
        ki = float(self.cfg.get("ki_per_s", 2.0))
        dead = math.radians(float(self.cfg.get("integ_deadzone_deg", 0.12)))
        v_gate = math.radians(float(self.cfg.get("integ_speed_gate_deg_s", 5.0)))
        mx = math.radians(self.cfg["sag_corr_max_deg"])
        lo = np.full(5, -mx) if integ_lo is None else np.asarray(integ_lo, dtype=float)
        hi = np.full(5, mx) if integ_hi is None else np.asarray(integ_hi, dtype=float)
        integ_start = np.clip(np.array(integ0, dtype=float), lo, hi)
        integ = integ_start.copy()
        abort_rad = math.radians(self.cfg["track_err_abort_deg"])
        abort_s = float(self.cfg["track_err_abort_s"])
        tol = math.radians(tol_deg)
        dwell = float(self._c("settle_dwell_s"))
        pinned_s = float(self._c("pinned_abort_s"))
        mask = int(self._c("fault_abort_mask"))
        arrive = None if arrive_tol_deg is None else math.radians(arrive_tol_deg)
        dt = 1.0 / hz
        q_end = np.asarray(plan(duration), dtype=float)
        # direction of the final approach per joint (for the overshoot figure)
        d_end = q_end - np.asarray(plan(max(0.0, duration - 0.25)), dtype=float)
        dirn = np.where(np.abs(d_end) > 1e-3, np.sign(d_end), 0.0)
        bad_since = ok_since = pin_since = None
        st = {"peak_err": 0.0, "peak_err_joint": np.zeros(5), "overshoot": 0.0, "peak_load": 0, "min_volt": None,
              "max_temp": None,
              "settled": False, "pinned": False, "t_plan_end": None, "t_coarse": None, "cycles": 0}
        coarse = math.radians(1.0)
        t = 0.0
        t_wall0 = last = self._clock()
        first = True
        q_prev = np.asarray(plan(0.0), dtype=float)
        raw_start = self.q_to_raw_f(q_prev)
        q_meas, fb = self._feedback()
        max_cycles = int((duration + settle_s + arrive_grace_s + 2.0) * hz * 20) + 1000
        while True:
            if st["cycles"] > max_cycles:  # a clock that does not advance must not hang the arm in a loop
                self._hold(q_meas, integ_start)
                raise MotionAborted("control loop clock is not advancing; holding the measured pose",
                                    info={"reason": "clock", "t": round(t, 3)})
            now = self._clock()
            step = 0.0 if first else min(max(now - last, 0.0), 2.0 * dt)
            last, first = now, False
            t += step
            q_plan = np.asarray(plan(min(t, duration)), dtype=float)
            qd = (q_plan - q_prev) / max(step, 1e-3)
            err = q_plan - q_meas
            active = (np.abs(qd) < v_gate) & (np.abs(err) > dead)
            integ = np.clip(integ + np.where(active, ki * err * min(step, 0.1), 0.0), lo, hi)
            self._write_q(q_plan, integ if fine is None else self._fine_corr(fine, integ, qd, lo, hi), raw_start)
            q_meas, fb = self._feedback()
            st["cycles"] += 1
            err = q_plan - q_meas
            worst = float(np.max(np.abs(err)))
            if t <= duration:
                st["peak_err"] = max(st["peak_err"], worst)
                st["peak_err_joint"] = np.maximum(st["peak_err_joint"], np.abs(err))
            if fb is not None:
                st["peak_load"] = max(st["peak_load"], max(abs(fb[j]["load"]) for j in ARM))
                v = min(fb[j]["volt"] for j in JOINTS)
                st["min_volt"] = v if st["min_volt"] is None else min(st["min_volt"], v)
                c = max(fb[j]["temp"] for j in JOINTS)
                st["max_temp"] = c if st["max_temp"] is None else max(st["max_temp"], c)
            faults = self._faults(fb, mask)
            if faults:
                self._hold(q_meas, integ_start)
                txt = "; ".join(f"{j}: {', '.join(self._fault_names(s))} (status {s})" for j, s in faults.items())
                raise MotionAborted(f"servo fault during the motion: {txt}. Holding the measured pose.",
                                    info={"reason": "servo_fault", "faults": faults, "t": round(t, 3)})
            if worst > abort_rad:
                bad_since = bad_since if bad_since is not None else t
                if t - bad_since > abort_s:
                    self._hold(q_meas, integ_start)
                    w = int(np.argmax(np.abs(err)))
                    raise MotionAborted(
                        f"tracking error {math.degrees(worst):.1f} deg on {ARM[w]} for > {abort_s} s; "
                        f"holding the measured pose",
                        info={"reason": "tracking_error", "joint": ARM[w], "joint_err_deg": _deg(err),
                              "t": round(t, 3)})
            else:
                bad_since = None
            if monitor is not None and monitor(t, q_plan, q_meas, integ):
                return {"q_meas": q_meas, "integ": integ, "stopped": True, "t": t, "integ_start": integ_start,
                        "duration_s": self._clock() - t_wall0, **st}
            if t >= duration:
                if st["t_plan_end"] is None:
                    st["t_plan_end"] = t
                st["overshoot"] = max(st["overshoot"], float(np.max(dirn * (q_meas - q_end))))
                if worst > coarse:
                    st["t_coarse"] = t + dt  # the arm is still more than 1 deg off
                if settle_s > 0.0:
                    if worst < tol:
                        ok_since = ok_since if ok_since is not None else t
                        if t - ok_since >= dwell:
                            st["settled"] = True
                            break
                    else:
                        ok_since = None
                    if self._pushing(integ, err, lo, hi, tol).any():
                        pin_since = pin_since if pin_since is not None else t
                        if t - pin_since >= pinned_s:
                            st["pinned"] = True  # blocked: do not keep pushing for the rest of the settle
                            break
                    else:
                        pin_since = None
                    if t >= duration + settle_s:
                        break
                elif arrive is None or worst <= arrive or t >= duration + arrive_grace_s:
                    break
            q_prev = q_plan
            self._sleep(dt)
        return {"q_meas": q_meas, "integ": integ, "stopped": False, "t": t, "integ_start": integ_start,
                "duration_s": self._clock() - t_wall0, **st}

    def _time_law(self, total: float, speed_deg_s: float, law: str):
        """Path length travelled as a function of time for a path of `total` rad (Chebyshev arc
        length) whose fastest joint may reach speed_deg_s: (duration, s(t)).

        "minjerk": one min-jerk law, s = total * minjerk(t / T) with T = 1.875 * total / speed
            (the peak speed is 1.875 x the average and is reached only at mid-path).
        "cruise": speed ramps up with a min-jerk shaped ramp of cruise_ramp_s, stays at
            speed_deg_s and ramps down; T = total / speed + ramp. Used only when the path is long
            enough to gain from it, otherwise the min-jerk law."""
        v = math.radians(speed_deg_s)
        t_mj = max(float(self._c("min_move_s")), 1.875 * total / v)
        ramp = float(self._c("cruise_ramp_s"))
        if law == "cruise" and ramp > 0.0 and total > 1.2 * v * ramp:
            t_c = total / v + ramp

            def ramp_len(tau):  # integral of minjerk: distance covered during a ramp, / (v * ramp)
                tau = max(0.0, min(1.0, tau))
                return tau ** 4 * (2.5 - 3.0 * tau + tau * tau)

            def s_cruise(t):
                if t <= 0.0:
                    return 0.0
                if t < ramp:
                    return v * ramp * ramp_len(t / ramp)
                if t <= t_c - ramp:
                    return v * (t - 0.5 * ramp)
                if t < t_c:
                    return total - v * ramp * ramp_len((t_c - t) / ramp)
                return total

            return t_c, s_cruise
        return t_mj, (lambda t: min_jerk(t / t_mj) * total)

    def _path_plan(self, wp, speed_deg_s: float, law: str = "minjerk", scale=None):
        """Time law over a joint path: (plan(t), duration, total length, blend time).

        ONE time law over the whole path (Chebyshev arc length, see _time_law), so intermediate
        waypoints are passed without stopping and the fastest joint peaks at speed_deg_s. With
        `scale` (5 factors) the length of a segment is max_j(|dq_j| / scale_j), so joint j may
        move at up to scale_j x speed_deg_s. A path with real corners (direction change between
        segments) is smoothed with a moving average of corner_blend_s over the reference, which
        rounds each corner with bounded acceleration; the motion then takes that much longer."""
        sc = np.ones(5) if scale is None else np.asarray(scale, dtype=float)
        seg = [(a, b, float(np.max(np.abs(b - a) / sc))) for a, b in zip(wp[:-1], wp[1:])]
        seg = [s for s in seg if s[2] > 1e-9]
        if not seg:
            end = wp[-1]
            return (lambda t: end), 0.0, 0.0, 0.0
        lens = np.array([s[2] for s in seg])
        total = float(lens.sum())
        cum = np.concatenate([[0.0], np.cumsum(lens)])
        duration, s_of_t = self._time_law(total, speed_deg_s, law)

        def raw(t):
            if t <= 0.0:
                return seg[0][0]
            if t >= duration:
                return seg[-1][1]
            s = s_of_t(t)
            i = max(0, min(len(seg) - 1, int(np.searchsorted(cum, s, side="right") - 1)))
            a, b, length = seg[i]
            return a + (b - a) * max(0.0, min(1.0, (s - cum[i]) / length))

        kink = 0.0
        for (a0, b0, l0), (a1, b1, l1) in zip(seg[:-1], seg[1:]):
            kink = max(kink, float(np.max(np.abs((b1 - a1) / l1 - (b0 - a0) / l0) / sc)))
        blend = float(self._c("corner_blend_s")) if kink > float(self._c("corner_kink_min")) else 0.0
        if blend <= 0.0:
            return raw, duration, total, 0.0
        offs = np.linspace(0.0, blend, max(3, int(round(blend * 50.0)) + 1))

        def smooth(t):
            return np.mean([raw(t - o) for o in offs], axis=0)

        return smooth, duration + blend, total, blend

    def run_path(self, waypoints, speed_deg_s: float, allow_contact: bool = False,
                 ignore_keepouts=(), settle=True, hz: float = 50.0, monitor=None, tol_deg: float = 0.25,
                 gripper_goal_raw=None, contact_depth=None, time_law: str | None = None, fine: bool = False):
        """Follow joint-space waypoints (list of 5-vectors, radians; the first one is the
        current measured pose) as ONE continuous motion: a single time law over the whole
        path, no stop at intermediate waypoints. speed_deg_s is the peak speed of the fastest
        joint; time_law None = the law of the active speed tier ("minjerk" unless the tier
        says "cruise", see _time_law). A joint with a joint_speed_scale above 1 (wrist_roll)
        may move that much faster, so a move it dominates is that much shorter.

        settle: True = regulate on the target for up to cfg["settle_s"], False = none,
            float = at most that many seconds. The loop leaves as soon as all joints are
            inside tol_deg (no fixed waiting).
        gripper_goal_raw: gripper goal written once when the motion starts, so the gripper
            moves while the arm moves (not waited for; "gripper_pos_raw" reports where it is).
        allow_contact / contact_depth: see check_q.
        monitor(t, q_plan, q_meas) -> True stops the motion (MotionAborted).
        fine: False (default) = the motion exactly as it always was. True = the opt-in slow-motion
            mode (friction feed-forward at reversals, see _fine_corr; parameters FINE_DEFAULTS /
            cfg["fine_motion"]): same checks, same correction bounds, same aborts; the result then
            also carries "fine": True and "fine_ff_deg". Not validated on the hardware until
            `fd track-test --fine --repeat 3 --accept` passed there (tracking.fine_accepted).

        Returns {"q_meas", "joint_err_deg", "corr_deg", "duration_s", "settle_s", "peak_err_deg",
        "overshoot_deg", "settled", "aborted": False, "peak_load", "min_volt", "max_temp", ...}.

        Raises SafetyError before any motion (limits, floor, keep-outs, speed, servo state) and
        MotionAborted when the target is not reached: the arm stands short of it while the
        standing command (correction + error) is beyond the correction limits for
        pinned_abort_s, or the residual is larger than free air leaves behind (blocked_err_deg
        after a settle of 2 s or more, arrive_tol_deg otherwise). The push is released first
        (MS-0)."""
        snap = self._require_torque()
        wp = [np.array(w, dtype=float) for w in waypoints]
        if not wp or any(w.shape != (5,) or not np.all(np.isfinite(w)) for w in wp):
            raise SafetyError("path must be a list of finite 5-joint poses")
        speed = float(speed_deg_s)
        if not (math.isfinite(speed) and speed > 0.0):
            raise SafetyError(f"speed must be a positive, finite number (got {speed_deg_s})")
        prof = self._profile()
        speed = min(speed, float(self.cfg["max_joint_speed_deg_s"]), prof["arm"][0] / TICKS_PER_DEG / 1.25)
        if isinstance(settle, bool):
            settle_max = float(self._c("settle_s")) if settle else 0.0
        else:
            settle_max = float(settle)
            if not (math.isfinite(settle_max) and settle_max >= 0.0):
                raise SafetyError(f"settle must be True, False or a non-negative number of seconds (got {settle})")
        # the path starts at the MEASURED pose: one reading for the path, the checks and the stream
        q_now = self.raw_to_q({j: snap[j]["pos"] for j in ARM})
        d0 = np.abs(wp[0] - q_now)
        if float(np.max(d0)) > math.radians(float(self._c("path_start_tol_deg"))):
            w = int(np.argmax(d0))
            raise SafetyError(f"path starts {math.degrees(d0[w]):.1f} deg from the measured pose on {ARM[w]}; "
                              f"plan it from the current pose")
        wp[0] = q_now
        grip_goal = None
        if gripper_goal_raw is not None:
            grip_goal = self._gripper_clamp(gripper_goal_raw)
        law = prof["time_law"] if time_law is None else time_law
        if law not in ("minjerk", "cruise"):
            raise SafetyError(f"time_law {law!r} is not 'minjerk' or 'cruise'")
        fine_par = self._fine_params() if fine else None  # a bad cfg["fine_motion"] refuses before anything moves
        # per-joint speed factor, clamped so that no joint is planned above the limits `speed` obeys
        v_top = min(float(self.cfg["max_joint_speed_deg_s"]), prof["arm"][0] / TICKS_PER_DEG / 1.25)
        scale = np.array([min(float(prof["joint_speed_scale"][j]), v_top / speed) for j in ARM])
        plan, duration, total, blend = self._path_plan(wp, speed, law, scale)
        # safety check of every <= 1.5 deg of joint motion (and of the smoothed reference)
        ctx = self._check_ctx(wp[0], None, grip_goal)
        for i in range(len(wp) - 1):
            n = max(1, int(math.ceil(math.degrees(float(np.max(np.abs(wp[i + 1] - wp[i])))) / 1.5)))
            for k in range(1, n + 1):
                self._check_pose(wp[i] + (wp[i + 1] - wp[i]) * (k / n), ctx, allow_contact, ignore_keepouts,
                                 f"path point {i}.{k}", contact_depth)
        if blend > 0.0:
            n = max(2, int(math.ceil(duration * speed * float(np.max(scale)) / 1.5)))
            for k in range(1, n + 1):
                self._check_pose(np.asarray(plan(duration * k / n)), ctx, allow_contact, ignore_keepouts,
                                 f"blended path at t={duration * k / n:.2f} s", contact_depth)
        integ0 = self._current_offset(snap)
        self._comp_hint = integ0.copy()
        lo, hi = self._path_bounds(wp, float(self._c("friction_corr_max_deg")))
        wrapped = None
        if monitor is not None:
            def wrapped(t, q_plan, q_meas, integ):
                return monitor(t, q_plan, q_meas)
        if grip_goal is not None:
            self.bus.write("gripper", "Goal_Position", grip_goal)
        track_kw = {} if fine_par is None else {"fine": fine_par}
        res = self._track(plan, duration, integ0, hz=hz, monitor=wrapped, integ_lo=lo, integ_hi=hi,
                          settle_s=settle_max, tol_deg=tol_deg,
                          arrive_tol_deg=float(self._c("arrive_tol_deg")),
                          arrive_grace_s=float(self._c("arrive_grace_s")), **track_kw)
        q_end = np.asarray(plan(duration), dtype=float)
        err = q_end - res["q_meas"]
        info = {"joint_err_deg": _deg(err), "corr_deg": _deg(res["integ"]), "duration_s": round(res["duration_s"], 3),
                "peak_err_deg": round(math.degrees(res["peak_err"]), 2)}
        if res["stopped"]:
            self._hold(res["q_meas"], res["integ_start"])
            raise MotionAborted("stopped by monitor; holding the measured pose", info={"reason": "monitor", **info})
        tol = math.radians(tol_deg)
        # blocked = the arm stands still short of the target while pushing harder than friction
        # explains (seen for pinned_abort_s during the settle, or at the end of a full settle),
        # or a residual no free-air motion leaves behind
        full = settle_max >= 2.0
        pinned = np.zeros(5, dtype=bool)
        if res["pinned"] or (full and not res["settled"]):
            pinned = self._pushing(res["integ"], err, lo, hi, tol)
        far_deg = float(self._c("blocked_err_deg")) if full else float(self._c("arrive_tol_deg"))
        far = np.abs(err) > math.radians(far_deg)
        if pinned.any() or far.any():
            self._hold(res["q_meas"], res["integ_start"])  # stop pushing, keep the gravity compensation
            w = int(np.argmax(np.where(pinned | far, np.abs(err), 0.0)))
            why = "while pushing beyond the correction limit" if pinned[w] else f"(more than {far_deg} deg)"
            raise MotionAborted(
                f"target not reached: {ARM[w]} is {math.degrees(err[w]):+.2f} deg short {why}; something is in "
                f"the way? Push released; holding the measured pose.",
                info={"reason": "target_not_reached", "joint": ARM[w], **info})
        out = {
            "q_meas": res["q_meas"],
            "joint_err_deg": info["joint_err_deg"],
            "corr_deg": info["corr_deg"],
            "duration_s": info["duration_s"],
            "settle_s": round(max(0.0, res["t"] - (res["t"] if res["t_plan_end"] is None else res["t_plan_end"])), 3),
            "peak_err_deg": info["peak_err_deg"],
            "peak_err_joint_deg": _deg(res["peak_err_joint"]),
            "overshoot_deg": round(math.degrees(max(0.0, res["overshoot"])), 2),
            "settled": bool(res["settled"]) if settle_max > 0.0 else None,
            # time after the end of the plan until every joint was (and stayed) within 1 deg
            "settle_1deg_s": None if settle_max <= 0.0 else round(
                max(0.0, (res["t_coarse"] or res["t_plan_end"] or 0.0) - (res["t_plan_end"] or 0.0)), 3),
            "aborted": False,
            "speed_deg_s": round(speed, 2),
            "joint_speed_scale": {j: round(float(scale[i]), 3) for i, j in enumerate(ARM)},
            "plan_s": round(duration, 3),
            "blend_s": blend,
            "time_law": law,
            "peak_load": res["peak_load"],
            "min_volt": res["min_volt"],
            "max_temp": res["max_temp"],
            # control cycles per second actually achieved (about 45 on the real bus)
            "loop_hz": round(res["cycles"] / res["duration_s"], 1) if res["duration_s"] > 0.2 else None,
        }
        if grip_goal is not None:
            out["gripper_goal_raw"] = grip_goal
            out["gripper_pos_raw"] = self._jaw_raw
        if fine_par is not None:
            out["fine"] = True
            out["fine_ff_deg"] = _deg(fine_par["ff"])
        return out

    def check_path(self, waypoints, speed_deg_s: float, allow_contact: bool = False, ignore_keepouts=(),
                   gripper_goal_raw=None, contact_depth=None, time_law: str | None = None,
                   tier: str | None = None) -> dict:
        """The safety checks run_path makes before it moves, WITHOUT moving and without needing
        torque: every <= 1.5 deg of joint motion along the path and along the corner-blended
        reference that would actually be streamed. The path starts at waypoints[0] (it need not
        be the measured pose), so a whole sequence can be checked before its first motion.

        tier: use that speed tier's time law / joint speed scale / servo cap instead of the
        active one (the blended reference depends on them). gripper_goal_raw: the jaw opening
        the gripper will have on this path (the floor check of the moving jaw depends on it).
        Raises SafetyError like run_path would; returns {"duration_s", "blend_s"}."""
        wp = [np.array(w, dtype=float) for w in waypoints]
        if not wp or any(w.shape != (5,) or not np.all(np.isfinite(w)) for w in wp):
            raise SafetyError("path must be a list of finite 5-joint poses")
        speed = float(speed_deg_s)
        if not (math.isfinite(speed) and speed > 0.0):
            raise SafetyError(f"speed must be a positive, finite number (got {speed_deg_s})")
        saved = self._active
        try:
            if tier is not None:
                self._active = self.speeds(tier)
            prof = self._profile()
        finally:
            self._active = saved
        v_top = min(float(self.cfg["max_joint_speed_deg_s"]), prof["arm"][0] / TICKS_PER_DEG / 1.25)
        speed = min(speed, v_top)
        law = prof["time_law"] if time_law is None else time_law
        if law not in ("minjerk", "cruise"):
            raise SafetyError(f"time_law {law!r} is not 'minjerk' or 'cruise'")
        grip_goal = None if gripper_goal_raw is None else self._gripper_clamp(gripper_goal_raw)
        scale = np.array([min(float(prof["joint_speed_scale"][j]), v_top / speed) for j in ARM])
        plan, duration, _total, blend = self._path_plan(wp, speed, law, scale)
        ctx = self._check_ctx(wp[0], None, grip_goal)
        for i in range(len(wp) - 1):
            n = max(1, int(math.ceil(math.degrees(float(np.max(np.abs(wp[i + 1] - wp[i])))) / 1.5)))
            for k in range(1, n + 1):
                self._check_pose(wp[i] + (wp[i + 1] - wp[i]) * (k / n), ctx, allow_contact, ignore_keepouts,
                                 f"path point {i}.{k}", contact_depth)
        if blend > 0.0:
            n = max(2, int(math.ceil(duration * speed * float(np.max(scale)) / 1.5)))
            for k in range(1, n + 1):
                self._check_pose(np.asarray(plan(duration * k / n)), ctx, allow_contact, ignore_keepouts,
                                 f"blended path at t={duration * k / n:.2f} s", contact_depth)
        return {"duration_s": float(duration), "blend_s": float(blend)}

    # ------------------------------------------------------------ moves
    def move_joints(self, q_target, speed_deg_s: float = 35.0, **kw):
        q_now = self.read_q()
        return self.run_path([q_now, np.asarray(q_target, dtype=float)], speed_deg_s, **kw)

    def solve(self, xyz, pitch_down_deg: float, roll_deg: float, q_seed=None, use_tip: bool = True):
        lim = self.q_limits()
        tool = self.tip_tool() if use_tip else None
        q, info = K.ik(xyz, pitch_down_deg, math.radians(roll_deg), q_init=q_seed, tool=tool, limits=lim)
        if not info["ok"]:
            rng = K.pitch_range(xyz, math.radians(roll_deg), tool=tool, limits=lim)
            hint = f"feasible pitch range at this point: {rng[0]}..{rng[1]} deg" if rng else "no pitch is feasible at this point"
            raise SafetyError(
                f"IK failed for xyz={np.round(xyz, 4).tolist()} pitch={pitch_down_deg} roll={roll_deg}: "
                f"{info['reason']}; {hint}"
            )
        return q

    def plan_tip_path(self, points, q_start=None, step_m: float = 0.005, step_deg: float = 4.0) -> list:
        """Joint waypoints for straight Cartesian fingertip segments. No motion.

        points: [(xyz, pitch_down_deg, roll_deg), ...] (pitch / roll None = keep the previous
        value). The path starts at q_start (default: the measured pose) and visits the points
        in order; position, pitch and roll are interpolated linearly along each segment with
        at least one waypoint per step_m of travel and per step_deg of pitch or roll change
        (KI-6). Raises SafetyError when a waypoint has no IK solution or two consecutive
        waypoints are more than 12 deg apart in pan / lift / elbow / wrist_flex (IK branch
        change or a near-singular pose). Returns [q_start, q_1, ...] for run_path."""
        q0 = self.read_q() if q_start is None else np.array(q_start, dtype=float)
        tool = self.tip_tool()
        p_prev = K.fk(q0, tool)[:3, 3].copy()
        pitch_prev = K.pose_summary(q0, tool)["pitch_down_deg"]
        roll_prev = math.degrees(q0[4])
        wps = [q0]
        seed = q0
        for idx, pt in enumerate(points):
            xyz, pitch, roll = pt
            xyz = np.asarray(xyz, dtype=float)
            pitch = pitch_prev if pitch is None else float(pitch)
            roll = roll_prev if roll is None else float(roll)
            if xyz.shape != (3,) or not (np.all(np.isfinite(xyz)) and math.isfinite(pitch) and math.isfinite(roll)):
                raise SafetyError(f"path point {idx}: xyz / pitch / roll must be finite numbers")
            n = max(1, int(math.ceil(float(np.linalg.norm(xyz - p_prev)) / step_m)),
                    int(math.ceil(abs(pitch - pitch_prev) / step_deg)),
                    int(math.ceil(abs(roll - roll_prev) / step_deg)))
            for k in range(1, n + 1):
                a = k / n
                q_k = self.solve(p_prev + (xyz - p_prev) * a, pitch_prev + (pitch - pitch_prev) * a,
                                 roll_prev + (roll - roll_prev) * a, q_seed=seed)
                jump = math.degrees(float(np.max(np.abs(q_k[:4] - seed[:4]))))
                if jump > 12.0:
                    raise SafetyError(f"joint step of {jump:.1f} deg between waypoints {k - 1} and {k} of {n} on "
                                      f"segment {idx} (IK branch change or near-singular pose)")
                wps.append(q_k)
                seed = q_k
            p_prev, pitch_prev, roll_prev = xyz, pitch, roll
        return wps

    def move_tip_path(self, points, speed_deg_s: float, **kw):
        """plan_tip_path + run_path: the fingertip visits the points along straight segments
        as ONE continuous motion (no stop at intermediate points)."""
        return self.run_path(self.plan_tip_path(points), speed_deg_s, **kw)

    def move_tip(self, xyz, pitch_down_deg: float, roll_deg: float | None = None, speed_deg_s: float = 30.0,
                 straight: bool = False, step_m: float = 0.005, **kw):
        """Move the fingertip to xyz (base frame, metres) with the approach axis pitched
        pitch_down_deg below horizontal. straight=True follows a Cartesian line."""
        q_now = self.read_q()
        if roll_deg is None:
            roll_deg = math.degrees(q_now[4])
        xyz = np.asarray(xyz, dtype=float)
        if not straight:
            q_t = self.solve(xyz, pitch_down_deg, roll_deg, q_seed=q_now)
            return self.run_path([q_now, q_t], speed_deg_s, **kw)
        wps = self.plan_tip_path([(xyz, pitch_down_deg, roll_deg)], q_start=q_now, step_m=step_m)
        return self.run_path(wps, speed_deg_s, **kw)

    def ready_q(self) -> np.ndarray:
        """Joint pose of the stand-by pose cfg["ready"] (fingertip xyz, pitch, roll). With a tool
        attached: the TOOL tip at that point, pitched as for working with the tool (tooling.ready_q)."""
        if self.tool is not None:
            import tooling
            return tooling.ready_q(self, self.tool)
        r = self._c("ready")
        return self.solve(np.array(r["xyz"], dtype=float), float(r["pitch"]), float(r["roll"]))

    def rest_q(self) -> np.ndarray:
        """The folded parking pose cfg["rest_raw"], clamped 0.005 rad inside the joint limits
        (the physical rest pose sits on the mechanical stops, just outside them). With a tool attached:
        the tool's rest pose (tooling.rest_q: the same fold with the wrist raised until the tool tip is
        5 cm above the table; SafetyError when the arm cannot rest with that tool)."""
        if self.tool is not None:
            import tooling
            return tooling.rest_q(self, self.tool)
        lim = self.q_limits()
        q = self.raw_to_q(self._c("rest_raw"))
        return np.array([min(max(q[i], lim[j][0] + 0.005), lim[j][1] - 0.005) for i, j in enumerate(ARM)])

    # ------------------------------------------------------------ gripper
    def _tool_lock(self, what: str):
        """Refuse a gripper command while a tool is taped to the gripper (before any bus traffic): opening
        the jaws drops or bends the tool, closing them squeezes it. Unlocked by `fd tool detach`."""
        if self.tool is not None:
            import tooling
            raise SafetyError(tooling.lock_message(self.tool, what))

    def _gripper_clamp(self, raw=None):
        """Allowed gripper goal range (lo, hi); with `raw`, that goal clamped into it. The lower
        end is where the fingertips touch (cfg["gripper_touch_raw"]): a goal below it only
        stalls the servo against its own jaw (MS-9). With a tool attached no gripper goal exists:
        asking for one raises SafetyError, so nothing built on it (run_path / check_path with
        gripper_goal_raw, gripper_raw_for) can send the gripper anywhere."""
        if raw is not None:
            self._tool_lock("a gripper goal")
        g = self.cfg["gripper"]
        lo = max(g["closed_raw"] + g["margin"], int(self._c("gripper_touch_raw")))
        hi = g["open_raw"] - g["margin"]
        if raw is None:
            return lo, hi
        if not math.isfinite(float(raw)):
            raise SafetyError(f"gripper goal {raw} is not a number")
        return int(round(max(lo, min(hi, float(raw)))))

    def gripper_raw_for(self, frac: float) -> int:
        g = self.cfg["gripper"]
        if not math.isfinite(float(frac)):
            raise SafetyError(f"gripper fraction {frac} is not a number")
        return self._gripper_clamp(g["closed_raw"] + float(frac) * (g["open_raw"] - g["closed_raw"]))

    def gripper_q_deg(self, raw) -> float:
        """URDF jaw angle (deg) of a raw gripper position: -11.41 deg where the tips touch."""
        return JAW_Q_TOUCH_DEG + (float(raw) - float(self._c("gripper_touch_raw"))) / TICKS_PER_DEG

    def gripper_raw_for_q(self, q_deg: float) -> int:
        return int(round(float(self._c("gripper_touch_raw")) + (float(q_deg) - JAW_Q_TOUCH_DEG) * TICKS_PER_DEG))

    def _gripper_read(self):
        """(position, load, status) of the gripper from one bus transaction."""
        if self._c("stream_feedback") and hasattr(self.bus, "feedback"):
            fb = self.bus.feedback(("gripper",))["gripper"]
            pos, load, status = fb["pos"], fb["load"], fb["status"]
        else:
            pos = self.bus.read("gripper", "Present_Position")
            load, status = self.bus.read("gripper", "Present_Load"), 0
        status |= int((getattr(self.bus, "status", None) or {}).get("gripper", 0))
        self._jaw_raw = pos
        return pos, load, status

    def _gripper_fault(self, status: int, pos: int):
        if status & int(self._c("fault_abort_mask")):
            self.bus.write("gripper", "Goal_Position", self._gripper_clamp(pos))  # stop squeezing
            raise MotionAborted(f"gripper servo fault: {', '.join(self._fault_names(status))} (status {status}); "
                                f"goal set to the present position",
                                info={"reason": "servo_fault", "faults": {"gripper": status}})

    def set_gripper(self, frac: float | None = None, wait_s: float = 1.2, raw: int | None = None,
                    wait: bool = True) -> dict:
        """Open fraction 0 (closed) .. 1 (fully open), or a raw goal (overrides frac).

        wait=False returns at once (the jaw moves on its own). Otherwise the position is
        watched until the jaw is at the goal, has stopped short of it, or wait_s is over.
        A jaw that stops more than gripper_blocked_ticks short is blocked: the goal is pulled
        back to squeeze_ticks beyond where it stands (bounded holding force instead of a
        stalled servo) and the result says "blocked": True (MS-9)."""
        self._tool_lock("moving the gripper (set_gripper)")
        if raw is None and frac is None:
            raise SafetyError("set_gripper needs an open fraction or a raw goal")
        if self.bus.read("gripper", "Torque_Enable") != 1:
            raise SafetyError("gripper torque is off; run torque_on first")
        target = self._gripper_clamp(raw) if raw is not None else self.gripper_raw_for(frac)
        self.bus.write("gripper", "Goal_Position", target)
        if not wait:
            return {"target_raw": target, "waited": False}
        t0 = self._clock()
        hist = []
        done = blocked = False
        pos = load = None
        while True:
            self._sleep(0.04)
            pos, load, status = self._gripper_read()
            self._gripper_fault(status, pos)
            t = self._clock() - t0
            hist.append(pos)
            if len(hist) > 2000:
                raise SafetyError("set_gripper: the clock is not advancing")
            still = len(hist) >= 3 and max(hist[-3:]) - min(hist[-3:]) <= 2
            if still and abs(pos - target) < 25:
                done = True
                break
            still4 = len(hist) >= 4 and max(hist[-4:]) - min(hist[-4:]) <= 2
            if still4 and t >= 0.3 and abs(pos - target) > int(self._c("gripper_blocked_ticks")):
                blocked = True  # (the first 0.3 s are not judged: the servo needs a moment to start)
                break
            if t >= wait_s:
                break
        out = {"target_raw": target, "pos_raw": pos, "open_frac": round(self.gripper_frac(pos), 3), "load": load,
               "blocked": blocked, "moving": not (done or blocked), "duration_s": round(self._clock() - t0, 3)}
        if blocked:
            sq = int(self._c("squeeze_ticks"))
            hold = self._gripper_clamp(pos - sq if target < pos else pos + sq)
            self.bus.write("gripper", "Goal_Position", hold)
            out["hold_goal_raw"] = hold
        return out

    def _jaw_width_mm(self, q_jaw_deg: float) -> float:
        """Opening between the fixed pad and the back corner of the moving pad (mm): the width
        of a flat-faced object standing against the fixed jaw when the jaw touches it."""
        return float(GM.tip_opening(q_jaw_deg)[0])

    def grasp(self, min_frac: float = 0.0, step_frac: float = 0.03, squeeze_ticks: int | None = None,
              stall_ticks: int = 6, max_s: float = 6.0):
        """Close in small steps until the jaws stop following (object contact), then command
        a small extra squeeze so the soft P-loop keeps a holding force. (grasp_fast does the
        same in one command.)"""
        self._tool_lock("closing the gripper (grasp)")
        if self.bus.read("gripper", "Torque_Enable") != 1:
            raise SafetyError("gripper torque is off; run torque_on first")
        squeeze = int(self._c("squeeze_ticks")) if squeeze_ticks is None else int(squeeze_ticks)
        g = self.cfg["gripper"]
        span = g["open_raw"] - g["closed_raw"]
        pos = self.bus.read("gripper", "Present_Position")
        floor_raw = self.gripper_raw_for(min_frac)
        if pos <= floor_raw:
            # closing "to min_frac" from below it would OPEN the gripper (and drop what it holds)
            self._jaw_raw = pos
            return {"contact": False, "pos_raw": pos, "hold_goal_raw": None,
                    "open_frac": round(self.gripper_frac(pos), 3), "load": self.bus.read("gripper", "Present_Load"),
                    "note": "already at or below min_frac; nothing was commanded"}
        goal = pos
        t0 = self._clock()
        stalled = 0
        history = []
        while self._clock() - t0 < max_s:
            goal = max(floor_raw, goal - int(step_frac * span))
            self.bus.write("gripper", "Goal_Position", goal)
            self._sleep(0.22)
            new, load, status = self._gripper_read()
            self._gripper_fault(status, new)
            history.append((goal, new, load))
            lag = new - goal
            moved = pos - new
            pos = new
            if lag > 3 * stall_ticks and moved < stall_ticks:
                stalled += 1
                if stalled >= 2:
                    break
            else:
                stalled = 0
            if goal <= floor_raw:
                break
        # the jaw standing far above the last goal is held open by something, however the loop ended
        contact = stalled >= 2 or (pos - goal) > 3 * stall_ticks
        hold = max(floor_raw, pos - squeeze) if contact else goal
        self.bus.write("gripper", "Goal_Position", hold)
        self._sleep(0.3)
        pos, load, status = self._gripper_read()
        return {"contact": contact, "pos_raw": pos, "hold_goal_raw": hold,
                "open_frac": round(self.gripper_frac(pos), 3), "load": load, "status": status,
                "current": self.bus.read("gripper", "Present_Current"), "history": history[-6:]}

    def pinch(self, extra_ticks: int | None = None, wait_s: float = 1.2) -> dict:
        """Hold something THIN (a sheet, the wall of a cup or of a bottle cap): thinner than the
        jaws' own closing error, so that grasp()/grasp_fast() cannot press on it.

        The jaws close to the clamp (where the fingertips touch); the goal is then set
        `extra_ticks` beyond where the jaw stopped, so the soft position loop presses with the
        same bounded force as a normal hold. The goal may lie below the closed clamp, but by no
        more than extra_ticks (<= squeeze_ticks): with nothing between the jaws the servo then
        stalls against its own jaw with at most the normal holding error, which is what MS-9
        protects against. Nothing is detected here (a thin object and empty jaws stop at almost
        the same position): check the result with the camera after lifting."""
        self._tool_lock("closing the gripper (pinch)")
        if self.bus.read("gripper", "Torque_Enable") != 1:
            raise SafetyError("gripper torque is off; run torque_on first")
        squeeze = int(self._c("squeeze_ticks"))
        extra = squeeze if extra_ticks is None else int(extra_ticks)
        if not 0 < extra <= squeeze:
            raise SafetyError(f"pinch: extra_ticks must be 1..{squeeze} (squeeze_ticks)")
        lo, _ = self._gripper_clamp()
        g = self.cfg["gripper"]
        first = self.set_gripper(raw=lo, wait_s=wait_s)
        pos, load, status = self._gripper_read()
        self._gripper_fault(status, pos)
        hold = int(max(pos - extra, lo - extra, g["closed_raw"] + g["margin"]))
        self.bus.write("gripper", "Goal_Position", hold)
        self._sleep(0.4)
        pos2, load2, status2 = self._gripper_read()
        self._gripper_fault(status2, pos2)
        self._jaw_raw = pos2
        return {"pos_raw": pos2, "pos_before_squeeze": pos, "hold_goal_raw": hold, "load": load2, "status": status2,
                "above_touch_ticks": pos2 - int(self._c("gripper_touch_raw")), "blocked_while_closing": bool(first.get("blocked")),
                "note": "thin hold: not verified by the jaw position; look with the camera"}

    def grasp_fast(self, expected_raw: int | None = None, squeeze_ticks: int | None = None, timeout_s: float = 1.5,
                   empty_margin: int = 45, hz: float = 50.0) -> dict:
        """Close in ONE command and watch the position at loop rate; a jaw that stops short of
        the closed position is held open by an object.

        On contact the goal is set to pos - squeeze_ticks (never below the closed clamp), which
        leaves a bounded holding force. "empty" = the jaws ended within empty_margin ticks of
        where the fingertips touch (nothing, or something thinner than about 5 mm, between
        them). Returns {"contact", "pos_raw", "hold_goal_raw", "q_jaw_deg", "width_mm", "empty",
        "duration_s", "load", "status", "stalled", "arrived", "timeout", "start_raw", "moved_ticks",
        "expected_raw", "delta_raw"}; width_mm is the geometric pad opening at pos_raw (it reads a
        few mm narrow under the squeeze). "contact" with moved_ticks near 0 means the jaw never
        moved: an object was already clamped, or the gripper servo does not respond."""
        self._tool_lock("closing the gripper (grasp_fast)")
        if self.bus.read("gripper", "Torque_Enable") != 1:
            raise SafetyError("gripper torque is off; run torque_on first")
        squeeze = int(self._c("squeeze_ticks")) if squeeze_ticks is None else int(squeeze_ticks)
        if squeeze < 0 or not (math.isfinite(timeout_s) and timeout_s > 0):
            raise SafetyError("grasp_fast: squeeze_ticks must be >= 0 and timeout_s > 0")
        lo, _ = self._gripper_clamp()
        touch = int(self._c("gripper_touch_raw"))
        pos0, load, status = self._gripper_read()
        dt = 1.0 / hz
        t0 = self._clock()
        self.bus.write("gripper", "Goal_Position", lo)
        hist = [(0.0, pos0)]
        stalled = arrived = timed_out = False
        pos, t = pos0, 0.0
        while True:
            self._sleep(dt)
            pos, load, status = self._gripper_read()
            self._gripper_fault(status, pos)
            t = self._clock() - t0
            hist.append((t, pos))
            if len(hist) > 5000:
                raise SafetyError("grasp_fast: the clock is not advancing")
            win = [p for (tt, p) in hist if tt >= t - 0.1]
            moved = pos0 - min(p for _, p in hist) >= 5
            # a standstill counts only once the jaw has started to close, or when it has not started
            # 0.35 s after the command (it is already against something, or already closed)
            if (moved or t >= 0.35) and len(win) >= 4 and max(win) - min(win) <= 3:
                if pos - lo > 12:
                    stalled = True  # stopped short of the goal: something holds the jaw open
                else:
                    arrived = True
                break
            if t >= timeout_s:
                timed_out = True
                break
        empty = pos <= touch + int(empty_margin)
        # a jaw that has not reached the closed position when the time is up (something soft that
        # keeps yielding) is held open by an object just like one that stalled
        contact = (stalled or timed_out) and not empty
        if contact:
            hold = max(lo, pos - squeeze)
        else:
            hold = max(lo, pos - 5) if (stalled or timed_out) else lo  # jaws on each other: no standing load
        if hold != lo:
            self.bus.write("gripper", "Goal_Position", hold)
        q_jaw = self.gripper_q_deg(pos)
        return {"contact": contact, "pos_raw": pos, "hold_goal_raw": hold, "q_jaw_deg": round(q_jaw, 2),
                "width_mm": round(self._jaw_width_mm(q_jaw), 1), "empty": empty, "duration_s": round(t, 3),
                "load": load, "status": status, "stalled": stalled, "arrived": arrived, "timeout": timed_out,
                "start_raw": pos0, "moved_ticks": pos0 - pos,
                "open_frac": round(self.gripper_frac(pos), 3),
                "expected_raw": None if expected_raw is None else int(expected_raw),
                "delta_raw": None if expected_raw is None else pos - int(expected_raw)}

    # ------------------------------------------------------------ probing
    def probe_down(self, max_drop_m: float = 0.06, speed_m_s: float = 0.006, backoff_m: float = 0.015,
                   gap_m: float = 0.004, hold_s: float = 0.35, push_deg: float = 2.0):
        """Lower the fingertip straight down slowly until it lands on something.

        The descent is tracked with the outer integral loop, so in free air the measured tip
        follows the commanded height within about a millimetre once it is moving. When the
        tip lands, the measured height falls behind the commanded one. Contact is declared
        when that gap exceeds its free-air level by `gap_m` for `hold_s`; the reported
        contact height is the measured height at the ONSET of the gap growth (after landing
        the gears run through their backlash and the tip slides, so later readings sink
        below the true surface). The integral term may push down by at most `push_deg`
        beyond the free-air compensation, which keeps the contact force small.

        Result "contact": True (surface found, "contact_tip_z"), False (free travel down to
        "lowest_tip_z"; a surface in the last few mm of the range cannot be excluded) or None
        with a "reason" (inconclusive: the tip was lagging when the descent ended, or no
        free-air baseline could be formed). A descent too short or too fast for the detector
        is refused before any motion (MS-2)."""
        snap = self._require_torque()
        warm_s = 1.2  # start-up stiction: the arm needs a moment to break free
        if not (math.isfinite(speed_m_s) and 0.002 <= speed_m_s <= 0.012):
            raise SafetyError(f"probe speed {speed_m_s} m/s outside 0.002..0.012")
        if not (math.isfinite(max_drop_m) and max_drop_m > 0):
            raise SafetyError(f"probe max_drop {max_drop_m} must be a positive number")
        duration = max_drop_m / speed_m_s
        # baseline (warm-up + 0.5 s + 15 samples at >= 25 ms) + gap_m of extra lag + hold_s
        t_need = warm_s + 0.5 + 0.4 + hold_s + gap_m / speed_m_s
        if duration < t_need:
            raise SafetyError(
                f"probe of {max_drop_m * 1000:.0f} mm at {speed_m_s * 1000:.1f} mm/s lasts {duration:.1f} s, but "
                f"contact cannot be detected in less than {t_need:.1f} s: use max_drop >= {t_need * speed_m_s:.3f} m "
                f"(and start at least {(warm_s + 0.9) * speed_m_s * 1000:.0f} mm above the expected surface)")
        q0 = self.raw_to_q({j: snap[j]["pos"] for j in ARM})
        tool = self.tip_tool()
        start = K.fk(q0, tool)[:3, 3].copy()
        pitch = K.pose_summary(q0, tool)["pitch_down_deg"]
        roll = math.degrees(q0[4])
        step = 0.002
        n = max(1, int(round(max_drop_m / step)))
        wps = [q0]
        seed = q0
        ctx = self._check_ctx(q0)
        for k in range(1, n + 1):
            q_k = self.solve(start + np.array([0.0, 0.0, -k * step]), pitch, roll, q_seed=seed)
            if math.degrees(float(np.max(np.abs(q_k - seed)))) > 6.0:
                raise SafetyError("IK jump on the probe path")
            self._check_pose(q_k, ctx, True, (), "probe path", math.inf)
            wps.append(q_k)
            seed = q_k

        def plan(t):
            a = max(0.0, min(1.0, t / duration)) * n
            i = min(n - 1, int(a))
            return wps[i] + (wps[i + 1] - wps[i]) * (a - i)

        integ0 = self._current_offset(snap)
        self._comp_hint = integ0.copy()
        push = math.radians(push_deg)
        # Bounded push: per joint, cap the correction on the side that moves the fingertip DOWN
        # (sign of d z_tip / d q at both ends of the descent; where it flips, as wrist_flex does
        # near pitch 88 deg, cap both sides). Upward holding authority stays as in any motion.
        lo, hi = self._path_bounds(wps, float(self._c("friction_corr_max_deg")))
        for i in (1, 2, 3):
            sg = []
            for qq in (wps[0], wps[-1]):
                d = np.zeros(5)
                d[i] = 1e-4
                g = (K.fk(qq + d, tool)[2, 3] - K.fk(qq - d, tool)[2, 3]) / 2e-4
                sg.append(0.0 if abs(g) < 0.005 else math.copysign(1.0, g))
            if sg[0] <= 0 or sg[1] <= 0:  # +q lowers the tip somewhere on the path
                hi[i] = integ0[i] + push
            if sg[0] >= 0 or sg[1] >= 0:  # -q lowers the tip somewhere on the path
                lo[i] = integ0[i] - push
        hist = []  # (t, z_plan, z_meas)
        found = {}
        base = {}

        def monitor(t, q_plan, q_meas, integ):
            z_meas = float(K.fk(q_meas, tool)[2, 3])
            z_plan = float(K.fk(q_plan, tool)[2, 3])
            hist.append((t, z_plan, z_meas))
            steady = [zm - zp for (tt, zp, zm) in hist if warm_s <= tt <= t - 0.5]
            if len(steady) < 15:
                return False
            g0 = float(np.median(steady))
            base["g0"] = g0
            recent = [(zm - zp) for (tt, zp, zm) in hist if tt > t - hold_s]
            if min(recent) > g0 + gap_m:
                # onset: last sample whose gap was still at the free-air level
                onset = None
                for (tt, zp, zm) in hist:
                    if tt >= warm_s and (zm - zp) < g0 + 0.001:
                        onset = (tt, zm)
                found.update(z_stall=z_meas, g0=g0, t=t)
                if onset is not None:
                    near = [zm for (tt, zp, zm) in hist if abs(tt - onset[0]) <= 0.06]
                    found.update(z=float(np.mean(near)), t_onset=onset[0])
                return True
            return False

        error = None
        res = None
        try:
            res = self._track(plan, duration, integ0, monitor=monitor, integ_lo=lo, integ_hi=hi)
        except (SafetyError, BusError, MotionAborted) as e:  # never leave the arm pressing down
            error = e
        reason = None
        if "z" not in found and hist:
            tail = [zm - zp for (tt, zp, zm) in hist[-5:]]
            if "g0" not in base:
                reason = "inconclusive: no free-air baseline could be formed"
            elif float(np.median(tail)) > base["g0"] + 0.5 * gap_m:
                reason = (f"inconclusive: the tip was {1000 * (float(np.median(tail)) - base['g0']):.1f} mm behind "
                          f"the plan at the end of the descent (a surface is probably within the last "
                          f"{1000 * (gap_m + hold_s * speed_m_s):.0f} mm of the range); probe again with a larger "
                          f"max_drop")
        # release the pressure first (goal := measured pose + free-air compensation), then back off straight up
        if res is not None:
            self._hold(res["q_meas"], res["integ_start"])
        else:
            self._hold_here()
        self._sleep(0.2)
        q_now = self.read_q()
        up_from = K.fk(q_now, tool)[:3, 3]
        q_up = self.solve(up_from + np.array([0.0, 0.0, backoff_m]), pitch, roll, q_seed=q_now)
        back = self.run_path([q_now, q_up], 10.0, allow_contact=True, contact_depth=math.inf)
        if error is not None:
            raise error
        trace = [(round(t, 2), round(zp, 4), round(zm, 4)) for (t, zp, zm) in hist[::10]]
        contact = True if "z" in found else (None if reason else False)
        return {"contact": contact, "reason": reason,
                "contact_tip_z": found.get("z"), "stall_tip_z": found.get("z_stall"),
                "free_air_gap": found.get("g0", base.get("g0")), "t_onset": found.get("t_onset"),
                "lowest_tip_z": round(min(zm for (_, _, zm) in hist), 4) if hist else None,
                "xy": [round(float(start[0]), 4), round(float(start[1]), 4)],
                "start_z": round(float(start[2]), 4), "trace": trace, "final": back["joint_err_deg"]}

    # ------------------------------------------------------------ benchmark
    def bench_moves(self) -> list:
        """Default move list of bench_motion: fingertip poses 9 cm or more above the table
        around the ready pose (nothing on the table is that tall), each at one of the tier's
        speeds: a joint-space transit, a 90 deg wrist-roll swing (the joint with a speed scale),
        a straight descent, a short precise slide, one continuous lift / transfer / lower path
        that rolls back on the way, and the return."""
        r = self._c("ready")
        ready = (tuple(r["xyz"]), float(r["pitch"]), float(r["roll"]))
        return [
            {"name": "transit_out", "kind": "joints", "speed": "transit", "to": ((0.24, -0.07, 0.10), 75.0, 20.0)},
            {"name": "roll_swing_90", "kind": "joints", "speed": "transit", "to": ((0.24, -0.07, 0.10), 75.0, -70.0)},
            {"name": "descent_30mm", "kind": "path", "speed": "approach", "points": [((0.24, -0.07, 0.07), 75.0, -70.0)]},
            {"name": "precise_15mm", "kind": "path", "speed": "precise", "points": [((0.24, -0.055, 0.07), 75.0, -70.0)]},
            {"name": "lift_transfer_lower", "kind": "path", "speed": "transit",
             "points": [((0.24, -0.055, 0.11), 75.0, -70.0), ((0.22, 0.08, 0.11), 75.0, 20.0),
                        ((0.22, 0.08, 0.08), 75.0, 20.0)]},
            {"name": "transit_back", "kind": "joints", "speed": "transit", "to": ready},
        ]

    def bench_motion(self, tier: str, moves: list | None = None) -> dict:
        """Free-space benchmark of a speed tier (the one place an unvalidated tier may run).

        Refuses to start unless all servos are torqued and the arm is within
        cfg["bench"]["start_tol_deg"] of the ready pose (wrist roll free). Applies the tier's servo profile,
        runs the moves (bench_moves() by default; every pose is planned and checked before
        the first one starts), and measures per move: duration, peak tracking error,
        overshoot, settle time, final error, peak load; overall: temperature rise and lowest
        supply voltage. Each figure is compared with cfg["bench"]; the benchmark stops at the
        first move that fails or aborts (also: load above abort_load for abort_load_s, rail
        below abort_volt, temperature above max_temp_c). The previous profile is restored and,
        unless a fault stopped it, the arm returns to the ready pose at the previous tier.

        Returns {"tier", "pass" (= "passed"), "aborted", "reason", "fails", "moves": [...], "peak_err_deg",
        "overshoot_deg", "settle_s", "settle_1deg_s", "peak_load", "temp_rise_c", "min_volt",
        "duration_s"}.
        A pass does NOT edit the config: add the tier to speed_tiers_validated by hand."""
        th = dict(DEFAULT_CONFIG["bench"])
        th.update(self.cfg.get("bench", {}))
        sp = self.speeds(tier, allow_unvalidated=True)
        snap = self._require_torque()
        q_now = self.raw_to_q({j: snap[j]["pos"] for j in ARM})
        q_ready = self.ready_q()
        off = np.abs(q_now - q_ready)[:4]  # the wrist roll is free at stand-by (it does not move the fingertip)
        if float(np.max(off)) > math.radians(th["start_tol_deg"]):
            w = int(np.argmax(off))
            raise SafetyError(f"bench_motion starts from the ready pose: {ARM[w]} is {math.degrees(off[w]):.1f} deg "
                              f"away (limit {th['start_tol_deg']}); move there first")
        # plan and check everything before the first move
        plans = []
        q_from = q_now
        ctx = self._check_ctx(q_now)
        for m in (self.bench_moves() if moves is None else moves):
            if not isinstance(m, dict) or not {"name", "kind", "speed"} <= set(m):
                raise SafetyError(f"bench move {m!r}: needs 'name', 'kind' and 'speed'")
            if m["speed"] not in ("transit", "approach", "precise"):
                raise SafetyError(f"bench move {m.get('name')}: speed must be transit / approach / precise")
            if m["kind"] == "joints":
                xyz, pitch, roll = m["to"]
                wps = [q_from, self.solve(np.array(xyz, dtype=float), pitch, roll)]
            elif m["kind"] == "path":
                wps = self.plan_tip_path(m["points"], q_start=q_from)
            else:
                raise SafetyError(f"bench move {m.get('name')}: kind must be 'joints' or 'path'")
            for i in range(len(wps) - 1):
                n = max(1, int(math.ceil(math.degrees(float(np.max(np.abs(wps[i + 1] - wps[i])))) / 1.5)))
                for k in range(1, n + 1):
                    self._check_pose(wps[i] + (wps[i + 1] - wps[i]) * (k / n), ctx, False, (),
                                     f"bench move {m['name']}", None)
            plans.append((m, wps))
            q_from = wps[-1]
        prev = self._active
        temp0 = {j: snap[j]["temp"] for j in JOINTS}
        out = {"tier": sp["tier"], "validated": sp["validated"],
               "speeds": {k: sp[k] for k in ("transit", "approach", "precise")},
               "servo_velocity_cap": sp["servo_velocity_cap"], "servo_acceleration": sp["servo_acceleration"],
               "time_law": sp["time_law"], "joint_speed_scale": sp["joint_speed_scale"],
               "thresholds": th, "moves": [], "fails": [], "pass": False, "passed": False, "aborted": False,
               "reason": None}
        over = {"since": None, "why": None}

        def monitor(t, q_plan, q_meas):
            fb = self._last_fb
            if fb is None:
                return False
            if min(fb[j]["volt"] for j in JOINTS) < th["abort_volt"]:
                over["why"] = f"supply voltage below {th['abort_volt']} V"
                return True
            if max(fb[j]["temp"] for j in JOINTS) > self.cfg["max_temp_c"]:
                over["why"] = f"temperature above {self.cfg['max_temp_c']} C"
                return True
            if max(abs(fb[j]["load"]) for j in ARM) > th["abort_load"]:
                over["since"] = t if over["since"] is None else over["since"]
                if t - over["since"] >= th["abort_load_s"]:
                    over["why"] = f"load above {th['abort_load']} for {th['abort_load_s']} s"
                    return True
            else:
                over["since"] = None
            return False

        t_start = self._clock()
        faulted = False
        temp_max = dict(temp0)
        try:
            self.apply_profile(sp["tier"], allow_unvalidated=True)
            for m, wps in plans:
                over["since"] = None
                rec = {"name": m["name"], "speed_deg_s": sp[m["speed"]]}
                try:
                    r = self.run_path(wps, sp[m["speed"]], settle=True, monitor=monitor)
                except MotionAborted as e:
                    rec.update(ok=False, aborted=True, reason=over["why"] or str(e))
                    out["moves"].append(rec)
                    out.update(aborted=True, reason=f"{m['name']}: {rec['reason']}")
                    faulted = e.info.get("reason") in ("servo_fault", "tracking_error", "target_not_reached")
                    break
                except (SafetyError, BusError) as e:
                    rec.update(ok=False, aborted=True, reason=f"{type(e).__name__}: {e}")
                    out["moves"].append(rec)
                    out.update(aborted=True, reason=f"{m['name']}: {rec['reason']}")
                    faulted = True
                    break
                final = max(abs(v) for v in r["joint_err_deg"])
                rec.update(duration_s=r["duration_s"], peak_err_deg=r["peak_err_deg"],
                           peak_err_joint_deg=r["peak_err_joint_deg"], overshoot_deg=r["overshoot_deg"],
                           settle_s=r["settle_s"], settle_1deg_s=r["settle_1deg_s"], final_err_deg=round(final, 2),
                           peak_load=r["peak_load"], min_volt=r["min_volt"], loop_hz=r["loop_hz"])
                fails = []
                for key, val in (("peak_err_deg", r["peak_err_deg"]), ("overshoot_deg", r["overshoot_deg"]),
                                 ("settle_1deg_s", r["settle_1deg_s"]), ("final_err_deg", final),
                                 ("peak_load", r["peak_load"])):
                    if val is not None and val > th[key]:
                        fails.append(f"{key} {val} above {th[key]}")
                if r["min_volt"] is not None and r["min_volt"] < th["min_volt"]:
                    fails.append(f"min_volt {r['min_volt']} below {th['min_volt']}")
                if r["loop_hz"] is not None and r["loop_hz"] < th["min_loop_hz"]:
                    fails.append(f"loop_hz {r['loop_hz']} below {th['min_loop_hz']} (bus retries / fallbacks?)")
                if self._last_fb is not None:
                    for j in JOINTS:
                        temp_max[j] = max(temp_max[j], self._last_fb[j]["temp"])
                rise = max(temp_max[j] - temp0[j] for j in JOINTS)
                if rise > th["temp_rise_c"]:
                    fails.append(f"temperature rise {rise} C above {th['temp_rise_c']}")
                rec.update(ok=not fails, fails=fails)
                out["moves"].append(rec)
                if fails:
                    out["fails"] += [f"{m['name']}: {f}" for f in fails]
                    out["reason"] = f"{m['name']} failed its thresholds; benchmark stopped"
                    break
        finally:
            # back to the profile that was in force before, whatever happened
            try:
                self.apply_profile(prev["tier"] if prev else None, allow_unvalidated=prev is not None)
                self._active = prev
            except (SafetyError, BusError) as e:
                out["profile_restore_error"] = f"{type(e).__name__}: {e}"
        done = [m for m in out["moves"] if "peak_err_deg" in m]
        out["peak_err_deg"] = max((m["peak_err_deg"] for m in done), default=None)
        out["overshoot_deg"] = max((m["overshoot_deg"] for m in done), default=None)
        out["settle_s"] = max((m["settle_s"] for m in done), default=None)
        out["settle_1deg_s"] = max((m["settle_1deg_s"] for m in done), default=None)
        out["peak_load"] = max((m["peak_load"] for m in done), default=None)
        volts = [m["min_volt"] for m in done if m["min_volt"] is not None]
        out["min_volt"] = min(volts) if volts else None
        out["temp_rise_c"] = max(temp_max[j] - temp0[j] for j in JOINTS)
        hz = [m["loop_hz"] for m in done if m.get("loop_hz") is not None]
        out["loop_hz"] = min(hz) if hz else None
        out["bus_stats"] = dict(getattr(self.bus, "stats", {}) or {})
        complete = len(done) == len(plans) and not out["aborted"]
        out["pass"] = out["passed"] = bool(complete and not out["fails"])
        if not complete and not faulted and "profile_restore_error" not in out:
            # stopped on a threshold or an early-abort limit: return to the ready pose at the previous tier
            try:
                back = self.speeds(prev["tier"] if prev else None, allow_unvalidated=prev is not None)
                self.move_joints(q_ready, back["transit"])
                out["returned_to_ready"] = True
            except (SafetyError, MotionAborted, BusError) as e:
                out["returned_to_ready"] = False
                out["return_error"] = f"{type(e).__name__}: {e}"
        out["duration_s"] = round(self._clock() - t_start, 2)
        return out
