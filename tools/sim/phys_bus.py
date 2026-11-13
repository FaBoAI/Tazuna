"""PhysBus: a torque-level servo model with the SimBus interface (SIMULATION ONLY).

Origin: the motion-safety verifier's model (refs/patches/verify/blocked-windup/phys_bus.py with
its gravity.py / inertials.json), made importable from the project. Its default parameters are
the ones the verifier calibrated against the real run log (runs/20261110_230151/events.jsonl):

    tau_stall = 3.0 N m, tau_fric = 0.25 N m   reproduce the logged free-air holding offsets at
                                               P = 32 (lift -1.4 .. -2.4 deg, elbow about -2 deg)
    kp = 8.2 permille PWM per tick at P = 32   Present_Load / (goal - present) in the log
                                               (4.1 at P = 16; saturation near 122 ticks)
    k_contact about 1 N/mm                     reproduces the ~5 mm (encoder terms) the real table
                                               probes sank between contact onset and detection

THE CALIBRATION DATA ARE STATIC HOLDS AND MOTIONS OF 25 deg/s OR LESS (most of them 8-20 deg/s).
Anything this model says about faster motion is a prediction, not a measurement.

Model per arm joint (URDF radians, config sign / zero tick):
    u       = clip(kp * (ref - pos), -1, 1)          PWM fraction; ref = Goal_Position, or the
                                                     servo-internal profiled setpoint (servo_profile)
    tau_m   = u * tau_stall                          motor torque at the joint
    tau_net = tau_m + tau_gravity(q) + tau_contact(q)
  quasi-static (default, the verifier's calibrated model; gear-train inertia neglected):
    omega   = 0                                      if |tau_net| <= tau_fric (Coulomb, static = kinetic)
            = omega0 * (tau_net -+ tau_fric) / tau_stall   otherwise (DC-motor torque / speed line),
              capped at the Goal_Velocity register
  inertia=True (extension, NOT calibrated: no dynamic trace exists in the log):
    J(q) * d(omega)/dt = tau_net -+ tau_fric - (tau_stall / omega0) * omega
    with J(q) = j_rotor + sum of m * r^2 of the link masses carried by the joint (diagonal
    approximation, no coupling between joints). The quasi-static model is its J -> 0 limit; only
    this variant can overshoot.
Gravity: URDF link masses and centres of mass (INERTIALS below, from so101_new_calib.urdf).
Contact: a one-sided vertical spring acting on the fingertip, F = k_contact * (z_surface - z_tip)
[N], standing in for everything that yields when the tip is stopped (link / bracket flex, base
clamp, mat). k_contact -> large = rigid.
Gripper: SimBus' kinematic follower (optionally with the servo profile); `blocker` acts on it.

Parameters (constructor):
    cfg               robot config (zero ticks, signs, tip offset)
    tau_stall         N m     joint torque at 100 % PWM          (calibrated: 3.0; plausible 1.3 .. 3.0)
    tau_fric          N m     Coulomb friction, one value or one per arm joint
                                                                 (calibrated: 0.25; plausible 0.10 .. 0.35)
    omega0_deg_s      deg/s   no-load speed at 100 % PWM         (180; the datasheet value derated to 5 V is ~205)
    z_surface         m       height of the surface under the fingertip (None = free air)
    k_contact         N/m     stiffness of that contact          (1000 reproduces the real probe sink)
    payload_kg        kg      point mass at the fingertip
    h                 s       integration step                   (use <= 0.00025 for k_contact > 5000)
    overload          (threshold, seconds, protective fraction) model of the STS overload protection,
                      e.g. (0.8, 2.0, 0.2); sets status bit 32 when it trips. None = off
    inertia, j_rotor  second-order variant; j_rotor [kg m^2] = reflected rotor + gear inertia (0.012:
                      order-of-magnitude estimate for a 1:345 gear train)
    servo_profile     model the in-servo trapezoid profile (Goal_Velocity / Acceleration registers),
                      see sim_bus.profile_step (inferred behaviour, not measured)
    volt_idle, volt_sag_per_duty   crude rail model: volt = idle - sag * sum(|u|) (log: 4.9 V idle,
                      4.7 V with one joint near 45 % PWM)
    io_s              extra time per control cycle (0.003 gives the 45 Hz loop seen on the real bus)
    tau_static        N m     OPTION (default None = off, the model above unchanged): stiction. A joint
                      that stands still only starts when |tau_net| exceeds tau_static (one value or one
                      per arm joint, never below tau_fric); once it moves, tau_fric applies until
                      |tau_net| falls to it, where it stops. STICK_SLIP_20261111 below holds values
                      fitted to the pen-tip traces of 2026-11-11.
    stick_speed_deg_s deg/s   (only with tau_static; default 0 = off) a sliding joint whose speed falls below
                      this stops and needs tau_static again: the falling part of a Stribeck friction curve,
                      where slow sliding is unstable. Without it a joint driven by a slow ramp slides
                      smoothly once it has let go (tau_static then only acts at starts and reversals);
                      with it the joint keeps standing and jumping for as long as the ramp is slower than
                      this, which is what the real joints did at 2.4 deg/s.
Read-outs: u / pwm() (PWM fraction per joint), fz / peak_fz (contact force, N), heat (integral of
u^2 dt per joint, a proxy for I^2R heating), protect, q() (joint angles). Test helpers: set_q()
(teleport the arm to a pose), plus everything SimBus offers (now / sleep, log, io, fault, temp,
power_cycle, blocker for the gripper).

What the model does NOT contain: backlash, structural compliance other than the tip spring,
joint coupling through inertia, bus latency, temperature. Its speed-dependent tracking error is
therefore a lower bound that grows with speed exactly as u = omega / omega0 + friction demands.
Friction is Coulomb friction with static = kinetic unless tau_static is given: WITHOUT that
option the model has no stick-slip at all (a slowly driven joint follows smoothly, one friction
band behind), which is why it could not show what the pen drawing of 2026-11-11 showed.
"""

from __future__ import annotations

if __name__ == "__main__":  # run as a script: put tools/ and its feature folders on the import path first
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import tazuna_paths  # noqa: F401

import math

import numpy as np

import kinematics as K
from sim_bus import UNCAPPED, SimBus, profile_step

ARM = K.ARM_JOINTS
TPR = 4096.0 / (2.0 * math.pi)
G = 9.81
KP_PER_TICK_P32 = 8.2e-3  # PWM fraction per tick of (goal - present) at P_Coefficient = 32

# mass [kg] and centre of mass [m, in the link's own frame] of the moving links, copied from the
# <inertial> elements of so101_new_calib.urdf (refs/patches/verify/blocked-windup/inertials.json).
# index = which K.link_frames() frame the link is rigidly attached to.
INERTIALS = (
    ("shoulder_link", 0, 0.100006, (-0.0307604, -1.66727e-05, -0.0252713)),
    ("upper_arm_link", 1, 0.103, (-0.0898471, -0.00838224, 0.0184089)),
    ("lower_arm_link", 2, 0.104, (-0.0980701, 0.00324376, 0.0182831)),
    ("wrist_link", 3, 0.079, (-0.000103312, -0.0386143, 0.0281156)),
    ("gripper_link", 4, 0.087, (0.000213627, 0.000245138, -0.025187)),
)
MOVING_JAW = (0.012, (-0.00157495, -0.0300244, 0.0192755))  # in the moving-jaw link frame
_JAW_ORIGIN = ((0.0202, 0.0188, -0.0234), (1.5708, -5.24284e-08, -1.41553e-15))  # gripper_link -> jaw link (URDF)


def _rpy_transform(xyz, rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    t = np.eye(4)
    t[:3, :3] = (np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
                 @ np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]]))
    t[:3, 3] = xyz
    return t


_COMS = [(np.array([*c, 1.0]), m, i) for (_, i, m, c) in INERTIALS]
_COMS.append((_rpy_transform(*_JAW_ORIGIN) @ np.array([*MOVING_JAW[1], 1.0]), MOVING_JAW[0], 4))
_CL = np.array([c for c, m, i in _COMS])  # (6, 4) centres of mass in their link frames (homogeneous)
_CM = np.array([m for c, m, i in _COMS])  # (6,)   masses
_CI = np.array([i for c, m, i in _COMS])  # (6,)   link index
_MASK = (_CI[None, :] >= np.arange(5)[:, None]).astype(float)  # joint i carries mass k


def _statics(q, tool, z_surface=None, k_contact=0.0, payload_kg=0.0, want_inertia=False):
    fr = K.link_frames(q)
    A = np.array([f[:3, 2] for f in fr])  # (5, 3) joint axes
    P = np.array([f[:3, 3] for f in fr])  # (5, 3) joint origins
    C = np.array([(fr[i] @ c)[:3] for c, i in zip(_CL, _CI)])  # (6, 3) centres of mass
    tip = (fr[4] @ tool)[:3, 3]
    # for a vertical force f at point c, the torque about joint axis a through p is
    # a . ((c - p) x (0, 0, f)) = f * (a_x * r_y - a_y * r_x)
    rx = C[None, :, 0] - P[:, None, 0]
    ry = C[None, :, 1] - P[:, None, 1]
    tg = ((A[:, 0:1] * ry - A[:, 1:2] * rx) * _MASK) @ (-_CM * G)
    rtx, rty = tip[0] - P[:, 0], tip[1] - P[:, 1]
    lever = A[:, 0] * rty - A[:, 1] * rtx  # torque per unit vertical force at the tip
    if payload_kg:
        tg = tg + lever * (-payload_kg * G)
    fz = 0.0
    tc = np.zeros(5)
    if z_surface is not None and tip[2] < z_surface:
        fz = k_contact * (z_surface - tip[2])
        tc = lever * fz
    inertia = None
    if want_inertia:
        r = C[None, :, :] - P[:, None, :]  # (5, 6, 3)
        along = np.einsum("ikc,ic->ik", r, A)
        perp2 = np.einsum("ikc,ikc->ik", r, r) - along * along
        inertia = (perp2 * _MASK) @ _CM
        if payload_kg:
            rt = tip[None, :] - P
            at = np.einsum("ic,ic->i", rt, A)
            inertia = inertia + payload_kg * (np.einsum("ic,ic->i", rt, rt) - at * at)
    return tg, tc, tip, fz, inertia


def torques(q, tool, z_surface=None, k_contact=0.0, payload_kg=0.0):
    """(tau_gravity[5], tau_contact[5], tip_xyz, contact_force_N) for joint vector q (rad)."""
    return _statics(q, tool, z_surface, k_contact, payload_kg)[:4]


def potential(q, payload_kg=0.0, tool=None) -> float:
    """Gravitational potential energy (J) of the moving links; -dU/dq is the gravity torque."""
    fr = K.link_frames(q)
    u = sum(m * G * float((fr[i] @ c)[2]) for c, m, i in _COMS)
    if payload_kg:
        u += payload_kg * G * float(K.fk(q, tool)[2, 3])
    return float(u)


# Stick-slip parameters fitted to the pen-tip traces of 2026-11-11 (tests/testdata/pen_20261111/tracks, the
# encoder ticks behind them recovered with tracking.joints_from_tips). In the two traces taken with the
# correction frozen (goal = plan + a constant) each joint stood still until the plan was S ticks away
# and then moved until it was K ticks away, per direction; half the difference between the two
# directions gives, in ticks of (goal - present) at P = 32:
#     joint          lets go at S   stops at K   slip (ticks)
#     shoulder_pan        7.7           1.5          7 - 9
#     shoulder_lift      17.3          11.0          9 - 11
#     elbow_flex         16.1          10.8          7 - 8
#     wrist_flex          7.2           2.1          5 - 6
# 1 tick = 8.2e-3 x tau_stall = 0.0246 N m, which gives the torques below (wrist_roll was not moved in
# those traces: a guess). K is where the slip ENDS, taken as the kinetic friction of this model, which
# has no inertia; a real joint coasts, so its true kinetic friction lies between K and S.
# stick_speed_deg_s = 2: with the integral correction on, the real elbow at a planned 2.4 deg/s stood
# for 0.2 - 0.3 s and then jumped 9 - 12 ticks, from 2.6 ticks behind the plan to 2.8 ahead of it, again
# and again; this model with 2 deg/s does the same (10 ticks, from 2.2 behind to 2.5 ahead), with 0 it
# slides smoothly, with 4 it hardly ever sticks. At 5 - 6 deg/s the real joints moved in long runs.
# Use: PhysBus(cfg, io_s=0.003, **STICK_SLIP_20261111). NOT a calibration of the arm: one evening, one
# region of the workspace (x 0.21 .. 0.27 m, 3 cm above the table), joint speeds of 0.5 .. 6 deg/s.
STICK_SLIP_20261111 = {
    "tau_fric": (0.04, 0.27, 0.265, 0.05, 0.05),
    "tau_static": (0.19, 0.43, 0.40, 0.18, 0.12),
    "stick_speed_deg_s": 2.0,
}


class PhysBus(SimBus):
    def __init__(self, cfg, tau_stall=3.0, tau_fric=0.25, omega0_deg_s=180.0, z_surface=None, k_contact=0.0,
                 payload_kg=0.0, h=0.001, overload=None, inertia=False, j_rotor=0.012, servo_profile=False,
                 volt_idle=4.9, volt_sag_per_duty=0.45, io_s=0.0, blocker=None, tau_static=None,
                 stick_speed_deg_s=0.0):
        super().__init__(sag_ticks=0.0, blocker=blocker, io_s=io_s, servo_profile=servo_profile)
        self.cfg = cfg
        self.tool = K.tool_offset(*cfg["tip_offset"])
        self.tau_stall = float(tau_stall)
        self.tau_fric = np.broadcast_to(np.asarray(tau_fric, dtype=float), (5,)).copy()  # per joint
        # stiction option (None = off: every line of the dynamics below then computes what it always did)
        self.tau_static = None if tau_static is None else np.maximum(
            np.broadcast_to(np.asarray(tau_static, dtype=float), (5,)), self.tau_fric).copy()
        self._slipping = np.zeros(5, dtype=bool)  # stiction option: the joint is in motion (kinetic friction applies)
        self.w_stick = math.radians(float(stick_speed_deg_s)) if tau_static is not None else 0.0
        self.omega0 = math.radians(omega0_deg_s)
        self.z_surface, self.k_contact, self.payload = z_surface, float(k_contact), payload_kg
        self.h = h
        self.inertia, self.j_rotor = bool(inertia), float(j_rotor)
        self.volt_idle, self.volt_sag = float(volt_idle), float(volt_sag_per_duty)
        self.volt = self.volt_idle
        self.w = np.zeros(5)  # joint speeds (rad/s), inertia variant only
        self.u = dict.fromkeys(ARM, 0.0)
        self.fz = 0.0
        self.peak_fz = 0.0
        self.heat = dict.fromkeys(ARM, 0.0)  # integral of u^2 dt (s): proxy for I^2R heating
        # optional model of the STS overload protection: (threshold, seconds, protective fraction)
        self.overload = overload
        self._ovl_t = dict.fromkeys(ARM, 0.0)
        self.protect = dict.fromkeys(ARM, False)

    # --- helpers
    def q(self) -> np.ndarray:
        z, s = self.cfg["zero_ticks"], self.cfg["sign"]
        return np.array([s[j] * (self.pos[j] - z[j]) / TPR for j in ARM])

    def set_q(self, q, hold: bool = True):
        """Teleport the arm to joint pose q (rad). hold=True also seats goals / setpoints there."""
        z, s = self.cfg["zero_ticks"], self.cfg["sign"]
        for i, j in enumerate(ARM):
            self.pos[j] = z[j] + s[j] * float(q[i]) * TPR
            if hold:
                self.goal[j] = float(int(round(self.pos[j])))
                self.sp[j], self.sp_v[j] = self.pos[j], 0.0
        self.w[:] = 0.0

    def pwm(self) -> dict:
        return {j: round(100 * self.u[j]) for j in ARM}

    def _load(self, j: str) -> int:
        if j in self.u:
            return int(round(-1000 * self.u[j])) if self.torque[j] else 0
        return super()._load(j)

    # --- dynamics
    def _advance(self, dt_total: float):
        n = max(1, int(math.ceil(dt_total / self.h - 1e-9)))
        dt = dt_total / n
        for _ in range(n):
            self._substep(dt)
        self._advance_joint("gripper", dt_total)
        self.volt = round(self.volt_idle - self.volt_sag * sum(abs(v) for v in self.u.values()), 2)

    def _substep(self, dt: float):
        q = self.q()
        tg, tc, tip, fz, inertia = _statics(q, self.tool, self.z_surface, self.k_contact, self.payload,
                                            want_inertia=self.inertia)
        self.fz = fz
        self.peak_fz = max(self.peak_fz, fz)
        b = self.tau_stall / self.omega0  # back-EMF damping (N m s / rad)
        for i, j in enumerate(ARM):
            if not self.torque[j]:
                self.u[j] = 0.0
                self.w[i] = 0.0
                self._slipping[i] = False
                continue
            cap_ticks = float(self.vel_cap[j]) if self.vel_cap[j] > 0 else UNCAPPED
            if self.servo_profile:
                self.sp[j], self.sp_v[j] = profile_step(self.sp[j], self.sp_v[j], self._eff_goal(j), cap_ticks,
                                                        100.0 * self.acc[j], dt)
            kp = KP_PER_TICK_P32 * self.p_gain.get(j, 16) / 32.0
            u = max(-1.0, min(1.0, kp * (self._drive_ref(j) - self.pos[j])))
            if self.overload is not None:
                thr, secs, prot = self.overload
                if abs(u) > thr:
                    self._ovl_t[j] += dt
                    if self._ovl_t[j] >= secs:
                        self.protect[j] = True
                        self.fault[j] |= 32
                else:
                    self._ovl_t[j] = 0.0
                if self.protect[j]:
                    u = max(-prot, min(prot, u))
            self.u[j] = u
            self.heat[j] += u * u * dt
            # joint torque in URDF sign (u is in tick sign; the config sign maps ticks to URDF angle)
            sgn = self.cfg["sign"][j]
            net = sgn * u * self.tau_stall + tg[i] + tc[i]
            # torque a joint at rest must exceed to start: tau_fric, or tau_static with the stiction option
            hold = self.tau_fric[i] if (self.tau_static is None or self._slipping[i]) else self.tau_static[i]
            if self.inertia:
                w = self.w[i]
                if abs(w) < 1e-4 and abs(net) <= hold:
                    w = 0.0  # stuck
                else:
                    drive = net - math.copysign(self.tau_fric[i], w if abs(w) >= 1e-4 else net)
                    w_inf = drive / b
                    w_new = w_inf + (w - w_inf) * math.exp(-b * dt / (self.j_rotor + inertia[i]))
                    w = 0.0 if w * w_new < 0.0 else w_new  # stops at a reversal; next step decides
                self._slipping[i] = w != 0.0
            else:
                if abs(net) <= hold:
                    self.w[i] = 0.0
                    self._slipping[i] = False
                    continue
                w = self.omega0 * (net - math.copysign(self.tau_fric[i], net)) / self.tau_stall
                if abs(w) < self.w_stick:  # stiction option: too slow to keep sliding, the joint stops again
                    self.w[i] = 0.0
                    self._slipping[i] = False
                    continue
                self._slipping[i] = True
            if not self.servo_profile:
                cap = cap_ticks / TPR
                w = max(-cap, min(cap, w))
            self.w[i] = w
            lo, hi = self.limits_raw[j]
            self.pos[j] = max(lo - 40.0, min(hi + 40.0, self.pos[j] + sgn * w * dt * TPR))


if __name__ == "__main__":
    # self-check: analytic gravity / contact torques against numeric derivatives
    rng = np.random.default_rng(0)
    tool = K.tool_offset(-0.001, 0.0, 0.0068)
    worst = 0.0
    for _ in range(200):
        q = rng.uniform(-1.2, 1.2, 5)
        tg = torques(q, tool)[0]
        num = np.array([-(potential(q + d) - potential(q - d)) / 2e-4 for d in np.eye(5) * 1e-4])
        worst = max(worst, float(np.max(np.abs(tg - num))))
    print("analytic gravity torque vs -dU/dq: worst abs difference", worst, "N m")
    q = np.radians([10.0, 5.0, 20.0, 65.0, 21.0])
    z = K.fk(q, tool)[2, 3] + 0.01
    tg, tc, tip, fz = torques(q, tool, z_surface=z, k_contact=1000.0)
    jz = np.array([(K.fk(q + d, tool)[2, 3] - K.fk(q - d, tool)[2, 3]) / 2e-5 for d in np.eye(5) * 1e-5])
    print("contact torque", tc, "vs Jz^T F", jz * fz)
