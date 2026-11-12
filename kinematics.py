"""SO-101 forward / inverse kinematics from the official so101_new_calib.urdf.

Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll (radians,
URDF convention; every joint rotates about its local +z). The tool frame is the URDF
`gripper_frame_link`. Base frame: +x forward (away from the base towards the workspace),
+y left, +z up.

Contents: FK (fk, link_frames, tool_offset), the analytic IK (ik), feasible approach pitches
(feasible_pitches, pitch_range, best_pitch) and the jaw geometry in the base frame
(jaw_open_axis, jaw_tilt_deg, jaw_azimuth_deg, roll_for_azimuth, jaw_points). Pure math: nothing
here talks to hardware. Tests: test_kinematics.py.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

import gripper_model as GM  # closed-form gripper geometry: millimetres / degrees, gripper_frame_link axes

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

# (xyz, rpy) of each joint origin in its parent link, copied from so101_new_calib.urdf
_ORIGINS = (
    ((0.0388353, -8.97657e-09, 0.0624), (3.14159, 4.18253e-17, -3.14159)),  # shoulder_pan
    ((-0.0303992, -0.0182778, -0.0542), (-1.5708, -1.5708, 0.0)),  # shoulder_lift
    ((-0.11257, -0.028, 1.73763e-16), (-3.63608e-16, 8.74301e-16, 1.5708)),  # elbow_flex
    ((-0.1349, 0.0052, 3.62355e-17), (4.02456e-15, 8.67362e-16, -1.5708)),  # wrist_flex
    ((5.55112e-17, -0.0611, 0.0181), (1.5708, 0.0486795, 3.14159)),  # wrist_roll
)
# gripper_frame_joint (fixed): gripper_link -> gripper_frame_link
_TOOL = ((-0.0079, -0.000218121, -0.0981274), (0.0, 3.14159, 0.0))
# gripper joint (moving jaw pivot): gripper_link -> moving_jaw link
_JAW = ((0.0202, 0.0188, -0.0234), (1.5708, -5.24284e-08, -1.41553e-15))

URDF_LIMITS = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": (-0.174533, 1.74533),
}


def rpy_matrix(rpy) -> np.ndarray:
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def transform(xyz, rpy) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = rpy_matrix(rpy)
    t[:3, 3] = xyz
    return t


def rot_z(q: float) -> np.ndarray:
    c, s = math.cos(q), math.sin(q)
    t = np.eye(4)
    t[0, 0], t[0, 1], t[1, 0], t[1, 1] = c, -s, s, c
    return t


_FIXED = [transform(*o) for o in _ORIGINS]
_T_TOOL = transform(*_TOOL)
_T_JAW = transform(*_JAW)


def link_frames(q) -> list[np.ndarray]:
    """Base-frame transforms of the 5 joint frames (after joint rotation)."""
    out = []
    t = np.eye(4)
    for i in range(5):
        t = t @ _FIXED[i] @ rot_z(float(q[i]))
        out.append(t.copy())
    return out


def fk(q, tool: np.ndarray | None = None) -> np.ndarray:
    """4x4 pose of the tool (default gripper_frame_link) in the base frame."""
    t = np.eye(4)
    for i in range(5):
        t = t @ _FIXED[i] @ rot_z(float(q[i]))
    return t @ (_T_TOOL if tool is None else tool)


def gripper_link_pose(q) -> np.ndarray:
    return fk(q, tool=np.eye(4))


def tool_offset(dx=0.0, dy=0.0, dz=0.0) -> np.ndarray:
    """A tool frame displaced from gripper_frame_link by (dx,dy,dz) in ITS axes."""
    t = np.eye(4)
    t[:3, 3] = (dx, dy, dz)
    return _T_TOOL @ t


def pitch_down_of(t: np.ndarray, pan: float) -> float:
    """Pitch of the approach axis below horizontal in radians, measured in the arm's
    vertical plane (continuous through 90 deg = straight down; >90 = pointing back)."""
    a = approach_axis(t)
    forward = a[0] * math.cos(pan) - a[1] * math.sin(pan)  # +pan turns the arm towards -y
    return math.atan2(-a[2], forward)


def pose_summary(q, tool=None) -> dict:
    """Position plus the approach direction expressed as pitch (deg below horizontal,
    90 = straight down)."""
    t = fk(q, tool)
    return {"xyz": t[:3, 3].copy(), "approach": approach_axis(t),
            "pitch_down_deg": math.degrees(pitch_down_of(t, float(q[0]))), "T": t}


# Which local axis of gripper_frame_link points out along the fingers is decided numerically
# once (see _detect_axes) and cached here.
_APPROACH_LOCAL = None
_JAW_OPEN_LOCAL = None


def _detect_axes():
    global _APPROACH_LOCAL, _JAW_OPEN_LOCAL
    # Fingers extend from the gripper_link origin (on the wrist-roll axis) to the tool origin.
    v = _T_TOOL[:3, 3] / np.linalg.norm(_T_TOOL[:3, 3])  # in gripper_link axes
    v_tool = _T_TOOL[:3, :3].T @ v  # same direction in tool axes
    k = int(np.argmax(np.abs(v_tool)))
    e = np.zeros(3)
    e[k] = 1.0 if v_tool[k] > 0 else -1.0
    _APPROACH_LOCAL = e
    # The moving jaw rotates about the jaw joint z axis; opening direction is
    # perpendicular to both that axis and the approach axis.
    jaw_axis_tool = _T_TOOL[:3, :3].T @ _T_JAW[:3, 2]
    o = np.cross(jaw_axis_tool, e)
    _JAW_OPEN_LOCAL = o / np.linalg.norm(o)


_detect_axes()


def approach_axis(t: np.ndarray) -> np.ndarray:
    return t[:3, :3] @ _APPROACH_LOCAL


def jaw_open_axis(t: np.ndarray) -> np.ndarray:
    """Unit vector (base frame) along which the two jaws are separated."""
    return t[:3, :3] @ _JAW_OPEN_LOCAL


# --------------------------------------------------------------------------- IK
# The lift / elbow / wrist_flex axes are parallel, so after the pan rotation the arm is a
# planar 3-link chain in the vertical plane (rho = horizontal distance from the pan axis
# along the arm heading, z = height). Constants are derived from the URDF numerically.
_X_PAN = _ORIGINS[0][0][0]


def _plane(p) -> np.ndarray:
    return np.array([p[0] - _X_PAN, p[2]])


_F0 = link_frames(np.zeros(5))
_S = _plane(_F0[1][:3, 3])  # shoulder-lift axis
_U1 = _plane(_F0[2][:3, 3]) - _S  # lift axis -> elbow axis at zero pose
_U2 = _plane(_F0[3][:3, 3]) - _plane(_F0[2][:3, 3])  # elbow axis -> wrist-flex axis
_L1, _L2 = float(np.linalg.norm(_U1)), float(np.linalg.norm(_U2))
_A1, _A2 = math.atan2(_U1[1], _U1[0]), math.atan2(_U2[1], _U2[0])


def _rot2(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]])


def _wrap_pi(a: float) -> float:
    """The angle a (rad) expressed in (-pi, pi]. A value that is already inside is returned
    unchanged, bit for bit."""
    if -math.pi < a <= math.pi:
        return a
    a = math.fmod(a, 2.0 * math.pi)
    if a > math.pi:
        a -= 2.0 * math.pi
    elif a <= -math.pi:
        a += 2.0 * math.pi
    return a


def _residual(q4, wrist_roll, tool, target, pitch_rad):
    t = fk(np.array([*q4, wrist_roll]), tool)
    pitch = pitch_down_of(t, float(q4[0]))
    # position error (m) and pitch error scaled to metres with a 0.1 m lever arm; the pitch
    # error is an angle difference, so it is taken modulo 2*pi (a request of 180 deg must not
    # be compared with a measured -180 deg as if they were a full turn apart)
    return np.array([*(t[:3, 3] - target), 0.1 * _wrap_pi(pitch - pitch_rad)])


def ik(
    target_xyz,
    pitch_down_deg: float,
    wrist_roll: float = 0.0,
    q_init=None,
    tool: np.ndarray | None = None,
    limits: dict | None = None,
    pos_tol: float = 1e-5,
):
    """Solve pan/lift/elbow/wrist_flex so the tool origin reaches target_xyz with the
    approach axis pitched `pitch_down_deg` below horizontal (90 = straight down), elbow-up
    branch, wrist_roll held at the given value (radians, not normalised: pass it inside the
    limits).

    Analytic planar solution + a few Newton polish steps. pan, lift and wrist_flex are
    normalised into (-pi, pi] before the limit test, so a reachable pose is never refused
    because the formula produced the same angle plus or minus a full turn. A solution outside
    the joint limits is reported as not ok (never clipped silently).

    Returns (q5, info) with info = {ok, pos_err_m, pitch_err_deg, reason}. Use q5 only when
    info["ok"] is True.
    """
    del q_init  # the solution branch is unique (elbow-up); kept for API compatibility
    lim = limits or URDF_LIMITS
    target = np.asarray(target_xyz, dtype=float)
    theta = math.radians(pitch_down_deg)
    tl = _T_TOOL if tool is None else tool
    if not (math.isfinite(target[0]) and math.isfinite(target[1]) and math.isfinite(target[2])
            and math.isfinite(theta) and math.isfinite(wrist_roll)):
        return np.zeros(5), {"ok": False, "pos_err_m": float("inf"), "pitch_err_deg": float("inf"),
                             "reason": "target, pitch or roll is not a finite number"}

    # wrist-flex axis -> tool origin for this roll at zero pose: in-plane part u3 and the
    # lateral part e (the tool origin of the whole chain relative to the arm plane y=0)
    t0 = fk(np.array([0.0, 0.0, 0.0, 0.0, wrist_roll]), tl)
    u3 = _plane(t0[:3, 3]) - _plane(_F0[3][:3, 3])
    e = float(t0[1, 3])

    def fail(reason, q=None):
        return (np.zeros(5) if q is None else q), {
            "ok": False, "pos_err_m": float("inf"), "pitch_err_deg": float("inf"), "reason": reason}

    r_h = math.hypot(target[0] - _X_PAN, target[1])
    if r_h <= abs(e) + 1e-6:
        return fail("target on the pan axis")
    rho = math.sqrt(r_h * r_h - e * e)
    # base = Rz(-pan) applied to (rho, e): +pan turns the arm towards -y
    pan = _wrap_pi(-(math.atan2(target[1], target[0] - _X_PAN) - math.atan2(e, rho)))
    w = np.array([rho, target[2]]) - _rot2(-theta) @ u3  # wrist-flex axis in the plane
    d = w - _S
    dist = float(np.linalg.norm(d))
    cos_g = (dist * dist - _L1 * _L1 - _L2 * _L2) / (2.0 * _L1 * _L2)
    if cos_g > 1.0 or cos_g < -1.0:
        return fail(f"wrist centre out of reach (distance {dist:.3f} m, max {_L1 + _L2:.3f})")
    gamma = math.acos(cos_g)  # elbow-up: link 1 points more upward than link 2
    elbow = gamma - (_A1 - _A2)
    beta = math.atan2(d[1], d[0])
    delta = math.atan2(_L2 * math.sin(gamma), _L1 + _L2 * math.cos(gamma))
    lift = _wrap_pi(_A1 - beta - delta)
    wrist = _wrap_pi(theta - lift - elbow)
    q4 = np.array([pan, lift, elbow, wrist])

    # Newton polish. For gripper_frame_link and every tool made by tool_offset() the analytic
    # answer is already exact to about 3 um (the URDF prints pi with 6 digits), so no step is
    # taken; the polish only works for tools that carry their own orientation.
    r = _residual(q4, wrist_roll, tl, target, theta)
    for _ in range(6):
        if float(np.linalg.norm(r)) < pos_tol:
            break
        jac = np.empty((4, 4))
        eps = 1e-6
        for i in range(4):
            dq = q4.copy()
            dq[i] += eps
            jac[:, i] = (_residual(dq, wrist_roll, tl, target, theta) - r) / eps
        try:
            q4 = q4 + np.linalg.solve(jac, -r)
        except np.linalg.LinAlgError:
            return fail("singular configuration")
        r = _residual(q4, wrist_roll, tl, target, theta)
    q = np.array([*q4, wrist_roll])
    pos_err = float(np.linalg.norm(r[:3]))
    pitch_err = math.degrees(abs(r[3]) / 0.1)
    info = {"ok": True, "pos_err_m": pos_err, "pitch_err_deg": pitch_err, "reason": ""}
    if not (pos_err <= 5e-4 and pitch_err <= 0.2):
        info.update(ok=False, reason="did not converge")
        return q, info
    for i, j in enumerate(ARM_JOINTS):
        lo, hi = lim[j]
        if not (lo <= q[i] <= hi):
            info.update(ok=False, reason=f"{j}={math.degrees(q[i]):.1f} deg outside "
                                         f"[{math.degrees(lo):.1f}, {math.degrees(hi):.1f}]")
            return q, info
    return q, info


# ------------------------------------------------------------ feasible approach pitches
_SX, _SZ = float(_S[0]), float(_S[1])
_SCAN_FALLBACKS = 0  # how often the closed-form pitch scan disagreed with ik() (expected: never)


def _note_scan_fallback():
    global _SCAN_FALLBACKS
    _SCAN_FALLBACKS += 1
    warnings.warn("kinematics: the closed-form pitch scan disagreed with ik(); the exact (slow) scan was used",
                  RuntimeWarning, stacklevel=3)


class _PitchScan:
    """Feasibility of approach pitches at ONE tool position and wrist roll.

    The pan angle and the lever arms of the tool do not depend on the pitch, so they are
    computed once (one FK); each pitch then costs a few float operations (about 3 us) instead
    of a full ik() call (about 170 us). The formulas are those of ik().

    feasible(pitch_deg) is the fast test, exact(pitch_deg) asks ik() itself. The closed form
    needs a tool whose axes are those of gripper_frame_link (anything made by tool_offset());
    for any other tool `fast` is False and feasible() simply calls exact().
    """

    def __init__(self, target_xyz, wrist_roll: float = 0.0, tool=None, limits=None):
        self.target = np.asarray(target_xyz, dtype=float)
        self.wrist_roll = float(wrist_roll)
        self.tool = tool
        self.limits = limits
        tl = _T_TOOL if tool is None else tool
        self.fast = bool(np.abs(tl[:3, :3] - _T_TOOL[:3, :3]).max() < 1e-12)
        self.dead = False  # True: no pitch can be feasible at this position / roll
        if not self.fast:
            return
        lim = limits or URDF_LIMITS
        self._lim = [lim[j] for j in ARM_JOINTS[1:4]]  # lift, elbow, wrist_flex
        t0 = fk(np.array([0.0, 0.0, 0.0, 0.0, self.wrist_roll]), tl)
        u3 = _plane(t0[:3, 3]) - _plane(_F0[3][:3, 3])
        self._u3x, self._u3z = float(u3[0]), float(u3[1])
        e = float(t0[1, 3])
        x, y, self._z = (float(v) for v in self.target)
        r_h = math.hypot(x - _X_PAN, y)
        if not r_h > abs(e) + 1e-6:  # on the pan axis (or not a number)
            self.dead = True
            return
        self._rho = math.sqrt(r_h * r_h - e * e)
        pan = _wrap_pi(-(math.atan2(y, x - _X_PAN) - math.atan2(e, self._rho)))
        lo_p, hi_p = lim[ARM_JOINTS[0]]
        lo_r, hi_r = lim[ARM_JOINTS[4]]
        if not (lo_p <= pan <= hi_p and lo_r <= self.wrist_roll <= hi_r and math.isfinite(self._z)):
            self.dead = True

    def exact(self, pitch_deg: float) -> bool:
        return bool(ik(self.target, pitch_deg, self.wrist_roll, tool=self.tool, limits=self.limits)[1]["ok"])

    def feasible(self, pitch_deg: float) -> bool:
        if not self.fast:
            return self.exact(pitch_deg)
        if self.dead:
            return False
        theta = math.radians(pitch_deg)
        c, s = math.cos(theta), math.sin(theta)
        # wrist-flex axis relative to the shoulder-lift axis, in the arm's vertical plane
        dx = self._rho - (c * self._u3x + s * self._u3z) - _SX
        dz = self._z - (c * self._u3z - s * self._u3x) - _SZ
        cos_g = (dx * dx + dz * dz - _L1 * _L1 - _L2 * _L2) / (2.0 * _L1 * _L2)
        if not -1.0 <= cos_g <= 1.0:
            return False
        gamma = math.acos(cos_g)
        elbow = gamma - (_A1 - _A2)
        lift = _wrap_pi(_A1 - math.atan2(dz, dx) - math.atan2(_L2 * math.sin(gamma), _L1 + _L2 * math.cos(gamma)))
        wrist = _wrap_pi(theta - lift - elbow)
        (lo1, hi1), (lo2, hi2), (lo3, hi3) = self._lim
        return lo1 <= lift <= hi1 and lo2 <= elbow <= hi2 and lo3 <= wrist <= hi3


def _runs(grid, flags) -> list:
    """Contiguous runs of True in flags as [(first, last), ...] of the matching grid values."""
    out = []
    start = None
    for i, f in enumerate(flags):
        if f and start is None:
            start = i
        if start is not None and (not f or i == len(flags) - 1):
            out.append((grid[start], grid[i] if f else grid[i - 1]))
            start = None
    return out


def feasible_pitches(xyz, wrist_roll: float = 0.0, tool=None, limits=None, lo: int = -20, hi: int = 120,
                     step: int = 2) -> list:
    """Feasible approach pitches (deg below horizontal) for a tool position and wrist roll, as
    the list of contiguous intervals [(first, last), ...] in ascending order, on the grid
    lo, lo + step, ... <= hi; [] when no pitch on the grid is feasible. An isolated feasible
    grid point p is the interval (p, p).

    The feasible set is not always one interval: there can be a band of pitches in between
    where elbow_flex leaves its range (the wrist centre comes too close to the shoulder), e.g.
    [(16, 20), (28, 102)]. A grid point is feasible when ik() accepts it. The scan uses the
    closed form of ik() (about 0.7 ms for the default grid instead of 12 ms) and both ends of
    every returned interval are confirmed with ik() itself.
    """
    if int(step) <= 0:
        raise ValueError("feasible_pitches: step must be a positive number of degrees")
    grid = list(range(int(lo), int(hi) + 1, int(step)))
    scan = _PitchScan(xyz, wrist_roll, tool, limits)
    flags = [scan.feasible(p) for p in grid]
    runs = _runs(grid, flags)
    if scan.fast and not all(scan.exact(a) and (b == a or scan.exact(b)) for a, b in runs):
        _note_scan_fallback()
        runs = _runs(grid, [scan.exact(p) for p in grid])
    return runs


def pitch_range(target_xyz, wrist_roll: float = 0.0, tool=None, limits=None, lo: int = -20, hi: int = 120, step: int = 2):
    """Smallest and largest feasible pitch_down angle (deg) for a tool position, as (min, max),
    or None when no pitch is feasible. NOT every pitch in between is necessarily feasible:
    feasible_pitches() returns the actual intervals and best_pitch() picks a usable one."""
    intervals = feasible_pitches(target_xyz, wrist_roll, tool=tool, limits=limits, lo=lo, hi=hi, step=step)
    return (intervals[0][0], intervals[-1][1]) if intervals else None


def _edge(feas, bad: float, good: float, iters: int = 10) -> float:
    """Bisection between an infeasible and a feasible pitch; returns the feasible end."""
    for _ in range(iters):
        mid = 0.5 * (bad + good)
        if feas(mid):
            good = mid
        else:
            bad = mid
    return good


def _nearest_feasible(feas, want: float, lo: float, hi: float, margin: float):
    """Feasible pitch nearest to `want` inside [lo, hi] (deg), or None. Feasible intervals are
    found on a 1 deg grid, their ends refined to 0.001 deg; the answer keeps `margin` from the
    ends of its interval (the middle of an interval narrower than 2 * margin)."""
    g0, g1 = math.ceil(lo), math.floor(hi)
    grid = list(range(g0, g1 + 1))
    flags = [feas(p) for p in grid]
    best = None
    for a, b in _runs(grid, flags):
        a_f = _edge(feas, a - 1.0, float(a)) if a > g0 else float(a)
        b_f = _edge(feas, b + 1.0, float(b)) if b < g1 else float(b)
        m = min(margin, 0.5 * (b_f - a_f))
        p = min(max(want, a_f + m), b_f - m)
        if not feas(p):  # a hole narrower than the grid: fall back to a grid point that was tested
            p = float(min(range(a, b + 1), key=lambda g: abs(g - want)))
        if best is None or abs(p - want) < abs(best - want):
            best = p
    return best


def best_pitch(xyz, want: float = 90.0, wrist_roll: float = 0.0, tool=None, limits=None,
               lo: float = -20.0, hi: float = 120.0, margin_deg: float = 1.0):
    """Approach pitch (deg below horizontal) to use at a tool position when `want` is preferred.

    Returns (pitch, degraded):
      (want, False)   when ik() accepts `want` itself;
      (pitch, True)   otherwise: the feasible pitch nearest to `want` inside [lo, hi], taken from
                      the real feasible set (holes included, see feasible_pitches) and kept
                      margin_deg inside its interval so the arm does not sit on a joint limit;
      (None, True)    when no pitch in [lo, hi] is feasible (position out of reach).
    A returned pitch is always accepted by ik() with the same xyz / wrist_roll / tool / limits.
    Callers that need the wanted pitch (e.g. a vertical grasp) must check `degraded`:
    abs(pitch - want) says by how much. Feasible slivers narrower than 1 deg are ignored.
    """
    want = float(want)
    if not math.isfinite(want):
        raise ValueError("best_pitch: `want` must be a finite angle in degrees")
    if ik(xyz, want, wrist_roll, tool=tool, limits=limits)[1]["ok"]:
        return want, False
    scan = _PitchScan(xyz, wrist_roll, tool, limits)
    pitch = _nearest_feasible(scan.feasible, want, float(lo), float(hi), float(margin_deg))
    if scan.fast and pitch is not None and not scan.exact(pitch):
        _note_scan_fallback()
        pitch = _nearest_feasible(scan.exact, want, float(lo), float(hi), float(margin_deg))
    return pitch, True


# ------------------------------------------------------------ jaws in the base frame
# The wrist_roll joint origin carries a pitch of 0.0486795 rad (2.789 deg): at roll 0 the jaw
# line is that far off the arm's vertical plane.
WRIST_ROLL_SKEW = float(_ORIGINS[4][1][1])


def jaw_tilt_deg(q) -> float:
    """Elevation (deg) of the jaw-open axis (fixed jaw -> moving jaw) above the horizontal
    plane for arm joints q. 0 = both jaws level (always the case at pitch 90); positive = the
    moving jaw is the higher one; negative = the moving jaw hangs BELOW the fixed fingertip and
    reaches the table first. Closed form: asin(cos(pitch) * cos(wrist_roll - 2.789 deg))."""
    jaw = jaw_open_axis(fk(q))
    return math.degrees(math.asin(max(-1.0, min(1.0, float(jaw[2])))))


def jaw_azimuth_deg(q, min_horizontal: float = 0.5):
    """Base-frame azimuth (deg, counter-clockwise from +x seen from above, in (-180, 180]) of
    the horizontal direction from the fixed jaw to the moving jaw, or None when the jaw axis
    is too steep for that to mean anything (horizontal part shorter than min_horizontal, i.e.
    |jaw_tilt_deg| > 60 deg by default)."""
    jaw = jaw_open_axis(fk(q))
    if math.hypot(jaw[0], jaw[1]) < min_horizontal:
        return None
    return math.degrees(math.atan2(jaw[1], jaw[0]))


def roll_for_azimuth(psi_deg: float, pan_deg: float, pitch_down_deg: float = 90.0) -> float:
    """URDF wrist_roll (deg, normalised to (-180, 180]) that makes the moving jaw open toward
    base-frame azimuth psi_deg (counter-clockwise from +x seen from above; the fixed jaw sits
    on the opposite side) when shoulder_pan is pan_deg (URDF degrees).

    Closed form. For a vertical gripper (pitch_down_deg = 90) it is the exact relation
        roll = psi + pan + 2.789 deg.
    For 0 < pitch < 180 it is the roll that puts the HORIZONTAL PROJECTION of the jaw-open axis
    on azimuth psi: rho = atan2(sin(pitch) * sin(psi + pan), cos(psi + pan)), roll = rho + 2.789.
    The azimuth is then still exact, but the jaws are no longer level: the jaw line is tilted by
    asin(cos(pitch) * cos(rho)) (see jaw_tilt_deg), e.g. 10 deg at pitch 80 when the jaws open
    along the arm's heading, 0 when they open across it. Near a horizontal gripper the azimuth
    is ill-conditioned; pitch 0 / 180 raises ValueError. The opposite jaw order (psi + 180) is
    always this roll + 180 deg; joint limits are not checked here.

    pan itself depends slightly on the roll, because the tool point is off the roll axis (by
    7 to 8 mm for the fingertip). Start with pan = -atan2(y, x - _X_PAN) of the target (azimuth
    error up to 5 deg at 0.1 m from the pan axis), solve ik with the resulting roll, read pan
    from the answer and call this again: one repetition leaves < 0.2 deg, two < 0.02 deg.
    """
    theta = math.radians(pitch_down_deg)
    s = math.sin(theta)
    if not abs(s) >= 1e-6:
        raise ValueError(f"roll_for_azimuth: jaw azimuth is undefined for a horizontal gripper (pitch {pitch_down_deg})")
    a = math.radians(psi_deg + pan_deg)
    rho = math.atan2(abs(s) * math.sin(a), math.copysign(1.0, s) * math.cos(a))
    return math.degrees(_wrap_pi(rho + WRIST_ROLL_SKEW))


def jaw_points(q, jaw_q_deg: float, tip_tool: np.ndarray | None = None) -> dict:
    """Landmarks of the gripper in the BASE frame (metres, each a numpy array of 3) for arm
    joints q (5 values, rad) and URDF gripper angle jaw_q_deg (deg, e.g. Arm.gripper_q_deg(raw)):

      "fixed_nose"   centre of the distal end of the fixed jaw (6.3 mm beyond the tool origin)
      "moving_nose"  centre of the distal end face of the moving jaw
      "moving_pad"   centre of the moving jaw's tip pad (the face that presses on the object)
      "pivot"        the moving jaw's pivot axis where it crosses the finger centre plane
      "palm"         where the wrist-roll axis pierces the palm wall (deepest an object can sit)
      "tip"          only when tip_tool is given: origin of that tool frame, i.e. the harness
                     fingertip (Arm.tip_tool()); it is NOT the same point as "fixed_nose"
                     (3 mm apart with the default tip offset)

    The geometry comes from gripper_model (CAD, checked against the URDF meshes). Both noses
    are rounded and about 3 mm across, so the lowest material point of a jaw is a little lower
    than its nose point: measured on the meshes for jaw angles up to 60 deg, at most 1.6 mm
    (moving) / 0.2 mm (fixed) for pitch 80..100 and 2.5 mm / 0.8 mm for pitch 60..120. Keep a
    3 mm margin when these points are used for a floor check.
    """
    g = gripper_link_pose(q)
    t = g @ _T_TOOL  # pose of gripper_frame_link, identical to fk(q)
    nose = GM.nose_points(jaw_q_deg)
    pad_x, pad_z = GM.jaw_point(jaw_q_deg, GM.MOVING_PAD_CENTER_Q0)
    names = ("fixed_nose", "moving_nose", "moving_pad", "pivot", "palm")
    local_mm = np.array([
        nose["fixed"],
        nose["moving"],
        (pad_x, GM.MOVING_FINGER_Y, pad_z),
        (GM.PIVOT_X, GM.MOVING_FINGER_Y, GM.PIVOT_Z),
        (GM.ROLL_AXIS_X, GM.ROLL_AXIS_Y, GM.PALM_Z),
    ], dtype=float)
    base_m = (local_mm * 1e-3) @ t[:3, :3].T + t[:3, 3]
    out = {name: base_m[i].copy() for i, name in enumerate(names)}
    if tip_tool is not None:
        out["tip"] = (g @ tip_tool)[:3, 3].copy()
    return out


if __name__ == "__main__":
    np.set_printoptions(precision=4, suppress=True)
    q0 = np.zeros(5)
    print("approach local axis:", _APPROACH_LOCAL, " jaw-open local axis:", _JAW_OPEN_LOCAL)
    for i, t in enumerate(link_frames(q0)):
        print(f"zero pose {ARM_JOINTS[i]:14s} origin {t[:3, 3]}")
    s = pose_summary(q0)
    print("zero pose tool xyz", s["xyz"], "approach", s["approach"], "pitch_down", round(s["pitch_down_deg"], 2))
    print("zero pose jaw-open axis (base):", jaw_open_axis(s["T"]))
    for name, idx in (("pan", 0), ("lift", 1), ("elbow", 2), ("wrist_flex", 3), ("wrist_roll", 4)):
        q = np.zeros(5)
        q[idx] = math.radians(10)
        d = pose_summary(q)
        print(f"+10deg {name:10s}: tool dxyz {(d['xyz'] - s['xyz']) * 1000} mm, pitch_down {d['pitch_down_deg']:.2f}, jaw axis {jaw_open_axis(d['T'])}")
