"""Fast variants of the kinematics, for code that needs many solutions at once.

kinematics.py is the reference: one call, one pose, numpy matrices. Planners try thousands of candidates and a
recorded motion has thousands of poses, so the same formulas exist here in a cheaper form:

- closed-form inverse kinematics on plain floats (about 10 x faster than kinematics.ik):
    tool_model(tool)                                   where the tool point is for any wrist roll (coefficients)
    tool_plane(roll_deg, model)                        the tool point in the arm's plane at this roll
    roll_for_azimuth(psi_deg, x, y, pitch_deg, lat)    wrist roll that opens the jaws towards a base azimuth
    ik_fast(xyz, pitch_deg, roll_deg, model, lim)      elbow-up joint vector, or (None, reason)
    pitch_feasible(xy, zs, roll_deg, pitches, model, lim)   which approach pitches stay inside the joint limits
- forward kinematics for many joint vectors at once:
    tool_frame(tool), fk_points(q, tool)

Nothing here moves the arm or reads a device. Whatever a planner finds with ik_fast is confirmed with the reference
before it is executed (robot.Arm.solve). tests/test_grasp.py compares the inverse kinematics with kinematics.ik,
tests/test_tracking.py and tests/test_integration_pen.py compare fk_points with kinematics.fk.

Until 2026-11-13 this code lived in tools/pick/grasp.py (the planner still uses it under its old private names)
and in tools/draw/tracking.py (tool_frame, fk_points).

Units: metres and radians; angles named *_deg are degrees.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

import gripper_model as gm
import kinematics as K


def _wrap180(a_deg: float) -> float:
    """Angle in [-180, 180)."""
    return (float(a_deg) + 180.0) % 360.0 - 180.0


# ----------------------------------------------------------------------- closed-form IK in plain floats
_F0 = K.link_frames(np.zeros(5))
X_PAN = float(_F0[0][0, 3])  # x of the shoulder-pan axis


def _plane(p) -> np.ndarray:
    """(distance from the pan axis along the arm heading, height) of a zero-pose point."""
    return np.array([float(p[0]) - X_PAN, float(p[2])])


_SH = _plane(_F0[1][:3, 3])  # shoulder-lift axis
_EL = _plane(_F0[2][:3, 3])  # elbow axis (zero pose)
WR = _plane(_F0[3][:3, 3])  # wrist-flex axis (zero pose)
_L1, _L2 = float(np.linalg.norm(_EL - _SH)), float(np.linalg.norm(WR - _EL))
_A1 = math.atan2(_EL[1] - _SH[1], _EL[0] - _SH[0])
_A2 = math.atan2(WR[1] - _EL[1], WR[0] - _EL[0])
_SKEW = math.radians(gm.WRIST_ROLL_SKEW_DEG)


@lru_cache(maxsize=8)
def _tool_model_cached(tool_bytes: bytes):
    tool = np.frombuffer(tool_bytes, dtype=float).reshape(4, 4)
    vals = []
    for roll in (0.0, 0.5 * math.pi, math.pi):
        t0 = K.fk(np.array([0.0, 0.0, 0.0, 0.0, roll]), tool)
        u3 = _plane(t0[:3, 3]) - WR
        vals.append((float(t0[1, 3]), float(u3[0]), float(u3[1])))
    out = []
    for k in range(3):
        v0, v90, v180 = vals[0][k], vals[1][k], vals[2][k]
        c0 = 0.5 * (v0 + v180)
        out.append((c0, 0.5 * (v0 - v180), v90 - c0))
    return tuple(out)


def tool_model(tool: np.ndarray):
    """Where the tool point sits relative to the wrist as a function of the wrist roll. The point is a few
    mm off the roll axis, so each coordinate is c0 + c1*cos(roll) + c2*sin(roll) (three FK calls fix the
    coefficients exactly). Returns the coefficient triples of (e, u3x, u3z): e = distance from the arm's
    vertical plane, (u3x, u3z) = vector wrist-flex axis -> tool point in that plane at the zero pose, i.e.
    the quantities kinematics.ik computes with one FK call per solution."""
    return _tool_model_cached(np.ascontiguousarray(tool, dtype=float).tobytes())


def _wrap_pi(a: float) -> float:
    """Angle (rad) modulo a full turn, in [-pi, pi)."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def tool_plane(roll_deg: float, model):
    """(u3x, u3z, e) at this wrist roll (see tool_model)."""
    c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
    (e0, e1, e2), (x0, x1, x2), (z0, z1, z2) = model
    return x0 + x1 * c + x2 * s, z0 + z1 * c + z2 * s, e0 + e1 * c + e2 * s


def roll_for_azimuth(psi_deg: float, x: float, y: float, pitch_deg: float, lat) -> float:
    """Wrist roll (deg, in [-180, 180)) that makes the jaws open towards base azimuth psi when the tool
    point is at (x, y) with approach pitch pitch_deg. Closed form:
        pan   = -(atan2(y, x - x_pan) - asin(e(roll) / r))          (as in kinematics.ik)
        alpha = psi + pan                                           (azimuth relative to the arm heading)
        roll  = atan2(sin(alpha) * sin(pitch), cos(alpha)) + 2.789 deg
    i.e. roll = psi + pan + 2.789 deg at pitch 90. e(roll) is a few mm (lat = its coefficient triple, see
    tool_model), so three passes over the pan <-> roll coupling converge far below 0.001 deg (no search)."""
    e_c, a, b = lat
    az_t = math.atan2(y, x - X_PAN)
    r_h = max(math.hypot(x - X_PAN, y), 1e-6)
    psi, th = math.radians(psi_deg), math.radians(pitch_deg)
    roll = psi - az_t + _SKEW
    for _ in range(3):
        e = e_c + a * math.cos(roll) + b * math.sin(roll)
        pan = -(az_t - math.asin(max(-1.0, min(1.0, e / r_h))))
        alpha = psi + pan
        roll = math.atan2(math.sin(alpha) * math.sin(th), math.cos(alpha)) + _SKEW
    return _wrap180(math.degrees(roll))


def ik_fast(xyz, pitch_deg: float, roll_deg: float, model, lim: dict):
    """Closed-form joint vector for the tool point at xyz with this approach pitch and wrist roll: the
    analytic elbow-up solution of kinematics.ik in plain floats (about 10 x faster). Returns (q, "") or
    (None, reason) with kinematics.ik's wording. Whatever is planned with it is confirmed by arm.solve."""
    u3x, u3z, e = tool_plane(roll_deg, model)
    x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
    r_h = math.hypot(x - X_PAN, y)
    if r_h <= abs(e) + 1e-6:
        return None, "target on the pan axis"
    rho = math.sqrt(r_h * r_h - e * e)
    pan = _wrap_pi(-(math.atan2(y, x - X_PAN) - math.atan2(e, rho)))
    th = math.radians(pitch_deg)
    c, s = math.cos(th), math.sin(th)
    dx = rho - (c * u3x + s * u3z) - _SH[0]  # wrist-flex axis relative to the shoulder-lift axis
    dz = z - (-s * u3x + c * u3z) - _SH[1]
    dist = math.hypot(dx, dz)
    cos_g = (dist * dist - _L1 * _L1 - _L2 * _L2) / (2.0 * _L1 * _L2)
    if cos_g > 1.0 or cos_g < -1.0:
        return None, f"wrist centre out of reach (distance {dist:.3f} m, max {_L1 + _L2:.3f})"
    gamma = math.acos(cos_g)  # elbow-up
    elbow = gamma - (_A1 - _A2)
    lift = _wrap_pi(_A1 - math.atan2(dz, dx) - math.atan2(_L2 * math.sin(gamma), _L1 + _L2 * math.cos(gamma)))
    q = (pan, lift, elbow, _wrap_pi(th - lift - elbow), math.radians(roll_deg))
    for name, v in zip(K.ARM_JOINTS, q):
        lo, hi = lim[name]
        if not (lo <= v <= hi):
            return None, f"{name}={math.degrees(v):.1f} deg outside [{math.degrees(lo):.1f}, {math.degrees(hi):.1f}]"
    return np.array(q), ""


def pitch_feasible(xy, zs, roll_deg: float, pitches_deg, model, lim: dict) -> np.ndarray:
    """Which approach pitches put the tool point at (xy, z) inside the joint limits? ik_fast for all the
    heights zs and all the pitches at once: bool array (len(zs), len(pitches)). Used to pick the pitch
    when 90 deg does not work."""
    pitches_deg = np.asarray(pitches_deg, dtype=float)
    zs = np.asarray(zs, dtype=float).reshape(-1, 1)
    u3x, u3z, e = tool_plane(roll_deg, model)
    roll = math.radians(roll_deg)
    r_h = math.hypot(xy[0] - X_PAN, xy[1])
    none = np.zeros((len(zs), len(pitches_deg)), dtype=bool)
    if r_h <= abs(e) + 1e-6:
        return none
    rho = math.sqrt(r_h * r_h - e * e)
    pan = _wrap_pi(-(math.atan2(xy[1], xy[0] - X_PAN) - math.atan2(e, rho)))
    for name, val in (("shoulder_pan", pan), ("wrist_roll", roll)):
        if not (lim[name][0] <= val <= lim[name][1]):
            return none
    th = np.radians(pitches_deg)[None, :]
    c, s = np.cos(th), np.sin(th)
    dx = rho - (c * u3x + s * u3z) - _SH[0]
    dz = zs - (-s * u3x + c * u3z) - _SH[1]
    cos_g = (dx * dx + dz * dz - _L1 * _L1 - _L2 * _L2) / (2.0 * _L1 * _L2)
    ok = np.abs(cos_g) <= 1.0
    gamma = np.arccos(np.clip(cos_g, -1.0, 1.0))
    elbow = gamma - (_A1 - _A2)
    two_pi = 2.0 * math.pi  # lift and wrist_flex modulo a full turn, as kinematics.ik takes them
    lift = (_A1 - np.arctan2(dz, dx) - np.arctan2(_L2 * np.sin(gamma), _L1 + _L2 * np.cos(gamma)) + math.pi) % two_pi \
        - math.pi
    wrist = (th - lift - elbow + math.pi) % two_pi - math.pi
    for name, val in (("shoulder_lift", lift), ("elbow_flex", elbow), ("wrist_flex", wrist)):
        ok &= (val >= lim[name][0]) & (val <= lim[name][1])
    return ok

# ----------------------------------------------------------------------- forward kinematics for many poses
def tool_frame(tool=None) -> np.ndarray:
    """The 4x4 frame a `tool` argument means here and in tracking.py: None = gripper_frame_link, a 4x4
    array (arm.tip_tool()), or an object with a `tip_offset` (a tooling.Tool, which is what `tool`
    means everywhere else in the harness; tooling is not imported here)."""
    if tool is None:
        return K._T_TOOL
    off = getattr(tool, "tip_offset", None)
    if off is not None:
        return K.tool_offset(*(float(v) for v in off))
    t = np.asarray(tool, dtype=float)
    if t.shape != (4, 4):
        raise ValueError(f"tool must be None, a 4x4 frame (arm.tip_tool()) or a tooling.Tool, got shape {t.shape}")
    return t


def fk_points(q, tool=None) -> np.ndarray:
    """Tool-tip positions (N, 3) for joint angles (N, 5): the same chain as kinematics.fk,
    vectorised (a trace has thousands of poses). tool: see tool_frame()."""
    q = np.asarray(q, dtype=float).reshape(-1, 5)
    n = len(q)
    t = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    for i in range(5):
        c, s = np.cos(q[:, i]), np.sin(q[:, i])
        r = np.zeros((n, 4, 4))
        r[:, 0, 0], r[:, 0, 1], r[:, 1, 0], r[:, 1, 1], r[:, 2, 2], r[:, 3, 3] = c, -s, s, c, 1.0, 1.0
        t = t @ K._FIXED[i] @ r
    return (t @ tool_frame(tool))[:, :3, 3]
