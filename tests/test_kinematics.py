"""Tests for kinematics.py and gripper_model.py. Pure math: no hardware, no serial port, no
camera, and no import of robot.py / servo_bus.py (their owners are editing them).

Run:  python tests/test_kinematics.py
      (optionally followed by words; only tests whose name contains one of them are run)

What is checked against what:
  * FK                 against an independent implementation parsed from tests/refs/urdf/so101_new_calib.urdf
                       (kinematic tree from parent/child links, quaternions, Rodrigues' formula)
  * IK                 round trips, the independent FK, the joint limits, and a frozen copy of the
                       kinematics.py that drove the real arm on 2026-11-10 (regression, bit for bit)
  * pitch intervals    against a brute-force ik() scan and the concrete points of review finding KI-4
  * jaw geometry       against gripper_model, the URDF `gripper` joint and the URDF meshes (tests/refs/urdf/assets)
  * gripper_model      its own self-test, the meshes, and the stall positions measured on the real arm
"""

from __future__ import annotations

import os as _os  # the harness modules live in tools/<feature>/ (see tools/tazuna_paths.py)
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "tools"))
import tazuna_paths  # noqa: E402,F401

import hashlib
import inspect
import math
import os
import subprocess
import sys
import time
import types
import unittest
import warnings
import xml.etree.ElementTree as ET

import numpy as np

import gripper_model as GM
import kinematics as K

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = HERE  # tests/: the reference URDF, meshes and gripper model are in tests/refs/
URDF_PATH = os.path.join(PROJECT, "refs", "urdf", "so101_new_calib.urdf")
ASSETS = os.path.join(PROJECT, "refs", "urdf", "assets")
GRIPPER_REF = os.path.join(PROJECT, "refs", "gripper", "so101_gripper_model.py")

NOTES = []  # lines printed after the PASS line of the running test


def note(msg: str):
    NOTES.append(msg)


class Skip(unittest.SkipTest):
    """Raised by a test that cannot run here (reference meshes missing); pytest reports it as skipped."""


def wrap(a):
    """Angle(s) into [-pi, pi)."""
    return (np.asarray(a, dtype=float) + math.pi) % (2.0 * math.pi) - math.pi


def ang_diff_deg(a, b):
    return abs((a - b + 180.0) % 360.0 - 180.0)


# ---------------------------------------------------------------------------- rig constants
# Joint limits exactly as robot.Arm.q_limits() builds them for this arm (DESIGN.md section 3):
# servo limits in ticks, 45 ticks margin, zero ticks 2047 (wrist_roll 1944.6), soft limits.
TICKS_PER_RAD = 4096.0 / (2.0 * math.pi)
SERVO_LIMITS = {"shoulder_pan": (864, 3418), "shoulder_lift": (931, 3356), "elbow_flex": (775, 3064),
                "wrist_flex": (835, 3266), "wrist_roll": (0, 4095)}
ZERO_TICKS = {"shoulder_pan": 2047.0, "shoulder_lift": 2047.0, "elbow_flex": 2047.0, "wrist_flex": 2047.0,
              "wrist_roll": 1944.6}
SOFT_LIMITS_DEG = {"shoulder_pan": (-100.0, 100.0), "wrist_roll": (-160.0, 160.0)}


def real_limits() -> dict:
    out = {}
    for j in K.ARM_JOINTS:
        lo, hi = SERVO_LIMITS[j]
        a, b = (lo + 45 - ZERO_TICKS[j]) / TICKS_PER_RAD, (hi - 45 - ZERO_TICKS[j]) / TICKS_PER_RAD
        if j in SOFT_LIMITS_DEG:
            a, b = max(a, math.radians(SOFT_LIMITS_DEG[j][0])), min(b, math.radians(SOFT_LIMITS_DEG[j][1]))
        out[j] = (a, b)
    return out


REAL = real_limits()
WIDE = {j: (-10.0, 10.0) for j in K.ARM_JOINTS}
LO = np.array([REAL[j][0] for j in K.ARM_JOINTS])
HI = np.array([REAL[j][1] for j in K.ARM_JOINTS])
TIP_OFFSET = (-0.001, 0.0, 0.0068)  # the harness fingertip (config.json "tip_offset")
TIP = K.tool_offset(*TIP_OFFSET)
BIG_OFFSET = (0.02, -0.015, 0.03)
BIG = K.tool_offset(*BIG_OFFSET)
TOOLS = (("gripper_frame", None, (0.0, 0.0, 0.0)), ("fingertip", TIP, TIP_OFFSET), ("20/-15/30 mm", BIG, BIG_OFFSET))


def in_limits(q, lim) -> bool:
    return all(lim[j][0] <= q[i] <= lim[j][1] for i, j in enumerate(K.ARM_JOINTS))


def branch_of(q, tool):
    """(forward distance of the tool from the pan axis in the arm plane, planar bend gamma)."""
    t = K.fk(q, tool)
    fwd = (t[0, 3] - K._X_PAN) * math.cos(q[0]) - t[1, 3] * math.sin(q[0])
    return fwd, q[2] + (K._A1 - K._A2)


# ---------------------------------------------------------------------------- independent URDF model
def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw)


def _quat_matrix(q):
    w, x, y, z = q
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _rpy_rotation(rpy):
    """URDF fixed-axis roll, pitch, yaw: R = Rz(yaw) Ry(pitch) Rx(roll), built from quaternions."""
    r, p, y = rpy
    qx = (math.cos(r / 2), math.sin(r / 2), 0.0, 0.0)
    qy = (math.cos(p / 2), 0.0, math.sin(p / 2), 0.0)
    qz = (math.cos(y / 2), 0.0, 0.0, math.sin(y / 2))
    return _quat_matrix(_qmul(qz, _qmul(qy, qx)))


def _rodrigues(axis, angle):
    k = np.asarray(axis, dtype=float)
    k = k / np.linalg.norm(k)
    kx = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + math.sin(angle) * kx + (1.0 - math.cos(angle)) * (kx @ kx)


def _pose(rot, xyz):
    t = np.eye(4)
    t[:3, :3] = rot
    t[:3, 3] = xyz
    return t


class UrdfModel:
    """Forward kinematics straight from the URDF text, written independently of kinematics.py."""

    def __init__(self, path=URDF_PATH):
        root = ET.parse(path).getroot()
        self.joints = {}
        for j in root.findall("joint"):
            origin, axis, limit = j.find("origin"), j.find("axis"), j.find("limit")
            self.joints[j.get("name")] = {
                "type": j.get("type"), "parent": j.find("parent").get("link"), "child": j.find("child").get("link"),
                "xyz": [float(v) for v in origin.get("xyz").split()],
                "rpy": [float(v) for v in origin.get("rpy").split()],
                "axis": [float(v) for v in axis.get("xyz").split()] if axis is not None else None,
                "limit": (float(limit.get("lower")), float(limit.get("upper"))) if limit is not None else None,
            }
        self.visuals = {}
        for link in root.findall("link"):
            self.visuals[link.get("name")] = [
                (v.find("geometry").find("mesh").get("filename"),
                 [float(x) for x in v.find("origin").get("xyz").split()],
                 [float(x) for x in v.find("origin").get("rpy").split()]) for v in link.findall("visual")]
        self.by_child = {j["child"]: name for name, j in self.joints.items()}

    def chain(self, link):
        """Joint names from base_link to `link`, found by walking child -> parent."""
        names = []
        while link != "base_link":
            name = self.by_child[link]
            names.append(name)
            link = self.joints[name]["parent"]
        return names[::-1]

    def link_pose(self, link, angles: dict) -> np.ndarray:
        """Pose of a link in base_link for joint angles given by joint name (missing = 0)."""
        t = np.eye(4)
        for name in self.chain(link):
            j = self.joints[name]
            t = t @ _pose(_rpy_rotation(j["rpy"]), j["xyz"])
            if j["type"] == "revolute":
                t = t @ _pose(_rodrigues(j["axis"], angles.get(name, 0.0)), [0.0, 0.0, 0.0])
        return t

    def arm_angles(self, q5) -> dict:
        return dict(zip(K.ARM_JOINTS, (float(v) for v in q5)))

    def tool_pose(self, q5, offset=(0.0, 0.0, 0.0)) -> np.ndarray:
        """gripper_frame_link pose, optionally displaced by `offset` in ITS axes."""
        return self.link_pose("gripper_frame_link", self.arm_angles(q5)) @ _pose(np.eye(3), offset)

    def pitch_down_deg(self, t, pan) -> float:
        a = t[:3, 2]  # +z of gripper_frame_link = approach direction
        return math.degrees(math.atan2(-a[2], a[0] * math.cos(pan) - a[1] * math.sin(pan)))


_MODEL = None


def model() -> UrdfModel:
    global _MODEL
    if _MODEL is None:
        _MODEL = UrdfModel()
    return _MODEL


# ---------------------------------------------------------------------------- meshes (optional reference)
def _load_stl(path):
    with open(path, "rb") as f:
        data = f.read()
    n = int(np.frombuffer(data[80:84], dtype="<u4")[0])
    assert len(data) == 84 + 50 * n, f"{path}: not a binary STL"
    rec = np.frombuffer(data[84:], dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]))
    return rec["v"].astype(float)  # (n, 3, 3), metres


_MESH_CACHE = {}


def gripper_mesh_gf_mm(link: str, only: str = "", jaw_deg: float = 0.0):
    """Triangles (n, 3, 3) of a link's visual meshes in gripper_frame_link axes, millimetres."""
    key = (link, only, jaw_deg)
    if key not in _MESH_CACHE:
        if not os.path.isdir(ASSETS):
            raise Skip("tests/refs/urdf/assets is missing")
        m = model()
        angles = {"gripper": math.radians(jaw_deg)}
        rel = np.linalg.inv(m.link_pose("gripper_frame_link", angles)) @ m.link_pose(link, angles)
        parts = []
        for filename, xyz, rpy in m.visuals[link]:
            if only and only not in filename:
                continue
            t = rel @ _pose(_rpy_rotation(rpy), xyz)
            tris = _load_stl(os.path.join(PROJECT, "refs", "urdf", filename))
            parts.append((tris @ t[:3, :3].T + t[:3, 3]) * 1000.0)
        _MESH_CACHE[key] = np.concatenate(parts)
    return _MESH_CACHE[key]


def slab_points(tris, axis, lo, hi):
    """All points of a triangle mesh with lo <= coordinate[axis] <= hi: its vertices plus the
    points where triangle edges cross the two planes (exact extremes, not sampled)."""
    pts = [tris.reshape(-1, 3)]
    for a, b in ((0, 1), (1, 2), (2, 0)):
        p, q = tris[:, a], tris[:, b]
        d = q[:, axis] - p[:, axis]
        for c in (lo, hi):
            with np.errstate(divide="ignore", invalid="ignore"):
                s = (c - p[:, axis]) / d
            m = np.isfinite(s) & (s > 0.0) & (s < 1.0)
            pts.append(p[m] + s[m, None] * (q[m] - p[m]))
    allp = np.concatenate(pts)
    return allp[(allp[:, axis] >= lo - 1e-9) & (allp[:, axis] <= hi + 1e-9)]


# ---------------------------------------------------------------------------- frozen copy of kinematics.py v1
# Verbatim text of kinematics.py as it was when the real arm completed its three pick-and-place
# runs (file dated 2026-11-10 22:52, md5 9bbdc95df0ffa3a33429f65adf06fc12, the version examined
# by the kinematics review). Used only as the regression reference for ik() and pitch_range().
_V1_MD5 = "9bbdc95df0ffa3a33429f65adf06fc12"
_V1_SOURCE = r'''"""SO-101 forward / inverse kinematics from the official so101_new_calib.urdf.

Joint order: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll (radians,
URDF convention; every joint rotates about its local +z). The tool frame is the URDF
`gripper_frame_link`. Base frame: +x forward (away from the base towards the workspace),
+y left, +z up.
"""

from __future__ import annotations

import math

import numpy as np

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


def _residual(q4, wrist_roll, tool, target, pitch_rad):
    t = fk(np.array([*q4, wrist_roll]), tool)
    pitch = pitch_down_of(t, float(q4[0]))
    # position error (m) and pitch error scaled to metres with a 0.1 m lever arm
    return np.array([*(t[:3, 3] - target), 0.1 * (pitch - pitch_rad)])


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
    branch, wrist_roll held at the given value.

    Analytic planar solution + a few Newton polish steps. A solution outside the joint
    limits is reported as not ok (never clipped silently).

    Returns (q5, info) with info = {ok, pos_err_m, pitch_err_deg, reason}.
    """
    del q_init  # the solution branch is unique (elbow-up); kept for API compatibility
    lim = limits or URDF_LIMITS
    target = np.asarray(target_xyz, dtype=float)
    theta = math.radians(pitch_down_deg)
    tl = _T_TOOL if tool is None else tool

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
    pan = -(math.atan2(target[1], target[0] - _X_PAN) - math.atan2(e, rho))
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
    lift = _A1 - beta - delta
    wrist = theta - lift - elbow
    q4 = np.array([pan, lift, elbow, wrist])

    # Newton polish (absorbs the small roll-frame skew of the real chain)
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
    if pos_err > 5e-4 or pitch_err > 0.2:
        info.update(ok=False, reason="did not converge")
        return q, info
    for i, j in enumerate(ARM_JOINTS):
        lo, hi = lim[j]
        if not (lo <= q[i] <= hi):
            info.update(ok=False, reason=f"{j}={math.degrees(q[i]):.1f} deg outside "
                                         f"[{math.degrees(lo):.1f}, {math.degrees(hi):.1f}]")
            return q, info
    return q, info


def pitch_range(target_xyz, wrist_roll: float = 0.0, tool=None, limits=None, lo: int = -20, hi: int = 120, step: int = 2):
    """All feasible pitch_down angles (deg) for a tool position, as (min, max) or None."""
    ok = [p for p in range(lo, hi + 1, step) if ik(target_xyz, p, wrist_roll, tool=tool, limits=limits)[1]["ok"]]
    return (min(ok), max(ok)) if ok else None


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
'''
_V1 = None


def v1():
    global _V1
    if _V1 is None:
        assert hashlib.md5(_V1_SOURCE.encode()).hexdigest() == _V1_MD5, "the frozen v1 source was altered"
        mod = types.ModuleType("kinematics_v1")
        exec(compile(_V1_SOURCE, "<kinematics v1, frozen inside test_kinematics.py>", "exec"), mod.__dict__)
        _V1 = mod
    return _V1


# ============================================================================ API
def test_api_names_and_signatures():
    """Every public name of DESIGN.md 5.2 exists with the documented parameters."""
    for name in ("ARM_JOINTS", "URDF_LIMITS", "fk", "link_frames", "gripper_link_pose", "tool_offset", "pose_summary",
                 "pitch_down_of", "approach_axis", "jaw_open_axis", "ik", "pitch_range", "_X_PAN", "_A1", "_A2",
                 "feasible_pitches", "best_pitch", "jaw_tilt_deg", "roll_for_azimuth", "jaw_points",
                 "rpy_matrix", "transform", "rot_z", "_T_TOOL", "_F0", "_plane", "_rot2", "_S", "_L1", "_L2"):
        assert hasattr(K, name), name

    def params(fn):
        return [(p.name, p.default) for p in inspect.signature(fn).parameters.values()]

    e = inspect.Parameter.empty
    assert params(K.fk) == [("q", e), ("tool", None)]
    assert params(K.tool_offset) == [("dx", 0.0), ("dy", 0.0), ("dz", 0.0)]
    assert params(K.ik) == [("target_xyz", e), ("pitch_down_deg", e), ("wrist_roll", 0.0), ("q_init", None),
                            ("tool", None), ("limits", None), ("pos_tol", 1e-5)]
    assert params(K.pitch_range) == [("target_xyz", e), ("wrist_roll", 0.0), ("tool", None), ("limits", None),
                                     ("lo", -20), ("hi", 120), ("step", 2)]
    assert params(K.feasible_pitches) == [("xyz", e), ("wrist_roll", 0.0), ("tool", None), ("limits", None),
                                          ("lo", -20), ("hi", 120), ("step", 2)]
    assert params(K.best_pitch)[:5] == [("xyz", e), ("want", 90.0), ("wrist_roll", 0.0), ("tool", None),
                                        ("limits", None)]
    assert K.best_pitch(xyz=(0.2, 0.0, 0.03), want=90.0, tool=TIP, limits=REAL) == (90.0, False)
    assert K.feasible_pitches(xyz=(0.2, 0.0, 0.03), tool=TIP, limits=REAL) == K.feasible_pitches(
        (0.2, 0.0, 0.03), 0.0, TIP, REAL)
    assert params(K.jaw_tilt_deg) == [("q", e)]
    assert params(K.roll_for_azimuth) == [("psi_deg", e), ("pan_deg", e), ("pitch_down_deg", 90.0)]
    assert params(K.jaw_points) == [("q", e), ("jaw_q_deg", e), ("tip_tool", None)]
    assert K.ARM_JOINTS == ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
    # the call the integrator's skills.py makes
    p, degraded = K.best_pitch((0.2, 0.0, 0.03), want=90.0, wrist_roll=math.radians(10.0), tool=TIP, limits=REAL)
    assert p == 90.0 and degraded is False
    # every public name of the v1 module is still there
    old = v1()
    missing = [n for n in dir(old) if not n.startswith("__") and not hasattr(K, n)]
    assert not missing, missing

    assert params(GM.nose_points) == [("q_jaw_deg", e)]
    assert params(GM.finger_back_x) == [("z", e)]
    assert params(GM.moving_outer_x) == [("q", e), ("z", e)]
    assert params(GM.q_for_width) == [("w_mm", e)]
    assert params(GM.q_from_raw) == [("raw", e), ("touch_raw", 2050)]
    assert params(GM.raw_from_q) == [("q", e), ("touch_raw", 2050)]
    for fn in (GM.nose_points, GM.finger_back_x, GM.moving_outer_x, GM.q_for_width, GM.width_for_q, GM.q_from_raw,
               GM.raw_from_q, GM.stall_raw_for_width, GM.width_from_stall_raw, K.feasible_pitches, K.best_pitch,
               K.jaw_tilt_deg, K.roll_for_azimuth, K.jaw_points):
        assert fn.__doc__ and len(fn.__doc__) > 40, fn.__name__


def test_gripper_model_keeps_the_research_file():
    """gripper_model.py still contains the research file verbatim and all of its names."""
    with open(GM.__file__) as f:
        mine = f.read()
    with open(GRIPPER_REF) as f:
        ref = f.read()
    head = ref[:ref.index('if __name__ == "__main__":')]
    assert mine.startswith(head), "the part of gripper_model.py copied from the research was modified"
    ref_main = ref[ref.index('if __name__ == "__main__":'):]
    kept = ref_main[:ref_main.index('    print("so101_gripper_model self-test OK")')]
    assert kept in mine, "the original self-test lines were modified"
    ns = {}
    exec(compile(ref, GRIPPER_REF, "exec"), {"__name__": "ref_gripper_model"}, ns)
    names = [n for n in ns if not n.startswith("_") and n != "math"]
    assert len(names) >= 45, len(names)
    for n in names:
        assert hasattr(GM, n), n
        if not callable(ns[n]):
            assert getattr(GM, n) == ns[n], n
    note(f"{len(names)} original names present and unchanged")


# ============================================================================ FK
def test_constants_match_urdf():
    """Every constant copied into kinematics.py / gripper_model.py equals the URDF text."""
    m = model()
    assert tuple(n for n in m.chain("gripper_frame_link") if m.joints[n]["type"] == "revolute") == K.ARM_JOINTS
    assert m.chain("gripper_frame_link")[-1] == "gripper_frame_joint"
    for i, name in enumerate(K.ARM_JOINTS):
        j = m.joints[name]
        assert j["xyz"] == list(K._ORIGINS[i][0]) and j["rpy"] == list(K._ORIGINS[i][1]), name
        assert j["axis"] == [0.0, 0.0, 1.0], name
        assert j["limit"] == K.URDF_LIMITS[name], name
    j = m.joints["gripper_frame_joint"]
    assert j["type"] == "fixed" and j["parent"] == "gripper_link"
    assert j["xyz"] == list(K._TOOL[0]) and j["rpy"] == list(K._TOOL[1])
    j = m.joints["gripper"]
    assert j["parent"] == "gripper_link" and j["axis"] == [0.0, 0.0, 1.0]
    assert j["xyz"] == list(K._JAW[0]) and j["rpy"] == list(K._JAW[1]) and j["limit"] == K.URDF_LIMITS["gripper"]
    assert K.WRIST_ROLL_SKEW == m.joints["wrist_roll"]["rpy"][1] == 0.0486795
    assert abs(math.degrees(K.WRIST_ROLL_SKEW) - GM.WRIST_ROLL_SKEW_DEG) < 2e-4
    # gripper_model frame constants: gripper_link -> gripper_frame_link, jaw pivot, roll axis
    assert np.allclose(np.array(GM.GL_T_GF_XYZ_MM), 1000.0 * np.array(m.joints["gripper_frame_joint"]["xyz"]), atol=1e-9)
    rel = np.linalg.inv(m.link_pose("gripper_frame_link", {})) @ m.link_pose("moving_jaw_so101_v1_link", {})
    pivot_mm, axis = rel[:3, 3] * 1000.0, rel[:3, 2]
    assert np.allclose(pivot_mm, (GM.PIVOT_X, GM.PIVOT_Y_JOINT_ORIGIN, GM.PIVOT_Z), atol=2e-4), pivot_mm
    assert np.allclose(axis, (0.0, -1.0, 0.0), atol=1e-5), axis  # +q turns the jaw about -y of the tool frame
    rel = np.linalg.inv(m.link_pose("gripper_frame_link", {})) @ m.link_pose("gripper_link", {})
    assert np.allclose(rel[:3, 3] * 1000.0, (GM.ROLL_AXIS_X, GM.ROLL_AXIS_Y, -98.1274), atol=2e-4)
    assert np.allclose(rel[:3, 2], (0.0, 0.0, -1.0), atol=1e-5)  # roll axis is parallel to the approach axis


def test_fk_matches_independent_urdf_model():
    """fk / link_frames / gripper_link_pose / tool_offset equal the independent URDF model."""
    m = model()
    rng = np.random.default_rng(0)
    worst_p = worst_r = 0.0
    for _ in range(20000):
        q = rng.uniform(-math.pi, math.pi, 5)
        a, b = K.fk(q), m.tool_pose(q)
        worst_p = max(worst_p, float(np.abs(a[:3, 3] - b[:3, 3]).max()))
        worst_r = max(worst_r, float(np.abs(a[:3, :3] - b[:3, :3]).max()))
    assert worst_p < 1e-12 and worst_r < 1e-12, (worst_p, worst_r)
    links = ("shoulder_link", "upper_arm_link", "lower_arm_link", "wrist_link", "gripper_link")
    worst_l = worst_t = 0.0
    for _ in range(2000):
        q = rng.uniform(-math.pi, math.pi, 5)
        ang = m.arm_angles(q)
        frames = K.link_frames(q)
        assert len(frames) == 5
        for i, link in enumerate(links):
            worst_l = max(worst_l, float(np.abs(frames[i] - m.link_pose(link, ang)).max()))
        worst_l = max(worst_l, float(np.abs(K.gripper_link_pose(q) - m.link_pose("gripper_link", ang)).max()))
        off = rng.uniform(-0.03, 0.03, 3)
        worst_t = max(worst_t, float(np.abs(K.fk(q, K.tool_offset(*off)) - m.tool_pose(q, off)).max()))
    assert worst_l < 1e-12 and worst_t < 1e-12, (worst_l, worst_t)
    note(f"20000 random q: max |dp| {worst_p:.1e} m, max |dR| {worst_r:.1e}; link frames {worst_l:.1e}; "
         f"tool_offset {worst_t:.1e}")


def test_frame_conventions():
    """Zero pose, sign of every joint, approach / jaw-open axes (numbers from refs/research/urdf-chain.md)."""
    q0 = np.zeros(5)
    s0 = K.pose_summary(q0)
    assert np.allclose(s0["xyz"], (0.3913619, -0.0000112, 0.2264687), atol=5e-6), s0["xyz"]
    assert np.allclose(s0["approach"], (1.0, 0.0, 0.0), atol=1e-5)
    assert abs(s0["pitch_down_deg"]) < 1e-3
    jaw0 = K.jaw_open_axis(s0["T"])
    assert np.allclose(jaw0, (0.0, -0.0487, 0.9988), atol=1e-4), jaw0  # moving jaw on top at the zero pose
    assert np.allclose(K._APPROACH_LOCAL, (0, 0, 1)) and np.allclose(K._JAW_OPEN_LOCAL, (-1, 0, 0))
    expect_mm = {0: (-5.36, -61.22, 0.00), 1: (14.18, 0.0, -57.61), 2: (-4.94, 0.0, -51.03), 3: (-3.79, 0.0, -27.53),
                 4: (0.00, -1.37, 0.09)}
    for i, exp in expect_mm.items():
        q = np.zeros(5)
        q[i] = math.radians(10.0)
        s = K.pose_summary(q)
        assert np.allclose((s["xyz"] - s0["xyz"]) * 1000.0, exp, atol=0.02), (i, (s["xyz"] - s0["xyz"]) * 1000.0)
        assert abs(s["pitch_down_deg"] - (10.0 if i in (1, 2, 3) else 0.0)) < 1e-3, (i, s["pitch_down_deg"])
    # +pan turns the arm towards -y; pitch is measured in the arm's own vertical plane
    q = np.array([0.7, 0.3, 0.2, 0.9, 0.4])
    t = K.fk(q)
    assert t[1, 3] < 0.0
    assert abs(K.pitch_down_of(t, q[0]) - (0.3 + 0.2 + 0.9)) < 2e-5
    # tool_offset displaces along the tool's own axes: +z is out of the fingertips
    tip = K.fk(q, K.tool_offset(0.0, 0.0, 0.01))
    assert np.allclose(tip[:3, 3] - t[:3, 3], 0.01 * K.approach_axis(t), atol=1e-12)
    side = K.fk(q, K.tool_offset(-0.01, 0.0, 0.0))
    assert np.allclose(side[:3, 3] - t[:3, 3], 0.01 * K.jaw_open_axis(t), atol=1e-12)


# ============================================================================ IK
def test_wrap_pi():
    rng = np.random.default_rng(1)
    for _ in range(20000):
        a = float(rng.uniform(-math.pi, math.pi))
        assert K._wrap_pi(a) is a or K._wrap_pi(a) == a  # unchanged, bit for bit
        b = float(rng.uniform(-40.0, 40.0))
        w = K._wrap_pi(b)
        assert -math.pi < w <= math.pi
        assert abs(math.remainder(w - b, 2.0 * math.pi)) < 1e-12
    assert K._wrap_pi(math.pi) == math.pi and K._wrap_pi(-math.pi) == math.pi  # the interval is (-pi, pi]
    for k in (3.0, -3.0, 5.0, -7.0):
        assert abs(abs(K._wrap_pi(k * math.pi)) - math.pi) < 1e-12
    assert abs(K._wrap_pi(2.0 * math.pi + 0.25) - 0.25) < 1e-12 and abs(K._wrap_pi(-2.0 * math.pi - 0.25) + 0.25) < 1e-12
    assert K._wrap_pi(0.0) == 0.0 and math.isnan(K._wrap_pi(float("nan")))


def test_ik_round_trip_all_rolls_and_tools():
    """ik(fk(q)) over the real joint-limit box, every wrist roll, three tools.
    Elbow-up poses in front of the pan axis must come back as the same joint vector; no pose may
    be refused; whatever is returned as ok must be the requested pose."""
    rng = np.random.default_rng(7)
    lines = []
    for label, tool, _ in TOOLS:
        n_front = n_strict = n_other_ok = 0
        worst_dq = worst_pos = worst_pitch = 0.0
        for _ in range(6000):
            q = rng.uniform(LO, HI)
            t = K.fk(q, tool)
            pitch = math.degrees(K.pitch_down_of(t, q[0]))
            fwd, gamma = branch_of(q, tool)
            q2, info = K.ik(t[:3, 3], pitch, q[4], tool=tool, limits=REAL)
            front = fwd > 0.002 and gamma > 0.02
            if front:
                n_front += 1
                assert info["ok"], (label, np.degrees(q).round(2), info["reason"])
            if not info["ok"]:
                continue
            # an ok answer is always inside the limits and reproduces the requested pose
            assert in_limits(q2, REAL), (label, np.degrees(q2))
            t2 = K.fk(q2, tool)
            pos = float(np.linalg.norm(t2[:3, 3] - t[:3, 3]))
            pit = abs(float(wrap(K.pitch_down_of(t2, q2[0]) - math.radians(pitch))))
            worst_pos, worst_pitch = max(worst_pos, pos), max(worst_pitch, pit)
            assert pos < 1e-5 and pit < 1e-4, (label, np.degrees(q).round(2), pos, pit)
            assert q2[4] == q[4]
            if not front:
                n_other_ok += 1
            elif fwd > 0.02 and 0.05 < gamma < math.pi - 0.05:  # away from the two singularities
                n_strict += 1
                dq = float(np.max(np.abs(wrap(q2 - q))))
                worst_dq = max(worst_dq, dq)
                assert dq < 1e-3, (label, np.degrees(q).round(3), np.degrees(q2).round(3))
        assert n_front > 3000 and n_strict > 2500
        lines.append(f"{label}: {n_front} elbow-up/front poses all solved, {n_strict} away from singularities "
                     f"returned the same q (max |dq| {worst_dq:.1e} rad); {n_other_ok} other-branch poses solved "
                     f"with a different q; worst pose error of any ok answer {worst_pos * 1e6:.2f} um / "
                     f"{math.degrees(worst_pitch):.5f} deg")
    for line in lines:
        note(line)


def test_ik_random_targets_verified_by_urdf_model():
    """Every answer ik() calls ok is re-checked with the independent FK and the limits."""
    m = model()
    rng = np.random.default_rng(3)
    n_ok = 0
    worst_pos = worst_pitch = 0.0
    reasons = {}
    n = 30000
    for k in range(n):
        label, tool, off = TOOLS[k % 3]
        xyz = np.array([rng.uniform(-0.15, 0.48), rng.uniform(-0.40, 0.40), rng.uniform(-0.03, 0.45)])
        pitch, roll = rng.uniform(-60.0, 150.0), math.radians(rng.uniform(-160.0, 160.0))
        q, info = K.ik(xyz, pitch, roll, tool=tool, limits=REAL)
        if not info["ok"]:
            key = info["reason"].split("=")[0].split("(")[0].strip()
            reasons[key] = reasons.get(key, 0) + 1
            assert info["reason"]
            continue
        n_ok += 1
        assert in_limits(q, REAL) and q[4] == roll
        gamma = q[2] + (K._A1 - K._A2)
        assert 0.0 <= gamma <= math.pi  # elbow-up branch only
        t = m.tool_pose(q, off)
        pos = float(np.linalg.norm(t[:3, 3] - xyz))
        pit = ang_diff_deg(m.pitch_down_deg(t, q[0]), pitch)
        worst_pos, worst_pitch = max(worst_pos, pos), max(worst_pitch, pit)
        assert pos < 1e-5 and pit < 0.01, (xyz, pitch, pos, pit)
        assert info["pos_err_m"] < 1e-5 and info["pitch_err_deg"] < 0.01 and info["reason"] == ""
    assert n_ok > 3000, n_ok
    note(f"{n} random targets, {n_ok} ok; independent FK: max position error {worst_pos * 1e6:.2f} um, "
         f"max pitch error {worst_pitch:.5f} deg; refusals {reasons}")


def test_ik_limit_handling():
    """A solution outside the limits is reported (joint named), never clipped; limits are inclusive
    boxes taken from the `limits` argument (URDF limits by default)."""
    cases = {  # joint index -> a pose that violates only that joint's REAL limit
        0: np.radians([103.0, 20.0, 30.0, 40.0, 10.0]),      # pan limit +100
        1: np.radians([10.0, -97.0, 60.0, 80.0, 0.0]),       # lift limit -94.13
        2: np.radians([10.0, -40.0, 88.0, 40.0, 0.0]),       # elbow limit +85.43
        3: np.radians([10.0, 0.0, 20.0, 105.5, 0.0]),        # wrist_flex limit +103.18
        4: np.radians([10.0, 20.0, 30.0, 40.0, 165.0]),      # wrist_roll limit +160
    }
    for idx, q in cases.items():
        for label, tool, _ in TOOLS:
            fwd, gamma = branch_of(q, tool)
            assert fwd > 0.02 and 0.05 < gamma < math.pi - 0.05
            bad = [i for i, j in enumerate(K.ARM_JOINTS) if not REAL[j][0] <= q[i] <= REAL[j][1]]
            assert bad == [idx], (idx, bad)
            t = K.fk(q, tool)
            pitch = math.degrees(K.pitch_down_of(t, q[0]))
            q2, info = K.ik(t[:3, 3], pitch, q[4], tool=tool, limits=REAL)
            name = K.ARM_JOINTS[idx]
            assert info["ok"] is False and info["reason"].startswith(name + "="), info
            assert "outside" in info["reason"]
            assert float(np.max(np.abs(wrap(q2 - q)))) < 1e-3, "the reported joint vector was clipped"
            q3, info3 = K.ik(t[:3, 3], pitch, q[4], tool=tool, limits=WIDE)
            assert info3["ok"] and float(np.max(np.abs(wrap(q3 - q)))) < 1e-3
    # default limits are the URDF ones: wrist_flex 97 deg is outside +-95 but inside the real 103.18
    q = np.radians([10.0, 0.0, 20.0, 97.0, 0.0])
    t = K.fk(q)
    pitch = math.degrees(K.pitch_down_of(t, q[0]))
    assert K.ik(t[:3, 3], pitch, 0.0)[1]["ok"] is False
    assert K.ik(t[:3, 3], pitch, 0.0, limits=None)[1]["reason"].startswith("wrist_flex=97.0 deg outside [-95.0, 95.0]")
    assert K.ik(t[:3, 3], pitch, 0.0, limits=REAL)[1]["ok"] is True
    # q_init is accepted and ignored (API compatibility): the branch is unique
    a = K.ik(t[:3, 3], pitch, 0.0, q_init=np.ones(5), limits=REAL)[0]
    b = K.ik(t[:3, 3], pitch, 0.0, limits=REAL)[0]
    assert a.tobytes() == b.tobytes()
    # grid: whatever is ok lies inside the box that was passed
    n_ok = 0
    for x in np.arange(0.08, 0.46, 0.03):
        for z in (0.0, 0.05, 0.1, 0.2):
            for pitch in range(-20, 121, 10):
                q, info = K.ik((x, 0.03, z), pitch, 0.2, limits=REAL)
                if info["ok"]:
                    n_ok += 1
                    assert in_limits(q, REAL)
                    assert np.linalg.norm(K.fk(q)[:3, 3] - np.array([x, 0.03, z])) < 1e-5
    assert n_ok > 200


def test_ik_rejects_bad_input_and_unreachable_targets():
    q, info = K.ik((0.9, 0.0, 0.1), 45.0, 0.0, limits=REAL)
    assert info["ok"] is False and "out of reach" in info["reason"] and info["pos_err_m"] == float("inf")
    q, info = K.ik((K._X_PAN, 0.0, 0.2), 45.0, 0.0, limits=REAL)
    assert info["ok"] is False and info["reason"] == "target on the pan axis"
    for bad in ((float("nan"), 0.0, 0.1), (0.2, float("inf"), 0.1), (0.2, 0.0, float("nan"))):
        q, info = K.ik(bad, 90.0, 0.0, limits=REAL)
        assert info["ok"] is False and "finite" in info["reason"], info
        assert K.pitch_range(bad, 0.0, limits=REAL) is None and K.feasible_pitches(bad, 0.0, limits=REAL) == []
        assert K.best_pitch(bad, 90.0, 0.0, limits=REAL) == (None, True)
    assert K.ik((0.2, 0.0, 0.1), float("nan"), 0.0, limits=REAL)[1]["ok"] is False
    assert K.ik((0.2, 0.0, 0.1), 90.0, float("nan"), limits=REAL)[1]["ok"] is False
    assert K.ik((0.2, 0.0, 0.05), 90.0, math.radians(170.0), limits=REAL)[1]["reason"].startswith("wrist_roll=170.0")


def test_ik_wrap_cases_ki5():
    """Review finding KI-5: reachable poses were refused because pan / wrist_flex came out of the
    formula one full turn away from their value inside the limits."""
    old = v1()
    cases = (
        ("pan near +100 deg with the tool almost on the pan axis", np.radians([98.64, 60.0, 21.57, 59.73, 7.92]), BIG,
         "shoulder_pan"),
        ("gripper pointing up, arm folded forward", np.radians([35.37, 96.28, 75.12, 98.96, 115.11]), None,
         "wrist_flex"),
        ("the pose quoted in the review text", np.radians([-94.1, 104.5, 80.2, 94.5, 0.05]), TIP, "wrist_flex"),
    )
    for label, q, tool, joint in cases:
        assert in_limits(q, REAL)
        t = K.fk(q, tool)
        pitch = math.degrees(K.pitch_down_of(t, q[0]))
        q_old, info_old = old.ik(t[:3, 3], pitch, q[4], tool=tool, limits=REAL)
        assert info_old["ok"] is False and info_old["reason"].startswith(joint + "="), (label, info_old)
        assert abs(math.degrees(q_old[K.ARM_JOINTS.index(joint)])) > 180.0
        q_new, info = K.ik(t[:3, 3], pitch, q[4], tool=tool, limits=REAL)
        assert info["ok"], (label, info)
        assert float(np.max(np.abs(q_new - q))) < 2e-3, (label, np.degrees(q_new), np.degrees(q))
        note(f"{label}: v1 '{info_old['reason']}' -> now solved")
    # statistics over the real limit box: v1 refused a few reachable elbow-up/front poses, now none is
    rng = np.random.default_rng(5)
    n = refused_old = refused_new = 0
    for _ in range(25000):
        q = rng.uniform(LO, HI)
        fwd, gamma = branch_of(q, TIP)
        if fwd <= 0.002 or gamma < 0.02:
            continue
        n += 1
        t = K.fk(q, TIP)
        pitch = math.degrees(K.pitch_down_of(t, q[0]))
        refused_new += not K.ik(t[:3, 3], pitch, q[4], tool=TIP, limits=REAL)[1]["ok"]
        refused_old += not old.ik(t[:3, 3], pitch, q[4], tool=TIP, limits=REAL)[1]["ok"]
    assert refused_new == 0, refused_new
    assert refused_old > 0  # the defect really existed
    note(f"{n} random in-limit elbow-up/front poses: v1 refused {refused_old}, now {refused_new}")
    # the requested pitch is an angle too: +-360 deg and the +-180 deg seam give the same answer
    # (the seam is only reachable with the gripper folded back under the forearm, i.e. outside the real
    # wrist_flex limit, hence the +-10 rad limits here)
    q = np.array([0.3, 1.2, -0.9, math.pi - 0.3, 0.2])  # approach axis pointing straight back: pitch 180
    fwd, gamma = branch_of(q, TIP)
    assert fwd > 0.02 and gamma > 0.05
    t = K.fk(q, TIP)
    assert abs(abs(K.pitch_down_of(t, q[0])) - math.pi) < 1e-4
    for pitch in (180.0, -180.0, 540.0, -540.0):
        q2, info = K.ik(t[:3, 3], pitch, q[4], tool=TIP, limits=WIDE)
        assert info["ok"], (pitch, info)
        assert float(np.max(np.abs(wrap(q2 - q)))) < 1e-3 and float(np.max(np.abs(q2[:4]))) <= math.pi
    xyz = (0.22, 0.05, 0.04)
    qa, ia = K.ik(xyz, 80.0, 0.3, tool=TIP, limits=REAL)
    for turns in (-2, -1, 1, 2):
        qb, ib = K.ik(xyz, 80.0 + 360.0 * turns, 0.3, tool=TIP, limits=REAL)
        assert ia["ok"] and ib["ok"] and float(np.max(np.abs(qa - qb))) < 1e-9, (turns, ib)


def test_ik_regression_against_v1():
    """ik() gives bit-for-bit the answers of the version that drove the real arm, for every
    target that version solved; the only new answers are the wrap cases of KI-5."""
    old = v1()
    rng = np.random.default_rng(11)
    totals = {"targets": 0, "v1_ok": 0, "identical": 0, "same_mod_2pi": 0, "newly_solved": 0}
    for label, tool, off in TOOLS:
        for lname, lim in (("urdf", None), ("real", REAL), ("wide", WIDE)):
            lim_used = lim or K.URDF_LIMITS
            for k in range(2400):
                if k % 2:  # the working volume of the harness
                    xyz = (rng.uniform(0.05, 0.42), rng.uniform(-0.25, 0.25), rng.uniform(-0.03, 0.30))
                    pitch = rng.uniform(-20.0, 120.0)
                    roll = math.radians(rng.uniform(-160.0, 160.0))
                else:      # everything else, including behind the base and the gripper pointing up
                    xyz = (rng.uniform(-0.15, 0.48), rng.uniform(-0.40, 0.40), rng.uniform(-0.03, 0.45))
                    pitch = rng.uniform(-179.9, 180.0)
                    roll = math.radians(rng.uniform(-170.0, 170.0))
                q_old, i_old = old.ik(xyz, pitch, roll, tool=tool, limits=lim)
                q_new, i_new = K.ik(xyz, pitch, roll, tool=tool, limits=lim)
                totals["targets"] += 1
                if i_old["ok"]:
                    totals["v1_ok"] += 1
                    assert i_new["ok"], (label, lname, xyz, pitch, i_new)
                    if lname != "wide":
                        assert q_new.tobytes() == q_old.tobytes() and i_new == i_old, (label, lname, xyz, pitch)
                        totals["identical"] += 1
                    elif q_new.tobytes() == q_old.tobytes():
                        assert i_new == i_old
                        totals["identical"] += 1
                    else:  # +-10 rad limits let v1 return e.g. wrist_flex = -190 deg; now normalised
                        assert float(np.max(np.abs(wrap(q_new - q_old)))) < 1e-9
                        assert float(np.max(np.abs(q_old))) > math.pi and float(np.max(np.abs(q_new[:4]))) <= math.pi
                        assert np.allclose(K.fk(q_new, tool), old.fk(q_old, tool), atol=1e-9)
                        totals["same_mod_2pi"] += 1
                elif i_new["ok"]:
                    totals["newly_solved"] += 1
                    r = i_old["reason"]  # v1 must have refused it for an angle outside +-180 deg
                    assert "outside" in r and abs(float(r.split("=")[1].split(" deg")[0])) > 180.0, (label, lname, xyz, r)
                    assert in_limits(q_new, lim_used)
                    t = model().tool_pose(q_new, off)
                    assert np.linalg.norm(t[:3, 3] - np.array(xyz)) < 1e-5
                    assert ang_diff_deg(model().pitch_down_deg(t, q_new[0]), pitch) < 0.01
    assert totals["v1_ok"] > 3000
    note(f"{totals['targets']} random targets (3 tools x URDF / real / +-10 rad limits): v1 solved {totals['v1_ok']}; "
         f"{totals['identical']} answers identical bit for bit, {totals['same_mod_2pi']} equal modulo 2*pi "
         f"(only with +-10 rad limits), {totals['newly_solved']} newly solved (all KI-5 wrap cases, verified by FK)")
    # requested pitches beyond +-180 deg: v1 could only fail (or, within 0.2 deg of the seam, approximate);
    # now they are the same request as the pitch modulo 360
    n_beyond = 0
    for k in range(1500):
        label, tool, off = TOOLS[k % 3]
        xyz = (rng.uniform(-0.15, 0.48), rng.uniform(-0.40, 0.40), rng.uniform(-0.03, 0.45))
        pitch = rng.uniform(180.5, 400.0) * (1 if k % 2 else -1)
        roll = math.radians(rng.uniform(-160.0, 160.0))
        assert old.ik(xyz, pitch, roll, tool=tool, limits=REAL)[1]["ok"] is False
        q_new, i_new = K.ik(xyz, pitch, roll, tool=tool, limits=REAL)
        q_eq, i_eq = K.ik(xyz, (pitch + 180.0) % 360.0 - 180.0, roll, tool=tool, limits=REAL)
        assert i_new["ok"] == i_eq["ok"]
        if i_new["ok"]:
            n_beyond += 1
            assert float(np.max(np.abs(q_new - q_eq))) < 1e-9
    assert n_beyond > 50
    note(f"1500 targets with a requested pitch beyond +-180 deg: v1 refused all, {n_beyond} are now solved exactly "
         f"like the same pitch modulo 360")


# ============================================================================ pitch intervals (KI-4)
REVIEW_HOLES = (  # (x, y, z, roll_deg), pitch_range, infeasible pitches inside it: refs/patches/kinematics-review/t04.out
    ((0.18, 0.15, 0.0, 0.0), (22, 110), list(range(26, 37, 2))),
    ((0.18, 0.15, 0.0, 90.0), (24, 110), list(range(28, 39, 2))),
    ((0.20, 0.15, 0.02, 0.0), (16, 102), list(range(22, 29, 2))),
    ((0.20, 0.15, 0.02, 90.0), (18, 104), list(range(24, 31, 2))),
    ((0.22, 0.15, 0.15, 0.0), (-20, 68), list(range(-18, -3, 2))),
    ((0.22, 0.15, 0.15, 90.0), (-18, 68), list(range(-16, -1, 2))),
    ((0.26, 0.0, 0.02, 0.0), (16, 102), [22, 24, 26]),
    ((0.26, 0.0, 0.02, 90.0), (18, 104), list(range(24, 31, 2))),
    ((0.28, 0.0, 0.10, 0.0), (-6, 80), list(range(-2, 7, 2))),
    ((0.28, 0.0, 0.10, 90.0), (-4, 82), list(range(0, 11, 2))),
    ((0.28, 0.0, 0.15, 0.0), (-20, 66), list(range(-14, -7, 2))),
    ((0.28, 0.0, 0.15, 90.0), (-18, 68), list(range(-12, -5, 2))),
)


def brute_force_intervals(xyz, roll, tool, lim, lo=-20, hi=120, step=2):
    grid = list(range(lo, hi + 1, step))
    return K._runs(grid, [bool(K.ik(xyz, p, roll, tool=tool, limits=lim)[1]["ok"]) for p in grid])


def test_runs_helper():
    assert K._runs([], []) == []
    assert K._runs([1, 2, 3], [False, False, False]) == []
    assert K._runs([1, 2, 3], [True, True, True]) == [(1, 3)]
    assert K._runs([0, 2, 4, 6, 8, 10], [True, False, True, True, False, True]) == [(0, 0), (4, 6), (10, 10)]
    assert K._runs([5], [True]) == [(5, 5)]
    assert K._runs([0, 1, 2, 3], [False, True, True, False]) == [(1, 2)]


def test_feasible_pitches_holes_ki4():
    """The concrete points of review finding KI-4 (fingertip tool, real limits)."""
    assert K.feasible_pitches((0.26, 0.0, 0.02), 0.0, tool=TIP, limits=REAL) == [(16, 20), (28, 102)]
    assert K.feasible_pitches((0.28, 0.0, 0.10), 0.0, tool=TIP, limits=REAL) == [(-6, -4), (8, 80)]
    assert K.feasible_pitches((0.22, 0.15, 0.15), 0.0, tool=TIP, limits=REAL) == [(-20, -20), (-2, 68)]
    assert K.feasible_pitches((0.18, 0.15, 0.0), 0.0, tool=TIP, limits=REAL) == [(22, 24), (38, 110)]
    q, info = K.ik((0.26, 0.0, 0.02), 24.0, 0.0, tool=TIP, limits=REAL)
    assert info["ok"] is False and info["reason"].startswith("elbow_flex=86.0 deg outside")
    for (x, y, z, roll_deg), rng_expected, holes in REVIEW_HOLES:
        roll = math.radians(roll_deg)
        iv = K.feasible_pitches((x, y, z), roll, tool=TIP, limits=REAL)
        assert K.pitch_range((x, y, z), roll, tool=TIP, limits=REAL) == rng_expected, ((x, y, z, roll_deg), iv)
        feasible = {p for a, b in iv for p in range(a, b + 1, 2)}
        got_holes = [p for p in range(rng_expected[0], rng_expected[1] + 1, 2) if p not in feasible]
        assert got_holes == holes, ((x, y, z, roll_deg), iv, got_holes)
        assert len(iv) == 2 and iv == brute_force_intervals((x, y, z), roll, TIP, REAL)
        assert all(isinstance(v, int) for pair in iv for v in pair)
    # the reviewer's grid: how common are holes?
    reachable = with_holes = 0
    for x in np.arange(-0.06, 0.46, 0.02):
        for y in (0.0, 0.15):
            for z in (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4):
                for roll_deg in (0.0, 90.0):
                    iv = K.feasible_pitches((float(x), y, z), math.radians(roll_deg), tool=TIP, limits=REAL)
                    reachable += bool(iv)
                    with_holes += len(iv) > 1
    assert with_holes >= 12 and reachable > 500
    note(f"review grid: {reachable} reachable fingertip positions, {with_holes} with a hole in the feasible pitch set")


def test_feasible_pitches_equal_brute_force_and_pitch_range_contract():
    """feasible_pitches == a brute-force ik() scan; pitch_range == (min, max) of it == v1's answer."""
    old = v1()
    rng = np.random.default_rng(21)
    n = reachable = holes = same_as_v1 = 0
    for k in range(240):
        label, tool, _ = TOOLS[k % 3]
        lim = (None, REAL)[(k // 3) % 2]
        xyz = (rng.uniform(0.03, 0.44), rng.uniform(-0.25, 0.25), rng.uniform(-0.03, 0.32))
        roll = math.radians(rng.uniform(-160.0, 160.0))
        iv = K.feasible_pitches(xyz, roll, tool=tool, limits=lim)
        assert iv == brute_force_intervals(xyz, roll, tool, lim), (xyz, roll, iv)
        pr = K.pitch_range(xyz, roll, tool=tool, limits=lim)
        assert pr == ((iv[0][0], iv[-1][1]) if iv else None)
        assert pr is None or (isinstance(pr, tuple) and len(pr) == 2 and all(isinstance(v, int) for v in pr))
        pr_old = old.pitch_range(xyz, roll, tool=tool, limits=lim)
        if pr_old != pr:  # only possible where v1 refused a wrap case (KI-5): the new range is the wider one
            assert pr is not None and (pr_old is None or (pr[0] <= pr_old[0] and pr[1] >= pr_old[1]))
        else:
            same_as_v1 += 1
        n += 1
        reachable += bool(iv)
        holes += len(iv) > 1
    assert same_as_v1 >= n - 2, (same_as_v1, n)
    # other grids
    xyz = (0.24, 0.05, 0.05)
    for lo, hi, step in ((0, 90, 5), (-20, 120, 1), (80, 100, 2), (90, 90, 2), (-90, 180, 3)):
        got = K.feasible_pitches(xyz, 0.2, tool=TIP, limits=REAL, lo=lo, hi=hi, step=step)
        assert got == brute_force_intervals(xyz, 0.2, TIP, REAL, lo, hi, step), (lo, hi, step, got)
        assert K.pitch_range(xyz, 0.2, tool=TIP, limits=REAL, lo=lo, hi=hi, step=step) == (
            (got[0][0], got[-1][1]) if got else None)
    assert K.feasible_pitches(xyz, 0.2, tool=TIP, limits=REAL, lo=10, hi=0) == []
    for bad_step in (0, -2):
        try:
            K.feasible_pitches(xyz, 0.2, tool=TIP, limits=REAL, step=bad_step)
        except ValueError:
            pass
        else:
            raise AssertionError("step <= 0 must be refused")
    # wrist roll outside its limit, or pan outside its limit: nothing is feasible
    assert K.feasible_pitches(xyz, math.radians(170.0), tool=TIP, limits=REAL) == []
    assert K.feasible_pitches((0.0, -0.2, 0.05), 0.0, tool=TIP, limits=REAL) == brute_force_intervals(
        (0.0, -0.2, 0.05), 0.0, TIP, REAL)
    note(f"{n} random positions ({reachable} reachable, {holes} with holes): intervals equal the brute-force scan; "
         f"pitch_range equals v1 at {same_as_v1}")


def test_pitch_scan_matches_ik():
    """The closed-form feasibility test used by the scans agrees with ik() itself."""
    rng = np.random.default_rng(5)
    total = feasible = 0
    for k in range(3600):
        label, tool, _ = TOOLS[k % 3]
        lim = (None, REAL, WIDE)[(k // 3) % 3]
        if rng.random() < 0.7:
            xyz = (rng.uniform(0.05, 0.42), rng.uniform(-0.25, 0.25), rng.uniform(-0.03, 0.30))
            roll = math.radians(rng.uniform(-160.0, 160.0))
            pitches = rng.uniform(-30.0, 130.0, 10)
        else:
            xyz = (rng.uniform(-0.15, 0.48), rng.uniform(-0.40, 0.40), rng.uniform(-0.03, 0.45))
            roll = math.radians(rng.uniform(-185.0, 185.0))
            pitches = rng.uniform(-400.0, 400.0, 10)
        scan = K._PitchScan(xyz, roll, tool, lim)
        assert scan.fast
        for p in pitches:
            exact = bool(K.ik(xyz, float(p), roll, tool=tool, limits=lim)[1]["ok"])
            assert scan.feasible(float(p)) == exact == scan.exact(float(p)), (label, xyz, roll, p)
            total += 1
            feasible += exact
    assert feasible > 5000
    note(f"{total} random (position, roll, pitch, tool, limits) samples, {feasible} feasible: 0 disagreements")


def test_pitch_scan_exact_path_and_fallback():
    """Tools that are not plain offsets use ik() directly; a disagreement between the closed form
    and ik() is detected, reported (RuntimeWarning) and replaced by the exact scan."""
    c, s = math.cos(0.2), math.sin(0.2)
    tilted = K.tool_offset(0.0, 0.0, 0.01) @ np.array([[c, 0, s, 0], [0, 1, 0, 0], [-s, 0, c, 0], [0, 0, 0, 1.0]])
    xyz = (0.22, 0.03, 0.05)
    scan = K._PitchScan(xyz, 0.1, tilted, REAL)
    assert scan.fast is False
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        iv = K.feasible_pitches(xyz, 0.1, tool=tilted, limits=REAL, lo=0, hi=120, step=8)
        assert iv == brute_force_intervals(xyz, 0.1, tilted, REAL, 0, 120, 8) and iv
        p, degraded = K.best_pitch(xyz, 150.0, 0.1, tool=tilted, limits=REAL, lo=60.0, hi=120.0)
        assert degraded and p is not None and K.ik(xyz, p, 0.1, tool=tilted, limits=REAL)[1]["ok"]
    # make the closed form lie and check that the public functions notice
    before = K._SCAN_FALLBACKS
    original = K._PitchScan.feasible
    K._PitchScan.feasible = lambda self, pitch_deg: True
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            iv = K.feasible_pitches((0.26, 0.0, 0.02), 0.0, tool=TIP, limits=REAL)
            p, degraded = K.best_pitch((0.20, 0.0, 0.12), 90.0, 0.0, tool=TIP, limits=REAL)
        assert iv == [(16, 20), (28, 102)]
        assert degraded and p is not None and K.ik((0.20, 0.0, 0.12), p, 0.0, tool=TIP, limits=REAL)[1]["ok"]
        assert sum(issubclass(w.category, RuntimeWarning) for w in caught) == 2
        assert K._SCAN_FALLBACKS == before + 2
    finally:
        K._PitchScan.feasible = original
        K._SCAN_FALLBACKS = before
    assert K.feasible_pitches((0.26, 0.0, 0.02), 0.0, tool=TIP, limits=REAL) == [(16, 20), (28, 102)]


def test_best_pitch():
    def ok(xyz, p, roll=0.0, tool=TIP, lim=REAL):
        return bool(K.ik(xyz, p, roll, tool=tool, limits=lim)[1]["ok"])

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a fallback would be a failure here
        # 1. the wanted pitch itself when it is feasible, untouched
        for want in (90.0, 90, 75.5, 50):
            p, degraded = K.best_pitch((0.22, 0.02, 0.03), want, 0.2, tool=TIP, limits=REAL)
            assert p == float(want) and isinstance(p, float) and degraded is False
        # 2. inside the hole of KI-4: nearest feasible pitch, on the correct side, 1 deg inside
        xyz = (0.26, 0.0, 0.02)  # feasible: about 15.3..21.1 and 27.8..102.9
        p, degraded = K.best_pitch(xyz, 24.0, 0.0, tool=TIP, limits=REAL)
        assert degraded is True and 19.5 < p < 20.5 and ok(xyz, p), p
        assert ok(xyz, p + 0.99) and not ok(xyz, p + 1.01)
        p, degraded = K.best_pitch(xyz, 26.0, 0.0, tool=TIP, limits=REAL)
        assert degraded is True and 28.5 < p < 29.5 and ok(xyz, p) and not ok(xyz, p - 1.01), p
        assert K.best_pitch(xyz, 90.0, 0.0, tool=TIP, limits=REAL) == (90.0, False)
        # v1's helper (fd._best_pitch) returned an infeasible pitch here: clamp(want, min + 1, max - 1) = 24
        assert not ok(xyz, 24.0)
        # 3. hover height: straight down is impossible above z = 0.12 (wrist_flex limit)
        hover = (0.20, 0.0, 0.12)
        p, degraded = K.best_pitch(hover, 90.0, 0.0, tool=TIP, limits=REAL)
        assert degraded is True and 70.0 < p < 90.0 and ok(hover, p) and not ok(hover, 90.0)
        assert ok(hover, p + 0.99) and not ok(hover, p + 1.01)
        p0, degraded = K.best_pitch(hover, 90.0, 0.0, tool=TIP, limits=REAL, margin_deg=0.0)
        assert degraded and abs(p0 - (p + 1.0)) < 0.01 and ok(hover, p0) and not ok(hover, p0 + 0.01)
        # 4. nothing feasible
        assert K.best_pitch((0.60, 0.0, 0.0), 90.0, 0.0, tool=TIP, limits=REAL) == (None, True)
        assert K.best_pitch((0.22, 0.0, 0.03), 90.0, math.radians(170.0), tool=TIP, limits=REAL) == (None, True)
        # 5. search window
        p, degraded = K.best_pitch(hover, 90.0, 0.0, tool=TIP, limits=REAL, lo=85.0, hi=120.0)
        assert (p, degraded) == (None, True)
        # 6. random cases: feasible, flagged, and nothing feasible is closer to `want`
        rng = np.random.default_rng(9)
        n = n_degraded = n_none = 0
        for k in range(300):
            label, tool, _ = TOOLS[k % 3]
            lim = (None, REAL)[(k // 3) % 2]
            xyz = (rng.uniform(0.05, 0.42), rng.uniform(-0.22, 0.22), rng.uniform(-0.03, 0.30))
            roll = math.radians(rng.uniform(-150.0, 150.0))
            want = float(rng.choice([90.0, 90.0, 60.0, 30.0, 0.0, 110.0, rng.uniform(-20.0, 120.0)]))
            p, degraded = K.best_pitch(xyz, want, roll, tool=tool, limits=lim)
            n += 1
            if p is None:
                n_none += 1
                assert degraded is True and K.feasible_pitches(xyz, roll, tool=tool, limits=lim, step=1) == []
                continue
            assert ok(xyz, p, roll, tool, lim), (xyz, want, p)
            assert -20.0 <= p <= 120.0
            if not degraded:
                assert p == want
                continue
            n_degraded += 1
            assert not ok(xyz, want, roll, tool, lim)
            reach = abs(p - want) - 1.0 - 0.05  # everything closer than (answer - margin) must be infeasible
            for d in np.linspace(0.0, max(reach, 0.0), 25):
                for cand in (want - d, want + d):
                    if -20.0 <= cand <= 120.0 and reach > 0.0:
                        assert not ok(xyz, float(cand), roll, tool, lim), (xyz, want, p, cand)
    for bad in (float("nan"), float("inf")):
        try:
            K.best_pitch((0.2, 0.0, 0.05), bad, 0.0, tool=TIP, limits=REAL)
        except ValueError:
            pass
        else:
            raise AssertionError("a non-finite wanted pitch must be refused")
    note(f"{n} random cases: {n - n_degraded - n_none} exact, {n_degraded} degraded (all feasible, none with a "
         f"feasible pitch more than 1 deg closer to the wanted one), {n_none} out of reach")


# ============================================================================ jaws in the base frame
def test_jaw_tilt_and_azimuth():
    rng = np.random.default_rng(13)
    worst = 0.0
    for _ in range(5000):
        q = rng.uniform(LO, HI)
        theta = q[1] + q[2] + q[3]
        rho = q[4] - K.WRIST_ROLL_SKEW
        expect = math.degrees(math.asin(math.cos(theta) * math.cos(rho)))
        worst = max(worst, abs(K.jaw_tilt_deg(q) - expect))
        t = K.fk(q)
        assert abs(K.jaw_tilt_deg(q) - math.degrees(math.asin(K.jaw_open_axis(t)[2]))) < 1e-9
        az = K.jaw_azimuth_deg(q)
        jaw = K.jaw_open_axis(t)
        if math.hypot(jaw[0], jaw[1]) < 0.5:
            assert az is None and abs(K.jaw_tilt_deg(q)) > 59.9
        else:
            assert abs(az - math.degrees(math.atan2(jaw[1], jaw[0]))) < 1e-9 and -180.0 <= az <= 180.0
    assert worst < 2e-3, worst
    # straight down: jaws level for every roll, azimuth = roll - pan - 2.789 deg
    for roll_deg in (-150.0, -90.0, -30.0, 0.0, 2.789, 45.0, 92.789, 150.0):
        for xyz in ((0.25, 0.0, 0.05), (0.20, -0.12, 0.02), (0.18, 0.10, 0.06)):
            q, info = K.ik(xyz, 90.0, math.radians(roll_deg), tool=TIP, limits=REAL)
            assert info["ok"]
            assert abs(K.jaw_tilt_deg(q)) < 2e-3
            assert ang_diff_deg(K.jaw_azimuth_deg(q), roll_deg - math.degrees(q[0]) - 2.789) < 2e-3
    # review numbers (KI-2): pitch 70 / 60 with roll 0 -> jaw line tilted 20 / 30 deg, a point 40 mm across
    # the gap is 13.7 / 20 mm higher than the fingertip; |roll| > 90 puts the moving jaw below
    for pitch, tilt, rise in ((70.0, 20.0, 13.7), (60.0, 30.0, 20.0)):
        q, info = K.ik((0.22, 0.0, 0.04), pitch, 0.0, tool=TIP, limits=REAL)
        assert info["ok"] and abs(K.jaw_tilt_deg(q) - tilt) < 0.1, K.jaw_tilt_deg(q)
        assert abs(40.0 * math.sin(math.radians(K.jaw_tilt_deg(q))) - rise) < 0.1
        q, info = K.ik((0.22, 0.0, 0.04), pitch, math.radians(150.0), tool=TIP, limits=REAL)
        assert info["ok"] and K.jaw_tilt_deg(q) < -15.0
    q0 = np.zeros(5)  # horizontal gripper, moving jaw on top: tilt 87.2 deg, no meaningful azimuth
    assert abs(K.jaw_tilt_deg(q0) - 87.21) < 0.01 and K.jaw_azimuth_deg(q0) is None
    assert K.jaw_azimuth_deg(q0, min_horizontal=0.01) is not None
    note(f"jaw_tilt_deg equals asin(cos(pitch) * cos(roll - 2.789 deg)) within {worst:.1e} deg on 5000 random poses")


def test_roll_for_azimuth_vs_fk():
    """The closed form against jaw_open_axis(fk(q)) for random pan / azimuth / pitch."""
    rng = np.random.default_rng(17)
    worst_down = worst_other = worst_tilt = 0.0
    for _ in range(12000):
        pan, psi = rng.uniform(-100.0, 100.0), rng.uniform(-180.0, 180.0)
        for pitch in (90.0, rng.uniform(20.0, 160.0), rng.uniform(-160.0, -20.0)):
            roll = K.roll_for_azimuth(psi, pan, pitch)
            assert -180.0 < roll <= 180.0
            lift, elbow = rng.uniform(-1.5, 1.5), rng.uniform(-1.5, 1.4)
            q = np.array([math.radians(pan), lift, elbow, math.radians(pitch) - lift - elbow, math.radians(roll)])
            jaw = K.jaw_open_axis(K.fk(q))
            err = ang_diff_deg(math.degrees(math.atan2(jaw[1], jaw[0])), psi)
            if pitch == 90.0:
                worst_down = max(worst_down, err)
                assert abs(jaw[2]) < 1e-4  # jaws level
                assert ang_diff_deg(roll, psi + pan + math.degrees(K.WRIST_ROLL_SKEW)) < 1e-9
            else:
                worst_other = max(worst_other, err)
                rho = math.radians(roll) - K.WRIST_ROLL_SKEW
                tilt = math.degrees(math.asin(math.cos(math.radians(pitch)) * math.cos(rho)))
                worst_tilt = max(worst_tilt, abs(K.jaw_tilt_deg(q) - tilt))
            # the other jaw order is exactly half a turn away
            assert ang_diff_deg(K.roll_for_azimuth(psi + 180.0, pan, pitch), roll + 180.0) < 1e-9
    assert worst_down < 0.005 and worst_other < 0.01 and worst_tilt < 0.01, (worst_down, worst_other, worst_tilt)
    # same convention as the research helper in gripper_model (which also applies the URDF roll limits)
    for psi, pan in ((0.0, 0.0), (30.0, -20.0), (-120.0, 40.0), (100.0, 55.0), (170.0, 0.0)):
        roll, flipped = GM.wrist_roll_for_azimuth(psi, pan)
        mine = K.roll_for_azimuth(psi + (180.0 if flipped else 0.0), pan)
        assert ang_diff_deg(mine, roll) < 1e-3, (psi, pan, roll, mine)
    assert abs(K.roll_for_azimuth(0.0, 0.0) - 2.789) < 2e-4
    assert abs(K.roll_for_azimuth(90.0, 0.0, 90.0) - 92.789) < 2e-4      # roll +92.8 opens toward +y
    assert abs(K.roll_for_azimuth(400.0, 0.0) - K.roll_for_azimuth(40.0, 0.0)) < 1e-9
    seam = K.roll_for_azimuth(180.0 - math.degrees(K.WRIST_ROLL_SKEW), 0.0)  # roll of exactly half a turn
    assert -180.0 < seam <= 180.0 and abs(abs(seam) - 180.0) < 1e-9
    for flat in (0.0, 180.0, -180.0, 360.0):
        try:
            K.roll_for_azimuth(10.0, 0.0, flat)
        except ValueError:
            pass
        else:
            raise AssertionError("a horizontal gripper has no jaw azimuth")
    note(f"36000 random (pan, azimuth, pitch): azimuth error {worst_down:.4f} deg at pitch 90, {worst_other:.4f} deg "
         f"at other pitches; predicted jaw tilt within {worst_tilt:.4f} deg")


def test_roll_for_azimuth_with_ik():
    """The documented recipe: guess pan from the target heading, then repeat ik -> pan -> roll."""
    rng = np.random.default_rng(19)
    errs = {0: 0.0, 1: 0.0, 2: 0.0}
    n = 0
    while n < 800:
        xyz = (rng.uniform(0.12, 0.30), rng.uniform(-0.18, 0.18), rng.uniform(-0.02, 0.06))
        psi = rng.uniform(-180.0, 180.0)
        pitch = 90.0 if n % 2 else rng.uniform(60.0, 100.0)
        pan = -math.degrees(math.atan2(xyz[1], xyz[0] - K._X_PAN))
        rounds = []
        for it in range(3):
            roll = K.roll_for_azimuth(psi, pan, pitch)
            q, info = K.ik(xyz, pitch, math.radians(roll), tool=TIP, limits=WIDE)
            if not info["ok"]:
                break
            rounds.append(ang_diff_deg(K.jaw_azimuth_deg(q), psi))
            pan = math.degrees(q[0])
        if len(rounds) < 3:
            continue
        n += 1
        for it in range(3):
            errs[it] = max(errs[it], rounds[it])
    assert errs[0] < 6.0 and errs[1] < 0.3 and errs[2] < 0.02, errs
    # consistency with the research's top-down grasp pose (gripper_model.topdown_gripper_frame_pose)
    for psi, centre in ((7.1, (254.1, -37.3, -5.5)), (-83.2, (278.3, -66.3, -14.6)), (160.0, (200.0, 60.0, -8.5))):
        pose = np.array(GM.topdown_gripper_frame_pose(centre, 22.0, 18.0, psi, e=14.0, standoff=7.0))
        xyz = pose[:3, 3] / 1000.0
        pan = -math.degrees(math.atan2(xyz[1], xyz[0] - K._X_PAN))
        for _ in range(4):
            roll = K.roll_for_azimuth(psi, pan)
            q, info = K.ik(xyz, 90.0, math.radians(roll), limits=WIDE)
            assert info["ok"], info
            pan = math.degrees(q[0])
        assert np.allclose(K.fk(q)[:3, :3], pose[:3, :3], atol=2e-4), (psi, K.fk(q)[:3, :3], pose[:3, :3])
    note(f"{n} fingertip targets: worst azimuth error {errs[0]:.2f} deg with pan guessed from the target heading, "
         f"{errs[1]:.3f} deg after one repetition, {errs[2]:.4f} deg after two")


def test_jaw_points_vs_gripper_model_and_tool_frame():
    rng = np.random.default_rng(23)
    names = {"fixed_nose", "moving_nose", "moving_pad", "pivot", "palm"}
    worst = 0.0
    for _ in range(2000):
        q = rng.uniform(LO, HI)
        jaw_deg = float(rng.uniform(-11.41, 100.0))
        pts = K.jaw_points(q, jaw_deg)
        assert set(pts) == names
        t = K.fk(q)
        inv = np.linalg.inv(t)
        local = {k: (inv[:3, :3] @ v + inv[:3, 3]) * 1000.0 for k, v in pts.items()}  # tool frame, mm
        for v in pts.values():
            assert isinstance(v, np.ndarray) and v.shape == (3,) and v.dtype == np.float64
        nose = GM.nose_points(jaw_deg)
        pad = GM.jaw_point(jaw_deg, GM.MOVING_PAD_CENTER_Q0)
        expect = {"fixed_nose": nose["fixed"], "moving_nose": nose["moving"],
                  "moving_pad": (pad[0], GM.MOVING_FINGER_Y, pad[1]),
                  "pivot": (GM.PIVOT_X, GM.MOVING_FINGER_Y, GM.PIVOT_Z),
                  "palm": (GM.ROLL_AXIS_X, GM.ROLL_AXIS_Y, GM.PALM_Z)}
        for k in names:
            worst = max(worst, float(np.abs(local[k] - np.array(expect[k])).max()))
        # rigid-body facts
        assert abs(np.linalg.norm(pts["moving_nose"] - pts["pivot"]) * 1000.0 - math.hypot(
            GM.MOVING_NOSE_Q0[0] - GM.PIVOT_X, GM.MOVING_NOSE_Q0[1] - GM.PIVOT_Z)) < 1e-9
        assert abs((pts["fixed_nose"] - t[:3, 3]) @ K.approach_axis(t) * 1000.0 - GM.FIXED_TIP_Z) < 1e-9
        assert (pts["moving_pad"] - t[:3, 3]) @ K.jaw_open_axis(t) > 0.0  # the moving jaw is on the +jaw_open side
        # with the harness tip tool: one more point, the others unchanged
        with_tip = K.jaw_points(q, jaw_deg, TIP)
        assert set(with_tip) == names | {"tip"}
        assert np.array_equal(with_tip["tip"], K.fk(q, TIP)[:3, 3])
        for k in names:
            assert np.array_equal(with_tip[k], pts[k])
    assert worst < 1e-9, worst
    # tool-frame values at q = 0: pads parallel 15.8 mm apart, noses 6.298 / 7.273 mm beyond the origin
    q = K.ik((0.22, 0.0, 0.05), 90.0, 0.0, limits=REAL)[0]
    pts = K.jaw_points(q, 0.0, TIP)
    t = K.fk(q)
    assert abs((pts["moving_pad"] - t[:3, 3]) @ K.jaw_open_axis(t) - 0.0158) < 1e-9
    assert abs(t[2, 3] - pts["fixed_nose"][2] - 0.006298) < 1e-6   # vertical gripper: fixed nose 6.3 mm lower
    assert abs(t[2, 3] - pts["moving_nose"][2] - 0.007273) < 1e-6
    assert abs(t[2, 3] - pts["tip"][2] - 0.0068) < 1e-6
    assert abs(np.linalg.norm(pts["tip"] - pts["fixed_nose"]) - math.hypot(3.06, 0.502) / 1000.0) < 1e-9
    assert abs(pts["palm"][2] - t[2, 3] - 0.06108) < 1e-6          # palm 61 mm above the tool origin
    note(f"2000 random poses / jaw angles: base-frame points map back to the gripper_model values within {worst:.1e} mm")


def test_jaw_points_vs_urdf_gripper_joint():
    """moving_nose / moving_pad / pivot through the URDF chain including the `gripper` joint
    (independent of kinematics.fk and of gripper_model.jaw_point's rotation formula)."""
    m = model()
    gf0 = m.link_pose("gripper_frame_link", {})
    to_jaw0 = np.linalg.inv(m.link_pose("moving_jaw_so101_v1_link", {})) @ gf0  # tool frame -> jaw link at q = 0
    nose0 = np.array([GM.MOVING_NOSE_Q0[0], GM.MOVING_FINGER_Y, GM.MOVING_NOSE_Q0[1], 1000.0]) / 1000.0
    pad0 = np.array([GM.MOVING_PAD_CENTER_Q0[0], GM.MOVING_FINGER_Y, GM.MOVING_PAD_CENTER_Q0[1], 1000.0]) / 1000.0
    nose_link, pad_link = to_jaw0 @ nose0, to_jaw0 @ pad0
    rng = np.random.default_rng(29)
    worst = worst_axis = 0.0
    for _ in range(1500):
        q = rng.uniform(LO, HI)
        jaw_deg = float(rng.uniform(-11.41, 110.0))
        ang = m.arm_angles(q)
        ang["gripper"] = math.radians(jaw_deg)
        t_jaw = m.link_pose("moving_jaw_so101_v1_link", ang)
        pts = K.jaw_points(q, jaw_deg)
        worst = max(worst, float(np.abs((t_jaw @ nose_link)[:3] - pts["moving_nose"]).max()),
                    float(np.abs((t_jaw @ pad_link)[:3] - pts["moving_pad"]).max()))
        d = pts["pivot"] - t_jaw[:3, 3]
        worst_axis = max(worst_axis, float(np.linalg.norm(d - (d @ t_jaw[:3, 2]) * t_jaw[:3, 2])))
        t_gf = m.tool_pose(q)
        for name, local in (("fixed_nose", GM.FIXED_NOSE), ("palm", (GM.ROLL_AXIS_X, GM.ROLL_AXIS_Y, GM.PALM_Z))):
            worst = max(worst, float(np.abs((t_gf @ np.array([*local, 1000.0]) / 1000.0)[:3] - pts[name]).max()))
    # (the URDF prints pi/2 as 1.5708, so its jaw axis is 4e-6 rad off the tool's y axis: 0.4 um at the nose)
    assert worst < 1e-6 and worst_axis < 2e-7, (worst, worst_axis)
    note(f"1500 random poses: moving_nose / moving_pad / fixed_nose / palm within {worst * 1e6:.3f} um of the URDF "
         f"chain, pivot {worst_axis * 1e6:.3f} um from the URDF gripper joint axis")


def test_jaw_points_show_the_moving_jaw_below_the_fingertip_ki2():
    """Review finding KI-2: with a non-vertical approach and |roll| > 90 deg the open moving jaw hangs
    below the fingertip. Expected values = lowest point of the moving jaw relative to the fingertip
    as measured by the review on a thinned-out mesh (tip at (0.22, 0, 0.03), jaw opened 20 deg)."""
    review_mm = {(80.0, -131.0): -4.8, (80.0, -150.0): -6.3, (70.0, -131.0): -10.0, (70.0, -150.0): -12.3,
                 (70.0, 150.0): -13.0, (60.0, -131.0): -15.1, (60.0, -150.0): -18.5, (60.0, 150.0): -19.6}
    worst = 0.0
    for (pitch, roll_deg), lowest in review_mm.items():
        q, info = K.ik((0.22, 0.0, 0.03), pitch, math.radians(roll_deg), tool=TIP, limits=REAL)
        assert info["ok"], (pitch, roll_deg, info)
        pts = K.jaw_points(q, 20.0, TIP)
        rel = (pts["moving_nose"][2] - pts["tip"][2]) * 1000.0
        # the nose point is the centre of a rounded end, at most 3 mm above the lowest point of the jaw; the
        # review kept every 7th mesh vertex, so its figure can be up to 1 mm above the true lowest point
        assert -1.0 <= rel - lowest <= 3.0, (pitch, roll_deg, rel, lowest)
        worst = max(worst, rel - lowest)
        assert K.jaw_tilt_deg(q) < 0.0
    for roll_deg in (-150.0, -90.0, 0.0, 49.0, 91.0, 150.0):  # straight down: the fingertip is the lowest point
        q, info = K.ik((0.22, 0.0, 0.03), 90.0, math.radians(roll_deg), tool=TIP, limits=REAL)
        assert info["ok"]
        pts = K.jaw_points(q, 20.0, TIP)
        assert 0.0 < (pts["moving_nose"][2] - pts["tip"][2]) * 1000.0 < 2.5
        assert abs(pts["fixed_nose"][2] - pts["tip"][2]) * 1000.0 < 0.6
    for pitch in (80.0, 70.0, 60.0):  # |roll| < 90: the moving jaw is the higher one
        q, info = K.ik((0.22, 0.0, 0.03), pitch, 0.0, tool=TIP, limits=REAL)
        assert info["ok"] and K.jaw_tilt_deg(q) > 0.0
        pts = K.jaw_points(q, 20.0, TIP)
        assert pts["moving_nose"][2] > pts["tip"][2] + 0.004
    note(f"8 review cases (moving jaw 5..20 mm below the fingertip): moving_nose z agrees within "
         f"-1.0..+{worst:.1f} mm")


def test_jaw_points_vs_meshes():
    """How far below the nose points the lowest material point of each jaw can be (URDF meshes)."""
    jaw0 = np.unique(gripper_mesh_gf_mm("moving_jaw_so101_v1_link").reshape(-1, 3).round(4), axis=0)
    fixed = np.unique(gripper_mesh_gf_mm("gripper_link", "follower").reshape(-1, 3).round(4), axis=0)
    finger = fixed[fixed[:, 2] > -55.0]
    rng = np.random.default_rng(31)
    bounds = {}
    for band, (lo, hi), max_moving, max_fixed in (("pitch 80..100", (80.0, 100.0), 1.8, 0.4),
                                                    ("pitch 60..120", (60.0, 120.0), 2.7, 1.0)):
        worst_m = worst_f = 0.0
        for _ in range(400):
            pitch, roll, pan = rng.uniform(lo, hi), rng.uniform(-160.0, 160.0), rng.uniform(-90.0, 90.0)
            jaw_deg = float(rng.uniform(-11.41, 60.0))
            lift, elbow = rng.uniform(-0.5, 1.0), rng.uniform(-0.5, 1.2)
            q = np.array([math.radians(pan), lift, elbow, math.radians(pitch) - lift - elbow, math.radians(roll)])
            t = K.fk(q)
            pts = K.jaw_points(q, jaw_deg)
            c, s = math.cos(math.radians(jaw_deg)), math.sin(math.radians(jaw_deg))
            turned = jaw0.copy()
            turned[:, 0] = GM.PIVOT_X + c * (jaw0[:, 0] - GM.PIVOT_X) - s * (jaw0[:, 2] - GM.PIVOT_Z)
            turned[:, 2] = GM.PIVOT_Z + s * (jaw0[:, 0] - GM.PIVOT_X) + c * (jaw0[:, 2] - GM.PIVOT_Z)
            jaw_base = (turned / 1000.0) @ t[:3, :3].T + t[:3, 3]
            k = int(np.argmin(jaw_base[:, 2]))
            assert np.linalg.norm(jaw_base[k] - pts["moving_nose"]) < 0.012  # the lowest point is at the nose
            worst_m = max(worst_m, (pts["moving_nose"][2] - jaw_base[k, 2]) * 1000.0)
            finger_base = (finger / 1000.0) @ t[:3, :3].T + t[:3, 3]
            k = int(np.argmin(finger_base[:, 2]))
            assert np.linalg.norm(finger_base[k] - pts["fixed_nose"]) < 0.012
            worst_f = max(worst_f, (pts["fixed_nose"][2] - finger_base[k, 2]) * 1000.0)
        assert worst_m <= max_moving and worst_f <= max_fixed, (band, worst_m, worst_f)
        bounds[band] = (round(float(worst_m), 2), round(float(worst_f), 2))
    note("lowest mesh point below the nose point, mm (moving, fixed), jaw angle -11.4..60: " + str(bounds))


# ============================================================================ gripper_model
def test_gripper_model_selftest_script():
    """gripper_model.py run as a script: its self-test (research numbers + the v2 additions) passes."""
    res = subprocess.run([sys.executable, GM.__file__], capture_output=True, text=True, timeout=120)
    assert res.returncode == 0, res.stderr[-2000:]
    assert "so101_gripper_model self-test OK" in res.stdout, res.stdout[-500:]


def test_gripper_model_new_api():
    # nose points
    n = GM.nose_points(0.0)
    assert set(n) == {"fixed", "moving"} and len(n["fixed"]) == 3 and len(n["moving"]) == 3
    assert n["fixed"] == (2.06, 0.0, 6.298) and abs(n["moving"][0] + 17.864) < 1e-9 and abs(n["moving"][2] - 7.273) < 1e-9
    for q in (-11.41, 0.0, 20.0, 45.0, 90.0):
        nq = GM.nose_points(q)
        assert nq["fixed"] == n["fixed"]
        x, z = GM.jaw_point(q, GM.MOVING_NOSE_Q0)
        assert nq["moving"] == (x, GM.MOVING_FINGER_Y, z)
        if q <= 64.0:  # the exact most distal z is never below the nose point and at most 1.6 mm beyond it
            assert -0.001 <= GM.moving_tip_depth(q) - nq["moving"][2] <= 1.6
    assert GM.nose_points(30.0)["moving"][0] < GM.nose_points(0.0)["moving"][0]  # +q opens toward -x
    # envelopes
    for z in (-70.0, -55.0, -30.0, 0.0, 5.0, 6.3):
        assert GM.finger_back_x(z) == GM.fixed_back_face_x(z)
    assert GM.finger_back_x(0.0) == 7.47 and GM.finger_back_x(-80.0) == 27.30
    assert all(GM.finger_back_x(z) >= GM.finger_back_x(z + 1.0) for z in range(-70, 6))  # monotonic envelope
    assert abs(GM.moving_outer_x(0.0, -1e9) + 38.10) < 0.05
    assert GM.moving_outer_x(0.0, 7.5) is None and GM.moving_outer_x(60.0, 0.0) is None
    for q in (0.0, 10.0, 20.0, 30.0, 45.0, 60.0, 80.0, 100.0):
        assert abs(-GM.moving_outer_x(q, -1e9) - GM.jaw_reach_minus_x(q)) < 0.6, q
        prev, ended = None, False
        for z in np.arange(-90.0, 9.0, 0.5):  # an envelope: it can only shrink as the depth z increases
            x = GM.moving_outer_x(q, float(z))
            if x is None:
                ended = True
                continue
            assert not ended and (prev is None or x >= prev - 1e-9), (q, z)
            prev = x
        assert prev is not None and ended
        outline = GM.moving_outline(q)
        assert len(outline) == len(GM.OUTLINE_Q0) and abs(min(p[0] for p in outline) - GM.moving_outer_x(q, -1e9)) < 1e-9
        assert GM.moving_outer_x(q, max(p[1] for p in outline) + 0.01) is None
    # width <-> angle
    assert abs(GM.q_for_width(0.0) - GM.Q_CLOSED) < 0.01 and abs(GM.q_for_width(15.8)) < 1e-6
    for w in (16.0, 20.4, 25.0, 30.0, 32.0, 45.0, 64.0):
        assert abs(GM.q_for_width(w) - GM.q_contact_pad_corner(w)) < 1e-3
    for w in (25.0, 28.0, 30.0):  # mesh-exact contact angles of the research for 10..15 mm engagement
        for e in (10.0, 12.0, 15.0):
            assert abs(GM.contact_angle(w, GM.FIXED_TIP_Z - e) - GM.q_for_width(w)) < 0.01, (w, e)
    for w in np.arange(0.0, 100.01, 2.5):
        assert abs(GM.width_for_q(GM.q_for_width(float(w))) - w) < 1e-9
    qs = [GM.q_for_width(float(w)) for w in np.arange(0.0, 100.01, 0.5)]
    assert all(b > a for a, b in zip(qs, qs[1:]))  # wider object -> jaw stops further open
    assert GM.width_for_q(GM.Q_CLOSED - 2.0) == 0.0 and abs(GM.width_for_q(0.0) - 15.8) < 1e-9
    for bad in (-1.0, 100.5, float("nan"), float("inf")):
        try:
            GM.q_for_width(bad)
        except ValueError:
            continue
        raise AssertionError(f"q_for_width({bad}) must raise")
    # raw ticks <-> q: the formula of DESIGN.md section 3
    for raw in (1998, 2050, 2194, 2263, 2819, 3648):
        assert abs(GM.q_from_raw(raw) - (-11.41 + (raw - 2050) * 360.0 / 4096.0)) < 1e-12
        back = GM.raw_from_q(GM.q_from_raw(raw))
        assert back == raw and isinstance(back, int)
        assert GM.raw_from_q(GM.q_from_raw(raw, 2061), touch_raw=2061) == raw
        assert abs(GM.q_from_raw(raw, touch_raw=2061) - GM.q_from_raw(raw) + 11 * 360.0 / 4096.0) < 1e-12
    assert GM.q_from_raw(2050) == GM.Q_CLOSED and GM.raw_from_q(0.0) == 2180 and GM.raw_from_q(GM.Q_CLOSED) == 2050
    assert abs(GM.q_from_raw(GM.raw_from_q(33.3)) - 33.3) <= 0.044


def test_gripper_model_vs_meshes():
    """nose_points, finger_back_x, moving_outer_x, width_for_q against the URDF meshes."""
    fixed = gripper_mesh_gf_mm("gripper_link", "follower")
    fv = fixed.reshape(-1, 3)
    # fixed jaw: pad plane x = 0, nose 6.298 mm beyond the origin, nose point in the middle of the end
    assert abs(fv[:, 2].max() - GM.FIXED_TIP_Z) < 1e-3
    cap = slab_points(fixed, 2, GM.FIXED_TIP_Z - 0.05, 99.0)
    assert abs(0.5 * (cap[:, 0].min() + cap[:, 0].max()) - GM.FIXED_NOSE[0]) < 0.05
    assert abs(0.5 * (cap[:, 1].min() + cap[:, 1].max()) - GM.FIXED_NOSE[1]) < 0.05
    pad = slab_points(fixed, 2, -3.6, 5.3)
    assert abs(pad[np.abs(pad[:, 1]) < 4.0][:, 0].min()) < 1e-3  # the tool origin lies on the fixed pad
    worst_back = [1e9, -1e9]
    for z in np.arange(-70.0, 6.2, 0.5):
        env = slab_points(fixed, 2, float(z), 99.0)[:, 0].max()  # largest x of the finger from z to the tip
        d = GM.finger_back_x(float(z)) - env
        worst_back = [min(worst_back[0], d), max(worst_back[1], d)]
    assert -1e-3 <= worst_back[0] and worst_back[1] < 2.0, worst_back  # never inside the material
    # moving jaw
    worst_nose = worst_tip = worst_env = worst_env90 = worst_w = 0.0
    for q in (-11.41, -5.0, 0.0, 8.0, 11.2, 20.0, 30.0, 40.0, 50.0, 60.0, 75.0, 90.0, 110.0):
        jaw = gripper_mesh_gf_mm("moving_jaw_so101_v1_link", jaw_deg=q)
        jv = jaw.reshape(-1, 3)
        if q <= 64.0:
            k = int(np.argmax(jv[:, 2]))
            worst_tip = max(worst_tip, abs(jv[k, 2] - GM.moving_tip_depth(q)))
            nose = GM.nose_points(q)["moving"]
            worst_nose = max(worst_nose, math.hypot(jv[k, 0] - nose[0], jv[k, 2] - nose[2]))
        for z in np.arange(-95.0, 12.0, 1.0):
            got = GM.moving_outer_x(q, float(z))
            pts = slab_points(jaw, 2, float(z), 1e9)
            if got is None or not len(pts):
                assert got is None or GM.moving_outer_x(q, float(z) + 0.2) is None, (q, z)   # reach differs < 0.2 mm
                assert not len(pts) or not len(slab_points(jaw, 2, float(z) + 0.2, 1e9)), (q, z)
                continue
            d = abs(got - pts[:, 0].min())
            if q <= 90.0:
                worst_env90 = max(worst_env90, d)
            worst_env = max(worst_env, d)
        if -11.0 < q <= 23.0:  # free width between the tip pads = distance of the moving pad zone from x = 0
            zone = slab_points(jaw, 2, -3.70, 6.298)
            worst_w = max(worst_w, abs(-zone[:, 0].max() - GM.width_for_q(q)))
    assert worst_tip < 2e-3 and worst_nose < 2.2, (worst_tip, worst_nose)
    assert worst_env90 < 0.25 and worst_env < 0.6, (worst_env90, worst_env)
    assert worst_w < 0.05, worst_w
    note(f"fixed back face: finger_back_x is {worst_back[0]:.2f}..{worst_back[1]:.2f} mm outside the mesh; "
         f"moving_outer_x within {worst_env90:.2f} mm of the mesh for q <= 90 ({worst_env:.2f} mm up to 110); "
         f"nose point within {worst_nose:.2f} mm of the most distal mesh vertex; width_for_q within {worst_w:.3f} mm "
         f"of the mesh gap")


def test_gripper_model_vs_real_arm_stalls():
    """The grasps logged on the real arm (runs/20261110_230151/events.jsonl, events 96, 125, 151):
    tips touch at raw 2050; the jaws stalled at raw 2194 on blocks the camera measured as 20.4 mm
    and 22.9 mm wide, and at 2263 on a 30 mm bottle cap."""
    measured = ((20.4, 2194), (22.9, 2194), (30.0, 2263))
    rigid = [GM.TOUCH_RAW + (GM.q_for_width(w) - GM.Q_CLOSED) / GM.DEG_PER_TICK for w, _ in measured]
    assert [round(r) for r in rigid] == [2221, 2244, 2307]          # what fd.py logged as expected_contact_raw
    gaps = [stall - r for (w, stall), r in zip(measured, rigid)]     # the jaws stall this far past rigid contact
    assert all(-52.0 < g < -25.0 for g in gaps), gaps
    assert abs(-sum(gaps) / 3.0 - GM.STALL_COMPLIANCE_TICKS) < 2.0
    model_width_at_stall = [GM.width_for_q(GM.q_from_raw(stall)) for _, stall in measured]
    assert all(2.5 < w - m < 6.0 for (w, _), m in zip(measured, model_width_at_stall))
    residuals = [stall - GM.stall_raw_for_width(w) for w, stall in measured]
    assert max(abs(r) for r in residuals) <= 13, residuals
    assert abs(GM.width_from_stall_raw(2194) - 21.82) < 0.02 and abs(GM.width_from_stall_raw(2263) - 29.54) < 0.02
    assert 20.4 - 1.5 < GM.width_from_stall_raw(2194) < 22.9 + 1.5 and abs(GM.width_from_stall_raw(2263) - 30.0) < 1.5
    # first contact (load still near zero in the logged histories): raw about 2221 and 2300..2310
    assert abs(GM.width_for_q(GM.q_from_raw(2221)) - 20.4) < 0.1
    assert 29.0 < GM.width_for_q(GM.q_from_raw(2300)) < GM.width_for_q(GM.q_from_raw(2310)) < 30.5
    # helper properties
    assert GM.stall_raw_for_width(0.5) == GM.TOUCH_RAW and isinstance(GM.stall_raw_for_width(20.0), int)
    assert GM.stall_raw_for_width(20.0, compliance_ticks=0) == GM.raw_from_q(GM.q_for_width(20.0))
    assert GM.stall_raw_for_width(20.0, touch_raw=2060) == GM.stall_raw_for_width(20.0) + 10
    for w in (12.0, 20.4, 30.0, 45.0):
        assert abs(GM.width_from_stall_raw(GM.stall_raw_for_width(w)) - w) < 0.07
    note("rigid contact raw " + str([round(r, 1) for r in rigid]) + ", measured stalls [2194, 2194, 2263]: the jaws "
         "stall " + str([round(-g, 1) for g in gaps]) + " ticks = " + str([round(-g * GM.DEG_PER_TICK, 2) for g in gaps])
         + " deg past rigid contact, i.e. the rigid model reads " + str([round(w - m, 1) for (w, _), m in
                                                                          zip(measured, model_width_at_stall)])
         + " mm too narrow at the stall; with STALL_COMPLIANCE_TICKS = 40 the prediction errors are "
         + str(residuals) + " ticks")


# ============================================================================ timing
def test_timing():
    def per_call(fn, n):
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            best = min(best, (time.perf_counter() - t0) / n)
        return best * 1e6  # microseconds

    q = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    old = v1()
    hover = (0.20, 0.0, 0.12)
    t = {
        "fk": per_call(lambda: K.fk(q, TIP), 1000),
        "ik": per_call(lambda: K.ik((0.2, 0.05, 0.03), 90.0, 0.3, tool=TIP, limits=REAL), 1000),
        "ik v1": per_call(lambda: old.ik((0.2, 0.05, 0.03), 90.0, 0.3, tool=TIP, limits=REAL), 1000),
        "best_pitch (wanted pitch feasible)": per_call(
            lambda: K.best_pitch((0.2, 0.05, 0.03), 90.0, 0.3, tool=TIP, limits=REAL), 500),
        "best_pitch (degraded)": per_call(lambda: K.best_pitch(hover, 90.0, 0.0, tool=TIP, limits=REAL), 200),
        "best_pitch (out of reach)": per_call(lambda: K.best_pitch((0.6, 0, 0), 90.0, 0.0, tool=TIP, limits=REAL), 200),
        "feasible_pitches": per_call(lambda: K.feasible_pitches(hover, 0.0, tool=TIP, limits=REAL), 200),
        "pitch_range": per_call(lambda: K.pitch_range(hover, 0.0, tool=TIP, limits=REAL), 200),
        "pitch_range v1": per_call(lambda: old.pitch_range(hover, 0.0, tool=TIP, limits=REAL), 20),
        "jaw_points": per_call(lambda: K.jaw_points(q, 20.0, TIP), 1000),
        "jaw_tilt_deg": per_call(lambda: K.jaw_tilt_deg(q), 1000),
        "roll_for_azimuth": per_call(lambda: K.roll_for_azimuth(30.0, 10.0, 80.0), 5000),
        "gripper_model.moving_outer_x": per_call(lambda: GM.moving_outer_x(30.0, -10.0), 1000),
    }
    for name, us in t.items():
        note(f"{name:36s} {us:8.1f} us")
    # generous bounds (the Jetson is shared tonight); typical values are 3 to 5 times lower
    assert t["ik"] < 1000.0, t["ik"]
    assert t["ik"] < 1.5 * t["ik v1"] + 30.0, (t["ik"], t["ik v1"])    # the wrap handling costs next to nothing
    assert t["fk"] < 200.0
    assert t["best_pitch (degraded)"] < 3000.0 and t["feasible_pitches"] < 3000.0
    assert t["pitch_range"] < 0.25 * t["pitch_range v1"]               # the scan is at least 4 times faster than v1
    assert t["jaw_points"] < 400.0 and t["roll_for_azimuth"] < 20.0


# ============================================================================ runner
if __name__ == "__main__":
    import traceback

    only = sys.argv[1:]
    tests = [(n, f) for n, f in globals().items() if n.startswith("test_") and callable(f)]
    passed = failed = skipped = 0
    t_all = time.time()
    for name, fn in tests:
        if only and not any(o in name for o in only):
            continue
        NOTES.clear()
        t0 = time.time()
        try:
            fn()
        except Skip as e:
            skipped += 1
            print(f"SKIP {name}: {e}")
            continue
        except Exception:
            failed += 1
            print(f"FAIL {name}")
            traceback.print_exc()
            continue
        passed += 1
        print(f"PASS {name} ({time.time() - t0:.1f} s)")
        for line in NOTES:
            print("       " + line)
    print(f"tests: {passed + failed + skipped}  passed: {passed}  failures: {failed}  skipped: {skipped}  "
          f"({time.time() - t_all:.0f} s)")
    sys.exit(1 if failed else 0)
