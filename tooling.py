"""A tool taped to the gripper (a pen, a marker, a stylus): what is attached, where its tip is, and what
that changes for safety.

Why this exists (2026-11-11): a pen was taped to the gripper and the harness knew nothing of it. `fd park`
opens the gripper to 10 % and folds the arm so that the FINGERTIP ends 28 mm above the table: with a 6 cm
tool the tip would have been driven 18 mm into the table. `fd ready` put the fingertip, not the tool tip,
at the stand-by point. That night's scripts got by with a second config file (FD_CONFIG trick) and a list
of commands "never to use". Here the tool is state of the harness itself:

- ONE file says what is on the gripper: tool_active.json next to config.json (TOOL_FILE). No file = no
  tool = exactly the old behaviour. $FD_TOOL overrides the path (the tests use it, so that no test ever
  writes the real file). A file that cannot be read is an error, never "no tool": the harness must not
  unlock the gripper because a file is damaged.
- `robot.Arm` reads it when it is built (`arm.tool`); `Arm.tip_tool()` is then the TOOL tip, so the arm
  plans, checks (floor, keep-outs) and moves the tool tip. The jaw and wrist lines of the floor check
  stay what they were (arm.cfg["tip_offset"] remains the fingertip).
- With a tool attached the gripper never receives a goal (robot.Arm refuses every gripper method and
  every gripper goal), `park` is refused (skills.park), `ready` goes to the tool stand-by pose
  (go_ready here), torque off is only possible in the tool's rest pose, and `fd tool rest` (rest() here)
  is the way to fold the arm away: the pose that worked on the real arm that night, generalised.
- A free move with a tool (go_ready, rest) never brings the tool tip lower than MIN_FREE_HEIGHT above
  the table: a tool that is low is first lifted straight up, the way is checked every 1.5 deg before
  anything moves, and the measured height is watched while it moves.

Simulation: `fd --sim` never writes the real tool file. A tool attached with `fd --sim tool attach` lives
in SIM_TOOL_FILE and is what the simulated arm carries until `fd --sim tool detach`; without it the
simulation mirrors the real tool (read only), so a real action can be rehearsed with the real tool's data.

Units: metres, base frame; angles in degrees in files and results, radians for joint vectors.
Tests: test_tooling.py. Nothing here opens the servo bus or the camera by itself.
"""

from __future__ import annotations

import json
import math
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np

import kinematics as K
from robot import ARM, CONFIG_PATH, MotionAborted, SafetyError

TOOL_FILE = os.path.join(os.path.dirname(CONFIG_PATH), "tool_active.json")
SIM_TOOL_FILE = os.path.join(os.path.dirname(CONFIG_PATH), "tool_active.sim.json")
MOUNTS = ("along_fingers", "across_fingers", "other")
# gripper pitch (deg below horizontal) a tool is worked with unless `fd tool attach --pitch` says otherwise.
# along the fingers: 75 is what drew the cat (the pen leans 15 deg, tip away from the robot);
# across the fingers (tip on the fixed-jaw side): the tool is upright with the gripper level (pitch 0), which
# the arm can hold almost nowhere on the table; 30 (the tool leans 30 deg, tip towards the robot) it can.
DEFAULT_PITCH = {"along_fingers": 75.0, "across_fingers": 30.0, "other": 75.0}
MIN_FREE_HEIGHT = 0.02    # a free move never brings the tool tip lower than this above the table
PLAN_MARGIN = 0.005       # ... and is planned with this much to spare (tracking error, encoder noise)
LIFT_HEIGHT = 0.035       # a tool found lower than this is first lifted straight up to it
REST_CLEARANCE = 0.05     # tool tip above the table in the rest pose (56 mm is what held on the real arm)
REST_PAN_DEG = 20.0
REST_TOL_DEG = 10.0       # "in the rest pose": lift / elbow / wrist_flex this close to it (as the old torque-off guard)
REST_MIN_HEIGHT = 0.03    # ... and the tool tip at least this far above the table
MAX_OFFSET_M = 0.30       # a tip further than this from the gripper is a typing error, not a tool
_FINGER_AXIS = np.asarray(K._APPROACH_LOCAL, dtype=float)       # along the fingers, in gripper_frame_link axes
_FIXED_JAW_AXIS = -np.asarray(K._JAW_OPEN_LOCAL, dtype=float)   # across them, towards the fixed jaw


# ----------------------------------------------------------------------------- the tool and its file
@dataclass
class Tool:
    """What is taped to the gripper. Everything position-like is in metres; `tip_offset` is the tool tip
    in gripper_frame_link axes (x: towards the fixed jaw, y: sideways, z: along the fingers)."""
    name: str
    tip_offset: list                 # tool tip in gripper_frame_link axes (m)
    mount: str = "other"             # "along_fingers" | "across_fingers" | "other"
    pitch_deg: float = 75.0          # gripper pitch used for working with it
    gripper_raw: int | None = None   # gripper position when it was attached (never commanded while attached)
    contact: dict | None = None      # {"x0","y0","a","bx","by","rms_mm","points"}: plan z of first contact
    corr_map: dict | None = None     # {"centre","kind","coef","rms_mm","max_mm","n"}: commanded -> actual
    region: list | None = None       # [x0, y0, x1, y1] where contact / map hold
    calibrated: dict = field(default_factory=dict)   # {"tip": {...}, "contact": {...}, "map": {...}} with "at"
    notes: list = field(default_factory=list)
    attached_at: str | None = None   # ISO time of `fd tool attach`
    extra: dict = field(default_factory=dict)        # keys of the file this version does not know (kept on save)

    _FIELDS = ("name", "tip_offset", "mount", "pitch_deg", "gripper_raw", "contact", "corr_map", "region",
               "calibrated", "notes", "attached_at")

    def to_dict(self) -> dict:
        d = dict(self.extra)
        d.update({k: getattr(self, k) for k in self._FIELDS})
        d["tip_offset"] = [round(float(v), 5) for v in self.tip_offset]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Tool":
        """Build a Tool from a file's content. Raises SafetyError for anything motion would depend on that
        is missing or not a sane number (the caller must not treat that as "no tool")."""
        if not isinstance(d, dict):
            raise SafetyError("tool description is not a JSON object")
        name = d.get("name")
        if not isinstance(name, str) or not name.strip():
            raise SafetyError("tool description has no name")
        tool = cls(name=name.strip(), tip_offset=check_offset(d.get("tip_offset")),
                   mount=d.get("mount") if d.get("mount") in MOUNTS else "other",
                   pitch_deg=check_pitch(d.get("pitch_deg", DEFAULT_PITCH["other"])))
        raw = d.get("gripper_raw")
        tool.gripper_raw = None if raw is None else int(raw)
        for key, kind in (("contact", dict), ("corr_map", dict), ("region", list)):
            v = d.get(key)
            if v is not None and not isinstance(v, kind):
                raise SafetyError(f"tool description: {key!r} must be {kind.__name__} or null")
            setattr(tool, key, v)
        tool.calibrated = dict(d.get("calibrated") or {})
        tool.notes = [str(n) for n in (d.get("notes") or [])]
        tool.attached_at = d.get("attached_at")
        tool.extra = {k: v for k, v in d.items() if k not in cls._FIELDS}
        return tool


def check_offset(v) -> list:
    """A tool tip offset as three finite numbers of metres (SafetyError otherwise)."""
    try:
        off = [float(x) for x in v]
    except (TypeError, ValueError):
        off = []
    if len(off) != 3 or not all(math.isfinite(x) for x in off):
        raise SafetyError(f"tool tip offset must be three numbers x,y,z in metres (got {v!r})")
    if math.sqrt(sum(x * x for x in off)) > MAX_OFFSET_M:
        raise SafetyError(f"tool tip offset {off} is more than {MAX_OFFSET_M * 100:.0f} cm from the gripper: "
                          "metres are expected (0.06 = 6 cm)")
    return off


def check_pitch(v) -> float:
    try:
        p = float(v)
    except (TypeError, ValueError):
        p = float("nan")
    if not (math.isfinite(p) and -90.0 <= p <= 120.0):
        raise SafetyError(f"tool working pitch must be a number of degrees in -90 .. 120 (got {v!r})")
    return p


def tool_path(sim: bool = False) -> str:
    """The tool file in force: $FD_TOOL when set (tests, experiments), else TOOL_FILE, or SIM_TOOL_FILE
    for what `fd --sim` writes (a simulated command never touches the real arm's tool state)."""
    env = os.environ.get("FD_TOOL")
    if env:
        return env
    return SIM_TOOL_FILE if sim else TOOL_FILE


def load(path: str) -> Tool:
    """The tool described by a file. Raises SafetyError when the file exists but cannot be used: a damaged
    file must stop the arm, not silently unlock the gripper."""
    try:
        with open(path) as f:
            return Tool.from_dict(json.load(f))
    except (OSError, ValueError, TypeError, KeyError, SafetyError) as e:
        raise SafetyError(f"the tool file {path} cannot be used ({e}): the harness does not know what is on the "
                          "gripper. Repair the file, or take the tool off by hand and run `fd tool detach`") from e


def active(sim: bool = False) -> Tool | None:
    """The tool that is attached, or None (no file = no tool). sim=True: the simulated tool when one was
    attached with `fd --sim tool attach`, else the real one (the simulation mirrors the real arm)."""
    for path in ([tool_path(True)] if sim else []) + [tool_path(False)]:
        if os.path.exists(path):
            return load(path)
    return None


def source(sim: bool = False) -> str | None:
    """Path of the file active(sim) reads the tool from (None: no tool)."""
    for path in ([tool_path(True)] if sim else []) + [tool_path(False)]:
        if os.path.exists(path):
            return path
    return None


def save(tool: Tool, sim: bool = False) -> None:
    """Write the tool file (atomically). An Arm that is already built keeps the tool it was built with:
    assign `arm.tool = tool` when it should follow."""
    try:
        Tool.from_dict(tool.to_dict())  # refuse to store what could not be read back
    except (TypeError, ValueError) as e:
        raise SafetyError(f"tool '{getattr(tool, 'name', '?')}' cannot be stored: {e}") from e
    path = tool_path(sim)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(tool.to_dict(), f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def clear(sim: bool = False) -> None:
    """Remove the tool file: nothing is attached any more (the gripper and `park` are unlocked)."""
    path = tool_path(sim)
    if os.path.exists(path):
        os.remove(path)


def config_with_tool(cfg: dict, tool: Tool | None) -> dict:
    """A deep copy of the configuration with "tip_offset" replaced by the tool tip (the FD_CONFIG trick of
    2026-11-11, for code that takes a config instead of an Arm). tool None: a plain deep copy."""
    out = json.loads(json.dumps(cfg))
    if tool is not None:
        out["tip_offset"] = [float(v) for v in tool.tip_offset]
        out["tool_name"] = tool.name
    return out


# ----------------------------------------------------------------------------- geometry of a mount
def offset_for(cfg: dict, mount: str, length_m: float) -> list:
    """Tool tip offset of a straight tool that sticks out `length_m` beyond the fingertip: along the
    fingers, or across them towards the fixed jaw (the side that points down when the gripper is level
    at wrist roll 0)."""
    if mount not in ("along_fingers", "across_fingers"):
        raise SafetyError(f"mount {mount!r}: give the tip with --tip x,y,z")
    try:
        length = float(length_m)
    except (TypeError, ValueError):
        length = float("nan")
    if not (math.isfinite(length) and 0.0 < length <= 0.25):
        raise SafetyError(f"tool length must be in metres, more than 0 and at most 0.25 (got {length_m!r})")
    axis = _FINGER_AXIS if mount == "along_fingers" else _FIXED_JAW_AXIS
    return [round(float(v), 5) for v in np.asarray(cfg.get("tip_offset", (0.0, 0.0, 0.0)), dtype=float) + axis * length]


def _reach(cfg: dict, tip_offset) -> np.ndarray:
    """Fingertip -> tool tip, in gripper_frame_link axes."""
    return np.asarray(tip_offset, dtype=float) - np.asarray(cfg.get("tip_offset", (0.0, 0.0, 0.0)), dtype=float)


def infer_mount(cfg: dict, tip_offset) -> str:
    """How a tool with this tip is probably mounted: within 30 deg of the finger direction = along the
    fingers, within 30 deg of the jaw line = across them, anything else (or shorter than 5 mm) = other."""
    d = _reach(cfg, tip_offset)
    n = float(np.linalg.norm(d))
    if n < 0.005:
        return "other"
    if float(d @ _FINGER_AXIS) / n > math.cos(math.radians(30.0)):
        return "along_fingers"
    if abs(float(d @ _FIXED_JAW_AXIS)) / n > math.cos(math.radians(30.0)):
        return "across_fingers"
    return "other"


def tool_axis(cfg: dict, tool: Tool) -> np.ndarray:
    """Unit vector from the tool's body to its tip in gripper_frame_link axes. A tool taped along (across)
    the fingers is taken to be parallel to them (to the jaw line): the sideways part of its tip offset is
    where the tape holds it, not a tilt. Any other tool: the line from the fingertip to its tip."""
    d = _reach(cfg, tool.tip_offset)
    if tool.mount == "along_fingers":
        return _FINGER_AXIS.copy()
    if tool.mount == "across_fingers":
        return _FIXED_JAW_AXIS * (1.0 if float(d @ _FIXED_JAW_AXIS) >= 0.0 else -1.0)
    n = float(np.linalg.norm(d))
    return _FINGER_AXIS.copy() if n < 1e-6 else d / n


def length_of(cfg: dict, tool: Tool) -> float:
    """How far the tool tip lies from the fingertip (m)."""
    return float(np.linalg.norm(_reach(cfg, tool.tip_offset)))


def table_z(arm) -> float:
    return float(arm.cfg.get("table_z", 0.0))


def tip_height(arm, q, tool: Tool) -> float:
    """Height of the tool tip above the table (m) at joint pose q."""
    return float(K.fk(np.asarray(q, dtype=float), K.tool_offset(*tool.tip_offset))[2, 3]) - table_z(arm)


def body_points(arm, q, tool: Tool, n: int = 5) -> np.ndarray:
    """(n, 3) base-frame points on the straight line from the fingertip to the tool tip (both included):
    what stands in for the tool's body in the checks below."""
    t = K.fk(np.asarray(q, dtype=float))
    a = np.asarray(arm.cfg["tip_offset"], dtype=float)
    pts = a + np.linspace(0.0, 1.0, n)[:, None] * (np.asarray(tool.tip_offset, dtype=float) - a)
    return pts @ t[:3, :3].T + t[:3, 3]


@contextmanager
def _as_tool(arm, tool):
    """Run the arm's own planning and checks with `tool` on its gripper (what it had is put back)."""
    had, prev = hasattr(arm, "tool"), getattr(arm, "tool", None)
    arm.tool = tool
    try:
        yield arm
    finally:
        if had:
            arm.tool = prev
        else:
            del arm.tool


def _keepout_of(cfg: dict, p):
    for name, box in cfg.get("keepouts", {}).items():
        x0, x1, y0, y1, z0, z1 = box
        if x0 <= p[0] <= x1 and y0 <= p[1] <= y1 and z0 <= p[2] <= z1:
            return name
    return None


def check_pose(arm, tool: Tool, q, q_start=None, min_height: float | None = None, what: str = "tool pose") -> float:
    """Everything a pose must satisfy with this tool on the gripper; returns the tool tip height above the
    table. Raises SafetyError.

    - the arm's own check (Arm.check_q) with the tool tip as its tip: joint limits, floor, the moving jaw,
      the wrist and gripper-body lines, keep-outs;
    - the tool's body (fingertip .. tool tip): not below the floor limit, not inside a keep-out. The arm
      itself tests only ONE point against the keep-outs, and with a tool that point is the tool tip;
    - min_height: the tool tip at least this far above the table.
    q_start relaxes each of them only as far as the start pose already violates it (as Arm.check_q does)."""
    q = np.asarray(q, dtype=float)
    with _as_tool(arm, tool):
        arm.check_q(q, what=what, q_start=q_start)
    pts = body_points(arm, q, tool)
    pts0 = None if q_start is None else body_points(arm, q_start, tool)
    floor = float(arm.cfg["floor_z"]) + float(arm.cfg["floor_clearance"])
    for i, p in enumerate(pts):
        if p[2] < floor and not (pts0 is not None and p[2] >= pts0[i][2] - 1e-3):
            raise SafetyError(f"{what}: the tool's body would be at z={p[2]:.4f}, below the floor limit {floor:.4f}")
        name = _keepout_of(arm.cfg, p)
        if name and not (pts0 is not None and _keepout_of(arm.cfg, pts0[i]) == name):
            raise SafetyError(f"{what}: the tool ({'tip' if i == len(pts) - 1 else 'body'}) would be inside "
                              f"keep-out '{name}' at {np.round(p, 3).tolist()}")
    h = float(pts[-1][2]) - table_z(arm)
    if min_height is not None and h < min_height:
        raise SafetyError(f"{what}: the tool tip would be {h * 1000:.0f} mm above the table "
                          f"(limit {min_height * 1000:.0f} mm)")
    return h


# ----------------------------------------------------------------------------- poses
def rest_q(arm, tool: Tool, clearance_m: float = REST_CLEARANCE, pan_deg: float = REST_PAN_DEG) -> np.ndarray:
    """The folded pose in which the arm can be de-energised with this tool on: shoulder and elbow folded
    back as in the plain rest pose (they lie on their stops there), wrist roll 0, the base turned to
    pan_deg, and the wrist raised just far enough that the tool tip is clearance_m above the table.

    What held on the real arm (2026-11-11): pan 20, lift -100.2, elbow 85.2, wrist_flex 42, roll 0 with a
    62 mm tool along the fingers: tip 56 mm above the table, torque off, nothing moved. This function gives
    wrist_flex 44 for that tool (tip at 50 mm). Raises SafetyError when no wrist angle does it or the pose
    fails check_pose."""
    try:
        clearance, pan = float(clearance_m), float(pan_deg)
    except (TypeError, ValueError):
        clearance = pan = float("nan")
    if not (math.isfinite(clearance) and 0.03 <= clearance <= 0.15):
        raise SafetyError(f"rest clearance must be 0.03 .. 0.15 m above the table (got {clearance_m!r})")
    lim = arm.q_limits()
    if not (math.isfinite(pan) and lim[ARM[0]][0] <= math.radians(pan) <= lim[ARM[0]][1]):
        raise SafetyError(f"rest pan {pan_deg!r} deg is outside the pan limits")
    with _as_tool(arm, None):
        q = np.array(arm.rest_q(), dtype=float)  # the plain folded pose, clamped inside the joint limits
    q[0], q[4] = math.radians(pan), 0.0
    w_lo, w_hi = lim[ARM[3]][0] + 0.005, min(float(q[3]), lim[ARM[3]][1] - 0.005)

    def height(w):
        q[3] = w
        return tip_height(arm, q, tool)

    # from the plain rest pose's wrist angle towards "gripper raised", in 1 deg steps: the first angle
    # that gives the clearance is the pose nearest to the fold the arm is known to rest in
    step = math.radians(1.0)
    last = None
    w = w_hi
    if height(w) < clearance:
        while w - step >= w_lo and height(w - step) < clearance:
            w -= step
        if w - step < w_lo:
            raise SafetyError(f"tool '{tool.name}': no wrist angle of the folded pose puts its tip "
                              f"{clearance * 1000:.0f} mm above the table; the arm cannot rest with it. Support "
                              "the arm by hand, take the tool off and run `fd tool detach`")
        a, b = w - step, w  # height(a) >= clearance > height(b)
        for _ in range(30):
            m = 0.5 * (a + b)
            if height(m) >= clearance:
                a = m
            else:
                b = m
        w = a
    # the pose must also pass every check; if it does not, raise the wrist further (a higher tool tip)
    while w >= w_lo:
        q[3] = w
        try:
            check_pose(arm, tool, q, min_height=clearance - 1e-6, what="tool rest pose")
            return q.copy()
        except SafetyError as e:
            last = e
            w -= step
            if tip_height(arm, np.array([*q[:3], max(w, w_lo), q[4]]), tool) > clearance + 0.02:
                break
    raise SafetyError(f"tool '{tool.name}': the folded pose that keeps its tip {clearance * 1000:.0f} mm above the "
                      f"table is not safe ({last}); the arm cannot rest with it. Support the arm by hand, take the "
                      "tool off and run `fd tool detach`")


def ready_pose(arm, tool: Tool) -> dict:
    """The stand-by pose with a tool: {"q", "xyz", "pitch", "moved_m"}. The TOOL tip is at cfg["ready"]["xyz"]
    (the plain ready pose puts the fingertip there, and the tool below it), the gripper pitched as for
    working with the tool (tool.pitch_deg, or the nearest pitch the arm can hold there), the configured roll.

    A tool too long for that point (10 cm and more along the fingers: the arm cannot fold far enough back)
    stands by at the nearest point straight ahead of it, same height, in steps of 2 cm ("moved_m" says how
    far). Raises SafetyError when no such pose exists or it fails check_pose."""
    r = arm._c("ready")
    base, roll = np.array(r["xyz"], dtype=float), float(r["roll"])
    tl, lim = K.tool_offset(*tool.tip_offset), arm.q_limits()
    last = "out of reach"
    for k in range(7):
        xyz = base + np.array([0.02 * k, 0.0, 0.0])
        pitch, _ = K.best_pitch(xyz, want=float(tool.pitch_deg), wrist_roll=math.radians(roll), tool=tl, limits=lim,
                                lo=-60.0)
        if pitch is None:
            continue
        q, info = K.ik(xyz, pitch, math.radians(roll), tool=tl, limits=lim)
        if not info["ok"]:
            last = info["reason"]
            continue
        try:
            check_pose(arm, tool, q, min_height=MIN_FREE_HEIGHT + PLAN_MARGIN, what="tool stand-by pose")
        except SafetyError as e:
            last = str(e)
            continue
        return {"q": q, "xyz": [round(float(v), 4) for v in xyz], "pitch": round(float(pitch), 1),
                "moved_m": round(0.02 * k, 3)}
    raise SafetyError(f"tool '{tool.name}': the stand-by point {base.round(3).tolist()} (and up to 12 cm ahead of "
                      f"it) cannot be reached with its tip: {last}. Check the tip offset (`fd tool status`)")


def ready_q(arm, tool: Tool) -> np.ndarray:
    """Joint pose of the tool stand-by pose (see ready_pose)."""
    return ready_pose(arm, tool)["q"]


def work_q(arm, tool: Tool, xy, height_m: float = 0.0) -> tuple:
    """(q, pitch) of the working pose with the tool tip `height_m` above the table at xy: the gripper at
    tool.pitch_deg or the nearest pitch the arm can hold there, the configured stand-by roll. Kinematics
    only (no floor / keep-out check). Raises SafetyError when the place is out of reach."""
    roll = float(arm._c("ready")["roll"])
    p = np.array([float(xy[0]), float(xy[1]), table_z(arm) + float(height_m)])
    tl, lim = K.tool_offset(*tool.tip_offset), arm.q_limits()
    pitch, _ = K.best_pitch(p, want=float(tool.pitch_deg), wrist_roll=math.radians(roll), tool=tl, limits=lim,
                            lo=-60.0)
    if pitch is None:
        raise SafetyError(f"({p[0]:.3f}, {p[1]:.3f}) is out of reach for the tip of tool '{tool.name}'")
    q, info = K.ik(p, pitch, math.radians(roll), tool=tl, limits=lim)
    if not info["ok"]:
        raise SafetyError(f"({p[0]:.3f}, {p[1]:.3f}) is out of reach for the tip of tool '{tool.name}': {info['reason']}")
    return q, float(pitch)


def lean_xy(arm, tool: Tool, xy) -> np.ndarray:
    """Unit vector on the table: where the tool tip points to, seen from the tool's body, when the tool
    works at xy (tip on the table, working pose of work_q). A stroke that travels ALONG this vector pushes
    the tip ahead of the tool; one that travels against it pulls. Which of the two loses ink is NOT known
    (on 2026-11-11 the pushed lines lost less: measure it per tool, `fd tool calibrate --stage direction`).
    A tool that stands within 1 deg of upright has no lean: the direction away from the robot is
    returned then (what it would lean to with a slightly smaller pitch). Raises SafetyError out of reach."""
    q, _pitch = work_q(arm, tool, xy)
    axis = K.fk(q)[:3, :3] @ tool_axis(arm.cfg, tool)
    h = np.array([float(axis[0]), float(axis[1])])
    n = float(np.linalg.norm(h))
    if n < math.sin(math.radians(1.0)):
        h = np.array([float(xy[0]) - K._X_PAN, float(xy[1])])
        n = float(np.linalg.norm(h))
    return h / n


def contact_z(tool: Tool, xy) -> float | None:
    """Plan height (base frame z of the MODELLED tip) at which the tool first touches at xy, from the
    measured contact plane tool.contact; None when it has not been measured."""
    c = tool.contact
    if not c:
        return None
    return float(c["a"]) + float(c.get("bx", 0.0)) * (float(xy[0]) - float(c.get("x0", 0.0))) \
        + float(c.get("by", 0.0)) * (float(xy[1]) - float(c.get("y0", 0.0)))


def calibration_state(tool: Tool) -> dict:
    """Which of the three measurements the tool carries: {"tip", "contact", "map"} -> bool."""
    return {"tip": bool((tool.calibrated or {}).get("tip")), "contact": tool.contact is not None,
            "map": tool.corr_map is not None}


# ----------------------------------------------------------------------------- free moves with a tool
def lift_path(arm, tool: Tool, q, to_height: float = LIFT_HEIGHT) -> list:
    """Joint waypoints (starting with q) that raise the tool tip straight up to to_height above the table,
    keeping the gripper's pitch; where the arm cannot hold that pitch higher up, the gripper may tip back
    by up to 12 deg while the tip is drawn up to 2 cm towards the robot (as the pen scripts did). Every
    waypoint passes check_pose and none is lower than the one before. No motion. Raises SafetyError."""
    q = np.asarray(q, dtype=float)
    tl = K.tool_offset(*tool.tip_offset)
    p = K.fk(q, tl)[:3, 3].copy()
    pitch = K.pose_summary(q, tl)["pitch_down_deg"]
    roll = math.degrees(float(q[4]))
    out = np.array([p[0] - K._X_PAN, p[1]])
    out = out / max(float(np.linalg.norm(out)), 1e-6)
    z_top = table_z(arm) + float(to_height)
    last = None
    with _as_tool(arm, tool):
        for dp, back in ((0.0, 0.0), (-6.0, 0.01), (-12.0, 0.02), (6.0, 0.0), (12.0, 0.0)):
            top = np.array([p[0] - back * out[0], p[1] - back * out[1], z_top])
            try:
                wps = arm.plan_tip_path([(top, pitch + dp, roll)], q_start=q, step_m=0.003)
                h_prev = tip_height(arm, q, tool)
                for k, w in enumerate(wps[1:], 1):
                    h = check_pose(arm, tool, w, q_start=q, what=f"lift point {k}")
                    if h < h_prev - 5e-4:
                        raise SafetyError("the tool tip would go down on the way up")
                    h_prev = h
                return wps
            except SafetyError as e:
                last = e
    raise SafetyError(f"the tool tip is {tip_height(arm, q, tool) * 1000:.0f} mm above the table and cannot be "
                      f"lifted straight up from this pose ({last}). Nothing moved. Lift it by hand-driven steps "
                      "(`fd tip 0 0 0.01 --rel`) or support the arm and use `fd torque off --force`")


def _segment(arm, tool: Tool, a, b, min_height: float, what: str) -> float:
    """Check the straight joint-space piece a -> b every 1.5 deg (as run_path does) with check_pose; the
    lowest tool tip height on it (the start pose is not judged: the arm is there)."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = max(1, int(math.ceil(math.degrees(float(np.max(np.abs(b - a)))) / 1.5)))
    lo = tip_height(arm, a, tool)
    for k in range(1, n + 1):
        lo = min(lo, check_pose(arm, tool, a + (b - a) * (k / n), q_start=a, min_height=min_height,
                                what=f"{what}, point {k}/{n}"))
    return lo


def _ways(arm, tool: Tool, q, q_to):
    """Candidate ways from q to q_to, simplest first: (name, [intermediate poses])."""
    yield "direct", []
    first = q.copy()
    first[3:] = q_to[3:]
    yield "wrist first", [first]
    late = q_to.copy()
    late[3:] = q[3:]
    yield "wrist last", [late]
    try:
        r = ready_q(arm, tool)
    except SafetyError:
        return
    if min(float(np.max(np.abs(r - q))), float(np.max(np.abs(r - q_to)))) < math.radians(2.0):
        return
    yield "via the stand-by pose", [r]
    a = q.copy()
    a[3:] = r[3:]
    yield "wrist first, then via the stand-by pose", [a, r]
    b = q_to.copy()
    b[3:] = r[3:]
    yield "via the stand-by pose, wrist last", [r, b]
    yield "wrist first, via the stand-by pose, wrist last", [a, r, b]


def plan_free_move(arm, tool: Tool, q_from, q_to, min_height: float = MIN_FREE_HEIGHT) -> dict:
    """Plan a free move (nothing held against the table) from q_from to q_to with a tool on the gripper.
    No motion. Returns {"lift": [q_from, ...] | None, "path": [q, ..., q_to], "via": str, "min_height_m"}.

    A tool tip lower than LIFT_HEIGHT is first lifted straight up ("lift"). From there the way is a chain
    of straight joint-space pieces ("path": direct when that is clear, else with the wrist moved first or
    last, or through the stand-by pose); on every 1.5 deg of it check_pose holds and the tool tip is at
    least min_height + PLAN_MARGIN above the table. Among the ways that pass, the first that does not dip
    is taken, else the one that stays highest. Raises SafetyError when no way passes."""
    q0, q_to = np.asarray(q_from, dtype=float), np.asarray(q_to, dtype=float)
    limit = float(min_height) + PLAN_MARGIN
    h_to = tip_height(arm, q_to, tool)
    if h_to < limit:
        raise SafetyError(f"the pose to go to has the tool tip {h_to * 1000:.0f} mm above the table; a free move "
                          f"keeps it above {limit * 1000:.0f} mm")
    lift, q = None, q0
    if tip_height(arm, q0, tool) < LIFT_HEIGHT - 0.002:
        lift = lift_path(arm, tool, q0)
        q = np.asarray(lift[-1], dtype=float)
    # the simplest way that does not dip is taken; a way that passes but comes more than 3 mm lower than it
    # starts (or than 1 cm above the limit) is kept only if no other way stays higher: on the real arm the
    # measured tip runs up to a centimetre below the plan, and free_move stops the motion at min_height
    roomy = min(tip_height(arm, q, tool), limit + 0.01) - 0.003
    tried, best = [], None
    for name, mids in _ways(arm, tool, q, q_to):
        pts = [q] + list(mids) + [q_to]
        try:
            lo = min(_segment(arm, tool, pts[i], pts[i + 1], limit, name) for i in range(len(pts) - 1))
        except SafetyError as e:
            tried.append(f"{name}: {e}")
            continue
        if best is None or lo > best["min_height_m"] + 0.002:
            best = {"lift": lift, "path": pts, "via": name, "min_height_m": round(float(lo), 4)}
        if lo >= roomy:
            break
    if best is None:
        raise SafetyError(f"no way to that pose keeps the tip of tool '{tool.name}' {limit * 1000:.0f} mm above the "
                          f"table and passes the safety checks. Nothing moved. Tried: {' | '.join(tried[:3])}")
    return best


def free_move(arm, q_to, speed_deg_s: float, settle=0.5, min_height: float = MIN_FREE_HEIGHT,
              lift_speed_deg_s: float | None = None, plan: dict | None = None) -> dict:
    """plan_free_move from the measured pose, then run it. Every piece is its own Arm.run_path (the
    reference is then exactly the checked straight piece: no corner rounding), re-checked from the pose the
    arm really stands in; the measured tool tip height is watched and the motion stops (MotionAborted, the
    arm holds) the moment it is below min_height. Returns the last run_path result plus "tool", "via",
    "lifted_first", "min_tip_height_m" (lowest MEASURED height) and the total "duration_s"."""
    tool = arm.tool
    if tool is None:
        raise SafetyError("free_move is for an arm with a tool attached")
    q = arm.read_q()
    if plan is not None:
        start = np.asarray((plan["lift"] or plan["path"])[0], dtype=float)
        if float(np.max(np.abs(start - q))) > math.radians(0.5):
            plan = None  # the arm is not where the plan starts (it settled when the torque came on): plan again
    if plan is None:
        plan = plan_free_move(arm, tool, q, q_to, min_height)
    low = {"h": tip_height(arm, q, tool)}
    total = 0.0

    def run(wps, speed, settle_, monitor, what):
        try:
            return arm.run_path(wps, speed, settle=settle_, monitor=monitor)
        except MotionAborted as e:
            if e.info.get("reason") == "monitor":
                raise MotionAborted(f"{what}: stopped because the tool tip was {low['h'] * 1000:.0f} mm above the "
                                    "table (measured); the arm holds where it stopped",
                                    info={**e.info, "tool_tip_height_m": round(low["h"], 4)}) from e
            raise

    if plan["lift"]:
        h0 = low["h"]

        def rising(t, q_plan, q_meas):
            h = tip_height(arm, q_meas, tool)
            low["h"] = min(low["h"], h)
            # it is supposed to go UP. (A tip that stands on the table cannot go down; the encoders may
            # still say so for a moment while the servos take the arm's weight over.)
            return h0 > 0.006 and h < h0 - 0.004

        r = run(plan["lift"], lift_speed_deg_s or min(float(speed_deg_s), 12.0), 0.3, rising, "lifting the tool")
        total += r["duration_s"]

    def watch(t, q_plan, q_meas):
        h = tip_height(arm, q_meas, tool)
        low["h"] = min(low["h"], h)
        return h < min_height

    pts = plan["path"]
    r = None
    for i in range(1, len(pts)):
        last = i == len(pts) - 1
        q_now = arm.read_q()
        if i > 1 or plan["lift"]:
            # the previous piece ended within the arrival tolerance of its target, not on it
            _segment(arm, tool, q_now, pts[i], min_height, f"{plan['via']} (piece {i})")
        r = run([q_now, pts[i]], speed_deg_s, settle if last else 0.3, watch, f"free move ({plan['via']})")
        total += r["duration_s"]
    out = dict(r)
    out.update(tool=tool.name, via=plan["via"], lifted_first=bool(plan["lift"]), pieces=len(pts) - 1,
               min_tip_height_m=round(float(low["h"]), 4), duration_s=round(total, 3))
    return out


def lift(arm, to_height: float = LIFT_HEIGHT, speed_deg_s: float = 12.0, settle=0.3) -> dict:
    """Tool up: raise the tool tip straight up to to_height above the table when it is lower (lift_path),
    e.g. before a command that stops reports where the tool is. The torque must be on. Returns
    {"lifted": bool, "tip_height_m"} (+ the run_path figures when it moved)."""
    tool = arm.tool
    if tool is None:
        raise SafetyError("lift is for an arm with a tool attached")
    q = arm.read_q()
    h0 = tip_height(arm, q, tool)
    if h0 >= float(to_height) - 0.002:
        return {"lifted": False, "tip_height_m": round(h0, 4)}
    wps = lift_path(arm, tool, q, to_height)

    def rising(t, q_plan, q_meas):
        return h0 > 0.006 and tip_height(arm, q_meas, tool) < h0 - 0.004

    r = arm.run_path(wps, speed_deg_s, settle=settle, monitor=rising)
    return {"lifted": True, "tip_height_m": round(tip_height(arm, arm.read_q(), tool), 4),
            "joint_err_deg": r["joint_err_deg"], "duration_s": r["duration_s"]}


def _power(arm, tier):
    """Torque on if needed (the arm holds where it is, the gripper keeps the position it has) and the
    speed tier's servo profile; returns the tier's speeds."""
    if not arm.is_torqued():
        arm.torque_on()
        getattr(arm, "_sleep", time.sleep)(0.3)  # the servos take the arm's weight over before heights are judged
    return arm.apply_profile(tier)


def go_ready(arm, tier: str | None = None) -> dict:
    """`ready` with a tool attached (skills.go_ready hands over to this): the TOOL tip goes to the
    stand-by point (ready_q) by free_move. The whole way is planned and checked before the torque goes on,
    so a refusal leaves the arm exactly as it was."""
    tool = arm.tool
    arm.speeds(tier)  # an unknown / unvalidated tier refuses first
    pose = ready_pose(arm, tool)
    plan = plan_free_move(arm, tool, arm.read_q(), pose["q"])
    sp = _power(arm, tier)
    out = free_move(arm, pose["q"], sp["transit"], settle=0.5, lift_speed_deg_s=sp["approach"], plan=plan)
    out.update(tool_tip_xyz=[round(float(v), 4) for v in K.fk(arm.read_q(), arm.tip_tool())[:3, 3]],
               standby_xyz=pose["xyz"], standby_pitch_deg=pose["pitch"], standby_moved_m=pose["moved_m"])
    return out


def in_rest_pose(arm, tool: Tool, q=None) -> tuple:
    """(True, "") when the arm may be de-energised with this tool on, else (False, why): lift and elbow
    within REST_TOL_DEG of the fold (they lie on their stops there), the wrist where a rest pose has it
    (between the angles rest_q gives for the default and for the largest clearance, REST_TOL_DEG to
    spare), and the tool tip at least REST_MIN_HEIGHT above the table."""
    q = arm.read_q() if q is None else np.asarray(q, dtype=float)
    try:
        ref = rest_q(arm, tool)
    except SafetyError as e:
        return False, f"no rest pose exists for it ({e})"
    try:
        w_high = float(rest_q(arm, tool, clearance_m=0.15)[3])   # the wrist raised as far as a rest pose goes
    except SafetyError:
        w_high = float(ref[3])
    off = np.degrees(np.abs(q - ref))[1:4]
    w = float(q[3])
    off[2] = math.degrees(max(0.0, min(w_high, float(ref[3])) - w, w - max(w_high, float(ref[3]))))
    if float(off.max()) > REST_TOL_DEG:
        return False, f"{ARM[1 + int(np.argmax(off))]} is {off.max():.0f} deg away from the tool's rest pose"
    h = tip_height(arm, q, tool)
    if h < REST_MIN_HEIGHT:
        return False, f"the tool tip is only {h * 1000:.0f} mm above the table"
    return True, ""


def torque_off_block(arm) -> str | None:
    """Why torque off must be refused with the tool that is on the arm, or None when it is fine: nothing
    holds the arm up anyway, or the arm is in the tool's rest pose."""
    tool = getattr(arm, "tool", None)
    if tool is None:
        return None
    holding = [j for j in ("shoulder_lift", "elbow_flex", "wrist_flex") if arm.bus.read(j, "Torque_Enable") == 1]
    if not holding:
        return None
    ok, why = in_rest_pose(arm, tool)
    if ok:
        return None
    return (f"refusing torque off: tool '{tool.name}' is attached and {why}; released here the arm would fall "
            "and the tool would hit the table. Use `fd tool rest` (folds the arm away with the tool, then "
            "switches the torque off), or `fd torque off --force` as an emergency stop while a person holds the arm")


def rest(arm, tier: str | None = None, clearance_m: float = REST_CLEARANCE, pan_deg: float = REST_PAN_DEG,
         torque_off: bool = True, dry: bool = False, watch_s: float = 1.5) -> dict:
    """`fd tool rest`: fold the arm into the tool's rest pose (free_move), switch the torque off and watch
    that it stays there. The replacement for `park` while a tool is attached.

    dry=True plans only. An arm that is already limp in the rest pose is left alone. Everything is planned
    and checked before the torque goes on. Returns a result with "ok", "stage" ("planned" for a dry run,
    else "rest"), and "reason" / "next" when it did not end resting."""
    tool = arm.tool
    if tool is None:
        raise SafetyError("no tool is attached: `fd park` folds the arm away")
    arm.speeds(tier)  # an unknown / unvalidated tier refuses first
    q_rest = rest_q(arm, tool, clearance_m, pan_deg)
    q = arm.read_q()
    torqued = arm.is_torqued()
    out = {"tool": tool.name, "rest_q_deg": [round(math.degrees(float(v)), 2) for v in q_rest],
           "rest_tip_height_mm": round(tip_height(arm, q_rest, tool) * 1000.0, 1),
           "tip_height_mm": round(tip_height(arm, q, tool) * 1000.0, 1)}
    if not torqued and in_rest_pose(arm, tool, q)[0]:
        out.update(ok=True, stage="rest", torque="off", moved=False,
                   note="the arm already rests in the tool's rest pose with the torque off; nothing moved")
        return out
    plan = plan_free_move(arm, tool, q, q_rest)
    out.update(via=plan["via"], lift_first=bool(plan["lift"]), planned_min_tip_height_mm=round(plan["min_height_m"] * 1000.0, 1))
    if dry:
        out.update(ok=True, stage="planned", moved=False)
        return out
    sp = _power(arm, tier)
    r = free_move(arm, q_rest, min(float(sp["transit"]), 30.0), settle=1.0, lift_speed_deg_s=sp["approach"],
                  plan=plan)
    out.update(moved=True, joint_err_deg=r["joint_err_deg"], duration_s=r["duration_s"],
               min_tip_height_mm=round(r["min_tip_height_m"] * 1000.0, 1), start_tip_height_mm=out["tip_height_mm"],
               tip_height_mm=round(tip_height(arm, arm.read_q(), tool) * 1000.0, 1))
    ok, why = in_rest_pose(arm, tool)
    short = np.degrees(np.abs(arm.read_q() - q_rest))[1:4]
    if ok and float(short.max()) > 3.0:
        ok, why = False, f"{ARM[1 + int(np.argmax(short))]} stopped {short.max():.1f} deg short of it"
    if not ok:
        out.update(ok=False, stage="rest", torque="on", reason=f"the arm did not end in the rest pose: {why}",
                   next="the torque is still ON and the arm holds; `fd state`, then `fd tool rest` again")
        return out
    if not torque_off:
        out.update(ok=True, stage="rest", torque="on")
        return out
    sleep = getattr(arm, "_sleep", time.sleep)
    sleep(0.4)
    h_before = tip_height(arm, arm.read_q(), tool)
    arm.torque_off()
    # 2026-11-11: "torque off: nothing moved in 3 s" was checked by eye; here by the encoders
    h_low = h_before
    for _ in range(max(1, int(round(watch_s / 0.1)))):
        sleep(0.1)
        h_low = min(h_low, tip_height(arm, arm.read_q(), tool))
    out.update(torque="off", tip_height_mm=round(h_low * 1000.0, 1),
               sag_after_release_mm=round(max(0.0, h_before - h_low) * 1000.0, 1))
    if h_low < MIN_FREE_HEIGHT:
        out.update(ok=False, stage="rest",
                   reason=f"after the torque went off the arm sagged: the tool tip is {h_low * 1000:.0f} mm above the table",
                   next="support the arm by hand; `fd torque on` holds it where it is, then `fd tool rest --clearance 0.07`")
    else:
        out.update(ok=True, stage="rest")
    return out


# ----------------------------------------------------------------------------- refusals
NEXT_LOCKED = ("`fd tool rest` folds the arm away with the tool on (torque off); once the tool is off the "
               "gripper, `fd tool detach` unlocks the gripper and `park`")


def lock_message(tool: Tool, what: str) -> str:
    """Text of the SafetyError for something that must not happen while a tool is attached."""
    return (f"tool '{tool.name}' is attached to the gripper: {what} is refused (the gripper must not move and "
            f"the arm must not fold as if nothing were taped to it). Nothing moved. {NEXT_LOCKED}")


def refusal(tool: Tool, what: str) -> dict:
    """The complete result of a command that is refused because a tool is attached."""
    msg = lock_message(tool, what)
    return {"ok": False, "error": "SafetyError", "stage": "tool", "reason": msg, "message": msg, "tool": tool.name,
            "next": NEXT_LOCKED,
            "say": f"グリッパーにツール「{tool.name}」が付いているので、{what} は実行しませんでした（アームは動かしていません）。"
                   "片付けるときは fd tool rest、ツールを外したあとは fd tool detach を使ってください。"}


# ----------------------------------------------------------------------------- simulation
def sim_start(arm) -> bool:
    """Simulated arm only: start in the tool's rest pose instead of the plain rest pose (which would put a
    6 cm tool 17 mm INSIDE the table: no real arm can be there). True when the pose was set."""
    tool = getattr(arm, "tool", None)
    bus = getattr(arm, "bus", None)
    if tool is None or not (hasattr(bus, "pos") and hasattr(bus, "goal")):
        return False
    if arm.raw_limits is None:
        arm.raw_limits = bus.limits()
    try:
        raw = arm.q_to_raw(rest_q(arm, tool))
    except SafetyError:
        return False
    if tool.gripper_raw is not None:
        raw["gripper"] = int(tool.gripper_raw)
    for j, v in raw.items():
        bus.pos[j] = float(v)
        bus.goal[j] = float(v)
        if hasattr(bus, "sp"):
            bus.sp[j] = float(v)
    return True
