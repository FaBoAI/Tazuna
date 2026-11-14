"""A tool taped to the gripper (a pen, a marker, a stylus): what is attached, where its tip is, and what
that changes for safety.

Why this exists (2026-11-11): a pen was taped to the gripper and the harness knew nothing of it. `fd park`
opens the gripper to 10 % and folds the arm so that the FINGERTIP ends 28 mm above the table: with a 6 cm
tool the tip would have been driven 18 mm into the table. `fd ready` put the fingertip, not the tool tip,
at the stand-by point. That night's scripts got by with a second config file (FD_CONFIG trick) and a list
of commands "never to use". Here the tool is state of the harness itself:

- ONE file says what is on the gripper: tool_active.json in the project folder (TOOL_FILE). No file = no
  tool = exactly the old behaviour. $FD_TOOL overrides the path (the tests use it, so that no test ever
  writes the real file). A file that cannot be read is an error, never "no tool": the harness must not
  unlock the gripper because a file is damaged.
- `robot.Arm` reads it when it is built (`arm.tool`); `Arm.tip_tool()` is then the TOOL tip, so the arm
  plans, checks (floor, keep-outs) and moves the tool tip. The jaw and wrist lines of the floor check
  stay what they were (arm.cfg["tip_offset"] remains the fingertip).
- With a tool attached the gripper never receives a goal (robot.Arm refuses every gripper method and
  every gripper goal; the one exception is the jaw of a tool on the moving jaw, below), `park` is refused
  (skills.park), `ready` goes to the tool stand-by pose (go_ready here), torque off is only possible in the
  tool's rest pose, and `fd tool rest` (rest() here) is the way to fold the arm away: the pose that worked
  on the real arm that night, generalised.
- A free move with a tool (go_ready, rest) never brings the tool tip lower than MIN_FREE_HEIGHT above
  the table: a tool that is low is first lifted straight up, the way is checked every 1.5 deg before
  anything moves, and the measured height is watched while it moves.
- A tool can carry joint limits of its own (Tool.joint_limits_deg; 2026-11-13: the 30 cm handle of a brush
  lay on the forearm when the wrist straightened, and hit it when the gripper rolled). Every pose made here
  lies inside them or is refused with the limit named; an arm that was moved by hand to a pose outside them
  can be lifted and moved back inside, never further out (lift_path).

A tool on the MOVING jaw (2026-11-13). A thin brush was taped to the outer face of the moving jaw, three
times. Open (35 deg) its tip stood 9 cm beside the finger line; with the jaw at 0 deg it lay along the
fingers but floated several millimetres in the jaw's gear backlash (pulled strokes painted, pushed and
sideways ones skipped). Rigid, and painting in every direction, it was only with the jaw closed AGAINST THE
FIXED JAW and held there by a light squeeze: raw 2041 where it met the jaw, goal 25 ticks beyond (2016),
load about 110, 30 C after 90 minutes. That was done by scratch scripts that wrote gripper goals past the
lock. Here it is an operation of the harness (move_jaw, hold_jaw; the command is `fd tool jaw`, tooljaw.py):
- the tool file says on which jaw the tool is (Tool.on_jaw). Only "moving_outer" (the outer face of the
  moving jaw, nothing between the jaws) lets the jaw move; a tool held between the jaws would be crushed by
  closing and dropped by opening, so "between" and "not said" stay locked;
- the jaw goal walks in steps of at most 15 ticks every 0.12 s, the position is read after every step, and
  the walk stops where the jaw no longer follows. Against the fixed jaw it is then held with the squeeze;
  stopped anywhere else, the push is taken off;
- the tool MOVES with the jaw: afterwards its tip is where nobody measured it. The file gets the new jaw
  position, the tip the jaw model predicts (jaw_tip_estimate: about 1 cm off that evening) and loses
  tip_measured and the pen calibration; `fd calib-air --tip-only --save` measures the tip again;
- the squeeze is a servo goal and is lost when the torque goes off (`fd tool rest`): hold_jaw writes it again
  when the jaw still stands where the file says, and refuses when it does not;
- the tip offset of a tool holds for ONE jaw position, the one the file records (gripper_raw). Review of
  2026-11-13: that was compared only for a tool with a recorded squeeze, and the rig's own file of that evening
  had none (the squeeze stood in its notes), so a limp jaw 100 ticks from its record, the tip 21 mm from the
  file's, went on to paint. jaw_in_place now answers for every tool that records a jaw position, except one
  recorded on the fixed jaw (jaw_tolerance): a jaw that stands elsewhere stops `ready` and brush work before
  the torque comes on, and the refusal names the `fd tool jaw` command that puts it right;
- when the servo bus stops answering in the middle of a jaw move, where the jaw stands is not known: the file
  then says exactly that (jaw_lost: no jaw position, no hold, the tip not measured, a note).

Simulation: `fd --sim` never writes the real tool file. A tool attached with `fd --sim tool attach` lives
in SIM_TOOL_FILE and is what the simulated arm carries until `fd --sim tool detach`; without it the
simulation mirrors the real tool (read only), so a real action can be rehearsed with the real tool's data.
A simulated command that CHANGES the mirrored tool (`fd --sim tool jaw mount`, `fd --sim tool set`) stores
its result as a copy in SIM_TOOL_FILE, marked as a copy (SIM_COPY_KEY). Such a copy is read only as long as
the real tool file is what it was copied from: once the real tool is changed or replaced, the copy is left
behind and the simulation mirrors the real tool again (review of 2026-11-13: a rehearsal went on with the
old brush after the real one had been registered anew, and nothing said so). sim_tool_state() tells a
command what to say about it.

Units: metres, base frame; angles in degrees in files and results, radians for joint vectors; the jaw in
raw servo ticks and URDF degrees (0 = jaws parallel, -11.41 = fingertips touch, more = further open).
Tests: test_tooling.py, test_tooljaw.py. Nothing here opens the servo bus or the camera by itself.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shlex
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace

import numpy as np

import gripper_model as GM
import kinematics as K
import tazuna_paths
from robot import ARM, DEFAULT_CONFIG, MotionAborted, SafetyError

TOOL_FILE = os.path.join(tazuna_paths.ROOT, "tool_active.json")            # in the project folder, as before
SIM_TOOL_FILE = os.path.join(tazuna_paths.ROOT, "tool_active.sim.json")
# Key of a SIMULATED tool file that is a copy of the real tool's record (not a tool attached in simulation):
# {"file", "name", "attached_at", "sha1"} of the real tool file it was copied from. Never in a real tool file.
SIM_COPY_KEY = "sim_copy_of"
MOUNTS = ("along_fingers", "across_fingers", "other")
# On which jaw a tool is taped (Tool.on_jaw). "moving_outer": the outer face of the moving jaw, nothing between
# the jaws: the only value that lets `fd tool jaw` move the jaw. "between": held between the jaws.
ON_JAW = ("fixed", "moving_outer", "between")
# --- the jaw of a tool on its outer face (numbers of 2026-11-13, reference/thin_close.py and thin_squeeze.py)
JAW_STEP_TICKS = 15         # the jaw goal moves at most this far per step ...
JAW_STEP_S = 0.12           # ... one step every 0.12 s (125 ticks/s = 11 deg/s: the jaw followed that)
JAW_NEAR_STEP_TICKS = 10    # steps while the fixed jaw is looked for (what met it at raw 2041) ...
JAW_NEAR_TICKS = 60         # ... from this far above where the fingertips touch (cfg["gripper_touch_raw"])
JAW_LAG_TICKS = 30          # "the jaw no longer follows": it STANDS this far behind its goal ...
JAW_LAG_MAX_TICKS = 70      # ... or is this far behind it while still moving (the stop of thin_close.py)
JAW_STILL_TICKS = 6         # "stands": it came less than this far in one step (pressed on the fixed jaw it creeps 2..3)
JAW_SQUEEZE_TICKS = 25      # goal beyond where the jaw met the fixed jaw: load about 110, 30 C after 90 minutes
JAW_SQUEEZE_CAP_TICKS = 60  # no tool file may ask for more ("60 sits on the overload threshold", robot.py)
JAW_CLOSED_BAND_TICKS = 25  # the fixed jaw is met at most this far above cfg["gripper_touch_raw"] (tape between
#                             the jaws); a jaw that stops earlier is held open by something
JAW_BELOW_TOUCH_TICKS = 60  # the search for the fixed jaw commands no goal further below the touch position
JAW_HOLD_TOL_TICKS = 40     # hold_jaw: the limp jaw may stand this far from its record (the squeeze pulls it back)
JAW_END_TOL_TICKS = 8       # ... and with the goal written again it must stand this near to it (0.7 deg). The same
#                             for a jaw that NO squeeze pulls back (jaw_tolerance): where it stands is where the tool
#                             is. The thin brush's tip is 138 mm from the jaw's pivot: 0.21 mm per tick, so 8 ticks
#                             are 1.7 mm at the tip, and 40 would be 8.5 mm (the line press is 3 mm, the band's 6)
JAW_SAME_TICKS = 3          # a jaw that ends this near to its record did not move the tool (0.6 mm at the tip)
JAW_ESTIMATE_ERR_M = 0.010  # how far jaw_tip_estimate was off on the real arm (8.3 mm): "about 1 cm"
# gripper pitch (deg below horizontal) a tool is worked with unless `fd tool attach --pitch` says otherwise.
# along the fingers: 75 is what drew the cat (the pen leans 15 deg, tip away from the robot);
# across the fingers (tip on the fixed-jaw side): the tool is upright with the gripper level (pitch 0), which
# the arm can hold almost nowhere on the table; 30 (the tool leans 30 deg, tip towards the robot) it can.
DEFAULT_PITCH = {"along_fingers": 75.0, "across_fingers": 30.0, "other": 75.0}
MIN_FREE_HEIGHT = 0.02    # a free move never brings the tool tip lower than this above the table
PLAN_MARGIN = 0.005       # ... and is planned with this much to spare (tracking error, encoder noise)
LIFT_HEIGHT = 0.035       # a tool found lower than this is first lifted straight up to it
LIFT_ARC_MAX_M = 0.04     # a lift by ONE joint carries the tip sideways; further than this it is told, not done
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
    gripper_raw: int | None = None   # jaw position at attach time, or where `fd tool jaw` left it: where tip_offset holds
    #                                  (compared with the jaw before a move: jaw_in_place; None: not known, not compared)
    contact: dict | None = None      # {"x0","y0","a","bx","by","rms_mm","points"}: plan z of first contact
    corr_map: dict | None = None     # {"centre","kind","coef","rms_mm","max_mm","n"}: commanded -> actual
    region: list | None = None       # [x0, y0, x1, y1] where contact / map hold
    calibrated: dict = field(default_factory=dict)   # {"tip": {...}, "contact": {...}, "map": {...}} with "at"
    notes: list = field(default_factory=list)
    attached_at: str | None = None   # ISO time of `fd tool attach`
    # Extra joint limits (URDF degrees) that hold while this tool is on the gripper: {"wrist_flex": [20, 97]}
    # (the brush of 2026-11-13; 97 and not the wrist's own 103: a move that ended at 103 stopped 3.7 deg short,
    # the handle against the arm). For a tool whose BODY reaches back over the arm (a long brush handle taped
    # along the gripper: it lies on the forearm when the wrist straightens, and swings into the wrist when the
    # gripper rolls). Arm.q_limits() intersects them with the servo and soft limits, so planning, IK and every
    # check respect them.
    joint_limits_deg: dict | None = None
    extra: dict = field(default_factory=dict)        # keys of the file this version does not know (kept on save)
    # --- a tool taped to a JAW (2026-11-13). None in every file older than these keys, and such a file is
    # written back without them: an old tool is exactly what it was.
    # On which jaw: one of ON_JAW, or None = not said. Only "moving_outer" lets the jaw be moved (move_jaw).
    on_jaw: str | None = None
    # The gripper goal that holds the jaw closed against the fixed jaw with a light squeeze: gripper_raw is then
    # where the jaw stands, this goal a few ticks beyond it. Written by move_jaw, restored by hold_jaw after the
    # torque was off. None: the jaw is not held by a squeeze.
    gripper_goal_raw: int | None = None
    # How tip_offset was measured (what `fd calib-air --tip-only --save` writes: tools/calib). None: the tip is typed
    # in, a first guess or an estimate. A CHANGED tip_offset clears it (__setattr__): assign the tip first, then
    # this record (set_measured_tip does it in that order).
    tip_measured: dict | None = None

    _FIELDS = ("name", "tip_offset", "mount", "pitch_deg", "gripper_raw", "contact", "corr_map", "region",
               "calibrated", "notes", "attached_at", "joint_limits_deg")
    _JAW_FIELDS = ("on_jaw", "gripper_goal_raw", "tip_measured")     # written only when they say something

    def __setattr__(self, name, value):
        # whoever gives the tool another tip (`fd tool set --tip`, the ink stage of `fd tool calibrate`, a jaw
        # that moved) leaves a tip the air measurement does not describe: its record must not survive that
        if (name == "tip_offset" and self.__dict__.get("tip_measured") is not None
                and not _same_tip(self.__dict__.get("tip_offset"), value)):
            self.__dict__["tip_measured"] = None
        object.__setattr__(self, name, value)

    def to_dict(self) -> dict:
        d = dict(self.extra)
        d.update({k: getattr(self, k) for k in self._FIELDS})
        d["tip_offset"] = [round(float(v), 5) for v in self.tip_offset]
        for k in self._JAW_FIELDS:
            if getattr(self, k) is not None:
                d[k] = getattr(self, k)
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
        tool.joint_limits_deg = check_joint_limits(d.get("joint_limits_deg"))
        tool.on_jaw, tool.gripper_goal_raw = check_jaw_record(d.get("on_jaw"), tool.gripper_raw, d.get("gripper_goal_raw"))
        measured = d.get("tip_measured")
        if measured is not None and not isinstance(measured, dict):
            raise SafetyError("tool description: 'tip_measured' must be dict or null")
        tool.tip_measured = None if measured is None else dict(measured)
        tool.extra = {k: v for k, v in d.items() if k not in cls._FIELDS and k not in cls._JAW_FIELDS}
        return tool


def _same_tip(a, b) -> bool:
    """Two tip offsets that are the same to the 0.01 mm the tool file keeps."""
    try:
        a, b = [round(float(v), 5) for v in a], [round(float(v), 5) for v in b]
    except (TypeError, ValueError):
        return False
    return len(a) == 3 and a == b


def check_jaw_record(on_jaw, gripper_raw, gripper_goal_raw) -> tuple:
    """(on_jaw, gripper_goal_raw) of a tool description, checked. SafetyError for an on_jaw that is not one of
    ON_JAW (never read as "the jaw may move"), and for a holding goal that would not be the light squeeze of a
    jaw closed on the fixed jaw: recorded for a tool that is not on the moving jaw's outer face, without the
    position it belongs to, not below that position, or more than JAW_SQUEEZE_CAP_TICKS beyond it (a goal typed
    in by hand must not become a stalled servo)."""
    if on_jaw is not None and on_jaw not in ON_JAW:
        raise SafetyError(f"tool description: 'on_jaw' must be one of {', '.join(ON_JAW)} or null (got {on_jaw!r})")
    if gripper_goal_raw is None:
        return on_jaw, None
    try:
        goal = float(gripper_goal_raw) if isinstance(gripper_goal_raw, (int, float)) else float("nan")   # (a number, not text)
        raw = float("nan") if gripper_raw is None else float(gripper_raw)
    except (TypeError, ValueError):
        goal = raw = float("nan")
    if isinstance(gripper_goal_raw, bool) or not (math.isfinite(goal) and goal == int(goal) and 0 <= goal <= 4095):
        raise SafetyError(f"tool description: 'gripper_goal_raw' must be a servo position 0 .. 4095 or null "
                          f"(got {gripper_goal_raw!r})")
    if on_jaw != "moving_outer":
        raise SafetyError("tool description: 'gripper_goal_raw' (a jaw held with a squeeze) is recorded, but the tool "
                          "is not recorded as taped to the outer face of the moving jaw (on_jaw = moving_outer)")
    if not (math.isfinite(raw) and 0 < raw - goal <= JAW_SQUEEZE_CAP_TICKS):
        raise SafetyError(f"tool description: 'gripper_goal_raw' {int(goal)} must lie 1 .. {JAW_SQUEEZE_CAP_TICKS} ticks "
                          f"below 'gripper_raw' ({gripper_raw!r}): it is the squeeze of a jaw closed on the fixed jaw")
    return on_jaw, int(goal)


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


def check_joint_limits(v) -> dict | None:
    """Extra joint limits of a tool as {joint: [lo_deg, hi_deg]} (None when there are none). SafetyError for
    anything else: an unknown joint, a range that is not two finite numbers lo < hi."""
    if v is None or v == {}:
        return None
    if not isinstance(v, dict):
        raise SafetyError(f"tool joint limits must be an object {{joint: [lo_deg, hi_deg]}} (got {v!r})")
    out = {}
    for j, rng in v.items():
        if j not in ARM:
            raise SafetyError(f"tool joint limits: unknown joint {j!r} (one of {', '.join(ARM)})")
        try:
            lo, hi = (float(x) for x in rng)
        except (TypeError, ValueError):
            lo = hi = float("nan")
        if not (math.isfinite(lo) and math.isfinite(hi) and -360.0 <= lo < hi <= 360.0):
            raise SafetyError(f"tool joint limits: {j} needs [lo_deg, hi_deg] with lo < hi (got {rng!r})")
        out[j] = [lo, hi]
    return out


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


def _real_record() -> dict | None:
    """The real tool file as it is now, for telling whether a simulated copy still describes it: {"file",
    "name", "attached_at", "sha1"}; None without a real tool file. Name and attach time say which
    registration it is (`fd tool attach` writes both anew), the digest of the file's bytes whether anything in
    it was changed since (a file time would miss a change within the same clock tick). A file that cannot be
    read has name and attached_at None: no copy is a copy of that."""
    path = tool_path(False)
    if not os.path.exists(path):
        return None
    rec = {"file": path, "name": None, "attached_at": None, "sha1": None}
    try:
        with open(path, "rb") as f:
            raw = f.read()
        rec["sha1"] = hashlib.sha1(raw).hexdigest()
        d = json.loads(raw.decode("utf-8"))
    except (OSError, ValueError):
        return rec
    if isinstance(d, dict):
        rec.update(name=d["name"].strip() if isinstance(d.get("name"), str) else None, attached_at=d.get("attached_at"))
    return rec


def _stale_copy(path: str) -> str | None:
    """Why the simulated tool file `path` is left behind (not read any more), or None when it is read: it is a
    COPY of the real tool's record, and the real tool has been changed or replaced since the copy was made.
    A marked copy (SIM_COPY_KEY) is compared with what it was copied from. A file without the mark is a tool
    attached in simulation (`fd --sim tool attach`: never left behind), unless it carries the real tool's own
    name and attach time and is older than the real tool file: that is a copy made before the mark existed.
    A simulated file that cannot be read is not "stale": load() says what is wrong with it."""
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(d, dict):
        return None
    real = _real_record()
    mark = d.get(SIM_COPY_KEY)
    if isinstance(mark, dict):
        was = f"the real tool '{mark.get('name')}'" + (f" (attached {mark['attached_at']})" if mark.get("attached_at") else "")
        if real is None:
            return f"it is a copy of {was}, and the real arm carries no tool any more"
        if (real["name"], real["attached_at"]) != (mark.get("name"), mark.get("attached_at")):
            now = (f"'{real['name']}'" + (f" (attached {real['attached_at']})" if real["attached_at"] else "")
                   if real["name"] is not None else "a file that cannot be read")
            return f"it is a copy of {was}, and the real tool is now {now}"
        if real["sha1"] != mark.get("sha1"):
            return f"it is a copy of {was} as it was before the real tool file was last changed"
        return None
    try:
        older = os.stat(real["file"]).st_mtime_ns > os.stat(path).st_mtime_ns if real is not None else False
    except OSError:
        older = False
    if (older and real["name"] is not None and real["attached_at"]
            and (d.get("name"), d.get("attached_at")) == (real["name"], real["attached_at"])):
        return f"it is a copy of the real tool '{real['name']}' that is older than the real tool file"
    return None


def _sim_tool_file() -> tuple:
    """(file, left behind): the simulated tool file when `fd --sim` reads its tool from it (else None: it
    mirrors the real arm's tool file), and why a simulated file that exists is NOT read (_stale_copy; else
    None). With $FD_TOOL there is one file for both: (None, None)."""
    path = tool_path(True)
    if path == tool_path(False) or not os.path.exists(path):
        return None, None
    why = _stale_copy(path)
    return (None, why) if why else (path, None)


def active(sim: bool = False) -> Tool | None:
    """The tool that is attached, or None (no file = no tool). sim=True: the simulated tool when there is one
    (attached with `fd --sim tool attach`, or the copy a simulated command made of the real tool, as long as
    the real tool file is what was copied: _stale_copy), else the real one (the simulation mirrors the real
    arm; such a tool remembers what it mirrors, so that save(tool, sim=True) can mark the copy it writes)."""
    if sim:
        path, _left = _sim_tool_file()
        if path is not None:
            return load(path)
    path = tool_path(False)
    if not os.path.exists(path):
        return None
    mirrored = _real_record() if sim and tool_path(True) != path else None      # (read before the file itself)
    tool = load(path)
    if mirrored is not None:
        tool._mirrors = mirrored
    return tool


def source(sim: bool = False) -> str | None:
    """Path of the file active(sim) reads the tool from (None: no tool)."""
    if sim:
        path, _left = _sim_tool_file()
        if path is not None:
            return path
    path = tool_path(False)
    return path if os.path.exists(path) else None


def _copy_mark(tool: Tool) -> dict | None:
    """The mark for a simulated tool file that is about to be written: what it is a copy of, or None when it is
    a tool of the simulation's own. A copy: the tool was read from a marked copy; or it is the real tool as the
    simulation mirrored it (active(sim=True)), whatever the simulated command changed of it; or it carries the
    real tool's own name and attach time."""
    mark = (tool.extra or {}).get(SIM_COPY_KEY)
    if not isinstance(mark, dict):
        mark = getattr(tool, "_mirrors", None)
    if not isinstance(mark, dict):
        real = _real_record()
        same = (real is not None and real["name"] is not None and tool.attached_at
                and (tool.name, tool.attached_at) == (real["name"], real["attached_at"]))
        mark = real if same else None
    return None if mark is None else {k: mark.get(k) for k in ("file", "name", "attached_at", "sha1")}


def save(tool: Tool, sim: bool = False) -> None:
    """Write the tool file (atomically). An Arm that is already built keeps the tool it was built with:
    assign `arm.tool = tool` when it should follow. A simulated tool file that is a copy of the real tool's
    record is marked as one (SIM_COPY_KEY, _copy_mark); a real tool file never carries the mark."""
    try:
        Tool.from_dict(tool.to_dict())  # refuse to store what could not be read back
    except (TypeError, ValueError) as e:
        raise SafetyError(f"tool '{getattr(tool, 'name', '?')}' cannot be stored: {e}") from e
    path = tool_path(sim)
    d = tool.to_dict()
    d.pop(SIM_COPY_KEY, None)
    if sim and path != tool_path(False):
        mark = _copy_mark(tool)
        if mark is not None:
            d[SIM_COPY_KEY] = mark
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(d, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def sim_tool_state() -> dict:
    """What `fd --sim` has for a tool file, for the result of a simulated command (reads only):
    {"simulated_tool": the simulation reads a tool file of its own (it does NOT mirror the real arm's);
     "file": the file the simulated tool is read from (None: no tool);
     "copy_of_real": that file is a copy of the real tool's record, made when a simulated command changed the
                     tool the simulation was mirroring; it is read until the real tool file changes;
     "dropped": None, or {"file", "why"} of a simulated copy that is NOT read any more because the real tool was
                changed or replaced since (the simulation mirrors the real tool again; the next simulated command
                that stores the tool writes a new copy over it);
     "note": what to tell the operator, one English line (None when the simulation simply mirrors the real
             tool and nothing was left behind)}.
    With $FD_TOOL there is one tool file for both: nothing is simulated about it."""
    path, left = _sim_tool_file()
    out = {"simulated_tool": path is not None, "file": source(True), "copy_of_real": False, "dropped": None, "note": None}
    if left:
        out["dropped"] = {"file": tool_path(True), "why": left}
        out["note"] = (f"the simulated tool file {tool_path(True)} is not read any more ({left}): the simulation mirrors "
                       "the real tool again")
    if path is None:
        return out
    try:
        with open(path) as f:
            d = json.load(f)
    except (OSError, ValueError):
        d = None
    out["copy_of_real"] = isinstance(d, dict) and isinstance(d.get(SIM_COPY_KEY), dict)
    # (one complete command per pair of backticks: `fd --sim` and `fd --sim tool attach` alone are names, not commands)
    out["note"] = (("the SIMULATED tool file is a copy of the real tool's record that a simulated command changed (the "
                    "real tool file is as it was): the simulated commands work on that copy until the real tool file "
                    "changes; " if out["copy_of_real"] else
                    "the simulated arm carries a tool of its own (attached in simulation), not the real arm's; ")
                   + "`fd --sim tool detach` returns to the real tool")
    return out


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


def _limits(arm, tool) -> dict:
    """The arm's joint limits with `tool` on its gripper (Arm.q_limits): the arm an attach or a set looks at
    still carries the tool that was on it before, and it is the NEW tool's limits a pose must respect."""
    with _as_tool(arm, tool):
        return arm.q_limits()


def _limits_text(tool) -> str:
    """The tool's own joint limits as text: "wrist_flex 20..97, wrist_roll -5..5 deg"."""
    lim = getattr(tool, "joint_limits_deg", None) or {}
    return ", ".join(f"{j} {lo:g}..{hi:g}" for j, (lo, hi) in lim.items()) + (" deg" if lim else "")


def _outside_tool_limits(tool, q, slack_deg: float = 0.0) -> list:
    """[(joint, value_deg, lo_deg, hi_deg)] for the joints of pose q that lie outside the tool's own limits."""
    out = []
    for j, (lo, hi) in (getattr(tool, "joint_limits_deg", None) or {}).items():
        if j in ARM:
            v = math.degrees(float(q[ARM.index(j)]))
            if v < lo - slack_deg or v > hi + slack_deg:
                out.append((j, v, float(lo), float(hi)))
    return out


def _with_limits(tool, limits):
    """A copy of the tool with other joint limits (None: without any)."""
    return replace(tool, joint_limits_deg=limits)


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
    fails check_pose. The pose lies inside the tool's own joint limits (Tool.joint_limits_deg); a tool whose
    limits exclude the fold (its shoulder, elbow, the wrist angles that keep the tip up) cannot rest, and the
    refusal names the limit."""
    try:
        clearance, pan = float(clearance_m), float(pan_deg)
    except (TypeError, ValueError):
        clearance = pan = float("nan")
    if not (math.isfinite(clearance) and 0.03 <= clearance <= 0.15):
        raise SafetyError(f"rest clearance must be 0.03 .. 0.15 m above the table (got {clearance_m!r})")
    lim = _limits(arm, tool)
    if not (math.isfinite(pan) and lim[ARM[0]][0] <= math.radians(pan) <= lim[ARM[0]][1]):
        raise SafetyError(f"rest pan {pan_deg!r} deg is outside the pan limits "
                          f"[{math.degrees(lim[ARM[0]][0]):.0f}, {math.degrees(lim[ARM[0]][1]):.0f}] deg"
                          + (f" (tool '{tool.name}' limits {_limits_text(tool)})"
                             if ARM[0] in (getattr(tool, "joint_limits_deg", None) or {}) else ""))
    with _as_tool(arm, None):
        q = np.array(arm.rest_q(), dtype=float)  # the plain folded pose, clamped inside the joint limits
    q[0], q[4] = math.radians(pan), 0.0
    w_lo, w_hi = lim[ARM[3]][0] + 0.005, min(float(q[3]), lim[ARM[3]][1] - 0.005)
    give_up = "upport the arm by hand, take the tool off and run `fd tool detach`"
    own = getattr(tool, "joint_limits_deg", None) or {}

    def change(j) -> str:
        """What to do when it is the tool's own limit of joint j that keeps the arm from resting."""
        # (words, no `fd tool set --joint-limit JOINT=LO:HI`: the new range is the operator's to say, and typed as it
        # was printed that command is refused. `fd tool status` works as typed and shows the limit that is stored)
        return (f"Change that limit (fd tool set with --joint-limit {j}=LO:HI in degrees; `fd tool status` shows what is "
                f"stored), or s{give_up}")

    def height(w):
        q[3] = w
        return tip_height(arm, q, tool)

    # from the plain rest pose's wrist angle towards "gripper raised", in 1 deg steps: the first angle
    # that gives the clearance is the pose nearest to the fold the arm is known to rest in
    step = math.radians(1.0)
    last = first = None
    w = w_hi
    wrist = ARM[3]
    if w_hi < w_lo:      # (only a tool's wrist limit does that: the arm's own limits contain the fold)
        raise SafetyError(f"tool '{tool.name}': its joint limit {wrist} [{own[wrist][0]:g}, {own[wrist][1]:g}] deg excludes "
                          f"the folded pose (the wrist is at {math.degrees(q[3]):.0f} deg or less there); the arm cannot "
                          f"rest with it. {change(wrist)}")
    if height(w) < clearance:
        while w - step >= w_lo and height(w - step) < clearance:
            w -= step
        if w - step < w_lo:
            if wrist in own and math.radians(own[wrist][0]) + 0.005 >= w_lo - 1e-9:     # the tool's limit ends the search
                raise SafetyError(f"tool '{tool.name}': no wrist angle of the folded pose inside its joint limit {wrist} "
                                  f"[{own[wrist][0]:g}, {own[wrist][1]:g}] deg puts its tip {clearance * 1000:.0f} mm above "
                                  f"the table; the arm cannot rest with it. {change(wrist)}")
            raise SafetyError(f"tool '{tool.name}': no wrist angle of the folded pose puts its tip "
                              f"{clearance * 1000:.0f} mm above the table; the arm cannot rest with it. S{give_up}")
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
            first = q.copy() if first is None else first
            w -= step
            if tip_height(arm, np.array([*q[:3], max(w, w_lo), q[4]]), tool) > clearance + 0.02:
                break
    out = _outside_tool_limits(tool, q if first is None else first)
    if out:              # the fold itself (shoulder, elbow, roll 0) lies outside a limit of the tool
        j, v, lo, hi = out[0]
        raise SafetyError(f"tool '{tool.name}': its joint limit {j} [{lo:g}, {hi:g}] deg excludes the folded pose "
                          f"({j} = {v:.0f} deg there: {last}); the arm cannot rest with it. {change(j)}")
    raise SafetyError(f"tool '{tool.name}': the folded pose that keeps its tip {clearance * 1000:.0f} mm above the "
                      f"table is not safe ({last}); the arm cannot rest with it. S{give_up}")


def ready_pose(arm, tool: Tool) -> dict:
    """The stand-by pose with a tool: {"q", "xyz", "pitch", "moved_m"}. The TOOL tip is at cfg["ready"]["xyz"]
    (the plain ready pose puts the fingertip there, and the tool below it), the gripper pitched as for
    working with the tool (tool.pitch_deg, or the nearest pitch the arm can hold there), the configured roll.

    A tool too long for that point (10 cm and more along the fingers: the arm cannot fold far enough back)
    stands by at the nearest point straight ahead of it, same height, in steps of 2 cm ("moved_m" says how
    far). Raises SafetyError when no such pose exists or it fails check_pose; when it is the tool's own joint
    limits that leave none, the refusal says so and how to change them."""
    r = arm._c("ready")
    base, roll = np.array(r["xyz"], dtype=float), float(r["roll"])
    tl, lim = K.tool_offset(*tool.tip_offset), _limits(arm, tool)

    def pose_at(k, limits):
        """(q, pitch, None) of the stand-by pose 2 cm * k ahead of the point, or (None, None, why not)."""
        xyz = base + np.array([0.02 * k, 0.0, 0.0])
        pitch, _ = K.best_pitch(xyz, want=float(tool.pitch_deg), wrist_roll=math.radians(roll), tool=tl, limits=limits,
                                lo=-60.0)
        if pitch is None:        # (asked again at the tool's own pitch: a joint limit then names its joint)
            return None, None, K.ik(xyz, float(tool.pitch_deg), math.radians(roll), tool=tl, limits=limits)[1]["reason"]
        q, info = K.ik(xyz, pitch, math.radians(roll), tool=tl, limits=limits)
        return (q, pitch, None) if info["ok"] else (None, None, info["reason"])

    last = "out of reach"
    for k in range(7):
        q, pitch, why = pose_at(k, lim)
        if q is None:
            if why and (k == 0 or last == "out of reach"):       # the nearest point's reason says most
                last = why
            continue
        try:
            check_pose(arm, tool, q, min_height=MIN_FREE_HEIGHT + PLAN_MARGIN, what="tool stand-by pose")
        except SafetyError as e:
            last = str(e)
            continue
        return {"q": q, "xyz": [round(float(v), 4) for v in base + np.array([0.02 * k, 0.0, 0.0])],
                "pitch": round(float(pitch), 1), "moved_m": round(0.02 * k, 3)}
    if getattr(tool, "joint_limits_deg", None):
        free = _limits(arm, _with_limits(tool, None))
        for k in range(7):               # would the arm stand by there without the tool's own joint limits?
            q = pose_at(k, free)[0]
            if q is None:
                continue
            what = "; ".join(f"{j} would be {v:.1f} deg there, outside the tool's limit [{lo:g}, {hi:g}] deg"
                             for j, v, lo, hi in _outside_tool_limits(tool, q)) or last
            raise SafetyError(f"tool '{tool.name}': the stand-by point {base.round(3).tolist()} (and up to 12 cm ahead of "
                              f"it) cannot be reached with its tip: {what}. It is the tool's own joint limits "
                              f"({_limits_text(tool)}) that leave no stand-by pose: change them with fd tool set and "
                              "--joint-limit JOINT=LO:HI in degrees (`--joint-limit none` removes them); `fd tool rest` "
                              "folds the arm away meanwhile")
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
    tl, lim = K.tool_offset(*tool.tip_offset), _limits(arm, tool)
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
def _holding_start(tool: Tool, q):
    """The tool as a lift is PLANNED with: its joint limits widened just as far as the start pose q lies outside
    them. An arm that was moved by hand with the torque off stands where it was left: on 2026-11-13 wrist_roll
    stood at +15.6 deg under a limit of +-5, and the lift refused, because the path planner asks the inverse
    kinematics for poses INSIDE the limits and none keeps a roll the limits exclude. The waypoints are still
    checked against the real limits (check_pose with q_start), which let a joint that starts outside come back
    in and never go further out."""
    out = _outside_tool_limits(tool, q)
    if not out:
        return tool
    lim = {j: [float(lo), float(hi)] for j, (lo, hi) in tool.joint_limits_deg.items()}
    for j, v, lo, hi in out:
        lim[j] = [min(lo, v - 0.01), max(hi, v + 0.01)]
    return _with_limits(tool, lim)


def _lift_by_joint(arm, tool: Tool, q, j: int, to_height: float, step_deg: float = 1.0, max_deg: float = 45.0):
    """The lift when the tip cannot go straight up: ONE joint alone (the elbow, the shoulder lift), turned the
    way that raises the tool tip, until the tip is to_height above the table. This is how the brush was got
    out on 2026-11-13 when the straight lift refused (the elbow joint, from a scratch script), and it needs no
    inverse kinematics, so it works from any pose the checks let the arm leave. The tip rises on an arc: it
    also travels sideways. (waypoints, how, sideways travel in m, the joint's turn in deg), or None when that
    joint cannot do it: a pose on the way fails check_pose, or the tip stops rising first."""
    d = np.zeros(5)
    d[j] = math.radians(step_deg)
    sgn = 1.0 if tip_height(arm, q + d, tool) >= tip_height(arm, q - d, tool) else -1.0
    wps, h_prev = [np.array(q, dtype=float)], tip_height(arm, q, tool)
    for _ in range(int(max_deg / step_deg)):
        w = wps[-1] + sgn * d
        try:
            h = check_pose(arm, tool, w, q_start=q, what=f"lift by {ARM[j]}")
            if h >= to_height:               # the last step ends at the height asked for, not beyond it
                w = wps[-1] + sgn * d * max(0.05, (to_height - h_prev) / (h - h_prev))
                h = check_pose(arm, tool, w, q_start=q, what=f"lift by {ARM[j]}")
        except SafetyError:
            return None
        if h <= h_prev:
            return None
        wps.append(w)
        h_prev = h
        if h >= to_height - 1e-4:
            tl = K.tool_offset(*tool.tip_offset)
            side = float(np.linalg.norm((K.fk(w, tl)[:3, 3] - K.fk(q, tl)[:3, 3])[:2]))
            turn = math.degrees(float(w[j] - q[j]))
            return wps, f"{ARM[j]} alone ({abs(turn):.0f} deg; the tip also moves {side * 1000:.0f} mm sideways)", side, turn
    return None


def _lift_plan(arm, tool: Tool, q, to_height: float = LIFT_HEIGHT) -> tuple:
    """(waypoints, how) of the lift lift_path describes."""
    q = np.asarray(q, dtype=float)
    tl = K.tool_offset(*tool.tip_offset)
    p = K.fk(q, tl)[:3, 3].copy()
    pitch = K.pose_summary(q, tl)["pitch_down_deg"]
    roll = math.degrees(float(q[4]))
    out = np.array([p[0] - K._X_PAN, p[1]])
    out = out / max(float(np.linalg.norm(out)), 1e-6)
    z_top = table_z(arm) + float(to_height)
    planner = _holding_start(tool, q)
    last = None
    for dp, back in ((0.0, 0.0), (-6.0, 0.01), (-12.0, 0.02), (6.0, 0.0), (12.0, 0.0)):
        top = np.array([p[0] - back * out[0], p[1] - back * out[1], z_top])
        try:
            with _as_tool(arm, planner):
                wps = arm.plan_tip_path([(top, pitch + dp, roll)], q_start=q, step_m=0.003)
            h_prev = tip_height(arm, q, tool)
            for k, w in enumerate(wps[1:], 1):
                h = check_pose(arm, tool, w, q_start=q, what=f"lift point {k}")
                if h < h_prev - 5e-4:
                    raise SafetyError("the tool tip would go down on the way up")
                h_prev = h
            return wps, ("straight up" if dp == 0.0 else f"up, the gripper tipped by {dp:+.0f} deg"
                         + (f" and the tip drawn {back * 1000:.0f} mm towards the robot" if back else ""))
        except SafetyError as e:
            last = e
    arcs = []
    for j in (2, 1):                         # the elbow first: it is what lifted the brush out that afternoon
        got = _lift_by_joint(arm, tool, q, j, float(to_height))
        if got is not None and got[2] <= LIFT_ARC_MAX_M:
            return got[0], got[1]
        if got is not None:                  # it would do it, on an arc too long to be taken without being asked
            arcs.append((got[2], ARM[j], got[3]))
    msg = (f"the tool tip is {tip_height(arm, q, tool) * 1000:.0f} mm above the table and cannot be lifted straight up "
           f"from this pose ({last})" + ("" if arcs else ", nor by the elbow or the shoulder joint alone")
           + ". Nothing moved. ")
    if arcs:
        side, joint, turn = arcs[0]
        msg += (f"One joint alone would raise it to {float(to_height) * 1000:.0f} mm, on an arc that carries the tip "
                f"{side * 1000:.0f} mm sideways (a lift takes at most {LIFT_ARC_MAX_M * 1000:.0f} mm of that upon itself): "
                f"`fd joints {joint}={turn:+.1f} --rel`, when nothing stands in the way of that arc. ")
    outside = _outside_tool_limits(tool, q, slack_deg=0.05)
    if not outside:
        raise SafetyError(msg if arcs else msg + "Lift it by hand-driven steps (`fd tip 0 0 0.01 --rel`) or support "
                                                 "the arm and use `fd torque off --force`")
    # the arm stands outside a joint limit of its tool: say which joint move brings it back (and that it was checked)
    j, v, lo, hi = outside[0]
    target = lo + 0.5 if v < lo else hi - 0.5
    back_q = q.copy()
    back_q[ARM.index(j)] = math.radians(target)
    try:
        low = _segment(arm, tool, q, back_q, None, f"{j} back inside its limit")
        way = (f"`fd joints {j}={target:.1f}` turns it back inside (checked: every pose on the way passes, the tool tip "
               f"stays {low * 1000:.0f} mm or more above the table)")
    except SafetyError as e:
        way = (f"Turning {j} back inside by itself does not pass the safety checks from here ({e}): "
               + ("raise the tip first (the joint move above); any free move then brings it back inside (`fd ready`)"
                  if arcs else "support the arm by hand and use `fd torque off --force`"))
    raise SafetyError(msg + f"{j} stands at {v:.1f} deg, outside the limit [{lo:g}, {hi:g}] deg of tool '{tool.name}' "
                            f"(was the arm moved by hand with the torque off?); a move may bring it back inside, not "
                            f"further out. {way}")


def lift_path(arm, tool: Tool, q, to_height: float = LIFT_HEIGHT) -> list:
    """Joint waypoints (starting with q) that raise the tool tip straight up to to_height above the table,
    keeping the gripper's pitch; where the arm cannot hold that pitch higher up, the gripper may tip back
    by up to 12 deg while the tip is drawn up to 2 cm towards the robot (as the pen scripts did). Every
    waypoint passes check_pose and none is lower than the one before. No motion. Raises SafetyError.

    Two things since 2026-11-13. A start pose that lies OUTSIDE a joint limit of the tool (the arm was moved
    by hand) is lifted with that joint staying where it is (_holding_start): it may come back inside on a later
    move, it never goes further out. And when the tip cannot go straight up at all (close to the base the
    wrist is at its limit), one joint alone raises it (_lift_by_joint), as long as that arc carries the tip no
    more than LIFT_ARC_MAX_M sideways. A refusal names the joint move that would free the arm: the one-joint
    arc that was too long to be taken unasked, and, from a pose outside a limit, the move back inside."""
    return _lift_plan(arm, tool, q, to_height)[0]


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
    lift, lift_via, q = None, None, q0
    if tip_height(arm, q0, tool) < LIFT_HEIGHT - 0.002:
        lift, lift_via = _lift_plan(arm, tool, q0)
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
            best = {"lift": lift, "path": pts, "via": name, "min_height_m": round(float(lo), 4), "lift_via": lift_via}
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
    if plan["lift"] and plan.get("lift_via"):
        out["lift_via"] = plan["lift_via"]
    return out


def lift(arm, to_height: float = LIFT_HEIGHT, speed_deg_s: float = 12.0, settle=0.3) -> dict:
    """Tool up: raise the tool tip straight up to to_height above the table when it is lower (lift_path),
    e.g. before a command that stops reports where the tool is. The torque must be on. Returns
    {"lifted": bool, "tip_height_m"} (+ the run_path figures and "via", how it went up, when it moved)."""
    tool = arm.tool
    if tool is None:
        raise SafetyError("lift is for an arm with a tool attached")
    q = arm.read_q()
    h0 = tip_height(arm, q, tool)
    if h0 >= float(to_height) - 0.002:
        return {"lifted": False, "tip_height_m": round(h0, 4)}
    wps, via = _lift_plan(arm, tool, q, to_height)

    def rising(t, q_plan, q_meas):
        return h0 > 0.006 and tip_height(arm, q_meas, tool) < h0 - 0.004

    r = arm.run_path(wps, speed_deg_s, settle=settle, monitor=rising)
    return {"lifted": True, "tip_height_m": round(tip_height(arm, arm.read_q(), tool), 4),
            "joint_err_deg": r["joint_err_deg"], "duration_s": r["duration_s"], "via": via}


def _power(arm, tier):
    """Torque on if needed (the arm holds where it is, the gripper keeps the position it has) and the
    speed tier's servo profile; returns the tier's speeds.

    A tool whose jaw is held closed by a squeeze (Tool.gripper_goal_raw) loses that squeeze whenever the
    torque was off: it is put back here (hold_jaw), so that whatever is powered through this function (the
    stand-by pose, the poses of the air calibration) has the tool as rigid as it was measured. The speeds
    then carry "jaw": what hold_jaw did. A jaw that stands elsewhere than the tool file says, held by a
    squeeze or not (jaw_in_place), does NOT stop the caller: this is also how the arm is folded away. "jaw"
    then has the refusal as "error"; whoever needs the tool where its tip was measured asks jaw_in_place
    before the torque comes on, or calls hold_jaw and lets it refuse (go_ready, BrushRun.power). A tool
    without a hold whose jaw stands where the file has it has nothing to report: the speeds are the arm's
    own dict, as for a tool without a jaw record."""
    if not arm.is_torqued():
        arm.torque_on()
        getattr(arm, "_sleep", time.sleep)(0.3)  # the servos take the arm's weight over before heights are judged
    sp = arm.apply_profile(tier)
    tool = getattr(arm, "tool", None)
    if jaw_tolerance(tool) is None:
        return sp
    try:
        jaw = hold_jaw(arm, tool)
    except SafetyError as e:
        jaw = {"held": False, "wrote": False, "error": str(e)}
    return dict(sp, jaw=jaw) if jaw.get("held") or "error" in jaw else sp


def go_ready(arm, tier: str | None = None) -> dict:
    """`ready` with a tool attached (skills.go_ready hands over to this): the TOOL tip goes to the
    stand-by point (ready_q) by free_move. The whole way is planned and checked before the torque goes on,
    so a refusal leaves the arm exactly as it was. That includes the jaw: when it does not stand where the
    tool file has it, the tool tip is not where the plan puts it (jaw_in_place: every tool that records a
    jaw position, held by a squeeze or not, except one recorded on the fixed jaw)."""
    tool = arm.tool
    arm.speeds(tier)  # an unknown / unvalidated tier refuses first
    pose = ready_pose(arm, tool)
    plan = plan_free_move(arm, tool, arm.read_q(), pose["q"])
    jaw_in_place(arm, tool)
    sp = _power(arm, tier)
    out = free_move(arm, pose["q"], sp["transit"], settle=0.5, lift_speed_deg_s=sp["approach"], plan=plan)
    out.update(tool_tip_xyz=[round(float(v), 4) for v in K.fk(arm.read_q(), arm.tip_tool())[:3, 3]],
               standby_xyz=pose["xyz"], standby_pitch_deg=pose["pitch"], standby_moved_m=pose["moved_m"])
    if "jaw" in sp:
        out["jaw"] = sp["jaw"]
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


def torque_off_block(arm, fold: str = "fd tool rest") -> str | None:
    """Why torque off must be refused with the tool that is on the arm, or None when it is fine: nothing
    holds the arm up anyway, or the arm is in the tool's rest pose. fold: the command that folds the arm away
    with this tool (the caller names `fd paint rest` for a painting tool beside a painting set-up: next_locked)."""
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
            f"and the tool would hit the table. Use `{fold}` (folds the arm away with the tool, then "
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
    if "jaw" in sp:      # (a jaw that is not where the tool file has it is told, and the arm is folded away all the same)
        out["jaw"] = sp["jaw"]
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


def next_locked(fold: str | None = None, detach: str | None = None) -> str:
    """What to do instead of a command that a tool on the gripper locks. fold: the command that folds the arm
    away with this tool on, when it is not `fd tool rest`: the caller names `fd paint rest` for a painting tool
    with a painting set-up on record (fd._fold_words: the tool's own fold has a brush over the canvas there, and
    `fd tool rest` is refused); detach: the spelling of the detach command that goes with it (`fd --sim tool
    detach` after a simulated call)."""
    if fold is None:
        return NEXT_LOCKED
    return (f"`{fold}` folds the arm away with the tool on, clear of the canvas of the painting set-up (torque off); once "
            f"the tool is off the gripper, `{detach or 'fd tool detach'}` unlocks the gripper and `park`")


def lock_message(tool: Tool, what: str, fold: str | None = None, detach: str | None = None) -> str:
    """Text of the SafetyError for something that must not happen while a tool is attached (fold, detach:
    next_locked)."""
    msg = (f"tool '{tool.name}' is attached to the gripper: {what} is refused (the gripper must not move and "
           f"the arm must not fold as if nothing were taped to it). Nothing moved. {next_locked(fold, detach)}")
    if getattr(tool, "on_jaw", None) == "moving_outer":     # the one tool whose jaw has a command of its own
        # (one command per pair of backticks: close|open in one pair, typed as it was printed, is a shell pipe; and
        # no placeholder inside them: `fd tool jaw open --to DEG`, typed as it was printed, is refused)
        msg += (". The jaw of this tool (taped to the outer face of the moving jaw) is moved with `fd tool jaw close` "
                "(`fd tool jaw help` lists the words for opening it or closing it to an angle)")
    return msg


def refusal(tool: Tool, what: str, fold: str | None = None, detach: str | None = None) -> dict:
    """The complete result of a command that is refused because a tool is attached (fold, detach: next_locked)."""
    msg = lock_message(tool, what, fold, detach)
    return {"ok": False, "error": "SafetyError", "stage": "tool", "reason": msg, "message": msg, "tool": tool.name,
            "next": next_locked(fold, detach),
            "say": f"グリッパーにツール「{tool.name}」が付いているので、{what} は実行しませんでした（アームは動かしていません）。"
                   + ("片付けるときは fd tool rest、ツールを外したあとは fd tool detach を使ってください。" if fold is None else
                      "絵のセットアップが登録されているので、片付けるときは fd paint rest を使ってください（fd tool rest は"
                      "キャンバスの上に畳むため断られます）。ツールを外したあとは fd tool detach です。")}


# ----------------------------------------------------------------------------- the tip of a tool that changed
def tip_changed(tool: Tool) -> list:
    """The tool's tip is no longer where it was measured (its tip offset was changed, or the jaw that carries
    it moved): drop what was measured with the old tip, exactly as `fd tool set --tip` does for the pen
    calibration (contact plane, correction map, the records in `calibrated`), and the air measurement of the
    tip (tip_measured). Returns the names of what was dropped."""
    dropped = [k for k, v in (("tip_measured", tool.tip_measured), ("contact", tool.contact), ("map", tool.corr_map),
                              ("calibrated", tool.calibrated)) if v]
    tool.tip_measured, tool.contact, tool.corr_map, tool.calibrated = None, None, None, {}
    return dropped


def set_measured_tip(tool: Tool, tip_offset, record: dict) -> list:
    """Give the tool a MEASURED tip (the air calibration): the new offset first, which drops what was measured
    with the old one when it differs (tip_changed), then the record of the measurement. Returns what was
    dropped. (Assigning tool.tip_measured before tool.tip_offset would lose the record: see Tool.__setattr__.)"""
    if not isinstance(record, dict):
        raise SafetyError(f"the record of a tip measurement must be a dict (got {record!r})")
    tip = check_offset(tip_offset)
    dropped = [] if _same_tip(tool.tip_offset, tip) else tip_changed(tool)
    tool.tip_offset = [round(float(v), 5) for v in tip]
    tool.tip_measured = dict(record)
    return dropped


# ----------------------------------------------------------------------------- the jaw of a tool on its outer face
def _cfg(cfg: dict, key: str):
    """Config value, with the built-in default for a key an older config dict lacks (as Arm._c)."""
    return cfg[key] if key in cfg else DEFAULT_CONFIG[key]


def jaw_q_deg(cfg: dict, raw) -> float:
    """URDF jaw angle (deg) of a raw gripper position, without an arm (the numbers of Arm.gripper_q_deg)."""
    return GM.q_from_raw(float(raw), float(_cfg(cfg, "gripper_touch_raw")))


def jaw_range_raw(cfg: dict) -> tuple:
    """(lo, hi): the gripper positions the harness commands on a bare gripper (Arm._gripper_clamp): from where
    the fingertips touch to the opened jaw. `fd tool jaw close --to DEG` and `open --to DEG` stay inside it; only
    the search for the fixed jaw and the squeeze go below lo."""
    g = _cfg(cfg, "gripper")
    return (max(int(g["closed_raw"]) + int(g["margin"]), int(_cfg(cfg, "gripper_touch_raw"))),
            int(g["open_raw"]) - int(g["margin"]))


def jaw_locked(tool: Tool | None, what: str = "moving the jaw") -> str | None:
    """Why the jaw must NOT be moved with this tool on the gripper (the text of the refusal), or None when it
    may: the tool is recorded as taped to the OUTER face of the moving jaw. Then nothing is between the jaws,
    and closing them crushes nothing. For a tool between the jaws, on the fixed jaw, or of which the file does
    not say where it is, the gripper stays locked; the text says what to record first."""
    if tool is None:
        return (f"no tool is attached: {what} with the tool jaw commands is for a tool taped to the outer face of the "
                "moving jaw. Nothing moved. The bare gripper is moved with `fd grip 0.5` (half open; 0 = closed, 1 = open) "
                "and `fd grasp`")
    on = getattr(tool, "on_jaw", None)
    if on == "moving_outer":
        return None
    record = ("look at a photo in which the gripper and the tool are both visible, then record where the tool is: "
              "`fd tool jaw mount moving_outer` (the outer face of the moving jaw: the jaw may then be moved), "
              "`fd tool jaw mount fixed` or `fd tool jaw mount between`")
    if on == "between":
        return (f"tool '{tool.name}' is recorded as held BETWEEN the jaws (on_jaw = between): {what} is refused, because "
                "closing the jaw would crush the tool and opening it would drop it. Nothing moved. The jaw is moved "
                f"only for a tool taped to the OUTER face of the moving jaw. If the record is wrong, {record}")
    if on == "fixed":
        return (f"tool '{tool.name}' is recorded as taped to the FIXED jaw (on_jaw = fixed): {what} is refused: the tool "
                "does not move with the jaw, and a jaw that closes on its tape or its body shifts it. Nothing moved. "
                f"The jaw is moved only for a tool taped to the OUTER face of the moving jaw. If the record is wrong, {record}")
    return (f"tool '{tool.name}': the tool file does not say on which jaw the tool is taped (on_jaw is not recorded): "
            f"{what} is refused, because a tool held BETWEEN the jaws would be crushed by closing them and dropped by "
            f"opening them. Nothing moved. First {record}")


def jaw_tip_estimate(tip_offset, q_from_deg: float, q_to_deg: float) -> list:
    """Where the tip of a tool taped to the MOVING jaw is once the jaw has turned from q_from_deg to q_to_deg
    (URDF jaw angles): the tip taken as rigid with the jaw and turned about the jaw's pivot axis
    (gripper_model.PIVOT_X / PIVOT_Z; y, along that axis, stays). Metres in gripper_frame_link axes.

    AN ESTIMATE FOR THE OPERATOR, never a measurement. 2026-11-13: tip (-5.1, -2.5, 60.4) mm with the jaw at
    0.2 deg; closed to -12.2 deg the rotation gives (23.4, -2.5, 52.3) mm, the air measurement found
    (16.0, -1.1, 55.8) mm: 8 mm off (JAW_ESTIMATE_ERR_M). Under the squeeze the printed jaw gives way and the
    brush, lying on the fixed jaw, is pushed back in its tape; the model knows neither."""
    x, y, z = (float(v) * 1000.0 for v in check_offset(tip_offset))
    xe, ze = GM.jaw_point(float(q_to_deg) - float(q_from_deg), (x, z))
    return [round(xe / 1000.0, 5), round(y / 1000.0, 5), round(ze / 1000.0, 5)]


def jaw_estimate(cfg: dict, tool: Tool, to_raw) -> dict | None:
    """What the jaw model says about the tool's tip once the jaw stands at to_raw: {"tip_offset" (m),
    "tip_offset_mm", "from_raw", "from_q_deg", "to_raw", "to_q_deg", "turned_deg", "tip_moves_mm",
    "expect_error_mm"}. None when it cannot say: the tool file does not record where the jaw stood for the
    tip it holds (gripper_raw), or the turned tip would be no tip the file accepts."""
    if tool.gripper_raw is None:
        return None
    q0, q1 = jaw_q_deg(cfg, tool.gripper_raw), jaw_q_deg(cfg, to_raw)
    try:
        tip = check_offset(jaw_tip_estimate(tool.tip_offset, q0, q1))
    except SafetyError:
        return None
    return {"tip_offset": tip, "tip_offset_mm": [round(v * 1000.0, 1) for v in tip],
            "from_raw": int(tool.gripper_raw), "from_q_deg": round(q0, 2), "to_raw": int(to_raw), "to_q_deg": round(q1, 2),
            "turned_deg": round(q1 - q0, 2),
            "tip_moves_mm": round(float(np.linalg.norm(np.asarray(tip) - np.asarray(tool.tip_offset, dtype=float))) * 1000.0, 1),
            "expect_error_mm": round(JAW_ESTIMATE_ERR_M * 1000.0, 1)}


def _whole(v, what: str, lo: int, hi: int) -> int:
    try:
        f = float(v)
    except (TypeError, ValueError):
        f = float("nan")
    if isinstance(v, bool) or not (math.isfinite(f) and f == int(f) and lo <= f <= hi):
        raise SafetyError(f"{what} must be a whole number {lo} .. {hi} (got {v!r})")
    return int(f)


def jaw_plan(cfg: dict, tool: Tool | None, direction: str = "close", to="closed", squeeze_ticks=None, step_ticks=None,
             lag_ticks=None, pos_raw=None, servo_limits=None) -> dict:
    """Everything `fd tool jaw close` / `fd tool jaw open` decides before the jaw moves, checked (SafetyError otherwise); the dry
    run shows exactly this. No bus traffic.

    direction "close" | "open"; to: "closed" (close only: until the jaw meets the fixed jaw, then hold it with
    squeeze_ticks) or a URDF jaw angle in degrees inside jaw_range_raw. squeeze_ticks: default
    JAW_SQUEEZE_TICKS, never more than cfg["squeeze_ticks"] (the holding offset the harness allows after a
    grasp). step_ticks 1 .. JAW_STEP_TICKS, lag_ticks 10 .. 40. pos_raw (where the jaw stands, when known):
    `close` then refuses to open and `open` to close. servo_limits: the gripper servo's own (min, max).
    Returns {"direction", "to", "target_raw" (None: until the fixed jaw), "squeeze_ticks", "floor_raw" (the lowest
    goal of the search for the fixed jaw), "step_ticks", "lag_ticks", "touch_raw", "expect_raw" (where the jaw is
    expected to end)} (+ "from_raw", "steps", "seconds" with pos_raw)."""
    if direction not in ("close", "open"):
        raise SafetyError(f"the jaw is moved with `close` or `open` (got {direction!r})")
    why = jaw_locked(tool, "closing the jaw" if direction == "close" else "opening the jaw")
    if why:
        raise SafetyError(why)
    step = JAW_STEP_TICKS if step_ticks is None else _whole(step_ticks, "the jaw step (--step, ticks)", 1, JAW_STEP_TICKS)
    lag = JAW_LAG_TICKS if lag_ticks is None else _whole(lag_ticks, "the lag that stops the jaw (--lag, ticks)", 10, 40)
    touch = int(_cfg(cfg, "gripper_touch_raw"))
    lo, hi = jaw_range_raw(cfg)
    g = _cfg(cfg, "gripper")
    s_lo, s_hi = (g["closed_raw"], g["open_raw"]) if servo_limits is None else servo_limits
    s_lo, s_hi = int(s_lo), int(s_hi)
    plan = {"direction": direction, "step_ticks": step, "lag_ticks": lag, "touch_raw": touch}
    if isinstance(to, str) and to.strip().lower() == "closed":
        if direction != "close":
            raise SafetyError("fd tool jaw open needs the angle to open to: --to DEG (0 = jaws parallel)")
        limit = min(int(_cfg(cfg, "squeeze_ticks")), JAW_SQUEEZE_CAP_TICKS)
        if squeeze_ticks is None:
            squeeze = min(JAW_SQUEEZE_TICKS, limit)
        else:
            squeeze = _whole(squeeze_ticks, "the squeeze (--squeeze, ticks)", 0, 4095)
            if squeeze > limit:
                raise SafetyError(f"a squeeze of {squeeze} ticks is refused: the holding offset this harness allows is "
                                  f"{limit} ticks (config \"squeeze_ticks\": 60 sits on the gripper's overload threshold); "
                                  f"{JAW_SQUEEZE_TICKS} held the brush for 90 minutes. Nothing moved")
        # where it is expected to end: where the fingertips touch; or, when the tool file has the jaw closed on the
        # fixed jaw already (held there, or recorded beyond the touch position), exactly there
        raw = getattr(tool, "gripper_raw", None)
        known = raw is not None and (getattr(tool, "gripper_goal_raw", None) is not None or int(raw) <= touch)
        plan.update(to="closed", target_raw=None, squeeze_ticks=squeeze, expect_raw=int(raw) if known else touch,
                    floor_raw=max(s_lo, touch - JAW_BELOW_TOUCH_TICKS))
    else:
        try:
            deg = float(to)
        except (TypeError, ValueError):
            deg = float("nan")
        if not math.isfinite(deg):
            raise SafetyError(f"--to {to!r}: a jaw angle in degrees (0 = jaws parallel), or `closed`")
        if squeeze_ticks is not None:
            raise SafetyError("--squeeze belongs to `close --to closed`: a jaw that stops at an angle presses on nothing")
        target = GM.raw_from_q(deg, touch)
        if not lo <= target <= hi or not s_lo <= target <= s_hi:
            raise SafetyError(f"--to {deg:g}: the jaw goes from {jaw_q_deg(cfg, lo):.1f} deg (the fingertips touch) to "
                              f"{jaw_q_deg(cfg, hi):.1f} deg; `fd tool jaw close --to closed` closes it against the "
                              "fixed jaw and holds it")
        plan.update(to=round(deg, 2), target_raw=int(target), squeeze_ticks=None, expect_raw=int(target), floor_raw=None)
    if pos_raw is not None:
        pos = int(pos_raw)
        if plan["target_raw"] is not None:
            if direction == "close" and plan["target_raw"] > pos + JAW_SAME_TICKS:
                raise SafetyError(f"`close --to {plan['to']:g}` would OPEN the jaw: it stands at {jaw_q_deg(cfg, pos):.1f} deg "
                                  f"(raw {pos}). Nothing moved. `fd tool jaw open --to {plan['to']:g}` opens it")
            if direction == "open" and plan["target_raw"] < pos - JAW_SAME_TICKS:
                raise SafetyError(f"`open --to {plan['to']:g}` would CLOSE the jaw: it stands at {jaw_q_deg(cfg, pos):.1f} deg "
                                  f"(raw {pos}). Nothing moved. `fd tool jaw close --to {plan['to']:g}` closes it")
        if plan["target_raw"] is None and plan["expect_raw"] == touch:
            plan["expect_raw"] = min(touch, pos)                   # (a jaw that is closed already will not open to get there)
        steps = int(math.ceil(abs(pos - plan["expect_raw"]) / step)) + (6 if plan["target_raw"] is None else 0)
        plan.update(from_raw=pos, steps=steps, seconds=round(steps * JAW_STEP_S + 1.5, 1))    # (about)
    return plan


def jaw_swing(cfg: dict, tool: Tool, q, from_raw, to_raw, min_height: float = MIN_FREE_HEIGHT) -> dict:
    """What a jaw move from from_raw to to_raw does to the tool tip while the ARM stands still at pose q, by the
    jaw model: the tool swings with the jaw (closing from 35 deg carried a tip 9 cm across), and it swings
    towards the fixed jaw, the side that points DOWN when the gripper is level. The scripts of 2026-11-13 left
    that to the person watching. Here the tip must be min_height above the table at every jaw angle on the way
    (jaw_tip_estimate every 4 deg; the estimate is good to about a centimetre, so 2 cm keep the real tip off
    the table) and must not enter a keep-out. SafetyError otherwise, with the way out (raise the arm first).
    A tool file that does not say where the jaw stood for its tip is read as "where it stands now".
    Returns {"tip_height_mm": {"now", "lowest", "end"}, "turn_deg"}."""
    tz = float(cfg.get("table_z", 0.0))
    a, b = jaw_q_deg(cfg, from_raw), jaw_q_deg(cfg, to_raw)
    q_rec = a if tool.gripper_raw is None else jaw_q_deg(cfg, tool.gripper_raw)
    n = max(1, int(math.ceil(abs(b - a) / 4.0)))
    # (the refusals below name the jaw command in words: this function does not know the options it was typed with,
    # and `fd tool jaw open` without its --to is refused when it is typed; the caller's next spells the whole command)
    doing, cmd = ("closing", "close") if b < a else ("opening", "open")
    q = np.asarray(q, dtype=float)
    hs = []
    for k in range(n + 1):
        try:
            tip = check_offset(jaw_tip_estimate(tool.tip_offset, q_rec, a + (b - a) * k / n))
        except SafetyError:
            raise SafetyError(f"{doing} the jaw to {b:.1f} deg would carry the tip of tool '{tool.name}' more than "
                              f"{MAX_OFFSET_M * 100:.0f} cm from the gripper (by the jaw model): the recorded tip offset "
                              "or jaw position cannot be right (`fd tool jaw status`). Nothing moved") from None
        p = K.fk(q, K.tool_offset(*tip))[:3, 3]
        h = float(p[2]) - tz
        hs.append(h)
        if h < min_height:
            if k == 0:
                raise SafetyError(f"the tip of tool '{tool.name}' is {h * 1000:.0f} mm above the table: the jaw is not "
                                  f"moved with the tool that low (the tool swings with it; the limit is "
                                  f"{min_height * 1000:.0f} mm). Nothing moved. Raise the arm first: `fd ready`, then "
                                  f"the same jaw command ({cmd}) again")
            raise SafetyError(f"with the arm where it stands, {doing} the jaw to {b:.1f} deg would swing the tip of tool "
                              f"'{tool.name}' down to {h * 1000:.0f} mm above the table (by the jaw model, which is good to "
                              f"about {JAW_ESTIMATE_ERR_M * 1000:.0f} mm; the limit is {min_height * 1000:.0f} mm). Nothing "
                              f"moved. Raise the arm first: `fd ready`, then the same jaw command ({cmd}) again")
        name = _keepout_of(cfg, p)
        if name and k > 0:
            raise SafetyError(f"with the arm where it stands, {doing} the jaw to {b:.1f} deg would swing the tip of tool "
                              f"'{tool.name}' into keep-out '{name}' (by the jaw model). Nothing moved. Move the arm "
                              f"clear of it first: `fd ready`, then the same jaw command ({cmd}) again")
    return {"tip_height_mm": {"now": round(hs[0] * 1000.0, 1), "lowest": round(min(hs) * 1000.0, 1),
                              "end": round(hs[-1] * 1000.0, 1)}, "turn_deg": round(b - a, 1)}


def _jaw_power(arm) -> bool:
    """The gripper servo on, with the in-servo caps of the active speed tier. Only the gripper: an arm that lies
    limp in its rest pose stays limp. bus.torque_on is idempotent, so a jaw that is holding keeps its goal.
    True when the torque was off."""
    prof = arm._profile()
    res = arm.bus.torque_on(("gripper",), velocity=prof["gripper"][0], acceleration=prof["gripper"][1])
    newly = isinstance(res, dict) and "gripper" in res.get("newly_enabled", [])
    if newly:
        getattr(arm, "_sleep", time.sleep)(0.2)
    return newly


def _jaw_release(arm):
    """Take the push off the jaw: goal := where it stands. Returns that position (None when the gripper
    torque is off: nothing pushes then, and a goal must not be written)."""
    if arm.bus.read("gripper", "Torque_Enable") != 1:
        return None
    pos = int(arm.bus.read("gripper", "Present_Position"))
    arm.jaw_goal(pos)
    return pos


def _jaw_fault(arm, status: int, pos: int):
    """A fault bit of the gripper servo while the jaw is walked: stop pushing, then MotionAborted (as
    Arm._gripper_fault does for the bare gripper, whose goal write a tool refuses)."""
    if status & int(arm._c("fault_abort_mask")):
        try:
            _jaw_release(arm)
        except SafetyError:
            pass
        raise MotionAborted(f"gripper servo fault: {', '.join(arm._fault_names(status))} (status {status}) with the jaw "
                            f"at raw {pos}; its goal was set to where it stands",
                            info={"reason": "servo_fault", "faults": {"gripper": status}})


def _jaw_walk(arm, start_raw: int, end_raw: int, step: int, lag: int, expect_stop=None) -> dict:
    """Walk the jaw goal from start_raw (where the jaw stands) to end_raw: one step of at most `step` ticks
    (JAW_NEAR_STEP_TICKS while closing near the fixed jaw) every JAW_STEP_S, the position read after each.

    The walk stops where the jaw no longer follows: it stands (came less than JAW_STILL_TICKS in the step)
    more than `lag` ticks behind where it should be, or is more than JAW_LAG_MAX_TICKS behind in any case.
    "Where it should be" is the goal; with expect_stop (hold_jaw: the position at which the jaw lies on the
    fixed jaw) the goal only as far as that position, so that the squeeze beyond it is not taken for a jaw
    that fails to follow. Returns {"goal", "pos", "load", "status", "stopped", "lag", "worst_lag", "steps"}."""
    sleep = getattr(arm, "_sleep", time.sleep)
    touch = int(arm._c("gripper_touch_raw"))
    sgn = -1 if end_raw < start_raw else 1
    goal = pos = int(start_raw)
    out = {"goal": goal, "pos": pos, "load": 0, "status": 0, "stopped": False, "lag": 0, "worst_lag": 0, "steps": 0}
    while goal != end_raw:
        d = min(step, JAW_NEAR_STEP_TICKS) if (sgn < 0 and goal <= touch + JAW_NEAR_TICKS) else step
        goal = max(int(end_raw), goal - d) if sgn < 0 else min(int(end_raw), goal + d)
        arm.jaw_goal(goal)
        sleep(JAW_STEP_S)
        new, load, status = arm._gripper_read()
        _jaw_fault(arm, status, new)
        should = goal if expect_stop is None else (max(goal, int(expect_stop)) if sgn < 0 else min(goal, int(expect_stop)))
        behind, came = -sgn * (new - should), sgn * (new - pos)
        pos = int(new)
        out.update(goal=goal, pos=pos, load=load, status=status, lag=int(behind), steps=out["steps"] + 1,
                   worst_lag=max(out["worst_lag"], int(behind)))
        if behind > JAW_LAG_MAX_TICKS or (behind > lag and came < JAW_STILL_TICKS):
            out["stopped"] = True
            break
    return out


def jaw_moved(cfg: dict, tool: Tool, pos_raw: int, goal_raw=None, aborted: bool = False) -> dict:
    """Bookkeeping after the jaw of a tool on its outer face was commanded: the jaw stands at pos_raw and is
    held by goal_raw (None: no squeeze). Changes `tool` in memory (the caller stores it: save()).

    A jaw that ends within JAW_SAME_TICKS of where the file has it did not move the tool (closed again on the
    fixed jaw it comes back to the tick): only the holding goal is recorded, the tip and what was measured
    with it stay. (aborted=True, a walk that was cut short: the recorded goal stays too; hold_jaw puts it
    back.) Otherwise the tool has MOVED with the jaw: gripper_raw and the goal are recorded, the tip becomes
    the jaw model's estimate (jaw_estimate; the old tip stays when there is none), and tip_measured and the
    pen calibration are dropped as for any changed tip (tip_changed). Returns {"tool_moved", "moved_ticks",
    "gripper_raw", "gripper_goal_raw", "tip_offset_before", "estimate", "dropped"}."""
    pos = int(pos_raw)
    goal = None if goal_raw is None else int(goal_raw)
    before, old_raw = [round(float(v), 5) for v in tool.tip_offset], tool.gripper_raw
    rec = {"tool_moved": False, "moved_ticks": None if old_raw is None else pos - int(old_raw), "gripper_raw": old_raw,
           "gripper_goal_raw": goal, "tip_offset_before": before, "estimate": None, "dropped": []}
    if old_raw is not None and abs(pos - int(old_raw)) <= JAW_SAME_TICKS:
        if goal is not None and not 0 < int(old_raw) - goal <= JAW_SQUEEZE_CAP_TICKS:
            goal = None       # (cannot be a squeeze on the recorded position: record none rather than a wrong one)
        if not aborted:
            tool.gripper_goal_raw = goal
        rec["gripper_goal_raw"] = tool.gripper_goal_raw
        return rec
    est = jaw_estimate(cfg, tool, pos)
    rec.update(tool_moved=True, gripper_raw=pos, estimate=est, dropped=tip_changed(tool))
    stamp = time.strftime("%Y-%m-%d %H:%M")
    if est is not None:
        tool.tip_offset = list(est["tip_offset"])
        was = [round(v * 1000.0, 1) for v in before]
        tool.notes.append(f"{stamp}: the jaw moved from raw {est['from_raw']} ({est['from_q_deg']:.1f} deg) to raw {pos} "
                          f"({est['to_q_deg']:.1f} deg) with the tool on its outer face; tip {was} mm -> "
                          f"{est['tip_offset_mm']} mm by the jaw model (an ESTIMATE, about {JAW_ESTIMATE_ERR_M * 1000:.0f} mm "
                          "off on 2026-11-13): to be measured, `fd calib-air --tip-only --save`")
    else:
        tool.notes.append(f"{stamp}: the jaw moved to raw {pos} ({jaw_q_deg(cfg, pos):.1f} deg) with the tool on its outer "
                          "face; where the jaw stood for the recorded tip is not known, so the tip offset is NOT "
                          "corrected: to be measured, `fd calib-air --tip-only --save`")
    if goal is not None and not 0 < pos - goal <= JAW_SQUEEZE_CAP_TICKS:
        goal = None
    tool.gripper_raw, tool.gripper_goal_raw = pos, goal
    rec["gripper_goal_raw"] = goal
    return rec


def jaw_lost(cfg: dict, tool: Tool, last_seen_raw=None) -> dict:
    """Bookkeeping when the jaw of a tool was being moved and where it stands can no longer be read: the servo
    bus stopped answering in the middle of `fd tool jaw close` / `fd tool jaw open`. Changes `tool` in memory (the caller
    stores it: save()).

    Review of 2026-11-13: in that case the tool file went on saying that the jaw was where it had been and
    that the tip was measured (the jaw really stood 75 ticks further, the tip about 1.5 cm from the file's).
    Nobody knows where the tool is now, and the file says so: gripper_raw and gripper_goal_raw are removed
    (unknown; no squeeze is put back, no jaw position is compared: jaw_tolerance), tip_measured and the pen
    calibration are dropped as for any changed tip (tip_changed), and a note says why. The tip offset itself
    stays, for want of a better one: it is the tip of the jaw position before the move, and the note says
    that too. last_seen_raw: the last position the walk read (arm._jaw_raw), for the note only.
    Returns {"tool_moved": None (not known), "moved_ticks": None, "gripper_raw": None, "gripper_goal_raw": None,
    "tip_offset_before", "estimate": None, "dropped", "last_seen_raw", "position_known": False}."""
    before, old_raw = [round(float(v), 5) for v in tool.tip_offset], tool.gripper_raw
    dropped = tip_changed(tool)
    seen = None if last_seen_raw is None else int(last_seen_raw)
    tool.notes.append(
        f"{time.strftime('%Y-%m-%d %H:%M')}: the jaw was being moved with the tool on it when the servo bus stopped "
        "answering: where the jaw stands is NOT known"
        + (f" (last read at raw {seen} = {jaw_q_deg(cfg, seen):.1f} deg" if seen is not None else " (never read during the move")
        + ("" if old_raw is None else f"; the tool file had it at raw {int(old_raw)} = {jaw_q_deg(cfg, old_raw):.1f} deg")
        + "). The recorded jaw position and holding goal were removed; the tip offset is still the one of the jaw "
          "position before the move and is NOT measured any more: `fd tool jaw status`, `fd tool jaw close`, then "
          "`fd calib-air --tip-only --save`")
    tool.gripper_raw = tool.gripper_goal_raw = None
    return {"tool_moved": None, "moved_ticks": None, "gripper_raw": None, "gripper_goal_raw": None,
            "tip_offset_before": before, "estimate": None, "dropped": dropped, "last_seen_raw": seen, "position_known": False}


def _jaw_cut_short(arm, cfg: dict, tool: Tool) -> dict:
    """The way out of a jaw walk that an error ended (a servo fault, an interrupt, a bus that stopped answering):
    take the push off, find out where the jaw stands, and make `tool` say it (jaw_moved; jaw_lost when the
    position cannot be read). The bus may be gone: nothing here raises because of it (the error that ended the
    walk is the one to report). Returns {"pos_raw" (None: could not be read), "released" (the push was taken
    off: the goal is where the jaw stands, or the gripper torque is off), "record"}."""
    left, released = None, False
    try:
        left = _jaw_release(arm)                 # goal := where it stands (None: the gripper torque is off)
        released = True
    except Exception:                            # (BusError, or whatever the dead port raises)
        pass
    if left is None:
        try:
            left = int(arm.bus.read("gripper", "Present_Position"))
        except Exception:
            left = None
    rec = (jaw_lost(cfg, tool, getattr(arm, "_jaw_raw", None)) if left is None else
           jaw_moved(cfg, tool, left, None, aborted=True))
    return {"pos_raw": left, "released": released, "record": rec}


def jaw_state(arm, tool: Tool | None = None) -> dict:
    """Where the jaw stands and whether it is held (reads only): {"pos_raw", "q_deg", "goal_raw", "torque",
    "load", "temp_c", "status", "squeeze_ticks" (goal beyond the position, towards closed), "squeeze_on",
    "from_record_ticks" (position minus the tool file's gripper_raw; None without one)}."""
    tool = getattr(arm, "tool", None) if tool is None else tool
    g = arm.bus.snapshot()["gripper"]
    arm._jaw_raw = g["pos"]
    torque = g["torque"] == 1
    beyond = int(g["pos"]) - int(g["goal"]) if torque else 0
    raw = getattr(tool, "gripper_raw", None)
    return {"pos_raw": int(g["pos"]), "q_deg": round(arm.gripper_q_deg(g["pos"]), 2), "goal_raw": int(g["goal"]),
            "torque": torque, "load": g["load"], "temp_c": g["temp"], "status": g["status"],
            "squeeze_ticks": beyond, "squeeze_on": bool(torque and beyond >= 5),
            "from_record_ticks": None if raw is None else int(g["pos"]) - int(raw)}


def move_jaw(arm, direction: str = "close", to="closed", squeeze_ticks=None, step_ticks=None, lag_ticks=None) -> dict:
    """The checked jaw move of `fd tool jaw close` / `fd tool jaw open` for the tool on the arm (arm.tool), which must be
    recorded as taped to the outer face of the moving jaw (jaw_locked). What thin_close.py and
    thin_squeeze.py did by hand on 2026-11-13, with the checks they left to the person watching:

    - everything is decided before anything is written (jaw_plan): a refusal is a SafetyError and the jaw has
      not moved; the gripper torque is switched on only then, and only the gripper's;
    - the tool swings with the jaw while the arm stands still: where the jaw model has its tip on the way must
      be clear of the table (jaw_swing);
    - the goal walks from where the jaw stands in small steps and stops where the jaw no longer follows
      (_jaw_walk);
    - to="closed": the walk goes on until the jaw meets the fixed jaw. Met within JAW_CLOSED_BAND_TICKS of
      where the fingertips touch, the jaw is held there with the squeeze (goal squeeze_ticks beyond where it
      stopped). Stopped earlier, something is in the way: the push is taken off and the result says so
      ("stopped_early"). Not met at all down to JAW_BELOW_TOUCH_TICKS below the touch position: held where
      it stands, without a squeeze;
    - to=DEG: the walk ends at that angle; a jaw that does not get there is released where it stands;
    - a servo fault, an interrupt or any other error on the way takes the push off first. When it is the servo
      bus that stopped answering, neither that nor a look at the jaw is possible: arm.tool then says that the
      jaw's position is not known (jaw_lost). The exception that ended the walk carries what was found as
      `jaw_left` ({"pos_raw", "released", "record"}; absent when no goal had been written yet).
    Afterwards arm.tool says where the jaw is and what that did to the tool (jaw_moved); the caller stores it,
    also after an exception.

    Returns {"ok", "direction", "to", "from_raw", "from_q_deg", "pos_raw", "q_deg", "goal_raw",
    "squeeze_ticks", "load", "temp_c", "status", "stopped_early", "met_fixed_jaw", "why", "worst_lag_ticks",
    "steps", "duration_s", "torque_switched_on", "record"}. Raises SafetyError (nothing moved), MotionAborted
    (servo fault; push released), BusError (the servos stopped answering: `jaw_left` says what is known)."""
    tool = getattr(arm, "tool", None)
    sleep = getattr(arm, "_sleep", time.sleep)
    clock = getattr(arm, "_clock", time.perf_counter)
    jaw_plan(arm.cfg, tool, direction, to, squeeze_ticks, step_ticks, lag_ticks)   # a locked tool, a bad option: no bus read
    if arm.raw_limits is None:
        arm.raw_limits = arm.bus.limits()
    pos0 = int(arm.bus.read("gripper", "Present_Position"))      # a look; nothing is written before the plan stands
    plan = jaw_plan(arm.cfg, tool, direction, to, squeeze_ticks, step_ticks, lag_ticks, pos_raw=pos0,
                    servo_limits=arm.raw_limits["gripper"])
    swing = jaw_swing(arm.cfg, tool, arm.read_q(), pos0, plan["expect_raw"])   # the tool swings with the jaw
    arm.health_check()
    _sim_fixed_jaw(arm, tool)
    cfg, touch, s_lo = arm.cfg, plan["touch_raw"], int(arm.raw_limits["gripper"][0])
    out = {"ok": False, "direction": direction, "to": plan["to"], "from_raw": pos0,
           "from_q_deg": round(jaw_q_deg(cfg, pos0), 2), "stopped_early": False, "met_fixed_jaw": None, "why": "",
           "swing": swing}
    t0 = clock()
    hold = None
    walking = False
    try:
        out["torque_switched_on"] = _jaw_power(arm)
        walking = True                                           # from here on a goal may have been written
        if plan["target_raw"] is None:                           # until it meets the fixed jaw
            w = _jaw_walk(arm, pos0, min(pos0, plan["floor_raw"]), plan["step_ticks"], plan["lag_ticks"])
            sleep(0.25)                                          # let it settle against what stopped it
            stop, _load, status = arm._gripper_read()
            _jaw_fault(arm, status, stop)
            if w["stopped"] and stop > touch + JAW_CLOSED_BAND_TICKS:
                _jaw_release(arm)
                out.update(stopped_early=True, met_fixed_jaw=False,
                           why=f"the jaw stopped following at raw {stop} = {jaw_q_deg(cfg, stop):.1f} deg, {stop - touch} "
                               f"ticks before the fingertips touch (raw {touch}): something is in its way. The push "
                               "was taken off; the jaw is not held")
            elif w["stopped"]:
                goal = max(s_lo, int(stop) - plan["squeeze_ticks"])
                arm.jaw_goal(goal)
                hold = goal if stop - goal > 0 else None
                out.update(ok=True, met_fixed_jaw=True)
            else:
                _jaw_release(arm)
                out.update(met_fixed_jaw=False,
                           why=f"the jaw followed down to raw {stop}, {touch - stop} ticks beyond where the fingertips "
                               f"should touch (config \"gripper_touch_raw\" {touch}), without a clear stop against the "
                               "fixed jaw: it is held where it stands, without a squeeze")
            sleep(0.8 if hold is not None else 0.3)
        else:
            w = _jaw_walk(arm, pos0, plan["target_raw"], plan["step_ticks"], plan["lag_ticks"])
            if not w["steps"]:                                   # it stands there already: the goal too, from now on
                arm.jaw_goal(plan["target_raw"])                 # (a squeeze that was on is off: this jaw is held by position)
            sleep(0.6)
            stop, _load, status = arm._gripper_read()
            _jaw_fault(arm, status, stop)
            if w["stopped"] or abs(stop - plan["target_raw"]) > 12:
                _jaw_release(arm)
                sleep(0.3)
                out.update(stopped_early=True,
                           why=f"the jaw stopped following at raw {stop} = {jaw_q_deg(cfg, stop):.1f} deg, "
                               f"{abs(stop - plan['target_raw'])} ticks before {plan['to']:g} deg (raw {plan['target_raw']}): "
                               "something is in its way. The push was taken off")
            else:
                out["ok"] = True
        pos, load, status = arm._gripper_read()
        # (the last two looks belong inside: a bus that dies here would otherwise leave the tool saying what it
        # said before the move, with the jaw closed and squeezed 141 ticks from that record)
        goal_now, temp = int(arm.bus.read("gripper", "Goal_Position")), arm.bus.read("gripper", "Present_Temperature")
    except BaseException as cut:
        # a fault, an interrupt, a bus error: no push may be left behind, and the tool file must not go on
        # saying that the jaw is where it was. When the bus itself is gone the position cannot be read: the
        # tool then says that the jaw's position is unknown (jaw_lost). What was found is handed to the caller
        # on the exception (`jaw_left`: {"pos_raw", "released", "record"}); without it nothing was commanded.
        if walking:
            cut.jaw_left = _jaw_cut_short(arm, cfg, tool)
        raise
    out.update(pos_raw=int(pos), q_deg=round(jaw_q_deg(cfg, pos), 2), goal_raw=goal_now,
               load=load, status=status, temp_c=temp,
               worst_lag_ticks=w["worst_lag"], steps=w["steps"], duration_s=round(clock() - t0, 2))
    out["squeeze_ticks"] = max(0, out["pos_raw"] - out["goal_raw"]) if hold is not None else 0
    out["record"] = jaw_moved(cfg, tool, pos, hold)
    return out


# (`--save` is what writes the record of the measured tip into the tool file: without it the painting commands
# go on refusing the tool, whatever was measured)
_JAW_AGAIN = ("Next: `fd tool jaw status` shows the jaw; `fd tool jaw close` closes it on the fixed jaw again, then "
              "`fd calib-air --tip-only --save` measures the tip and records it")


def jaw_tolerance(tool: Tool | None) -> int | None:
    """How far (ticks) the jaw may stand from the tool file's gripper_raw before the tool counts as "not where
    its tip was measured" (jaw_in_place); None when there is nothing to compare. No bus traffic.

    - A jaw that is held shut by a squeeze (gripper_goal_raw): JAW_HOLD_TOL_TICKS. The squeeze pulls the limp
      jaw back onto the fixed jaw (hold_jaw), and with the goal written it must stand within JAW_END_TOL_TICKS.
    - Any other jaw with a recorded position: JAW_END_TOL_TICKS at once. Nothing pulls it back: where it
      stands is where the tool is (a tool on the moving jaw goes with it, a tool between the jaws is held by
      it), and the tool file does not always say on which jaw the tool is: the rig's own file of 2026-11-13
      did not, with the brush on the moving jaw.
    - None: no tool; a tool recorded on the FIXED jaw (the moving jaw does not carry it); a tool file without a
      jaw position (attached with --no-robot or with the arm off, or after jaw_lost). Such a tool gets no jaw
      check at all, as before these checks existed: there is nothing to compare the jaw with, and refusing
      every tool that was attached without a look at the arm would stop work that needs no jaw (a pen along the
      fingers). What protects brush work then is the tip measurement: after jaw_lost the tip counts as not
      measured, and `fd tool jaw close` / `fd tool jaw open` (which records the jaw position) are the commands that move it."""
    if tool is None or getattr(tool, "gripper_raw", None) is None:
        return None
    if getattr(tool, "gripper_goal_raw", None) is not None:
        return JAW_HOLD_TOL_TICKS
    return None if getattr(tool, "on_jaw", None) == "fixed" else JAW_END_TOL_TICKS


def _jaw_elsewhere(arm, tool: Tool, pos: int) -> str:
    """Text of the refusal for a jaw that stands elsewhere than its tool file says. Every variant names the
    `fd tool jaw` command that is the way out for that tool."""
    raw = int(tool.gripper_raw)
    on = getattr(tool, "on_jaw", None)
    where = (f"the jaw stands at raw {pos} = {arm.gripper_q_deg(pos):.1f} deg, {abs(pos - raw)} ticks from where it was "
             f"when the tip was measured (raw {raw} = {arm.gripper_q_deg(raw):.1f} deg)")
    if tool.gripper_goal_raw is not None:
        return (f"tool '{tool.name}' is taped to the moving jaw, and {where}: the tool is no longer where its tip was "
                f"measured. Nothing was written to the gripper. {_JAW_AGAIN}")
    if on == "between":
        return (f"tool '{tool.name}' is recorded as held BETWEEN the jaws, and {where}: the jaws do not hold it as they "
                "did, so the tool is no longer where its tip was measured. Nothing was written to the gripper. Next: "
                "`fd tool jaw status` shows the jaw; set the tool back by hand and register it again with its tip as it is "
                f"then (fd tool attach with --replace, --name {shlex.quote(tool.name)} and the tip: `fd tool attach --help` "
                "lists the words), or, when the record is wrong, `fd tool jaw mount moving_outer` / `fd tool jaw mount fixed`")
    try:      # how far that carries the tip, if the tool goes with the jaw (the jaw model: good to about a centimetre)
        est = jaw_tip_estimate(tool.tip_offset, arm.gripper_q_deg(raw), arm.gripper_q_deg(pos))
        off_mm = float(np.linalg.norm(np.asarray(est) - np.asarray(tool.tip_offset, dtype=float))) * 1000.0
        far = f" (by the jaw model its tip is about {off_mm:.0f} mm from there)"
    except SafetyError:
        far = ""
    if on == "moving_outer":
        return (f"tool '{tool.name}' is taped to the moving jaw (no squeeze holds it), and {where}: the tool is no longer "
                f"where its tip was measured{far}. Nothing was written to the gripper. Next: `fd tool jaw status` shows "
                "the jaw; `fd tool jaw close` closes it on the fixed jaw and holds it (to an angle instead: `fd tool jaw "
                "help` lists the words), then `fd calib-air --tip-only --save` measures the tip and records it")
    return (f"tool '{tool.name}': {where}, and the tool file does not say on which jaw the tool is taped (on_jaw is not "
            f"recorded). On the moving jaw, or held between the jaws, the tool is no longer where its tip was measured{far}. "
            "Nothing was written to the gripper. Next: `fd tool jaw status` shows the jaw; look at a photo of the gripper, "
            "then `fd tool jaw mount moving_outer` and `fd tool jaw close` (a tool on the outer face of the moving jaw: "
            "the jaw is closed on the fixed jaw and held; then `fd calib-air --tip-only --save`), or `fd tool jaw mount fixed` "
            "(a tool on the fixed jaw: the moving jaw does not carry it)")


def jaw_in_place(arm, tool: Tool | None = None) -> dict | None:
    """Reads only: does the jaw still stand where the tool file has it (gripper_raw: the jaw position the tip
    offset belongs to)? For every tool that records a jaw position, held by a squeeze or not, except one
    recorded on the fixed jaw (jaw_tolerance). None when there is nothing to compare (no bus traffic). Else
    {"pos_raw", "from_record_ticks", "tol_ticks", "held" (a squeeze is on record)}; SafetyError when the jaw
    stands more than the tolerance from the record: the tool is no longer where its tip was measured, and a move
    planned with that tip must not start. Asked before the torque comes on (go_ready, BrushRun.power), so that a
    refusal leaves the arm exactly as it was; the text names the `fd tool jaw` command that puts it right."""
    tool = getattr(arm, "tool", None) if tool is None else tool
    tol = jaw_tolerance(tool)
    if tol is None:
        return None
    check_jaw_record(tool.on_jaw, tool.gripper_raw, tool.gripper_goal_raw)
    pos = int(arm.bus.read("gripper", "Present_Position"))
    arm._jaw_raw = pos
    if abs(pos - int(tool.gripper_raw)) > tol:
        raise SafetyError(_jaw_elsewhere(arm, tool, pos))
    return {"pos_raw": pos, "from_record_ticks": pos - int(tool.gripper_raw), "tol_ticks": tol,
            "held": tool.gripper_goal_raw is not None}


def hold_jaw(arm, tool: Tool | None = None) -> dict:
    """Put the squeeze back that holds a tool's jaw closed against the fixed jaw. The squeeze is a servo goal:
    it is lost whenever the torque was off (`fd tool rest`), and the jaw then has its gear backlash again (the
    brush of 2026-11-13 floated several millimetres in it). Call it when the torque has just come on, before a
    move that needs the tool rigid (BrushRun.power() does); the gripper torque must be on.

    Nothing to hold (no tool, or its file records no gripper_goal_raw): returns {"held": False, "wrote":
    False, "why"} and writes nothing. A tool whose file records a jaw position is still compared with it
    (jaw_in_place: one read; "pos_raw" and "from_record_ticks" are then in the answer): nothing can put this
    jaw back, so one that stands elsewhere is a refusal here too. A tool without a jaw position, or on the
    fixed jaw, is answered without touching the bus. Otherwise the jaw must stand within JAW_HOLD_TOL_TICKS of
    the recorded gripper_raw: then the recorded goal is written again, in steps (_jaw_walk), and the jaw must
    end within JAW_END_TOL_TICKS of the record. Returns {"held": True, "wrote": bool, "pos_raw", "goal_raw",
    "squeeze_ticks", "load", "from_record_ticks"}.

    SafetyError when the jaw stands elsewhere, before a goal is written, or does not come back to its place
    (the push is then taken off): the tool is no longer where its tip was measured. Next: `fd tool jaw close`,
    then `fd calib-air --tip-only --save` (for a tool without a hold: what _jaw_elsewhere names for it)."""
    tool = getattr(arm, "tool", None) if tool is None else tool
    if tool is None or getattr(tool, "gripper_goal_raw", None) is None:
        out = {"held": False, "wrote": False, "why": "no jaw hold is recorded for the tool on the gripper"}
        place = jaw_in_place(arm, tool)          # (SafetyError: the jaw stands elsewhere than its record)
        if place is not None:
            out.update(pos_raw=place["pos_raw"], from_record_ticks=place["from_record_ticks"])
        return out
    check_jaw_record(tool.on_jaw, tool.gripper_raw, tool.gripper_goal_raw)
    raw, goal = int(tool.gripper_raw), int(tool.gripper_goal_raw)
    if raw - goal > int(arm._c("squeeze_ticks")):
        raise SafetyError(f"tool '{tool.name}' records a jaw squeeze of {raw - goal} ticks; the holding offset this "
                          f"harness allows is {int(arm._c('squeeze_ticks'))} (config \"squeeze_ticks\"). Nothing was "
                          "written. `fd tool jaw close` closes and holds the jaw anew")
    again = _JAW_AGAIN
    if arm.bus.read("gripper", "Torque_Enable") != 1:
        raise SafetyError("gripper torque is off; run torque_on first (the squeeze that holds the jaw of tool "
                          f"'{tool.name}' is a servo goal)")
    pos, load, status = arm._gripper_read()
    if abs(pos - raw) > JAW_HOLD_TOL_TICKS:
        raise SafetyError(_jaw_elsewhere(arm, tool, pos))
    out = {"held": True, "wrote": False, "pos_raw": int(pos), "goal_raw": goal, "squeeze_ticks": int(pos) - goal,
           "load": load, "from_record_ticks": int(pos) - raw}
    if int(arm.bus.read("gripper", "Goal_Position")) == goal:
        return out
    _sim_fixed_jaw(arm, tool)
    try:
        # (a jaw that stands 20 ticks short of its place is not coming back: with the squeeze on top the goal
        # would then be 45 ticks ahead of it, as far as move_jaw ever lets it get)
        w = _jaw_walk(arm, int(pos), goal, JAW_STEP_TICKS, 20, expect_stop=raw)
        getattr(arm, "_sleep", time.sleep)(0.3)
        end, load, status = arm._gripper_read()
        _jaw_fault(arm, status, end)
    except BaseException:
        try:
            _jaw_release(arm)
        except Exception:
            pass
        raise
    if w["stopped"] or abs(end - raw) > JAW_END_TOL_TICKS:
        _jaw_release(arm)
        raise SafetyError(f"tool '{tool.name}': with its holding goal written again the jaw stands at raw {end}, "
                          f"{abs(end - raw)} ticks from where it was when the tip was measured (raw {raw}): "
                          + ("something keeps it open" if end > raw else "it closes further than it did")
                          + f". The push was taken off; the jaw is not held and the tool is not where its tip was "
                            f"measured. {again}")
    out.update(wrote=True, pos_raw=int(end), squeeze_ticks=int(end) - goal, load=load, from_record_ticks=int(end) - raw,
               steps=w["steps"])
    return out


# ----------------------------------------------------------------------------- simulation
def _sim_fixed_jaw(arm, tool, jaw_at=None) -> bool:
    """Simulated arm only: give the simulated gripper its FIXED jaw. SimBus has none (its jaw closes through to
    the servo's own limit), so a jaw commanded to close on the fixed jaw would neither stop nor carry a
    squeeze. For a tool on the moving jaw's outer face the simulated jaw is stopped where the tool file has it
    held (gripper_raw, when a holding goal is recorded), else where the fingertips touch
    (cfg["gripper_touch_raw"]), or where the jaw already stands (jaw_at; default: where the simulated jaw is)
    when that is further closed. An obstacle the bus already has (a test's) is left alone. True when installed."""
    bus = getattr(arm, "bus", None)
    if (getattr(tool, "on_jaw", None) != "moving_outer" or not (hasattr(bus, "blocker") and hasattr(bus, "pos"))
            or bus.blocker is not None):
        return False
    if tool.gripper_goal_raw is not None and tool.gripper_raw is not None:
        stop = float(tool.gripper_raw)
    else:
        stop = min(float(arm._c("gripper_touch_raw")), float(bus.pos["gripper"] if jaw_at is None else jaw_at))
    bus.blocker = lambda pos, _stop=stop: pos["gripper"] < _stop
    return True


def sim_place(arm, q) -> bool:
    """Simulated arm only: stand the arm at joint pose q, as if an earlier command had brought it there (every
    `fd --sim` call starts a new simulated arm, folded in its rest pose). The jaw is not touched. False, and
    nothing done, for a real bus."""
    bus = getattr(arm, "bus", None)
    if not (hasattr(bus, "pos") and hasattr(bus, "goal") and hasattr(bus, "blocker")):
        return False
    for j, v in arm.q_to_raw(q).items():
        bus.pos[j] = bus.goal[j] = float(v)
        if hasattr(bus, "sp"):
            bus.sp[j] = float(v)
    return True


def sim_start(arm) -> bool:
    """Simulated arm only: start in the tool's rest pose instead of the plain rest pose (which would put a
    6 cm tool 17 mm INSIDE the table: no real arm can be there), with the jaw where the tool file has it, and
    with a fixed jaw to close on for a tool on the moving jaw (_sim_fixed_jaw). True when the pose was set."""
    tool = getattr(arm, "tool", None)
    bus = getattr(arm, "bus", None)
    if tool is None or not (hasattr(bus, "pos") and hasattr(bus, "goal")):
        return False
    _sim_fixed_jaw(arm, tool, jaw_at=tool.gripper_raw)
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
