"""Air calibration: where the camera stands and where the tip of the attached tool is, from photos of the
tool tip held in the AIR at known joint poses.

Why this exists (2026-11-13): the camera was moved to see a whole painting set-up (paper, palette, water) and a
flat brush with a 30 cm handle was taped to the gripper. The camera model of 2026-11-11 no longer fitted, the
fingertip calibration (`fd calib-at`) is locked while a tool is on, and a brush leaves no ink to measure its tip
with before it has found the paint. What still works without touching anything: the arm holds the tool tip at
N places in the air, a photo is taken at each, and

    pixel_i = project(camera pose, FK(q_i) * tool tip offset)

is solved for the camera pose (6 numbers) AND the tool tip offset (3) together. The pan and pitch of the
poses differ, so the offset turns differently against the base in each photo: that is what makes it
observable. The focal length is pinned (camera.F_PINNED), as everywhere in this harness.

Two solves, both measured on the real arm that day:

- FULL (morning; `solve`): camera pose + tip. 61 poses (31 on a grid 7..17 cm up, then 30 at 3..6 cm), the flat
  brush's gold head found in 59 photos (`detect_tip`), 55 used, rms 8.9 px, tip +- 1.5 mm.
- TIP ONLY (evening; `solve_tip`): the camera had not moved, the brush had been swapped for a thin round one
  (dark handle, pointed orange tip 4 mm wide: `detect_tip` sees nothing of it) and was re-taped four times.
  Every bristle-coloured, elongated piece that differs from the background is a CANDIDATE (`candidates`: a
  person's arm, paint and a second brush on the desk gave wrong ones in most photos); the tip offset for
  which the most photos have a candidate near its projection wins a vote over a grid of offsets, then least
  squares on those. 33 to 36 photos, rms 6.6 to 7.7 px against the 8.9 px the camera's own fit had left,
  formal error 0.3 mm. A first attempt with one start value and outlier rejection had collapsed onto a single
  photo: hence the vote. The unknown is always the full 3-vector: the brush was taped 9 cm beside the finger
  line, then 17 cm out, before it lay along the fingers.

What such a tip is worth: right to a few millimetres IN THE IMAGE; along the camera's line of sight it carries
the bias of the camera model (the paper turned out 3 .. 9 mm above where the model said: the contact height is
measured on the paper, not here).

Look first (`look_pose`, `look_check`): the second mount was registered as "8 cm along the fingers" without
looking, and 20 poses were flown with that. The tip was 17 cm from the gripper. (Taped as it was, 12 cm beside
the finger line, the records put it 11 .. 20 cm above the table in those poses; the same length ALONG the
fingers would have been 1 .. 8 cm up, and the low set would have gone into the table.) So before any pose that
relies on the registered tip the gripper is brought into the picture in a pose that is harmless for a tool of
unknown length, one photo is taken, and the registered tip must have a candidate near its projection. One photo
sees two of the three numbers: how far the tip is along the camera's line of sight it cannot say.
The look needs a second photo, of the scene WITHOUT the tool there: by colour alone the evening's desk had 28
to 41 "tips" (the strokes of the picture, paint on scrap strips), and a registered tip that was 5 and 9 cm off
passed with the nearest of them 21 and 35 px away (review of 2026-11-13). A look without that photo is "not
checked", never "passed"; an arm that already stands at the look pose is first turned aside about its base
(`look_aside`). So is an arm
that stands anywhere the camera model sees it at the place of the look (`look_hidden`, `silhouette`): what the
arm hides in the photo "before" is revealed in the look photo, and paint among it passed for the tip of a tool
9 cm longer than registered (review of 2026-11-14); and nothing the look photo shows where the arm stood for
the photo before can make the look pass (`look_check`, q_bg).
Round 3 (2026-11-14, later): "the arm" in that drawing is the arm's links AND everything a tool of up to 30 cm
can reach from the gripper, wherever it points (`silhouette`, SIL_REACH_M): a tool whose handle lay 6 cm beside
the finger line had hidden paint outside the two lines the tool was drawn as, and passed. With this rig's camera
that leaves one place for the photo before: the look pose turned 90 deg and more about the base (`look_aside`
goes there from wherever the arm stands), and for some registrations none: the look is then refused before
anything moves. And no look is taken where the photo before shows bristle-coloured paint near the place of the
look (`look_background`): the look cannot tell a brush from paint behind it.
The turn itself (2026-11-14, evening): the base turns only with the tool LIFTED. Turned at the look pose's own
height, a tool longer than registered passed 12 .. 47 mm above the table over the rig's water dish and palette
(simulated), before the look had judged it. So the arm first raises the gripper 10 cm straight up without turning
it (`straight_lift`: every point of whatever is taped on goes up by the same 10 cm), turns its base there (no
point changes its height), and comes straight down onto the pose (`look_way`, flown by `fly_look_way`, which asks
the measured arm for the lift before and while the base turns). Where the arm cannot do that, the look is refused
before anything moves.

Stages (each its own function; nothing here writes the rig's files, `fd calib-air` (aircal_cli.py) does that):
  plan(arm, tool) / plan_points(...)      joint poses: the tool tip on a grid / at given places, all checks passed
  plan_set(arm, tool, spec)               ... for one of the pose sets that were flown that day (POSE_SETS)
  plannable(arm, tool, poses, q_from)     the poses that can be reached one after the other by free moves
  capture(arm, poses, photo, out_dir)     free moves (tooling.free_move), one photo per pose -> records (joint pose,
                                          where the jaw stood, when; it stops when a photo cannot be stored)
  detect_tip(img, background)             the far end of the FLAT brush (gold head on a blue handle) in one photo
  candidates(img, background, near)       everything that could be the tip of a tool (the thin brush's detector)
  solve(qs, uvs, tip_guess, f, cx, cy)    camera pose + tip offset (+ residuals); robust to a few bad detections
  solve_tip(qs, cands, cam)               tip offset alone, camera model fixed: vote, then least squares
  measure_full(...) / measure_tip(...)    photos -> detection -> solve, with the reason for every unused photo
  look_pose(arm, tool, cam) / look_aside(...) / look_check(img, cam, q, tip, bg, q_bg=...)
                                          the first look at a tool whose length is a guess
  straight_lift(arm, tool, q) / tool_rise(q_a, q_b) / look_way(arm, tool, q_from, q_to) / fly_look_way(arm, q_to)
                                          the ways of that look: lift straight up, turn the base, straight down; how
                                          much the point of a tool that gained least has gained between two poses
  silhouette(cam, q, tip) / look_hidden(cam, q, q_look, tip)
                                          where the camera model sees the arm and whatever a tool could reach, and
                                          whether a photo taken with the arm there shows the place of the look without
  look_background(img, cam, q_look, tip)  is the photo before clean (no bristle-coloured paint) where the look judges?

Every pose keeps the MODELLED tool tip `min_height` above the table and passes tooling.check_pose. The modelled
tip is only as good as the registered offset: that is what the look and the order of the sets (high first, the
low set planned with the tip solved from the high one) are for.

Units: metres, base frame, radians for joint vectors; pixels as (u, v).
Tests: test_aircal.py. Nothing here opens the servo bus or the camera by itself.
"""

from __future__ import annotations

import json
import math
import os
import time

import cv2
import numpy as np

import kinematics as K
import tooling as TL
from robot import MotionAborted, SafetyError

GRID_X = (0.15, 0.23, 0.31)
GRID_Y = (-0.10, 0.03, 0.16, 0.28)
GRID_H = (0.07, 0.12, 0.17)          # modelled tool tip above the table
PITCH_WANTS = (75.0, 50.0, 88.0)     # varied from pose to pose (the arm takes the nearest pitch it can hold)
MIN_HEIGHT = 0.045

# The pose sets that were flown on 2026-11-13 (heights of the MODELLED tool tip above the table). A "grid" set is
# plan() (pitch varied from place to place), a "points" set is plan_points() at the tool's working pitch with
# three heights: over the sheet, beside it (the palette), and beside it far out (the water dishes: LOW_SPLIT).
# The low sets go 3 .. 3.5 cm over the paper: fly them only with a tip that has been measured (the high set
# first; see aircal_cli).
POSE_SETS = {
    # morning, camera pose + tip: 31 of the 36 grid places could be held, all 30 low ones; rms 8.9 px
    "full": {
        "air": {"grid": (GRID_X, GRID_Y, GRID_H), "min_height": MIN_HEIGHT},
        "low": {"points": ((0.12, 0.18, 0.24, 0.30, 0.36), (-0.12, -0.04, 0.04, 0.12, 0.20, 0.28)),
                "heights": (0.03, 0.045, 0.06), "min_height": 0.02},
    },
    # evening, tip only (reference thin_air.py): 19 .. 24 of 24 grid places and 12 low ones; 90 + 55 s of arm time
    "tip": {
        "high": {"grid": ((0.17, 0.25, 0.33), (-0.10, 0.03, 0.16, 0.28), (0.08, 0.13)), "min_height": 0.06},
        "low": {"points": ((0.16, 0.24, 0.32), (-0.10, 0.02, 0.14, 0.26)),
                "heights": (0.035, 0.05, 0.06), "min_height": 0.03},
    },
}
LOW_SPLIT = (0.27, 0.10)             # (x, y): y >= 0.10 is beside the sheet, and there x >= 0.27 is the dishes

# --- candidates: the evening's values (thin round brush under the "paper" exposure: the bristles are dark
# orange, v 75 .. 100; the morning's thresholds s >= 80, v >= 80 with a later v >= 95 cut the outer half off)
BRISTLE_HUE = (172, 21)              # OpenCV hue 0 .. 179, from .. to THROUGH red: h >= 172 or h <= 21
FLAT_HEAD = {"hue": (6, 30), "s_min": 80, "v_min": 80, "area": (250, 20000), "len_min": 30.0, "width_max": 90.0}
"""candidates(**FLAT_HEAD): the gold head of the morning's flat brush (detect_tip's colour range and sizes)."""

# --- tip-only solve: the evening's values
VOTE_BOX = ((-0.12, 0.12), (-0.08, 0.08), (0.03, 0.25))   # tip offset searched, gripper_frame_link axes (m)
VOTE_STEP = 0.005
VOTE_PX = 30.0                       # a photo votes for an offset when a candidate is this near its projection
FIT_RADII = (45.0, 30.0, 25.0, 25.0)  # least squares on the nearest candidates, the radius shrinking
MIN_TIP_PHOTOS = 6                   # fewer photos agreeing on one offset is no measurement (3 unknowns, wrong
#                                      candidates in most photos; the weakest real run had 8 of 20)
MIN_FULL_PHOTOS = 8                  # camera pose + tip: 9 unknowns
HEAD_ENOUGH = 0.85                   # measure_full: detect_tip explains this share of the photos = it knows the tool
#                                      (the flat brush: 55 of 61; the thin brush, of which it sees a part: 9 .. 22 of 36)
IMAGE_SIZE = (1920, 1080)

# --- the first look
# Fingertip (x, y, height above the table, pitch below the horizontal) of the look pose, tried in this order
# (reference thin_look.py; the first is where the evening's looks were taken). All of them keep the bound
# below; the later ones stand further back and higher, for a registered tip that would leave the picture.
LOOK_POSES = ((0.28, 0.0, 0.12, 30.0), (0.26, 0.0, 0.13, 30.0), (0.24, 0.0, 0.14, 30.0), (0.22, 0.0, 0.16, 30.0),
              (0.26, 0.0, 0.14, 25.0), (0.24, 0.0, 0.16, 20.0), (0.20, 0.0, 0.18, 30.0))
LOOK_MIN_HEIGHT = 0.12               # fingertip above the table, at least
LOOK_MAX_PITCH = 30.0                # fingers below the horizontal, at most
LOOK_PX = 60.0                       # a candidate this near the registered tip's projection = the guess is usable
LOOK_EDGE_PX = 80.0                  # the registered tip must project this far inside the picture
LOOK_SPEED_DEG_S = 15.0              # what the evening's looks were driven with
# The photo "before" of an arm that already stands at the look pose: the base alone is turned by one of these
# angles (tried in this order; a turn that takes the registered tip out of the picture is preferred), until the
# camera model sees gripper and registered tip this far from where the look photo will have them. With this
# rig's camera a turn of 60 deg puts the gripper at the picture's edge and the tool outside it (600 .. 800 px).
LOOK_ASIDE_DEG = (60.0, -60.0, 75.0, -75.0, 45.0, -45.0, 90.0, -90.0, 97.0, -97.0)
#                 (since round 3 the arm must stand where NO tool of up to 30 cm could cover the place of the look
#                 (SIL_REACH_M): with this rig's camera only the last turns do that, -90 / -97 for the tools of that
#                 day; 97 is 3 deg inside the base's soft limit)
LOOK_ASIDE_PX = 500.0
# The base turns only with the tool LIFTED (2026-11-14, evening). Until then the look pose was turned to the side
# of the photo "before" and back with the fingertip 12 cm up: a tool longer than registered passed 12 .. 47 mm above
# the table over the water dish and the palette of the rig (simulated: 26 mm for a brush 9 cm longer, 24 mm for the
# mistake of 2026-11-13, 12 mm for 20 cm), before the look had judged anything. Now the arm first raises the gripper
# straight up without turning it (straight_lift), turns its base at that height, and comes straight down again
# (look_way). A lift that does not turn the gripper raises EVERY point of whatever is taped to it by the same
# amount, and a turn of the base alone changes no height: so during the turn every point of the tool is LOOK_LIFT_M
# above where it hung before the lift, hence at least that far above the table, whatever the tool's length and
# direction (it could not hang inside the table). Nothing of that depends on the registration.
LOOK_LIFT_M = 0.10                   # 4 cm clear of things 6 cm high (the dish's rim was 2.5 cm, the palette's dividers
#                                      1.5 cm) for a tool that hung ON the table; the look pose lifts 14 cm at most
LOOK_CLEAR_M = 0.08                  # what the MEASURED arm must still show of that lift while its base turns (2 cm
#                                      for sag and tracking: the measured tip ran up to 1 cm below the plan at 20 deg/s)
LOOK_TURN_FREE_DEG = 1.0             # a change of the base angle up to this is no turn (it moves a 30 cm tool 5 mm)
"""What the look pose is safe for. With the fingertip h >= 0.12 m above the table and the fingers at most 30 deg
below the horizontal (wrist roll 0: the fixed jaw is the lower one), a tool whose tip lies L from the fingertip
is above the table by at least  0.12 - L * sin(30 deg + a),  a = how far the tool leans from the finger line
TOWARDS THE FIXED JAW (down). So, with 2 cm to spare:
  - a tool along the fingers (a = 0), or leaning to the moving-jaw side or sideways: up to L = 20 cm
    (it would touch the table at 24 cm);
  - a tool leaning 30 deg towards the fixed jaw: up to 11.5 cm;
  - ANY direction, straight down included: up to 10 cm.
The tools taped on so far were 6 to 17 cm (the 17 cm one pointed up and out). `tooling.MAX_OFFSET_M` (30 cm) is
NOT covered: a tool longer than the bound has to be looked at by a person before the arm moves.
The way to the look pose is checked for the same unknown tools (unknown_tool_dip); where it turns the base, the
tool is lifted first (look_way), and THAT holds for a tool of any length."""

# What the ARM hides in the photo "before" is revealed in the look photo, and where that is paint it differs
# from the photo before and is bristle-coloured: a candidate. Simulated on the evening's desk (review of
# 2026-11-14): with the arm standing 13 .. 25 deg from the look pose, a tool registered as 8 cm and really 17 cm
# along the fingers passed the look on a strip of the lantern's red that its own handle had hidden, 16 px from
# the registered tip's place (6 of 267 such starts, 3 of 12 with the evening's second mount), and 20 poses were
# flown with the real tip down to the table. So the camera model draws the arm as it stood for the photo before
# (silhouette), and (a) a photo before is only taken where that drawing stays clear of the place of the look
# (look_hidden: else the arm is turned aside first), (b) what the look photo shows inside it cannot pass the look.
# The sizes are choices, not measurements, and generous: all they cost is a turn of the base.
SIL_ARM_M = 0.08                     # the arm's links, drawn this wide (the SO-101's are 4 .. 6 cm, plus cables)
SIL_REACH_M = TL.MAX_OFFSET_M        # the tool: EVERYTHING within this of the gripper frame, in any direction (30 cm:
#                                      what tooling accepts as a tip offset). Until round 3 the tool was two lines, along
#                                      the fingers and through the registered tip, 3 cm wide: a tool whose body lay 4 .. 6 cm
#                                      beside the finger line (tip really at (0, -60, 177) mm, registered 8 cm along the
#                                      fingers) hid the lantern's red outside that drawing, the look passed on it in 10 of
#                                      214 starts, and 20 poses were flown with the real tip at -3 .. 14 mm (simulated on
#                                      the evening's desk, 2026-11-14). Where a tool is, is what the look is about to
#                                      judge: the drawing must not depend on it
SIL_PAD_PX = 30.0                    # the drawing is grown by this (camera model 9 px rms, the arm's own calibration)
# A clean background (round 3). The look compares two photos, and what it takes for a tip is whatever differs and is
# bristle-coloured. Paint that lies near the place of the look in the photo "before" differs as soon as the arm's
# shadow or the brush itself falls on it: the look cannot tell a brush from paint behind it. So no look is taken
# where the photo before shows bristle-coloured paint (the colour test of the candidates, any shape) within
# LOOK_CLEAN_PX of the place where the registered tip will be seen: the sheet is covered with a blank one, or taken
# away. Choices, not measurements: twice the 60 px of the look; a handful of pixels is sensor noise.
LOOK_CLEAN_PX = 120.0
LOOK_CLEAN_MIN_PX = 12
LOOK_NEAR_PX = 80.0                  # the drawing this near the registered tip's place in the look photo = the arm
#                                      hides the place of the look (LOOK_PX, and 20 px for the blur of an edge)
LOOK_CAMERA_M = 0.30                 # a gripper nearer to the camera than this hides whatever its tool points at: 2 of
#                                      60 000 random poses, 20 and 24 cm from the lens, hid the place of the look behind
#                                      the handle of the evening's second mount, far from any drawing of the registered
#                                      tool (the look pose stands 36 cm from the camera, the folded arm 44 .. 62 cm)


# ----------------------------------------------------------------------------- poses
def plan(arm, tool, xs=GRID_X, ys=GRID_Y, heights=GRID_H, min_height: float = MIN_HEIGHT, roll_deg: float = 0.0) -> list:
    """Calibration poses: [{"xyz", "pitch", "q"}], ordered so that neighbours are close (rows of constant
    height, snaking in y, then x). A grid point the arm cannot hold with every check passed is left out."""
    tl, lim = K.tool_offset(*tool.tip_offset), arm.q_limits()
    tz = TL.table_z(arm)
    out = []
    for iz, h in enumerate(heights):
        level = []
        for ix, x in enumerate(xs):
            row = []
            for iy, y in enumerate(ys):
                p = np.array([float(x), float(y), tz + float(h)])
                want = PITCH_WANTS[(ix + iy + iz) % len(PITCH_WANTS)]
                pitch, _ = K.best_pitch(p, want=want, wrist_roll=math.radians(roll_deg), tool=tl, limits=lim, lo=15.0)
                if pitch is None:
                    continue
                q, info = K.ik(p, pitch, math.radians(roll_deg), tool=tl, limits=lim)
                if not info["ok"]:
                    continue
                try:
                    TL.check_pose(arm, tool, q, min_height=min_height, what="calibration pose")
                except SafetyError:
                    continue
                row.append({"xyz": [round(float(v), 4) for v in p], "pitch": round(float(pitch), 1), "q": q})
            level += row if ix % 2 == 0 else row[::-1]
        out += level if iz % 2 == 0 else level[::-1]
    return out


def plan_points(arm, tool, points, min_height: float = 0.02, roll_deg: float = 0.0) -> list:
    """Calibration poses for explicit tool tip places: points = [(x, y, height above the table, wanted pitch)],
    visited in the order given. A place the arm cannot hold with every check passed is left out."""
    tl, lim = K.tool_offset(*tool.tip_offset), arm.q_limits()
    tz = TL.table_z(arm)
    out = []
    for x, y, h, want in points:
        p = np.array([float(x), float(y), tz + float(h)])
        pitch, _ = K.best_pitch(p, want=float(want), wrist_roll=math.radians(roll_deg), tool=tl, limits=lim, lo=15.0)
        if pitch is None:
            continue
        q, info = K.ik(p, pitch, math.radians(roll_deg), tool=tl, limits=lim)
        if not info["ok"]:
            continue
        try:
            TL.check_pose(arm, tool, q, min_height=min_height, what="calibration pose")
        except SafetyError:
            continue
        out.append({"xyz": [round(float(v), 4) for v in p], "pitch": round(float(pitch), 1), "q": q})
    return out


def low_points(xs, ys, heights, pitch: float) -> list:
    """[(x, y, height, pitch)] of a low set: rows of constant x, snaking in y; heights = (over the sheet,
    beside it, beside it far out), split at LOW_SPLIT (where the palette and the dishes stood on 2026-11-13)."""
    pts = []
    for ix, x in enumerate(xs):
        row = [(float(x), float(y),
                float(heights[0]) if y < LOW_SPLIT[1] else float(heights[2] if x >= LOW_SPLIT[0] else heights[1]),
                float(pitch)) for y in ys]
        pts += row if ix % 2 == 0 else row[::-1]
    return pts


def is_low(spec: dict) -> bool:
    """A set that flies close to the table (3 .. 6 cm): planned only with a tip that has been measured."""
    return "points" in spec


def plan_set(arm, tool, spec: dict) -> list:
    """The poses of one entry of POSE_SETS for this tool (plan() or plan_points(); a low set is flown at the
    tool's working pitch, as on 2026-11-13: 75 with the flat brush, 65 with the thin one)."""
    if "grid" in spec:
        xs, ys, hs = spec["grid"]
        return plan(arm, tool, xs=xs, ys=ys, heights=hs, min_height=float(spec["min_height"]))
    xs, ys = spec["points"]
    return plan_points(arm, tool, low_points(xs, ys, spec["heights"], float(tool.pitch_deg)),
                       min_height=float(spec["min_height"]))


def plannable(arm, tool, poses, q_from) -> dict:
    """Which of the poses can be flown one after the other, starting at q_from (tooling.plan_free_move from pose
    to pose; a pose with no safe way to it is left out and the next one is planned from the last that was kept).
    No motion. -> {"poses": [kept], "refused": [{"i", "xyz", "why"}], "lowest_m": lowest MODELLED tip height on
    the ways and at the poses (None when nothing is kept), "q_end": where the arm would stand afterwards}"""
    q = np.asarray(q_from, dtype=float)
    keep, refused, lowest = [], [], None
    for i, pose in enumerate(poses):
        try:
            pl = TL.plan_free_move(arm, tool, q, pose["q"])
        except SafetyError as e:
            refused.append({"i": i, "xyz": pose["xyz"], "why": str(e)[:200]})
            continue
        lo = min(float(pl["min_height_m"]), TL.tip_height(arm, pose["q"], tool))
        lowest = lo if lowest is None else min(lowest, lo)
        keep.append(pose)
        q = np.asarray(pose["q"], dtype=float)
    return {"poses": keep, "refused": refused, "lowest_m": lowest, "q_end": q}


def store_photo(path: str, img) -> str | None:
    """Write one photo (JPEG, quality 95). None when it is on the disk, else why it is not. cv2.imwrite does not
    raise for a folder that cannot be written or a disk that is full: it returns False and leaves no file (this
    computer had 2.7 GB free, 98 % used, on 2026-11-13)."""
    try:
        if not cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            return "OpenCV could not write the file"
        if not os.path.getsize(path):
            return "the file is empty"
    except (OSError, cv2.error) as e:
        return f"{type(e).__name__}: {e}"
    return None


def jaw_reading(arm) -> dict:
    """Where the jaw stands, and where it is told to stand, read from the gripper servo: {"gripper_raw":
    Present_Position, "gripper_goal_raw": Goal_Position, or None with the gripper's torque off}. A jaw held shut
    by a squeeze has its goal some ticks beyond its position (tooling.hold_jaw: 25).

    capture() writes this into every record: a tip is the tip of the tool with the jaw where it stood for the
    photos (2026-11-13: (-5.1, -2.5, 60.4) mm at raw 2182, (16.0, -1.1, 55.8) mm closed at raw 2041), and what
    the tool FILE said at that moment is not proof of it (that evening the jaw was closed by scripts)."""
    pos = int(arm.bus.read("gripper", "Present_Position"))
    held = int(arm.bus.read("gripper", "Torque_Enable")) == 1
    return {"gripper_raw": pos, "gripper_goal_raw": int(arm.bus.read("gripper", "Goal_Position")) if held else None}


def capture(arm, poses, photo, out_dir: str, speed_deg_s: float | None = None, settle_s: float = 0.5,
            log=lambda *a: None, prefix: str = "air") -> dict:
    """Visit the poses by free moves and take one photo at each. Returns {"records": [...], "skipped": [...],
    "stopped": None | str, "interrupted": bool, "photo_error": None | str}. A pose that is refused before it
    moves (SafetyError) is skipped; a motion that is stopped on the way (MotionAborted) ends the capture: the
    arm holds where it is. An interrupt (Ctrl-C, a cancelled tool call) ends it the same way, with the records
    taken so far. So does a photo that cannot be stored ("photo_error" says why): the remaining poses would be
    three to five minutes of arm time for photos nobody can solve.

    A record: {"i", "xyz_model", "pitch", "q_cmd", "q" (measured, rad), "moved_deg", "image", "t" (time of day),
    "at" (date and time), "gripper_raw", "gripper_goal_raw" (jaw_reading)}."""
    tool = arm.tool
    if tool is None:
        raise SafetyError("air calibration needs a registered tool (fd tool attach registers one; `fd tool status` shows it)")
    os.makedirs(out_dir, exist_ok=True)
    sp = TL._power(arm, None)
    speed = min(float(sp["transit"]), 30.0) if speed_deg_s is None else float(speed_deg_s)
    records, skipped, stopped, interrupted, photo_error = [], [], None, False, None
    for i, pose in enumerate(poses):
        try:
            try:
                TL.free_move(arm, pose["q"], speed, settle=settle_s)
            except SafetyError as e:
                skipped.append({"i": i, "xyz": pose["xyz"], "why": str(e)[:200]})
                log(f"pose {i}: skipped ({str(e)[:100]})")
                continue
            except MotionAborted as e:
                stopped = f"pose {i}: {e}"
                break
            arm._sleep(0.35)
            q = arm.read_q()
            img = photo()
            path = os.path.join(out_dir, f"{prefix}_{i:02d}.jpg")
            why = store_photo(path, img)
            q2 = arm.read_q()
            jaw = jaw_reading(arm)
        except KeyboardInterrupt:
            stopped, interrupted = f"pose {i}: interrupted", True
            break
        if why:
            stopped = photo_error = f"pose {i}: the photo was taken but could not be stored as {path} ({why})"
            log(stopped)
            break
        now = time.localtime()
        rec = {"i": i, "xyz_model": pose["xyz"], "pitch": pose["pitch"], "q_cmd": [float(v) for v in pose["q"]],
               "q": [float(v) for v in 0.5 * (q + q2)], "moved_deg": round(float(np.degrees(np.max(np.abs(q2 - q)))), 3),
               "image": path, "t": time.strftime("%H:%M:%S", now), "at": time.strftime("%Y-%m-%dT%H:%M:%S", now), **jaw}
        records.append(rec)
        log(f"pose {i}: photo {os.path.basename(path)}  q_deg={np.round(np.degrees(q), 1).tolist()}")
    return {"records": records, "skipped": skipped, "stopped": stopped, "interrupted": interrupted,
            "photo_error": photo_error}


def save_records(out_dir: str, tool, result: dict, name: str = "aircal_records.json") -> str:
    """The records of one capture, with the tool they were taken with, as out_dir/name. Written to a temporary
    file first: an interrupt in the middle leaves the file that was there (or none), never half a one (`fd
    calib-air --from DIR` refuses a folder with a record file it cannot read)."""
    path = os.path.join(out_dir, name)
    tmp = path + ".tmp"
    try:
        with open(tmp, "w") as f:
            json.dump({"tool": tool.to_dict(), **result}, f, indent=1)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    return path


# ----------------------------------------------------------------------------- the brush in a photo
def background(images, rows: int = 120) -> np.ndarray:
    """Per-pixel median of the photos: the scene without the arm (it stands somewhere else in each). Computed
    in bands of `rows` image rows: the same numbers as np.median over the whole stack, without holding 61
    full photos three times over (1.1 GB on this 8 GB computer)."""
    out = np.empty(images[0].shape, dtype=np.uint8)
    for y in range(0, out.shape[0], rows):
        out[y:y + rows] = np.median(np.stack([im[y:y + rows] for im in images], axis=0), axis=0).astype(np.uint8)
    return out


def detect_tip(img, bg, diff_min: int = 26, debug: bool = False) -> dict | None:
    """The FLAT brush of the morning in one photo: {"tip": (u, v), "junction": (u, v) | None, "axis": (du, dv),
    "n_head", "how"} or None when no brush head is seen.

    "tip" is the middle of the far end of the bristles; "junction" the place where the coloured handle ends and
    the metal ferrule begins (None when the handle is not in the picture). The head (ferrule + bristles: gold /
    orange) is the largest orange piece of what differs from the background photo. Which of its two ends is the
    tip: the one away from the blue handle; with the handle cut off by the picture's edge, the end away from
    that edge; else the lower one ("how" says which rule decided).

    Not for a tool with a small tip or without a blue handle: candidates() is the detector for those."""
    hh, ww = img.shape[:2]
    diff = cv2.absdiff(img, bg).max(axis=2)
    fg = diff >= diff_min
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0].astype(int), hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    orange = ((h >= 6) & (h <= 30) & (s >= 80) & (v >= 80) & fg).astype(np.uint8)
    orange = cv2.morphologyEx(orange, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(orange, 8)
    if n < 2:
        return None
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[k, cv2.CC_STAT_AREA] < 250:
        return None
    ys, xs = np.nonzero(lab == k)
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean, vecs, vals = cv2.PCACompute2(pts, mean=None)
    c, d = mean[0].astype(float), vecs[0].astype(float)
    d = d / np.linalg.norm(d)
    if 4.0 * math.sqrt(float(vals[0, 0])) < 30.0:
        return None                                    # a blob, not a brush head
    nrm = np.array([-d[1], d[0]])
    t, lat = (pts - c) @ d, (pts - c) @ nrm
    t_lo, t_hi = float(np.percentile(t, 1.0)), float(np.percentile(t, 99.0))
    # the handle: blue, on the head's line, just beyond one of its ends
    blue = (h >= 95) & (h <= 125) & (s >= 110) & (v >= 50) & fg
    by, bx = np.nonzero(blue)
    n_lo = n_hi = 0
    if len(bx):
        bp = np.column_stack([bx, by]).astype(float)
        bt, bl = (bp - c) @ d, (bp - c) @ nrm
        on_line = np.abs(bl) < 25.0
        n_lo = int(np.sum(on_line & (bt < t_lo + 5.0) & (bt > t_lo - 80.0)))
        n_hi = int(np.sum(on_line & (bt > t_hi - 5.0) & (bt < t_hi + 80.0)))
    end_lo, end_hi = c + d * t_lo, c + d * t_hi

    def at_edge(p):
        return p[0] <= 3 or p[1] <= 3 or p[0] >= ww - 4 or p[1] >= hh - 4

    if max(n_lo, n_hi) >= 40 and min(n_lo, n_hi) < 0.3 * max(n_lo, n_hi):
        tip_hi, how = n_lo > n_hi, "handle"
    elif at_edge(end_lo) != at_edge(end_hi):
        tip_hi, how = at_edge(end_lo), "edge"
    else:
        tip_hi, how = end_hi[1] > end_lo[1], "lower end"
    sign = 1.0 if tip_hi else -1.0
    ts = sign * t
    t_tip, t_j = float(np.percentile(ts, 99.0)), float(np.percentile(ts, 1.0))
    lat_tip = float(np.median(lat[ts >= t_tip - 8.0]))
    lat_j = float(np.median(lat[ts <= t_j + 8.0]))
    tip = c + d * (sign * t_tip) + nrm * lat_tip
    junction = c + d * (sign * t_j) + nrm * lat_j
    seen_j = how == "handle" and not at_edge(junction)
    out = {"tip": (float(tip[0]), float(tip[1])), "junction": (float(junction[0]), float(junction[1])) if seen_j else None,
           "axis": (float(sign * d[0]), float(sign * d[1])), "n_head": int(len(pts)), "how": how,
           "head_len_px": round(t_tip - t_j, 1), "tip_at_edge": bool(at_edge(tip))}
    if debug:
        out["masks"] = {"fg": fg.astype(np.uint8), "orange": orange}
    return out


def candidates(img, bg=None, near=None, hue=BRISTLE_HUE, s_min: int = 110, v_min: int = 60, diff_min: int = 24,
               area=(45, 1500), len_min: float = 16.0, aspect_min: float = 2.0, width_max: float = 22.0,
               r_max: float = 700.0, close: int = 5) -> list:
    """Everything in one photo that could be the tip of a tool: the bristle-coloured, elongated pieces.
    -> [{"tip": (u, v), "tail": (u, v), "centre": (u, v), "area", "len_px", "width_px"}], in the order of the
    picture's rows.

    A piece counts when it is bristle-coloured in HSV (hue from hue[0] to hue[1], THROUGH red when hue[0] >
    hue[1]; s >= s_min, v >= v_min), differs from the background photo `bg` by diff_min (bg None: colour alone,
    for a single photo), has area[0] .. area[1] pixels, is elongated (4 sigma length >= len_min, length / width
    >= aspect_min, width <= width_max) and, when `near` (the pixel where the camera model sees the gripper) is
    given, lies within r_max of it. Its "tip" is the end FARTHER from `near`: no handle colour is needed, so a
    dark handle and a brush loaded with red or black paint are fine. Without `near` the lower end is called the
    tip and the caller should treat both ends as possible.

    The defaults are the evening's values for the thin round brush under the "paper" exposure; FLAT_HEAD holds
    the ones for the morning's gold head. Wrong candidates are normal (a person's arm, paint, another brush
    on the desk, the arm's own red wires): never take "the largest piece", let the poses decide (solve_tip)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lo, hi = int(hue[0]), int(hue[1])
    mask = ((h >= lo) | (h <= hi)) if lo > hi else ((h >= lo) & (h <= hi))
    mask &= (s >= int(s_min)) & (v >= int(v_min))
    if bg is not None:
        mask &= cv2.absdiff(img, bg).max(axis=2) >= int(diff_min)
    mask = mask.astype(np.uint8)
    if close and close > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((int(close), int(close)), np.uint8))
    n, lab, stats, cen = cv2.connectedComponentsWithStats(mask, 8)
    org = None if near is None else np.asarray(near, dtype=float).reshape(2)
    out = []
    for k in range(1, n):
        a = int(stats[k, cv2.CC_STAT_AREA])
        if a < area[0] or a > area[1] or (org is not None and float(np.linalg.norm(cen[k] - org)) > r_max):
            continue
        x0, y0 = int(stats[k, cv2.CC_STAT_LEFT]), int(stats[k, cv2.CC_STAT_TOP])
        ys, xs = np.nonzero(lab[y0:y0 + int(stats[k, cv2.CC_STAT_HEIGHT]), x0:x0 + int(stats[k, cv2.CC_STAT_WIDTH])] == k)
        pts = np.column_stack([xs + x0, ys + y0]).astype(np.float32)
        mean, vecs, vals = cv2.PCACompute2(pts, mean=None)
        c, d = mean[0].astype(float), vecs[0].astype(float)
        d /= np.linalg.norm(d)
        length, width = 4.0 * math.sqrt(float(vals[0, 0])), 4.0 * math.sqrt(max(float(vals[1, 0]), 1e-6))
        if length < len_min or length / max(width, 1.0) < aspect_min or width > width_max:
            continue
        t = (pts - c) @ d
        e1, e2 = c + d * float(np.percentile(t, 0.5)), c + d * float(np.percentile(t, 99.5))
        if org is not None:
            far, other = (e1, e2) if np.linalg.norm(e1 - org) > np.linalg.norm(e2 - org) else (e2, e1)
        else:
            far, other = (e1, e2) if e1[1] > e2[1] else (e2, e1)
        out.append({"tip": (float(far[0]), float(far[1])), "tail": (float(other[0]), float(other[1])),
                    "centre": (float(cen[k][0]), float(cen[k][1])), "area": a, "len_px": round(length, 1),
                    "width_px": round(width, 1)})
    return out


def tip_points(cands, both_ends: bool = False) -> np.ndarray:
    """(k, 2) pixels of candidates(): their tips, or tips and tails when it is not known which end is which."""
    pts = [c["tip"] for c in cands] + ([c["tail"] for c in cands] if both_ends else [])
    return np.asarray(pts, dtype=float).reshape(-1, 2)


def _cand_list(uvs) -> list:
    """Observations as a list of (k, 2) arrays, one per photo (k = 0: nothing was seen in it). Accepts an
    (N, 2) array (one pixel per photo) or any sequence of pixel lists."""
    if isinstance(uvs, np.ndarray) and uvs.ndim == 2 and uvs.shape[1] == 2:
        return [row.reshape(1, 2).astype(float) for row in uvs]
    return [np.asarray(c, dtype=float).reshape(-1, 2) for c in uvs]


def _gripper_frames(qs):
    """(rot (N, 3, 3), org (N, 3)) of gripper_frame_link in the base frame for N joint poses."""
    frames = [K.fk(q) for q in np.asarray(qs, dtype=float).reshape(-1, 5)]
    return np.stack([t[:3, :3] for t in frames]), np.stack([t[:3, 3] for t in frames])


EDGE_PX = 25.0                       # a tip this near the picture's border is cut off with its brush head


def _why_unused(n_cands: int, nearest: float, pixel, size) -> str:
    """Why a photo did not take part in a fit, for the operator: where the fit puts the tip in it comes first
    (outside the picture, or at its edge: 2026-11-13 morning, two photos with the brush head cut off at the top),
    then whether the photo had a candidate at all."""
    u, v = float(pixel[0]), float(pixel[1])
    if not (0.0 <= u <= size[0] - 1.0 and 0.0 <= v <= size[1] - 1.0):
        return "tip outside the picture"
    if min(u, v, size[0] - 1.0 - u, size[1] - 1.0 - v) < EDGE_PX:
        return "tip at the edge of the picture"
    if n_cands == 0:
        return "no candidate"
    return f"no candidate near the solution (the nearest is {nearest:.0f} px away)"


# ----------------------------------------------------------------------------- camera pose + tool tip
def _project(rvec, tvec, f, cx, cy, pts):
    r, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3))
    pc = np.asarray(pts, dtype=float) @ r.T + np.asarray(tvec, dtype=float).reshape(3)
    return np.column_stack([f * pc[:, 0] / pc[:, 2] + cx, f * pc[:, 1] / pc[:, 2] + cy]), pc[:, 2]


def _jacobian(fun, x, f0, eps=1e-6):
    jac = np.empty((len(f0), len(x)))
    for k in range(len(x)):
        dx = np.zeros(len(x))
        dx[k] = eps * max(1.0, abs(x[k]))
        jac[:, k] = (fun(x + dx) - fun(x - dx)) / (2.0 * dx[k])
    return jac


def _levenberg(fun, x0, iters: int = 80, lam: float = 1e-3):
    """Levenberg-Marquardt with a numeric Jacobian (numpy only: scipy is not part of this rig's environment).
    -> (x, jacobian at x, residuals at x)"""
    x = np.asarray(x0, dtype=float).copy()
    f0 = fun(x)
    cost = float(f0 @ f0)
    for _ in range(iters):
        jac = _jacobian(fun, x, f0)
        g, hmat = jac.T @ f0, jac.T @ jac
        improved = False
        for _try in range(12):
            try:
                step = np.linalg.solve(hmat + lam * np.diag(np.maximum(np.diag(hmat), 1e-12)), -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            f1 = fun(x + step)
            c1 = float(f1 @ f1)
            if np.isfinite(c1) and c1 < cost:
                gain = cost - c1
                x, f0, cost, lam, improved = x + step, f1, c1, max(lam / 5.0, 1e-9), True
                break
            lam *= 10.0
        if not improved or gain < 1e-10 * max(cost, 1e-12):
            break
    return x, _jacobian(fun, x, f0), f0


def _p3p_start(pts3, cs, f: float, cx: float, cy: float, iters: int = 3000, tol: float = 60.0, seed: int = 0):
    """A camera pose (rvec, tvec) under which as many photos as possible have a candidate within tol of their
    point pts3[i], or None. For photos with SEVERAL candidates: RANSAC over all (point, candidate) pairs cannot
    work (with 4 .. 8 pixels per photo one pair in ten is right, and a 5-point sample is all right once in
    100 000), so a sample is four photos with ONE candidate each (P3P), photos with few candidates preferred.
    The seed is fixed: the same photos give the same answer."""
    n = len(cs)
    if n < 4:
        return None
    pad = np.full((n, max(len(c) for c in cs), 2), np.inf)
    for i, c in enumerate(cs):
        pad[i, :len(c)] = c
    w = np.array([1.0 / len(c) ** 2 for c in cs])
    w /= w.sum()
    kmat = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    rng = np.random.default_rng(seed)
    best_n, best = -1, None
    for _ in range(iters):
        idx = rng.choice(n, 4, replace=False, p=w)
        img = np.array([cs[i][int(rng.integers(len(cs[i])))] for i in idx], dtype=float)
        try:
            ok, rvec, tvec = cv2.solvePnP(pts3[idx].reshape(-1, 1, 3), img.reshape(-1, 1, 2), kmat, None,
                                          flags=cv2.SOLVEPNP_P3P)
        except cv2.error:
            continue
        if not ok:
            continue
        uv, depth = _project(rvec, tvec, f, cx, cy, pts3)
        if not (np.all(np.isfinite(uv)) and np.all(depth > 0.05)):
            continue
        score = int(np.sum(np.min(np.linalg.norm(pad - uv[:, None, :], axis=2), axis=1) <= tol))
        if score > best_n:
            best_n, best = score, (rvec.ravel().copy(), tvec.ravel().copy())
            if best_n >= 0.9 * n:
                break
    return best


def solve(qs, uvs, tip_guess, f: float, cx: float, cy: float, max_px: float = 25.0, start=None,
          size=IMAGE_SIZE) -> dict:
    """Camera pose and tool tip offset from N poses: qs (N, 5) rad; uvs the tool tip in each photo, either
    (N, 2) px (ONE pixel per photo: the morning's use, with detect_tip) or a list of N pixel lists (the
    CANDIDATES of each photo, none or several: the nearest to the fit counts, re-chosen in every round);
    tip_guess (3,) the registered tip offset in gripper_frame_link axes (the start value); start: optionally
    (rvec, tvec) of a camera pose to try as the start besides the PnP one (the model the rig had).

    Robust: a first fit on everything (soft loss), then observations further than max_px from it are dropped
    and the fit is repeated. With candidate lists the first rounds also leave out what is further than a
    shrinking radius (a wrong candidate hundreds of pixels away is no "bad detection", it is another thing).

    -> {"rvec", "tvec", "tip_offset", "center", "rms_px", "max_px", "residuals_px" (per photo given, nan = not
        used), "all_px" (distance of the nearest candidate, nan = none), "used", "n" (photos with a candidate),
        "photos", "index" (photos used), "uv" (their pixels), "pixels" (where the fit puts the tip in every
        photo), "unused": [{"i", "why"}], "sigma": standard errors of the nine numbers (m / rad), "cov" (their
        9 x 9 covariance), "center_sigma" (m), "cond": condition number of the fit}
    Raises ValueError when fewer than MIN_FULL_PHOTOS photos have a candidate or agree with one camera pose."""
    qs = np.asarray(qs, dtype=float).reshape(-1, 5)
    every = _cand_list(uvs)
    if len(every) != len(qs):
        raise ValueError(f"{len(qs)} poses but {len(every)} observations")
    have = np.array([i for i, c in enumerate(every) if len(c)], dtype=int)
    n = len(have)
    if n < MIN_FULL_PHOTOS:
        raise ValueError(f"air calibration needs at least {MIN_FULL_PHOTOS} poses with a detected tip (got {n})")
    rot_all, org_all = _gripper_frames(qs)              # gripper_frame_link in the base frame
    rot, org = rot_all[have], org_all[have]
    cs = [every[i] for i in have]
    several = any(len(c) > 1 for c in cs)
    first = np.stack([c[0] for c in cs])

    def tips(off):
        return org + rot @ np.asarray(off, dtype=float)

    def pick(x):
        """The candidate of every photo that is nearest to where x puts the tip, and how far it is."""
        uv, _depth = _project(x[0:3], x[3:6], f, cx, cy, tips(x[6:9]))
        if not several:
            return first, np.linalg.norm(first - uv, axis=1)
        sel = np.stack([c[int(np.argmin(np.linalg.norm(c - uv[i], axis=1)))] for i, c in enumerate(cs)])
        return sel, np.linalg.norm(sel - uv, axis=1)

    def res(x, obs, use):
        uv, _depth = _project(x[0:3], x[3:6], f, cx, cy, tips(x[6:9])[use])
        return (uv - obs[use]).ravel()

    kmat = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
    guess = np.asarray(tip_guess, dtype=float).reshape(3)
    starts = []
    if several:
        first_try = _p3p_start(tips(guess), cs, f, cx, cy)
        if first_try is not None:
            starts.append(np.concatenate([first_try[0], first_try[1], guess]))
    else:
        try:
            ok, rvec, tvec, _inl = cv2.solvePnPRansac(tips(guess).reshape(-1, 1, 3), first.reshape(-1, 1, 2), kmat, None,
                                                      reprojectionError=40.0, iterationsCount=500, flags=cv2.SOLVEPNP_EPNP)
        except cv2.error:
            ok = False
        if ok:
            starts.append(np.concatenate([rvec.ravel(), tvec.ravel(), guess]))
    if start is not None:
        starts.append(np.concatenate([np.asarray(start[0], dtype=float).ravel(), np.asarray(start[1], dtype=float).ravel(), guess]))
    if not starts:
        raise ValueError("no camera pose fits the detections (PnP failed)")
    # the start that explains more photos (the PnP one when they tie: what the morning's solve started from)
    support = []
    for x0 in starts:
        d0 = pick(x0)[1]
        support.append(int(np.sum(d0 <= 40.0)) if np.all(np.isfinite(d0)) else -1)
    x = starts[int(np.argmax(support))]
    everything = np.ones(n, dtype=bool)
    # 1) robust fit on everything (Huber weights, re-computed each round), 2) drop what is further than
    #    max_px from it, 3) plain least squares on the rest
    radii = (150.0, 100.0, 70.0, 50.0, 40.0, 30.0) if several else (None,) * 6
    for radius in radii:
        obs, r0 = pick(x)
        sel = everything if radius is None else r0 <= radius
        if sel.sum() < MIN_FULL_PHOTOS:
            raise ValueError(f"only {int(sel.sum())} photos have a candidate within {radius:.0f} px of one camera pose")
        w = np.repeat(np.sqrt(np.minimum(1.0, 8.0 / np.maximum(r0[sel], 1e-9))), 2)
        x, _jac, _fun = _levenberg(lambda y: w * res(y, obs, sel), x)
    obs, r_all = pick(x)
    use = r_all <= max_px
    if use.sum() < MIN_FULL_PHOTOS:
        raise ValueError(f"only {int(use.sum())} detections agree with one camera pose within {max_px} px")
    x, jac, fun = _levenberg(lambda y: res(y, obs, use), x)
    obs, r = pick(x)                                    # (the same choice unless two candidates lie side by side)
    # a fit that ran away (the camera at infinity, a tool tip metres from the gripper) is no calibration
    _uv, depth = _project(x[0:3], x[3:6], f, cx, cy, tips(x[6:9]))
    if not (np.all(np.isfinite(x)) and float(np.linalg.norm(x[6:9])) <= TL.MAX_OFFSET_M + 0.05
            and float(np.linalg.norm(x[3:6])) <= 5.0 and np.all(depth[use] > 0.0)):
        raise ValueError("the fit left every plausible range (camera pose or tip offset): the detections do not "
                         "belong to one camera and one tool tip")
    rr = r[use]
    dof = max(1, 2 * int(use.sum()) - 9)
    s2 = float(np.sum(fun ** 2)) / dof
    rmat, _ = cv2.Rodrigues(x[0:3])
    try:
        cov = np.linalg.inv(jac.T @ jac) * s2
        sigma = np.sqrt(np.diag(cov))
        sv = np.linalg.svd(jac, compute_uv=False)
        cond = float(sv[0] / sv[-1])
        # the camera centre -R^T t depends on six of the nine numbers: its standard error through their covariance
        jc = _jacobian(lambda y: -cv2.Rodrigues(y[0:3])[0].T @ y[3:6], x[0:6], np.zeros(3))
        center_sigma = np.sqrt(np.diag(jc @ cov[0:6, 0:6] @ jc.T))
    except np.linalg.LinAlgError:
        cov = np.full((9, 9), np.nan)
        sigma, cond, center_sigma = np.full(9, np.nan), float("inf"), np.full(3, np.nan)
    npix = len(qs)
    pixels, _depth = _project(x[0:3], x[3:6], f, cx, cy, org_all + rot_all @ x[6:9])
    resid, nearest = np.full(npix, np.nan), np.full(npix, np.nan)
    resid[have[use]], nearest[have] = r[use], r
    unused = [{"i": int(i), "why": _why_unused(len(every[i]), nearest[i], pixels[i], size)}
              for i in range(npix) if not np.isfinite(resid[i])]
    return {"rvec": x[0:3].tolist(), "tvec": x[3:6].tolist(), "tip_offset": x[6:9].tolist(),
            "center": (-rmat.T @ x[3:6]).tolist(), "rms_px": float(np.sqrt(np.mean(rr ** 2))),
            "max_px": float(rr.max()), "residuals_px": resid.tolist(), "all_px": nearest.tolist(), "used": int(use.sum()),
            "n": n, "photos": npix, "index": [int(i) for i in have[use]], "uv": obs[use].tolist(),
            "pixels": pixels.tolist(), "unused": unused, "sigma": sigma.tolist(), "center_sigma": center_sigma.tolist(),
            "cond": cond, "cov": cov.tolist()}


# ----------------------------------------------------------------------------- tool tip alone (camera fixed)
def vote_box(tip_guess=None, near: float = 0.02, margin: float = 0.05, box=VOTE_BOX, step: float = VOTE_STEP) -> tuple:
    """The box of tip offsets the vote searches: the evening's box (tools from the finger line out to 12 cm
    beside it, 3 .. 25 cm ahead of the gripper frame). Where the registered tip lies within `near` of a side of
    it, or outside, that side is moved out to `margin` beyond the registration (on the same grid): a tool taped
    across the fingers is searched around where it was registered, while for the usual tools the box stays the
    evening's and the arm's own red wires behind the gripper stay outside it."""
    if tip_guess is None:
        return tuple((float(lo), float(hi)) for lo, hi in box)
    out = []
    for (lo, hi), g in zip(box, np.asarray(tip_guess, dtype=float).reshape(3)):
        lo2 = math.floor((g - margin) / step + 1e-9) * step if g - near < lo else float(lo)
        hi2 = math.ceil((g + margin) / step - 1e-9) * step if g + near > hi else float(hi)
        out.append((min(float(lo), lo2), max(float(hi), hi2)))
    return tuple(out)


def solve_tip(qs, cands, cam, tip_guess=None, box=None, step: float = VOTE_STEP, vote_px: float = VOTE_PX,
              radii=FIT_RADII, min_photos: int = MIN_TIP_PHOTOS, size=IMAGE_SIZE) -> dict:
    """Tool tip offset with the camera model FIXED: qs (N, 5) rad, cands the candidate pixels of each photo
    (a list of N pixel lists, or (N, 2)), cam a camera model with .project(points) (camera.CameraModel).

    1) a vote: every offset of a grid (box, step: default vote_box(tip_guess)) is projected into every photo;
       the offset for which the most photos have a candidate within vote_px wins (among equals, the one whose
       candidates are nearest). This is what stands up to wrong candidates in most photos: they do not move
       with the gripper the way one fixed offset does.
    2) least squares of the three numbers on the nearest candidate of each photo, within a radius that shrinks
       (radii).
    tip_guess only widens the box (vote_box); it is not a start value.

    -> {"tip_offset", "sigma" (standard errors, m), "rms_px", "max_px", "used", "n" (photos with a candidate),
        "photos", "index" (photos used), "uv" (their pixels), "residuals_px" (per photo, nan = not used),
        "all_px" (distance of the nearest candidate, nan = none), "pixels" (where the fit puts the tip in every
        photo), "unused": [{"i", "why"}], "vote": {"offset", "photos", "cells"}}
    Raises ValueError when fewer than min_photos photos have a candidate, or agree on one offset."""
    qs = np.asarray(qs, dtype=float).reshape(-1, 5)
    cs = _cand_list(cands)
    if len(cs) != len(qs):
        raise ValueError(f"{len(qs)} poses but {len(cs)} observations")
    npix = len(qs)
    have = [i for i, c in enumerate(cs) if len(c)]
    if len(have) < min_photos:
        raise ValueError(f"the tip measurement needs at least {min_photos} photos with a candidate (got {len(have)} "
                         f"of {npix})")
    rot, org = _gripper_frames(qs)
    box = vote_box(tip_guess, step=step) if box is None else box
    axes = [lo + step * np.arange(int(round((hi - lo) / step)) + 1) for lo, hi in box]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    grid = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    count, cost = np.zeros(len(grid), dtype=int), np.zeros(len(grid))
    for i in have:
        pr = cam.project(org[i] + grid @ rot[i].T)
        dmin = np.full(len(grid), np.inf)
        for c in cs[i]:
            dmin = np.minimum(dmin, np.hypot(pr[:, 0] - c[0], pr[:, 1] - c[1]))
        dmin = np.where(np.isfinite(dmin), dmin, np.inf)       # (a grid point behind the camera projects to nan)
        count += dmin <= vote_px
        cost += np.minimum(dmin, vote_px)
    best = int(count.max())
    if best < min_photos:
        raise ValueError(f"only {best} of the {len(have)} photos with a candidate agree on one tip offset (at least "
                         f"{min_photos} are needed): the tip is hidden, outside the picture, or not bristle-coloured")
    tied = np.nonzero(count == best)[0]
    k = int(tied[int(np.argmin(cost[tied]))])
    x = grid[k].copy()

    def nearest(x, radius):
        pr = cam.project(org + rot @ x)
        idx, uv, dist = [], [], np.full(npix, np.nan)
        for i in have:
            d = np.linalg.norm(cs[i] - pr[i], axis=1)
            j = int(np.argmin(d))
            dist[i] = float(d[j])
            if d[j] <= radius:
                idx.append(i)
                uv.append(cs[i][j])
        return np.array(idx, dtype=int), np.array(uv, dtype=float).reshape(-1, 2), dist, pr

    for radius in radii:
        idx, uv, _dist, _pr = nearest(x, radius)
        if len(idx) < 2:
            raise ValueError(f"only {len(idx)} photos have a candidate within {radius:.0f} px of one tip offset")
        x, jac, fres = _levenberg(lambda y: (cam.project(org[idx] + rot[idx] @ y) - uv).ravel(), x)
    r = np.linalg.norm(fres.reshape(-1, 2), axis=1)
    dof = max(1, 2 * len(idx) - 3)
    try:
        sigma = np.sqrt(np.diag(np.linalg.inv(jac.T @ jac) * float(np.sum(fres ** 2)) / dof))
    except np.linalg.LinAlgError:
        sigma = np.full(3, np.nan)
    _idx, _uv, dist, pixels = nearest(x, float(radii[-1]))
    resid = np.full(npix, np.nan)
    resid[idx] = r
    unused = [{"i": int(i), "why": _why_unused(len(cs[i]), dist[i], pixels[i], size)}
              for i in range(npix) if not np.isfinite(resid[i])]
    return {"tip_offset": x.tolist(), "sigma": sigma.tolist(), "rms_px": float(np.sqrt(np.mean(r ** 2))),
            "max_px": float(r.max()), "used": int(len(idx)), "n": len(have), "photos": npix,
            "index": [int(i) for i in idx], "uv": uv.tolist(), "residuals_px": resid.tolist(), "all_px": dist.tolist(),
            "pixels": pixels.tolist(), "unused": unused,
            "vote": {"offset": [round(float(v), 4) for v in grid[k]], "photos": best, "cells": int(len(grid))}}


def camera_check(qs, cands, cam, tip_offset) -> dict | None:
    """Has the camera moved? The cross-check of a tip-only solve: the SAME photos are solved for camera pose and
    tip together (solve(), started at the model and at the tip just found) and the camera that comes out is
    compared with the model.

    Why the residuals of the tip-only solve are not enough (simulated with the evening's 36 poses, 5 px noise):
    a camera that has moved by 2 cm shifts the solved tip by 12 .. 21 mm while the rms only goes from 6.4 to
    9.7 .. 14 px; moved 3 cm towards the robot (roughly along its line of sight), 11.6 px (tip 18 mm off) with
    nearly every photo still used. The tip absorbs the move. The nine-parameter fit does not: from 33 .. 36
    photos it places the camera centre to 2 .. 4 mm (formal), and on the evening's real photos (camera not
    moved) it came out 0.6, 2.0 and 5.9 mm from the model; on the morning's photos against the model of
    2026-11-11 (the camera HAD been moved) it came out 52 mm and 3.5 deg from that model, which is where the
    morning's own calibration put the camera.

    -> {"moved_mm", "turned_deg", "sigma_mm" (formal, of the centre), "d" (how many formal standard errors the
        six pose numbers are from the model: Mahalanobis), "rms_px", "used", "tip_offset"} or None when the
        photos do not carry a nine-parameter fit (too few, or they do not agree)."""
    try:
        fit = solve(qs, cands, tip_offset, cam.f, cam.cx, cam.cy, start=(cam.rvec, cam.tvec))
    except ValueError:
        return None
    cov = np.asarray(fit["cov"], dtype=float)[0:6, 0:6]
    dx = np.concatenate([np.asarray(fit["rvec"]) - cam.rvec, np.asarray(fit["tvec"]) - cam.tvec])
    try:
        d = float(math.sqrt(max(0.0, float(dx @ np.linalg.solve(cov, dx)))))
    except np.linalg.LinAlgError:
        d = float("nan")
    rmat, _ = cv2.Rodrigues(np.asarray(fit["rvec"], dtype=float))
    turn = float(np.linalg.norm(cv2.Rodrigues(rmat @ cam.R.T)[0]))
    return {"moved_mm": round(float(np.linalg.norm(np.asarray(fit["center"]) - cam.center)) * 1000.0, 1),
            "turned_deg": round(math.degrees(turn), 2),
            "sigma_mm": [round(float(v) * 1000.0, 1) if math.isfinite(v) else None for v in fit["center_sigma"]],
            "d": round(d, 1) if math.isfinite(d) else None,
            "rms_px": round(float(fit["rms_px"]), 2), "used": int(fit["used"]),
            "tip_offset": [round(float(v), 5) for v in fit["tip_offset"]]}


# ----------------------------------------------------------------------------- photos -> a measurement
def measure_tip(images, qs, cam, tip_guess=None, bg=None, **detector) -> dict:
    """Tip only, from the photos: candidates in every photo (`near` = where the camera model sees the
    gripper), then solve_tip: the evening's method. detector: keyword arguments of candidates().

    The flat brush's gold head is too large and too yellow for candidates(): when that fit does not exist or
    uses fewer than half of the photos AND detect_tip knows the tool (a head in HEAD_ENOUGH of the photos), the
    heads' tips join the candidates and the fit that uses more photos is kept.

    -> solve_tip's result plus "candidates" (how many pixels each photo offered), "points" (those pixels, for
    camera_check) and "detector" ("candidates" | "candidates+head").
    Raises ValueError when no fit exists."""
    bg = background(images) if bg is None else bg
    size = (images[0].shape[1], images[0].shape[0])
    _rot, org = _gripper_frames(qs)
    near = cam.project(org)
    cands = [tip_points(candidates(img, bg, near[i], **detector)) for i, img in enumerate(images)]
    fit, error, name = None, None, "candidates"
    try:
        fit = solve_tip(qs, cands, cam, tip_guess=tip_guess, size=size)
    except ValueError as e:
        error = e
    if fit is None or fit["used"] < 0.5 * len(images):
        heads = [detect_tip(img, bg) for img in images]
        if sum(1 for h in heads if h is not None) >= HEAD_ENOUGH * len(images):
            both = [np.vstack([c, np.asarray([h["tip"]], dtype=float)]) if h is not None else c for c, h in zip(cands, heads)]
            try:
                other = solve_tip(qs, both, cam, tip_guess=tip_guess, size=size)
                if fit is None or other["used"] > fit["used"]:
                    fit, cands, name = other, both, "candidates+head"
            except ValueError as e:
                error = error or e
    if fit is None:
        raise error
    fit.update(candidates=[int(len(c)) for c in cands], points=cands, detector=name)
    return fit


def measure_full(images, qs, tip_guess, f: float, cx: float, cy: float, prior=None, bg=None, r_max: float = 700.0,
                 **detector) -> dict:
    """Camera pose + tip, from the photos. Two detectors, two fits, and the one that explains more photos is
    kept:
    - "head": detect_tip in every photo (the flat brush's gold head): the morning's method. A photo in which
      it finds nothing has no observation.
    - "candidates": candidates() in every photo, for a tool detect_tip does not know (tried when the head fit
      does not exist or uses less than HEAD_ENOUGH of the photos). Which end of a candidate piece is the tip cannot
      be said before the camera is known: both ends are offered to a first solve, then, as seen with the
      solved camera, the far ends of the pieces within r_max of the gripper to a second one.

    prior: the camera model the rig had (anything with .rvec / .tvec), tried as a second start value.
    detector: keyword arguments of candidates().
    -> solve's result plus "detector" ("head" | "candidates"), "heads" (photos in which detect_tip found a
    head) and "candidates" (how many pixels each photo offered to the fit that was kept).
    Raises ValueError when neither detector gives a fit."""
    bg = background(images) if bg is None else bg
    size = (images[0].shape[1], images[0].shape[0])
    start = None if prior is None else (prior.rvec, prior.tvec)
    heads = [detect_tip(img, bg) for img in images]
    n_head = sum(1 for d in heads if d is not None)
    fits, errors = [], []
    if n_head >= MIN_FULL_PHOTOS:
        obs = [np.asarray([d["tip"]] if d is not None else [], dtype=float).reshape(-1, 2) for d in heads]
        try:
            fit = solve(qs, obs, tip_guess, f, cx, cy, start=start, size=size)
            fit.update(detector="head", candidates=[int(len(o)) for o in obs])
            fits.append(fit)
        except ValueError as e:
            errors.append(f"flat-brush detector: {e}")
    else:
        errors.append(f"flat-brush detector: a brush head in only {n_head} of {len(images)} photos")
    if not fits or fits[0]["used"] < HEAD_ENOUGH * len(images):
        cand = [candidates(img, bg, None, **detector) for img in images]
        try:
            obs = [tip_points(c, both_ends=True) for c in cand]
            fit = solve(qs, obs, tip_guess, f, cx, cy, start=start, size=size)
            _rot, org = _gripper_frames(qs)
            near, _depth = _project(fit["rvec"], fit["tvec"], f, cx, cy, org)
            obs = []
            for i, c in enumerate(cand):
                ends = [(e["tip"], e["tail"]) for e in c if np.linalg.norm(np.subtract(e["centre"], near[i])) <= r_max]
                obs.append(np.asarray([a if np.linalg.norm(np.subtract(a, near[i])) >= np.linalg.norm(np.subtract(b, near[i]))
                                       else b for a, b in ends], dtype=float).reshape(-1, 2))
            fit = solve(qs, obs, fit["tip_offset"], f, cx, cy, start=(fit["rvec"], fit["tvec"]), size=size)
            fit.update(detector="candidates", candidates=[int(len(o)) for o in obs])
            fits.append(fit)
        except ValueError as e:
            errors.append(f"candidates: {e}")
    if not fits:
        raise ValueError("; ".join(errors))
    fit = max(fits, key=lambda d: d["used"])
    fit["heads"] = int(n_head)
    return fit


# ----------------------------------------------------------------------------- the first look
def look_pose(arm, tool, cam=None, poses=LOOK_POSES, size=IMAGE_SIZE) -> dict:
    """The pose for the first look at a tool whose registered length is a guess: the first of `poses` (fingertip
    x, y, height above the table, pitch) the arm can hold with the tool's joint limits and every check passed,
    and in which the camera model (when there is one) sees the gripper inside the picture and the registered
    tip LOOK_EDGE_PX inside it. Every one of them keeps the bound the module text states (fingertip >=
    LOOK_MIN_HEIGHT above the table, fingers <= LOOK_MAX_PITCH below the horizontal); that is verified on the
    solved pose, not assumed. No motion.
    -> {"q", "fingertip": [x, y, height], "pitch", "tip_height_m" (modelled), "gripper_px", "tip_px"}
    Raises SafetyError when no pose qualifies (the text says why the last ones failed)."""
    lim = arm.q_limits()
    roll = min(max(0.0, lim[K.ARM_JOINTS[4]][0]), lim[K.ARM_JOINTS[4]][1])
    finger = K.tool_offset(*arm.cfg["tip_offset"])
    tz = TL.table_z(arm)
    why = []
    for x, y, h, pitch in poses:
        if h < LOOK_MIN_HEIGHT - 1e-9 or pitch > LOOK_MAX_PITCH + 1e-9:
            why.append(f"({x}, {y}, {h}, {pitch}) is outside the look bound")
            continue
        q, info = K.ik(np.array([x, y, tz + h]), float(pitch), roll, tool=finger, limits=lim)
        if not info["ok"]:
            why.append(f"({x}, {h}): {info['reason']}")
            continue
        got = K.pose_summary(q, finger)
        if got["xyz"][2] - tz < LOOK_MIN_HEIGHT - 1e-3 or got["pitch_down_deg"] > LOOK_MAX_PITCH + 0.5:
            why.append(f"({x}, {h}): the solved pose misses the look bound")
            continue
        try:
            tip_h = TL.check_pose(arm, tool, q, min_height=TL.MIN_FREE_HEIGHT + TL.PLAN_MARGIN, what="look pose")
        except SafetyError as e:
            why.append(f"({x}, {h}): {str(e)[:120]}")
            continue
        out = {"q": q, "fingertip": [float(x), float(y), float(h)], "pitch": float(pitch), "tip_height_m": float(tip_h),
               "gripper_px": None, "tip_px": None}
        if cam is not None:
            t = K.fk(q)
            g, fpx, tpx = cam.project([t[:3, 3], K.fk(q, finger)[:3, 3], K.fk(q, K.tool_offset(*tool.tip_offset))[:3, 3]])
            if not all(0.0 <= p[0] <= size[0] - 1.0 and 0.0 <= p[1] <= size[1] - 1.0 for p in (g, fpx)):
                why.append(f"({x}, {h}): the gripper would be outside the picture")
                continue
            if not (LOOK_EDGE_PX <= tpx[0] <= size[0] - 1.0 - LOOK_EDGE_PX and LOOK_EDGE_PX <= tpx[1] <= size[1] - 1.0 - LOOK_EDGE_PX):
                why.append(f"({x}, {h}): the registered tip would be at the picture's edge ({tpx[0]:.0f}, {tpx[1]:.0f})")
                continue
            out.update(gripper_px=[round(float(v), 1) for v in g], tip_px=[round(float(v), 1) for v in tpx])
        return out
    raise SafetyError(f"no look pose for tool '{tool.name}': none of the {len(poses)} poses (fingertip >= "
                      f"{LOOK_MIN_HEIGHT * 100:.0f} cm up, fingers <= {LOOK_MAX_PITCH:.0f} deg down) can be held and seen. "
                      f"Last reasons: {' | '.join(why[-3:])}")


def plan_waypoints(plan: dict) -> list:
    """The joint poses of a way, in the order they are flown: of a look_way result (its legs, one after the
    other) or of a tooling.plan_free_move result (its lift, then its path)."""
    if "legs" in plan:
        out = []
        for leg in plan["legs"]:
            out += list(leg["path"]) if not out else list(leg["path"][1:])
        return out
    return (list(plan["lift"]) + list(plan["path"][1:])) if plan.get("lift") else list(plan["path"])


# ----------------------------------------------------------------------------- the way of the look: lift, turn, lower
def tool_rise(q_a, q_b, reach: float = SIL_REACH_M) -> float:
    """How much higher (m) is, at joint pose q_b, the point of a tool that gained LEAST since q_a? Taken over
    every point within `reach` of the gripper frame, in any direction (tooling.MAX_OFFSET_M: whatever may be
    taped on). A point p (gripper axes) stands at  z_gripper + up . p,  `up` being the base frame's vertical
    seen from the gripper (the third row of its rotation); so the least gain of all of them is
        (z_b - z_a) - reach * |up_b - up_a|.
    A turn of the base alone changes neither term: it is 0 for it. A lift that keeps the gripper's direction
    gives its own height; every degree the gripper tips on the way costs 5 mm of it. Negative: some point of a
    tool that long may have come down by that much."""
    ta, tb = K.fk(np.asarray(q_a, dtype=float)), K.fk(np.asarray(q_b, dtype=float))
    return float(tb[2, 3] - ta[2, 3]) - float(reach) * float(np.linalg.norm(tb[2, :3] - ta[2, :3]))


def straight_lift(arm, tool, q, rise_m: float = LOOK_LIFT_M) -> list:
    """Joint waypoints (starting with q) that raise the gripper rise_m STRAIGHT UP without turning it: pan, roll
    and the gripper's pitch stay as they are, so every point of whatever is taped to the gripper goes up by the
    same rise_m, along a vertical line. That is what makes the lift safe for a tool nobody has measured: nothing
    of it moves sideways, nothing swings, and afterwards all of it is rise_m above where it hung.

    Planned as a straight line of the registered tip (Arm.plan_tip_path, a waypoint every 3 mm) and then
    VERIFIED on the solved poses, not assumed: each passes tooling.check_pose with the registered tool (its
    joint limits, the keep-outs), no point of a tool of up to SIL_REACH_M comes lower on the way than it was a
    waypoint before (tool_rise), and at the end all of them have gained rise_m (to 3 mm). A start pose that
    lies outside a joint limit of the tool is held there, as tooling.lift_path does. No motion.
    Raises SafetyError when the arm cannot do that from q (the text says what stopped it: mostly the wrist's
    limit, for a gripper that points steeply down)."""
    q = np.asarray(q, dtype=float)
    rise = float(rise_m)
    tl = K.tool_offset(*tool.tip_offset)
    try:
        planner = TL._holding_start(tool, q)
        # an arm that was folded onto a soft limit reads a few hundredths of a degree beyond it (pan -100.02 under a
        # limit of -100): the line is planned from the pose just inside, which the path's start is re-anchored to
        q_in = q.copy()
        for i, (lo, hi) in enumerate(TL._limits(arm, planner)[j] for j in K.ARM_JOINTS):
            if not lo <= q_in[i] <= hi and min(abs(q_in[i] - lo), abs(q_in[i] - hi)) <= math.radians(0.5):
                q_in[i] = min(max(q_in[i], lo + 1e-5), hi - 1e-5)
        p = K.fk(q_in, tl)[:3, 3].copy()
        pitch = K.pose_summary(q_in, tl)["pitch_down_deg"]
        with TL._as_tool(arm, planner):
            wps = arm.plan_tip_path([(np.array([p[0], p[1], p[2] + rise]), pitch, math.degrees(float(q_in[4])))],
                                    q_start=q_in, step_m=0.003)
        wps = [q] + list(wps[1:])
        gained = 0.0
        for k, w in enumerate(wps[1:], 1):
            TL.check_pose(arm, tool, w, q_start=q, what=f"lift point {k}")
            g = tool_rise(q, w)
            if g < gained - 5e-4:
                raise SafetyError("something on the gripper would come down on the way up")
            gained = max(gained, g)
        if gained < rise - 0.003:
            raise SafetyError(f"the lift would end {gained * 1000:.0f} mm up")
    except SafetyError as e:
        raise SafetyError(f"the arm cannot lift what is on its gripper {rise * 100:.0f} cm straight up from that pose "
                          f"({str(e)[:160]})") from e
    return [np.asarray(w, dtype=float) for w in wps]


def plane_move(arm, tool, q_from, q_to) -> dict:
    """A free move that does NOT turn the base: tooling.plan_free_move for two poses on one base angle, with
    the ways that keep it (direct, the wrist first, the wrist last; never through the stand-by pose, which lies
    on another base angle). A tool tip lower than tooling.LIFT_HEIGHT is lifted first, as there. -> the same
    dict ({"lift", "path", "via", "min_height_m", "lift_via"}). SafetyError when no such way passes. No motion."""
    q0, q1 = np.asarray(q_from, dtype=float), np.asarray(q_to, dtype=float)
    if abs(float(q1[0] - q0[0])) > math.radians(LOOK_TURN_FREE_DEG) + 1e-9:
        raise SafetyError("plane_move is for two poses on one base angle")
    limit = TL.MIN_FREE_HEIGHT + TL.PLAN_MARGIN
    h_to = TL.tip_height(arm, q1, tool)
    if h_to < limit:
        raise SafetyError(f"the pose to go to has the tool tip {h_to * 1000:.0f} mm above the table; a free move keeps "
                          f"it above {limit * 1000:.0f} mm")
    lift, lift_via, q = None, None, q0
    if TL.tip_height(arm, q0, tool) < TL.LIFT_HEIGHT - 0.002:
        lift, lift_via = TL._lift_plan(arm, tool, q0)
        q = np.asarray(lift[-1], dtype=float)
    first, late = q.copy(), q1.copy()
    first[3:], late[3:] = q1[3:], q[3:]
    roomy = min(TL.tip_height(arm, q, tool), limit + 0.01) - 0.003
    tried, best = [], None
    for name, mids in (("direct", []), ("wrist first", [first]), ("wrist last", [late])):
        pts = [q] + mids + [q1]
        try:
            lo = min(TL._segment(arm, tool, pts[i], pts[i + 1], limit, name) for i in range(len(pts) - 1))
        except SafetyError as e:
            tried.append(f"{name}: {e}")
            continue
        if best is None or lo > best["min_height_m"] + 0.002:
            best = {"lift": lift, "path": pts, "via": name, "min_height_m": round(float(lo), 4), "lift_via": lift_via}
        if lo >= roomy:
            break
    if best is None:
        raise SafetyError(f"no way to that pose that leaves the base where it is keeps the tip of tool '{tool.name}' "
                          f"{limit * 1000:.0f} mm above the table and passes the safety checks. Tried: {' | '.join(tried[:3])}")
    return best


def _plane_legs(plan: dict) -> list:
    """The legs of a plane_move result: its lift (kind "raise"), then every piece of its path (kind "plane")."""
    legs = [{"kind": "raise", "path": [np.asarray(w, dtype=float) for w in plan["lift"]]}] if plan.get("lift") else []
    pts = plan["path"]
    return legs + [{"kind": "plane", "path": [np.asarray(pts[i], dtype=float), np.asarray(pts[i + 1], dtype=float)]}
                   for i in range(len(pts) - 1)]


def look_way(arm, tool, q_from, q_to, lift_m: float = LOOK_LIFT_M) -> dict:
    """The way of the first look from joint pose q_from to q_to (the look pose, or the look pose turned aside
    for the photo "before"). No motion. The base turns ONLY with the tool lifted:

      lift    where the arm stands: straight up by lift_m, the gripper not turning (straight_lift);
      turn    the base alone, to the base angle of q_to: no point of the tool changes its height;
      plane   (only when the arm did not start in the joints of q_to: from a fold) at that height and on that
              base angle to the place straight above q_to;
      lower   straight down onto q_to: straight_lift of q_to, backwards.

    So while the base turns, every point of whatever is taped to the gripper is lift_m above where it was when
    the arm stood still before the lift (tool_rise says so for every sample of the turn; it is asked again of
    the measured arm when the way is flown: fly_look_way), which is at least lift_m above the table. And where
    it comes down, it comes down along a vertical line, onto where the look pose has it.

    Where the arm cannot lift straight up from q_from (a gripper that points steeply down has its wrist at the
    limit: the photo pose of the painting commands, most poses of a capture), it first goes to the look pose on
    the base angle it STANDS at, without turning the base (plane_move; the tool moves in the vertical plane it
    is in, as on every way to the look pose before), and lifts, turns and lowers from there.
    Two poses on one base angle (LOOK_TURN_FREE_DEG) need no turn: plane_move alone.

    -> {"legs": [{"kind": "lift" | "turn" | "plane" | "raise" | "lower", "path": [q, ...]}, ...], "via": text,
        "turn": bool, "lift_m": lift_m when the base turns, "turn_rise_m": the least any point of a tool has
        gained at the turn (tool_rise), "min_height_m": the lowest modelled height of the REGISTERED tip}.
    Raises SafetyError when no such way passes the checks: nothing is flown then (the caller refuses)."""
    q0, q1 = np.asarray(q_from, dtype=float), np.asarray(q_to, dtype=float)
    limit = TL.MIN_FREE_HEIGHT + TL.PLAN_MARGIN
    out = {"legs": [], "via": "the arm is there", "turn": False, "lift_m": None, "turn_rise_m": None}
    if float(np.max(np.abs(q1 - q0))) <= math.radians(0.3):
        pass
    elif abs(float(q1[0] - q0[0])) <= math.radians(LOOK_TURN_FREE_DEG):
        plan = plane_move(arm, tool, q0, q1)
        out.update(legs=_plane_legs(plan), via=plan["via"] + " (the base does not turn)")
    else:
        down = straight_lift(arm, tool, q1, lift_m)             # (flown backwards: the last leg of either way)

        def turned(up):
            """The turn at the top of the lift `up`: (its two poses, the least gain of a tool point there)."""
            a = np.asarray(up[-1], dtype=float)
            b = a.copy()
            b[0] = q1[0]
            TL._segment(arm, tool, a, b, limit, "the turn of the base")
            gain = min(tool_rise(up[0], a), tool_rise(up[0], b))
            if gain < float(lift_m) - 0.005:
                raise SafetyError(f"at the turn a tool would be only {gain * 1000:.0f} mm above where it hung")
            return [a, b], gain

        try:
            up = straight_lift(arm, tool, q0, lift_m)
            turn, gain = turned(up)
            legs = [{"kind": "lift", "path": up}, {"kind": "turn", "path": turn}]
            via = f"lifted {float(lift_m) * 100:.0f} cm straight up, the base turned at that height"
            if float(np.max(np.abs(turn[1] - down[-1]))) > math.radians(0.3):
                across = plane_move(arm, tool, turn[1], down[-1])
                legs += _plane_legs(across)
                via += f", across to the place above the pose ({across['via']})"
        except SafetyError as e:
            # the arm cannot lift from where it stands: to the look pose on ITS base angle first, the base not turning
            mid = q1.copy()
            mid[0] = q0[0]
            try:
                there = plane_move(arm, tool, q0, mid)
                up = straight_lift(arm, tool, mid, lift_m)
                turn, gain = turned(up)
            except SafetyError as e2:
                raise SafetyError(f"no way of the look turns the base with the tool lifted: {e}; and by the look pose on the "
                                  f"base angle the arm stands at: {e2}") from e2
            legs = _plane_legs(there) + [{"kind": "lift", "path": up}, {"kind": "turn", "path": turn}]
            via = (f"to the look pose on the base angle it stands at ({there['via']}, the base not turning), lifted "
                   f"{float(lift_m) * 100:.0f} cm straight up, the base turned at that height")
        legs.append({"kind": "lower", "path": [np.asarray(w, dtype=float) for w in down[::-1]]})
        out.update(legs=legs, via=via + ", straight down", turn=True, lift_m=float(lift_m), turn_rise_m=round(float(gain), 4))
    wps = plan_waypoints(out) if out["legs"] else [q0]
    low = TL.tip_height(arm, wps[0], tool)
    for a, b in zip(wps[:-1], wps[1:]):
        n = max(1, int(math.ceil(math.degrees(float(np.max(np.abs(b - a)))) / 1.5)))
        low = min(low, min(TL.tip_height(arm, a + (b - a) * (k / n), tool) for k in range(1, n + 1)))
    out["min_height_m"] = round(float(low), 4)
    return out


def fly_look_way(arm, q_to, speed_deg_s: float = LOOK_SPEED_DEG_S, settle=0.5, way: dict | None = None) -> dict:
    """look_way from the measured pose to q_to, then fly it, leg by leg (each its own Arm.run_path, from the pose
    the arm really stands in). way: a look_way result planned before; it is planned again when the arm does not
    stand where it starts (an arm settles when its torque comes on).

    What is asked of the MEASURED arm, not of the plan:
    - a lift is planned from where the arm stands when the lift begins (straight_lift), and must not sink;
    - before the base turns, and all the while it turns, every point of a tool of up to SIL_REACH_M is still
      LOOK_CLEAR_M above where it was when the lift began (tool_rise of the measured poses): else the base does
      not turn, or stops turning, and the arm holds (MotionAborted);
    - on every leg the registered tip stays tooling.MIN_FREE_HEIGHT above the table, as in tooling.free_move.
    SafetyError before anything moves (no way, or the first leg does not pass from the measured pose); a leg
    that cannot be flown after the arm has moved is a MotionAborted: the arm holds where it stands.
    -> {"via", "legs": how many were flown, "turned": bool, "rise_at_turn_m": the least gain measured before the
        turn (None: no turn), "min_tip_height_m": the lowest MEASURED height of the registered tip, "duration_s"}."""
    tool = arm.tool
    if tool is None:
        raise SafetyError("fly_look_way is for an arm with a tool attached")
    q = arm.read_q()
    q_to = np.asarray(q_to, dtype=float)
    if way is not None and way["legs"] and float(np.max(np.abs(np.asarray(way["legs"][0]["path"][0]) - q))) > math.radians(0.5):
        way = None                                              # (the arm is not where the plan starts: plan again)
    if way is None:
        way = look_way(arm, tool, q, q_to)
    low = {"h": TL.tip_height(arm, q, tool)}
    out = {"via": way["via"], "legs": 0, "turned": False, "rise_at_turn_m": None, "duration_s": 0.0}
    ground = None                                               # the measured pose from which the last lift began
    legs = way["legs"]

    def run(wps, settle_s, monitor, what):
        try:
            r = arm.run_path(wps, speed_deg_s, settle=settle_s, monitor=monitor)
        except MotionAborted as e:
            if e.info.get("reason") == "monitor":
                raise MotionAborted(f"{what}; the arm holds where it stopped (the registered tip {low['h'] * 1000:.0f} mm "
                                    "above the table, measured)",
                                    info={**e.info, "tool_tip_height_m": round(low["h"], 4)}) from e
            raise
        out["duration_s"] = round(out["duration_s"] + float(r["duration_s"]), 3)
        out["legs"] += 1
        return r

    for i, leg in enumerate(legs):
        kind, path = leg["kind"], leg["path"]
        last = i == len(legs) - 1
        q_now = arm.read_q()
        try:
            if kind == "lift":
                ground, h0 = q_now, TL.tip_height(arm, q_now, tool)
                wps = straight_lift(arm, tool, q_now, way["lift_m"])

                def rising(t, q_plan, q_meas, _h0=h0):
                    h = TL.tip_height(arm, q_meas, tool)
                    low["h"] = min(low["h"], h)
                    return h < _h0 - 0.004                     # (it is supposed to go UP)

                run(wps, 1.0, rising, "the lift before the base turns was stopped: the tool sank")
            elif kind == "turn":
                gain = None if ground is None else tool_rise(ground, q_now)
                if gain is None or gain < LOOK_CLEAR_M:
                    raise MotionAborted("the base was not turned: the arm does not hold the tool lifted (" +
                                        ("no lift was flown before" if gain is None else
                                         f"the least a tool point has gained is {gain * 1000:.0f} mm, measured; "
                                         f"{LOOK_CLEAR_M * 1000:.0f} mm are asked") + "); the arm holds where it stands",
                                        info={"reason": "not lifted"})
                TL._segment(arm, tool, q_now, path[-1], TL.MIN_FREE_HEIGHT, "the turn of the base")
                out["rise_at_turn_m"] = round(float(gain), 4)

                def lifted(t, q_plan, q_meas, _g=ground):
                    low["h"] = min(low["h"], TL.tip_height(arm, q_meas, tool))
                    return tool_rise(_g, q_meas) < LOOK_CLEAR_M

                run([q_now, path[-1]], 0.3, lifted, "the base stopped turning: the arm no longer held the tool "
                                                    f"{LOOK_CLEAR_M * 1000:.0f} mm above where it hung before the lift")
                out["turned"] = True
            else:
                # "plane" / "raise": pieces on one base angle; "lower": straight down, from the top the plan has
                wps = [q_now] + [np.asarray(w, dtype=float) for w in (path if kind == "lower" else path[1:])]
                if kind == "lower" and float(np.max(np.abs(wps[1] - q_now))) < math.radians(0.05):
                    wps = [q_now] + wps[2:]
                if abs(float(wps[-1][0] - q_now[0])) > math.radians(LOOK_TURN_FREE_DEG):
                    raise SafetyError("the arm does not stand on the base angle this leg was planned for")
                for a, b in zip(wps[:-1], wps[1:]):
                    TL._segment(arm, tool, a, b, None if kind == "raise" else TL.MIN_FREE_HEIGHT, f"the way of the look ({kind})")

                def watch(t, q_plan, q_meas):
                    h = TL.tip_height(arm, q_meas, tool)
                    low["h"] = min(low["h"], h)
                    return kind != "raise" and h < TL.MIN_FREE_HEIGHT

                run(wps, settle if last else 0.3, watch, f"the way of the look ({kind}) was stopped: the registered tip "
                                                         "came too near the table")
        except SafetyError as e:
            if out["legs"]:             # (the arm has moved: this is no refusal "before anything moved")
                raise MotionAborted(f"the way of the look could not go on from where the arm stands ({e}); the arm holds "
                                    "there", info={"reason": "replan"}) from e
            raise
    out["min_tip_height_m"] = round(float(low["h"]), 4)
    return out


def _reach_mask(cam, centre, radius_m: float, size, pad_px: float, step: int = 4) -> np.ndarray:
    """The picture of a ball: every pixel whose line of sight passes within radius_m of `centre` (base frame), as
    a mask (uint8, 255 inside), grown by pad_px. Worked out on every step-th pixel (the lens distortion of the
    model included) and enlarged: the growth covers the coarse grid. A camera that stands inside the ball sees
    it everywhere."""
    w, h = int(size[0]), int(size[1])
    g = np.asarray(centre, dtype=float).reshape(3) - np.asarray(cam.center, dtype=float)
    dist = float(np.linalg.norm(g))
    if dist <= radius_m:
        return np.full((h, w), 255, np.uint8)
    us, vs = np.arange(0, w, step) + (step - 1) / 2.0, np.arange(0, h, step) + (step - 1) / 2.0
    uu, vv = np.meshgrid(us, vs)
    px = np.column_stack([uu.ravel(), vv.ravel()]).astype(np.float64)
    xy = cv2.undistortPoints(px.reshape(-1, 1, 2), cam.K, cam.dist).reshape(-1, 2)
    rays = np.column_stack([xy, np.ones(len(xy))]) @ cam.R          # (camera axes -> base axes, row vectors)
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    along = rays @ g
    off = np.sqrt(np.maximum(0.0, dist * dist - along * along))
    grow = (float(pad_px) + step) * dist / float(cam.f)             # the pad, in metres at the ball's distance
    small = ((off <= radius_m + grow) & (along > 0.0)).astype(np.uint8).reshape(len(vs), len(us)) * 255
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def silhouette(cam, q, tip_offset=None, size=IMAGE_SIZE, pad_px: float = SIL_PAD_PX) -> np.ndarray:
    """Where the camera model sees the arm and ANYTHING that may be taped to its gripper, with the arm at joint
    pose q: a mask of the picture (uint8, 255 inside). Drawn: the links from the base to the gripper (SIL_ARM_M
    wide), and around the gripper frame everything a tool can reach from it in any direction (a ball of
    SIL_REACH_M = tooling.MAX_OFFSET_M: its picture is a disc around the gripper, 700 .. 1200 px in radius with
    this rig's camera). Everything is grown by pad_px. A link that would be behind the camera, or nearer to it
    than 8 cm, is left out. tip_offset is not used any more (the parameter stays for the callers): where the
    tool is, is what a look is about to judge, so the drawing must not depend on the registration.

    What it is for: a photo "before" shows the scene WITHOUT the tool only outside this mask (look_check lets
    nothing inside it pass the look; look_hidden asks whether it reaches the place of the look). Until round 3
    the tool was drawn as two lines (along the fingers, through the registered tip), and a tool whose body lay
    beside both hid paint outside the drawing (SIL_REACH_M)."""
    w, h = int(size[0]), int(size[1])
    q = np.asarray(q, dtype=float)
    t = K.fk(q)
    org = t[:3, 3]
    mask = _reach_mask(cam, org, SIL_REACH_M, (w, h), pad_px)
    chain = [np.zeros(3)] + [f[:3, 3] for f in K.link_frames(q)] + [org]
    lines = [(a, b, SIL_ARM_M) for a, b in zip(chain[:-1], chain[1:])]
    for a, b, width in lines:
        pts = a + (b - a) * np.linspace(0.0, 1.0, 13)[:, None]
        depth = (pts @ cam.R.T + cam.tvec)[:, 2]
        ok = depth > 0.08
        uv = np.full((len(pts), 2), np.nan)
        if ok.any():
            uv[ok] = cam.project(pts[ok])
        for i in range(len(pts) - 1):
            if not (ok[i] and ok[i + 1] and np.all(np.isfinite(uv[i:i + 2])) and float(np.abs(uv[i:i + 2]).max()) < 1e5):
                continue
            thick = width * cam.f / float(min(depth[i], depth[i + 1])) + 2.0 * float(pad_px)
            cv2.line(mask, (int(round(uv[i][0])), int(round(uv[i][1]))), (int(round(uv[i + 1][0])), int(round(uv[i + 1][1]))),
                     255, int(min(max(1.0, round(thick)), 2000.0)))
    return mask


def look_hidden(cam, q, q_look, tip_offset, size=IMAGE_SIZE, near_px: float = LOOK_NEAR_PX) -> str | None:
    """Would a photo taken with the arm at joint pose q show the place of the look WITHOUT arm and tool? None
    when it would; else why not (text). The place of the look: where the look photo (the arm at q_look) will
    have the REGISTERED tip. A photo taken at q is no photo "before" when

    - the arm's drawing (silhouette: its links, and everything a tool of up to SIL_REACH_M can reach from the
      gripper, wherever it points) comes within near_px of that place: what arm or tool hide there, the look
      photo reveals, and paint among it is "new" and would be a candidate right where the registration is judged;
    - the gripper stands nearer to the camera than LOOK_CAMERA_M (the reach then covers the whole picture; the
      rule is older than the reach and stays as it is).
    With this rig's camera that leaves the look pose turned 90 deg and more about the base (look_aside), and
    little else: a folded arm is never a place for the photo "before". No motion, no photo."""
    q, off = np.asarray(q, dtype=float), np.asarray(tip_offset, dtype=float).reshape(3)
    t0, t1 = K.fk(np.asarray(q_look, dtype=float)), K.fk(q)
    p0 = cam.project([t0[:3, 3] + t0[:3, :3] @ off])[0]
    if not np.all(np.isfinite(p0)):
        return "the camera model cannot say where the look photo will have the registered tip"
    d_cam = float(np.linalg.norm(t1[:3, 3] - cam.center))
    if d_cam < LOOK_CAMERA_M:
        return f"the gripper would stand {d_cam * 100:.0f} cm from the camera"
    mask = silhouette(cam, q, off, size)
    disc = np.zeros_like(mask)
    cv2.circle(disc, (int(round(float(p0[0]))), int(round(float(p0[1])))), int(round(near_px)), 255, -1)
    if np.any(mask & disc):
        return "the arm would hide the place of the look"
    return None


def look_aside(arm, tool, cam, q_from, q_look, turns=LOOK_ASIDE_DEG, size=IMAGE_SIZE) -> dict:
    """Where an arm that stands at the look pose, or anywhere the camera could see it or its tool at the place of
    the look (look_hidden), goes for the photo "before" (a look that was refused leaves it at the look pose, and
    the next run starts from it): the LOOK POSE with only the base turned, far enough that the camera sees
    gripper and registered tip LOOK_ASIDE_PX from where the look photo will have them, and the drawing of the
    arm and of everything a tool can reach (silhouette) clear of the place of the look. Among the turns that
    qualify, the first that takes the registered tip out of the picture, else the first.

    Why the look pose turned, and not a fold: the registered length is what the look is about to judge, and a
    fold is planned with it. Simulated with a tool registered as 8 cm along the fingers and really 17 cm: folded
    from the look pose into the tool's rest pose its tip came down to 10 mm above the table, into the pose `fd
    paint rest` chooses 5 mm UNDER it. The look pose is the one pose made for a tool of unknown length. (Until
    round 3 the arm kept the joints it STOOD with and only its base was turned; a folded arm, whose gripper is
    25 .. 32 cm from the look's line of sight wherever its base points, cannot clear the reach of a 30 cm tool
    that way.)

    The way there and the way back are look_way: the base turns only with the tool lifted LOOK_LIFT_M straight
    up, and the arm comes straight down onto the pose (2026-11-14, evening. Until then the look pose was turned
    at its own height, the fingertip 12 cm up: every point of the tool kept the height it has there, and for a
    tool longer than registered that was 12 .. 47 mm above the table, over the water dish and the palette). A
    turn qualifies only when both ways exist and do not dip for a tool of unknown length (unknown_tool_dip).
    What the lift does not know: things that stand higher than LOOK_LIFT_M above where the tool hangs, and what
    lies under the place where the tool comes down beside the table's working area (the look pose turned aside:
    with the fingertip 12 cm up, as the look pose itself). No motion.

    -> {"q", "turn_deg", "pan_deg", "plan" (the way from q_from, a look_way result), "back" (the way from there
        to the look pose), "gripper_px", "tip_px", "tip_in_picture"}
    Raises SafetyError when no turn qualifies (the text says why the last ones failed): "cannot be turned
    aside" when no turn clears the place of the look, whatever the start; "no safe way" when a turn does and the
    arm cannot get there and back with the tool lifted from where it stands."""
    q_from, q_look = np.asarray(q_from, dtype=float), np.asarray(q_look, dtype=float)
    lim = arm.q_limits()[K.ARM_JOINTS[0]]
    off = np.asarray(tool.tip_offset, dtype=float)

    def seen(q):
        t = K.fk(q)
        return cam.project([t[:3, 3], t[:3, 3] + t[:3, :3] @ off])

    g0, t0 = seen(q_look)
    found, why, no_way = None, [], []
    for turn in turns:
        q = q_look.copy()
        q[0] = float(q_look[0]) + math.radians(float(turn))
        if not lim[0] <= q[0] <= lim[1]:
            why.append(f"{turn:+g} deg: outside the pan limits")
            continue
        g, t = seen(q)
        if not (np.all(np.isfinite(g)) and np.all(np.isfinite(t))) or min(float(np.linalg.norm(g - g0)),
                                                                           float(np.linalg.norm(t - t0))) < LOOK_ASIDE_PX:
            why.append(f"{turn:+g} deg: the tool would still be near its place in the look photo")
            continue
        hidden = look_hidden(cam, q, q_look, off, size)
        if hidden:
            why.append(f"{turn:+g} deg: {hidden}")
            continue
        try:                                        # the way back first: it does not depend on where the arm stands
            back = look_way(arm, tool, q, q_look)
        except SafetyError as e:
            why.append(f"{turn:+g} deg: no way back to the look pose with the tool lifted ({str(e)[:200]})")
            continue
        if unknown_tool_dip(arm, tool, plan_waypoints(back)):
            why.append(f"{turn:+g} deg: the way back would bring a tool of unknown length lower than it is there")
            continue
        try:
            plan = look_way(arm, tool, q_from, q)
        except SafetyError as e:
            no_way.append(f"{turn:+g} deg: {str(e)[:260]}")
            continue
        if unknown_tool_dip(arm, tool, plan_waypoints(plan)):
            no_way.append(f"{turn:+g} deg: the way would bring a tool of unknown length lower than it is now")
            continue
        inside = bool(0.0 <= t[0] <= size[0] - 1.0 and 0.0 <= t[1] <= size[1] - 1.0)
        finger = K.fk(q, K.tool_offset(*arm.cfg["tip_offset"]))[:3, 3]
        got = {"q": q, "turn_deg": float(turn), "pan_deg": round(math.degrees(float(q[0])), 1), "plan": plan, "back": back,
               "gripper_px": [round(float(v), 1) for v in g], "tip_px": [round(float(v), 1) for v in t],
               "tip_in_picture": inside, "fingertip_xy": [round(float(finger[0]), 3), round(float(finger[1]), 3)]}
        if not inside:
            return got
        found = found or got
    if found is not None:
        return found
    if no_way:
        raise SafetyError(f"the arm stands at the look pose, or where the camera could see it or its tool at the place of "
                          f"the look, and there is no safe way from where it stands to the look pose turned aside: the base "
                          f"turns only with the tool lifted {LOOK_LIFT_M * 100:.0f} cm straight up (so that a tool of unknown "
                          f"length passes that far above the table). Last reasons: {' | '.join(no_way[-2:])}")
    raise SafetyError(f"the arm stands at the look pose, or where the camera could see it or its tool at the place of the "
                      f"look, and cannot be turned aside for a photo of the scene without tool '{tool.name}' in it (the look "
                      f"pose turned about the base, with everything a tool of up to {SIL_REACH_M * 100:.0f} cm can reach clear "
                      f"of that place, and back to the look pose with the tool lifted {LOOK_LIFT_M * 100:.0f} cm before the "
                      f"base turns). Last reasons: {' | '.join(why[-3:])}")


def look_background(img, cam, q_look, tip_offset, radius_px: float = LOOK_CLEAN_PX, min_px: int = LOOK_CLEAN_MIN_PX,
                    hue=BRISTLE_HUE, s_min: int = 110, v_min: int = 60) -> dict:
    """Is the photo "before" clean where the look will judge? img: the photo of the scene without the tool there;
    q_look: the look pose; tip_offset: the registered tip. Counted: the pixels within radius_px of the place where
    the look photo will have the registered tip that are bristle-coloured (the colour test of candidates(): hue
    THROUGH red, s >= s_min, v >= v_min), whatever their shape.

    Why (round 3): the look takes for a tip what differs between its two photos and is bristle-coloured. Paint
    that lies there differs as soon as the arm's shadow or the brush falls on it, and paint beside it makes the
    nearest "candidate" a matter of a few pixels: the look cannot tell a brush from paint behind it. On the
    evening's desk the place of the look lay on the lantern's red. So the look is only taken over a background
    that has none: a blank sheet over the picture, or the sheet taken away.

    -> {"clean": bool, "paint_px": how many such pixels, "radius_px", "min_px", "place_px": [u, v], "nearest_px":
        how far the nearest of them is from the place (None: none)}. No motion."""
    t = K.fk(np.asarray(q_look, dtype=float))
    p = cam.project([t[:3, 3] + t[:3, :3] @ np.asarray(tip_offset, dtype=float).reshape(3)])[0]
    hh, ww = img.shape[:2]
    out = {"clean": False, "paint_px": None, "radius_px": float(radius_px), "min_px": int(min_px),
           "place_px": [round(float(v), 1) for v in p], "nearest_px": None}
    if not np.all(np.isfinite(p)):
        return out
    r = int(math.ceil(radius_px))
    u0, v0 = int(round(float(p[0]))), int(round(float(p[1])))
    x0, x1, y0, y1 = max(0, u0 - r), min(ww, u0 + r + 1), max(0, v0 - r), min(hh, v0 + r + 1)
    if x0 >= x1 or y0 >= y1:
        out.update(clean=True, paint_px=0)             # (the place is outside the picture: look_pose does not allow it)
        return out
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    lo, hi = int(hue[0]), int(hue[1])
    mask = ((h >= lo) | (h <= hi)) if lo > hi else ((h >= lo) & (h <= hi))
    mask &= (s >= int(s_min)) & (v >= int(v_min))
    ys, xs = np.nonzero(mask)
    d = np.hypot(xs + x0 - float(p[0]), ys + y0 - float(p[1]))
    inside = d <= radius_px
    n = int(inside.sum())
    out.update(clean=n < int(min_px), paint_px=n, nearest_px=round(float(d[inside].min()), 1) if n else None)
    return out


def unknown_tool_dip(arm, tool, waypoints, step_deg: float = 1.5) -> dict | None:
    """Would a tool of UNKNOWN length come lower on this way than it is at its two ends? waypoints: joint poses
    joined by straight joint-space pieces (a plan_free_move result: its "lift", then its "path").

    Tried: tips 2 .. 20 cm from the fingertip along the fingers and along the registered direction, and up to
    10 cm leaning 30 deg to either jaw and sideways (the tools the look bound covers). A tip that would be
    inside the table at the start cannot be what is taped on (the arm is there) and is not judged. Returns None
    when every such tip stays within 5 mm of the lower of its two end heights or 2 cm above the table, else
    {"length_m", "direction", "lowest_m", "ends_m"} of the worst one. No motion."""
    wps = [np.asarray(w, dtype=float) for w in waypoints]
    if len(wps) < 2:                    # (a way of no length: the arm is there)
        return None
    samples = [wps[0]]
    for a, b in zip(wps[:-1], wps[1:]):
        n = max(1, int(math.ceil(math.degrees(float(np.max(np.abs(b - a)))) / step_deg)))
        samples += [a + (b - a) * (k / n) for k in range(1, n + 1)]
    f0 = np.asarray(arm.cfg["tip_offset"], dtype=float)
    ax, jaw = TL._FINGER_AXIS, TL._FIXED_JAW_AXIS
    side = np.cross(ax, jaw)
    c30, s30 = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    dirs = [("along the fingers", ax, 0.20), ("30 deg towards the fixed jaw", c30 * ax + s30 * jaw, 0.10),
            ("30 deg towards the moving jaw", c30 * ax - s30 * jaw, 0.10), ("30 deg sideways", c30 * ax + s30 * side, 0.10),
            ("30 deg sideways", c30 * ax - s30 * side, 0.10)]
    reach = np.asarray(tool.tip_offset, dtype=float) - f0
    if float(np.linalg.norm(reach)) > 0.005:
        dirs.append(("the registered direction", reach / np.linalg.norm(reach), 0.20))
    frames = [K.fk(q) for q in samples]
    tz = TL.table_z(arm)
    worst = None
    for name, d, l_max in dirs:
        for length in np.arange(0.02, l_max + 1e-9, 0.02):
            off = f0 + d * length
            hs = np.array([(t[:3, 3] + t[:3, :3] @ off)[2] - tz for t in frames])
            if hs[0] < 0.0:
                continue
            ends = min(float(hs[0]), float(hs[-1]))
            low = float(hs.min())
            if low < ends - 0.005 and low < TL.MIN_FREE_HEIGHT:
                if worst is None or low < worst["lowest_m"]:
                    worst = {"length_m": round(float(length), 3), "direction": name, "lowest_m": round(low, 4),
                             "ends_m": [round(float(hs[0]), 4), round(float(hs[-1]), 4)]}
    return worst


def look_check(img, cam, q, tip_offset, bg=None, limit_px: float = LOOK_PX, q_bg=None, **detector) -> dict:
    """Is the registered tip where the photo shows a tool tip? img: the look photo; q: the MEASURED joint pose
    it was taken in; tip_offset: the registered tip; bg: a photo of the same scene with the tool elsewhere (taken
    before the arm went to the look pose), so that paint on the sheet and things on the desk are no candidates;
    q_bg: the joint pose the arm stood in for that photo.

    Without bg the look is NOT CHECKED: "checked" False and "ok" False, whatever lies near. By colour alone the
    strokes of a picture are tips (28 to 41 pieces in the evening's photos; registrations 5 and 9 cm off had
    one within 21 and 35 px). The numbers are still filled in, for a person who looks at the photo.

    With q_bg, a candidate that lies where the arm stood for the photo before (silhouette) cannot make the look
    pass: bg does not show the scene there, it shows the arm, and paint the arm hid is "new" in the look photo.
    "hidden_before" counts those candidates, "nearest_clear_px" is the nearest of the others. They stay in the
    list and may be the nearest (a tool that is not where it is registered is often seen where the folded arm
    lay, and the suggestion for a better registration is made from it). The caller takes the photo before where
    the arm does not hide the place of the look (look_hidden): a real tip there would not count either.

    With bg, the background must be clean where the look judges (look_background: no bristle-coloured paint in
    bg within LOOK_CLEAN_PX of the registered tip's place): where it is not, nothing can make the look pass
    ("background" says what was found).

    -> {"ok", "checked", "registered_px", "candidates": [[u, v, distance] ...] (nearest first), "nearest_px",
        "nearest_mm" (in the image plane, at the registered tip's distance from the camera), "limit_px",
        "hidden_before", "nearest_clear_px" (both None without q_bg), "background" (look_background of bg; None
        without bg),
        "suggest_tip": the registered offset moved onto the line of sight of the nearest candidate (a better
        guess IF the photo shows that this candidate is the tool tip; along the line of sight nothing is known)}
    ok: checked, the background is clean, and a candidate that the arm did not hide in the photo before lies
    within limit_px of the registered tip's projection."""
    t = K.fk(np.asarray(q, dtype=float))
    tip = t[:3, 3] + t[:3, :3] @ np.asarray(tip_offset, dtype=float)
    g, p = cam.project([t[:3, 3], tip])
    pts = [c["tip"] for c in candidates(img, bg, g, **detector)]
    if bg is not None:
        head = detect_tip(img, bg)                      # the flat brush's gold head is too large for candidates()
        if head is not None:
            pts.append(head["tip"])
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    dist = np.linalg.norm(pts - p, axis=1) if len(pts) else np.zeros(0)
    order = np.argsort(dist)
    depth = float((cam.R @ tip + cam.tvec)[2])
    checked = bg is not None
    hidden, clear = None, dist
    if checked and q_bg is not None:
        hh, ww = img.shape[:2]
        mask = silhouette(cam, q_bg, tip_offset, size=(ww, hh))
        under = np.array([bool(0 <= int(round(c[0])) < ww and 0 <= int(round(c[1])) < hh
                               and mask[int(round(c[1])), int(round(c[0]))]) for c in pts], dtype=bool)
        hidden, clear = int(under.sum()), dist[~under]
    ground = look_background(bg, cam, q, tip_offset) if checked else None
    clean = ground is None or bool(ground["clean"])
    out = {"ok": bool(checked and clean and len(clear) and float(clear.min()) <= limit_px), "checked": checked,
           "background": ground,
           "registered_px": [round(float(v), 1) for v in p],
           "candidates": [[round(float(pts[k][0]), 1), round(float(pts[k][1]), 1), round(float(dist[k]), 1)] for k in order[:6]],
           "nearest_px": round(float(dist[order[0]]), 1) if len(dist) else None, "nearest_mm": None,
           "limit_px": float(limit_px), "hidden_before": hidden,
           "nearest_clear_px": None if hidden is None or not len(clear) else round(float(clear.min()), 1), "suggest_tip": None}
    if len(dist):
        out["nearest_mm"] = round(float(dist[order[0]]) * depth / cam.f * 1000.0, 1)
        o, d = cam.ray(pts[order[0]])
        on_ray = o + d * float((tip - o) @ d)           # the point of that line of sight nearest to the registered tip
        out["suggest_tip"] = [round(float(v), 4) for v in t[:3, :3].T @ (on_ray - t[:3, 3])]
    return out
