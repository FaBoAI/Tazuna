"""Painting with watercolours: water, paint wells, colour recipes, and a plan of brush strokes laid layer by layer.

The set-up of one session is a JSON file (paint_setup.json in the project folder, local state like
tool_active.json; $FD_PAINT_SETUP names another; `load_setup` checks it, `save_setup` writes it):

  {"photo_q_deg": [...],                          arm pose (deg) in which the camera sees paper and palette
   "canvas": {"origin_xy": [x, y], "size_mm": [W, H], "pitch_deg": 65},
        the picture's top left corner (as the viewer in front of the table sees it) in the base frame.
        Canvas u (to the right) runs along +y of the robot, canvas v (down) along +x.
   "contact": {"z0", "x0", "y0", "bx", "by", "measured": {...}},
        plan height at which the bristles first touch the paper: z = z0 + bx * (x - x0) + by * (y - y0), from
        ladders of test strokes (`fd paint contact`). "measured" is the brush state the plane holds for (tool tip,
        gripper position, on which jaw and whether it is held by a squeeze, canvas pitch): a layer is not
        painted with another one (contact_record_state). It is written by `fd paint contact --save` from the
        record that command keeps beside its two photos, or, for a plane that was measured before records
        existed, by `fd paint contact --adopt` (the operator's word: "adopted": true). A plane WITHOUT a record
        is not painted with: nothing says which brush it was measured with (adopt it, or measure it).
   "press_mm": 2.0,                               how far below that a stroke with "press": 1.0 runs
   "brush": {"kind": "round", "modes": {"fine": {"press": 0.6, "width_mm": 2.7},
                                        "broad": {"press": 1.2, "width_mm": 5.0}}, "load_mm": 140}
          | {"kind": "flat", "wide_mm": 9.0, "thin_mm": 2.5, "load_mm": 180},
        which brush is on the gripper. A round brush paints the same width in every direction, and one load
        lasts load_mm of FINE line (a stroke uses its length times its width / the fine width). Absent: the
        flat brush of the morning of 2026-11-13 (its band measured about 9 mm), reloaded after 180 mm of plain
        travel. A plan says the same in its own "brush" block; a plan WITHOUT one is a plan for that flat brush.
   "wells": {name: {"xy": [x, y], "z_mm": 2.5, "previous_xy": [x, y], "located": false, "expected_xy": [x, y]}},
        paint blobs; z_mm = brush tip above the table when scrubbing; previous_xy: where the well was before
        the last `fd paint locate` ("located" beside "wells" says when, and from which photo); "located": false:
        that command did not find the well in its photo, "xy" is where it was BEFORE the palette moved
        ("expected_xy": where the palette's move would put it; the next photo looks for it there), and no paint
        is loaded from it (require_wells) until a later photo shows it or the place is written by hand
   "water": {"rinse": {"xy", "z_mm"}, "clean": {"xy", "z_mm"}},
        ONE DISH: only "rinse" (then "clean" is the same dish, and it gets dirty fast)
   "blot": {"xy": [x, y], "size": [sx, sy], "z": height} | null,   scrap paper where a brush is pressed out
   "paints": {name: {"first": [step, ...], "reload": [step, ...], "after_without_rinse": [name, ...], "load_mm": 110}},
        a step is ["water", "clean" | "rinse", passes], ["scrub", well, passes], ["touch", well], ["blot"];
        "load_mm" (optional): one load of THIS paint lasts less than the brush's (a blotted brush carries less)
   "rest": {"pan_deg", "clearance_m"}}            optional: where `fd paint rest` folds the arm (else it chooses)

What the brush carries (and which tool that was: another tool on the gripper carries nothing known), how far
each layer of a plan got (by the plan's real path) and the running means of the time model is paint_state.json
($FD_PAINT_STATE names another). A simulated run has its own file, never that one: beside $FD_PAINT_STATE when
that is set, else outside the tree (fd.sim_state_dir(), where `fd --sim` keeps its simulated tool too). A state
file that cannot be read or holds something else than these is a refusal (stage "state"), never an empty
state: the next command would write over the progress of a finished picture.

What was measured on the real arm (2026-11-13):
- morning, a flat brush (its band about 9 mm wide), tube watercolour in a plastic palette: a water dip, then
  scrubbing the blob back and forth along the arm's radial direction (4 passes of +-6 mm at 2.5 mm above the
  table), twice, loads enough paint for 200 mm of a 9 mm band; a single light touch of the blob gives a few
  dark dabs and then only water;
- evening, a thin round brush (tip 4 mm wide, 15 mm long) on the moving jaw, the jaw squeezed shut:
  - the paper is NOT where table_z says: with the tool tip calibrated in the air the bristles first touched at
    plan z = +3 .. +9 mm (table_z is -1.5 mm), higher away from the robot (36 mm per metre). Ladders of short
    strokes at the four corners and the centre of the canvas, read from two photos, give the plane
    (read_contact). Ladders outside the picture (the far scrap strip, the raised margin) gave other heights and
    misled for an hour: measure where the picture will be. A loose brush touched 4 mm higher than a rigid one,
    and with the tip under the fingers the same plan z is another real height at another gripper pitch: so the
    plane is stored WITH the brush state it holds for;
  - two reliable modes: press 0.6 (3 mm below first touch) a line, press 1.2 (6 mm, the limit) a band of 4 to
    5 mm; between them the width jumps. The paint fades within 60 to 90 mm of band, about 140 mm of fine line;
  - tints: water twice and a touch of the blob is a good pale wash of red; with dark blue and black the same
    recipe came out nearly full strength (a real dilution step is still missing);
  - lettering needs the brush pressed out on the blotting strip after loading (recipe step "blot");
  - marks land about 6 mm beside and 9 mm nearer to the robot than the camera model puts a commanded point;
  - one water dish stood in reach (the second, 0.44 m from the pan axis, did not): check_reach says so before
    anything is painted;
  - 70 strokes took 50 minutes, the estimates had said 26 to 35: the time model here is fitted to five timed
    calls of that evening and follows every real run (fit_times, times_of, note_run);
  - folded with the torque off the wrist can sink until the brush stands on what is below it: `fd paint rest`
    folds where neither the tip nor the place it would sink to is over the picture (rest_choice; rest_report
    answers the same question for the folds of other commands).

Depths, all counted from the first-touch plane of the set-up: a stroke presses at most 6 mm (brushwork.
MAX_PRESS_M); press and `--lower-mm` together at most 7.5 mm (MAX_DEPTH_M: what the bands of that evening ran
at, 6 mm with the plane lowered by 1.5); a rung of a contact ladder at most 6 mm below the plane the set-up
already has there (check_rungs; the bristles are 15 mm long, and a squashed brush shifts in its tape). And
nothing goes below table_z - 6 mm, whatever the plane says.

What a painting command asks of the tool before it moves (the review of that day's update): a measured tip
(Tool.tip_measured: a first guess was 9 cm short that evening), a plane that is recorded for the brush as it
is, and a scrap sheet over the picture for contact ladders (the before photo is looked at: bare_paper).

Everything that moves goes through brushwork.BrushRun (checked moves, depth limits). Results are plain dicts
with ok / stage / reason / next / say (Japanese), as everywhere in this harness; what is refused before anything
moves is a Refusal (a SafetyError that carries its stage and the next command).
Tests: tests/test_paint.py (simulated arm, saved photos of the evening, synthetic photos).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
import time

import numpy as np

import brushwork as BW
import kinematics as K
import render as RD
import tazuna_paths
import tooling as TL
from robot import ARM, DEFAULT_CONFIG, MotionAborted, SafetyError

SETUP_FILE = os.path.join(tazuna_paths.ROOT, "paint_setup.json")
STATE_FILE = os.path.join(tazuna_paths.ROOT, "paint_state.json")
MOTION_ERRORS = (SafetyError, MotionAborted)
STEP_KINDS = ("water", "scrub", "touch", "blot")
DISHES = ("rinse", "clean")
SETUP_NEXT = ("the set-up file is written by hand (its blocks are described at the top of tools/paint/paintskill.py); "
              "`fd paint status` reads it again")
LADDER_GAP_M = 0.012          # from rung to rung of a ladder, along +y
# A contact plane holds for the brush state it was measured with: tip offset within 0.5 mm (as the pen's
# calibration), gripper position within 6 ticks (half a degree of the jaw: 0.5 mm at the brush tip, 0.2 mm of
# height), canvas pitch within 1 deg.
CONTACT_TIP_TOL_M = 0.0005
CONTACT_GRIPPER_TOL = 6
CONTACT_PITCH_TOL_DEG = 1.0
CONTACT_MIN_PLACES = 4        # `fd paint contact --save`: a plane through fewer places has no residual to judge
CONTACT_MAX_RESIDUAL_MM = 1.5
CANVAS_INSET_M = 0.015        # the standard ladders stand this far inside the canvas corners
MAX_LOWER_MM = 2.0            # `fd paint run --lower-mm`: the contact plane at most this much lower for one command
# Press and lowering are counted TOGETHER against this: the deepest a stroke ran below the measured first-touch
# plane on the real arm (2026-11-13, every band after 18:17: press 6 mm with the plane lowered by 1.5 mm). The
# lowering is not a second allowance on top of the 6 mm of press.
MAX_DEPTH_M = 0.0075
# (the sheet first, the command after it: this `next` is not to be typed before somebody has laid the sheet)
CONTACT_NEXT = ("lay a scrap sheet over the whole canvas, then `fd paint contact --scrap-laid --save` (it measures the paper "
                "height for the brush as it is; the first photo is looked at before any paint is loaded)")
ADOPT_NEXT = ("fd paint contact --adopt   (ONLY when the stored plane was measured with the brush exactly as it is now: no "
              "motion, it records the brush state; else lay a scrap sheet over the canvas and `fd paint contact "
              "--scrap-laid --save`)")
# `fd paint locate` stores the wells only when the photo shows the palette: most wells found by a blob of their
# own, and the palette's shift agreed on by several (one stray blob near one well moved all ten on 2026-11-13's
# review).
LOCATE_MIN_PART = 0.6
LOCATE_MIN_AGREE = 3
LOCATE_AGREE_M = 0.008        # a blob this near to where the palette's move puts a well is that well's blob
LOCATE_PAINT_R_M = 0.003      # "the photo shows paint there": a disc of this radius round the place ...
LOCATE_PAINT_PART = 0.5       # ... is paint-coloured to at least this part
# Test strokes and ladders keep this far from the wells and dishes of the set-up unless they are asked to go
# there (a well of the palette is 3 cm wide and has dividers 15 mm high, a dish a rim of 25 mm: one wrong sign
# drags the pressed brush through them).
NEAR_WELL_M = 0.04
NEAR_DISH_M = 0.08
# A test stroke that comes this near to the canvas of the set-up counts as ON it (on_canvas): painted on a scrap
# sheet only. (The evening's bands were 4 to 5 mm wide and landed 6 mm beside, 9 mm nearer than commanded.)
TEST_CANVAS_MARGIN_M = 0.010
# The flat brush of the morning (render.py's model of it, where that module still names the numbers): a band of
# 9 mm along the robot's direction, 2.5 mm sideways, one load for RD.LOAD_MM (180 mm) of travel.
FLAT_WIDE_MM = float(getattr(RD, "BRUSH_WIDE_MM", 9.0))
FLAT_THIN_MM = float(getattr(RD, "BRUSH_THIN_MM", 2.5))


class Refusal(SafetyError):
    """Something a painting command refuses with a text BEFORE anything moves: stage / reason / next as in
    every result of the harness (a SafetyError, so code that only knows SafetyError still stops)."""

    def __init__(self, stage: str, reason: str, next_: str = "", **detail):
        super().__init__(reason)
        self.stage, self.reason, self.next, self.detail = stage, reason, next_, detail


# ============================================================================= the set-up file
def setup_path() -> str:
    return os.environ.get("FD_PAINT_SETUP") or SETUP_FILE


def _num(v, what: str, lo: float, hi: float) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or not (lo <= float(v) <= hi):
        raise ValueError(f"{what} must be a number in {lo:g} .. {hi:g} (got {v!r})")
    return float(v)


def _pair(v, what: str, lo: float, hi: float) -> tuple:
    if not (isinstance(v, (list, tuple)) and len(v) == 2):
        raise ValueError(f"{what} must be two numbers (got {v!r})")
    return _num(v[0], what + "[0]", lo, hi), _num(v[1], what + "[1]", lo, hi)


def _count(v, what: str, lo: int, hi: int) -> int:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or float(v) != int(v) or not (lo <= int(v) <= hi):
        raise ValueError(f"{what} must be a whole number in {lo} .. {hi} (got {v!r})")
    return int(v)


def brush_of(setup: dict) -> dict:
    """The brush of a set-up, complete: {"kind": "flat", "wide_mm", "thin_mm", "load_mm"} or {"kind": "round",
    "modes": {"fine": {"press", "width_mm"}, "broad": {...}}, "load_mm"}, plus "stated" (False: the set-up has
    no "brush" block, and this is the flat brush of the morning of 2026-11-13). ValueError for a block that
    cannot be used."""
    b = setup.get("brush")
    if b is None:
        return {"kind": "flat", "wide_mm": FLAT_WIDE_MM, "thin_mm": FLAT_THIN_MM, "load_mm": RD.LOAD_MM, "stated": False}
    if not isinstance(b, dict):
        raise ValueError('"brush" must be an object')
    kind = b.get("kind")
    if kind == "flat":
        return {"kind": "flat", "wide_mm": _num(b.get("wide_mm", FLAT_WIDE_MM), '"brush.wide_mm"', 1.0, 60.0),
                "thin_mm": _num(b.get("thin_mm", FLAT_THIN_MM), '"brush.thin_mm"', 0.3, 30.0),
                "load_mm": _num(b.get("load_mm", RD.LOAD_MM), '"brush.load_mm"', 10.0, 2000.0), "stated": True}
    if kind == "round":
        modes = b.get("modes")
        if not (isinstance(modes, dict) and isinstance(modes.get("fine"), dict) and isinstance(modes.get("broad"), dict)):
            raise ValueError('"brush.modes" of a round brush must have "fine" and "broad", each {"press", "width_mm"}')
        out = {}
        for name in ("fine", "broad"):
            out[name] = {"press": _num(modes[name].get("press"), f'"brush.modes.{name}.press"', 0.05, 3.0),
                         "width_mm": _num(modes[name].get("width_mm"), f'"brush.modes.{name}.width_mm"', 0.3, 30.0)}
        if not (out["fine"]["press"] < out["broad"]["press"] and out["fine"]["width_mm"] <= out["broad"]["width_mm"]):
            raise ValueError('"brush.modes": the broad mode must press harder than the fine one and not be narrower')
        return {"kind": "round", "modes": out, "load_mm": _num(b.get("load_mm"), '"brush.load_mm"', 10.0, 2000.0),
                "stated": True}
    raise ValueError(f'"brush.kind" must be "round" or "flat" (got {kind!r})')


def brush_width_mm(brush: dict, press: float) -> float | None:
    """Width of the line a round brush leaves at a press factor: the fine width at the fine press, the broad
    width at the broad press, straight between and beyond (the design kit's rule; on the real brush only the
    two modes are reliable). None for a flat brush: its band depends on the direction of travel (render.py)."""
    if brush["kind"] != "round":
        return None
    f, b = brush["modes"]["fine"], brush["modes"]["broad"]
    w = f["width_mm"] + (float(press) - f["press"]) / (b["press"] - f["press"]) * (b["width_mm"] - f["width_mm"])
    return float(max(w, 0.25 * f["width_mm"]))


def use_mm(brush: dict, length_mm: float, press: float = 1.0) -> float:
    """How much of one load a stroke takes, in the unit of brush["load_mm"]: plain travel for a flat brush,
    millimetres of FINE line for a round one (length x width / fine width: a broad band empties the brush
    nearly twice as fast as a line)."""
    if brush["kind"] != "round":
        return float(length_mm)
    return float(length_mm) * brush_width_mm(brush, press) / brush["modes"]["fine"]["width_mm"]


def _check_dish(d, what: str):
    if not isinstance(d, dict):
        raise ValueError(f"{what} must be an object with \"xy\"")
    _pair(d.get("xy"), what + ".xy", -1.0, 1.0)
    for key in ("z_mm", "scrub_z_mm"):
        if key in d:
            _num(d[key], f"{what}.{key}", 0.0, 40.0)


def _check_setup(s) -> None:
    if not isinstance(s, dict):
        raise ValueError("it is not a JSON object")
    for key in ("canvas", "contact", "wells", "water", "paints", "photo_q_deg"):
        if key not in s:
            raise ValueError(f"it has no {key!r}")
    q = s["photo_q_deg"]
    if not (isinstance(q, (list, tuple)) and len(q) == 5):
        raise ValueError('"photo_q_deg" must be five joint angles in degrees')
    for i, v in enumerate(q):
        _num(v, f'"photo_q_deg"[{i}]', -360.0, 360.0)
    c = s["canvas"]
    if not isinstance(c, dict):
        raise ValueError('"canvas" must be an object with "origin_xy" and "size_mm"')
    _pair(c.get("origin_xy"), '"canvas.origin_xy" (metres)', -1.0, 1.0)
    _pair(c.get("size_mm"), '"canvas.size_mm"', 10.0, 600.0)
    if c.get("pitch_deg") is not None:
        _num(c["pitch_deg"], '"canvas.pitch_deg"', 15.0, 120.0)
    k = s["contact"]
    if not isinstance(k, dict):
        raise ValueError('"contact" must be an object with "z0"')
    _num(k.get("z0"), '"contact.z0" (metres)', -0.03, 0.05)
    for key in ("x0", "y0"):
        if key in k:
            _num(k[key], f'"contact.{key}" (metres)', -1.0, 1.0)
    for key in ("bx", "by"):
        if key in k:
            _num(k[key], f'"contact.{key}" (metres per metre)', -0.3, 0.3)
    if k.get("measured") is not None and not isinstance(k["measured"], dict):
        raise ValueError('"contact.measured" must be an object (written by fd paint contact with --save)')
    if "press_mm" in s:
        _num(s["press_mm"], '"press_mm"', 0.1, BW.MAX_PRESS_M * 1000.0)
    wells = s["wells"]
    if not isinstance(wells, dict):
        raise ValueError('"wells" must be an object {name: {"xy": [x, y]}}')
    for name, w in wells.items():
        if not isinstance(w, dict):
            raise ValueError(f'well {name!r} must be an object with "xy"')
        _pair(w.get("xy"), f'well {name!r}: "xy" (metres)', -1.0, 1.0)
        if "z_mm" in w:
            _num(w["z_mm"], f'well {name!r}: "z_mm"', 0.0, 30.0)
        if w.get("previous_xy") is not None:
            _pair(w["previous_xy"], f'well {name!r}: "previous_xy" (metres)', -1.0, 1.0)
        if "located" in w and not isinstance(w["located"], bool):
            raise ValueError(f'well {name!r}: "located" must be true or false (false: `fd paint locate` did not find '
                             "it; take the key out once its \"xy\" is right)")
        if w.get("expected_xy") is not None:
            _pair(w["expected_xy"], f'well {name!r}: "expected_xy" (metres)', -1.0, 1.0)
    water = s["water"]
    if not isinstance(water, dict) or "rinse" not in water:
        raise ValueError('"water" must have "rinse": {"xy": [x, y]} (and "clean" when a second dish stands in reach)')
    for name, d in water.items():
        if name == "note":
            continue
        if name not in DISHES:
            raise ValueError(f'"water" has an unknown dish {name!r} ("rinse", and "clean" when there is a second one)')
        _check_dish(d, f'"water.{name}"')
    blot = s.get("blot")
    if blot is not None:
        if not isinstance(blot, dict):
            raise ValueError('"blot" must be null or an object with "xy"')
        _pair(blot.get("xy"), '"blot.xy" (metres)', -1.0, 1.0)
        if "size" in blot:
            _pair(blot["size"], '"blot.size" (metres)', 0.008, 0.25)
        if "z" in blot:
            _num(blot["z"], '"blot.z" (metres)', -0.03, 0.05)
    paints = s["paints"]
    if not isinstance(paints, dict):
        raise ValueError('"paints" must be an object {name: {"first": [...], "reload": [...]}}')
    for name, rec in paints.items():
        if not isinstance(rec, dict):
            raise ValueError(f"paint {name!r} must be an object with \"first\" and \"reload\"")
        for part in ("first", "reload"):
            steps = rec.get(part)
            if not isinstance(steps, list) or (part == "first" and not steps):
                raise ValueError(f"paint {name!r} has no list of steps {part!r}")
            for st in steps:
                if not (isinstance(st, (list, tuple)) and st and st[0] in STEP_KINDS):
                    raise ValueError(f"paint {name!r}: unknown recipe step {st!r} (a step is [\"water\", dish, passes], "
                                     "[\"scrub\", well, passes], [\"touch\", well] or [\"blot\"])")
                kind = st[0]
                if kind == "water":
                    if len(st) > 1 and st[1] not in DISHES:
                        raise ValueError(f"paint {name!r}: step {list(st)!r} names the dish {st[1]!r} (\"clean\" or \"rinse\")")
                    if len(st) > 2:
                        _count(st[2], f"paint {name!r}: passes of {list(st)!r}", 1, 6)
                elif kind in ("scrub", "touch"):
                    if len(st) < 2 or st[1] not in wells:
                        raise ValueError(f"paint {name!r}: step {list(st)!r} needs a well of the set-up "
                                         f"(wells: {', '.join(sorted(wells)) or 'none'})")
                    if kind == "scrub" and len(st) > 2:
                        _count(st[2], f"paint {name!r}: passes of {list(st)!r}", 1, 8)
                elif kind == "blot":
                    if not blot:
                        raise ValueError(f"paint {name!r}: step [\"blot\"] needs a blotting strip (\"blot\": {{\"xy\": [x, y]}}) "
                                         "in the set-up")
                    if len(st) > 1:
                        _count(st[1], f"paint {name!r}: strokes of {list(st)!r}", 1, 6)
        if rec.get("load_mm") is not None:
            _num(rec["load_mm"], f"paint {name!r}: \"load_mm\"", 10.0, 2000.0)
        awr = rec.get("after_without_rinse")
        if awr is not None and not (isinstance(awr, list) and all(isinstance(n, str) for n in awr)):
            raise ValueError(f"paint {name!r}: \"after_without_rinse\" must be a list of paint names")
    brush_of(s)
    rest = s.get("rest")
    if rest is not None:
        if not isinstance(rest, dict):
            raise ValueError('"rest" must be an object {"pan_deg", "clearance_m"}')
        if "pan_deg" in rest:
            _num(rest["pan_deg"], '"rest.pan_deg"', -180.0, 180.0)
        if "clearance_m" in rest:
            _num(rest["clearance_m"], '"rest.clearance_m"', 0.03, 0.15)


def check_setup(s, path: str = "") -> dict:
    """The set-up, or a Refusal (stage "setup") that says what is wrong with it: canvas, contact, wells with
    xy, water, blot, the brush, recipes whose steps are known and whose wells and dishes exist."""
    try:
        _check_setup(s)
    except ValueError as e:
        raise Refusal("setup", f"the painting set-up {path} cannot be used: {e}".replace("  ", " "), SETUP_NEXT) from None
    return s


def load_setup(path: str | None = None) -> dict:
    path = path or setup_path()
    try:
        with open(path, encoding="utf-8") as f:
            s = json.load(f)
    except (OSError, ValueError) as e:
        raise Refusal("setup", f"the painting set-up {path} cannot be read ({e})", SETUP_NEXT) from e
    return check_setup(s, path)


def save_setup(setup: dict, path: str | None = None) -> str:
    """Write the set-up (atomically). What could not be read back is not stored.

    ("plane_at" in a contact block is taken out: for a few hours of 2026-11-13 a plane WITHOUT a record was
    dated by it and painted with while the tool file was the older of the two. A file edited by hand, as the
    hand-over of that night asked for, made the plane of the old brush look new again; a plane without a record
    is not painted with at all now: contact_record_state.)"""
    path = path or setup_path()
    check_setup(setup, path)
    c = setup.get("contact")
    if isinstance(c, dict):
        c.pop("plane_at", None)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(setup, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)
    return path


def one_dish(setup: dict) -> bool:
    """True when rinse and clean water are the same dish: no "clean" entry, or one that points into the dish
    of "rinse" (nearer than 5 cm: how the single dish was written down on 2026-11-13)."""
    w = setup["water"]
    c = w.get("clean")
    if not c:
        return True
    return math.hypot(c["xy"][0] - w["rinse"]["xy"][0], c["xy"][1] - w["rinse"]["xy"][1]) < 0.05


# ----------------------------------------------------------------------------- the canvas and its contact plane
def contact_z(setup: dict, x: float, y: float) -> float:
    c = setup["contact"]
    return float(c["z0"]) + float(c.get("bx", 0.0)) * (float(x) - float(c.get("x0", 0.0))) \
        + float(c.get("by", 0.0)) * (float(y) - float(c.get("y0", 0.0)))


def canvas_xy(setup: dict, u_mm: float, v_mm: float) -> tuple:
    ox, oy = setup["canvas"]["origin_xy"]
    return float(ox) + float(v_mm) / 1000.0, float(oy) + float(u_mm) / 1000.0


def canvas_rect(setup: dict) -> tuple:
    """(x0, x1, y0, y1) of the canvas on the table."""
    ox, oy = setup["canvas"]["origin_xy"]
    w, h = setup["canvas"]["size_mm"]
    return float(ox), float(ox) + float(h) / 1000.0, float(oy), float(oy) + float(w) / 1000.0


def canvas_distance(setup: dict, x: float, y: float) -> float:
    """How far a table point lies OUTSIDE the canvas (m); negative: that far inside it."""
    x0, x1, y0, y1 = canvas_rect(setup)
    dx, dy = max(x0 - x, x - x1), max(y0 - y, y - y1)
    if dx <= 0.0 and dy <= 0.0:
        return max(dx, dy)
    return math.hypot(max(dx, 0.0), max(dy, 0.0))


def on_canvas(setup: dict, pts_xy, margin_m: float | None = None) -> bool:
    """Does a stroke through these table points come onto the canvas of the set-up, or nearer to it than
    margin_m (default TEST_CANVAS_MARGIN_M: a band is 5 mm wide, and the paint lands up to 9 mm from where the
    stroke was commanded)? The whole polyline counts, not its ends. What is on the canvas is where the picture
    is: test strokes there are painted on a scrap sheet only (`fd paint test --scrap-laid`)."""
    margin = TEST_CANVAS_MARGIN_M if margin_m is None else float(margin_m)
    pts = np.asarray(pts_xy, dtype=float).reshape(-1, 2)
    for a, b in zip(pts[:-1], pts[1:]) if len(pts) > 1 else [(pts[0], pts[0])]:
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / 0.002)))
        for k in range(n + 1):
            p = a + (b - a) * (k / n)
            if canvas_distance(setup, float(p[0]), float(p[1])) < margin:
                return True
    return False


def near_palette(setup: dict, pts_xy) -> str | None:
    """Why a stroke through these table points must not be made as a TEST stroke or a ladder (None: it may): it
    comes within NEAR_WELL_M of a well or NEAR_DISH_M of a dish centre of the set-up. Test strokes used to be
    checked for reach and the keep-outs only: `line:0.2578,0.170:0.2638,0.262:1.2` ran through three wells and
    over two dividers of the palette, 0.7 mm above the table, and a ladder was accepted in the red well (review
    of 2026-11-13). A swapped x, y or one wrong sign is all it takes. The whole polyline counts, not its ends."""
    pts = np.asarray(pts_xy, dtype=float).reshape(-1, 2)
    things = [(f"the well {n!r}", w["xy"], NEAR_WELL_M) for n, w in setup["wells"].items()]
    things += [(f"the {n} dish", d["xy"], NEAR_DISH_M) for n, d in setup["water"].items() if n in DISHES]
    worst = None
    for name, xy, limit in things:
        c = np.asarray(xy, dtype=float)
        d = float("inf")
        for a, b in zip(pts[:-1], pts[1:]) if len(pts) > 1 else [(pts[0], pts[0])]:
            ab = b - a
            t = 0.0 if float(ab @ ab) < 1e-12 else min(1.0, max(0.0, float((c - a) @ ab) / float(ab @ ab)))
            d = min(d, float(np.linalg.norm(a + t * ab - c)))
        if d < limit and (worst is None or d / limit < worst[0]):
            worst = (d / limit, name, d, limit, c)
    if worst is None:
        return None
    _k, name, d, limit, c = worst
    return (f"it passes {d * 1000:.0f} mm from {name} at ({c[0]:.3f}, {c[1]:.3f}): test strokes and ladders keep "
            f"{limit * 1000:.0f} mm from it (the brush is pressed on the way; a well has dividers, a dish a rim)")


# ============================================================================= the state file
def sim_dir() -> str:
    """Where `fd --sim` keeps what its simulated world remembers, outside the tree (fd.sim_state_dir(): one
    folder per tree under the temporary directory). The simulated paint state and the pictures of simulated
    paint commands go there too, unless the environment names other places: a simulated command used to write
    paint_state.sim.json into the project folder and its pictures into runs/<date>_paint, beside the real
    photos (review of 2026-11-13)."""
    for name in ("fd", "__main__"):
        fn = getattr(sys.modules.get(name), "sim_state_dir", None)
        if callable(fn):
            return fn()
    try:
        import fd
        return fd.sim_state_dir()
    except (Exception, SystemExit):      # (fd.py cannot be loaded from here: the same rule, spelled out)
        return os.path.join(tempfile.gettempdir(), f"tazuna_sim_{os.getuid()}",
                            hashlib.sha1(tazuna_paths.ROOT.encode()).hexdigest()[:10])


def state_path(sim: bool = False) -> str:
    """What the brush carries, how far each layer of a plan got, the running means of the time model:
    paint_state.json, or $FD_PAINT_STATE. A simulated run has its own file: it must never tell the real arm
    that its brush holds a paint it does not hold. With $FD_PAINT_STATE that file lies beside it (the same name
    with .sim before the extension); without, outside the tree (sim_dir())."""
    env = os.environ.get("FD_PAINT_STATE")
    if not sim:
        return env or STATE_FILE
    if env:
        root, ext = os.path.splitext(env)
        return root + ".sim" + (ext or ".json")
    return os.path.join(sim_dir(), "paint_state.sim.json")


STATE_NEXT = ("repair the file, or move it aside (it is the only record of how far each layer was painted: look at it "
              "before a layer is run again with --start); `fd paint status` reads it again")


def _check_state(d) -> None:
    if not isinstance(d, dict):
        raise ValueError("it is not a JSON object")
    paint = d.get("paint")
    if paint is not None and not (isinstance(paint, str) and paint):
        raise ValueError(f'"paint" must be a paint name or null (got {paint!r})')
    for key in ("since_load_mm", "used_mm"):
        if d.get(key) is not None:
            _num(d[key], f'"{key}"', 0.0, 1e12)
    if d.get("blot_k") is not None:
        _count(d["blot_k"], '"blot_k"', 0, 10 ** 9)
    if d.get("paint_unknown") is not None and not isinstance(d["paint_unknown"], bool):
        raise ValueError(f'"paint_unknown" must be true or false (got {d["paint_unknown"]!r})')
    prog = d.get("progress")
    if prog is not None:
        if not isinstance(prog, dict):
            raise ValueError('"progress" must be an object {"PLAN|layer|digest": strokes painted}')
        for key, n in prog.items():
            _count(n, f'"progress" of {key!r}', 0, 10 ** 6)
    timing = d.get("timing")
    if timing is not None:
        if not isinstance(timing, dict):
            raise ValueError('"timing" must be an object (the running means of the time model)')
        for key, v in timing.items():
            if key == "at":
                if not isinstance(v, str):
                    raise ValueError('"timing.at" must be a time as text')
            else:
                _num(v, f'"timing.{key}"', 0.0, 1e12)
    tool = d.get("tool")
    if tool is not None and not isinstance(tool, dict):
        raise ValueError('"tool" must be an object {"name", "attached_at"} (the tool whose brush the state describes)')


def check_state(d, path: str = "") -> dict:
    """The state, or a Refusal (stage "state") that says what is wrong with it: paint (a name or null), progress
    (whole numbers per plan and layer), timing (numbers), blot_k (a whole number), tool (an object)."""
    try:
        _check_state(d)
    except ValueError as e:
        raise Refusal("state", f"the painting state file {path} cannot be used: {e}".replace("  ", " "), STATE_NEXT) from None
    return d


def load_state(sim: bool = False) -> dict:
    """The state file ({} when there is none). A file that is there but cannot be read, or holds something else
    than a state, is a Refusal: counted as empty, the next command wrote five lines over the progress of every
    layer, and a finished layer would have been painted again from stroke 0 (review of 2026-11-13)."""
    path = state_path(sim)
    if not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError) as e:
        raise Refusal("state", f"the painting state file {path} cannot be read ({e})", STATE_NEXT) from e
    return check_state(d, path)


def save_state(state: dict, sim: bool = False) -> None:
    """Write the state (atomically). What could not be read back is not stored, and a file that is there but
    unreadable is never written over: it is kept beside the new one as paint_state.damaged_<time>.json."""
    path = state_path(sim)
    check_state(state, path)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                _check_state(json.load(f))
        except (OSError, ValueError):
            root, ext = os.path.splitext(path)
            os.replace(path, f"{root}.damaged_{time.strftime('%Y%m%d_%H%M%S')}{ext or '.json'}")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def tool_id(tool) -> dict:
    """What ties the recorded brush content to a tool: its name and the time it was attached."""
    return {"name": getattr(tool, "name", None), "attached_at": getattr(tool, "attached_at", None)}


# What stands for "the brush carries something, and it is not known what" where a paint name is expected (the
# budgets of a layer: a load onto such a brush is preceded by a wash). No paint of a set-up has this name.
UNKNOWN_PAINT = "(not known)"


def carried_paint(state: dict, tool) -> tuple:
    """(what the brush on the gripper carries by the state file, a note or None). A note means: NOT KNOWN (the
    paint is then None). The record belongs to the tool it was written for: with another tool on the gripper
    (another name or attach time: a brush that was swapped, or taped on again) nothing is known of what this
    brush carries, whatever the record says of the other one, and the note says so. An older state file
    without "tool" is taken at its word unless it was written before the tool on the gripper was attached.
    "paint_unknown" in the state: a command moved such a brush without washing or loading it, and it is still
    not known. No state at all is an empty brush, as it always was.

    "Not known" is not "nothing": a brush that carried black, taped on again and registered anew (that happened
    three times on 2026-11-13), was taken to carry nothing, and pale yellow went onto it without a wash (second
    review). Painter.load washes such a brush first. `fd paint brush --carries none` (or a paint's name) writes
    the record by hand: a brush rinsed at the tap."""
    paint = state.get("paint")
    if tool is None:
        return paint, None
    now = tool_id(tool)
    tail = (f"what the brush that is on now ({now['name']!r}, attached {now['attached_at']}) carries is NOT KNOWN: it is "
            "washed before its first load. `fd paint brush --carries none` says that it is clean (the same with the "
            "name of a paint of the set-up in place of none: that it carries that paint)")
    if state.get("paint_unknown"):
        return None, "the state file says that the brush was not washed or loaded since another one was on the gripper; " + tail
    was = state.get("tool")
    said = f"the brush carries {paint!r}" if paint is not None else "the brush carries nothing"
    if not isinstance(was, dict):
        # an older state file: it does not say whose brush it describes. Written before this tool was attached, it
        # describes another brush; else it is taken at its word
        at = state.get("at")
        if isinstance(at, str) and isinstance(now["attached_at"], str) and at < now["attached_at"]:
            return None, f"the state file says {said}, but it was written ({at}) before this tool was attached; " + tail
        return paint, None
    if was.get("name") == now["name"] and was.get("attached_at") == now["attached_at"]:
        return paint, None
    return None, (f"the state file says {said}, but that was tool {was.get('name')!r} (attached "
                  f"{was.get('attached_at')}); " + tail)


def _layer_digest(plan: dict, li: int) -> str:
    return hashlib.sha1(json.dumps(plan["layers"][li]["strokes"], sort_keys=True).encode()).hexdigest()[:10]


def progress_key(plan_path: str, plan: dict, li: int) -> str:
    """Progress is remembered per plan file, layer AND content of the layer: a plan that was made anew starts
    at its first stroke (a stale index would skip strokes, or call the layer done). The file is named by its
    REAL path: the documented command reaches the tree through a symbolic link (~/lerobot/tazuna), and keyed
    by the path as typed the finished Kaminarimon had no progress there (review of 2026-11-13)."""
    return f"{os.path.realpath(plan_path)}|{plan['layers'][li]['name']}|{_layer_digest(plan, li)}"


def _progress_aliases(state: dict, plan_path: str, plan: dict, li: int) -> list:
    """The keys of state["progress"] that mean this layer of this plan file, however the path was spelled when
    they were stored."""
    tail = f"|{plan['layers'][li]['name']}|{_layer_digest(plan, li)}"
    real = os.path.realpath(plan_path)
    out = []
    for key in (state.get("progress") or {}):
        if isinstance(key, str) and key.endswith(tail) and os.path.realpath(key[:-len(tail)]) == real:
            out.append(key)
    return out


def progress_of(state: dict, plan_path: str, plan: dict, li: int) -> int:
    """How many strokes of this layer are on the paper by the state file: under the real path, or under any
    other spelling of the same file that an earlier version stored (the furthest of them: a second coat is the
    worse mistake)."""
    prog = state.get("progress") or {}
    return max((int(prog[k]) for k in _progress_aliases(state, plan_path, plan, li)), default=0)


def painted_elsewhere(state: dict, plan_path: str, plan: dict) -> list:
    """Layers of this plan whose strokes are recorded as painted under ANOTHER plan file: the same layer name
    and the same strokes, another path. Progress is kept per plan file (a copy is another plan: it starts at
    stroke 0), so a copy of a plan that was painted would be painted again over the picture; this is what
    `fd paint preview` and `run --dry` say about it. -> [{"layer", "strokes", "of", "plan"}] (the furthest
    record of each layer)."""
    prog = state.get("progress") or {}
    real = os.path.realpath(plan_path)
    out = []
    for li, layer in enumerate(plan.get("layers") or []):
        if not (isinstance(layer, dict) and isinstance(layer.get("strokes"), list) and "name" in layer):
            continue
        tail = f"|{layer['name']}|{_layer_digest(plan, li)}"
        best = None
        for key, n in prog.items():
            if isinstance(key, str) and key.endswith(tail) and int(n) > 0 and os.path.realpath(key[:-len(tail)]) != real:
                if best is None or int(n) > best[0]:
                    best = (int(n), os.path.realpath(key[:-len(tail)]))
        if best is not None:
            out.append({"layer": layer["name"], "strokes": best[0], "of": len(layer["strokes"]), "plan": best[1]})
    return out


def set_progress(state: dict, plan_path: str, plan: dict, li: int, n: int) -> str:
    """Store the progress of a layer under the real path of its plan; keys of other spellings of that file are
    taken out (one layer, one number). Returns the key."""
    prog = state.setdefault("progress", {})
    for k in _progress_aliases(state, plan_path, plan, li):
        del prog[k]
    key = progress_key(plan_path, plan, li)
    prog[key] = int(n)
    return key


# ============================================================================= time
# Five timed calls of `fd paint run` of the evening of 2026-11-13 (thin round brush, the water at the far edge
# of the reach, hops at 20 deg/s): (strokes, travel mm, loads, loads pressed out on the blotting strip, washes of
# the brush, seconds). 1: the check pattern, 2: the red glaze, 3: the red wash (a load about every 75 mm of
# band), 4: the written lines (every load blotted), 5: the roof wash, the brush washed after the reds.
EVENING_CALLS = ((20, 447, 5, 0, 0, 289.0), (14, 584, 11, 0, 0, 469.0), (16, 839, 15, 0, 0, 694.0),
                 (19, 497, 6, 6, 0, 409.0), (6, 427, 6, 0, 1, 342.0))
_TIME_UNITS = (("per_stroke_s", "strokes"), ("per_mm_s", "mm"), ("per_load_s", "loads"), ("per_blot_s", "blots"),
               ("per_wash_s", "washes"))
TIME_KEYS = tuple(k for k, _ in _TIME_UNITS)


def fit_times(calls=EVENING_CALLS, v_stroke_mm_s: float | None = None) -> dict:
    """Seconds per stroke, per millimetre of stroke, per load, per blotting of a load and per wash of the brush,
    by least squares (of the relative errors) from timed calls [(strokes, mm, loads, blotted loads, washes,
    seconds), ...], nothing negative.

    The seconds per millimetre are NOT left to the fit: the calls of 2026-11-13 cannot tell millimetres from
    loads or strokes (a band is reloaded every 75 mm: they grow together). A free fit meets all five calls
    exactly with -5.8 s per stroke, 0.86 s per mm and -16 s per wash; a non-negative one answers 0 s per stroke,
    0.39 s per mm and 23 s per load, which would make one 180 mm stroke last 71 s (at the commanded 20 mm/s it
    lasts 9). So a millimetre costs what the commanded stroke speed says (1 / v_stroke; a real run then
    MEASURES it: times_of). With that the evening gives 4.0 s per stroke, 37 s per load, 14 s more for a load
    that is blotted and 75 s per wash, and the five calls are met within 5 %. (The estimates of that day had
    6 s, 22 s and no blotting: 26 to 35 minutes for a picture that took 50.)

    "Within 5 %" is IN-SAMPLE: four numbers fitted to the five calls they are then compared with; the seconds
    per blotting and per wash rest on one call each, which therefore cannot miss. Out of sample
    (fit_times_loo: fitted on four calls, asked for the fifth) the errors are -20, +10, -10, -21 and -22 %.
    Take an estimate as good to about a quarter until real runs have fed the means (note_run).
    -> {"per_stroke_s", "per_mm_s", "per_load_s", "per_blot_s", "per_wash_s", "worst_error": largest
        |predicted / measured - 1| over the calls (in-sample)}"""
    v = float(BW.BrushOptions().v_stroke if v_stroke_mm_s is None else v_stroke_mm_s)
    rows = np.asarray(calls, dtype=float).reshape(-1, 6)
    a, t = rows[:, :5], rows[:, 5]
    aw, tw = a / t[:, None], np.ones(len(t))
    lo = np.array([0.0, 1.0 / v, 0.0, 0.0, 0.0])
    hi = np.array([np.inf, 1.0 / v, np.inf, np.inf, np.inf])
    n = a.shape[1]
    best = None
    for code in range(3 ** n):                    # every variable: free, on its lower bound, on its upper bound
        mode = [(code // 3 ** k) % 3 for k in range(n)]
        if any(m == 2 and not np.isfinite(hi[k]) for k, m in enumerate(mode)):
            continue
        x = np.array([lo[k] if m == 1 else (hi[k] if m == 2 else 0.0) for k, m in enumerate(mode)])
        free = [k for k, m in enumerate(mode) if m == 0]
        if free:
            sol, *_ = np.linalg.lstsq(aw[:, free], tw - aw @ x, rcond=None)
            x[free] = sol
            if np.any(x < lo - 1e-9) or np.any(x > hi + 1e-9):
                continue
        rss = float(np.sum((tw - aw @ x) ** 2))
        if best is None or rss < best[0] - 1e-12:
            best = (rss, x)
    x = best[1]
    out = {k: round(float(x[i]), 4) for i, k in enumerate(TIME_KEYS)}
    out["worst_error"] = round(float(np.max(np.abs(a @ x / t - 1.0))), 3)
    return out


def fit_times_loo(calls=EVENING_CALLS) -> list:
    """Leave one out: for every timed call, the relative error (predicted / measured - 1) of the model fitted
    to the OTHER calls. What the time model is worth on a call it has not seen (fit_times' own worst_error is
    in-sample). A unit that only the left-out call has (its blottings, its wash) is then worth 0 s."""
    rows = [tuple(c) for c in calls]
    out = []
    for k, (n, mm, loads, blots, washes, sec) in enumerate(rows):
        fit = fit_times(tuple(r for i, r in enumerate(rows) if i != k))
        est = estimate_s({"strokes": n, "travel_mm": mm, "loads": loads, "blots": blots, "washes": washes}, fit)
        out.append(round(float(est) / float(sec) - 1.0, 3))
    return out


_FIT = None


def evening_fit() -> dict:
    global _FIT
    if _FIT is None:
        _FIT = fit_times()
    return dict(_FIT)


def times_of(state: dict | None = None) -> dict:
    """The seconds per stroke / mm / load / blotting / wash to estimate with: the fit of the evening, pulled
    towards what the real runs since then measured. The fit counts as the evening's own 75 strokes, 2794 mm,
    43 loads, 6 blottings and 1 wash; every real run adds its counts and its seconds (note_run), so the means
    follow the rig as it is.
    -> {"per_stroke_s", "per_mm_s", "per_load_s", "per_blot_s", "per_wash_s", "runs": real runs included, "source"}"""
    fit = evening_fit()
    rows = np.asarray(EVENING_CALLS, dtype=float)
    prior = {unit: float(rows[:, i].sum()) for i, (_k, unit) in enumerate(_TIME_UNITS)}
    t = (state or {}).get("timing") or {}
    out = {}
    for key, unit in _TIME_UNITS:
        n, sec = float(t.get(unit, 0.0) or 0.0), float(t.get(unit + "_s", 0.0) or 0.0)
        out[key] = round((fit[key] * prior[unit] + sec) / (prior[unit] + n), 4)
    runs = int(t.get("runs", 0) or 0)
    out["runs"] = runs
    out["source"] = "five timed calls of 2026-11-13" + (f" and {runs} real run(s) since" if runs else "")
    return out


def note_run(state: dict, counts: dict, seconds: float, spent: dict, ok: bool = True) -> dict | None:
    """Add one REAL run to the running means of the time model (state["timing"]; the caller saves the state).
    counts: {"strokes", "travel_mm", "loads", "blots", "washes"} of the command; seconds: the whole command;
    spent: {"travel_s": the brush on the paper, "load_s" / "blot_s" / "wash_s": inside the loads, their
    blottings and the washes}. Whatever the command took besides (hops, lifts, the way to the photo pose, the
    photo) is the strokes' overhead, so that the five products add up to the whole command again. A run
    without a finished stroke is not counted, and neither is one that did not end ok (ok=False: stopped,
    interrupted, the servos gone): the seconds of the load that failed and of the way out would be booked as
    overhead per stroke, and nothing takes them out of the means again (four such runs made 9 s per stroke of
    4, review of 2026-11-13). A run that ended at its time limit ended ok."""
    strokes, mm = float(counts.get("strokes", 0) or 0), float(counts.get("travel_mm", 0.0) or 0.0)
    if not ok or not (strokes > 0 and mm > 0 and math.isfinite(float(seconds)) and float(seconds) > 0):
        return None
    part = {k: max(0.0, float(spent.get(k, 0.0) or 0.0)) for k in ("travel_s", "load_s", "blot_s", "wash_s")}
    over = max(0.0, float(seconds) - sum(part.values()))
    t = dict(state.get("timing") or {})
    for unit, n, sec in (("strokes", strokes, over), ("mm", mm, part["travel_s"]),
                         ("loads", float(counts.get("loads", 0) or 0), part["load_s"]),
                         ("blots", float(counts.get("blots", 0) or 0), part["blot_s"]),
                         ("washes", float(counts.get("washes", 0) or 0), part["wash_s"])):
        t[unit] = round(float(t.get(unit, 0.0) or 0.0) + n, 1)
        t[unit + "_s"] = round(float(t.get(unit + "_s", 0.0) or 0.0) + sec, 1)
    t["runs"] = int(t.get("runs", 0) or 0) + 1
    t["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    state["timing"] = t
    return t


def estimate_s(counts: dict, times: dict) -> float:
    return (float(counts.get("strokes", 0)) * times["per_stroke_s"] + float(counts.get("travel_mm", 0.0)) * times["per_mm_s"]
            + float(counts.get("loads", 0)) * times["per_load_s"] + float(counts.get("blots", 0)) * times["per_blot_s"]
            + float(counts.get("washes", 0)) * times["per_wash_s"])


# ============================================================================= plans
def plan_brush_kind(plan: dict) -> str | None:
    """The kind of brush a plan was made for: what its "brush" block says; "flat" for a plan WITHOUT a block
    (the plans of the morning of 2026-11-13 have none, and they are flat-brush plans: with a round brush on
    the gripper their zig-zag fills come out as thin lines); None for a block that does not say."""
    if not isinstance(plan, dict) or "brush" not in plan or plan["brush"] is None:
        return "flat"
    b = plan["brush"]
    return b.get("kind") if isinstance(b, dict) else None


RETRACE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "paintings", "kaminarimon_letters_touchup.json")


def _is_retrace(plan: dict) -> bool:
    """Is this plan the lettering touch-up of 2026-11-13 as it was run (the shipped record, or a copy of it)?"""
    try:
        with open(RETRACE_FILE, encoding="utf-8") as f:
            return json.load(f).get("layers") == plan.get("layers")
    except (OSError, ValueError, AttributeError):
        return False


def retrace_command(out: str | None = None) -> str:
    """One command line that writes the plan kaminarimon.retrace() returns (the lettering touch-up WITH the
    brush block of the picture) as a file: the shipped record of what ran stays as it is. `out`: where (default:
    kaminarimon_retrace_plan.json in the temporary folder)."""
    out = out or os.path.join(tempfile.gettempdir(), "kaminarimon_retrace_plan.json")
    code = (f"import sys, json; sys.path.insert(0, {os.path.dirname(RETRACE_FILE)!r}); import kaminarimon; "
            f"json.dump(kaminarimon.retrace(), open({out!r}, 'w'))")
    return f'{sys.executable} -c "{code}"'


def pan_axis_mm(setup: dict) -> list:
    """The robot's pan axis in canvas millimetres (what a plan must carry as pan_axis_mm)."""
    ox, oy = setup["canvas"]["origin_xy"]
    return [round((0.0 - float(oy)) * 1000.0, 1), round((float(K._X_PAN) - float(ox)) * 1000.0, 1)]


def press_m(setup: dict, press: float) -> float:
    """Metres below first touch for a stroke's press factor; SafetyError beyond what brush work allows (a
    plan is not quietly painted shallower than it says)."""
    try:
        p = float(setup.get("press_mm", 2.0)) / 1000.0 * float(press)
    except (TypeError, ValueError):
        p = float("nan")
    if not (math.isfinite(p) and 0.0 <= p <= BW.MAX_PRESS_M + 1e-9):
        raise SafetyError(f"press {press!r} x {float(setup.get('press_mm', 2.0)):g} mm is outside 0 .. "
                          f"{BW.MAX_PRESS_M * 1000:.0f} mm below first touch (the limit of brush work); nothing moved")
    return p


def plan_problems(plan, setup: dict) -> list:
    """Why this plan cannot be painted with this set-up (texts, the most fundamental first; [] = it can):
    not a plan, another canvas, another pan axis, another KIND of brush, paints the set-up does not have,
    strokes that leave the canvas, are shorter than 1 mm, take more than one load or press deeper than
    brush work allows."""
    if not (isinstance(plan, dict) and isinstance(plan.get("layers"), list) and plan["layers"]):
        return ["the file is not a painting plan (no \"layers\")"]
    out = []
    w, h = (float(v) for v in setup["canvas"]["size_mm"])
    try:
        got = [round(float(v), 1) for v in plan.get("canvas_mm", [])]
    except (TypeError, ValueError):
        got = []
    if got != [round(w, 1), round(h, 1)]:
        out.append(f"the plan is for a canvas of {plan.get('canvas_mm')} mm, the set-up has {[w, h]} mm")
    pan = pan_axis_mm(setup)
    pa = plan.get("pan_axis_mm")
    try:
        bad_pan = abs(float(pa[0]) - pan[0]) > 3.0 or abs(float(pa[1]) - pan[1]) > 3.0
    except (TypeError, ValueError, IndexError):
        bad_pan = True
    if bad_pan:
        out.append(f"the plan was made for the pan axis at {pa} mm on the canvas; with this set-up it is at {pan} mm: "
                   "make the plan anew (where a stroke lies for the arm depends on it)")
    brush = brush_of(setup)
    kind = plan_brush_kind(plan)
    if kind not in ("round", "flat"):
        out.append(f"the plan's \"brush\" must be an object {{\"kind\": \"round\" | \"flat\", ...}} (got {plan.get('brush')!r}); "
                   "a plan without one is a plan for the flat brush of the morning of 2026-11-13")
    elif kind != brush["kind"]:
        # Which brush is on is said by what is ON the gripper, never by a plan: no text here advises to change the
        # set-up's brush kind so that a plan fits (following that advice, the set-up said "flat" with a round
        # brush on: second review of 2026-11-13).
        if not isinstance(plan.get("brush"), dict):
            text = ("the plan has no \"brush\" block: it does not say which brush it was made for (it is taken for a plan "
                    "of the flat brush of the morning of 2026-11-13, whose plans have none), and the set-up's brush is "
                    f"{brush['kind']}. Make the plan anew for the brush that is on, so that it carries its block")
            if _is_retrace(plan):
                text += (". This is the lettering touch-up of 2026-11-13 as the executor ran it; with its block it is what "
                         f"kaminarimon.retrace() returns: `{retrace_command()}` writes that plan")
        elif not brush["stated"]:
            text = (f"the plan was made for a {kind} brush, and the set-up has no \"brush\" block (that means the flat brush "
                    f"of the morning of 2026-11-13). When a {kind} brush is on the gripper, the set-up has to say so "
                    "first: `fd paint status` names the command that stores which brush is on; else make the plan anew "
                    "for the brush that is on")
        else:
            text = (f"the plan was made for a {kind} brush, the set-up's brush is {brush['kind']}: make the plan anew for "
                    "the brush that is on (a plan is not painted with another kind of brush than its own)")
        out.append(text)
    for li, layer in enumerate(plan["layers"]):
        if not (isinstance(layer, dict) and isinstance(layer.get("name"), str) and isinstance(layer.get("strokes"), list)
                and "paint" in layer):
            out.append(f"layer {li} has no name, paint or strokes")
            continue
        if not isinstance(layer["paint"], str):
            out.append(f"layer {layer['name']!r}: its paint must be the name of a paint of the set-up (got {layer['paint']!r})")
            continue
        if layer["paint"] not in setup["paints"]:
            out.append(f"layer {layer['name']!r} wants paint {layer['paint']!r}, which the set-up does not have "
                       f"(paints: {', '.join(sorted(setup['paints']))})")
        for i, st in enumerate(layer["strokes"]):
            where = f"{layer['name']}[{i}]"
            try:
                pts = np.asarray(st["pts"], dtype=float)
                press = float(st.get("press", 1.0))
                if pts.ndim != 2 or pts.shape[1] != 2:          # ([[u, v, z], ...] reshaped into pairs made 3 points of 2)
                    raise ValueError("points must be pairs")
            except (TypeError, ValueError, KeyError, AttributeError):
                out.append(f"{where}: not a stroke ({{\"pts\": [[u, v], ...], \"press\": 1.0}}: every point a pair of "
                           "canvas millimetres)")
                continue
            if len(pts) < 2 or not np.all(np.isfinite(pts)):
                out.append(f"{where}: fewer than two points")
                continue
            ln = RD.length_mm(pts)
            if ln < 1.0:
                out.append(f"{where}: shorter than 1 mm")
            if pts[:, 0].min() < 0 or pts[:, 1].min() < 0 or pts[:, 0].max() > w or pts[:, 1].max() > h:
                out.append(f"{where}: leaves the canvas")
            try:
                press_m(setup, press)
            except SafetyError as e:
                out.append(f"{where}: {e}")
                continue
            limit = reload_rule(brush, None, setup["paints"].get(layer["paint"]))[1]
            if use_mm(brush, ln, press) > limit + 1e-6:
                out.append(f"{where}: {ln:.0f} mm at press {press:g} takes more than one load of {layer['paint']} "
                           f"({limit:.0f} mm{' of fine line' if brush['kind'] == 'round' else ''})")
    return out


def reload_rule(brush: dict, reload_mm: float | None = None, recipe: dict | None = None) -> tuple:
    """(use(length_mm, press), limit, name): how run_layer decides to reload. By the brush of the set-up (a
    round brush: paint used, in mm of fine line; a flat one: plain travel), a recipe's own "load_mm" when that
    is less (a brush that is blotted after loading carries less: the written lines of 2026-11-13 were reloaded
    every 110 mm, the retraced ones every 60), or, with reload_mm (`--reload-mm`), after that much plain travel
    whatever brush and recipe say."""
    if reload_mm is not None:
        try:
            lim = float(reload_mm)
        except (TypeError, ValueError):
            lim = float("nan")
        if not (math.isfinite(lim) and 5.0 <= lim <= 2000.0):
            raise Refusal("options", f"--reload-mm {reload_mm!r}: millimetres of travel on one load, 5 .. 2000",
                          "`fd paint run --help`")
        return (lambda ln, press: float(ln)), lim, "travel"
    limit = float(brush["load_mm"])
    own = (recipe or {}).get("load_mm")
    if own is not None:
        limit = min(limit, float(own))
    if brush["kind"] == "round":
        return (lambda ln, press: use_mm(brush, ln, press)), limit, "paint"
    return (lambda ln, press: float(ln)), limit, "travel"


def layer_budget(setup: dict, layer: dict, paint_now=None, carried: float = 1e9, start: int = 0, stop: int | None = None,
                 reload_mm: float | None = None) -> dict:
    """What run_layer will do with strokes start .. stop-1 of a layer, by its own rule: {"strokes", "travel_mm",
    "loads", "blots" (loads that are pressed out on the blotting strip), "washes", "paint": on the brush
    afterwards, "carried": used of the last load}. paint_now: what the brush carries when it starts (None:
    clean or unknown); carried: how much of that load is used (a new command always starts with a load: 1e9)."""
    paint = layer["paint"]
    recipe = setup["paints"].get(paint) or {}
    use_of, limit, _rule = reload_rule(brush_of(setup), reload_mm, recipe)
    strokes = layer["strokes"]
    stop = len(strokes) if stop is None else min(int(stop), len(strokes))
    n = loads = blots = washes = 0
    mm = 0.0
    for st in strokes[int(start):stop]:
        ln = RD.length_mm(st["pts"])
        use = use_of(ln, float(st.get("press", 1.0)))
        if paint_now != paint or carried + use > limit:
            first = paint_now != paint
            if first and paint_now is not None and paint_now not in (recipe.get("after_without_rinse") or []):
                washes += 1
            blots += sum(1 for step in (recipe.get("first" if first else "reload") or []) if step and step[0] == "blot")
            loads += 1
            paint_now, carried = paint, 0.0
        carried += use
        n += 1
        mm += ln
    return {"strokes": n, "travel_mm": round(mm), "loads": loads, "blots": blots, "washes": washes, "paint": paint_now,
            "carried": carried}


def plan_budget(setup: dict, plan: dict, state: dict | None = None, paint_now=None, reload_mm: float | None = None) -> dict:
    """Strokes, travel, loads, washes and minutes of every layer of a plan, painted one layer per call in the
    order of the plan with the brush as the layer before left it, and of the whole. Minutes by times_of(state).
    -> {"layers": [{"name", "paint", "strokes", "travel_mm", "loads", "blots", "washes", "minutes"}], "strokes",
        "travel_mm", "loads", "blots", "washes", "minutes", "times"}"""
    times = times_of(state)
    layers, total = [], {"strokes": 0, "travel_mm": 0, "loads": 0, "blots": 0, "washes": 0}
    for layer in plan["layers"]:
        b = layer_budget(setup, layer, paint_now, 1e9, reload_mm=reload_mm)
        paint_now = b["paint"]
        row = {"name": layer["name"], "paint": layer["paint"], **{k: b[k] for k in total}}
        row["minutes"] = round(estimate_s(row, times) / 60.0, 1)
        layers.append(row)
        for k in total:
            total[k] += row[k]
    return {"layers": layers, **total, "minutes": round(estimate_s(total, times) / 60.0, 1), "times": times}


# ============================================================================= what a brush must be before it moves
# The next command for a brush whose tip was not measured. Only `--save` writes the record that it was: a plain
# `fd calib-air` measures and stores nothing.
# (Third review, T12: what stands in backticks, or at the beginning of a `next`, is ONE command that works as it is
# typed. The folder of photos that were already taken is a value this program does not know: it is named in
# words, with the options that go with it. The same command with "DIR" for the folder, typed as it was printed,
# is refused.)
FROM_PHOTOS = "fd calib-air with --from, the folder of those photos, and --save"
TIP_NEXT = ("fd calib-air --save   (it looks at the tool first, then measures its tip and stores it: only --save writes "
            "the record that the tip was measured; photos of this brush that were already taken with the jaw as it "
            f"stands: {FROM_PHOTOS})")
# ... once the jaw is HELD by a squeeze on record: photos from before that are photos of another jaw state. That
# command refuses them AS THEY ARE, and its refusal names the way back (the hold off the record, the photos
# stored, the jaw held again): this text said "cannot be stored any more" while that refusal named a way that
# works as typed (third review, T9)
TIP_NEXT_HELD = ("fd calib-air --save   (it looks at the tool first, then measures its tip with the jaw held as it is and "
                 "stores it: only --save writes the record that the tip was measured; photos taken BEFORE the jaw was "
                 "closed and held are refused as they are; the refusal names the way back)")
# ... while the jaw is shut and NOT yet held: photos that were taken with the jaw as it stands are stored first.
# (The other order, jaw first, is refused at the third command for the rig's own files of 2026-11-13: its
# photos of that evening then count as taken "with no jaw hold on record".) {jaw}: what closes and holds the jaw.
TIP_NEXT_FIRST = ("ONLY when photos of this brush exist that were taken with the jaw as it stands now: " + FROM_PHOTOS
                  + "   (no motion: their tip is stored BEFORE the jaw is closed and held, because a hold on record "
                    "makes them photos of another jaw state. No such photos: {jaw}, then `fd calib-air --save` flies a "
                    "new measurement)")
TIP_WORD_NEXT = ". Only after looking at a photo of gripper and brush yourself: this command with --tip-looked-at"


def tip_next(tool, cfg: dict | None = None) -> str:
    """The next command for a brush whose tip was not measured, by the state of its jaw (TIP_NEXT_FIRST,
    TIP_NEXT_HELD, TIP_NEXT). One order for `fd paint status`, `check` and the refusal of every moving command."""
    if getattr(tool, "gripper_goal_raw", None) is not None:
        return TIP_NEXT_HELD
    if cfg is None:
        try:
            from robot import load_config
            cfg = load_config()
        except Exception:               # (no configuration to read: the touch position of the defaults)
            cfg = {}
    if jaw_hold_note(tool, cfg) is None:
        return TIP_NEXT
    jaw = ("`fd tool jaw close`" if getattr(tool, "on_jaw", None) == "moving_outer" else
           "`fd tool jaw mount moving_outer` after looking at a photo of gripper and brush, `fd tool jaw close`")
    return TIP_NEXT_FIRST.format(jaw=jaw)


def require_tool(tool):
    """Refusal (stage "tool") when no brush is registered."""
    if tool is None:
        raise Refusal("tool", "no brush is registered: painting needs a tool on the gripper",
                      "`fd tool attach --help` shows how the brush on the gripper is registered; then `fd calib-air --save` "
                      "for its tip (`fd tool status` shows what is on)")
    return tool


def require_measured_tip(tool):
    """Refusal (stage "tool", next: `fd calib-air`) for a brush whose tip was never measured. On 2026-11-13 a
    tool was registered as "8 cm along the fingers" from an instruction, without looking: its tip was 17 cm
    out. A painting command dips the tip to 2.5 mm above the table by the registered offset; with that tool the
    first dip would have pushed the brush 9 cm into the dish or the palette. The rule of that day (a photo of
    gripper and tool is looked at before the first motion that relies on the registered length) is what
    `fd calib-air` does first, and what the operator vouches for with --tip-looked-at."""
    if not BW.tip_is_measured(tool):
        raise Refusal("tool", BW.unmeasured_text(tool), tip_next(tool) + TIP_WORD_NEXT, tool=getattr(tool, "name", None),
                      tip_measured=False,
                      say=(f"実行しませんでした。筆（ツール「{getattr(tool, 'name', '?')}」）の先端の位置がまだ測定されていません"
                           "（ツールのファイルに測定の記録がありません）。fd calib-air で測定してください。グリッパーと筆が一緒に"
                           "写った写真で先端の位置を確かめた場合に限り、--tip-looked-at を付けて実行できます。"
                           "アームは動かしていません。"))
    return tool


def wells_of(setup: dict, paint: str) -> list:
    """The wells a paint's recipe scrubs or touches (its "first" and its "reload" steps), in their order."""
    rec = setup["paints"].get(paint) or {}
    out = []
    for st in list(rec.get("first") or []) + list(rec.get("reload") or []):
        if st and st[0] in ("scrub", "touch") and len(st) > 1 and st[1] not in out:
            out.append(st[1])
    return out


def wells_not_located(setup: dict) -> list:
    """The wells the last `fd paint locate` did not find ("located": false in the set-up)."""
    return [n for n, w in setup["wells"].items() if isinstance(w, dict) and w.get("located") is False]


def require_wells(setup: dict, paint: str) -> None:
    """Refusal (stage "wells") when `paint` is loaded from a well that `fd paint locate` did not find in its
    photo: the place the set-up has for it is where it was before the palette moved, and a load would scrub
    2.5 mm above the table there, on bare palette or on a divider. Asked before anything moves."""
    bad = [w for w in wells_of(setup, paint) if w in wells_not_located(setup)]
    if bad:
        names = ", ".join(repr(w) for w in bad)
        raise Refusal("wells", f"paint {paint!r} is loaded from the well {names}, which the last `fd paint locate` did not "
                               "find in its photo: the place the set-up has for it is where it was BEFORE the palette "
                               "moved. Nothing moved",
                      "`fd paint locate` again, with the whole palette in the photo and unshadowed (its picture shows what "
                      "was found; a blob that is too small or too pale is not found: put paint into the well). Or write "
                      "the well's \"xy\" into the set-up by hand and take its \"located\": false out (the file: "
                      "`fd paint status`)",
                      wells_not_located=bad,
                      say=(f"実行しませんでした。絵の具「{paint}」を取る場所（{'、'.join(bad)}）は、前回の fd paint locate の写真で"
                           "見つからず、パレットが動く前の位置のままです。パレット全体が写るようにして fd paint locate を"
                           "やり直してください（絵の具が少ないと見つかりません）。アームは動かしていません。"))


def jaw_hold_note(tool, cfg: dict) -> str | None:
    """What `fd paint status` and `check` say about the jaw (brushwork.jaw_note, with the touch position of
    the configuration): the jaw is shut and no hold is recorded."""
    try:
        touch = cfg["gripper_touch_raw"] if "gripper_touch_raw" in cfg else DEFAULT_CONFIG["gripper_touch_raw"]
    except (KeyError, TypeError, AttributeError):
        touch = None
    return BW.jaw_note(tool, touch)


# ============================================================================= the painter
class Painter:
    """The brush with a set-up: knows where water and paints are, what is on the brush, how much of its load
    is used, and where the canvas lies."""

    def __init__(self, arm, setup: dict, opts: BW.BrushOptions | None = None, photo=None, out_dir: str | None = None,
                 log=None, unmeasured_tip_ok: bool = False):
        """unmeasured_tip_ok: for what only computes (`fd paint check`, `preview`), and for the operator who has
        looked at a photo of gripper and brush (`--tip-looked-at`). Without it a tool whose tip was never
        measured is refused here, before anything can move (require_measured_tip)."""
        require_tool(getattr(arm, "tool", None))
        if not unmeasured_tip_ok:
            require_measured_tip(arm.tool)
        self.arm, self.s = arm, setup
        self.log = log or (lambda *a, **k: None)
        self.br = BW.BrushRun(arm, opts, log=self.log, unmeasured_tip_ok=True)      # (asked just above, with the next command)
        self.tip_measured = BW.tip_is_measured(arm.tool)
        self.brush = brush_of(setup)
        self.photo_fn, self.out_dir = photo, out_dir
        self.tz = self.br.tz
        self.clock = self.br.clock
        self.paint = None            # what is on the brush (a paint name of the set-up), None = nothing
        self.unknown = False         # ... or: it carries something, and it is NOT KNOWN what (carried_paint's note)
        self.washed_unknown = False  # such a brush was washed before its first load (said in the result)
        self.since_load = 1e9        # mm of stroke travel since the last load
        self.used = 1e9              # the same in the unit of the brush's load (mm of fine line for a round brush)
        self.loads = 0
        self.blots = 0               # loads that were pressed out on the blotting strip (a recipe's "blot" step)
        self.washes = 0              # clean() between two unrelated paints
        self.spent = {"load_s": 0.0, "blot_s": 0.0, "wash_s": 0.0, "travel_s": 0.0, "stroke_s": 0.0}   # by the arm's clock
        self.t0 = self.clock()
        self.canvas_pitch = None
        self.lower_m = 0.0           # this command paints with the contact plane so much LOWER than measured (set_lower)
        self.blot_k = 0              # strips of the blotting pad used so far (they cycle)
        self.photos = []
        self.last_photo = None

    # ------------------------------------------------------------------ geometry
    def zc(self, x: float, y: float) -> float:
        return contact_z(self.s, x, y) - self.lower_m

    def set_lower(self, mm: float) -> float:
        """Paint with the contact plane `mm` lower than the set-up has it, for this command only (0 .. MAX_LOWER_MM).
        What the bands of 2026-11-13 needed: the plane through the four corners and the centre was 1.5 mm too
        high at the far middle of the sheet, the first band there came out a thin line, and every band after
        it was run with the plane lowered by that much (by editing z0 in the file; the written lines were run
        with the plane as measured). The depth guard is not moved by it: nothing goes below table_z - 6 mm.
        And it is no second allowance on top of the press: press and lowering TOGETHER stay within MAX_DEPTH_M
        (7.5 mm below first touch, what those bands ran at): depth_m refuses a stroke that asks for more."""
        try:
            v = float(mm)
        except (TypeError, ValueError):
            v = float("nan")
        if not (math.isfinite(v) and 0.0 <= v <= MAX_LOWER_MM):
            raise Refusal("options", f"--lower-mm {mm!r}: 0 .. {MAX_LOWER_MM:g} mm (how much lower than the measured "
                                     f"contact plane this command paints; press and lowering together at most "
                                     f"{MAX_DEPTH_M * 1000:g} mm)", "`fd paint run --help`")
        self.lower_m = v / 1000.0
        self.canvas_pitch = None
        return v

    def depth_m(self, press: float) -> float:
        """How far below the measured first-touch plane a stroke of this press factor runs in this command:
        its press (press_m: at most 6 mm) plus the lowering (set_lower). SafetyError beyond MAX_DEPTH_M: the
        two are counted together (press 1.2 x 5 mm with --lower-mm 2 would be 8 mm: deeper than any stroke ran
        on the real arm)."""
        p = press_m(self.s, press)
        if p + self.lower_m > MAX_DEPTH_M + 1e-9:
            raise SafetyError(f"press {press:g} x {float(self.s.get('press_mm', 2.0)):g} mm with the plane lowered by "
                              f"{self.lower_m * 1000:g} mm (--lower-mm) is {(p + self.lower_m) * 1000:.1f} mm below first "
                              f"touch: press and lowering together may be {MAX_DEPTH_M * 1000:g} mm (what the bands of "
                              f"2026-11-13 ran at; --lower-mm {max(0.0, (MAX_DEPTH_M - p) * 1000):g} at most for this "
                              "press); nothing moved")
        return p + self.lower_m

    def canvas_xy(self, u_mm: float, v_mm: float) -> tuple:
        return canvas_xy(self.s, u_mm, v_mm)

    def pan_axis_mm(self) -> list:
        return pan_axis_mm(self.s)

    def canvas_points(self) -> list:
        """(name, x, y) of the corners, the edge middles and the centre of the canvas."""
        w, h = self.s["canvas"]["size_mm"]
        names = (("near-left", 0, 0), ("near-right", w, 0), ("far-left", 0, h), ("far-right", w, h), ("centre", w / 2, h / 2),
                 ("near-middle", w / 2, 0), ("far-middle", w / 2, h), ("left-middle", 0, h / 2), ("right-middle", w, h / 2))
        return [(n, *self.canvas_xy(u, v)) for n, u, v in names]

    def pitch_for_canvas(self) -> float:
        """One brush pitch for the whole picture (so that the band has the same width everywhere)."""
        if self.canvas_pitch is None:
            pts = []
            for _n, x, y in self.canvas_points():
                z = self.zc(x, y)
                pts += [np.array([x, y, z - BW.MAX_PRESS_M]), np.array([x, y, self.tz + self.br.o.above_h])]
            self.canvas_pitch = self.br.pitch_at(pts, want=self.s["canvas"].get("pitch_deg"))
        return self.canvas_pitch

    def contact_state(self) -> dict:
        """Does the contact plane of the set-up hold for the brush as it is now? (contact_record_state)"""
        return contact_record_state(self.s, self.arm.tool, self.pitch_for_canvas())

    def require_plane(self, what: str = "painting") -> dict:
        """Refusal (stage "contact") unless the contact plane is recorded for the brush as it is (a plane
        without a record included: next `fd paint contact --adopt`); else the state."""
        cs = self.contact_state()
        if not cs["ok"]:
            raise Refusal("contact", cs["reason"] + f". {what[:1].upper()}{what[1:]} with it would press the brush by a wrong amount",
                          cs["next"], contact={k: cs[k] for k in ("recorded", "ok", "why")})
        return cs

    # ------------------------------------------------------------------ water and paint
    def _radial(self, xy) -> np.ndarray:
        d = np.array([float(xy[0]) - float(K._X_PAN), float(xy[1])])
        return d / max(float(np.linalg.norm(d)), 1e-9)

    def dish(self, which: str = "clean") -> dict:
        """The dish for "clean" or "rinse" water. With ONE dish in the set-up both are that dish."""
        if which not in DISHES:
            raise SafetyError(f"unknown dish {which!r} (\"clean\" or \"rinse\")")
        w = self.s["water"]
        return w.get(which) or w["rinse"]

    @staticmethod
    def _water_pattern(passes: int = 2, amp: float = 0.007) -> list:
        pat = []
        for _ in range(int(passes)):
            pat += [(amp, 0.0, 0.0), (-amp, 0.0, 0.0)]
        return pat

    @staticmethod
    def _rinse_pattern(rounds: int = 2) -> list:
        a, b = 0.013, 0.009
        pat = []
        for _ in range(max(1, int(rounds))):
            pat += [(a, 0, 0), (-a, 0, 0), (a, 0, 0), (-a, 0, 0), (0, b, 0), (0, -b, 0), (0, b, 0), (0, -b, 0)]
            pat += [(0, 0, 0.012), (0, 0, 0)]              # out of the water and in again
        return pat

    def _clean_dish_pattern(self, xy) -> list:
        # the clean dish stands at the edge of the arm's reach: sideways (tangential) and towards the robot only
        rad = self._radial(xy)
        tan = np.array([-rad[1], rad[0]])

        def v(t, r, dz=0.0):
            return (float(t * tan[0] + r * rad[0]), float(t * tan[1] + r * rad[1]), dz)

        return [v(0.008, 0), v(-0.008, 0), v(0.008, 0), v(-0.008, 0), v(0, -0.008), v(0, 0), v(0, 0, 0.012), v(0, 0),
                v(0.008, 0), v(-0.008, 0)]

    def _scrub_pattern(self, xy, passes: int = 4, amp: float = 0.006, side: float = 0.003) -> list:
        rad = self._radial(xy)
        tan = np.array([-rad[1], rad[0]])
        pat = []
        n = max(1, int(passes))
        for k in range(n):
            s = side * ((k / (n - 1)) * 2.0 - 1.0) if n > 1 else 0.0
            for a in (amp, -amp):
                pat.append((float(a * rad[0] + s * tan[0]), float(a * rad[1] + s * tan[1]), 0.0))
        return pat

    def _touch_pattern(self, xy) -> list:
        rad = self._radial(xy)
        return [(float(0.003 * rad[0]), float(0.003 * rad[1]), 0.0), (float(-0.003 * rad[0]), float(-0.003 * rad[1]), 0.0)]

    def water(self, which: str = "clean", passes: int = 2, amp: float = 0.007) -> dict:
        w = self.dish(which)
        r = self.br.dab(w["xy"], self.tz + float(w.get("z_mm", 6.0)) / 1000.0, pattern=self._water_pattern(passes, amp),
                        dwell_s=0.3, what=f"water ({which})")
        self.arm._sleep(0.8)         # drops fall back into the dish
        return r

    def rinse(self, rounds: int = 2) -> dict:
        """Wash the brush. In the rinse dish the bristles are pressed onto the bottom and dragged to and fro in
        both directions (that squeezes the paint out of the belly of the brush; swishing the tip alone left
        enough cerulean in it to turn a yellow wash green, 2026-11-13), lifted and plunged again `rounds`
        times; then the same, shorter, in the clean dish (with ONE dish: in the same water once more)."""
        w = self.dish("rinse")
        z = self.tz + float(w.get("scrub_z_mm", 3.0)) / 1000.0
        r = self.br.dab(w["xy"], z, pattern=self._rinse_pattern(rounds), v_mm_s=35.0, dwell_s=0.3, what="rinse")
        self.arm._sleep(0.8)
        c = self.dish("clean")
        zc = self.tz + float(c.get("scrub_z_mm", 4.5)) / 1000.0
        self.br.dab(c["xy"], zc, pattern=self._clean_dish_pattern(c["xy"]), v_mm_s=35.0, dwell_s=0.3, what="rinse (clean dish)")
        self.arm._sleep(0.8)
        self.paint, self.since_load, self.used = None, 1e9, 1e9
        self.unknown = False         # (washed: it carries nothing now, and that is known)
        return r

    def scrub(self, well: str, passes: int = 4, amp: float = 0.006, side: float = 0.003) -> dict:
        """Work paint into the brush: back and forth through the blob along the arm's radial direction."""
        w = self.s["wells"][well]
        r = self.br.dab(w["xy"], self.tz + float(w.get("z_mm", 2.5)) / 1000.0,
                        pattern=self._scrub_pattern(w["xy"], passes, amp, side), what=f"paint ({well})")
        self.arm._sleep(0.5)
        return r

    def touch(self, well: str) -> dict:
        """A light pick-up: the brush tip into the blob and out (for a tint)."""
        w = self.s["wells"][well]
        return self.br.dab(w["xy"], self.tz + float(w.get("z_mm", 2.5)) / 1000.0 + 0.0005,
                           pattern=self._touch_pattern(w["xy"]), what=f"touch ({well})")

    def _blot_rows(self, strokes: int, k0: int = 0) -> tuple:
        """(z of the pad's top, [((x0, y), (x1, y)), ...]) of the next `strokes` pulls on the blotting pad."""
        b = self.s["blot"]
        cx, cy = b["xy"]
        sx, sy = b.get("size", [0.03, 0.05])
        z = float(b.get("z", contact_z(self.s, cx, cy) + 0.002))
        n = max(1, int(sy / 0.011))
        rows = []
        for i in range(int(strokes)):
            y = cy - sy / 2.0 + 0.0055 + 0.011 * ((int(k0) + i) % n)
            rows.append(((cx + sx / 2.0 - 0.004, y), (cx - sx / 2.0 + 0.004, y)))
        return z, rows

    def blot(self, strokes: int = 2) -> dict | None:
        """Press a soaked brush out on the blotting pad (a folded paper towel taped down), when the set-up has
        one: short pull strokes towards the robot, each on a fresh strip of the pad. The pad takes the water and
        what paint is left after a rinse. {"xy": centre, "z": height of its top (absolute), "size": [along x,
        along y] in metres}. The strips cycle so that the pad wears evenly."""
        if not self.s.get("blot"):
            return None
        z, rows = self._blot_rows(strokes, self.blot_k)
        out = None
        for a, b in rows:
            out = self.br.stroke([a, b], z, 0.003, v_mm_s=15.0)
            self.blot_k += 1
        return out

    def clean(self) -> dict:
        """A brush clean enough for a pale colour after a dark one: rinse, blot, rinse again, blot."""
        self.rinse()
        self.blot(2)
        r = self.rinse(rounds=1)
        self.blot(1)
        return r

    def load(self, paint: str) -> dict:
        """Put `paint` on the brush by its recipe: the "first" steps when the brush carries something else (it is
        washed before, unless the recipe lists what it carries under "after_without_rinse": a tint, then the
        stronger tint of the same paint), the "reload" steps when it already carries this paint. A brush whose
        content is NOT KNOWN (self.unknown: the record is another tool's) is washed first too: it is not "clean".

        From the first step that touches a well the brush CARRIES this paint (how much is not known until the
        recipe is through): a load that is stopped on the way to the blotting strip leaves a brush full of
        black, and recorded as clean it took pale yellow next without a wash (review of 2026-11-13; hops were
        stopped four times that evening)."""
        recipes = self.s["paints"]
        if paint not in recipes:
            raise Refusal("paint", f"paint {paint!r} is not in the set-up (paints: {', '.join(sorted(recipes))})",
                          "`fd paint status` lists the paints of the set-up")
        require_wells(self.s, paint)       # (a well `fd paint locate` did not find: before the wash, before anything)
        t0 = self.clock()
        first = self.paint != paint
        washed = unknown = False
        if first and self.unknown:
            self.clean()                 # (rinse() takes the "not known" off)
            washed = unknown = self.washed_unknown = True
        elif first and self.paint is not None:
            # paints that follow one another without washing (e.g. the blues, light to dark) say so in the set-up
            if self.paint not in (recipes[paint].get("after_without_rinse") or []):
                self.clean()
                washed = True
        t1 = self.clock()
        done, t_blot = [], 0.0
        for st in recipes[paint]["first" if first else "reload"]:
            kind = st[0]
            if kind in ("scrub", "touch"):
                self.paint, self.since_load, self.used = paint, 1e9, 1e9       # it carries this paint from here on
            if kind == "water":
                self.water(st[1] if len(st) > 1 else "clean", passes=int(st[2]) if len(st) > 2 else 2)
            elif kind == "scrub":
                self.scrub(st[1], passes=int(st[2]) if len(st) > 2 else 4)
            elif kind == "touch":
                self.touch(st[1])
            elif kind == "blot":
                tb = self.clock()
                self.blot(int(st[1]) if len(st) > 1 else 2)
                t_blot += float(self.clock() - tb)
                self.blots += 1
            else:
                raise SafetyError(f"paint {paint!r}: unknown recipe step {st!r}")
            done.append(kind)
        self.paint, self.since_load, self.used = paint, 0.0, 0.0
        self.loads += 1
        if washed:
            self.washes += 1
            self.spent["wash_s"] += float(t1 - t0)
        self.spent["blot_s"] += t_blot
        self.spent["load_s"] += float(self.clock() - t1) - t_blot
        return {"paint": paint, "first": first, "washed": washed, "steps": done, **({"content_was_unknown": True} if unknown else {})}

    # ------------------------------------------------------------------ strokes
    def stroke_xy(self, pts_xy, press: float = 1.0, pitch: float | None = None, v_mm_s: float | None = None,
                  planned_mm: float | None = None) -> dict:
        """One stroke through table points at press x press_mm below first touch. What it takes from the load
        is counted by planned_mm when given (a plan's stroke: the plan's own length, so that the reloads of a
        layer are what layer_budget says: on 2026-11-13 the MEASURED travel, a millimetre longer per band,
        made two bands that fill a load to the brim on paper take two loads), else by the travel."""
        self.depth_m(press)            # press and lowering together: refused beyond MAX_DEPTH_M, before anything moves
        r = self.br.stroke(pts_xy, self.zc, press_m(self.s, press), v_mm_s=v_mm_s, pitch=pitch)
        ln = r["length_m"] * 1000.0 if planned_mm is None else float(planned_mm)
        self.since_load += ln
        self.used += use_mm(self.brush, ln, press)
        self.spent["travel_s"] += float(r.get("travel_s", 0.0))
        self.spent["stroke_s"] += float(r.get("seconds", 0.0))
        return r

    def stroke_canvas(self, pts_uv, press: float = 1.0) -> dict:
        return self.stroke_xy([self.canvas_xy(u, v) for u, v in pts_uv], press, pitch=self.pitch_for_canvas(),
                              planned_mm=RD.length_mm(pts_uv))

    def ladder(self, x: float, y: float, zs_mm, length: float = 0.016, relative: bool = True, pitch: float | None = None,
               into: list | None = None) -> dict:
        """A ladder of short pull strokes towards the robot with what is on the brush: the first starts at
        (x, y), the next ones LADDER_GAP_M further along +y, each at its own height zs_mm[k]: relative to the
        contact height of the set-up (`fd paint test --ladder`), or as plan z itself (relative=False,
        `fd paint contact`: the contact height is what is being measured). No press: a rung runs AT its
        height. `into`: a list that gets every rung as it is finished (what was done before a stop)."""
        rungs = [] if into is None else into
        for k, dz in enumerate(zs_mm):
            yy = y + LADDER_GAP_M * k
            z = (self.zc(x, yy) if relative else 0.0) + float(dz) / 1000.0
            self.br.stroke([(x, yy), (x - length, yy)], z, 0.0, pitch=pitch)
            rungs.append({"y": round(yy, 4), "dz_mm": float(dz), "plan_z_mm": round(z * 1000.0, 2)})
        return {"at": [x, y], "rungs": rungs, "paint": self.paint}

    def look(self, tag: str, move: bool = True) -> str | None:
        """To the photo pose (move=False: the arm stays where it is), one photo with the paper exposure.
        Returns its path (None without a camera or without a folder; the picture itself is self.last_photo)."""
        if move:
            self.br.goto_q(self.s["photo_q_deg"])
        if self.photo_fn is None:
            return None
        import cv2
        try:
            img = self.photo_fn()
        except MOTION_ERRORS:
            raise
        except Exception as e:      # the camera is busy, unplugged, locked: a refusal with a text, not a traceback
            raise Refusal("camera", f"the camera gave no photo ({type(e).__name__}: {e})",
                          "`fd snap --exposure paper` tries the camera alone") from e
        self.last_photo = img
        if self.out_dir is None:
            return None
        path = os.path.join(self.out_dir, f"{time.strftime('%H%M%S')}_{tag}.jpg")
        try:
            os.makedirs(self.out_dir, exist_ok=True)
            why = None if cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95]) else "OpenCV could not write the file"
            if why is None and not os.path.getsize(path):
                why = "the file is empty"
        except (OSError, cv2.error) as e:
            why = f"{type(e).__name__}: {e}"
        if why:                     # (a full disk, a folder that cannot be made): said, not passed off as a photo
            raise Refusal("camera", f"the photo was taken but could not be stored as {path} ({why})",
                          "is the disk full (`df -h`)? $FD_PAINT_OUT names another folder for the photos")
        self.photos.append(path)
        return path

    # ------------------------------------------------------------------ a layer of a plan
    def check_plan(self, plan: dict) -> None:
        """Refusal (stage "plan") unless the plan can be painted with this set-up (plan_problems)."""
        problems = plan_problems(plan, self.s)
        if problems:
            more = f" (and {len(problems) - 1} more)" if len(problems) > 1 else ""
            raise Refusal("plan", problems[0] + more,
                          "make the plan anew for this set-up (`fd paint status` shows canvas, pan axis and brush); "
                          "fd paint preview with the plan's file lists every problem", problems=problems[:20])

    def check_depths(self, layer: dict, start: int = 0, stop: int | None = None) -> None:
        """Refusal (stage "options") when a stroke of the layer, with the lowering of this command, would run
        deeper below first touch than MAX_DEPTH_M (depth_m): said before the first stroke, not at the one that
        asks for it."""
        strokes = layer["strokes"]
        stop = len(strokes) if stop is None else min(int(stop), len(strokes))
        for i in range(int(start), stop):
            try:
                self.depth_m(float(strokes[i].get("press", 1.0)))
            except SafetyError as e:
                raise Refusal("options", f"{layer['name']}[{i}]: {e}", "the same command with a smaller --lower-mm "
                              "(`fd paint run --help`)") from None

    def check_strokes(self, layer: dict, start: int = 0, stop: int | None = None) -> list:
        """Strokes of a layer the arm cannot make at the canvas pitch: [{"stroke", "why"}] (kinematics only:
        every point at its contact height minus the press, and its lift)."""
        pitch = self.pitch_for_canvas()
        roll = math.radians(self.br.roll)
        lim = self.arm.q_limits()
        bad = []
        strokes = layer["strokes"]
        stop = len(strokes) if stop is None else min(int(stop), len(strokes))
        for i in range(int(start), stop):
            st = strokes[i]
            p_m = press_m(self.s, float(st.get("press", 1.0)))
            for u, v in st["pts"]:
                x, y = self.canvas_xy(u, v)
                z = self.zc(x, y)
                why = None
                for zz in (z - p_m, z + self.br.o.hover_m):
                    if zz < self.br.z_min - 1e-9:
                        why = f"({u}, {v}) mm would run at {zz * 1000:.1f} mm, below the limit of brush work"
                        break
                    info = K.ik(np.array([x, y, zz]), pitch, roll, tool=self.br.tl, limits=lim)[1]
                    if not info["ok"]:
                        why = f"({u}, {v}) mm is out of reach at pitch {pitch:.0f}: {info['reason']}"
                        break
                if why:
                    bad.append({"stroke": i, "why": why})
                    break
        return bad

    def run_layer(self, plan: dict, li: int, start: int = 0, stop: int | None = None, max_s: float | None = None,
                  reload_mm: float | None = None, on_stroke=None) -> dict:
        """Paint strokes start .. stop-1 of layer li. The brush is reloaded BETWEEN strokes, never inside one:
        when it carries another paint, and when the next stroke would take more than is left of the load (by the
        brush of the set-up: reload_rule; reload_mm: after that much plain travel instead). Ends early, cleanly
        (the stroke in hand is finished), when max_s seconds of the arm's clock are over.
        Refused before anything moves (Refusal): a plan that does not fit the set-up, a contact plane that was
        measured for another brush state or has no record of one. on_stroke(i): called after every finished stroke with the index of the
        next one, so that the caller knows what is on the paper whatever ends this call (also an error that is
        not a stopped motion, such as the servo bus failing: that one is raised).
        -> {"ok", "layer", "paint", "next_stroke": index of the first stroke not painted, "of", "done",
            "strokes", "travel_mm", "loads", "blots", "washes", "seconds", "reload", "reason"?, "interrupted"?}"""
        self.check_plan(plan)
        if not (isinstance(li, int) and 0 <= li < len(plan["layers"])):
            raise Refusal("plan", f"the plan has no layer {li!r} (it has {len(plan['layers'])})", "the layers of a plan are listed by fd paint preview with the plan's file")
        layer = plan["layers"][li]
        use_of, limit, rule = reload_rule(self.brush, reload_mm, self.s["paints"].get(layer["paint"]))
        self.require_plane("painting")
        require_wells(self.s, layer["paint"])
        strokes = layer["strokes"]
        stop = len(strokes) if stop is None else min(int(stop), len(strokes))
        self.check_depths(layer, start, stop)
        t_start = self.clock()
        l0, b0, w0 = self.loads, self.blots, self.washes
        out = {"ok": True, "layer": layer["name"], "paint": layer["paint"], "next_stroke": int(start), "of": len(strokes),
               "done": False, "strokes": 0, "travel_mm": 0.0, "loads": 0, "blots": 0, "washes": 0,
               "reload": {"by": "paint used" if rule == "paint" else "travel", "limit_mm": round(limit, 1)}}
        i = int(start)
        try:
            pitch = self.pitch_for_canvas()
            out["pitch"] = round(pitch, 1)
            while i < stop:
                if max_s is not None and self.clock() - t_start > max_s:
                    out.update(reason=f"time limit of {max_s:.0f} s reached", time_limit=True)
                    break
                st = strokes[i]
                press = float(st.get("press", 1.0))
                use = use_of(RD.length_mm(st["pts"]), press)
                if self.paint != layer["paint"] or (self.used if rule == "paint" else self.since_load) + use > limit:
                    self.load(layer["paint"])
                r = self.stroke_canvas(st["pts"], press)
                out["strokes"] += 1
                out["travel_mm"] += r["length_m"] * 1000.0
                i += 1
                out["next_stroke"] = i
                if on_stroke is not None:
                    on_stroke(i)
                self.log(f"layer {layer['name']}: stroke {i}/{len(strokes)}  {r['length_m'] * 1000:.0f} mm  "
                         f"t={self.clock() - t_start:.0f}s")
        except BW.JawRefusal:            # (the jaw is not where its record has it: the caller's refusal, with its way out)
            raise
        except MOTION_ERRORS as e:
            out.update(ok=False, reason=f"{type(e).__name__}: {e}")
        except KeyboardInterrupt:
            out.update(ok=False, reason="interrupted", interrupted=True)
        out["loads"], out["blots"], out["washes"] = self.loads - l0, self.blots - b0, self.washes - w0
        out["done"] = bool(out["ok"] and i >= len(strokes))
        out["travel_mm"] = round(out["travel_mm"])
        out["seconds"] = round(float(self.clock() - t_start), 1)
        return out

    # ------------------------------------------------------------------ where the brush goes (for `fd paint check`)
    def visits(self) -> list:
        """Every place the skill sends the brush to, with the tool tip points it visits there (absolute) and the
        height it travels to it at: wells (scrub and touch), the dish(es) (dip, rinse), the rows of the blotting
        strip, the corners and the centre of the canvas. "needed_by": the paints (or the washing) that need it."""
        o, tz = self.br.o, self.tz
        wells, dishes, blotters = {}, {d: [] for d in DISHES}, []
        for name, rec in self.s["paints"].items():
            for st in list(rec.get("first", [])) + list(rec.get("reload", [])):
                if st[0] in ("scrub", "touch") and name not in wells.setdefault(st[1], []):
                    wells[st[1]].append(name)
                elif st[0] == "water" and name not in dishes[st[1] if len(st) > 1 else "clean"]:
                    dishes[st[1] if len(st) > 1 else "clean"].append(name)
                elif st[0] == "blot" and name not in blotters:
                    blotters.append(name)
        out = []
        for name, w in self.s["wells"].items():
            x, y = (float(v) for v in w["xy"])
            z = tz + float(w.get("z_mm", 2.5)) / 1000.0
            low = [(x, y, z)] + [(x + dx, y + dy, z + dz) for dx, dy, dz in self._scrub_pattern(w["xy"], 4)]
            out.append({"name": name, "kind": "well", "xy": (x, y), "low": low, "top_h": o.travel_h,
                        "needed_by": wells.get(name, []), "needed": bool(wells.get(name))})
        water = self.s["water"]
        single = not water.get("clean")
        for which in DISHES:
            d = water.get(which)
            if not d:
                continue
            x, y = (float(v) for v in d["xy"])
            zd = tz + float(d.get("z_mm", 6.0)) / 1000.0
            low = [(x, y, zd)] + [(x + dx, y + dy, zd + dz) for dx, dy, dz in self._water_pattern(2)]
            pats = []
            if which == "rinse":
                pats.append((tz + float(d.get("scrub_z_mm", 3.0)) / 1000.0, self._rinse_pattern(2)))
            if which == "clean" or single:
                pats.append((tz + float(d.get("scrub_z_mm", 4.5)) / 1000.0, self._clean_dish_pattern(d["xy"])))
            for zz, pat in pats:
                low += [(x, y, zz)] + [(x + dx, y + dy, zz + dz) for dx, dy, dz in pat]
            users = list(dishes[which]) + (list(dishes["clean"]) if which == "rinse" and single else [])
            out.append({"name": which, "kind": "dish", "xy": (x, y), "low": low, "top_h": o.travel_h,
                        "needed_by": sorted(set(users)) + ["washing the brush"], "needed": True})
        if self.s.get("blot"):
            b = self.s["blot"]
            n = max(1, int(float(b.get("size", [0.03, 0.05])[1]) / 0.011))
            z, rows = self._blot_rows(n, 0)
            low = []
            for a, e in rows:
                low += [(a[0], a[1], z - 0.003), (e[0], e[1], z - 0.003), (a[0], a[1], z + o.hover_m)]
            out.append({"name": "blot", "kind": "blot", "xy": (float(b["xy"][0]), float(b["xy"][1])), "low": low,
                        "top_h": o.above_h, "needed_by": blotters + ["washing the brush"], "needed": True})
        for name, x, y in self.canvas_points()[:5]:
            out.append({"name": name, "kind": "canvas", "xy": (x, y), "low": [(x, y, self.zc(x, y) - BW.MAX_PRESS_M)],
                        "top_h": o.above_h, "needed_by": ["every stroke"], "needed": True})
        return out


# ============================================================================= the contact plane and its record
def brush_state(tool, pitch_deg: float) -> dict:
    """What a contact plane is measured with, kept with it: the tool (name, attach time), its tip offset and
    gripper position, the canvas pitch (and, where tooling.Tool has them, on which jaw the tool is taped, the
    squeeze that holds a tool on the moving jaw, and whether the tip was measured)."""
    d = {"tool": tool.name, "attached_at": getattr(tool, "attached_at", None),
         "tip_offset": [round(float(v), 5) for v in tool.tip_offset],
         "gripper_raw": None if tool.gripper_raw is None else int(tool.gripper_raw),
         "pitch_deg": round(float(pitch_deg), 2)}
    for key in ("gripper_goal_raw", "on_jaw"):
        v = getattr(tool, key, None)             # fields of newer tool files
        if v is not None:
            d[key] = v
    if hasattr(tool, "tip_measured"):
        d["tip_measured"] = bool(tool.tip_measured)
    return d


def brush_differs(rec: dict, tool, pitch_deg: float | None = None) -> list:
    """In what the brush as it is differs from a recorded brush state (brush_state), in words; [] = the same
    brush state. Compared: which tool it is (name and attach time: a brush that was taped on again is another
    mount, whatever numbers were typed for it), the tip offset (0.5 mm), the gripper position (6 ticks), on
    which jaw the tool is taped, whether the jaw is HELD by a squeeze and how far (6 ticks), and, when given,
    the canvas pitch (1 deg). The hold counts: first touch was 4 mm higher with the brush loose than with the
    jaw squeezed shut (2026-11-13), and a `fd tool jaw close` that puts the squeeze on keeps position and tip."""
    why = []
    if rec.get("tool") != getattr(tool, "name", None) or rec.get("attached_at") != getattr(tool, "attached_at", None):
        why.append(f"the registered tool is {getattr(tool, 'name', None)!r} (attached {getattr(tool, 'attached_at', None)}) "
                   f"now, it was {rec.get('tool')!r} (attached {rec.get('attached_at')}) then")
    tip = rec.get("tip_offset")
    try:
        d_tip = max(abs(float(a) - float(b)) for a, b in zip(tip, tool.tip_offset)) if len(tip) == 3 else float("inf")
    except (TypeError, ValueError):
        d_tip = float("inf")
    if d_tip > CONTACT_TIP_TOL_M:
        why.append(f"the tool tip is {[round(float(v), 4) for v in tool.tip_offset]} now, it was {tip} then")
    def apart(a, b) -> bool:
        """Do two servo positions differ by more than the tolerance (a value that is no number differs)?"""
        try:
            return abs(int(a) - int(b)) > CONTACT_GRIPPER_TOL
        except (TypeError, ValueError):
            return True

    g_rec, g_now = rec.get("gripper_raw"), tool.gripper_raw
    if (g_rec is None) != (g_now is None) or (g_rec is not None and apart(g_rec, g_now)):
        why.append(f"the gripper stands at raw {g_now} now, it stood at {g_rec} then")
    h_rec, h_now = rec.get("gripper_goal_raw"), getattr(tool, "gripper_goal_raw", None)
    if (h_rec is None) != (h_now is None):
        why.append("the jaw is held shut by a squeeze now, it was not held when the plane was measured" if h_rec is None else
                   "the jaw is not held by a squeeze now, it was held when the plane was measured (a loose brush touched "
                   "4 mm higher on 2026-11-13)")
    elif h_rec is not None and apart(h_rec, h_now):
        why.append(f"the jaw is squeezed to goal {h_now} now, it was squeezed to {h_rec} then")
    j_rec, j_now = rec.get("on_jaw"), getattr(tool, "on_jaw", None)
    if j_rec != j_now:
        why.append(f"the tool is recorded on_jaw = {j_now} now, it was {j_rec} then")
    if pitch_deg is not None:
        p = rec.get("pitch_deg")
        try:
            d_pitch = abs(float(p) - float(pitch_deg))
        except (TypeError, ValueError):
            d_pitch = float("inf")
        if d_pitch > CONTACT_PITCH_TOL_DEG:
            why.append(f"the canvas is painted at pitch {float(pitch_deg):.1f} now, the plane was measured at {p}")
    return why


def contact_record_state(setup: dict, tool, pitch_deg: float | None) -> dict:
    """Does the contact plane hold for the brush as it is? {"recorded": the plane says what it was measured
    with; "ok": False when the brush differs from that record (brush_differs: the tool, its tip, gripper
    position, jaw and squeeze, the canvas pitch; each difference in "why"), and False for EVERY plane without a
    record, until `fd paint contact --adopt` or a measurement gives it one; "reason" / "next": the text and the
    next command of the refusal when not ok ("adopted": True when the record that differs is an adopted one:
    the next is then the adoption beside the measurement, ADOPT_NEXT)}. pitch_deg None: the pitch is not
    compared (no single pitch reaches the canvas).

    A plane without a record used to be painted with while the tool file was the older of the two files (the
    set-up's date, or "plane_at" once `fd paint brush` or `locate` had rewritten it). That rule erred towards
    painting: the tool measured anew and THEN the set-up edited by hand (what the hand-over of 2026-11-13 asked
    for: "load_mm" in two recipes), and the stroke was painted with the plane of the old brush (second review of
    that update). No date says with which brush a plane was measured; only a record does."""
    rec = (setup.get("contact") or {}).get("measured")
    if not isinstance(rec, dict):
        why = "the contact plane of the set-up has no record of the brush state it was measured with"
        return {"recorded": False, "ok": False, "why": [why], "next": ADOPT_NEXT,
                "reason": why + " (a set-up written by hand, or from before such records were kept): nothing says that "
                                "it is the plane of the brush that is on now"}
    why = brush_differs(rec, tool, pitch_deg)
    out = {"recorded": True, "ok": not why, "why": why, "measured": rec}
    if why and rec.get("adopted") is True:
        # The record that differs is the operator's WORD, not ladders (adopt_block), and `fd paint contact --adopt`
        # takes that word again for the brush as it is now. Status and check named only a new measurement, and
        # called the plane "measured for" a brush state nothing was measured with: an adoption made before the tip
        # was stored and the jaw held sent the operator to fly ladders on the real arm for a plane that only
        # needed the word again (third review, T10).
        # (brush_differs speaks of when the plane "was measured": for this record that was when it was adopted)
        why = [w.replace("when the plane was measured", "when the plane was adopted")
                .replace("the plane was measured at", "the plane was adopted at") for w in why]
        out.update(why=why, reason="the contact plane of the set-up was adopted for another brush state (the operator's "
                                   "word, no ladders were painted for that record): " + "; ".join(why),
                   next=ADOPT_NEXT, adopted=True)
    elif why:
        out.update(reason="the contact plane of the set-up was measured for another brush state: " + "; ".join(why),
                   next=CONTACT_NEXT)
    return out


def check_zs(zs_mm, table_z: float) -> list:
    """The heights of a contact ladder (plan z, mm): 2 to 6 of them, falling, none above 30 mm, none below
    the limit of brush work. Refusal otherwise. (How deep a rung may go below the paper the set-up already
    knows is check_rungs.)"""
    try:
        zs = [float(v) for v in zs_mm]
    except (TypeError, ValueError):
        zs = []
    low = (table_z - BW.MAX_BELOW_TABLE_M) * 1000.0
    if not (2 <= len(zs) <= 6 and all(math.isfinite(z) for z in zs) and all(a > b for a, b in zip(zs, zs[1:]))
            and zs[0] <= 30.0 and zs[-1] >= low):
        raise Refusal("options", f"--zs {zs_mm!r}: 2 to 6 plan heights in mm, falling, at most 30 and not below "
                                 f"{low:.1f} (e.g. \"12,9,6,3\")", "`fd paint contact --help`")
    return zs


def check_rungs(setup: dict, places: list) -> None:
    """Refusal (stage "options") when a rung of a contact ladder would run more than the press limit of brush
    work (6 mm) below the first-touch height the set-up ALREADY has at its place. The depth limit of the
    ladders used to be counted from table_z alone, although the evening of 2026-11-13 showed the paper 5 to
    10 mm above it: `--zs 9,5,1,-3,-7` was accepted and put the lowest rung 15 mm below the stored plane at the
    far-left corner, as deep as the bristles of the thin brush are long (a brush squashed like that shifts in
    its tape, and then the tip is where nobody measured it). `fd paint contact --distrust-plane` skips this
    check for a stored plane that is known to be wrong (a new table, a thick pad under the sheet); the limit
    of brush work below the table holds in any case."""
    worst = None
    for p in places:
        for r in p["rungs"]:
            for x in (r["x0"], r["x1"]):
                plane = contact_z(setup, x, r["y"]) * 1000.0
                depth = plane - float(r["plan_z_mm"])
                if depth > BW.MAX_PRESS_M * 1000.0 + 1e-6 and (worst is None or depth > worst[0]):
                    worst = (depth, p["name"], float(r["plan_z_mm"]), plane)
    if worst is not None:
        depth, name, z, plane = worst
        raise Refusal("options", f"--zs: the rung at {z:g} mm of the ladder at {name} would run {depth:.1f} mm below the "
                                 f"paper height the set-up already has there (first touch at {plane:.1f} mm): more than the "
                                 f"{BW.MAX_PRESS_M * 1000:.0f} mm a stroke may press (the bristles are about 15 mm long: "
                                 "squashed, the brush shifts in its tape)",
                      f"rungs not lower than {plane - BW.MAX_PRESS_M * 1000.0:.1f} mm at that place (--zs), e.g. the default "
                      "\"12,9,6,3\" raised; only when the stored plane is known to be wrong (a new table, a pad under the "
                      "sheet): the same command with --distrust-plane")


def contact_places(setup: dict, at=None, zs_mm=(12.0, 9.0, 6.0, 3.0), pull: float = 0.010) -> list:
    """The ladders of `fd paint contact`. at: None or ["canvas"] = five ladders CENTRED on the four corners of
    the canvas, 15 mm inside it, and on its centre (what was measured on 2026-11-13); (x, y) = a ladder whose
    first rung STARTS there. Every rung is a pull of `pull` metres towards the robot, the next one
    LADDER_GAP_M further along +y, at plan heights zs_mm (falling).
    -> [{"name", "at": [x, y], "rungs": [{"y", "x0", "x1", "plan_z_mm"}], "inside_canvas"}]"""
    n = len(zs_mm)
    half_y = LADDER_GAP_M * (n - 1) / 2.0
    starts = []
    for item in (at or ["canvas"]):
        if isinstance(item, str):
            if item != "canvas":
                raise Refusal("options", f"--at {item!r}: \"canvas\" or x,y in metres", "`fd paint contact --help`")
            x0, x1, y0, y1 = canvas_rect(setup)
            d = CANVAS_INSET_M
            for name, cx, cy in (("near-left", x0 + d, y0 + d), ("near-right", x0 + d, y1 - d), ("far-left", x1 - d, y0 + d),
                                 ("far-right", x1 - d, y1 - d), ("centre", (x0 + x1) / 2.0, (y0 + y1) / 2.0)):
                starts.append((name, cx + pull / 2.0, cy - half_y))
        else:
            starts.append((f"({float(item[0]):.3f}, {float(item[1]):.3f})", float(item[0]), float(item[1])))
    places = []
    for name, x, y in starts:
        rungs = [{"y": round(y + LADDER_GAP_M * k, 4), "x0": round(x, 4), "x1": round(x - pull, 4), "plan_z_mm": float(z)}
                 for k, z in enumerate(zs_mm)]
        inside = all(canvas_distance(setup, xx, r["y"]) <= 0.004 for r in rungs for xx in (r["x0"], r["x1"]))
        places.append({"name": name, "at": [round(x, 4), round(y, 4)], "rungs": rungs, "inside_canvas": bool(inside)})
    return places


# ----------------------------------------------------------------------------- reading the ladders from two photos
# The rule, derived from the two photos of 2026-11-13 17:39 / 17:42 (runs/20261113_paint/173916_look.jpg and
# 174247_test.jpg: red paint, five ladders of four rungs at plan z 12, 9, 6, 3 mm, each rung a 10 mm pull towards
# the robot) seen from straight above (the photos warped onto the table plane with the camera model):
# - a new mark is where the paper got darker between the two photos (red paint takes the green and blue away:
#   120 to 150 grey levels where it is strong, the faintest dab still 26; the noise of the difference is under 1);
# - a mark does not lie on its rung: it lies BEYOND the end of the pull, towards the robot (its far end 1 to
#   14 mm beyond it, its near end up to 18 mm), and up to 4.5 mm beside the rung (-4.3 .. +1.2 mm that evening),
#   by the same amount for all rungs of one ladder. (Where the camera model puts a commanded point and where the
#   arm paints differ by that much.) So the marks of a ladder are looked for in a window from 22 mm beyond the
#   rung ends to 6 mm before the rung starts, the sideways shift of the ladder (at most 7 mm) is the one that
#   puts most paint on rungs, and a mark belongs to the rung it then lies within 3.5 mm of. A shift of about half
#   the rung spacing (6 mm) looks like its opposite: it is taken when only one of the two readings is a ladder
#   that touched (marks on every rung below the first marked one), else the ladder is not read;
# - first touch is at or above the HIGHEST rung that left a mark, and below the dry rung over it. How far above
#   says the mark: a dot or dabs covering a part c of the pull's length mean "just touching": + 0.75 mm * c
#   (the eye read +0.0 for a dot of 1.4 mm, +0.3 for faint dabs of 3.5 to 4 mm, +0.5 for two dabs of 6 mm); a line
#   over the whole pull means the tip was pressed: + 0.8 mm + its width (the one such line of that evening,
#   1.2 mm wide, was read by eye as 2 mm above its rung; at the working press of 3 mm a line along the robot's
#   direction was 2.5 to 3 mm wide to the eye), never more than up to the dry rung above; between three
#   quarters and the whole of the pull the two are blended;
# - every rung below the first marked one must have left a mark too, else the ladder is not read (a drip, a dry
#   brush); a full mark on the highest rung means first touch is above the ladder: not read either.
# With this rule the five places of that evening read 3.3, 3.3, 9.1, 6.5 and 5.0 mm; the eye had read 3.3, 3.3,
# 9.0, 6.5 and 5.0. A ladder whose first marked rung carries dabs is read to a few tenths of a millimetre, one
# that starts with a full line to about one millimetre (the width is the weak part: two observations stand
# behind it, and the photo's blur widens a line by up to half a millimetre).
MARK_BEYOND_M = 0.022
MARK_BEFORE_M = 0.006
MARK_SIDE_M = 0.007
MARK_RUNG_TOL_M = 0.0035
MARK_MIN_MM2 = 0.6
MARK_MIN_DARK = 8.0
MARK_FAINT_DARK = 40.0
PAPER_MIN_GREY = 90.0
TOUCH_PER_COVER_MM = 0.75
LINE_BASE_MM = 0.8
# Bare paper, for the look at the sheet BEFORE the ladders are painted (bare_paper). In the photos of 2026-11-13
# (paper exposure) blank paper has a saturation of 8 to 20 and is even to a few grey levels inside a ladder
# window; paint is saturated (a pale wash 40 to 60, a full colour 100 to 230) or dark (ink). Counted as "not
# paper": a pixel more than 20 saturation steps above the window's paper (its median, at most 25) or darker than
# 0.85 of the window's bright level. (IN-SAMPLE: these numbers were set on the photos they are quoted for below.
# Under another light or exposure a pale wash may pass for paper, or a blank sheet be refused: the first real
# `fd paint contact --scrap-laid` shows it, in "sheet" of its result.)
# With that the windows of the five standard ladders read 0.0 %, all five, on
# the 30 photos of that day in which the sheet there is blank or freshly covered (among them the clean scrap sheet
# of 17:39), and more than the limit in at least one window on each of the 21 photos with paint there: 34, 25, 5,
# 4 and 91 % on the finished Kaminarimon (19:08), 17 % at the centre after its first band (18:15), 3.3 and 1.9 %
# at the far ladders of the scrap sheet that had one set of ladders on it (17:42). More than half a per cent is
# not bare paper (one old mark of 10 x 2.5 mm is 1 % of a window). A hard shadow or a thing lying on the sheet
# reads as "not bare" too: that errs on the side of not painting.
PAPER_MAX_SAT = 25.0
PAINT_SAT_STEP = 20.0
PAPER_REL_GREY = 0.85
BARE_MAX_PART = 0.005


def table_view(img, cam, x0: float, x1: float, y0: float, y1: float, z: float, ppm: float = 4.0):
    """A photo seen from straight above: the table rectangle x0..x1, y0..y1 at height z, warped with the camera
    model. Rows run along +x (away from the robot, as down the canvas), columns along +y; ppm pixels per mm."""
    import cv2
    xs = x0 + (np.arange(max(1, int(round((x1 - x0) * 1000.0 * ppm))), dtype=np.float64) + 0.5) / (1000.0 * ppm)
    ys = y0 + (np.arange(max(1, int(round((y1 - y0) * 1000.0 * ppm))), dtype=np.float64) + 0.5) / (1000.0 * ppm)
    yy, xx = np.meshgrid(ys, xs)
    pts = np.column_stack([xx.ravel(), yy.ravel(), np.full(xx.size, float(z))])
    uv = cam.project(pts).reshape(xx.shape + (2,)).astype(np.float32)
    return cv2.remap(img, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))


def _new_marks(before, after, cam, win: tuple, table_z: float, ppm: float) -> dict:
    """What got darker inside a table window between two photos: {"blobs": [...], "paper": grey level of the
    window before, "noise", "threshold", "why": text when the window cannot show a mark}."""
    import cv2
    x0, x1, y0, y1 = win
    b = table_view(before, cam, x0, x1, y0, y1, table_z, ppm).astype(np.float32)
    a = table_view(after, cam, x0, x1, y0, y1, table_z, ppm).astype(np.float32)
    if b.ndim == 2:
        b, a = b[..., None], a[..., None]
    grey = b.mean(axis=2)
    paper = float(np.median(grey))
    out = {"blobs": [], "paper": round(paper, 1), "noise": None, "threshold": None, "why": None}
    if paper < PAPER_MIN_GREY:
        out["why"] = (f"the first photo shows no bright paper here (grey level {paper:.0f}): a mark would not show; "
                      "lay a clean scrap sheet over the place")
        return out
    d = (b - a).max(axis=2)                         # darkening, in the channel that lost most
    med = float(np.median(d))
    sig = max(1.0, 1.4826 * float(np.median(np.abs(d - med))))
    # slow changes of the light between the two photos (a person's shadow) are not marks
    bg = cv2.GaussianBlur(np.clip(d, med - 3.0 * sig, med + 3.0 * sig), (0, 0), 5.0 * ppm)
    hp = d - bg
    sig = 1.4826 * float(np.median(np.abs(hp - float(np.median(hp)))))
    thr = max(MARK_MIN_DARK, 4.5 * sig)
    out.update(noise=round(sig, 2), threshold=round(thr, 1))
    if sig > 6.0:
        out["why"] = (f"the two photos differ all over this place (noise {sig:.1f} grey levels): something moved or "
                      "the light changed; take both photos again")
        return out
    mask = cv2.morphologyEx((hp > thr).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, _cent = cv2.connectedComponentsWithStats(mask, 8)
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA]) / ppm ** 2
        if area < MARK_MIN_MM2:
            continue
        rows, cols = np.nonzero(lab == i)
        dark = float(hp[rows, cols].max())
        # Length and width are taken where the mark is at least a quarter as dark as its dark parts (a faint
        # mark: wherever it is over the threshold). A strong mark has a blurred fringe that is over the
        # threshold too: it would add 1 to 2 mm where the camera sees the sheet from far, and not where it is
        # near. Width: across the pull, the median over the rows along it.
        body = hp[rows, cols] > max(thr, 0.25 * float(np.percentile(hp[rows, cols], 90)))
        if body.any():
            rows, cols = rows[body], cols[body]
        across = np.bincount(rows - rows.min())
        across = across[across > 0]
        out["blobs"].append({
            "x": float(x0 + (rows.mean() + 0.5) / (1000.0 * ppm)), "y": float(y0 + (cols.mean() + 0.5) / (1000.0 * ppm)),
            "x_lo": float(x0 + rows.min() / (1000.0 * ppm)), "x_hi": float(x0 + (rows.max() + 1) / (1000.0 * ppm)),
            "y_lo": float(y0 + cols.min() / (1000.0 * ppm)), "y_hi": float(y0 + (cols.max() + 1) / (1000.0 * ppm)),
            "area_mm2": area, "dark": dark,
            "length_mm": float(rows.max() - rows.min() + 1) / ppm, "width_mm": float(np.median(across)) / ppm})
    return out


def ladder_window(place: dict) -> tuple:
    """(x0, x1, y0, y1): the table window in which the marks of a ladder are looked for (and which must be bare
    paper before it is painted): from MARK_BEYOND_M beyond the rung ends to MARK_BEFORE_M before the rung
    starts, and MARK_SIDE_M + 4 mm to both sides of the outer rungs."""
    ys = [float(r["y"]) for r in place["rungs"]]
    x_start, x_end = max(r["x0"] for r in place["rungs"]), min(r["x1"] for r in place["rungs"])
    return (x_end - MARK_BEYOND_M, x_start + MARK_BEFORE_M, min(ys) - MARK_SIDE_M - 0.004, max(ys) + MARK_SIDE_M + 0.004)


def stroke_window(pts_xy) -> tuple:
    """(x0, x1, y0, y1): the table window in which the paint of a stroke through these points lands, as
    ladder_window gives it for a ladder (the marks lie beyond the commanded points towards the robot, and
    beside them): what must be bare paper before a test stroke is painted on the canvas."""
    pts = np.asarray(pts_xy, dtype=float).reshape(-1, 2)
    return (float(pts[:, 0].min()) - MARK_BEYOND_M, float(pts[:, 0].max()) + MARK_BEFORE_M,
            float(pts[:, 1].min()) - MARK_SIDE_M - 0.004, float(pts[:, 1].max()) + MARK_SIDE_M + 0.004)


def bare_paper(img, cam, win: tuple, table_z: float, ppm: float = 4.0) -> dict:
    """Is a table window of a photo bare paper? {"bare": bool, "paper": median grey level, "not_paper_pct":
    the part of the window that is neither paper-coloured nor paper-bright, "why": text when it is not bare}.
    The numbers and what they were measured on are at the top of this section."""
    import cv2
    view = table_view(img, cam, *win, table_z, ppm)
    if view.ndim == 2:
        view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)
    grey = view.astype(np.float32).mean(axis=2)
    sat = cv2.cvtColor(view, cv2.COLOR_BGR2HSV)[..., 1].astype(np.float32)
    paper, bright = float(np.median(grey)), float(np.percentile(grey, 90))
    out = {"bare": False, "paper": round(paper, 1), "not_paper_pct": None, "why": None}
    if paper < PAPER_MIN_GREY:
        out.update(not_paper_pct=100.0, why=f"no bright paper there (grey level {paper:.0f})")
        return out
    s_paper = min(float(np.median(sat)), PAPER_MAX_SAT)
    part = float(((sat > s_paper + PAINT_SAT_STEP) | (grey < PAPER_REL_GREY * bright)).mean())
    out["not_paper_pct"] = round(100.0 * part, 1)
    if part > BARE_MAX_PART:
        out["why"] = f"{100.0 * part:.1f} % of it is painted or covered (paint, old marks, something lying on it, a hard shadow)"
    else:
        out["bare"] = True
    return out


def sheet_check(img, cam, places: list, table_z: float) -> dict:
    """The look at the sheet before contact ladders are painted: is every ladder window bare paper?
    -> {"ok", "places": [{"name", "bare", "paper", "not_paper_pct", "why"}], "why": text when not ok}.
    `fd paint contact` paints at the corners and the centre of the canvas, where the picture is: run as the
    `next` of four commands gave it, with the finished Kaminarimon uncovered, it would have put 20 rungs on the
    picture (review of 2026-11-13). A scrap sheet laid over it is bare paper in every window; the picture is
    not. What the photo cannot tell is a blank PAINTING sheet from scrap: that is the operator's word
    (--scrap-laid)."""
    rows = []
    for p in places:
        b = bare_paper(img, cam, ladder_window(p), table_z)
        rows.append({"name": p["name"], **b})
    bad = [r for r in rows if not r["bare"]]
    why = None
    if bad:
        why = ("the sheet is not bare paper where the ladders would be painted: "
               + "; ".join(f"{r['name']}: {r['why']}" for r in bad))
    return {"ok": not bad, "places": rows, "why": why}


def first_touch(rungs: list, pull_mm: float) -> tuple:
    """(plan z of first touch in mm or None, the reason in words) from the rungs of one ladder as read
    (highest first; each {"plan_z_mm", "mark": bool, "length_mm", "width_mm"}). The rule is at the top of
    this section."""
    marked = [bool(r.get("mark")) for r in rungs]
    zs = [float(r["plan_z_mm"]) for r in rungs]
    if not any(marked):
        return None, (f"no rung left a mark: the paper is lower than {zs[-1]:.1f} mm here, or the brush was dry "
                      "(lower --zs, or load the brush)")
    k = marked.index(True)
    if not all(marked[k:]):
        dry = zs[k + 1 + marked[k + 1:].index(False)]
        return None, (f"the rung at {zs[k]:.1f} mm left a mark but the lower one at {dry:.1f} mm did not (a drip, or "
                      "the brush ran dry): this ladder cannot be read")
    r = rungs[k]
    cover = min(1.0, float(r["length_mm"]) / float(pull_mm))
    s = min(1.0, max(0.0, (cover - 0.75) / 0.25))           # 0: dabs (just touching) .. 1: a line over the whole pull
    add = (1.0 - s) * TOUCH_PER_COVER_MM * cover + s * (LINE_BASE_MM + float(r["width_mm"]))
    if k == 0:
        if s > 0.0:
            return None, (f"the highest rung ({zs[0]:.1f} mm) already left a full mark: first touch is above the "
                          "ladder (start it higher: --zs)")
        return round(zs[0] + add, 2), f"dabs over {cover * 100:.0f} % of the pull at the highest rung ({zs[0]:.1f} mm): just touching"
    add = min(add, zs[k - 1] - zs[k] - 0.1)                 # the rung above stayed dry
    if s <= 0.0:
        what = f"a dot or dabs over {cover * 100:.0f} % of the pull at {zs[k]:.1f} mm: just touching"
    else:
        what = (f"a line {float(r['width_mm']):.1f} mm wide over {cover * 100:.0f} % of the pull at {zs[k]:.1f} mm, nothing at "
                f"{zs[k - 1]:.1f} mm: pressed by about {add:.1f} mm")
    return round(zs[k] + add, 2), what


def read_place(before, after, cam, place: dict, table_z: float, ppm: float = 5.0) -> dict:
    """One ladder read from the two photos: the place with, for every rung, "mark" (bool) and when there is
    one "length_mm", "width_mm", "pieces", "dark" (grey levels), "faint"; "sideways_mm" (shift of the marks
    of this ladder), "stray" (new marks that belong to no rung), "read" (bool), "first_touch_mm", "xy" (where
    that height holds: the middle of the rung that touched first) and "why" (the reading in words, or why the
    ladder could not be read)."""
    rungs = [dict(r) for r in place["rungs"]]
    ys = np.array([r["y"] for r in rungs])
    x_start, x_end = max(r["x0"] for r in rungs), min(r["x1"] for r in rungs)
    pull_mm = (x_start - x_end) * 1000.0
    win = ladder_window(place)
    out = {**place, "rungs": rungs, "read": False, "first_touch_mm": None, "sideways_mm": None, "stray": 0}
    for r in rungs:
        r["mark"] = False
    found = _new_marks(before, after, cam, win, table_z, ppm)
    out["photo"] = {k: found[k] for k in ("paper", "noise", "threshold")}
    if found["why"]:
        out["why"] = found["why"]
        return out
    blobs = found["blobs"]

    def assign(s):
        """The marks by rung for a sideways shift s of the whole ladder: ([[blob, ...] per rung], strays)."""
        per, stray = [[] for _ in rungs], 0
        for b in blobs:
            d = np.abs(b["y"] - (ys + s))
            k = int(d.argmin())
            if d[k] > MARK_RUNG_TOL_M:
                stray += 1
            else:
                per[k].append(b)
        return per, stray

    def downwards(per):
        """Marks from some rung on, on every rung below it: what a ladder that touched the paper looks like."""
        m = [bool(p) for p in per]
        return any(m) and all(m[m.index(True):])

    shift = 0.0
    if blobs:
        by = np.array([b["y"] for b in blobs])
        wt = np.array([b["area_mm2"] for b in blobs])

        def score(s):
            d = np.abs(by[:, None] - (ys[None, :] + s)).min(axis=1)
            on = d <= MARK_RUNG_TOL_M
            return float(wt[on].sum()), -float((wt[on] * d[on] ** 2).sum())

        cands = sorted({round(float(b - y), 5) for b in by for y in ys if abs(b - y) <= MARK_SIDE_M} | {0.0})
        shift = max(cands, key=lambda s: (score(s), -abs(s)))
        d = by[:, None] - (ys[None, :] + shift)
        on = np.abs(d).min(axis=1) <= MARK_RUNG_TOL_M
        if on.any():
            k_of = np.abs(d).argmin(axis=1)
            shift += float(np.average(d[np.arange(len(by)), k_of][on], weights=wt[on]))      # the common offset, refined
        # The same marks read one rung over: a shift of about half the rung spacing and its opposite put the
        # same paint on rungs. Only one of the two may look like a ladder that touched (marks on every rung
        # below the first marked one); else the photos cannot tell.
        best = score(shift)[0]
        others = [o for o in (shift - LADDER_GAP_M, shift + LADDER_GAP_M)
                  if abs(o) <= MARK_SIDE_M and score(o)[0] >= 0.999 * best > 0.0]
        if others:
            good = [s for s in [shift, *others] if downwards(assign(s)[0])]
            if len(good) != 1:
                out["why"] = (f"the marks lie about half way between the rungs (sideways by {shift * 1000:+.1f} or "
                              f"{others[0] * 1000:+.1f} mm): which rung left which mark cannot be told from the photos")
                out["sideways_mm"] = round(shift * 1000.0, 1)
                return out
            shift = good[0]
        out["sideways_mm"] = round(shift * 1000.0, 1)
    per, out["stray"] = assign(shift)
    for r, pieces in zip(rungs, per):
        if not pieces:
            continue
        spans = sorted((p["x_lo"], p["x_hi"]) for p in pieces)
        length, (lo, hi) = 0.0, spans[0]
        for a, e in spans[1:]:
            if a > hi:
                length += hi - lo
                lo, hi = a, e
            else:
                hi = max(hi, e)
        length += hi - lo
        big = max(pieces, key=lambda p: p["area_mm2"])
        dark = max(p["dark"] for p in pieces)
        r.update(mark=True, length_mm=round(length * 1000.0, 1), width_mm=round(big["width_mm"], 1), pieces=len(pieces),
                 dark=round(dark), faint=bool(dark < MARK_FAINT_DARK),
                 at_mm=[round((min(p["x_lo"] for p in pieces) - r["x1"]) * 1000.0, 1),
                        round((max(p["x_hi"] for p in pieces) - r["x1"]) * 1000.0, 1)],     # along x, from the rung's end
                 _box=[min(p["x_lo"] for p in pieces), max(p["x_hi"] for p in pieces), min(p["y_lo"] for p in pieces),
                       max(p["y_hi"] for p in pieces)])
    z, why = first_touch(rungs, pull_mm)
    out["why"] = why
    if z is not None:
        k = [bool(r["mark"]) for r in rungs].index(True)
        out.update(read=True, first_touch_mm=z, xy=[round((rungs[k]["x0"] + rungs[k]["x1"]) / 2.0, 4), rungs[k]["y"]])
    return out


def fit_plane(points) -> dict | None:
    """Least-squares plane z = z0 + bx (x - x0) + by (y - y0) through [(x, y, z), ...] (metres), (x0, y0) their
    centroid: {"z0", "x0", "y0", "bx", "by", "residuals_mm", "max_residual_mm", "n"}; None for fewer than three
    points or points that lie (nearly) on one line (their spread across is under 15 mm)."""
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    if len(p) < 3:
        return None
    c = p[:, :2].mean(axis=0)
    d = p[:, :2] - c
    if float(np.linalg.svd(d, compute_uv=False)[-1]) / math.sqrt(len(p)) < 0.015:
        return None
    a = np.column_stack([np.ones(len(p)), d])
    sol, *_ = np.linalg.lstsq(a, p[:, 2], rcond=None)
    res = (p[:, 2] - a @ sol) * 1000.0
    return {"z0": round(float(sol[0]), 5), "x0": round(float(c[0]), 4), "y0": round(float(c[1]), 4),
            "bx": round(float(sol[1]), 4), "by": round(float(sol[2]), 4),
            "residuals_mm": [round(float(v), 2) for v in res], "max_residual_mm": round(float(np.max(np.abs(res))), 2),
            "n": int(len(p))}


def read_contact(before, after, cam, setup: dict, places: list, table_z: float) -> dict:
    """`fd paint contact`, the reading: every ladder (read_place), the plane through the places that could be
    read (fit_plane), and whether that is a measurement to paint with.
    -> {"places": [...], "read": how many, "plane": {...} | None, "usable": bool, "why": text when not,
        "warnings": [...]}. usable: at least CONTACT_MIN_PLACES places read and no residual above
    CONTACT_MAX_RESIDUAL_MM (with fewer than that many places asked for: all of them read)."""
    out_places = [read_place(before, after, cam, p, table_z) for p in places]
    got = [p for p in out_places if p["read"]]
    plane = fit_plane([(p["xy"][0], p["xy"][1], p["first_touch_mm"] / 1000.0) for p in got])
    if plane is not None:
        for p, r in zip(got, plane["residuals_mm"]):
            p["residual_mm"] = r
    warnings = []
    outside = [p["name"] for p in out_places if not p["inside_canvas"]]
    if outside:
        warnings.append("ladder(s) outside the canvas (" + ", ".join(outside) + "): measure where the picture will be "
                        "(the far strip and the margin gave other heights on 2026-11-13)")
    stray = sum(p["stray"] for p in out_places)
    if stray:
        warnings.append(f"{stray} new mark(s) near the ladders belong to no rung (drips?)")
    unread = [f"{p['name']}: {p['why']}" for p in out_places if not p["read"]]
    usable, why = True, None
    if len(places) >= CONTACT_MIN_PLACES:
        if len(got) < CONTACT_MIN_PLACES:
            usable, why = False, f"only {len(got)} of {len(places)} places could be read ({'; '.join(unread)})"
        elif plane is None:
            usable, why = False, "the places that could be read lie on one line: no plane"
        elif plane["max_residual_mm"] > CONTACT_MAX_RESIDUAL_MM:
            usable, why = False, (f"the places do not lie on a plane: a residual of {plane['max_residual_mm']:.1f} mm "
                                  f"(limit {CONTACT_MAX_RESIDUAL_MM} mm; is the sheet flat, is the brush rigid?)")
    elif unread:
        usable, why = False, "; ".join(unread)
    if usable and unread:
        warnings.append("not read: " + "; ".join(unread))
    return {"places": out_places, "read": len(got), "plane": plane, "usable": usable, "why": why, "warnings": warnings}


def contact_block(plane: dict, record: dict, read: dict, photos=None, old: dict | None = None) -> dict:
    """The "contact" block `fd paint contact --save` writes: the plane, what it was measured with, the places.
    record: the brush state the LADDERS were painted with (brush_state, from the record beside the photos); its
    "at" is when they were painted, and is kept: the time of storing is "stored_at"."""
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    block = {k: plane[k] for k in ("z0", "x0", "y0", "bx", "by")}
    block["measured"] = {**record, "at": record.get("at") or now, "stored_at": now,
                         "places": [{"name": p["name"], "xy": p["xy"], "first_touch_mm": p["first_touch_mm"],
                                     "residual_mm": p.get("residual_mm")} for p in read["places"] if p["read"]],
                         "max_residual_mm": plane["max_residual_mm"], "photos": list(photos or [])}
    touches = ", ".join(f"{p['name']} {p['first_touch_mm']:.1f}" for p in read["places"] if p["read"])
    block["note"] = (f"fd paint contact, {block['measured']['at']}: first touch (plan z, mm) {touches}; plane by least "
                     f"squares, residuals up to {plane['max_residual_mm']:.1f} mm")
    if old:
        block["previous"] = {k: old[k] for k in ("z0", "x0", "y0", "bx", "by", "note") if k in old}
    return block


def adopt_block(old: dict, record: dict) -> dict:
    """The "contact" block `fd paint contact --adopt` writes: the plane as it stands, with the brush state as it
    is NOW as what it holds for. Nothing was measured: the operator vouches that the plane was measured with
    this brush state ("adopted": true says that it is their word; a record that came from ladders does not
    have it)."""
    now = time.strftime("%Y-%m-%dT%H:%M:%S")
    block = {k: v for k, v in old.items() if k != "plane_at"}
    block["measured"] = {**record, "at": now, "adopted": True,
                         "note": "adopted by `fd paint contact --adopt`: the operator's word that the plane was measured "
                                 "with this brush state; no ladders were painted for it"}
    return block


# ----------------------------------------------------------------------------- the record beside the photos
def contact_record_path(before_photo: str) -> str:
    """Where `fd paint contact` keeps what its two photos cannot say: the name of the first photo with .json."""
    return os.path.splitext(str(before_photo))[0] + ".json"


def write_contact_record(before_photo: str, brush: dict, places: list, zs_mm, ladder_len: float, paint, setup: dict,
                         after_photo: str | None = None, at: str | None = None, sim: bool = False) -> str | None:
    """The small record beside the photos of `fd paint contact`: the brush state the ladders were painted with
    (brush_state), where (the start of every ladder, how many of its rungs were made), the heights, the paint,
    where the canvas lay, the names of the photos. `--read BEFORE AFTER --save` takes the brush state from it:
    read back later, the photos used to be stamped with whatever tool was registered THEN (a re-taped brush, a
    jaw that had moved: review of 2026-11-13). Returns the path (None when it could not be written: a full
    disk; the photos stand, `--save` will then say that there is no record)."""
    path = contact_record_path(before_photo)
    rec = {"what": "fd paint contact: the brush state and the ladders of the two photos beside this file",
           "at": at or time.strftime("%Y-%m-%dT%H:%M:%S"), "brush": dict(brush), "paint": paint,
           "zs_mm": [float(z) for z in zs_mm], "ladder_len_m": round(float(ladder_len), 5),
           "places": [{"name": p["name"], "at": list(p["at"]), "rungs": len(p["rungs"]),
                       "rungs_done": int(p.get("rungs_done", len(p["rungs"])))} for p in places],
           "canvas": {k: setup["canvas"][k] for k in ("origin_xy", "size_mm") if k in setup["canvas"]},
           "photos": {"before": str(before_photo), "after": None if after_photo is None else str(after_photo)}}
    if sim:
        rec["sim"] = True
    try:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=1, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError:
        return None
    return path


def load_contact_record(before_photo: str) -> dict | None:
    """The record beside a before photo (write_contact_record), None when there is none. Refusal (stage
    "contact") for a file that is there and is no such record."""
    path = contact_record_path(before_photo)
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as f:
            rec = json.load(f)
        ok = (isinstance(rec, dict) and isinstance(rec.get("brush"), dict) and isinstance(rec.get("places"), list)
              and rec["places"] and isinstance(rec.get("zs_mm"), list)
              and all(isinstance(p, dict) and len(p.get("at", ())) == 2 for p in rec["places"]))
        if ok:
            _num(rec.get("ladder_len_m"), "ladder_len_m", 0.005, 0.03)
            [_num(z, "zs_mm", -30.0, 30.0) for z in rec["zs_mm"]]
            [_pair(p["at"], "at", -1.0, 1.0) for p in rec["places"]]
    except (OSError, ValueError, TypeError) as e:
        raise Refusal("contact", f"the record beside the photo ({path}) cannot be read ({e})",
                      "move that file aside and give --at / --zs as when the ladders were painted to read the photos "
                      "without it (nothing can be stored then), or measure again: " + CONTACT_NEXT) from e
    if not ok:
        raise Refusal("contact", f"{path} is not a record of fd paint contact (no brush state, places or heights)",
                      "move that file aside and give --at / --zs as when the ladders were painted to read the photos "
                      "without it (nothing can be stored then), or measure again: " + CONTACT_NEXT)
    rec["path"] = path
    return rec


def record_differs(rec: dict, tool) -> list:
    """In what the registered tool differs from the one a contact record's ladders were painted with
    (brush_differs: another tool by name or attach time, or the same tool in another state: tip, gripper
    position, jaw, squeeze). [] = it is the brush of the photos."""
    return brush_differs(rec.get("brush") or {}, tool)


def contact_picture(after, cam, setup: dict, read: dict, table_z: float, path: str, ppm: float = 4.0) -> str | None:
    """The picture for the operator: the second photo from straight above with the canvas, every rung where it
    was commanded (red: it left a mark, grey: it did not) with its plan height, the marks that were found
    (boxes) and what each ladder says. Written to `path`."""
    import cv2
    x0, x1, y0, y1 = canvas_rect(setup)
    for p in read["places"]:
        for r in p["rungs"]:
            x0, x1 = min(x0, r["x1"] - MARK_BEYOND_M), max(x1, r["x0"] + MARK_BEFORE_M)
            y0, y1 = min(y0, r["y"] - 0.012), max(y1, r["y"] + 0.012)
    m = 0.02
    x0, x1, y0, y1 = x0 - m, x1 + m, y0 - m, y1 + m
    view = table_view(after, cam, x0, x1, y0, y1, table_z, ppm)
    if view.ndim == 2:
        view = cv2.cvtColor(view, cv2.COLOR_GRAY2BGR)

    def px(x, y):
        return int(round((y - y0) * 1000.0 * ppm)), int(round((x - x0) * 1000.0 * ppm))

    cx0, cx1, cy0, cy1 = canvas_rect(setup)
    cv2.rectangle(view, px(cx0, cy0), px(cx1, cy1), (0, 150, 0), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for p in read["places"]:
        for r in p["rungs"]:
            col = (0, 0, 230) if r["mark"] else (140, 140, 140)
            cv2.line(view, px(r["x0"], r["y"]), px(r["x1"], r["y"]), col, 1, cv2.LINE_AA)
            cv2.circle(view, px(r["x0"], r["y"]), 3, col, 1, cv2.LINE_AA)
            c = px(r["x0"], r["y"])
            cv2.putText(view, f"{r['plan_z_mm']:g}", (c[0] - 8, c[1] + 16), font, 0.4, col, 1, cv2.LINE_AA)
            if r.get("_box"):
                a, b = px(r["_box"][0], r["_box"][2]), px(r["_box"][1], r["_box"][3])
                cv2.rectangle(view, (a[0] - 2, a[1] - 2), (b[0] + 2, b[1] + 2), (0, 160, 255), 1)
        top = px(min(r["x1"] for r in p["rungs"]) - MARK_BEYOND_M, p["rungs"][0]["y"])
        text = f"{p['name']}: {p['first_touch_mm']:.1f} mm" if p["read"] else f"{p['name']}: not read"
        cv2.putText(view, text, (max(2, top[0] - 4), max(12, top[1] - 6)), font, 0.45, (150, 0, 0) if p["read"] else (0, 0, 200), 1, cv2.LINE_AA)
    pl = read.get("plane")
    head = "fd paint contact: first touch (plan z, mm)"
    if pl:
        head += (f"; plane z0 {pl['z0'] * 1000:.2f} mm at ({pl['x0']:.3f}, {pl['y0']:.3f}), bx {pl['bx'] * 1000:+.1f}, "
                 f"by {pl['by'] * 1000:+.1f} mm/m, residuals up to {pl['max_residual_mm']:.1f} mm")
    cv2.putText(view, head, (6, 16), font, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        return path if cv2.imwrite(path, view, [cv2.IMWRITE_JPEG_QUALITY, 92]) else None
    except (OSError, cv2.error):        # (a full disk: the reading stands, there is no picture of it)
        return None


def strip_private(read: dict) -> dict:
    """The reading without the keys that are only for the picture (for a result that is printed as JSON)."""
    for p in read.get("places", []):
        for r in p.get("rungs", []):
            r.pop("_box", None)
    return read


# ============================================================================= what is in reach (`fd paint check`)
def _keepout_at(cfg: dict, p):
    for name, box in (cfg.get("keepouts") or {}).items():
        x0, x1, y0, y1, z0, z1 = box
        if x0 <= p[0] <= x1 and y0 <= p[1] <= y1 and z0 <= p[2] <= z1:
            return name
    return None


def check_reach(pt: Painter) -> dict:
    """No motion. For every well, the dish(es), the blotting strip and the corners and the centre of the canvas:
    can the brush be brought above it at the travel height and down to the heights the skill uses there, at
    which pitch, and how far wrist_flex then is from its limits (the tool's own limits included).
    -> {"ok": everything a recipe, the washing or the picture needs is in reach and outside the keep-outs,
        "items": [{"name", "kind", "xy", "r_mm" (from the pan axis), "needed_by", "ok", "pitch",
                   "wrist_flex_deg": [lowest, highest], "wrist_margin_deg", "heights_mm", "why", "reach_mm"}],
        "problems": [names], "warnings": [...], "pitch_canvas"}"""
    arm, br = pt.arm, pt.br
    lo3, hi3 = arm.q_limits()[ARM[3]]
    try:
        canvas_pitch, canvas_why = pt.pitch_for_canvas(), None
    except SafetyError as e:
        canvas_pitch, canvas_why = None, f"no single brush pitch reaches the whole canvas ({e})"
    items, problems, warnings = [], [], []
    for v in pt.visits():
        x, y = v["xy"]
        top = np.array([x, y, pt.tz + v["top_h"]])
        low = [np.asarray(p, dtype=float) for p in v["low"]]
        it = {"name": v["name"], "kind": v["kind"], "xy": [round(x, 4), round(y, 4)],
              "r_mm": round(math.hypot(x - K._X_PAN, y) * 1000.0), "needed_by": v["needed_by"],
              "heights_mm": {"low": round((min(float(p[2]) for p in low) - pt.tz) * 1000.0, 1), "travel": round(v["top_h"] * 1000.0)}}
        try:
            for p in [top, *low]:
                ko = _keepout_at(arm.cfg, p)
                if ko and ko not in br.o.ignore_keepouts:
                    raise SafetyError(f"({p[0]:.3f}, {p[1]:.3f}) lies inside the keep-out '{ko}'")
            if v["kind"] == "canvas":
                if canvas_pitch is None:
                    raise SafetyError(canvas_why)
                pitch = canvas_pitch
            else:
                pitch = br.pitch_at([top, *low])
            q = arm.solve(top, pitch, br.roll)
            TL.check_pose(arm, arm.tool, q, min_height=v["top_h"] - 0.004, what=f"above {v['name']}")
            wrists = [float(q[3])]
            for p in low:
                q_low = arm.solve(p, pitch, br.roll)
                if v["kind"] in ("canvas", "blot"):
                    # A stroke on paper is a contact move: no floor line for it, the depth limit of brush work
                    # instead. The blotting strip too: the executor blots with strokes (Painter.blot), and a strip
                    # at table height, the one the Akabeko was painted with, was reported out of reach here while
                    # `fd paint test --do blot:2` worked (review of 2026-11-13).
                    br.guard(float(p[2]), f"at {v['name']}")
                else:
                    TL.check_pose(arm, arm.tool, q_low, what=f"at {v['name']}")
                wrists.append(float(q_low[3]))
            it.update(ok=True, pitch=round(float(pitch), 1),
                      wrist_flex_deg=[round(math.degrees(min(wrists)), 1), round(math.degrees(max(wrists)), 1)],
                      wrist_margin_deg=round(math.degrees(min(min(w - lo3, hi3 - w) for w in wrists)), 1))
            if it["wrist_margin_deg"] < 3.0:
                warnings.append(f"{v['name']}: wrist_flex comes within {it['wrist_margin_deg']:.1f} deg of its limit "
                                "(a move that ends on a limit stopped 3.7 deg short on 2026-11-13)")
        except SafetyError as e:
            it.update(ok=False, why=str(e))
            # why it cannot be used decides the advice: inside a keep-out, too LOW (the floor line of a pose, the
            # depth limit of brush work), or out of reach (then: how far the brush does reach in that direction)
            text = str(e)
            it["cause"] = ("keep-out" if "keep-out" in text else
                           "too low" if ("floor limit" in text or "below the limit of brush work" in text) else "reach")
            if it["cause"] == "reach":
                it["reach_mm"] = _reach_mm(br, v, pt.tz)
            (problems if v["needed"] else warnings).append(v["name"] if v["needed"] else
                                                           f"{v['name']} (no recipe uses it) is out of reach: {e}")
        items.append(it)
    return {"ok": not problems, "items": items, "problems": problems, "warnings": warnings,
            "pitch_canvas": None if canvas_pitch is None else round(float(canvas_pitch), 1)}


def _reach_mm(br, visit: dict, tz: float) -> int | None:
    """How far from the pan axis, on the ray through this place, the brush still reaches the heights used there
    (mm), or None when it does not even 15 cm nearer."""
    x, y = visit["xy"]
    r0 = math.hypot(x - K._X_PAN, y)
    if r0 < 1e-6:
        return None
    ux, uy = (x - K._X_PAN) / r0, y / r0
    zs = sorted({round(float(p[2]), 4) for p in visit["low"]} | {round(tz + visit["top_h"], 4)})
    for k in range(1, 31):
        r = r0 - 0.005 * k
        try:
            br.pitch_at([np.array([K._X_PAN + ux * r, uy * r, z]) for z in zs])
            return int(round(r * 1000.0))
        except SafetyError:
            continue
    return None


# ============================================================================= where to fold the arm (`fd paint rest`)
REST_CLEARANCE_M = 0.04       # tool tip above the table when folded (what was used by hand on 2026-11-13, twice)
REST_CLEARANCE_RANGE_M = (0.03, 0.15)     # what tooling.rest_q takes
REST_MARGIN_M = 0.03          # the folded brush, and where it would sink to, keep this far from the canvas
REST_WELL_M = 0.025           # ... and, when a pose allows it, this far from every well, dish and the blotting strip


def sink_of(arm, tool, q) -> tuple:
    """Where the tool tip goes when the wrist sinks with the torque off (2026-11-13 morning: it sank until the
    brush stood on the paper): wrist_flex turning from pose q until the tip meets the table. -> (lands_xy: where
    it comes down, None when the wrist reaches its limit first; low_xy: the lowest place it gets to)."""
    tl = K.tool_offset(*tool.tip_offset)
    tz = TL.table_z(arm)
    w_hi = arm.q_limits()[ARM[3]][1]
    qq = np.array(q, dtype=float)
    w, lands = float(qq[3]), None
    low = K.fk(qq, tl)[:3, 3]
    while w <= w_hi + 1e-9:
        qq[3] = w
        low = K.fk(qq, tl)[:3, 3]
        if low[2] <= tz:
            lands = low
            break
        w += math.radians(0.5)
    xy = [round(float(low[0]), 4), round(float(low[1]), 4)]
    return (None if lands is None else xy), xy


def rest_pose(arm, tool, pan_deg: float, clearance_m: float) -> dict:
    """The folded pose of tooling.rest_q for one pan, and what the brush does there: {"q", "pan_deg", "tip_xy",
    "tip_height_mm", "lands_xy", "low_xy" (sink_of)}. SafetyError when the arm cannot rest like that."""
    q = TL.rest_q(arm, tool, clearance_m=clearance_m, pan_deg=pan_deg)
    p = K.fk(q, K.tool_offset(*tool.tip_offset))[:3, 3]
    lands, low = sink_of(arm, tool, q)
    return {"q": q, "pan_deg": round(float(pan_deg), 1), "tip_xy": [round(float(p[0]), 4), round(float(p[1]), 4)],
            "tip_height_mm": round((float(p[2]) - TL.table_z(arm)) * 1000.0, 1), "lands_xy": lands, "low_xy": low}


def _rest_judge(setup: dict, pose: dict) -> dict:
    """A folded pose (rest_pose) with what the set-up says about it: "clear_of_canvas_mm" (the nearer of the tip
    and the place it would sink to; negative: inside the canvas) and "near": [what stands nearest, mm]."""
    things = [(n, w["xy"]) for n, w in setup["wells"].items()] + [(f"the {n} dish", d["xy"]) for n, d in setup["water"].items()
                                                                  if n in DISHES]
    if setup.get("blot"):
        things.append(("the blotting strip", setup["blot"]["xy"]))
    pts = [pose["tip_xy"], pose["low_xy"]]
    pose["clear_of_canvas_mm"] = round(min(canvas_distance(setup, *p) for p in pts) * 1000.0, 1)
    near = min(((n, min(math.hypot(p[0] - xy[0], p[1] - xy[1]) for p in pts)) for n, xy in things), key=lambda t: t[1],
               default=None)
    pose["near"] = None if near is None else [near[0], round(near[1] * 1000.0)]
    return pose


def rest_as_it_lies(arm, setup: dict, q=None) -> dict | None:
    """The arm as it LIES, when that is a fold `fd paint rest` would accept: the torque off, in a rest pose of its
    tool (tooling.in_rest_pose), the brush tip and the place it would sink to REST_MARGIN_M outside the canvas.
    -> what rest_choice gives ("q", "pan_deg", "tip_xy", "tip_height_mm", "lands_xy", "low_xy", "clear_of_canvas_mm",
    "near", "clearance_m", "chosen": "as it lies"), or None. `fd paint rest` then leaves the arm alone, as
    `fd tool rest` does: on 2026-11-14 the arm lay folded at pan -100 (put there by hand-typed options the
    evening before, 157 mm clear of the canvas), and the command would have switched the torque on, turned it to
    the pan it chooses itself and switched off there, where torque-off had never been tried."""
    tool = arm.tool
    q = arm.read_q() if q is None else np.asarray(q, dtype=float)
    if arm.is_torqued() or not TL.in_rest_pose(arm, tool, q)[0]:
        return None
    p = K.fk(q, K.tool_offset(*tool.tip_offset))[:3, 3]
    lands, low = sink_of(arm, tool, q)
    h = float(p[2]) - TL.table_z(arm)
    pose = _rest_judge(setup, {"q": q, "pan_deg": round(math.degrees(float(q[0])), 1),
                               "tip_xy": [round(float(p[0]), 4), round(float(p[1]), 4)],
                               "tip_height_mm": round(h * 1000.0, 1), "lands_xy": lands, "low_xy": low})
    if pose["clear_of_canvas_mm"] < REST_MARGIN_M * 1000.0:
        return None
    pose.update(clearance_m=round(h, 4), chosen="as it lies")
    return pose


def highest_rest(arm, setup: dict, pan_deg: float | None = None) -> dict:
    """rest_choice with the LARGEST clearance this arm can fold with (from REST_CLEARANCE_RANGE_M's upper end
    down in centimetres to the clearance rest_choice would take by itself): the fold for a tool whose tip was
    never measured. Its real tip may be centimetres from the registered one (2026-11-13: 9 cm further out), and
    a fold planned with the registered tip 4 cm above the table then ends with the real one ON it (simulated,
    second review: 5 and 8 mm under the table surface). "Can fold": a pose that keeps the brush off the canvas
    (rest_choice) AND that tooling takes for a rest pose (tooling.in_rest_pose: only there does `fd tool rest`
    switch the torque off). Refusal as rest_choice gives it for the lowest."""
    try:
        base = float((setup.get("rest") or {}).get("clearance_m", REST_CLEARANCE_M))
    except (TypeError, ValueError):
        base = REST_CLEARANCE_M
    for k in range(int(round((REST_CLEARANCE_RANGE_M[1] - base) * 100.0)), 0, -1):
        try:
            pose = rest_choice(arm, setup, pan_deg, round(base + 0.01 * k, 3))
        except Refusal as e:
            if e.stage != "rest":       # (a pan that was asked for and cannot be used at ANY height says so below)
                raise
            continue
        if TL.in_rest_pose(arm, arm.tool, pose["q"])[0]:
            return pose
    return rest_choice(arm, setup, pan_deg, None)


def rest_choice(arm, setup: dict, pan_deg: float | None = None, clearance_m: float | None = None) -> dict:
    """Where `fd paint rest` folds the arm: a pose in which the brush tip, AND the place it comes down on if
    the wrist sinks after the torque went off (2026-11-13 morning: it sank until the brush stood on the paper),
    lie outside the canvas by REST_MARGIN_M. The pan is the one nearest to the photo pose's (where the arm
    stands free for every photo) that does it, preferring one that also keeps REST_WELL_M from the wells, the
    dishes and the blotting strip; "rest" in the set-up or an explicit pan_deg name it instead (and are
    refused when they put the brush over the picture). Refusal (stage "rest") when no pose does it.
    -> rest_pose(...) + {"clearance_m", "clear_of_canvas_mm", "near": [name, mm], "chosen": "set-up" | "asked" |
       "nearest to the photo pose", "note"?}"""
    tool = arm.tool
    cfg = setup.get("rest") or {}
    try:
        clearance = float(clearance_m if clearance_m is not None else cfg.get("clearance_m", REST_CLEARANCE_M))
    except (TypeError, ValueError):
        clearance = float("nan")
    if not (math.isfinite(clearance) and REST_CLEARANCE_RANGE_M[0] <= clearance <= REST_CLEARANCE_RANGE_M[1]):
        # (said here: inside the scan below the refusal of tooling.rest_q was swallowed pan by pan, and the answer
        # was "no folded pose keeps the brush off the canvas ... take the sheet away")
        raise Refusal("options", f"--clearance {clearance_m if clearance_m is not None else cfg.get('clearance_m')!r}: the "
                                 f"brush tip above the table when folded, {REST_CLEARANCE_RANGE_M[0]:g} .. "
                                 f"{REST_CLEARANCE_RANGE_M[1]:g} m (default {REST_CLEARANCE_M:g})",
                      "`fd paint rest` without --clearance, or `fd paint rest --clearance 0.05`")
    def judge(pan):
        pose = _rest_judge(setup, rest_pose(arm, tool, pan, clearance))
        pose["clearance_m"] = clearance
        return pose

    if pan_deg is not None or "pan_deg" in cfg:
        asked = pan_deg is not None
        pan = float(pan_deg if asked else cfg["pan_deg"])
        try:
            pose = judge(pan)
        except SafetyError as e:
            raise Refusal("rest", f"the arm cannot fold at pan {pan:g} with this brush: {e}",
                          "`fd paint rest` without --pan chooses a pose", pan_deg=pan) from e
        if pose["clear_of_canvas_mm"] < REST_MARGIN_M * 1000.0:
            raise Refusal("rest", f"folded at pan {pan:g} the brush would be {pose['clear_of_canvas_mm']:.0f} mm from the "
                                  f"canvas (tip at {pose['tip_xy']}, sinking to {pose['low_xy']}): it must keep "
                                  f"{REST_MARGIN_M * 1000:.0f} mm, a wet brush that sinks would stand on the picture",
                          "`fd paint rest` without --pan chooses a pose" if asked else
                          "take \"rest\" out of the set-up (or correct it); `fd paint rest` then chooses a pose", pan_deg=pan)
        pose["chosen"] = "asked" if asked else "set-up"
        return pose
    lim = arm.q_limits()[ARM[0]]
    pan0 = float(setup["photo_q_deg"][0])
    grid = [p for p in np.arange(-180.0, 180.1, 5.0) if lim[0] <= math.radians(p) <= lim[1]]
    hard, tried, cannot = None, 0, None
    for pan in sorted({round(pan0, 1), *grid}, key=lambda p: (abs(p - pan0), p)):
        if not lim[0] <= math.radians(pan) <= lim[1]:
            continue
        try:
            pose = judge(float(pan))
        except SafetyError as e:
            cannot = cannot or e
            continue
        tried += 1
        if pose["clear_of_canvas_mm"] < REST_MARGIN_M * 1000.0:
            continue
        if pose["near"] is None or pose["near"][1] >= REST_WELL_M * 1000.0:
            pose["chosen"] = "nearest to the photo pose"
            return pose
        hard = hard or pose
    if hard is not None:
        hard["chosen"] = "nearest to the photo pose"
        hard["note"] = (f"no pose keeps {REST_WELL_M * 1000:.0f} mm from everything on the table: this one is "
                        f"{hard['near'][1]} mm from {hard['near'][0]}")
        return hard
    if not tried and cannot is not None:
        # (not a matter of the canvas: the arm cannot fold like this at any pan, e.g. with the tip 15 cm up)
        if clearance_m is not None:
            next_ = f"`fd paint rest` without --clearance (the tip {REST_CLEARANCE_M * 1000:.0f} mm up), or with a smaller one"
        elif "clearance_m" in cfg:
            next_ = ("correct \"rest\": \"clearance_m\" in the set-up, or take it out (`fd paint rest` then folds with the tip "
                     f"{REST_CLEARANCE_M * 1000:.0f} mm up)")
        else:
            # No height was asked for: the same command again is the same refusal (it was the `next` here, and
            # `fd calib-air`, which holds the arm when no fold is safe, sends the operator to this command). What
            # tooling says for a tool the arm cannot rest with:
            next_ = ("the arm cannot be folded away with this tool on: it stays as it is (it holds while its torque is on). "
                     "Support the arm by hand, take the tool off, then `fd tool detach`; `fd tool status` shows the tool "
                     "and its joint limits (the reason names the limit when it is one of them that excludes the fold)")
        raise Refusal("rest", f"the arm cannot fold with the brush tip {clearance * 1000:.0f} mm above the table at any pan "
                              f"angle: {cannot}", next_, cannot_fold=True)
    raise Refusal("rest", f"no folded pose keeps the brush {REST_MARGIN_M * 1000:.0f} mm off the canvas "
                          f"({tried} pan angles tried with the tip {clearance * 1000:.0f} mm above the table): with the "
                          "torque off the brush could sink onto the picture",
                  "leave the torque on (the arm holds where it is), take the sheet away, then `fd tool rest --over-canvas` "
                  "(the word that the sheet is gone: without it fd tool rest refuses the same fold while the set-up "
                  "still has the canvas)")


def rest_report(arm, tool=None, setup: dict | None = None, pan_deg: float | None = None,
                clearance_m: float | None = None) -> dict:
    """For every command that folds the arm with a brush on (`fd tool rest`, the end of `fd calib-air`): would
    the folded brush be over the picture? The same rule as `fd paint rest` (rest_choice), for ONE fold: the
    tool's rest pose at pan_deg / clearance_m (None: tooling's defaults, pan 20 and 5 cm, which with the rig's
    canvas of 2026-11-13 put the brush tip 40 mm INSIDE the canvas and let it sink onto the picture).

    -> {"ok": False when the folded tip, or the place it comes down on if the wrist sinks with the torque off,
              lies over the canvas or nearer to it than the margin `fd paint rest` keeps,
        "why": the reason in words ("" when ok), "tip_xy", "sink_xy" (the lowest place the tip sinks to),
        "margin_mm": that margin (REST_MARGIN_M), "clear_of_canvas_mm": how far the nearer of the two is outside
        the canvas (negative: inside), "pan_deg", "clearance_m", "next": the command that folds the arm off the
        picture}.
    tool None: the arm's tool. setup None: the painting set-up file, when there is one (no file, or one that
    cannot be used: ok True, and "why" says that nothing is known of a canvas). Reads the arm's joint limits;
    moves nothing, writes nothing."""
    tool = getattr(arm, "tool", None) if tool is None else tool
    pan = float(TL.REST_PAN_DEG if pan_deg is None else pan_deg)
    clearance = float(TL.REST_CLEARANCE if clearance_m is None else clearance_m)
    out = {"ok": True, "why": "", "tip_xy": None, "sink_xy": None, "margin_mm": round(REST_MARGIN_M * 1000.0, 1),
           "clear_of_canvas_mm": None, "pan_deg": pan, "clearance_m": clearance,
           "next": "`fd paint rest` folds the arm where neither the brush nor the place it would sink to is over the picture"}
    if setup is None:
        if not os.path.exists(setup_path()):
            out["why"] = "no painting set-up: nothing is known of a canvas"
            return out
        try:
            setup = load_setup()
        except Refusal as e:
            out["why"] = f"the painting set-up cannot be used ({e}): nothing is known of a canvas"
            return out
    if tool is None:
        out["why"] = "no tool is registered"
        return out
    try:
        pose = rest_pose(arm, tool, pan, clearance)
    except SafetyError as e:
        out.update(ok=False, why=f"the arm cannot fold at pan {pan:g} with the tip {clearance * 1000:.0f} mm up: {e}")
        return out
    clear = min(canvas_distance(setup, *pose["tip_xy"]), canvas_distance(setup, *pose["low_xy"]))
    out.update(tip_xy=pose["tip_xy"], sink_xy=pose["low_xy"], clear_of_canvas_mm=round(clear * 1000.0, 1))
    if clear < REST_MARGIN_M:
        where = "INSIDE the canvas" if clear < 0.0 else "from the canvas"
        out.update(ok=False, why=(f"folded at pan {pan:g} the brush tip is at {pose['tip_xy']} and sinks to {pose['low_xy']} "
                                  f"when the wrist gives way with the torque off: {abs(clear) * 1000:.0f} mm {where} of the "
                                  f"painting set-up (it must keep {REST_MARGIN_M * 1000:.0f} mm; on 2026-11-13 the wrist sank "
                                  "until the brush stood on the paper)"))
    return out


# ============================================================================= pictures of a plan (`fd paint preview`)
_PAINT_RGB = {"red": (226, 52, 38), "crimson": (150, 24, 40), "brown": (122, 62, 48), "yellow": (246, 196, 44),
              "black": (28, 28, 34), "ink": (28, 28, 34), "ink2": (28, 28, 34), "cerulean": (36, 112, 200),
              "blue": (28, 48, 140), "navy": (22, 30, 85), "green": (44, 150, 76), "violet": (92, 52, 140),
              "magenta": (200, 40, 120)}
_TINT_OPACITY = {"pale": 0.34, "mid": 0.58}


def paint_rgb(plan: dict, name: str) -> tuple:
    """(r, g, b) of a layer's paint for a picture: what the plan says, else the tube colour by name."""
    v = (plan.get("paints") or {}).get(name)
    if isinstance(v, dict) and "rgb" in v:
        v = v["rgb"]
    if isinstance(v, (list, tuple)) and len(v) == 3:
        return tuple(int(c) for c in v)
    base = name.rsplit("_", 1)[0] if name.rsplit("_", 1)[-1] in _TINT_OPACITY else name
    known = getattr(RD, "DEFAULT_PAINTS", {})
    if base in known:
        return tuple(int(c) for c in known[base]["rgb"])
    return _PAINT_RGB.get(base, (90, 90, 90))


def render_plan(plan: dict, ppm: float = 4.0) -> tuple:
    """(BGR image of the simulated picture, which simulation made it). A plan for a round brush is drawn by
    roundbrush.render when this harness has that module, everything else by render.render (the flat brush of
    the morning: for a round brush's plan its bands are then only roughly right)."""
    note = None
    if plan_brush_kind(plan) == "round":
        try:
            import roundbrush
            return roundbrush.render(plan, ppm), "roundbrush.render"
        except ImportError:
            note = "render.render (flat-brush model: roundbrush.py is not in this harness)"
        except Exception as e:              # a picture is not worth a traceback: fall back to the other simulation
            note = f"render.render (roundbrush.render failed: {type(e).__name__}: {e})"
    paints = {}
    for layer in plan["layers"]:
        name = layer["paint"]
        v = (plan.get("paints") or {}).get(name)
        if isinstance(v, dict) and "rgb" in v:
            paints[name] = v
        elif name in getattr(RD, "DEFAULT_PAINTS", {}) and v is None:
            continue
        else:
            tint = name.rsplit("_", 1)[-1]
            paints[name] = {"rgb": list(paint_rgb(plan, name)), "opacity": _TINT_OPACITY.get(tint, 0.86)}
    return RD.render({**plan, "paints": paints}, px_per_mm=ppm), note or "render.render"


def design_stats(plan: dict) -> tuple:
    """(statistics of the plan by the kit it was designed with, which one): roundbrush.stats for a round
    brush's plan when that module is there, else render.plan_stats. For information: what the executor
    refuses is plan_problems, how long it takes plan_budget."""
    if plan_brush_kind(plan) == "round":
        try:
            import roundbrush
            return roundbrush.stats(plan), "roundbrush.stats"
        except Exception:                   # (not there, or it does not like the plan)
            pass
    try:
        return RD.plan_stats(plan), "render.plan_stats"
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}, "render.plan_stats"


def _every_mm(pts, step_mm: float) -> np.ndarray:
    """Points every step_mm along a polyline [(u, v), ...] (both ends kept)."""
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    out = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(math.ceil(float(np.linalg.norm(b - a)) / step_mm)))
        out += [a + (b - a) * (k / n) for k in range(1, n + 1)]
    return np.array(out)


def overlay_plan(img, cam, setup: dict, plan: dict):
    """The strokes of a plan drawn over a photo, where the camera model puts them at their contact height
    (the paint itself lands about 6 mm beside and 9 mm nearer to the robot: 2026-11-13). Returns a new image."""
    import cv2
    out = img.copy()
    x0, x1, y0, y1 = canvas_rect(setup)
    corners = [[x, y, contact_z(setup, x, y)] for x, y in ((x0, y0), (x0, y1), (x1, y1), (x1, y0))]
    cv2.polylines(out, [np.round(cam.project(corners)).astype(np.int32)], True, (0, 150, 0), 1, cv2.LINE_AA)
    brush = brush_of(setup)
    for layer in plan["layers"]:
        r, g, b = paint_rgb(plan, layer["paint"])
        for st in layer["strokes"]:
            pts = []
            for u, v in _every_mm(st["pts"], 2.0):
                x, y = canvas_xy(setup, u, v)
                pts.append([x, y, contact_z(setup, x, y)])
            uv = np.round(cam.project(pts)).astype(np.int32)
            w = brush_width_mm(brush, float(st.get("press", 1.0)))
            cv2.polylines(out, [uv], False, (int(b), int(g), int(r)), 2 if w is None or w < 4.0 else 4, cv2.LINE_AA)
    return out


# ============================================================================= where the paints are, from a photo
def _palette_fit(src, dst) -> tuple:
    """The move of the palette as ONE rigid piece: the rotation and shift that take the well places `src` onto
    the blobs `dst` by least squares -> (function: place before -> place after, the turn in degrees). With fewer
    than LOCATE_MIN_AGREE pairs, or pairs that lie within 4 cm of one another, the shift alone (a turn fitted
    to them would throw the wells at the other end of the palette anywhere)."""
    src, dst = np.asarray(src, dtype=float).reshape(-1, 2), np.asarray(dst, dtype=float).reshape(-1, 2)
    c_src, c_dst = src.mean(axis=0), dst.mean(axis=0)
    a, b = src - c_src, dst - c_dst
    turn = 0.0
    if len(src) >= LOCATE_MIN_AGREE and float(np.linalg.norm(a, axis=1).max()) >= 0.04:
        turn = math.atan2(float((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]).sum()), float((a * b).sum()))
    cs, sn = math.cos(turn), math.sin(turn)
    rot = np.array([[cs, -sn], [sn, cs]])
    return (lambda p: (np.asarray(p, dtype=float) - c_src) @ rot.T + c_dst), math.degrees(turn)


def relocate_wells(img, cam, setup: dict, table_z: float, max_move_m: float = 0.03) -> dict:
    """Find the paint blobs of the set-up again in a photo (the palette gets pushed when water is changed:
    12 mm on 2026-11-13). Blobs are the saturated or very dark pieces inside the palette's area (the hull of
    the known wells, 3 cm wider, at blob height) whose MIDDLE is paint (the middle of a ring, of two lumps joined
    across a divider or of a streak is nobody's blob). The palette moves as ONE piece:
    1. the shift that most wells agree on (a blob within 8 mm of where it puts them) is a first guess of it, and
       every well takes the free blob nearest to its shifted place (at most 12 mm from it);
    2. shift AND turn are then fitted to the wells that agree (_palette_fit), every well takes the free blob
       within LOCATE_AGREE_M of where the fit puts it, and the fit is made again from those (a palette that was
       turned as well as pushed: the shift alone leaves the wells at its far end without their blobs);
    3. a well WITHOUT a blob of its own is put where the fit puts it only when the photo shows paint there (its
       blob has run into a neighbour's: in the photos of the evening of 2026-11-13 two or three of the ten wells
       share a piece with the next one); else it KEEPS the place it had and is not located. Where the palette's
       move would put it is kept beside it ("expected_xy"): the next photo looks for it THERE (a well that was
       left behind once is otherwise never found again by a palette that moves as one piece).
    In 1 and 2 a piece that lies within reach of TWO wells (LOCATE_AGREE_M round where the shift, then the fit,
    puts them) is given to neither: its middle is the place of neither, and the fit is made without it. Only a
    fit that leaves ONE well within reach of a piece gives the piece to that well.
    No well is put where the photo shows no paint: EVERY place that is written, by a blob or by the fit, passes
    the same test at the place as it is written (paint_at: half of a disc of 3 mm), whatever the matching did;
    a well whose place fails it keeps the place it had and is not located (no paint is loaded from it:
    require_wells). It used to be: the wells that were not found were moved by the median shift of the others,
    and with the palette as it lay that morning (17 mm away and turned) four wells were written 14 to 18 mm
    beside their blobs, onto bare palette, where the next load would have scrubbed (second review of
    2026-11-13). And after that: with the set-up 3 to 6 deg off the palette's turn (eight real photos of the
    evening, 968 turns and pushes) a well took the middle of the piece that violet and magenta share, or a
    stray piece of 12 mm, as its own blob; that ONE pair bent the fit, and in 33 results a well was written onto
    the white divider, in 52 more over 8 mm from its blob (third review). With the three rules: none of the 968.
    The test is the photo's colour and no more: the dark desk beside the palette is "paint" to it as well.
    -> {"wells": {name: {"xy", "moved_mm", "located": bool, "by": "blob" | "paint" | None, "expected_xy"?}},
        "missing": [names without a blob of their own], "not_located": [names that keep their place: no paint
        where the palette's move puts them, or at the place that would be written], "shift_mm": [dx, dy] (median
        of the wells found by a blob), "turn_deg", "found": wells that took a blob of their own, "agree": wells
        that had a blob within 8 mm of where the first shift put them, "blobs": the coloured pieces (also those
        that are nobody's blob), "blob_xy": [[x, y], ...]}.
    Nothing is written: the caller decides (locate_usable says whether this is a palette)."""
    import cv2
    wells = setup["wells"]
    z = table_z + 0.004
    names = list(wells)
    # where each well is looked for: its place, or, for one an earlier photo did not find, where the palette's
    # move of then would have put it
    pts = np.array([wells[n]["expected_xy"] if wells[n].get("located") is False and wells[n].get("expected_xy") else
                    wells[n]["xy"] for n in names], dtype=float).reshape(-1, 2)
    kept = np.array([wells[n]["xy"] for n in names], dtype=float).reshape(-1, 2)
    if not len(pts):
        return {"wells": {}, "missing": [], "not_located": [], "shift_mm": [0.0, 0.0], "turn_deg": 0.0, "blobs": 0, "found": 0,
                "agree": 0, "blob_xy": []}
    lo, hi = pts.min(axis=0) - 0.03, pts.max(axis=0) + 0.03
    corners = [[lo[0], lo[1], z], [lo[0], hi[1], z], [hi[0], hi[1], z], [hi[0], lo[1], z]]
    roi = np.zeros(img.shape[:2], np.uint8)
    cv2.fillPoly(roi, [np.round(cam.project(corners)).astype(np.int32)], 1)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1].astype(int), hsv[..., 2].astype(int)
    paint = ((((s > 110) & (v > 40)) | ((v < 75) & (s > 30))) & (roi > 0)).astype(np.uint8)
    paint = cv2.morphologyEx(paint, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))         # what is paint-coloured
    mask = cv2.morphologyEx(paint, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))         # ... gathered into pieces
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)

    def disc(p, r_m):
        """The photo round table place p -> (its window, which pixels of it lie within r_m of p); None at the
        photo's edge."""
        uv = cam.project([[p[0], p[1], z], [p[0] + r_m, p[1], z], [p[0], p[1] + r_m, z]])
        r = max(2, int(round(float(max(np.linalg.norm(uv[1] - uv[0]), np.linalg.norm(uv[2] - uv[0]))))))
        x, y = int(round(float(uv[0][0]))), int(round(float(uv[0][1])))
        if not (r <= x < paint.shape[1] - r and r <= y < paint.shape[0] - r):
            return None
        yy, xx = np.ogrid[-r:r + 1, -r:r + 1]
        return (slice(y - r, y + r + 1), slice(x - r, x + r + 1)), xx * xx + yy * yy <= r * r

    def paint_at(p) -> bool:
        """Does the photo show paint at table place p (most of a disc of LOCATE_PAINT_R_M around it)?"""
        d = disc(p, LOCATE_PAINT_R_M)
        return d is not None and float(paint[d[0]][d[1]].mean()) >= LOCATE_PAINT_PART

    def pieces_near(p) -> set:
        """The pieces within reach of a well that stands at table place p (LOCATE_AGREE_M round it)."""
        d = disc(p, LOCATE_AGREE_M)
        return set() if d is None else {int(k) for k in np.unique(lab[d[0]][d[1]]) if k}

    blobs, b, piece = [], [], []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < 60:
            continue
        p = cam.to_plane(tuple(cent[k]), z)
        blobs.append((float(p[0]), float(p[1])))
        if paint_at(blobs[-1]):          # (a piece whose middle is not paint is nobody's blob: it takes no part)
            b.append(blobs[-1])
            piece.append(k)
    b = np.array(b, dtype=float).reshape(-1, 2)
    shift = np.zeros(2)
    best = (0, 0.0)
    if len(b):
        # every (blob - well) within reach is a candidate for the palette's shift; the one most wells agree on wins
        best = (-1, 0.0)
        for i in range(len(pts)):
            for j in range(len(b)):
                c = b[j] - pts[i]
                if float(np.hypot(*c)) > max_move_m:
                    continue
                d = np.linalg.norm((pts + c)[:, None, :] - b[None, :, :], axis=2).min(axis=1)
                key = (int((d <= LOCATE_AGREE_M).sum()), -float(np.hypot(*c)))
                if key > best:
                    best, shift = key, c

    def take(places, tol) -> dict:
        """{well index: blob index}: every well the free blob nearest to its place, the nearest pairs first, no
        blob further than tol from the place or than max_move_m from where the well was, and no piece that
        lies within reach of two of the wells as they stand at `places` (it is given to neither)."""
        if not len(b):
            return {}
        reach = {}
        for i in range(len(pts)):
            for k in pieces_near(places[i]):
                reach.setdefault(k, set()).add(i)
        d = np.linalg.norm(places[:, None, :] - b[None, :, :], axis=2)
        pairs, used = {}, set()
        for i, j in sorted(((i, j) for i in range(len(pts)) for j in range(len(b))), key=lambda ij: d[ij]):
            if d[i, j] > tol:
                break
            if i in pairs or j in used or float(np.hypot(*(b[j] - pts[i]))) > max_move_m or len(reach.get(piece[j], ())) > 1:
                continue
            pairs[i] = j
            used.add(j)
        return pairs

    move, turn = (lambda p: np.asarray(p, dtype=float) + shift), 0.0
    pairs = take(pts + shift, 0.012)
    agreeing = [i for i, j in pairs.items() if float(np.linalg.norm(pts[i] + shift - b[j])) <= LOCATE_AGREE_M]
    idx = sorted(agreeing) if len(agreeing) >= 2 else sorted(pairs)
    for _ in range(3):
        if not idx:
            break
        move, turn = _palette_fit(pts[idx], b[[pairs[i] for i in idx]])
        again = take(move(pts), LOCATE_AGREE_M)
        if again == pairs and idx == sorted(pairs):
            break
        pairs, idx = again, sorted(again)
    out, missing, lost, moves = {}, [], [], []
    for i, name in enumerate(names):
        expected = None
        xy, by = (b[pairs[i]], "blob") if i in pairs else (move(pts[i]), "paint")
        near = float(np.hypot(*(xy - pts[i]))) <= max_move_m
        # The last gate, whatever the matching did: the place AS IT IS WRITTEN shows paint in the photo, or it
        # is not written (also a place taken from a blob: the middle of a piece is not always paint)
        xy = np.array([round(float(xy[0]), 4), round(float(xy[1]), 4)])
        if not near or not paint_at(xy):
            fitted = move(pts[i])
            expected = fitted if float(np.hypot(*(fitted - pts[i]))) <= max_move_m else pts[i]
            xy, by = kept[i], None               # it keeps its place: nothing in the photo says where it is
            lost.append(name)
        if by == "blob":
            moves.append(xy - pts[i])
        else:
            missing.append(name)
        out[name] = {"xy": [float(v) for v in wells[name]["xy"]] if by is None else [float(xy[0]), float(xy[1])],
                     "moved_mm": round(float(np.hypot(*(xy - kept[i]))) * 1000.0, 1), "located": by is not None, "by": by}
        if expected is not None:
            out[name]["expected_xy"] = [round(float(expected[0]), 4), round(float(expected[1]), 4)]
    med = np.median(np.array(moves), axis=0) if moves else np.zeros(2)
    return {"wells": out, "missing": missing, "not_located": lost,
            "shift_mm": [round(float(med[0]) * 1000.0, 1), round(float(med[1]) * 1000.0, 1)], "turn_deg": round(float(turn), 1),
            "blobs": len(blobs), "found": len(names) - len(missing), "agree": max(0, int(best[0])),
            "blob_xy": [[round(x, 4), round(y, 4)] for x, y in blobs]}


def locate_usable(found: dict) -> tuple:
    """(may the wells of a relocate_wells result be stored?, why not). Stored only when the photo shows the
    palette: at least LOCATE_MIN_PART of the wells (6 of 10) found by a blob of their own, and the palette's
    shift agreed on by at least LOCATE_MIN_AGREE wells (all of them where the set-up has fewer). `fd paint
    locate` used to store whatever one photo gave: a photo without a palette "0 found, the 10 not found were
    shifted the same, stored"; ONE saturated blob 25 mm beside the red well moved all ten wells by 25 mm,
    onto the bare palette and its rim (review of 2026-11-13)."""
    n = len(found.get("wells") or {})
    if not n:
        return False, "the set-up has no wells"
    need = max(int(math.ceil(LOCATE_MIN_PART * n - 1e-9)), min(n, LOCATE_MIN_AGREE))
    agree = min(n, LOCATE_MIN_AGREE)
    got, ok = int(found.get("found", n - len(found.get("missing") or []))), int(found.get("agree", 0))
    if got < need:
        return False, (f"only {got} of {n} wells were found by a blob of their own in the photo ({found.get('blobs', 0)} "
                       f"coloured piece(s) in the palette's area); {need} are needed to call it the palette")
    if ok < agree:
        return False, (f"only {ok} well(s) agree on where the palette moved to; {agree} are needed (one stray blob is no palette)")
    return True, None


def locate_picture(img, cam, setup: dict, found: dict, table_z: float, path: str) -> str | None:
    """The picture for the operator of `fd paint locate`: the photo with every well where the set-up has it
    (grey ring, its name), every coloured piece that was found (yellow dot), and where each well would be put
    (green ring: found by a blob of its own; blue ring: no blob of its own, but paint where the palette's move
    puts it; red ring with a cross, at its old place: not located, it keeps that place). Written to `path`."""
    import cv2
    out = img.copy()
    z = table_z + 0.004
    font = cv2.FONT_HERSHEY_SIMPLEX

    def px(xy):
        u, v = cam.project([[float(xy[0]), float(xy[1]), z]])[0]
        return int(round(float(u))), int(round(float(v)))

    for xy in found.get("blob_xy") or []:
        cv2.circle(out, px(xy), 4, (0, 220, 255), -1, cv2.LINE_AA)
    for name, w in setup["wells"].items():
        c = px(w["xy"])
        cv2.circle(out, c, 16, (150, 150, 150), 1, cv2.LINE_AA)
        cv2.putText(out, name, (c[0] + 18, c[1] + 4), font, 0.45, (60, 60, 60), 1, cv2.LINE_AA)
        new = (found.get("wells") or {}).get(name)
        if new and not new.get("located", True):
            cv2.circle(out, c, 12, (0, 0, 220), 2, cv2.LINE_AA)
            cv2.line(out, (c[0] - 8, c[1] - 8), (c[0] + 8, c[1] + 8), (0, 0, 220), 2, cv2.LINE_AA)
            cv2.line(out, (c[0] - 8, c[1] + 8), (c[0] + 8, c[1] - 8), (0, 0, 220), 2, cv2.LINE_AA)
        elif new:
            cv2.circle(out, px(new["xy"]), 12, (0, 170, 0) if new.get("by") == "blob" else (220, 120, 0), 2, cv2.LINE_AA)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        return path if cv2.imwrite(path, out, [cv2.IMWRITE_JPEG_QUALITY, 90]) else None
    except (OSError, cv2.error):        # (a full disk: the result stands, there is no picture of it)
        return None


def canvas_view(img, cam, setup: dict, px_per_mm: float = 4.0, margin_mm: float = 12.0, z: float | None = None):
    """The canvas as seen from straight above: the photo warped onto the paper plane (BGR, px_per_mm pixels per
    millimetre, margin_mm of paper around the canvas; the plane is at the contact height z0 of the set-up
    unless z says otherwise). What the camera sees obliquely becomes comparable with a plan's preview."""
    import cv2
    ox, oy = setup["canvas"]["origin_xy"]
    w, h = setup["canvas"]["size_mm"]
    z = float(setup["contact"]["z0"]) if z is None else float(z)
    us = (np.arange(int(round((w + 2 * margin_mm) * px_per_mm)), dtype=np.float64) + 0.5) / px_per_mm - margin_mm
    vs = (np.arange(int(round((h + 2 * margin_mm) * px_per_mm)), dtype=np.float64) + 0.5) / px_per_mm - margin_mm
    uu, vv = np.meshgrid(us, vs)
    pts = np.column_stack([(ox + vv / 1000.0).ravel(), (oy + uu / 1000.0).ravel(), np.full(uu.size, z)])
    uv = cam.project(pts).reshape(uu.shape + (2,)).astype(np.float32)
    return cv2.remap(img, uv[..., 0], uv[..., 1], cv2.INTER_LINEAR, borderValue=(0, 0, 0))
