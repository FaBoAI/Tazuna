"""The thin round brush: a kit to design a picture for it, a picture of what it will really leave on the sheet,
and the numbers a plan is judged by (loads, minutes, what the robot would refuse).

Why this module exists. On 2026-11-13 at 15:50 the flat brush of the morning (bristles about 9 mm wide;
render.py, shapes.py) was replaced by a thin round brush with a pointed tip (about 4 mm wide, 15 mm long).
Nothing of the flat brush holds for it: it paints in every direction, so a wash is parallel bands in any
direction and not a radial zig-zag.
That evening a design kit was written for it; three designers made the Kaminarimon drafts with it and the
picture that was painted (paintings/kaminarimon.py) came out of it. This module is that kit with its names
unchanged (Plan, Layer with line / band / dot / fill_mask / fill_polygon / fill_ellipse, hatch_mask, split, arc,
render, stats, width_of, tints by the suffixes _pale / _mid), made honest about what the brush did on the
sheet: the kit showed the picture as it was designed and said 35 minutes; the sheet looked different in ways
that could have been known, and it took 50.

Measured on the real arm (2026-11-13, 17:40 to 19:10; brush rigid in the closed jaw, pitch 65, set-up press_mm
5.0; the photos are runs/20261113_paint/*.jpg of that rig, the check pattern 174946_run.jpg, the picture layer
by layer 181520 .. 190853_run.jpg):
- Two modes are reliable. press 0.6 (3 mm below first touch) is a LINE: 1 to 1.5 mm wide when it runs sideways,
  2.5 to 3 mm when it runs along the direction to the robot's pan axis. press 1.2 (6 mm, the guard's maximum)
  is a BAND of 4 to 5 mm in every direction (the belly of the brush lies on the paper). Between the two the
  width jumps (2 mm at 4 mm of press, 5 mm at 6): a plan uses 0.6 or 1.2. More than 1.2 the executor refuses
  at that set-up (brush work allows 6 mm below first touch), and `stats` says so beforehand (press_limit).
- The six band layers of the picture were NOT run at 6 mm: for them the contact plane was lowered by 1.5 mm
  (z0 3.9 instead of 5.42 mm, by hand in the set-up file), because the first band at the far middle of the
  sheet had come out a thin line; the written lines were run with the plane as measured. A plan cannot say
  that with its press. The command can: `fd paint run PLAN --layer N --lower-mm 1.5`.
- The first stroke after a load is the darkest and the widest, up to 5 mm even at the light press; the paint
  fades within 60 to 90 mm of band, about 140 mm of fine line. The bands of the picture were run with a load
  every 75 mm, the written lines with 110, the retraced lettering with 60.
- Tints. `red_pale` (water twice, a touch of the blob) is a good salmon wash; `red_mid` (water once, scrub
  once) is hardly darker. The same pale recipe with dark blue and with black came out nearly full strength
  after the first band (the touch takes too much of a dark, wet blob); in the photos the pale yellow did the
  same. A real dilution step is missing: a pale blue cannot be promised.
- Lettering: after loading, the brush is pressed out on the blotting strip (paints "ink", "ink2"), else the
  first stroke is a 5 mm band. Black written over red that was still damp turned grey; the same strokes
  retraced with dense black ("ink2": no water dip) are solid.
- Marks land about (-6, -9) mm (u, v) from the plan (LANDING_MM, where it is told in which view); the picture
  of the sheet (render with a margin) shows them there.
- Times of five calls of `fd paint run` (MEASURED_CALLS); `seconds` is fitted to them. It meets those five
  within 5 %, which is the fit looking at its own data. For work it has not seen it is good to about a quarter:
  fitted to four of the calls and asked for the fifth it is 10 to 22 % off (leave_one_out), and of eleven other
  calls of that evening ten are met within 12 % and one is 26 % too long (UNSEEN_CALLS).

Canvas frame as everywhere in tools/paint: millimetres, u to the right, v DOWN, (0, 0) = top left as the camera
and the person in front of the table see the sheet; the robot stands beyond the TOP edge, its pan axis at
plan["pan_axis_mm"] (negative v).

Plan (the file format of the harness, render.py):
  {"canvas_mm": [224, 153], "pan_axis_mm": [118.0, -107.2],
   "brush": {"kind": "round", "wide_mm": W, "broad_mm": WB, "load_mm": L},
   "paints": {name: [r, g, b]},
   "layers": [{"name": str, "paint": name, "strokes": [{"pts": [[u, v], ...], "press": 0.6 | 1.2}, ...]}, ...]}
A layer's paint is a tube paint ("red"), a tint of it ("red_mid", "red_pale") or a writing black ("ink",
"ink2": the black tube, blotted). Layers are painted in order, one `fd paint run PLAN --layer N` each; within a
layer the strokes in the order given; the robot lifts the brush between strokes and loads it again when the
paint used since the last load, plus the next stroke, would pass one load (never inside a stroke: no stroke
may use more than one load). Every call starts with a load: a layer that is painted in two calls (a call ends
after `--max-min` minutes, 8 unless told otherwise) takes one load more than is counted here.

Use:
    import roundbrush as RB
    p = RB.Plan(pan_axis_mm=painter.pan_axis_mm())          # W = 2.7 (fine), WB = 5.0 (broad), L = 140
    red = p.layer("red woodwork", "red")
    red.line([(30, 40), (30, 120)])                         # a fine line (press 0.6), cut by the load length
    red.band([(40, 40), (40, 120)])                         # a broad band (press 1.2)
    red.fill_polygon([(40, 60), (90, 60), (90, 110), (40, 110)], angle_deg=0)    # a wash, the brush kept inside
    red.fill_ellipse((112, 80), (30, 42), angle_deg=90, pitch=7.5)               # a broken wash: paper between
    p.save("plan.json"); cv2.imwrite("preview.png", p.render()); print(p.stats())
    p.stats(setup=paintskill.load_setup())                  # ... with the press limit of the set-up in force
    p.render(as_painted=False)                              # the picture as designed (what the kit showed)
    p.render(margin_mm=12)                                  # the sheet: where the marks land beside the canvas

Nothing here touches the arm or the camera (numpy and OpenCV only). Tests: tests/test_paint_model.py.
"""

from __future__ import annotations

import json
import math

import cv2
import numpy as np

CANVAS = (224.0, 153.0)
# The robot's pan axis in canvas millimetres for the canvas origin (0.146, -0.118) m of the evening's set-up
# (the kit had (131.0, -107.2): the canvas was moved by 13 mm before the picture was painted). The executor
# refuses a plan whose pan axis is not the set-up's (paintskill): pass the present one to Plan().
PAN_AXIS = (118.0, -107.2)
EDGE = 5.0                      # no stroke point nearer than this to the canvas edge
PAPER = (246, 244, 238)         # RGB of the bare sheet in the preview
# what the tube paints on the palette look like when brushed out (RGB), for the preview only
PAINTS = {
    "red": (226, 52, 38),       # bright scarlet: the big well, plenty of it
    "crimson": (150, 24, 40),   # dark red
    "brown": (122, 62, 48),
    "yellow": (246, 196, 44),
    "black": (28, 28, 34),
    "cerulean": (36, 112, 200), # mid blue
    "blue": (28, 48, 140),      # dark blue
    "green": (44, 150, 76),
    "violet": (92, 52, 140),
}
W_FINE, W_BROAD, P_FINE, P_BROAD = 2.7, 5.0, 0.6, 1.2
# What the executor accepts as a press: brush work never runs the modelled tip further than this below first
# touch (brushwork.MAX_PRESS_M, "the guard's maximum": press 1.2 at the evening's press_mm 5.0). The number is
# brushwork's and is read from there where that module can be imported (guard_mm); it stands here so that the
# kit also works where the motion modules are not.
MAX_PRESS_MM = 6.0
LOAD_MM = 140.0                 # one load of paint, in millimetres of fine line (a band uses it WB / W times faster)
TINT = {"": 0.86, "mid": 0.58, "pale": 0.34}    # how much of the paper one pass covers, as the picture is DESIGNED
BLACK_GAIN = 1.08               # black covers almost fully

# ----------------------------------------------------------------------------- what the sheet showed
P_MODE = 0.9                    # a press below this paints in the line mode, from it on in the band mode
W_SIDE = 1.2                    # width of a fine line that runs SIDEWAYS (across the direction to the pan axis):
#                                 1 to 1.5 mm in the check pattern, against 2.5 to 3 mm (= W) along that direction
BAND_LOSS = 0.2                 # a band was 4 to 5 mm where the kit drew WB = 5: taken as WB on a fresh load and
#                                 this part of it narrower where the load is used up (the first stroke after a
#                                 load is the widest)
WET_RUN_MM = 25.0               # The first stroke after a load is "up to 5 mm wide even at the light press": a
#                                 LINE is taken to be as wide as a band is designed (WB) at the very start of a
#                                 load and to have its own width after this many fine millimetres. In the check
#                                 pattern the first stroke of a load (24 mm) was fat as a whole, the second was
#                                 not: a length that was seen, not a curve that was measured.
TAIL_MM = 2.0                   # the last millimetres of a stroke taper as the brush lifts ("a short tail")
FADE_END = 0.45                 # strength of the mark where a load is used up, relative to its start (in the
#                                 photos a pale red band went from 0.24 to 0.14 of the paper's light in 54 mm,
#                                 the first rib of the lantern from 0.67 to 0.32 in 71 mm)
# Tints as they were SEEN: tube -> tint -> (first load of a layer, every load after it), as a part of the full
# paint. Read off the photos of the evening as the part of the paper's light a mark takes in the colour channel
# the paint absorbs most (full red 0.79 to 0.84, full black 0.80 to 0.83):
# - red: pale is a salmon wash and stays one (0.18 to 0.31 on bare paper), mid is hardly darker (0.25 to 0.40);
# - blue: the first band of the pale roof took 0.08, the five after it 0.46 to 0.56: a dark navy;
# - black: the first pale band 0.63, the next two 0.77 and 0.80: an almost black line;
# - yellow: the pale plaque and shoulder took 0.55 to 0.75 where the pure paint takes 0.83.
# Why: the reload of a pale recipe is water once and a touch of the blob, and the touch takes too much of a wet
# blob. Tubes that are not listed were not painted that evening: their tints are shown as designed.
SEEN_TINTS = {
    "red": {"pale": (0.40, 0.40), "mid": (0.47, 0.47)},
    "blue": {"pale": (0.15, 0.90)},
    "black": {"pale": (0.75, 0.95)},
    "yellow": {"pale": (0.50, 0.80)},
}
# Where the marks land, (du, dv) in mm from where the plan puts them: "about (-6, -9), the same everywhere"
# (the evening's note); the flat brush of the morning: (-4.5, -8.5) (render.fit_to_photo). It is not the brush: it
# is the difference between where the camera model projects a commanded point and where the arm really paints.
# In which view: one that is warped onto the PAPER (paintskill.canvas_view(..., z=0.0), the plane the morning
# used). Read again for this update in such views of 174946_run.jpg (the check pattern, five places) and
# 190853_run.jpg (the picture: the pillars' lower halves, the two short beams, the yellow light): u between 0
# and -5 (middle -3), v between -7 and -12 (middle -10): the note's v, a smaller u. The number of the note is
# kept. In the view canvas_view makes by default with the evening's set-up (the plane at the contact height,
# 5.4 mm ABOVE the paper) the same marks sit within about 5 mm of the plan (u -4 .. +3, v -5 .. 0): the tool tip
# had been measured through the same camera, and the two errors nearly cancel there. One thing does not fit
# either number: the roof of the picture, straight in the plan, sags about 10 mm in the middle of the sheet's
# near edge. A set-up that has measured its own landing passes it to render (shift_mm).
LANDING_MM = (-6.0, -9.0)
# Writing blacks: the black tube, pressed out on the blotting strip after every load (no fat first stroke, less
# paint on the brush). name -> (tube, millimetres of fine line one load was run for, loaded with a water dip?).
WRITING = {"ink": ("black", 110.0, True), "ink2": ("black", 60.0, False)}
# Black written with a wet brush ("ink") over a wash of the same sitting turned grey: it kept 0.28 to 0.49 of the
# paper's light where it crossed the pale red ribs, against 0.83 on bare paper (photo 190142; on the strongest
# red, which holds less water, it stayed black). The same strokes retraced with "ink2" (no water) are solid.
DAMP_GREY = 0.45                # part of its cover a wet writing black keeps over an earlier wash

# ----------------------------------------------------------------------------- time
# Calls of `fd paint run` on the real arm that evening: (strokes, travel mm, loads, loads that were blotted,
# washes of the brush, seconds). 1: the check pattern (red); 2: the red glaze; 3: the red wash; 4: the written
# lines ("ink"); 5: the roof wash, the brush washed after the reds.
MEASURED_CALLS = (
    (20, 447.0, 5, 0, 0, 289.0),
    (14, 584.0, 11, 0, 0, 469.0),
    (16, 839.0, 15, 0, 0, 694.0),
    (19, 497.0, 6, 6, 0, 409.0),
    (6, 427.0, 6, 0, 1, 342.0),
)
# seconds = T_STROKE * strokes + travel / V_MM_S + T_LOAD * loads + T_BLOT * blotted loads + T_WASH * washes.
# V_MM_S is the stroke speed the executor commands (BrushOptions.v_stroke); T_STROKE and T_LOAD are the least
# squares fit (relative errors) to calls 1 to 3, T_BLOT follows from call 4, T_WASH from call 5 (fit_seconds
# does it again from the table: 4.04, 36.9, 14.3, 75.0). (Three free numbers for stroke, millimetre and load
# fit calls 1 to 3 exactly with a NEGATIVE time per stroke: the calls cannot tell a millimetre from a stroke.)
# The kit had 6 s per stroke, 22 s per load and 120 s per change of paint: 35 minutes for the Kaminarimon. A
# load is the expensive thing: the water stands at the far edge of the reach and the hops go at 20 deg/s.
T_STROKE, V_MM_S, T_LOAD, T_BLOT, T_WASH = 4.0, 20.0, 37.0, 14.0, 75.0
# How good that is. The five calls come out at -0.6, +4.9, -4.8, -0.5 and +0.1 %, but that is IN-SAMPLE: the fit
# is asked for the data it was made from, and for calls 4 and 5 the figure says nothing at all (T_BLOT and T_WASH
# each rest on that one call and cannot miss it). What an estimate is worth for work the fit has NOT seen:
# - fitted to four of the calls and asked for the fifth (leave_one_out): -20, +10, -10, -21 and -22 %;
# - the other calls of `fd paint run` of that evening, which the fit never used (UNSEEN_CALLS, from the run log
#   of the rig): ten of eleven within 12 % (-12 to +8), the lettering in dense black 26 % too LONG: a load of
#   "ink2" takes no water, so it has no trip to the dish at the far edge, and took about 24 s instead of 37.
# So the minutes of a new picture are good to about a quarter (TIME_ERROR; `stats` passes it on), until real
# runs say more: the executor keeps running means of its own (paintskill.note_run).
# One thing in the table is not as the evening was: call 3 is listed without a wash, but by the run log the
# brush carried yellow_pale when that layer began, so a wash of the brush is inside its 694 s (the fit books it
# under the loads). Counted as a wash the same rule gives 4.9 s per stroke, 34 s per load, 15 s per blotting
# and 78 s per wash, and 38.3 instead of 39.4 minutes for the picture. The table is left as it was listed for
# this update, the same as paintskill.EVENING_CALLS, so that the two estimates of the harness stay one.
TIME_ERROR = 0.25
# (name, strokes, travel mm, loads, loads that were blotted, washes of the brush, seconds) of the calls the fit
# did not use: results of `fd paint run` in the run log of 2026-11-13 (runs/20261113_110358/events.jsonl of that
# rig), 17:55 to 19:09. Blottings and washes are not in those results: they are what the recipes of the set-up
# do with the paint the brush carried before (a wash where the tube changes or a paler tint follows).
UNSEEN_CALLS = (
    ("17:55 tint test, pale red after red", 3, 92.0, 3, 0, 1, 208.9),
    ("17:56 tint test, mid red", 3, 89.0, 2, 0, 0, 84.7),
    ("17:59 tint test, pale black after red", 3, 89.0, 2, 0, 1, 168.7),
    ("18:01 tint test, black lines", 2, 60.0, 2, 0, 0, 78.8),
    ("18:05 a test character in ink", 13, 244.0, 3, 3, 0, 215.9),
    ("18:15 layer 0: the light, pale yellow after ink", 6, 285.0, 4, 0, 1, 255.6),
    ("18:17 one band of layer 0 again", 1, 36.0, 1, 0, 0, 44.0),
    ("18:39 layer 3: the rim", 4, 115.0, 2, 0, 0, 106.5),
    ("18:49 layer 5: the eave, pale black after pale blue", 3, 213.0, 3, 0, 1, 219.4),
    ("18:50 layer 6: black bands", 2, 75.0, 1, 0, 0, 55.2),
    ("19:08 the lettering again in ink2 (no water in its loads)", 13, 240.0, 5, 5, 0, 252.2),
)
# (The loads of calls 2 and 3 are those of the executor of that evening: it added up the travel it MEASURED,
# which was about 1 mm a band longer than the plan's (839 for 823 mm, 584 for 577, 427 for 418), and was told
# to reload every 75 mm, so two bands that fill a load to the brim on paper (37.6 + 37.6 mm) took two loads.
# The executor of this update counts a stroke by the plan's own length, as `loads` does: for those two layers
# it will need 13 and 9 loads where the evening needed 15 and 11.)


def seconds(strokes: int, travel_mm: float, loads: int, blotted: int = 0, washes: int = 0) -> float:
    """Time the robot needs for that much work (see MEASURED_CALLS for what it is fitted to, and the lines
    under T_STROKE for how far to trust it: about a quarter for work the fit has not seen)."""
    return (T_STROKE * strokes + float(travel_mm) / V_MM_S + T_LOAD * loads + T_BLOT * blotted + T_WASH * washes)


def fit_seconds(calls=MEASURED_CALLS, v_mm_s: float = V_MM_S) -> tuple:
    """(seconds per stroke, per load, per blotting of a load, per wash of the brush) from timed calls [(strokes,
    travel mm, loads, blotted loads, washes, seconds), ...], the way the constants of `seconds` were found:
    a millimetre costs what the stroke speed says (1 / v_mm_s); stroke and load by least squares of the
    RELATIVE errors over the calls that neither blot nor wash, neither of the two negative (a time cannot be:
    calls 2 and 3 alone are met exactly by -17 s per stroke); the blotting is what those two leave over in the
    calls that blot, the wash what is left over in the calls that wash; a thing no call has costs 0 here
    (nothing is known about it). The five calls of 2026-11-13 give (4.04, 36.9, 14.3, 75.0); `seconds` has
    them rounded. With leave_one_out this is how the fit is judged; it changes no constant."""
    rows = np.asarray(calls, dtype=float).reshape(-1, 6)
    v = float(v_mm_s)
    plain = rows[(rows[:, 3] == 0) & (rows[:, 4] == 0)]
    part = np.column_stack([plain[:, 0], plain[:, 2]]) / plain[:, 5:6]      # strokes and loads, per second of the call
    want = 1.0 - plain[:, 1] / v / plain[:, 5]                              # the part of the call they have to explain
    best = None
    for free in ((0, 1), (0,), (1,), ()):                                   # both free; else one of them at 0; else both
        x = np.zeros(2)
        if free and len(plain):
            x[list(free)] = np.linalg.lstsq(part[:, list(free)], want, rcond=None)[0]
        if np.any(x < -1e-9):
            continue
        x = np.maximum(x, 0.0)
        rss = float(np.sum((part @ x - want) ** 2))
        if best is None or rss < best[0] - 1e-12:
            best = (rss, x)
    t_stroke, t_load = float(best[1][0]), float(best[1][1])
    left = rows[:, 5] - (t_stroke * rows[:, 0] + rows[:, 1] / v + t_load * rows[:, 2])
    blots = (rows[:, 3] > 0) & (rows[:, 4] == 0)
    t_blot = max(0.0, float(np.mean(left[blots] / rows[blots, 3]))) if blots.any() else 0.0
    washes = rows[:, 4] > 0
    t_wash = max(0.0, float(np.mean((left[washes] - t_blot * rows[washes, 3]) / rows[washes, 4]))) if washes.any() else 0.0
    return t_stroke, t_load, t_blot, t_wash


def leave_one_out(calls=MEASURED_CALLS, v_mm_s: float = V_MM_S) -> list:
    """What the estimate is worth for a call the fit has not seen: for every timed call the constants are
    fitted to the OTHER calls (fit_seconds) and that call is estimated with them. -> [estimate / measured - 1,
    ...] in the order of the calls. The five calls of 2026-11-13: -0.20, +0.10, -0.10, -0.21, -0.22 (the last
    two are the only call that blots and the only one that washes: estimated without a time for either)."""
    calls = [tuple(float(v) for v in c) for c in calls]
    out = []
    for i, (n, mm, ld, blot, wash, t) in enumerate(calls):
        ts, tl, tb, tw = fit_seconds(calls[:i] + calls[i + 1:], v_mm_s)
        out.append((ts * n + mm / float(v_mm_s) + tl * ld + tb * blot + tw * wash) / t - 1.0)
    return out


# ----------------------------------------------------------------------------- the brush and its paints
def width_of(press: float, W: float = W_FINE, WB: float = W_BROAD) -> float:
    """Line width (mm) at a press AS A PICTURE IS DESIGNED: W at 0.6, WB at 1.2, straight between and beyond.
    (What the sheet shows: painted_width.)"""
    return float(W + (float(press) - P_FINE) / (P_BROAD - P_FINE) * (WB - W))


def mode_of(press: float) -> str:
    """"fine" (a line) or "broad" (a band): the two things the brush really does."""
    return "fine" if float(press) < P_MODE else "broad"


def painted_width(press: float, along: float, W: float = W_FINE, WB: float = W_BROAD, side: float = W_SIDE,
                  used_mm: float = WET_RUN_MM, load_mm: float = LOAD_MM, wet: bool = True) -> float:
    """Width (mm) of the mark the brush really leaves. along: |cos| of the angle between the stroke and the
    direction to the robot's pan axis (1 = towards or away from the robot, 0 = sideways). used_mm: fine
    millimetres painted since the brush was loaded (default: the wet start is just over); wet: the brush was
    not blotted after loading.
    A band is as wide whichever way it runs: WB on a fresh load, BAND_LOSS of it less where the load is used up.
    A line is `side` wide sideways and W along the robot's direction, and as wide as a band at the very start
    of a wet load."""
    if mode_of(press) == "broad":
        return float(WB) * (1.0 - BAND_LOSS * min(1.0, max(0.0, float(used_mm) / float(load_mm))))
    side = min(float(side), float(W))
    w = side + (float(W) - side) * min(1.0, max(0.0, float(along)))
    if wet:
        w += max(0.0, float(WB) - w) * min(1.0, max(0.0, 1.0 - float(used_mm) / WET_RUN_MM))
    return float(w)


def base_paint(name: str) -> str:
    return name.rsplit("_", 1)[0] if name.rsplit("_", 1)[-1] in ("mid", "pale") else name


def tint_of(name: str) -> str:
    return name.rsplit("_", 1)[-1] if name.rsplit("_", 1)[-1] in ("mid", "pale") else ""


def tube_of(name: str) -> str:
    """The tube a layer's paint comes from: "red" for red_pale, "black" for ink2."""
    return WRITING[name][0] if name in WRITING else base_paint(name)


def blotted(name: str) -> bool:
    """Is the brush pressed out on the blotting strip after every load of this paint?"""
    return name in WRITING


def load_mm_of(name: str, L: float = LOAD_MM) -> float:
    """Fine millimetres one load of this paint lasts (a blotted brush carries less)."""
    return min(float(L), WRITING[name][1]) if name in WRITING else float(L)


def _strength(name: str) -> int:
    return {"pale": 0, "mid": 1}.get(tint_of(name), 2 + (list(WRITING).index(name) + 1 if name in WRITING else 0))


def needs_wash(before: str | None, paint: str) -> bool:
    """Is the brush washed (rinse, blot, rinse, blot) before `paint` when it carries `before`? Not for the first
    paint of a clean brush and not from a tint to a stronger one of the same tube (pale -> mid -> full -> ink ->
    ink2: the set-up's recipes say `after_without_rinse`); for everything else it is."""
    if before is None or before == paint:
        return False
    return tube_of(before) != tube_of(paint) or _strength(paint) < _strength(before)


def designed_alpha(name: str) -> float:
    return TINT[tint_of(name)] * (BLACK_GAIN if tube_of(name) == "black" else 1.0)


def painted_alpha(name: str, k: int = 0) -> float:
    """How much of the paper a fresh pass of this paint covers on the k-th load of its layer (0 = the first), as it
    was seen on the sheet (SEEN_TINTS); a tint of a tube that was not painted that evening: as designed."""
    full = TINT[""] * (BLACK_GAIN if tube_of(name) == "black" else 1.0)
    tint = tint_of(name)
    if not tint:
        return full
    seen = SEEN_TINTS.get(tube_of(name))
    if seen is None:
        return designed_alpha(name)
    if tint not in seen:
        # (the mid tint of a tube whose pale one came out strong was not painted: not paler than that pale one)
        first, later = seen["pale"]
        return max(TINT[tint], full * (first if k == 0 else later))
    first, later = seen[tint]
    return full * (first if k == 0 else later)


def brush_of(plan: dict) -> dict:
    """The round brush of a plan, every number filled in: its "brush" block {"wide_mm", "broad_mm", "load_mm"}
    (the block of a painting set-up, which names the two widths under "modes", is read as well)."""
    b = plan.get("brush") or {}
    modes = b.get("modes") or {}
    W = float(b.get("wide_mm", (modes.get("fine") or {}).get("width_mm", W_FINE)))
    WB = float(b.get("broad_mm", (modes.get("broad") or {}).get("width_mm", W_BROAD)))
    return {"kind": "round", "wide_mm": W, "broad_mm": WB, "load_mm": float(b.get("load_mm", LOAD_MM)),
            "side_mm": min(W, float(b.get("side_mm", W_SIDE)))}


def canvas_of(plan: dict) -> tuple:
    w, h = plan.get("canvas_mm") or CANVAS
    return float(w), float(h)


# ----------------------------------------------------------------------------- geometry
def length_mm(pts) -> float:
    p = np.asarray(pts, dtype=float)
    return float(np.sum(np.linalg.norm(np.diff(p, axis=0), axis=1))) if len(p) > 1 else 0.0


def resample(pts, step: float = 1.0) -> np.ndarray:
    p = np.asarray(pts, dtype=float)
    if len(p) < 2:
        return p
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    n = max(2, int(math.ceil(s[-1] / step)) + 1)
    t = np.linspace(0.0, s[-1], n)
    return np.column_stack([np.interp(t, s, p[:, 0]), np.interp(t, s, p[:, 1])])


def split(pts, max_len: float) -> list:
    """Cut a polyline into pieces no longer than max_len (pieces meet end to start)."""
    p = resample(pts, 1.0)
    total = length_mm(p)
    if total <= max_len:
        return [np.asarray(pts, dtype=float).round(2).tolist()]
    n = int(math.ceil(total / max_len))
    seg = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    out = []
    for k in range(n):
        a, b = total * k / n, total * (k + 1) / n
        idx = np.nonzero((seg >= a - 1e-9) & (seg <= b + 1e-9))[0]
        out.append(simplify(p[idx[0]:idx[-1] + 1]))
    return out


def simplify(pts, tol: float = 0.15) -> list:
    p = np.asarray(pts, dtype=np.float32).reshape(-1, 1, 2)
    q = cv2.approxPolyDP(p, tol, False).reshape(-1, 2)
    return np.asarray(q, dtype=float).round(2).tolist()


def arc(centre, radius, a0_deg: float, a1_deg: float, step_deg: float = 6.0) -> list:
    """Points of an ellipse arc; radius = r or (ru, rv); angles in degrees, 0 = +u, 90 = +v (down)."""
    ru, rv = (radius, radius) if np.isscalar(radius) else radius
    n = max(2, int(abs(a1_deg - a0_deg) / step_deg) + 1)
    return [[centre[0] + ru * math.cos(math.radians(a)), centre[1] + rv * math.sin(math.radians(a))]
            for a in np.linspace(a0_deg, a1_deg, n)]


def hatch_mask(mask, ppm: float, W: float, angle_deg: float = 90.0, pitch: float | None = None,
               inset: float | None = None, min_len: float = 3.0) -> list:
    """Parallel brush lines that cover a shape (mask: uint8 image at ppm px/mm, non-zero = paint here), the brush
    kept inside: the centre lines stay `inset` (default W/2) inside the outline. `pitch` is the distance between
    neighbouring lines (default 0.8 W: they overlap a little). Lines run at angle_deg (0 = along u, 90 = along v)
    and are ordered boustrophedon. -> list of polylines (each a straight piece inside the shape)."""
    pitch = 0.8 * W if pitch is None else float(pitch)
    inset = 0.5 * W if inset is None else float(inset)
    k = max(1, int(round(inset * ppm)))
    inner = cv2.erode((mask > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    ys, xs = np.nonzero(inner)
    if not len(xs):
        return []
    d = np.array([math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg))])
    n = np.array([-d[1], d[0]])
    pts = np.column_stack([xs, ys]) / ppm
    along, across = pts @ d, pts @ n
    lo, hi = across.min(), across.max()
    count = max(1, int(math.ceil((hi - lo) / pitch)) + 1)
    levels = np.linspace(lo, hi, count) if count > 1 else np.array([(lo + hi) / 2])
    out = []
    for i, c in enumerate(levels):
        sel = np.abs(across - c) <= 0.6 / ppm
        if not sel.any():
            continue
        a = np.sort(along[sel])
        runs, start, prev = [], a[0], a[0]
        for v in a[1:]:
            if v - prev > 2.5 / ppm:
                runs.append((start, prev))
                start = v
            prev = v
        runs.append((start, prev))
        row = [[(d * a0 + n * c).round(2).tolist(), (d * a1 + n * c).round(2).tolist()] for a0, a1 in runs if a1 - a0 >= min_len]
        if i % 2:
            row = [r[::-1] for r in row[::-1]]
        out += row
    return out


# ----------------------------------------------------------------------------- the kit: a plan made of brush lines
class Layer:
    def __init__(self, plan, name: str, paint: str):
        self.plan, self.name, self.paint, self.strokes = plan, name, paint, []

    def line(self, pts, press: float = P_FINE):
        """One brush line along a polyline: fine at press 0.6 (default), a broad band at 1.2. Cut into pieces of
        at most one load (a band uses the paint faster than a fine line, a blotted brush carries less)."""
        w = width_of(press, self.plan.W, self.plan.WB)
        for piece in split(pts, load_mm_of(self.paint, self.plan.L) * self.plan.W / w * 0.95):
            if length_mm(piece) >= 0.5:
                self.strokes.append({"pts": piece, "press": float(press)})
        return self

    def band(self, pts):
        return self.line(pts, P_BROAD)

    def dot(self, uv, press: float = P_FINE, length: float = 1.5):
        """A dab: a very short stroke (the brush touches down and drags 1.5 mm)."""
        u, v = uv
        self.strokes.append({"pts": [[round(u, 2), round(v - length / 2, 2)], [round(u, 2), round(v + length / 2, 2)]],
                             "press": float(press)})
        return self

    def _mask(self, draw, ppm: float):
        w, h = self.plan.canvas
        m = np.zeros((int(round(h * ppm)), int(round(w * ppm))), np.uint8)
        draw(m, ppm)
        return m

    def fill_mask(self, mask, angle_deg: float = 90.0, pitch=None, inset=None, press: float = P_BROAD, outline: bool = False):
        """A wash: parallel strokes over a shape (mask of the canvas at any px/mm), broad bands by default, the
        brush kept inside the shape. pitch = distance between neighbouring strokes (default 0.8 * their width: an
        even wash; 1.3 .. 1.6 * width: broken, paper sparkles between the bands)."""
        ppm = mask.shape[1] / self.plan.canvas[0]
        w = width_of(press, self.plan.W, self.plan.WB)
        if outline:
            k = max(1, int(round(0.5 * w * ppm)))
            inner = cv2.erode((mask > 0).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
            cs, _ = cv2.findContours(inner, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            for c in cs:
                if len(c) >= 8:
                    ring = c.reshape(-1, 2) / ppm
                    self.line(simplify(np.vstack([ring, ring[:1]]), 0.25), press)
        for ln in hatch_mask(mask, ppm, w, angle_deg, pitch, inset):
            self.line(ln, press)
        return self

    def fill_polygon(self, poly, angle_deg: float = 90.0, holes=(), ppm: float = 8.0, **kw):
        def draw(m, s):
            cv2.fillPoly(m, [np.round(np.asarray(poly, dtype=float) * s).astype(np.int32)], 255)
            for h in holes:
                cv2.fillPoly(m, [np.round(np.asarray(h, dtype=float) * s).astype(np.int32)], 0)
        return self.fill_mask(self._mask(draw, ppm), angle_deg, **kw)

    def fill_ellipse(self, centre, radii, angle_deg: float = 90.0, holes=(), ppm: float = 8.0, **kw):
        def draw(m, s):
            cv2.ellipse(m, (int(round(centre[0] * s)), int(round(centre[1] * s))),
                        (int(round(radii[0] * s)), int(round(radii[1] * s))), 0, 0, 360, 255, -1)
            for h in holes:
                cv2.fillPoly(m, [np.round(np.asarray(h, dtype=float) * s).astype(np.int32)], 0)
        return self.fill_mask(self._mask(draw, ppm), angle_deg, **kw)


class Plan:
    def __init__(self, W: float = W_FINE, L: float = LOAD_MM, paints: dict | None = None, WB: float = W_BROAD,
                 canvas_mm=CANVAS, pan_axis_mm=PAN_AXIS):
        self.W, self.WB, self.L, self.layers = float(W), float(WB), float(L), []
        self.paints = dict(PAINTS if paints is None else paints)
        self.canvas = (float(canvas_mm[0]), float(canvas_mm[1]))
        self.pan_axis = (float(pan_axis_mm[0]), float(pan_axis_mm[1]))

    def layer(self, name: str, paint: str) -> Layer:
        if tube_of(paint) not in self.paints:
            raise ValueError(f"unknown paint {paint!r}: one of {sorted(self.paints)}, optionally with _mid or _pale, "
                             f"or a writing black ({', '.join(WRITING)})")
        ly = Layer(self, name, paint)
        self.layers.append(ly)
        return ly

    def to_dict(self) -> dict:
        used = {ly.paint for ly in self.layers}
        num = lambda v: int(v) if float(v).is_integer() else float(v)      # noqa: E731  (224, not 224.0, as the kit wrote it)
        return {"canvas_mm": [num(self.canvas[0]), num(self.canvas[1])], "pan_axis_mm": list(self.pan_axis),
                "brush": {"kind": "round", "wide_mm": self.W, "broad_mm": self.WB, "load_mm": self.L},
                "paints": {k: list(self.paints[tube_of(k)]) for k in sorted(used)},
                "layers": [{"name": ly.name, "paint": ly.paint, "strokes": ly.strokes} for ly in self.layers if ly.strokes]}

    def save(self, path: str) -> str:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)
        return path

    def render(self, ppm: float = 4.0, W: float | None = None, upto_layer: int | None = None, WB: float | None = None,
               **kw) -> np.ndarray:
        return render(self.to_dict(), ppm, self.W if W is None else W, upto_layer, WB=self.WB if WB is None else WB, **kw)

    def stats(self, setup: dict | None = None) -> dict:
        return stats(self.to_dict(), self.L, setup)


# ----------------------------------------------------------------------------- loads
def paint_use(stroke: dict, W: float = W_FINE, WB: float = W_BROAD) -> float:
    """Paint a stroke takes, in millimetres of fine line: its length times width_of(press) / W (a broad
    millimetre counts WB / W fine ones). The executor counts the same (paintskill.use_mm)."""
    return length_mm(stroke["pts"]) * width_of(float(stroke.get("press", P_FINE)), W, WB) / W


def loads(plan: dict, L: float | None = None) -> list:
    """How the strokes fall on loads of paint: per layer a list with one (k, a, b) per stroke: k = which load of
    the layer the stroke is painted with (0 = the first; every layer starts with one), a and b = how much of that
    load is gone before and after the stroke (0 .. 1). The brush is loaded before a stroke when what was used
    since the last load, plus this stroke, would pass one load (load_mm_of the layer's paint). This is the one
    place where loads are counted here: stats (number, time) and render (where a mark is fresh and where it has
    faded) use it. The executor has the same rule (paintskill.reload_rule) and counts the same for every paint
    that is not blotted. A writing black's shorter load it knows only from the set-up's recipe ("load_mm": 110
    for "ink", 60 for "ink2") or from the command (`fd paint run ... --reload-mm 110`: plain travel, which for
    fine lines is the same number). The set-up of 2026-11-13 has no "load_mm": without either, the executor
    reloads a writing black every 140 mm like any paint, and the letters get fewer loads than are counted here."""
    br = brush_of(plan)
    W, WB = br["wide_mm"], br["broad_mm"]
    L = br["load_mm"] if L is None else float(L)
    out = []
    for layer in plan["layers"]:
        cap = load_mm_of(layer["paint"], L)
        rows, k, since = [], -1, float("inf")
        for st in layer["strokes"]:
            use = paint_use(st, W, WB)
            if since + use > cap:
                k, since = k + 1, 0.0
            rows.append((k, since / cap, (since + use) / cap))
            since += use
        out.append(rows)
    return out


# ----------------------------------------------------------------------------- what the sheet will look like
def painted_stroke(st: dict, br: dict, pan, a: float = 0.0, b: float = 0.0, alpha: float = 1.0, wet: bool = True,
                   step: float = 0.5, cap: float | None = None):
    """One stroke as the brush really leaves it: points every `step` mm along it with the width (mm) and the
    cover (0 .. 1) there. a, b: part of the load that is gone before and after the stroke (`loads`); alpha: cover
    of a fresh pass (painted_alpha); wet: the brush was not blotted after loading (a fresh line starts as wide
    as a band); cap: fine millimetres one load of this paint lasts (default: the brush's).
    -> (points n x 2, widths n, covers n)."""
    W, WB = br["wide_mm"], br["broad_mm"]
    press = float(st.get("press", P_FINE))
    p = resample(st["pts"], step)
    n = len(p)
    d = np.gradient(p, axis=0) if n > 1 else np.zeros((n, 2))
    d /= np.linalg.norm(d, axis=1)[:, None] + 1e-9
    rad = p - np.asarray(pan, dtype=float)
    rad /= np.linalg.norm(rad, axis=1)[:, None] + 1e-9
    along = np.abs(np.sum(d * rad, axis=1))
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]) if n > 1 else np.zeros(n)
    used = a + (b - a) * (s / max(float(s[-1]), 1e-9))                # part of the load that is gone at each point
    cap = float(br["load_mm"]) if cap is None else float(cap)
    w = np.array([painted_width(press, c, W, WB, br.get("side_mm", W_SIDE), u * cap, cap, wet) for c, u in zip(along, used)])
    w = w * (0.55 + 0.45 * np.clip((s[-1] - s) / TAIL_MM, 0.0, 1.0))
    cover = alpha * (1.0 - (1.0 - FADE_END) * np.clip(used, 0.0, 1.0))
    return p, w, cover


def render(plan: dict, ppm: float = 4.0, W: float | None = None, upto_layer: int | None = None, seed: int = 7,
           WB: float | None = None, as_painted: bool = True, margin_mm: float = 0.0, shift_mm=None,
           return_coverage: bool = False):
    """What the sheet will look like (BGR image, ppm px per mm), as watercolour: every layer is a translucent
    glaze over what is below (subtractive: red over yellow turns orange, a second pass is darker), edges are soft
    with a darker rim where the paint pooled, the tone wavers a little, the paper has grain.

    as_painted=True (the default: what the harness shows before it paints) is what the brush did on the sheet of
    2026-11-13: a fine line is W_SIDE wide where it runs sideways and W where it runs along the direction to the
    robot's pan axis; a band is WB on a fresh load and narrower as the load runs out; the first stroke after a
    load is the darkest and the widest and the mark fades along the load (the loads are the ones `stats` counts:
    `loads`); a blotted writing black starts thin, and grey where a wet one crosses a wash; tints are as strong
    as they were seen (painted_alpha: a pale blue or black is nearly full strength from its second load on).
    as_painted=False is the picture as it was designed, as the kit of the evening drew it: every line
    width_of(press) wide in every direction with a slightly fatter start, every tint as pale as its name.

    The image is the canvas, in the plan's own frame: the point (u, v) of the plan is at pixel (u * ppm,
    v * ppm). margin_mm > 0 gives the SHEET instead: that much paper on every side of the canvas (12: the frame
    of paintskill.canvas_view, so that preview and photo can be laid side by side), four small corner marks
    where the canvas is, and, as painted, every mark where it LANDS: LANDING_MM from where the plan puts it, so
    that what leaves the canvas is seen. shift_mm=(du, dv) moves the marks by that instead, with or without a
    margin (a set-up that has measured its own landing passes it; (0, 0): where the plan puts them).
    return_coverage: also {paint: float32 map of how much of that paint lies where}."""
    br = brush_of(plan)
    if W is not None:
        br["wide_mm"] = float(W)
        br["side_mm"] = min(br["side_mm"], br["wide_mm"])
    if WB is not None:
        br["broad_mm"] = float(WB)
    Wf, Wb = br["wide_mm"], br["broad_mm"]
    w_mm, h_mm = canvas_of(plan)
    m_mm = max(0.0, float(margin_mm))
    hh, ww = int(round((h_mm + 2.0 * m_mm) * ppm)), int(round((w_mm + 2.0 * m_mm) * ppm))
    pan = plan.get("pan_axis_mm") or PAN_AXIS
    if not as_painted:
        shift = np.zeros(2)                                          # the designed picture is where the plan puts it
    elif shift_mm is None:
        shift = np.array(LANDING_MM if m_mm > 0.0 else (0.0, 0.0), dtype=float)
    else:
        shift = np.array(shift_mm, dtype=float)
    rng = np.random.default_rng(seed)
    grain = cv2.GaussianBlur(rng.normal(0.0, 1.0, (hh, ww)).astype(np.float32), (0, 0), 0.6)
    cloud = cv2.resize(rng.normal(0.0, 1.0, (hh // 24 + 2, ww // 24 + 2)).astype(np.float32), (ww, hh), interpolation=cv2.INTER_CUBIC)
    img = np.empty((hh, ww, 3), np.float32)
    img[:] = np.array(PAPER[::-1], np.float32) / 255.0
    img *= (1.0 + 0.012 * grain)[..., None]
    layers = plan["layers"] if upto_layer is None else plan["layers"][:upto_layer + 1]
    on_load = loads({**plan, "brush": br})
    cover = {}
    damp = np.zeros((hh, ww), np.float32)
    F = 8                                                            # fixed point of the drawing calls (1/8 px)
    for li, layer in enumerate(layers):
        name = layer["paint"]
        col = (plan.get("paints") or {}).get(name, PAINTS.get(tube_of(name), (80, 80, 80)))
        col = col.get("rgb", (80, 80, 80)) if isinstance(col, dict) else col
        rgb = np.array(col, np.float32) / 255.0
        once = np.zeros((hh, ww), np.float32)
        twice = np.zeros((hh, ww), np.float32)
        for st, (k, a, b) in zip(layer["strokes"], on_load[li]):
            if len(st["pts"]) < 2:
                continue
            m = np.zeros((hh, ww), np.uint8)
            if as_painted:
                p, w, cv = painted_stroke(st, br, pan, a, b, painted_alpha(name, k), wet=not blotted(name),
                                          cap=load_mm_of(name, br["load_mm"]))
                p = (p + shift + m_mm) * ppm
                reach = int(math.ceil(0.5 * float(w.max()) * ppm)) + 3
                # (weakest first: where two stamps of a stroke overlap the stronger one stays)
                for i in np.argsort(cv, kind="stable"):
                    cv2.circle(m, (int(round(p[i, 0] * F)), int(round(p[i, 1] * F))), max(1, int(round(0.5 * w[i] * ppm * F))),
                               int(round(255.0 * min(1.0, float(cv[i])))), -1, cv2.LINE_AA, 3)
                scale = 1.0 / 255.0
            else:
                w0 = width_of(float(st.get("press", P_FINE)), Wf, Wb)
                p = (np.asarray(st["pts"], dtype=float) + m_mm) * ppm
                q = np.round(p * F).astype(np.int32)
                cv2.polylines(m, [q.reshape(-1, 1, 2)], False, 255, max(1, int(round(w0 * ppm))), cv2.LINE_AA, 3)
                cv2.circle(m, (int(q[0][0]), int(q[0][1])), int(round(0.60 * w0 * ppm * F)), 255, -1, cv2.LINE_AA, 3)
                cv2.circle(m, (int(q[-1][0]), int(q[-1][1])), int(round(0.46 * w0 * ppm * F)), 255, -1, cv2.LINE_AA, 3)
                reach = int(math.ceil(0.6 * w0 * ppm)) + 3
                scale = designed_alpha(name) / 255.0
            x0, y0 = max(0, int(math.floor(p[:, 0].min())) - reach), max(0, int(math.floor(p[:, 1].min())) - reach)
            x1, y1 = min(ww, int(math.ceil(p[:, 0].max())) + reach + 1), min(hh, int(math.ceil(p[:, 1].max())) + reach + 1)
            if x1 <= x0 or y1 <= y0:
                continue                                             # the mark lands outside the picture
            box = (slice(y0, y1), slice(x0, x1))
            mf = m[box].astype(np.float32) * scale
            twice[box] = np.maximum(twice[box], np.minimum(once[box], mf))
            once[box] = np.maximum(once[box], mf)
        soft = cv2.GaussianBlur(once, (0, 0), 0.30 * ppm)
        rim = np.clip(once - cv2.GaussianBlur(once, (0, 0), 0.9 * ppm), 0.0, 1.0)
        dens = soft * (1.0 + 0.10 * cloud + 0.04 * grain) + 0.30 * rim + 0.22 * cv2.GaussianBlur(twice, (0, 0), 0.4 * ppm)
        if as_painted and name in WRITING and WRITING[name][2]:
            dens = dens * (1.0 - (1.0 - DAMP_GREY) * np.clip(damp / 0.15, 0.0, 1.0))
        dens = np.clip(dens, 0.0, 0.97)
        if not blotted(name):
            damp = np.maximum(damp, dens)                           # a wash: what a later wet black runs into
        if return_coverage:
            cover[name] = cover.get(name, 0.0) + dens
        img = img * (1.0 - dens[..., None] * (1.0 - rgb[::-1]))
    out = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    if as_painted and m_mm >= 2.0:
        # where the canvas is on the sheet: a small pencil mark at each corner (the marks of the brush land beside it)
        t = max(2, int(round(3.0 * ppm)))
        for cx, cy, sx, sy in ((m_mm, m_mm, 1, 1), (m_mm + w_mm, m_mm, -1, 1), (m_mm, m_mm + h_mm, 1, -1), (m_mm + w_mm, m_mm + h_mm, -1, -1)):
            x, y = int(round(cx * ppm)), int(round(cy * ppm))
            cv2.line(out, (x, y), (x + sx * t, y), (150, 150, 150), 1, cv2.LINE_AA)
            cv2.line(out, (x, y), (x, y + sy * t), (150, 150, 150), 1, cv2.LINE_AA)
    return (out, cover) if return_coverage else out


# ----------------------------------------------------------------------------- the numbers
SHORT_MM = 1.0                  # the executor refuses a stroke that is shorter than this (paintskill.plan_problems)


def guard_mm() -> float:
    """Millimetres below first touch that brush work allows a stroke: brushwork.MAX_PRESS_M where that module
    can be imported (the executor's own number), MAX_PRESS_MM where the kit is used without the motion modules."""
    try:
        import brushwork
        return float(brushwork.MAX_PRESS_M) * 1000.0
    except Exception:           # (not on this computer, or not on the path: the kit alone)
        return MAX_PRESS_MM


def _press_mm(setup: dict) -> float:
    """The millimetres a press of 1.0 stands for in a painting set-up (2.0 when it does not say: paintskill.press_m)."""
    v = float(setup.get("press_mm", 2.0))
    if not (math.isfinite(v) and v > 0.0):
        raise ValueError(f"the set-up's press_mm must be a positive number of millimetres (got {setup.get('press_mm')!r})")
    return v


def press_limit(setup: dict | None = None) -> float:
    """The largest press a stroke of a plan may have. With a painting set-up (paintskill.load_setup()): what
    the executor accepts with it, brush work's millimetres below first touch (guard_mm, 6) over the set-up's
    press_mm. Without one: P_BROAD, the kit's own cap. The band mode is the deepest press a plan was ever run
    with, and at the evening's press_mm 5.0 it IS brush work's limit."""
    return P_BROAD if setup is None else guard_mm() / _press_mm(setup)


def press_problem(press: float, setup: dict | None = None) -> str | None:
    """Why the executor would refuse a stroke of this press, or None when it would not: press_limit, by the
    comparison paintskill.press_m makes. (With a set-up the rule is not the round brush's alone:
    render.plan_stats asks it for a flat plan too.)"""
    press = float(press)
    if setup is None:
        if math.isfinite(press) and 0.0 <= press <= P_BROAD + 1e-9:
            return None
        return (f"press {press:g} is outside 0 .. {P_BROAD:g}, the band mode and the kit's own cap (with the set-up of "
                f"2026-11-13, press_mm 5.0, the executor refuses more: brush work allows {MAX_PRESS_MM:g} mm below "
                "first touch)")
    mm, guard = _press_mm(setup), guard_mm()
    if math.isfinite(press) and 0.0 <= press * mm <= guard + 1e-6:
        return None
    return (f"press {press:g} x {mm:g} mm is outside 0 .. {guard:g} mm below first touch, the limit of brush work: the "
            f"executor refuses the plan (at most press {guard / mm:.3g} with this set-up)")


def stats(plan: dict, L: float | None = None, setup: dict | None = None) -> dict:
    """Counts, travel, loads and the time the robot will need, and "problems": what the executor would refuse
    of the plan, as far as the strokes themselves tell it:
    - a stroke whose points are not pairs of numbers, one with fewer than 2 points, or shorter than SHORT_MM;
    - one that takes more than one load (`loads`);
    - a press beyond what brush work allows (press_problem). setup: the painting set-up the plan is meant
      for (paintskill.load_setup()): its press_mm then gives the limit, as the executor computes it; without
      one the kit's own cap P_BROAD stands in, which is the same number at the set-up of 2026-11-13;
    - a point nearer than EDGE to the canvas edge. This one is the kit's own rule and stricter than the
      executor, which refuses only what leaves the canvas: the marks land up to 9 mm from the plan
      (LANDING_MM) and a band is 5 mm wide.
    What only the set-up and the arm know is not judged here: another canvas or pan axis, another kind of
    brush or a brush block with other widths or another load than the plan's (the executor counts with the
    set-up's), a paint the set-up does not have, a recipe with a shorter load, the reach, the contact plane.
    paintskill.plan_problems(plan, setup) is the executor's own word on the plan; `fd paint preview PLAN`
    reports both and `fd paint run PLAN --layer N --dry` the rest.

    Loads as `loads` counts them; minutes from `seconds`: a load per count, blotting for the writing blacks,
    a wash where the tube changes or a paler tint follows a stronger one (`needs_wash`). "minutes_off_percent"
    is how far to trust them: `seconds` is fitted to five calls of one evening and good to about a quarter for
    work it has not seen (TIME_ERROR). It is the robot's time in its calls: on 2026-11-13 the clock said 50
    minutes for the Kaminarimon because every layer was a call, with a photo to look at and twice a contact
    height to change between two of them."""
    if isinstance(L, dict) and setup is None:       # stats(plan, setup): the set-up in the place of the load length
        L, setup = None, L
    br = brush_of(plan)
    W, WB = br["wide_mm"], br["broad_mm"]
    L = br["load_mm"] if L is None else float(L)
    w_mm, h_mm = canvas_of(plan)
    on_load = loads(plan, L)
    area = 0.0
    problems, per_layer = [], []
    total_s, strokes, travel, n_loads, n_wash, before = 0.0, 0, 0.0, 0, 0, None
    for li, layer in enumerate(plan["layers"]):
        name = layer["paint"]
        cap = load_mm_of(name, L)
        n, mm = 0, 0.0
        for si, st in enumerate(layer["strokes"]):
            p = np.asarray(st["pts"], dtype=float)
            if p.ndim != 2 or p.shape[1] != 2 or not np.all(np.isfinite(p)):
                # ([[u, v, z], ...] read as pairs would be other points: the executor calls it "not a stroke")
                problems.append(f"layer {li} stroke {si}: its points are not pairs [u, v] of canvas millimetres")
                continue
            ln = length_mm(p)
            press = float(st.get("press", P_FINE))
            area += ln * width_of(press, W, WB) if math.isfinite(press) and mode_of(press) == "broad" else 0.0
            if len(p) < 2:
                problems.append(f"layer {li} stroke {si}: fewer than 2 points")
                continue
            if ln < SHORT_MM:
                problems.append(f"layer {li} stroke {si}: {ln:.2f} mm is shorter than {SHORT_MM:g} mm")
            if paint_use(st, W, WB) > cap:
                problems.append(f"layer {li} stroke {si}: {ln:.0f} mm at this press is more than one load "
                                f"({cap * W / width_of(press, W, WB):.0f} mm of {name})")
            why = press_problem(press, setup)
            if why:
                problems.append(f"layer {li} stroke {si}: {why}")
            if p[:, 0].min() < EDGE or p[:, 0].max() > w_mm - EDGE or p[:, 1].min() < EDGE or p[:, 1].max() > h_mm - EDGE:
                problems.append(f"layer {li} stroke {si}: a point is nearer than {EDGE:.0f} mm to the canvas edge")
            n, mm = n + 1, mm + ln
        ld = (on_load[li][-1][0] + 1) if on_load[li] else 0
        wash = bool(layer["strokes"]) and needs_wash(before, name)
        if layer["strokes"]:
            before = name
        sec = seconds(n, mm, ld, ld if blotted(name) else 0, 1 if wash else 0)
        per_layer.append({"name": layer["name"], "paint": name, "strokes": n, "travel_mm": round(mm), "loads": ld,
                          "wash": wash, "seconds": round(sec)})
        strokes, travel, n_loads, n_wash, total_s = strokes + n, travel + mm, n_loads + ld, n_wash + int(wash), total_s + sec
    return {"strokes": strokes, "travel_mm": round(travel), "loads": n_loads, "washes": n_wash,
            "minutes": round(total_s / 60.0, 1), "minutes_off_percent": round(100.0 * TIME_ERROR),
            "broad_cm2": round(area / 100.0, 1),
            "layers": per_layer, "problems": problems[:20], "n_problems": len(problems)}


if __name__ == "__main__":
    import sys
    plan = json.load(open(sys.argv[1]))
    cv2.imwrite(sys.argv[2] if len(sys.argv) > 2 else "preview.png", render(plan))
    print(json.dumps(stats(plan), indent=1, ensure_ascii=False))
