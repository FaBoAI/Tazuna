"""Akabeko (赤べこ), the red papier-mache cow of Aizu, for the flat brush, in side view, facing right.

What makes it an Akabeko (after the photo in the Japanese Wikipedia article, Akabeko0926.jpg): a red loaf of a
body on stubby legs, a thick BLACK COLLAR where the nodding head sits, white brush stripes on the back behind
the collar, black spots with white rings on the body, a red head with a blunt snout, black ears on top, a small
white eye with a black pupil, a white crescent along the cheek, black nostrils on the RED snout (a white muzzle
makes it a pig). White is unpainted paper; the colours are red, then black.

    python akabeko.py [out_dir]      writes akabeko_plan.json and akabeko_preview.png

The picture is made of shapes (shapes.Canvas) that are turned into strokes: once round every edge
(shapes.contour_strokes), then radial zig-zags (shapes.fill_strokes). Canvas and pan axis are the values of the
set-up of 2026-11-13 (224 x 153 mm, pan axis at [131, -107.2]); `build(size_mm, pan)` takes others.

This is the design as it was painted on 2026-11-13 (red 13:56 to 14:11, black to 14:22), kept as it was: the
strokes of build() are those of the plan the executor ran. The plan now says which brush it is for (the flat
brush as it was run that morning, render.FLAT_BRUSH), so that the preview shows the 12.5 mm band the sheet
showed and a set-up with another brush refuses it. What the sheet did to the white of this design is what
`report()` (shapes.design_report) now says beforehand: the eye, the crescent on the cheek and the two stripes on
the back close or become slivers, the 26 mm ring on the rump survives at about 23 x 16 mm. A next Akabeko
needs those features larger; this one is the record.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import tazuna_paths  # noqa: E402,F401

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import render as RD  # noqa: E402
import shapes as SH  # noqa: E402

CANVAS_MM = (224.0, 153.0)
PAN_AXIS_MM = (131.0, -107.2)
PAINTS = {"red": {"rgb": [214, 38, 44], "opacity": 0.9}, "black": {"rgb": [28, 28, 34], "opacity": 0.95}}

# The cow faces RIGHT: the head, where the detail is, then lies under the robot's pan axis (u = 131), where the
# brush's rays run nearly straight down the picture; on the left they lean by 30 deg, which only the big body
# has to put up with. Every white (unpainted) feature is at least 6 mm wide and, where it is not along a ray,
# has 10 mm or more of red around it: the brush is 9 mm wide (a first design with a 70 mm head and 9 mm margins
# came out in fragments). Canvas mm: u right, v down.
BODY = (8, 40, 128, 104, 22)             # u0, v0, u1, v1, corner radius
LEGS = ((16, 94, 44, 121), (90, 94, 118, 121))
RUMP_SPOT = ((38, 66), 13.0)             # white ring (unpainted disc) with a black centre
FLANK_SPOT = ((84, 87), 9.0)
STRIPES = (((114, 36), (106, 64)), ((97, 36), (89, 64)))   # white brush stripes on the back, along the rays
STRIPE_MM = 7.5
COLLAR = ((134.0, 34), (132.5, 70), (134.0, 108))          # black, in the gap between body (u <= 128) and head
HEAD = ((139, 46), (146, 38), (170, 36), (196, 44), (212, 54), (217, 70), (214, 88), (202, 99), (160, 103), (142, 98), (139, 80))
HEAD_ROUND_MM = 7.0
EARS = (((150, 39), (148, 24)), ((167, 37), (168, 22)))    # black, on top of the head
EYE = ((181, 56), (8.5, 6.5))            # white oval, black pupil
CHEEK = ((139, 80), (160, 87), (184, 86), (203, 79))       # white crescent, from the collar towards the mouth
CHEEK_MM = 7.0
NOSTRILS = ((207, 63), (208, 78))
EDGE_SPILL_MM = 1.0                      # the brush may stick out of a red shape's edge by this much


def red_mask(size_mm=CANVAS_MM, ppm: float = 4.0) -> SH.Canvas:
    c = SH.Canvas(size_mm, ppm)
    c.rounded_box(*BODY)
    for u0, v0, u1, v1 in LEGS:
        c.rounded_box(u0, v0, u1, v1, 6)
    head = SH.Canvas(size_mm, ppm)
    head.polygon(HEAD)
    k = int(round(HEAD_ROUND_MM * 2 * ppm)) | 1          # round the corners of the head
    head.mask = cv2.morphologyEx(head.mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    c.mask |= head.mask
    for a, b in STRIPES:
        c.band((a, b), STRIPE_MM, on=False)
    c.circle(RUMP_SPOT[0], RUMP_SPOT[1], on=False)
    c.circle(FLANK_SPOT[0], FLANK_SPOT[1], on=False)
    c.ellipse(EYE[0], EYE[1], on=False)
    c.band(CHEEK, CHEEK_MM, on=False)
    return c


def build(size_mm=CANVAS_MM, pan=PAN_AXIS_MM) -> dict:
    ppm = 4.0
    mask = red_mask(size_mm, ppm).mask
    red = SH.contour_strokes(mask, pan, ppm, margin_mm=0.0) + SH.fill_strokes(mask, pan, ppm, margin_mm=-EDGE_SPILL_MM)
    # Black. On the real paper (2026-11-13) the red band was 12.5 mm where the model had 9: the paint reached
    # 1.75 mm further on every side (render.BRUSH_SPREAD_MM), so the white rings came out smaller than drawn. The
    # 26 mm disc on the rump was left at about 23 x 16 mm, half its area (measured in the photo of the red
    # layer; report() says so beforehand now), and the eye closed almost completely. The black marks are kept
    # small for that reason: their sizes were chosen that morning from a first look at the wet sheet, on which
    # the white looked smaller still, and they are what was painted. Order: what lies on white paper or on the
    # red that has dried longest first (collar in the gap, the spots), the marks on the head, which was painted
    # last, at the end.
    black = [SH.polyline(COLLAR, press=1.1)]
    (ru, rv), _r = RUMP_SPOT
    black.append(SH.radial_dab(ru, rv - 2.0, pan, 4.0, press=0.6))
    black.append(SH.radial_dab(FLANK_SPOT[0][0], FLANK_SPOT[0][1] - 1.0, pan, 2.0, press=0.5))
    for a, b in EARS:
        black.append(SH.polyline([a, b], press=0.9))
    for n in NOSTRILS:
        black.append(SH.radial_dab(n[0], n[1] - 1.0, pan, 2.0, press=0.5))
    black.append(SH.radial_dab(EYE[0][0], EYE[0][1] - 1.0, pan, 2.0, press=0.5))
    return {"name": "akabeko", "canvas_mm": list(size_mm), "pan_axis_mm": list(pan), "brush": dict(RD.FLAT_BRUSH),
            "paints": PAINTS,
            "layers": [{"name": "red", "paint": "red", "strokes": red}, {"name": "black", "paint": "black", "strokes": black}]}


def report(size_mm=CANVAS_MM, pan=PAN_AXIS_MM, ppm: float = 4.0) -> dict:
    """What will be left of the white of the red shape (shapes.design_report): the ring on the rump, the spot on
    the flank, the eye, the crescent on the cheek, the stripes on the back, the gap for the collar."""
    return SH.design_report(red_mask(size_mm, ppm).mask, pan, ppm)


def main(out_dir: str = ".") -> dict:
    size, pan = CANVAS_MM, PAN_AXIS_MM
    plan = build(size, pan)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "akabeko_plan.json"), "w") as f:
        json.dump(plan, f)
    cv2.imwrite(os.path.join(out_dir, "akabeko_preview.png"), RD.render(plan))
    st = RD.plan_stats(plan)
    st["white"] = report(size, pan)
    print(json.dumps(st, indent=1))
    return plan


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
