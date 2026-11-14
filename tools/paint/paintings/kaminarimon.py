"""Kaminarimon (雷門), the gate of Senso-ji in Asakusa, for the thin round brush: the great red lantern hanging
in the gate, a blue-black roof above it, two red pillars, the characters 雷門 on the lantern.

How it came about (2026-11-13). The user had swapped the flat brush for a thin round one and asked
「筆を細くしました。浅草雷門を…水彩画で描けますか」, then
「雷門の門に焦点を当てて水彩画を」. Poster-like drafts were turned down with
「もう少し水彩画っぽくならないですかね？」; of three watercolour designs made with the kit (roundbrush.py) the
user chose "WB", "the lantern as a body": washes of thinned red laid as bowed ribs with paper between them, the
light as thinned yellow, nothing ruled. It was painted from about 18:10 to 19:01: 70 strokes in 8 layers
(yellow_pale, red_pale, red_mid, red, blue_pale, black_pale, black bands, "ink" written lines), canvas
224 x 153 mm with its origin at (0.146, -0.118) m, 50 minutes by the clock. After
「文字をもう少しはっきり」 the grey strokes of the lettering were written again in dense black (13 strokes,
"ink2"). Result: runs/20261113_paint/190853_run.jpg on that rig.

What came out differently from the preview the kit had shown (roundbrush.render(as_painted=True) now shows most
of it beforehand):
- the "pale" blue roof is a dark navy band and the "pale" black an almost black line: the pale recipe (water
  twice, a touch of the blob) carries nearly the full paint of a dark tube from the second load on;
- the lettering turned grey where it crossed the pale red wash, which was still damp when the writing began,
  25 minutes after the wash was finished (the times of the photos);
- the yellow light under the lantern is a thin line: at the far middle of the sheet the contact plane was
  1.5 mm too high, and it was the first band;
- the strong red ran to the right, so that the lettering sits left of the lantern's middle; the left pillar is
  fainter than the right; the roof sags about a centimetre in the middle (not in the list of the evening: seen
  in the photo for this update, cause not known).

    python kaminarimon.py [out_dir] [painted | next] [press of the far light, for next: 1.2 | 1.5]
writes kaminarimon_plan.json (kaminarimon_next_plan.json), kaminarimon_preview.png (the sheet as the brush will
leave it: 12 mm of paper round the canvas as in paintskill.canvas_view, the marks where they land) and
kaminarimon_design.png (the picture as it was designed) and prints the statistics (for next: with what is
still open and the command that paints each layer).

build() returns EXACTLY the plan that was painted: it is data (kaminarimon_plan.json beside this module, byte
for byte the file the executor ran; PAINTED holds the digests the executor recorded for its layers), because
the design script drew on a random generator and a plan that was painted must not change when numpy does.
retrace() is the touch-up (the file of 19:04 was written by hand and names no brush: what retrace() returns
carries the brush block of the picture it belongs to), done() both together: what is on the sheet.

How it was run, which no plan says (the run log of that rig, runs/20261113_110358/events.jsonl, and the note in
its set-up): layer 0 with the contact plane as it was measured (z0 5.42 mm); its first band, the light at the
far middle of the sheet, came out a thin line. At 18:17 z0 was lowered to 3.9 mm by hand (the last band of
layer 0 was painted a second time in that minute), and layers 1 to 6, all bands, were painted with the plane
1.5 mm lower (press 1.2 = 6 mm, so 7.5 mm below the measured first touch); at 18:55 z0 went back to 5.42 mm
for the written lines. Bands were reloaded after 75 mm of travel, the written lines after 110, the retraced
letters after 60. The executor has this as options of one command now: `fd paint run PLAN --layer N
--lower-mm 1.5` paints a layer with the plane that much lower than the set-up stores it (press and lowering
together at most 7.5 mm, what those bands ran at: paintskill.MAX_DEPTH_M), and `--reload-mm 110` sets the
travel on one load.

build(variant="next") is the same picture with what the evening taught, where it is certain, and it is a plan
the executor takes as it stands (paintskill.plan_problems finds nothing with the rig's set-up and a round
brush block). Its "run" list has the command of every layer:
- the lettering is written ONCE, in dense black ("ink2": no water dip, blotted, a load every 60 mm), as a layer
  of its own at the very end, to be started only when the red is dry. How long that takes was not measured:
  25 minutes were not enough for the pale wash.
- the yellow light at the far side keeps the press it was painted with, 1.2: at the set-up's press_mm 5.0
  that is 6 mm below first touch, all that brush work allows (brushwork.MAX_PRESS_M; the executor refuses a
  plan that asks for more, and roundbrush.stats flags it). The 1.5 mm that the contact plane was too high
  there come from the command instead, as they did for the bands after it:
      fd paint run PLAN --layer 0 --lower-mm 1.5
  and so for every band layer (0 to 6); the two written layers are run without. It is NOT proven that 1.5 mm
  is enough at the far middle: the black ring, 15 mm nearer to the robot than the light, was painted with the
  plane lowered so and is a thin line too (it was the second band on its load, so that may be the paint and
  not the height). Paint layer 0 first and look. 1.5 mm is also the most the executor gives a band (6 mm of
  press and the lowering together may be 7.5 mm): if the light is thin again, the stored plane is too high at
  that place and the paper has to be measured there (`fd paint contact --help`).
  build(variant="next", press_far=1.5) writes the 1.5 mm into the plan instead (7.5 mm, the light cut in two so
  that no piece takes more than one load). The executor refuses that plan as long as brush work's limit is
  6 mm; it is kept for the day that limit is another.
- the roof keeps its pale blue and pale black, and they are OPEN problems, not solved ones: no recipe of that
  evening gives a pale tone of a dark tube after the first band. Until the set-up has a real dilution step the
  roof comes out dark navy and black again, as the preview shows.
Nothing else is changed: the pillars, the ribs and the order of the strokes are those that were painted.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shlex
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import tazuna_paths  # noqa: E402,F401

import cv2  # noqa: E402

import roundbrush as RB  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
PLAN_FILE = os.path.join(HERE, "kaminarimon_plan.json")
RETRACE_FILE = os.path.join(HERE, "kaminarimon_letters_touchup.json")
# What the executor recorded as painted (paint_state.json of 2026-11-13 19:10): per layer the first ten digits of
# sha1(json.dumps(strokes, sort_keys=True)), the key `fd paint run` keeps its progress under, and the number of
# strokes it painted. build() and retrace() are checked against it (painted_as_recorded, tests).
PAINTED = (
    ("light: the plaque, the lit shoulder of the lantern", "7bc8dee94b", 6),
    ("red wash: the lantern", "1469618280", 16),
    ("red glaze: beam, pillars; lantern shadow side and lower third", "3a8b3b58d7", 14),
    ("red, full: the rim", "f2181f3dd8", 4),
    ("roof wash", "6afa421aef", 6),
    ("roof towards the eave", "042c6fe9ef", 3),
    ("black accents: eave, rings, 雷門 (bands)", "f4eb134508", 2),
    ("black accents: eave, rings, 雷門 (written lines)", "0c42a92494", 19),
)
RETRACED = (("letters again: the grey strokes in dense black", "469a03ca6c", 13),)
EAVE_LINES = 2          # the first strokes of the written layer are the two lines along the eave; the rest is 雷門
GLOW_V_MM = 130.0       # the yellow light is the one stroke of the first layer that lies below this
PRESS_FAR = RB.P_BROAD  # the far light in the next plan: the press it was painted with (6 mm at the set-up's press_mm 5.0)
PRESS_FAR_MAX = 1.5     # 7.5 mm: the 6 mm of a band and the 1.5 mm the plane was too high; more than the executor takes
LOWER_MM = 1.5          # `fd paint run --lower-mm`: how much lower than measured the plane was for the bands (z0 3.9 for 5.42 mm)


def _load(path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def digest(strokes) -> str:
    return hashlib.sha1(json.dumps(strokes, sort_keys=True).encode()).hexdigest()[:10]


def painted_as_recorded(plan: dict, record=PAINTED) -> bool:
    """Are the plan's layers, stroke by stroke, what the executor recorded as painted?"""
    got = tuple((ly["name"], digest(ly["strokes"]), len(ly["strokes"])) for ly in plan["layers"])
    return got == tuple(record)


def retrace() -> dict:
    """The touch-up of 19:04: thirteen strokes of 雷門 again, in dense black. The file is what the executor ran;
    it was written by hand and has no "brush" block, which makes it a plan for the flat brush by the rule of
    this update (render.brush_of). What is returned says which brush painted it: the block of the picture."""
    plan = _load(RETRACE_FILE)
    plan.setdefault("brush", _load(PLAN_FILE)["brush"])
    return plan


def done() -> dict:
    """What is on the sheet: the plan and the retraced lettering as one more layer (for previews)."""
    plan, again = _load(PLAN_FILE), retrace()
    plan["paints"] = {**plan["paints"], **again["paints"]}
    plan["layers"] = plan["layers"] + again["layers"]
    return plan


def run_commands(plan: dict, path: str = "PLAN") -> list:
    """The command that paints each layer of a plan of this picture, in the order of the layers, with the
    options the layers were run with on 2026-11-13 (module text): a layer of bands with the contact plane
    LOWER_MM lower than the set-up stores it, a written layer with the plane as stored and the travel its
    black lasts on one load. A layer whose bands already ask for more than the band press (press_far over 1.2)
    is not lowered on top of that. path: the plan file, as it is to stand in the commands."""
    out = []
    for li, layer in enumerate(plan["layers"]):
        presses = [float(st.get("press", RB.P_FINE)) for st in layer["strokes"]]
        cmd = f"fd paint run {shlex.quote(path)} --layer {li}"
        if presses and all(RB.mode_of(p) == "broad" for p in presses):
            if max(presses) <= RB.P_BROAD + 1e-9:
                cmd += f" --lower-mm {LOWER_MM:g}"
        elif layer["paint"] in RB.WRITING:
            cmd += f" --reload-mm {RB.WRITING[layer['paint']][1]:g}"
        out.append(cmd)
    return out


def build(variant: str = "painted", pan_axis_mm=None, press_far: float | None = None) -> dict:
    """variant "painted": the plan as it was painted. "next": with the lessons of that evening (module text),
    a plan the executor takes as it stands; its "run" list has the command of every layer and its "open" list
    what is not solved. pan_axis_mm moves it to another canvas origin (paintskill.pan_axis_mm(setup): a plan
    whose pan axis is not the set-up's is refused). press_far is the press of the yellow light at the far side:
    PRESS_FAR when it is not given (1.2, as it was painted: its 1.5 mm more come from `--lower-mm`), up to
    PRESS_FAR_MAX (1.5, which the executor refuses as long as brush work allows 6 mm)."""
    plan = _load(PLAN_FILE)
    if variant == "painted":
        if pan_axis_mm is not None or press_far is not None:
            raise ValueError("the painted plan is a record: it keeps its pan axis and its presses (use variant='next')")
        return plan
    if variant != "next":
        raise ValueError(f"unknown variant {variant!r}: 'painted' or 'next'")
    press_far = PRESS_FAR if press_far is None else float(press_far)
    if not (PRESS_FAR <= press_far <= PRESS_FAR_MAX):
        raise ValueError(f"press_far {press_far!r}: between {PRESS_FAR} (a band as it was painted) and {PRESS_FAR_MAX}")
    br = RB.brush_of(plan)
    W, WB, L = br["wide_mm"], br["broad_mm"], br["load_mm"]
    layers = copy.deepcopy(plan["layers"])
    # the yellow light: more press, and cut so that no piece takes more than one load at that press
    light = layers[0]
    strokes = []
    for st in light["strokes"]:
        if min(p[1] for p in st["pts"]) > GLOW_V_MM and press_far > float(st["press"]):
            longest = 0.95 * L * W / RB.width_of(press_far, W, WB)
            strokes += [{"pts": piece, "press": press_far} for piece in RB.split(st["pts"], longest)]
        else:
            strokes.append(st)
    light["strokes"] = strokes
    # the written layer: the lines along the eave stay where they are, the lettering becomes the last layer
    written = layers[-1]
    letters = written["strokes"][EAVE_LINES:]
    written["strokes"] = written["strokes"][:EAVE_LINES]
    written["name"] = "black accents: eave (written lines)"
    layers.append({"name": "雷門: written once, in dense black, when the red is DRY", "paint": "ink2", "strokes": letters})
    lowered = f"`fd paint run PLAN --layer 0 --lower-mm {LOWER_MM:g}`"
    if press_far <= RB.P_BROAD + 1e-9:
        far = (f"light: the band under the lantern keeps the press it was painted with ({press_far:g}: 6 mm below first "
               "touch at press_mm 5.0, all that brush work allows), and on 2026-11-13 it was a thin line: the contact "
               f"plane was {LOWER_MM:g} mm too high at the far middle of the sheet. Its extra press comes from the "
               f"command: {lowered} paints the layer with the plane that much lower than the set-up stores it, as the "
               "band layers after it were painted that evening (\"run\" has the command of every layer). Not proven "
               "to be enough there: the black ring 15 mm nearer to the robot was painted so and is a thin line too "
               "(the second band on its load: that may be the paint). Paint layer 0 first and look. It is also the "
               "most the executor gives a band (press and --lower-mm together at most 7.5 mm): if the light is thin "
               "again, the paper height has to be measured at that place (`fd paint contact --help`)")
    else:
        far = (f"light: the band under the lantern asks for press {press_far:g} ({press_far * 5.0:g} mm below first touch "
               "at press_mm 5.0). NOT PAINTABLE: brush work allows 6 mm (brushwork.MAX_PRESS_M), the executor refuses "
               "the plan and roundbrush.stats flags it. build(variant='next') without press_far is the plan the "
               f"executor takes: there the light keeps press {PRESS_FAR:g} and gets the {LOWER_MM:g} mm from {lowered}")
    out = {"name": "kaminarimon_next", "canvas_mm": plan["canvas_mm"],
           "pan_axis_mm": plan["pan_axis_mm"] if pan_axis_mm is None else [float(pan_axis_mm[0]), float(pan_axis_mm[1])],
           "brush": plan["brush"],
           "paints": {**plan["paints"], "black_pale": plan["paints"]["black"], "ink2": plan["paints"]["ink"]},
           "layers": layers,
           "source": "paintings/kaminarimon.py build(variant='next'): the plan painted on 2026-11-13 with the "
                     "lettering as a last layer in ink2"
                     + ("" if press_far <= RB.P_BROAD + 1e-9 else f" and the yellow light at press {press_far:g}"),
           "open": [
               "roof: blue_pale and black_pale come out nearly full strength from their second load on (no dilution "
               "step in the set-up): expect a dark navy and black roof, as on 2026-11-13",
               far,
               "lettering: start the last layer only when the red is dry (25 minutes were not enough for the pale wash; "
               "the drying time was not measured). The writing blacks last less than a load of other paint: \"ink2\" was "
               "reloaded every 60 mm, \"ink\" every 110 mm. \"run\" says so with --reload-mm, which has to be given with "
               "every call of such a layer, also one that continues it; \"load_mm\" in the set-up's recipes does the same "
               "without an option (the set-up of 2026-11-13 has none: without either the executor reloads them every "
               "140 mm, and the letters get 3 loads instead of 8)",
           ]}
    out["run"] = run_commands(out)
    return out


def main(out_dir: str = ".", variant: str = "painted", press_far: float | None = None) -> dict:
    plan = build(variant, press_far=press_far)
    stem = "kaminarimon" if variant == "painted" else f"kaminarimon_{variant}"
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{stem}_plan.json")
    with open(path, "w") as f:
        json.dump(plan, f)
    cv2.imwrite(os.path.join(out_dir, f"{stem}_preview.png"), RB.render(plan, margin_mm=12.0))
    cv2.imwrite(os.path.join(out_dir, f"{stem}_design.png"), RB.render(plan, as_painted=False))
    st = RB.stats(plan)
    if plan.get("open"):
        st["open"] = plan["open"]
    if plan.get("run"):
        st["run"] = run_commands(plan, os.path.abspath(path))       # the same commands, with the file that was just written
    print(json.dumps(st, indent=1, ensure_ascii=False))
    return plan


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".", sys.argv[2] if len(sys.argv) > 2 else "painted",
         float(sys.argv[3]) if len(sys.argv) > 3 else None)
