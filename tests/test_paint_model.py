"""Tests for the brush models and the pictures (tools/paint: roundbrush.py, render.py, shapes.py, paintings/).
Run: python tests/test_paint_model.py [name-part ...]      (no hardware; nothing moves)

What they show.
The thin round brush (roundbrush.py): a line and a band are as wide as designed in four directions in the
designed picture, and as wide as they were MEASURED in the picture as painted (a line 1.2 mm sideways, 2.7 mm
along the direction to the robot's pan axis; a band 4 to 5 mm); the first stroke after a load is the darkest and
the widest and the mark fades along the load; tints are as pale as their name when designed and as strong as
they were seen when painted; a wet writing black turns grey over a wash and a dense one does not; marks land
LANDING_MM from the plan; a wash covers its shape with the brush inside; no stroke of the kit uses more than
one load; the statistics count a band as more paint than a line and flag what the executor would refuse of a
plan's strokes (a point outside the canvas, a stroke over one load or under 1 mm, a press beyond what brush
work allows); the render is deterministic.
The time of the statistics: within 5 % of the five calls it was fitted to, which is IN-SAMPLE and says little;
fitted to four of them and asked for the fifth it is within 25 %; of eleven other calls of the same evening
(the run log of the rig) ten are met within 12 % and the lettering in dense black is 26 % too long.
The pictures (paintings/): kaminarimon.py returns the plan that was painted, stroke by stroke (the digests the
executor recorded); its "next" variant is paintable as it is built: the lettering last, the far light at the
press it was painted with, and a "run" list with the command of every layer.
The kit against the executor (paintskill, paintcli: imported by the tests that need them, on scratch files):
every plan a painting module builds by default passes paintskill.plan_problems with a copy of the rig's set-up
(the evening's with a round brush block for the Kaminarimon, the morning's for the Akabeko); what the
statistics call a problem the executor refuses, and what it refuses of a plan's strokes they call a problem;
the commands of the next plan's "run" list are taken by `fd --sim paint run ... --dry` on a simulated arm.
The flat brush (render.py, shapes.py): band widths with and without the spread of the real sheet; valid
centres keep the contact line inside a shape; contour and fill cover a shape; a plan without a brush block
renders as it did before the block existed but for the stated spread; design_report flags a 10 mm hole, passes
a 30 mm hole and says of the Akabeko what the sheet showed; fit_to_photo recovers a known shift and spread
from a synthetic photo and, when the photo of that morning is on this computer, the morning's numbers.

No file of the rig is written: the tool, the set-up, the state, the photo folder, the run log and the hardware
lock of this process are scratch files (the lock stays set: nothing here can reach the real arm or the
camera); the last test checks that the rig's files are what they were.
A pass says what the MODEL does, and that the executor's CHECKS take a plan. Whether the next sheet looks like
the preview, and whether a band at the far middle is a band, is decided on the real arm.
"""

import os as _os  # the harness modules live in tools/<feature>/ (see tools/tazuna_paths.py)
import sys as _sys

_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "tools"))
import tazuna_paths  # noqa: E402,F401

import contextlib
import copy
import hashlib
import io
import json
import math
import os
import shlex
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
_TMP = tempfile.mkdtemp(prefix="tz_paintmodel_")
# The brush models read no file of the rig. The tests that ask the EXECUTOR about a plan (paintskill, paintcli on
# a simulated arm) do read a tool, a set-up and a state file: scratch files, named here before any module that
# reads them is imported, with the calibration of tests/testdata and a hardware lock of this process's own (a
# scratch file that stays set: whatever the user's lock says, nothing here opens the real arm or the camera).
TOOL_PATH = os.path.join(_TMP, "tool_active.json")
SETUP_PATH = os.path.join(_TMP, "paint_setup.json")
LOCK_PATH = os.path.join(_TMP, "hardware.lock")
os.environ["FD_TOOL"] = TOOL_PATH
os.environ.update(FD_PAINT_SETUP=SETUP_PATH, FD_PAINT_STATE=os.path.join(_TMP, "paint_state.json"),
                  FD_PAINT_OUT=os.path.join(_TMP, "out"), FD_RUNS=os.path.join(_TMP, "runs"),
                  FD_CAMERA_MODEL=os.path.join(_TMP, "camera_model.json"),
                  FD_CONFIG=os.path.join(HERE, "testdata", "pen_20261111", "config.json"), TAZUNA_HW_LOCK=LOCK_PATH)
with open(LOCK_PATH, "w") as _f:
    json.dump({"reason": "tests/test_paint_model.py", "since": time.strftime("%Y-%m-%dT%H:%M:%S"), "by": "test"}, _f)
RIG_FILES = ("tool_active.json", "tool_active.sim.json", "paint_setup.json", "paint_state.json", "paint_state.sim.json",
             os.path.join("config", "config.json"), os.path.join("config", "camera_model.json"))


def _rig_files() -> dict:
    out = {}
    for name in RIG_FILES:
        try:
            with open(os.path.join(tazuna_paths.ROOT, name), "rb") as fh:
                out[name] = hashlib.md5(fh.read()).hexdigest()
        except FileNotFoundError:
            out[name] = None
    return out


_RIG_BEFORE = _rig_files()  # before any test ran


_TRIPPED = []       # what the hook stopped (a command of the executor answers an error with a result: the last test asks)


def _no_hardware(event, args):
    """Audit hook: these tests must never open the servo bus or the camera."""
    if event == "open" and args and isinstance(args[0], (str, bytes)):
        p = os.fsdecode(args[0])
        if p.startswith("/dev/ttyACM") or p.startswith("/dev/ttyUSB") or p.startswith("/dev/video"):
            _TRIPPED.append(p)
            raise RuntimeError(f"test touched hardware: open({p})")
    if event == "subprocess.Popen" and args and "v4l2" in str(args[0]) + str(args[1] if len(args) > 1 else ""):
        _TRIPPED.append(str(args[:2]))
        raise RuntimeError(f"test touched the camera: {args[:2]}")


sys.addaudithook(_no_hardware)

sys.path.insert(0, os.path.join(tazuna_paths.TOOLS, "paint", "paintings"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

import akabeko as AK  # noqa: E402
import kaminarimon as KM  # noqa: E402
import render as RD  # noqa: E402
import roundbrush as RB  # noqa: E402
import shapes as SH  # noqa: E402

PAN = (118.0, -107.2)                  # the pan axis of the evening's canvas, in canvas mm
RUNS = os.path.join(tazuna_paths.ROOT, "runs", "20261113_paint")
STAGE_RED = os.path.join(tazuna_paths.ROOT, "video", "work_paint", "assets", "stage_red.jpg")
KIT = "/home/jetson/lerobot/jetharness_work/paint_update/reference/kit.py"      # the kit as it was that evening
# The five calls that were timed on the real arm on 2026-11-13 (the task of this update lists them):
# strokes, travel mm, loads, loads with blotting, washes, seconds.
CALLS = ((20, 447, 5, 0, 0, 289), (14, 584, 11, 0, 0, 469), (16, 839, 15, 0, 0, 694), (19, 497, 6, 6, 0, 409),
         (6, 427, 6, 0, 1, 342))
# What `fd paint run` recorded as painted that day (paint_state.json, "progress": plan | layer | digest: strokes).
KAMINARIMON = (("7bc8dee94b", 6, "yellow_pale"), ("1469618280", 16, "red_pale"), ("3a8b3b58d7", 14, "red_mid"),
               ("f2181f3dd8", 4, "red"), ("6afa421aef", 6, "blue_pale"), ("042c6fe9ef", 3, "black_pale"),
               ("f4eb134508", 2, "black"), ("0c42a92494", 19, "ink"))
RETRACE = ("469a03ca6c", 13, "ink2")
AKABEKO = (("a0919cf55e", 26, "red"), ("da20b85842", 8, "black"))


class Skip(Exception):
    pass


def digest(strokes) -> str:
    return hashlib.sha1(json.dumps(strokes, sort_keys=True).encode()).hexdigest()[:10]


# ----------------------------------------------------------------------------- the executor, on scratch files
BACKUP = "/home/jetson/lerobot/jetharness_work/paint_update/state_backup"      # the rig as it stood on 2026-11-13 19:25
RUN_LOG = os.path.join(tazuna_paths.ROOT, "runs", "20261113_110358", "events.jsonl")     # the day's run log (this computer only)
# what `fd paint brush --kind round --fine 0.6:2.7 --broad 1.2:5.0 --load-mm 140` stores in a set-up
ROUND_BLOCK = {"kind": "round", "modes": {"fine": {"press": 0.6, "width_mm": 2.7}, "broad": {"press": 1.2, "width_mm": 5.0}},
               "load_mm": 140}
# the thin round brush as it was registered that evening (tool_active.json of the rig) ...
THIN_BRUSH = {"name": "brush_thin", "tip_offset": [0.016, -0.0011, 0.0558], "mount": "other", "pitch_deg": 65.0,
              "gripper_raw": 2041, "attached_at": "2026-11-13T17:17:45",
              "joint_limits_deg": {"wrist_flex": [20.0, 97.0], "wrist_roll": [-5.0, 5.0]}}
# ... and what the executor asks of a brush before it paints since the review of that day: the jaw it is taped
# to and the squeeze that holds it (`fd tool jaw mount moving_outer`, `close`), a tip that is recorded as measured
READY = {"on_jaw": "moving_outer", "gripper_goal_raw": 2016,
         "tip_measured": {"at": "2026-11-13T17:25:00", "photos": 34, "rms_px": 6.6, "by": "tests/test_paint_model.py"}}
PHOTO_Q = [-89.2, -61.7, 36.2, 85.6, 0.0]
WELLS = {"red": [0.0716, 0.2651], "blue": [0.1453, 0.2583], "yellow": [0.1987, 0.1882], "crimson": [0.2117, 0.2544],
         "black": [0.2578, 0.1763]}


def written_out_setup(which: str) -> dict:
    """The part of the rig's painting set-up a plan check reads, for a computer that does not have the rig's
    files: canvas, contact plane and press of the evening of 2026-11-13 ("evening": the Kaminarimon; no brush
    block, as the rig's file) or of that morning ("morning": the Akabeko), and the recipes of the paints the
    plans of the painting modules use."""
    def rec(first, again, after=None):
        d = {"first": first, "reload": again}
        if after:
            d["after_without_rinse"] = after
        return d
    base = {"photo_q_deg": list(PHOTO_Q), "wells": {n: {"xy": list(xy), "z_mm": 2.5} for n, xy in WELLS.items()},
            "water": {"rinse": {"xy": [0.346, 0.192], "z_mm": 6.0}, "clean": {"xy": [0.34, 0.186], "z_mm": 6.0}}}
    if which == "morning":
        paints = {"red": rec([["water", "clean", 2], ["scrub", "crimson", 4], ["scrub", "crimson", 3]],
                             [["water", "clean", 1], ["scrub", "crimson", 4]]),
                  "black": rec([["water", "clean", 1], ["scrub", "black", 4]], [["water", "clean", 1], ["scrub", "black", 4]])}
        return {**base, "canvas": {"origin_xy": [0.146, -0.131], "size_mm": [224, 153], "pitch_deg": 65.0},
                "contact": {"z0": 0.0, "x0": 0.354, "y0": 0.0, "bx": 0.0, "by": 0.0}, "press_mm": 2.3,
                "blot": {"xy": [0.354, 0.082], "size": [0.03, 0.066], "z": 0.0}, "paints": paints}
    paints = {}
    for tube in ("red", "black", "blue", "yellow"):
        paints[tube] = rec([["water", "clean", 1], ["scrub", tube, 3]], [["water", "clean", 1], ["scrub", tube, 2]],
                           [tube + "_pale", tube + "_mid"])
        paints[tube + "_mid"] = rec([["water", "clean", 1], ["scrub", tube, 2]], [["water", "clean", 1], ["scrub", tube, 2]],
                                    [tube + "_pale"])
        paints[tube + "_pale"] = rec([["water", "clean", 2], ["touch", tube]], [["water", "clean", 1], ["touch", tube]])
    paints["ink"] = rec([["water", "clean", 1], ["scrub", "black", 2], ["blot"]], [["water", "clean", 1], ["scrub", "black", 1], ["blot"]],
                        ["black", "black_pale", "black_mid"])
    paints["ink2"] = rec([["scrub", "black", 3], ["blot"]], [["scrub", "black", 2], ["blot"]],
                         ["ink", "black", "black_pale", "black_mid"])
    return {**base, "canvas": {"origin_xy": [0.146, -0.118], "size_mm": [224, 153], "pitch_deg": 65.0},
            "contact": {"z0": 0.00542, "x0": 0.2225, "y0": -0.019, "bx": 0.0362, "by": -0.0062}, "press_mm": 5.0,
            "blot": {"xy": [0.354, 0.012], "size": [0.03, 0.04], "z": 0.006}, "paints": paints}


def rig_setups(which: str) -> list:
    """[(painting set-up, where it is from), ...] for a plan check: always the one written out above, and the
    rig's own file of that time when this computer has it (state_backup, read only: the tests work on copies)."""
    out = [(written_out_setup(which), "written out in the test")]
    path = os.path.join(BACKUP, "paint_setup.json" if which == "evening" else "paint_setup.before_thin_1700.json")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            out.append((json.load(f), path))
    return out


def with_round_brush(setup: dict) -> dict:
    s = copy.deepcopy(setup)
    s["brush"] = copy.deepcopy(ROUND_BLOCK)
    return s


def scratch_rig(setup: dict | None = None):
    """Make the scratch files of this process a rig that is ready to paint the Kaminarimon: the thin brush with
    its jaw hold and a measured tip in the tool file, the evening's set-up with a round brush block and a
    contact plane that is recorded for that brush in the set-up file. -> (paintskill, paintcli, the set-up)."""
    import paintcli
    import paintskill as PS
    import tooling as TL
    with open(TOOL_PATH, "w", encoding="utf-8") as f:
        json.dump({**THIN_BRUSH, **READY}, f)
    s = with_round_brush(rig_setups("evening")[-1][0] if setup is None else setup)
    s["contact"] = {k: v for k, v in s["contact"].items() if k != "measured"}
    s["contact"]["measured"] = PS.brush_state(TL.active(), float(s["canvas"].get("pitch_deg", 65.0)))
    with open(SETUP_PATH, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1, ensure_ascii=False)
    return PS, paintcli, PS.load_setup()


def fd_paint(paintcli, *words) -> dict:
    """The result of `fd --sim paint WORDS` (paintcli.main on a simulated arm; what it logs is not shown)."""
    with contextlib.redirect_stderr(io.StringIO()):
        res = paintcli.main(["--sim", *[str(w) for w in words]])
    assert isinstance(res, dict) and isinstance(res.get("ok"), bool) and res.get("stage") and "trace" not in res, res
    return res


def one_layer(paint, strokes, pan=PAN) -> dict:
    return {"canvas_mm": [224, 153], "pan_axis_mm": list(pan), "brush": {"kind": "round", "wide_mm": 2.7, "broad_mm": 5.0, "load_mm": 140.0},
            "paints": {}, "layers": [{"name": "t", "paint": paint, "strokes": strokes}]}


def layers_of(paint, strokes, pan=PAN) -> dict:
    """Every stroke in a layer of its own (so that each starts on a fresh load)."""
    plan = one_layer(paint, [], pan)
    plan["layers"] = [{"name": f"t{i}", "paint": paint, "strokes": [st]} for i, st in enumerate(strokes)]
    return plan


def cover_of(plan, paint, ppm=8.0, **kw):
    """How much of `paint` lies where: the float map of roundbrush.render, canvas only, nothing shifted."""
    kw.setdefault("margin_mm", 0.0)
    kw.setdefault("shift_mm", (0.0, 0.0))
    _img, cover = RB.render(plan, ppm, return_coverage=True, **kw)
    return cover[paint]


def width_at(cover, at_mm, along_uv, ppm=8.0, reach_mm=7.0) -> float:
    """Width in mm of a mark at a point: the full width at half maximum across its direction."""
    d = np.asarray(along_uv, dtype=float)
    d /= np.linalg.norm(d)
    n = np.array([-d[1], d[0]])
    offs = np.arange(-reach_mm, reach_mm + 1e-9, 0.02)
    xs = ((at_mm[0] + n[0] * offs) * ppm - 0.5).astype(np.float32).reshape(1, -1)
    ys = ((at_mm[1] + n[1] * offs) * ppm - 0.5).astype(np.float32).reshape(1, -1)
    prof = cv2.remap(cover, xs, ys, cv2.INTER_LINEAR).ravel()
    return float((prof >= 0.5 * prof.max()).sum()) * 0.02


def strength_along(cover, st, ppm=8.0, lo=0.0, hi=1.0) -> float:
    """Mean of the cover on the centre line of a stroke between the parts lo and hi of its length."""
    p = RB.resample(st["pts"], 0.5)
    p = p[int(lo * (len(p) - 1)): int(hi * (len(p) - 1)) + 1]
    xs = (p[:, 0] * ppm - 0.5).astype(np.float32).reshape(1, -1)
    ys = (p[:, 1] * ppm - 0.5).astype(np.float32).reshape(1, -1)
    return float(cv2.remap(cover, xs, ys, cv2.INTER_LINEAR).mean())


# The four directions at a place straight under the pan axis: sideways both ways, away from the robot, towards it.
FOUR = {"right": [(97, 50), (127, 50)], "left": [(127, 64), (97, 64)], "away": [(104, 76), (104, 106)], "towards": [(120, 106), (120, 76)]}


def four_widths(paint, press, as_painted) -> dict:
    strokes = [{"pts": [list(a), list(b)], "press": press} for a, b in FOUR.values()]
    cover = cover_of(layers_of(paint, strokes), paint, as_painted=as_painted)
    out = {}
    for name, (a, b) in FOUR.items():
        mid = ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)
        out[name] = width_at(cover, mid, (b[0] - a[0], b[1] - a[1]))
    return out


# ----------------------------------------------------------------------------- round brush: widths
def test_round_designed_widths_are_the_same_in_four_directions():
    for press, want in ((0.6, 2.7), (1.2, 5.0)):
        w = four_widths("red", press, as_painted=False)
        assert all(abs(v - want) < 0.35 for v in w.values()), (press, w)
    assert abs(RB.width_of(0.6) - 2.7) < 1e-9 and abs(RB.width_of(1.2) - 5.0) < 1e-9
    assert abs(RB.width_of(0.9) - 3.85) < 1e-9 and abs(RB.width_of(1.2, 2.0, 4.0) - 4.0) < 1e-9     # straight between


def test_round_painted_line_is_thin_sideways_and_wide_along_the_robots_direction():
    # a blotted writing black: no fat start, so the width is the line's own from its first millimetre
    w = four_widths("ink", 0.6, as_painted=True)
    assert abs(w["right"] - 1.2) < 0.35 and abs(w["left"] - 1.2) < 0.35, w
    assert abs(w["away"] - 2.7) < 0.35 and abs(w["towards"] - 2.7) < 0.35, w
    assert min(w["away"], w["towards"]) > 1.8 * max(w["right"], w["left"]), w
    # the same numbers from the function the render uses, and a diagonal between them
    assert RB.painted_width(0.6, 0.0) == 1.2 and RB.painted_width(0.6, 1.0) == 2.7
    assert 1.2 < RB.painted_width(0.6, math.cos(math.radians(45))) < 2.7
    assert RB.mode_of(0.6) == "fine" and RB.mode_of(1.2) == "broad" and RB.mode_of(0.7) == "fine"


def test_round_painted_band_is_4_to_5_mm_in_four_directions():
    w = four_widths("red", 1.2, as_painted=True)
    assert all(4.0 <= v <= 5.2 for v in w.values()), w
    assert max(w.values()) - min(w.values()) < 0.4, w
    # 5 mm on a fresh load, 4 mm where the load is used up: the measured range, the fresh end the widest
    assert RB.painted_width(1.2, 0.0, used_mm=0.0) == 5.0 and RB.painted_width(1.2, 1.0, used_mm=0.0) == 5.0
    assert abs(RB.painted_width(1.2, 0.3, used_mm=140.0) - 4.0) < 1e-9


def test_round_first_stroke_after_a_load_is_the_widest_and_the_darkest():
    # three sideways lines of 40 mm on one load of red (not blotted), then a band that needs a new load; all of
    # them under the pan axis, where "along u" is sideways (further left or right a level line is partly radial)
    strokes = [{"pts": [[138, 40 + 8 * i], [98, 40 + 8 * i]], "press": 0.6} for i in range(3)]
    strokes.append({"pts": [[88, 80], [148, 80]], "press": 1.2})
    plan = one_layer("red", strokes)
    on = RB.loads(plan)[0]
    assert [k for k, _a, _b in on] == [0, 0, 0, 1], on
    cover = cover_of(plan, "red")
    w_start = width_at(cover, (136.0, 40.0), (1, 0))                           # 2 mm after the load
    w_later = width_at(cover, (102.0, 40.0), (1, 0))                           # 36 mm after it
    assert w_start > 4.0 and abs(w_later - 1.2) < 0.45, (w_start, w_later)     # "up to 5 mm even at the light press"
    assert abs(RB.painted_width(0.6, 0.0, used_mm=0.0) - 5.0) < 1e-9 and RB.painted_width(0.6, 0.0, used_mm=0.0, wet=False) == 1.2
    s = [strength_along(cover, st, lo=0.2, hi=0.8) for st in strokes[:3]]
    assert s[0] > s[1] * 1.1 and s[1] > s[2] * 1.1, s                         # it fades along the load ...
    a, b = strength_along(cover, strokes[3], lo=0.03, hi=0.15), strength_along(cover, strokes[3], lo=0.8, hi=0.95)
    assert a > s[2] * 1.3 and a > b * 1.3, (a, b, s)                          # ... is dark again after the next one, and fades again
    # a load that is used up has FADE_END of its strength left
    full = one_layer("red", [{"pts": [[80, 100], [80 + 140 * 2.7 / 5.0 - 0.5, 100]], "press": 1.2}])
    c2 = cover_of(full, "red")
    ratio = strength_along(c2, full["layers"][0]["strokes"][0], lo=0.93, hi=0.96) / strength_along(c2, full["layers"][0]["strokes"][0], lo=0.03, hi=0.06)
    assert abs(ratio - RB.FADE_END) < 0.12, ratio
    # the designed picture knows nothing of loads
    cd = cover_of(plan, "red", as_painted=False)
    sd = [strength_along(cd, st, lo=0.2, hi=0.8) for st in strokes[:3]]
    assert max(sd) - min(sd) < 0.08 * max(sd), sd


def test_round_blotted_black_starts_thin():
    st = {"pts": [[138, 40], [98, 40]], "press": 0.6}                       # sideways, under the pan axis
    wet = width_at(cover_of(one_layer("black", [st]), "black"), (136.0, 40.0), (1, 0))
    dry = width_at(cover_of(one_layer("ink", [st]), "ink"), (136.0, 40.0), (1, 0))
    assert wet > 4.0 and dry < 1.8, (wet, dry)
    assert RB.blotted("ink") and RB.blotted("ink2") and not RB.blotted("black")
    assert RB.tube_of("ink2") == "black" and RB.load_mm_of("ink") == 110.0 and RB.load_mm_of("ink2") == 60.0 and RB.load_mm_of("red") == 140.0


def test_round_marks_land_where_they_landed():
    st = {"pts": [[80, 60], [120, 60]], "press": 1.2}
    plan = one_layer("red", [st])

    def centre(**kw):
        _img, cov = RB.render(plan, 4.0, return_coverage=True, **kw)
        ys, xs = np.nonzero(cov["red"] > 0.2)
        return np.array([xs.mean() / 4.0, ys.mean() / 4.0])
    # the picture is the canvas, in the plan's own frame: a point (u, v) of the plan is at pixel (u * ppm, v * ppm)
    here = centre()
    assert RB.render(plan, 4.0).shape == RB.render(plan, 4.0, as_painted=False).shape == (612, 896, 3)
    assert abs(here[0] - 100.0) < 2.0 and abs(here[1] - 60.0) < 0.3, here    # (a little towards its start: the band is wider there)
    assert np.allclose(centre(shift_mm=(0.0, 0.0)), here)
    # the picture of the SHEET (12 mm of paper round the canvas, the frame of paintskill.canvas_view) shows the
    # marks where they land: about (-6, -9) mm from where the plan puts them
    assert RB.LANDING_MM == (-6.0, -9.0)
    assert RB.render(plan, 4.0, margin_mm=12.0).shape == (708, 992, 3)
    assert np.allclose(centre(margin_mm=12.0) - here, np.array(RB.LANDING_MM) + 12.0, atol=0.3), centre(margin_mm=12.0) - here
    assert np.allclose(centre(margin_mm=12.0, shift_mm=(0.0, 0.0)) - here, (12.0, 12.0), atol=0.3)
    assert np.allclose(centre(shift_mm=RB.LANDING_MM) - here, RB.LANDING_MM, atol=0.3)         # a landing of one's own
    assert np.allclose(centre(shift_mm=(2.0, -3.0)) - here, (2.0, -3.0), atol=0.3)
    # a band 6.5 mm from the top edge lands above the canvas: the sheet shows it, whole, beside the corner marks
    top = one_layer("blue", [{"pts": [[20, 6.5], [80, 6.5]], "press": 1.2}])
    _img, cov = RB.render(top, 4.0, margin_mm=12.0, return_coverage=True)
    rows = np.nonzero((cov["blue"] > 0.2).any(axis=1))[0]
    assert abs(rows.mean() / 4.0 - (6.5 - 9.0 + 12.0)) < 0.5 and rows.min() >= 0 and rows.max() / 4.0 < 12.5
    # the sheet has a small mark at each corner of the canvas (the marks of the brush land beside it); the canvas
    # alone has none
    sheet = RB.render(one_layer("red", []), 4.0, margin_mm=12.0)
    bare = RB.render(one_layer("red", []), 4.0, margin_mm=12.0, as_painted=False)
    marks = np.argwhere(np.abs(sheet.astype(int) - bare.astype(int)).max(axis=2) > 30)[:, ::-1] / 4.0 - 12.0      # (u, v) in mm
    corners = np.array([(0, 0), (224, 0), (0, 153), (224, 153)], dtype=float)
    nearest = np.linalg.norm(marks[:, None, :] - corners[None, :, :], axis=2)
    assert len(marks) > 40 and float(nearest.min(axis=1).max()) < 4.0 and set(nearest.argmin(axis=1).tolist()) == {0, 1, 2, 3}
    assert np.array_equal(RB.render(one_layer("red", []), 4.0), RB.render(one_layer("red", []), 4.0, as_painted=False))
    # the designed picture is where the plan puts it, whatever shift is asked for
    img, cov = RB.render(plan, 4.0, as_painted=False, shift_mm=(5.0, 5.0), return_coverage=True)
    assert np.array_equal(img, RB.render(plan, 4.0, as_painted=False))
    ys, xs = np.nonzero(cov["red"] > 0.2)
    assert abs(xs.mean() / 4.0 - 100.0) < 0.8 and abs(ys.mean() / 4.0 - 60.0) < 0.3      # (its start is a little fatter)


# ----------------------------------------------------------------------------- round brush: tints
def test_round_tints_as_designed_and_as_seen():
    assert RB.base_paint("red_pale") == "red" and RB.tint_of("red_mid") == "mid" and RB.tint_of("red") == "" and RB.base_paint("ink2") == "ink2"
    assert RB.designed_alpha("red_pale") < RB.designed_alpha("red_mid") < RB.designed_alpha("red") < RB.designed_alpha("black")
    full = RB.painted_alpha("red")
    # red: pale stays a wash on every load, mid is hardly darker
    assert RB.painted_alpha("red_pale", 0) == RB.painted_alpha("red_pale", 3) < 0.5 * full
    assert RB.painted_alpha("red_pale") < RB.painted_alpha("red_mid") < 1.3 * RB.painted_alpha("red_pale")
    # dark tubes: pale on the first load only
    for tube in ("blue", "black"):
        f = RB.painted_alpha(tube)
        assert RB.painted_alpha(f"{tube}_pale", 1) >= 0.85 * f and RB.painted_alpha(f"{tube}_pale", 5) == RB.painted_alpha(f"{tube}_pale", 1)
        assert RB.painted_alpha(f"{tube}_mid", 1) >= RB.painted_alpha(f"{tube}_pale", 1) * 0.85
    assert RB.painted_alpha("blue_pale", 0) < 0.3 * RB.painted_alpha("blue")
    # a tube that was not painted that evening is shown as designed
    assert RB.painted_alpha("green_pale", 0) == RB.painted_alpha("green_pale", 2) == RB.designed_alpha("green_pale")
    # in the picture: two bands of pale blue, each on its own load
    strokes = [{"pts": [[30, 40], [100, 40]], "press": 1.2}, {"pts": [[30, 60], [100, 60]], "press": 1.2}]
    plan = one_layer("blue_pale", strokes)
    assert [k for k, _a, _b in RB.loads(plan)[0]] == [0, 1]
    c = cover_of(plan, "blue_pale")
    first, second = (strength_along(c, st, lo=0.1, hi=0.3) for st in strokes)
    assert second > 3.0 * first, (first, second)
    d = cover_of(plan, "blue_pale", as_painted=False)
    first, second = (strength_along(d, st, lo=0.1, hi=0.3) for st in strokes)
    assert abs(first - second) < 0.1 * first and abs(first - RB.TINT["pale"]) < 0.08, (first, second)
    # the kit's own names: a plan refuses a paint it does not know, takes tints and the writing blacks
    p = RB.Plan()
    for name in ("red", "red_pale", "cerulean_mid", "ink", "ink2"):
        p.layer("x", name)
    try:
        p.layer("x", "pink_pale")
    except ValueError as e:
        assert "pink_pale" in str(e)
    else:
        raise AssertionError("an unknown paint was accepted")


def test_round_wet_black_over_a_wash_is_grey_and_dense_black_is_not():
    wash = {"name": "wash", "paint": "red_pale", "strokes": [{"pts": [[40, 50 + 4 * i], [110, 50 + 4 * i]], "press": 1.2} for i in range(5)]}
    line = {"pts": [[75, 44], [75, 76]], "press": 0.6}                      # crosses the wash, starts and ends on bare paper

    def over(paint):
        plan = one_layer(paint, [line])
        plan["layers"].insert(0, wash)
        c = cover_of(plan, paint)
        return strength_along(c, line, lo=0.35, hi=0.65), strength_along(c, line, lo=0.0, hi=0.04)
    wet_on_wash, wet_on_paper = over("ink")
    dense_on_wash, _ = over("ink2")
    assert wet_on_wash < 0.6 * wet_on_paper, (wet_on_wash, wet_on_paper)
    assert dense_on_wash > 1.6 * wet_on_wash, (dense_on_wash, wet_on_wash)
    assert abs(wet_on_wash / dense_on_wash - RB.DAMP_GREY) < 0.2


# ----------------------------------------------------------------------------- round brush: the kit
def test_round_wash_covers_its_shape_with_the_brush_inside():
    poly = [(40, 40), (110, 40), (110, 95), (40, 95)]
    hole = [(70, 60), (85, 60), (85, 75), (70, 75)]
    # (bands along an edge of the shape cover it to the corners; oblique ones leave small teeth at the edges)
    for angle, least in ((0.0, 0.95), (90.0, 0.95), (30.0, 0.9)):
        p = RB.Plan(pan_axis_mm=PAN)
        p.layer("wash", "red").fill_polygon(poly, angle_deg=angle, holes=[hole])
        plan = p.to_dict()
        strokes = plan["layers"][0]["strokes"]
        assert len(strokes) >= 10 and all(st["press"] == 1.2 for st in strokes)
        ppm = 8.0
        shape = np.zeros((int(153 * ppm), int(224 * ppm)), np.uint8)
        cv2.fillPoly(shape, [np.round(np.array(poly) * ppm).astype(np.int32)], 1)
        cv2.fillPoly(shape, [np.round(np.array(hole) * ppm).astype(np.int32)], 0)
        painted = RD.layer_mask(plan, 0, ppm)                              # every stroke as wide as designed (5 mm)
        assert float((painted & shape).sum()) / float(shape.sum()) > least, angle
        grown = cv2.dilate(shape, np.ones((9, 9), np.uint8))               # half a millimetre of slack
        assert int((painted & (1 - grown)).sum()) == 0, (angle, int((painted & (1 - grown)).sum()))
        # the brush inside: every point of every stroke is at least half a band (less the slack) from the outline
        dist = cv2.distanceTransform(shape, cv2.DIST_L2, 5) / ppm
        for st in strokes:
            for u, v in RB.resample(st["pts"], 1.0):
                assert dist[int(v * ppm), int(u * ppm)] > 2.5 - 0.5, (angle, u, v)
    # a broken wash leaves paper between its bands, an ellipse is filled too
    p = RB.Plan(pan_axis_mm=PAN)
    p.layer("even", "red").fill_ellipse((75, 70), (30, 22), angle_deg=0)
    p.layer("broken", "red").fill_ellipse((165, 70), (30, 22), angle_deg=0, pitch=7.5)
    plan = p.to_dict()
    ell = np.zeros((612, 896), np.uint8)
    cv2.ellipse(ell, (300, 280), (120, 88), 0, 0, 360, 1, -1)
    even = float((RD.layer_mask(plan, 0, 4.0) & ell).sum()) / float(ell.sum())
    assert even > 0.88, even                                               # (teeth where the bands end on the curved outline)
    assert int((RD.layer_mask(plan, 0, 4.0) & (1 - cv2.dilate(ell, np.ones((5, 5), np.uint8)))).sum()) == 0
    ell2 = np.zeros((612, 896), np.uint8)
    cv2.ellipse(ell2, (660, 280), (120, 88), 0, 0, 360, 1, -1)
    assert 0.5 < float((RD.layer_mask(plan, 1, 4.0) & ell2).sum()) / float(ell2.sum()) < 0.8
    assert RB.stats(plan)["n_problems"] == 0
    # with a band along the outline first, the teeth are closed and the brush is still inside
    p = RB.Plan(pan_axis_mm=PAN)
    p.layer("outlined", "red").fill_ellipse((75, 70), (30, 22), angle_deg=0, outline=True)
    outlined = RD.layer_mask(p.to_dict(), 0, 4.0)
    assert float((outlined & ell).sum()) / float(ell.sum()) > 0.97 and RB.stats(p.to_dict())["n_problems"] == 0
    assert int((outlined & (1 - cv2.dilate(ell, np.ones((5, 5), np.uint8)))).sum()) == 0


def test_round_no_stroke_uses_more_than_one_load():
    p = RB.Plan(pan_axis_mm=PAN)
    zig = [(20 + 30 * (i % 2), 20 + 6 * i) for i in range(20)]            # 580 mm
    p.layer("lines", "black").line(zig)
    p.layer("bands", "red").band([(60, 20), (200, 20), (200, 130), (60, 130)])      # 390 mm
    p.layer("letters", "ink2").line([(70, 40), (190, 40), (190, 110)])    # 190 mm, a load of ink2 is 60 mm
    p.layer("dots", "black").dot((100, 100)).dot((104, 100), press=1.2)
    plan = p.to_dict()
    assert plan["brush"] == {"kind": "round", "wide_mm": 2.7, "broad_mm": 5.0, "load_mm": 140.0}
    wanted = {"lines": 580.0, "bands": 390.0, "letters": 190.0}
    for layer, rows in zip(plan["layers"], RB.loads(plan)):
        cap = RB.load_mm_of(layer["paint"], 140.0)
        for st in layer["strokes"]:
            assert RB.paint_use(st) <= cap + 1e-6, (layer["name"], RB.paint_use(st), cap)
        ks = [k for k, _a, _b in rows]
        assert ks[0] == 0 and all(b - a in (0, 1) for a, b in zip(ks, ks[1:]))
        assert all(0.0 <= a <= b <= 1.0 + 1e-9 for _k, a, b in rows), rows
        if layer["name"] in wanted:
            total = sum(RB.length_mm(st["pts"]) for st in layer["strokes"])
            assert abs(total - wanted[layer["name"]]) < 0.02 * wanted[layer["name"]], (layer["name"], total)
            for a, b in zip(layer["strokes"], layer["strokes"][1:]):       # the pieces meet end to start
                assert math.dist(a["pts"][-1], b["pts"][0]) < 1.5
    longest = {ly["name"]: max(RB.length_mm(st["pts"]) for st in ly["strokes"]) for ly in plan["layers"]}
    assert 100 < longest["lines"] <= 140 and 60 < longest["bands"] <= 75.6 and 45 < longest["letters"] <= 60, longest
    st = RB.stats(plan)
    assert st["n_problems"] == 0 and st["strokes"] == sum(len(ly["strokes"]) for ly in plan["layers"])
    assert RB.split([(0, 0), (50, 0)], 60.0) == [[[0.0, 0.0], [50.0, 0.0]]]
    arc = RB.arc((100, 80), (30, 20), 0, 90)
    assert np.allclose(arc[0], (130, 80)) and np.allclose(arc[-1], (100, 100))


# ----------------------------------------------------------------------------- round brush: statistics
def rows_plan(press, n=8):
    return one_layer("red", [{"pts": [[20, 20 + 8 * i], [80, 20 + 8 * i]], "press": press} for i in range(n)])


def test_round_statistics_count_a_band_as_more_paint_than_a_line():
    fine, broad = RB.stats(rows_plan(0.6)), RB.stats(rows_plan(1.2))
    assert fine["strokes"] == broad["strokes"] == 8 and fine["travel_mm"] == broad["travel_mm"] == 480
    st = {"pts": [[20, 20], [80, 20]], "press": 0.6}
    assert abs(RB.paint_use(st) - 60.0) < 1e-9 and abs(RB.paint_use({**st, "press": 1.2}) - 60.0 * 5.0 / 2.7) < 1e-9
    assert fine["loads"] == 4 and broad["loads"] == 8, (fine["loads"], broad["loads"])      # 2 lines a load, 1 band a load
    assert broad["minutes"] > fine["minutes"] and broad["broad_cm2"] == 24.0 and fine["broad_cm2"] == 0.0
    for key in ("strokes", "travel_mm", "loads", "layers", "minutes", "problems", "n_problems"):
        assert key in fine
    assert fine["layers"][0]["loads"] == 4 and fine["layers"][0]["strokes"] == 8 and fine["layers"][0]["travel_mm"] == 480
    # render.plan_stats hands a round plan over, and the picture of a round plan too
    assert RD.plan_stats(rows_plan(1.2)) == broad
    assert np.array_equal(RD.render(rows_plan(1.2), 4.0), RB.render(rows_plan(1.2), 4.0))
    # a stroke counts by the plan's own length (as the executor counts it): two bands that fill a load to the
    # brim are one load, a little more is two
    brim = one_layer("red", [{"pts": [[20, 30], [57.6, 30]], "press": 1.2}, {"pts": [[20, 40], [57.6, 40]], "press": 1.2}])
    assert 2 * 37.6 * 5.0 / 2.7 < 140.0 and RB.stats(brim)["loads"] == 1
    over = one_layer("red", [{"pts": [[20, 30], [58, 30]], "press": 1.2}, {"pts": [[20, 40], [58, 40]], "press": 1.2}])
    assert 2 * 38.0 * 5.0 / 2.7 > 140.0 and RB.stats(over)["loads"] == 2
    # a writing black is blotted and carries less: 110 mm of "ink", 60 of "ink2"
    lines = [{"pts": [[20, 30 + 6 * i], [70, 30 + 6 * i]], "press": 0.6} for i in range(4)]          # 4 x 50 mm
    assert [RB.stats(one_layer(p, lines))["loads"] for p in ("black", "ink", "ink2")] == [2, 2, 4]
    assert RB.stats(one_layer("ink", lines))["minutes"] > RB.stats(one_layer("black", lines))["minutes"]       # the blotting


def test_round_statistics_flag_what_the_robot_would_refuse():
    good = rows_plan(0.6)
    assert RB.stats(good)["n_problems"] == 0 and RB.stats(good)["problems"] == []
    out = one_layer("red", [{"pts": [[200, 50], [230, 50]], "press": 0.6}])                 # leaves the 224 mm canvas
    s = RB.stats(out)
    assert s["n_problems"] == 1 and "canvas edge" in s["problems"][0], s["problems"]
    near = one_layer("red", [{"pts": [[3, 50], [40, 50]], "press": 0.6}])                   # inside, but nearer than 5 mm
    assert RB.stats(near)["n_problems"] == 1
    long_band = one_layer("red", [{"pts": [[20, 50], [120, 50]], "press": 1.2}])            # 100 mm of band: 185 fine mm
    s = RB.stats(long_band)
    assert s["n_problems"] == 1 and "more than one load" in s["problems"][0], s["problems"]
    assert RB.stats(one_layer("red", [{"pts": [[20, 50], [120, 50]], "press": 0.6}]))["n_problems"] == 0     # as a line it is one load
    assert RB.stats(one_layer("ink2", [{"pts": [[20, 50], [90, 50]], "press": 0.6}]))["n_problems"] == 1     # 70 mm of ink2: over its 60
    assert RB.stats(one_layer("red", [{"pts": [[20, 50]], "press": 0.6}]))["n_problems"] == 1
    assert RD.plan_stats(long_band)["n_problems"] == 1
    # a stroke under 1 mm (the kit keeps pieces from 0.5 mm on; the executor refuses what is shorter than 1 mm)
    p = RB.Plan(pan_axis_mm=PAN)
    p.layer("speck", "red").line([(50, 50), (50.7, 50)])
    s = p.stats()
    assert len(p.to_dict()["layers"][0]["strokes"]) == 1 and s["n_problems"] == 1 and "shorter than 1 mm" in s["problems"][0], s
    assert RB.Plan(pan_axis_mm=PAN).layer("dot", "red").dot((50, 50)).plan.stats()["n_problems"] == 0        # a dab is 1.5 mm
    # a press beyond what the executor accepts. Without a set-up the kit's own cap stands in: the band mode
    band = lambda press: one_layer("red", [{"pts": [[60, 50], [80, 50]], "press": press}])      # noqa: E731
    assert RB.press_limit() == RB.P_BROAD == 1.2 and RB.MAX_PRESS_MM == 6.0
    assert RB.stats(band(1.2))["n_problems"] == 0 and RB.stats(band(0.9))["n_problems"] == 0
    for press in (1.21, 1.5, -0.1, float("nan")):
        s = RB.stats(band(press))
        assert s["n_problems"] == 1 and "press" in s["problems"][0] and RD.plan_stats(band(press))["n_problems"] == 1, (press, s["problems"])
    assert "1.2" in RB.stats(band(1.5))["problems"][0] and RB.press_problem(1.2) is None and RB.press_problem(0.0) is None
    # with the set-up the plan is meant for: brush work's 6 mm below first touch over its press_mm
    evening, shallow = {"press_mm": 5.0}, {"press_mm": 4.0}
    assert abs(RB.press_limit(evening) - 1.2) < 1e-12 and abs(RB.press_limit(shallow) - 1.5) < 1e-12 and RB.press_limit({}) == 3.0
    assert RB.stats(band(1.2), setup=evening)["n_problems"] == 0 and RB.stats(band(1.21), setup=evening)["n_problems"] == 1
    assert RB.stats(band(1.5), setup=shallow)["n_problems"] == 0 and RB.stats(band(1.6), setup=shallow)["n_problems"] == 1
    s = RB.stats(band(1.5), setup=evening)
    assert s["n_problems"] == 1 and "1.5 x 5 mm" in s["problems"][0] and "6 mm" in s["problems"][0], s["problems"]
    assert RD.plan_stats(band(1.5), evening) == s and RB.Plan(pan_axis_mm=PAN).stats(setup=evening)["n_problems"] == 0
    assert RB.stats(band(1.5), evening) == s and RB.stats(band(1.5), 140.0, shallow)["n_problems"] == 0     # (the set-up given second)
    try:
        RB.press_limit({"press_mm": 0})
    except ValueError as e:
        assert "press_mm" in str(e)
    else:
        raise AssertionError("a set-up without a usable press_mm was accepted")
    # the flat brush: the same limit as soon as the set-up is known (the Akabeko was run at press_mm 2.3)
    flat = {"canvas_mm": [224, 153], "pan_axis_mm": list(PAN), "paints": {"red": {"rgb": [214, 38, 44]}},
            "layers": [{"name": "a", "paint": "red", "strokes": [{"pts": [[60, 50], [80, 50]], "press": 1.0}, {"pts": [[60, 60], [80, 60]], "press": 2.7}]}]}
    assert RD.plan_stats(flat)["n_problems"] == 0 and RD.plan_stats(flat, {"press_mm": 2.2})["n_problems"] == 0
    s = RD.plan_stats(flat, {"press_mm": 2.3})
    assert s["n_problems"] == 1 and s["problems"][0].startswith("a[1]: press 2.7 x 2.3 mm"), s["problems"]
    # points that are not pairs are not read as other points (round and flat)
    assert "not pairs" in RB.stats(one_layer("red", [{"pts": [[60, 50, 3], [80, 50, 3]], "press": 0.6}]))["problems"][0]
    flat["layers"][0]["strokes"] = [{"pts": [[60, 50, 3], [80, 50, 3]], "press": 1.0}, {"pts": [[60, 60], [80, 60]], "press": 1.0}]
    s = RD.plan_stats(flat)
    assert s["n_problems"] == 1 and "not pairs" in s["problems"][0] and s["strokes"] == 1, s


def test_round_statistics_and_the_executor_refuse_the_same_strokes():
    """ "problems lists what the executor would refuse": asked of the executor itself (paintskill.plan_problems),
    with a copy of the rig's set-up of the evening and a round brush block."""
    import brushwork as BW
    import paintskill as PS
    assert RB.guard_mm() == BW.MAX_PRESS_M * 1000.0 == RB.MAX_PRESS_MM           # the kit's number is brush work's
    for base, source in rig_setups("evening"):
        setup = PS.check_setup(with_round_brush(base), source)
        assert PS.pan_axis_mm(setup) == list(PAN) and setup["press_mm"] == 5.0, source
        stroke = lambda pts, press=0.6: one_layer("red", [{"pts": pts, "press": press}])       # noqa: E731
        both_refuse = {
            "a press of 7.5 mm": stroke([[60, 50], [80, 50]], 1.5),
            "a press just over the limit": stroke([[60, 50], [80, 50]], 1.21),
            "a press below nothing": stroke([[60, 50], [80, 50]], -0.1),
            "0.7 mm of stroke": stroke([[60, 50], [60.7, 50]]),
            "one point": stroke([[60, 50]]),
            "points with a height": stroke([[60, 50, 3], [80, 50, 3]]),        # (read as pairs they would be three other points)
            "a point that is no number": stroke([[60, 50], [float("nan"), 50]]),
            "no points": stroke([]),
            "a stroke that leaves the canvas": stroke([[200, 50], [230, 50]]),
            "a band longer than its load": stroke([[20, 50], [120, 50]], 1.2),
        }
        for what, plan in both_refuse.items():
            assert PS.plan_problems(plan, setup), (what, source)
            assert RB.stats(plan)["n_problems"] >= 1 and RB.stats(plan, setup=setup)["n_problems"] >= 1, (what, source)
            assert RD.plan_stats(plan, setup)["n_problems"] >= 1, what
        both_take = {
            "a line": stroke([[60, 50], [100, 50]]),
            "a band at the limit": stroke([[60, 50], [100, 50]], 1.2),
            "a dab of 1 mm": stroke([[60, 50], [61, 50]]),
            "a stroke 5 mm inside the edge": stroke([[5, 5], [5, 60]]),
        }
        for what, plan in both_take.items():
            assert PS.plan_problems(plan, setup) == [], (what, source, PS.plan_problems(plan, setup))
            assert RB.stats(plan)["n_problems"] == 0 and RB.stats(plan, setup=setup)["n_problems"] == 0, (what, source)
        # where the kit is the stricter one, and says so (roundbrush.stats): a point nearer than 5 mm to the edge
        # of the canvas (marks land up to 9 mm from the plan), and a writing black's shorter load, which the
        # executor knows only from the set-up's recipe
        near = stroke([[3, 50], [40, 50]])
        assert PS.plan_problems(near, setup) == [] and RB.stats(near, setup=setup)["n_problems"] == 1
        letters = one_layer("ink2", [{"pts": [[20, 50], [90, 50]], "press": 0.6}])                # 70 mm, a load of ink2 is 60
        assert RB.stats(letters, setup=setup)["n_problems"] == 1
        if "load_mm" not in setup["paints"]["ink2"]:                                              # (the rig's set-up of that evening)
            assert PS.plan_problems(letters, setup) == []
        with_load = copy.deepcopy(setup)
        with_load["paints"]["ink2"]["load_mm"] = 60
        assert len(PS.plan_problems(letters, with_load)) == 1
        # another press_mm, another limit, the same on both sides: 6 mm are press 1.5 at 4 mm per unit of press
        shallow = dict(setup, press_mm=4.0)
        assert PS.plan_problems(stroke([[60, 50], [80, 50]], 1.5), shallow) == []
        assert RB.stats(stroke([[60, 50], [80, 50]], 1.5), setup=shallow)["n_problems"] == 0
        assert PS.plan_problems(stroke([[60, 50], [80, 50]], 1.6), shallow) and RB.stats(stroke([[60, 50], [80, 50]], 1.6), setup=shallow)["n_problems"] == 1
        # loads are counted alike, layer by layer, once the recipes of the writing blacks say what a load lasts
        with_load["paints"]["ink"]["load_mm"] = 110
        plan = KM.build(variant="next")
        kit = RB.stats(plan, setup=setup)
        assert [ly["loads"] for ly in PS.plan_budget(with_load, plan)["layers"]] == [ly["loads"] for ly in kit["layers"]] \
            == [4, 13, 9, 2, 6, 3, 1, 2, 8], source
        # ... and without that (the rig's set-up) the executor gives the letters 3 loads where the kit counts 8
        if not any("load_mm" in setup["paints"][name] for name in ("ink", "ink2")):
            assert [ly["loads"] for ly in PS.plan_budget(setup, plan)["layers"]][7:] == [1, 3]


def test_round_time_in_sample_and_on_calls_the_fit_has_not_seen():
    assert tuple(tuple(float(v) for v in c) for c in RB.MEASURED_CALLS) == tuple(tuple(float(v) for v in c) for c in CALLS)
    # IN-SAMPLE: the five calls the constants were fitted to. It shows that the fit was done right and nothing
    # about a new picture; calls 4 and 5 cannot miss (the blotting and the wash each rest on that one call).
    worst = 0.0
    for n, mm, loads, blot, wash, t in CALLS:
        est = RB.seconds(n, mm, loads, blot, wash)
        worst = max(worst, abs(est - t) / t)
        assert abs(est - t) / t <= 0.15, (n, mm, loads, est, t)
    assert worst < 0.06, worst                                             # the fit is in fact within 5 %
    # the constants are that fit: stroke and load from the calls that neither blot nor wash, nothing negative
    fit = RB.fit_seconds()
    assert np.allclose(fit, (4.04, 36.9, 14.33, 75.0), atol=0.02), fit
    assert np.allclose(fit, (RB.T_STROKE, RB.T_LOAD, RB.T_BLOT, RB.T_WASH), atol=0.35) and RB.V_MM_S == 20.0, fit
    assert np.allclose(RB.fit_seconds(CALLS[1:3]), (0.0, 41.58, 0.0, 0.0), atol=0.02)      # (free, those two give -17 s per stroke)
    # every part of the time costs something, and a load is the expensive one
    assert RB.T_STROKE > 0 and RB.T_LOAD > RB.T_STROKE and RB.T_BLOT > 0 and RB.T_WASH > RB.T_LOAD
    # LEAVE ONE OUT: fitted to four calls and asked for the fifth. This is what the estimate is worth for work
    # it has not seen: within 25 %, mostly too short
    loo = RB.leave_one_out()
    assert np.allclose(loo, (-0.203, 0.098, -0.097, -0.210, -0.219), atol=0.004), loo
    assert max(abs(e) for e in loo) <= RB.TIME_ERROR == 0.25 and max(abs(e) for e in loo) > 3 * worst, loo
    assert RB.stats(KM.build())["minutes_off_percent"] == 25 and RB.stats(rows_plan(0.6))["minutes_off_percent"] == 25
    # THE OTHER CALLS of that evening, which the fit never used (the run log of the rig): ten within 12 %; the
    # lettering in dense black is 26 % too long (its loads take no water: no trip to the dish at the far edge)
    unseen = [RB.seconds(n, mm, loads, blot, wash) / t - 1.0 for _name, n, mm, loads, blot, wash, t in RB.UNSEEN_CALLS]
    assert len(unseen) == 11 and "ink2" in RB.UNSEEN_CALLS[-1][0] and all(abs(e) <= 0.12 for e in unseen[:-1]), unseen
    assert 0.24 < unseen[-1] < 0.29 and max(abs(e) for e in unseen) <= RB.TIME_ERROR + 0.02, unseen
    assert not set(c[1:] for c in RB.UNSEEN_CALLS) & set(tuple(float(v) for v in c) for c in CALLS)       # none of them is one of the five
    if os.path.exists(RUN_LOG):                                            # the table is the log's (this computer only)
        logged = {}
        with open(RUN_LOG, encoding="utf-8") as f:
            for line in f:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue
                res, words = ev.get("result") or {}, (ev.get("args") or {}).get("paint_args") or []
                if ev.get("cmd") == "paint" and words[:1] == ["run"] and res.get("ok") and not (ev.get("args") or {}).get("sim"):
                    logged[str(ev.get("t"))[11:16]] = (res.get("strokes"), res.get("travel_mm"), res.get("loads"), res.get("seconds"))
        for name, n, mm, loads, _blot, _wash, t in RB.UNSEEN_CALLS:
            assert logged.get(name[:5]) == (n, mm, loads, t), (name, logged.get(name[:5]))
    # the layers of the picture as the statistics count them, against the calls that painted them
    st = RB.stats(KM.build())
    for li, t in ((1, 694.0), (2, 469.0), (4, 342.0), (7, 409.0)):
        got = st["layers"][li]["seconds"]
        assert abs(got - t) / t <= 0.15, (li, got, t)
    assert st["layers"][4]["loads"] == 6 and st["layers"][7]["loads"] == 6                    # as the executor counted
    # (the red wash and the glaze: 13 and 9 loads by the plan's own lengths; the executor of that evening, which
    #  added up its measured travel, needed 15 and 11)
    assert st["layers"][1]["loads"] == 13 and st["layers"][2]["loads"] == 9 and st["loads"] == 44
    assert st["layers"][1]["wash"] and st["layers"][4]["wash"] and st["layers"][5]["wash"] and not st["layers"][2]["wash"] \
        and not st["layers"][3]["wash"] and not st["layers"][6]["wash"] and not st["layers"][7]["wash"]
    assert st["washes"] == 3
    # the kit said 35.4 minutes for this plan and the clock 50, of which the robot's calls were about 42
    assert 38.5 <= st["minutes"] <= 50.0, st["minutes"]
    # the check pattern of 17:49 (five places: a line, a band, two lines along the robot's direction), the first
    # of the timed calls: 5 loads, 289 s
    strokes = []
    for u, v in ((48, 32), (153, 32), (48, 122), (153, 122), (100, 100)):
        strokes += [{"pts": [[u, v], [u + 24, v]], "press": 0.6}, {"pts": [[u, v + 8], [u + 24, v + 8]], "press": 1.2},
                    {"pts": [[u + 30, v - 8], [u + 30, v + 12]], "press": 0.6}, {"pts": [[u + 36, v + 12], [u + 36, v - 8]], "press": 0.6}]
    check = RB.stats(one_layer("red", strokes))
    assert check["strokes"] == 20 and check["travel_mm"] == 440 and check["loads"] == 5 and abs(check["layers"][0]["seconds"] - 289.0) / 289.0 < 0.15, check
    assert RB.needs_wash("red", "blue_pale") and RB.needs_wash("red", "red_pale") and not RB.needs_wash("red_pale", "red_mid")
    assert not RB.needs_wash(None, "red") and not RB.needs_wash("black", "ink") and not RB.needs_wash("ink", "ink2") and RB.needs_wash("ink2", "black")
    # (the table lists the red wash, call 3, without a wash of the brush, and the statistics above count one for
    #  that layer: the brush carried yellow_pale when it began. The note under roundbrush.T_STROKE says what the
    #  fit would be with it: the table is kept as it was listed, the same as the executor's)
    assert RB.needs_wash("yellow_pale", "red_pale") and CALLS[2][4] == 0
    with_wash = RB.fit_seconds(CALLS[:2] + ((16, 839, 15, 0, 1, 694),) + CALLS[3:])
    assert np.allclose(with_wash, (4.89, 33.75, 14.77, 78.11), atol=0.02), with_wash
    # the executor estimates from the same five calls by the same rule: the two estimates of the harness are one
    import paintskill as PS
    assert tuple(tuple(float(v) for v in c) for c in PS.EVENING_CALLS) == tuple(tuple(float(v) for v in c) for c in CALLS)
    theirs = PS.fit_times()
    assert np.allclose([theirs[k] for k in ("per_stroke_s", "per_load_s", "per_blot_s", "per_wash_s")], fit, atol=0.02), theirs
    assert abs(theirs["per_mm_s"] - 1.0 / RB.V_MM_S) < 1e-9


def test_round_render_is_deterministic():
    plan = KM.build()
    a = RB.render(plan)
    np.random.seed(12345)                                                  # the global generator has no say
    b = RB.render(plan)
    assert a.dtype == np.uint8 and np.array_equal(a, b)
    assert np.array_equal(RB.render(plan, as_painted=False), RB.render(plan, as_painted=False))
    assert not np.array_equal(RB.render(plan, seed=8), a)                  # the grain is the seed's
    d = RB.render(plan, as_painted=False)
    assert d.shape == a.shape and int((np.abs(d.astype(int) - a.astype(int)).max(axis=2) > 40).sum()) > 5000      # the two pictures differ
    upto = RB.render(plan, upto_layer=0)
    assert int((np.abs(upto.astype(int) - a.astype(int)).max(axis=2) > 40).sum()) > 5000        # layers 1 .. 7 are missing


def test_round_designed_picture_is_the_kits():
    if not os.path.exists(KIT):
        raise Skip(f"the kit of the evening is not on this computer ({KIT})")
    import importlib.util
    spec = importlib.util.spec_from_file_location("kit_of_the_evening", KIT)
    kit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kit)
    plan = KM.build()
    plan["layers"] = plan["layers"][:7]                                    # (the kit did not know "ink" as a black)
    mine, theirs = RB.render(plan, as_painted=False), kit.render(plan)
    assert mine.shape == theirs.shape and int(np.abs(mine.astype(int) - theirs.astype(int)).max()) <= 1
    for name in ("Plan", "Layer", "hatch_mask", "split", "arc", "render", "stats", "width_of", "base_paint", "tint_of"):
        assert hasattr(RB, name), name
    for name in ("line", "band", "dot", "fill_mask", "fill_polygon", "fill_ellipse"):
        assert hasattr(RB.Layer, name), name
    assert kit.stats(KM.build())["minutes"] == 35.4 and kit.stats(KM.build())["loads"] == 42


# ----------------------------------------------------------------------------- the Kaminarimon
def test_kaminarimon_build_is_the_plan_that_was_painted():
    plan = KM.build()
    got = tuple((digest(ly["strokes"]), len(ly["strokes"]), ly["paint"]) for ly in plan["layers"])
    assert got == KAMINARIMON, got
    assert KM.painted_as_recorded(plan) and plan["canvas_mm"] == [224, 153] and plan["pan_axis_mm"] == [118.0, -107.2]
    assert plan["brush"] == {"kind": "round", "wide_mm": 2.7, "broad_mm": 5.0, "load_mm": 140.0}
    assert sum(len(ly["strokes"]) for ly in plan["layers"]) == 70
    again = KM.retrace()
    assert [(digest(ly["strokes"]), len(ly["strokes"]), ly["paint"]) for ly in again["layers"]] == [RETRACE]
    assert KM.painted_as_recorded(again, KM.RETRACED)
    done = KM.done()
    assert len(done["layers"]) == 9 and done["layers"][-1]["paint"] == "ink2" and "ink2" in done["paints"]
    st = RB.stats(plan)
    assert st["n_problems"] == 0 and st["strokes"] == 70 and RB.stats(done)["n_problems"] == 0
    plan["layers"][0]["strokes"][0]["pts"][0][0] += 0.01                   # one hundredth of a millimetre is another plan
    assert not KM.painted_as_recorded(plan)
    for path, name in ((KM.PLAN_FILE, "kaminarimon_plan.json"), (KM.RETRACE_FILE, "kaminarimon_letters_touchup.json")):
        rig = os.path.join(RUNS, name)
        if os.path.exists(rig):                                            # the file the executor ran, where it is kept
            with open(path, "rb") as a, open(rig, "rb") as b:
                assert a.read() == b.read(), name


def test_kaminarimon_next_applies_the_lessons_and_is_paintable_as_built():
    painted, nxt = KM.build(), KM.build(variant="next")
    st = RB.stats(nxt)
    assert st["n_problems"] == 0 and RD.plan_stats(nxt)["n_problems"] == 0, st["problems"]
    # no stroke asks for more than a band's press: 6 mm below first touch at the set-up's press_mm 5.0, which is
    # all that brush work allows and the executor takes (test_paintings_... asks the executor itself)
    assert max(s_["press"] for ly in nxt["layers"] for s_ in ly["strokes"]) == 1.2 == RB.press_limit({"press_mm": 5.0})
    assert KM.PRESS_FAR == RB.P_BROAD and nxt == KM.build(variant="next", press_far=1.2)
    # the lettering: once, in dense black, its own layer at the very end; the same strokes in the same order
    last = nxt["layers"][-1]
    assert last["paint"] == "ink2" and "DRY" in last["name"] and last["strokes"] == painted["layers"][7]["strokes"][2:] and len(last["strokes"]) == 17
    assert nxt["layers"][-2]["paint"] == "ink" and nxt["layers"][-2]["strokes"] == painted["layers"][7]["strokes"][:2]
    assert [ly["paint"] for ly in nxt["layers"]].count("ink2") == 1 and len(nxt["layers"]) == 9
    assert all(RB.paint_use(s) <= 60.0 for s in last["strokes"])
    # nothing else is redesigned: the far light too is the stroke that was painted
    for li in range(0, 7):
        assert nxt["layers"][li] == painted["layers"][li], li
    glow_old = [s for s in painted["layers"][0]["strokes"] if min(p[1] for p in s["pts"]) > 130]
    assert len(glow_old) == 1 and glow_old[0]["press"] == 1.2 and painted["layers"][0]["strokes"].index(glow_old[0]) == 0
    # how the layers are to be run is in the plan: the bands with the contact plane 1.5 mm lower than the set-up
    # stores it (so the far light gets the press it lacked), the written layers without and with their own load
    assert nxt["run"] == [f"fd paint run PLAN --layer {i} --lower-mm 1.5" for i in range(7)] + \
        ["fd paint run PLAN --layer 7 --reload-mm 110", "fd paint run PLAN --layer 8 --reload-mm 60"]
    assert KM.run_commands(nxt, "/tmp/my plans/k.json")[8] == "fd paint run '/tmp/my plans/k.json' --layer 8 --reload-mm 60"
    assert KM.LOWER_MM == 1.5 and "run" not in painted and "open" not in painted
    # what is not solved is said, in the plan itself, with the command where there is one
    text = " ".join(nxt["open"])
    assert "blue_pale" in text and "black_pale" in text and "dry" in text and "thin line" in text and "Not proven" in text
    assert "`fd paint run PLAN --layer 0 --lower-mm 1.5`" in text and nxt["run"][0] in text and "--reload-mm" in text
    assert "NOT PAINTABLE" not in text and "press 1.5" not in text
    # the 1.5 mm written into the plan instead: more press for the far light, the same line end to end, cut where
    # no piece takes more than one load at that press. Not paintable today, and the plan and its statistics say so
    deep = KM.build(variant="next", press_far=1.5)
    glow_new = [s for s in deep["layers"][0]["strokes"] if min(p[1] for p in s["pts"]) > 130]
    assert len(glow_new) == 2 and all(s["press"] == 1.5 for s in glow_new) and KM.PRESS_FAR_MAX == 1.5
    assert math.dist(glow_new[0]["pts"][0], glow_old[0]["pts"][0]) < 0.05 and math.dist(glow_new[-1]["pts"][-1], glow_old[0]["pts"][-1]) < 0.05
    assert abs(sum(RB.length_mm(s["pts"]) for s in glow_new) - RB.length_mm(glow_old[0]["pts"])) < 1.5
    old_line = RB.resample(glow_old[0]["pts"], 0.1)
    for s_new in glow_new:
        for q in s_new["pts"]:
            assert float(np.min(np.linalg.norm(old_line - np.asarray(q), axis=1))) < 0.3, q
    assert all(RB.paint_use(s_new) <= 140.0 for s_new in glow_new) and RB.paint_use(dict(glow_old[0], press=1.5)) > 140.0
    assert [s for s in deep["layers"][0]["strokes"] if s not in glow_new] == [s for s in painted["layers"][0]["strokes"] if s not in glow_old]
    assert deep["layers"][1:] == nxt["layers"][1:]
    st = RB.stats(deep)
    assert st["n_problems"] == 2 and all("press 1.5" in p for p in st["problems"]) and RD.plan_stats(deep)["n_problems"] == 2, st["problems"]
    text = " ".join(deep["open"])
    assert "NOT PAINTABLE" in text and "press 1.5" in text and "MAX_PRESS_M" in text and "7.5 mm" in text
    assert deep["run"][0] == "fd paint run PLAN --layer 0" and deep["run"][1:] == nxt["run"][1:]      # (not lowered on top of that)
    assert nxt["brush"] == painted["brush"] and nxt["canvas_mm"] == painted["canvas_mm"] and nxt["pan_axis_mm"] == painted["pan_axis_mm"]
    assert KM.build(variant="next", pan_axis_mm=(131.0, -107.2))["pan_axis_mm"] == [131.0, -107.2]
    for bad in (lambda: KM.build(variant="other"), lambda: KM.build(pan_axis_mm=(131.0, -107.2)), lambda: KM.build(press_far=1.2),
                lambda: KM.build(variant="next", press_far=2.0), lambda: KM.build(variant="next", press_far=1.0)):
        try:
            bad()
        except ValueError:
            pass
        else:
            raise AssertionError("a wrong call was accepted")
    # in the picture as painted the lettering of the evening is grey on the lantern, the next one black
    sample = painted["layers"][7]["strokes"][9]                           # a bar of 田, in the middle of the lantern
    grey = strength_along(cover_of(painted, "ink", ppm=4.0), sample, ppm=4.0, lo=0.2, hi=0.8)
    black = strength_along(cover_of(nxt, "ink2", ppm=4.0), sample, ppm=4.0, lo=0.2, hi=0.8)
    assert black > 1.6 * grey, (grey, black)


def test_kaminarimon_main_writes_plan_and_previews():
    out = os.path.join(_TMP, "kaminarimon")
    plan = KM.main(out)
    with open(os.path.join(out, "kaminarimon_plan.json"), "rb") as a, open(KM.PLAN_FILE, "rb") as b:
        assert a.read() == b.read()                                        # byte for byte the plan that was painted
    assert plan == KM.build()
    pre, des = cv2.imread(os.path.join(out, "kaminarimon_preview.png")), cv2.imread(os.path.join(out, "kaminarimon_design.png"))
    assert pre.shape == (708, 992, 3) and des.shape == (612, 896, 3)
    assert np.array_equal(pre, RB.render(plan, margin_mm=12.0)) and np.array_equal(des, RB.render(plan, as_painted=False))
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        KM.main(out, "next")
    path = os.path.join(out, "kaminarimon_next_plan.json")
    with open(path) as f:
        assert json.load(f) == KM.build(variant="next")
    assert os.path.exists(os.path.join(out, "kaminarimon_next_preview.png"))
    # what it prints: the statistics, what is open, and the command of every layer with the file it wrote
    said = json.loads(said.getvalue())
    assert said["n_problems"] == 0 and said["open"] == KM.build(variant="next")["open"] and said["minutes_off_percent"] == 25
    assert said["run"] == KM.run_commands(KM.build(variant="next"), path) and len(said["run"]) == 9 and all(path in c for c in said["run"])
    said = io.StringIO()
    with contextlib.redirect_stdout(said):
        KM.main(out, "next", 1.5)                                          # the press of the far light in the plan: said to be refused
    assert json.loads(said.getvalue())["n_problems"] == 2 and "NOT PAINTABLE" in " ".join(json.loads(said.getvalue())["open"])


# ----------------------------------------------------------------------------- the pictures and the executor
def test_paintings_every_default_plan_passes_the_executors_plan_check():
    """What a painting module builds when it is asked for nothing special is a plan the executor takes: asked
    of paintskill.plan_problems, with copies of the rig's set-ups (the Kaminarimon: the evening's, plus the
    round brush block that `fd paint brush` stores; the Akabeko: the morning's, which has no block = flat)."""
    import paintskill as PS
    kaminarimon = (("build()", KM.build), ("build(variant='next')", lambda: KM.build(variant="next")),
                   ("retrace()", KM.retrace), ("done()", KM.done))
    akabeko = AK.build()
    for base, source in rig_setups("evening"):
        setup = PS.check_setup(with_round_brush(base), source)
        assert PS.pan_axis_mm(setup) == KM.build()["pan_axis_mm"] == list(PAN), source
        for name, make in kaminarimon:
            plan = make()
            assert PS.plan_problems(plan, setup) == [], (name, source, PS.plan_problems(plan, setup)[:3])
            assert RB.stats(plan, setup=setup)["n_problems"] == 0 and RD.plan_stats(plan)["n_problems"] == 0, (name, source)
            assert all(ly["paint"] in setup["paints"] for ly in plan["layers"]), (name, source)
        # the 1.5 mm of the far light written into the plan: the executor refuses it, and the statistics had said so
        deep = KM.build(variant="next", press_far=1.5)
        refused = PS.plan_problems(deep, setup)
        assert len(refused) == 2 and all("press 1.5" in p for p in refused), refused
        assert RB.stats(deep)["n_problems"] == RB.stats(deep, setup=setup)["n_problems"] == RD.plan_stats(deep)["n_problems"] == 2
        # a plan is for ONE brush and one canvas: the Akabeko is refused here, and without the brush block (the
        # rig's file as it stands: that means the flat brush) the Kaminarimon is
        assert PS.plan_problems(akabeko, setup)
        if "brush" not in base:
            assert any("round" in p and "flat" in p for p in PS.plan_problems(KM.build(), PS.check_setup(copy.deepcopy(base), source)))
    for base, source in rig_setups("morning"):
        setup = PS.check_setup(copy.deepcopy(base), source)
        assert "brush" not in setup and PS.pan_axis_mm(setup) == akabeko["pan_axis_mm"] == list(FLAT_PAN), source
        assert PS.plan_problems(akabeko, setup) == [], (source, PS.plan_problems(akabeko, setup)[:3])
        assert RD.plan_stats(akabeko, setup)["n_problems"] == 0 and akabeko["brush"]["kind"] == "flat"
        assert PS.plan_problems(KM.build(), setup)                         # the round brush's picture, with the flat brush on
    # the two written-out set-ups are the rig's, in what a plan check reads (where this computer has the rig's files)
    for which in ("evening", "morning"):
        both = rig_setups(which)
        if len(both) == 2:
            (mine, _), (rig, source) = both
            assert mine["canvas"] == rig["canvas"] and mine["press_mm"] == rig["press_mm"] and set(mine["paints"]) <= set(rig["paints"]), source
            assert {k: mine["contact"][k] for k in ("z0", "x0", "y0", "bx", "by")} == {k: rig["contact"][k] for k in ("z0", "x0", "y0", "bx", "by")}


def test_kaminarimon_next_run_commands_are_taken_by_the_executor():
    """A plan that names commands must name commands that work: every line of the next plan's "run" list, and
    the one its "open" list gives for the far light, is given to the executor with --dry (`fd --sim paint run`
    on a simulated arm; the scratch files hold a rig that is ready to paint: the thin brush held and measured,
    the evening's set-up with a round brush block). Checks and kinematics only: nothing is painted here."""
    PS, paintcli, setup = scratch_rig()
    nxt = KM.build(variant="next")
    path = os.path.join(_TMP, "kaminarimon_next_plan.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nxt, f)
    kit = RB.stats(nxt, setup=setup)
    assert kit["n_problems"] == 0 and len(nxt["run"]) == len(nxt["layers"]) == 9
    for li, cmd in enumerate(nxt["run"]):
        words = shlex.split(cmd)
        assert words[:6] == ["fd", "paint", "run", "PLAN", "--layer", str(li)], cmd
        r = fd_paint(paintcli, "run", path, *words[4:], "--dry")
        assert r["ok"] is True and r["stage"] == "planned", (cmd, r.get("stage"), r.get("reason"))
        # what the executor will do is what the statistics counted: the strokes, and the loads of that layer
        assert r["budget"]["strokes"] == len(nxt["layers"][li]["strokes"]) and r["budget"]["loads"] == kit["layers"][li]["loads"], (cmd, r["budget"])
        if "--lower-mm" in words:
            assert r["contact_lowered_mm"] == KM.LOWER_MM and f"--lower-mm {KM.LOWER_MM:g}" in r["next"], (cmd, r.get("next"))
    assert [("--lower-mm" in c) for c in nxt["run"]] == [True] * 7 + [False] * 2
    # the command in the "open" list is one of them
    assert f"`{nxt['run'][0]}`" in " ".join(nxt["open"])
    # ... as the operator types it, through fd itself (its run log goes to this process's scratch folder)
    try:
        import fd
    except SystemExit as e:             # (fd.py answers a module it cannot import with one JSON line and exit code 1)
        raise AssertionError(f"tools/cli/fd.py cannot be imported (exit code {e.code})") from None
    assert fd.RUNS == os.environ["FD_RUNS"] and fd.RUNS.startswith(_TMP)
    said = io.StringIO()
    with contextlib.redirect_stdout(said), contextlib.redirect_stderr(io.StringIO()):
        code = fd.main(["--sim", *shlex.split(nxt["run"][0].replace("PLAN", shlex.quote(path)))[1:], "--dry"])
    r = json.loads(said.getvalue())
    assert code == 0 and r["ok"] is True and r["stage"] == "planned" and r["sim"] is True and r["contact_lowered_mm"] == 1.5, r
    assert os.path.isdir(os.environ["FD_RUNS"])                            # (the record of that call)
    # 1.5 mm is all the executor gives a band: press and lowering together stay within what the bands of
    # 2026-11-13 ran at, and the plan's texts say so
    depth = PS.MAX_DEPTH_M                                                 # (the executor's name for that limit: it must exist)
    assert abs(depth * 1000.0 - (RB.MAX_PRESS_MM + KM.LOWER_MM)) < 1e-9
    r = fd_paint(paintcli, "run", path, "--layer", "0", "--lower-mm", "2", "--dry")
    assert r["ok"] is False and r["stage"] != "planned" and "7.5" in r["reason"], r
    assert "7.5 mm" in " ".join(nxt["open"]) and "7.5 mm" in KM.__doc__
    # the far light with its 1.5 mm in the plan (press 1.5): refused before anything moves, by `run` and by
    # `preview`, and the kit's statistics in that result say the same (they said "no problems" until the review)
    deep = os.path.join(_TMP, "kaminarimon_next_deep_plan.json")
    with open(deep, "w", encoding="utf-8") as f:
        json.dump(KM.build(variant="next", press_far=1.5), f)
    r = fd_paint(paintcli, "run", deep, "--layer", "0", "--dry")
    assert r["ok"] is False and r["stage"] == "plan" and "press 1.5" in r["reason"], r
    r = fd_paint(paintcli, "preview", deep, "--out", os.path.join(_TMP, "deep.png"))
    assert r["ok"] is False and r["stage"] == "plan" and r["design_stats"]["n_problems"] == 2 == len(r["problems"]), r
    r = fd_paint(paintcli, "preview", path, "--out", os.path.join(_TMP, "next.png"))
    assert r["ok"] is True and r["design_stats"]["n_problems"] == 0 and r["loads"] >= 40, (r.get("stage"), r.get("reason"))
    # nothing of this wrote the state file of the scratch rig (dry runs and previews), and the tool file is as it was
    assert not os.path.exists(PS.state_path(True)) and not os.path.exists(PS.state_path(False))
    with open(TOOL_PATH, encoding="utf-8") as f:
        assert json.load(f) == {**THIN_BRUSH, **READY}


def test_texts_say_what_was_measured():
    """Three numbers the module texts had wrong until the review of 2026-11-13 (texts only: what is computed
    did not change, and the tests above pin that down)."""
    def source(mod):
        with open(mod.__file__, encoding="utf-8") as f:
            return f.read()
    # the white disc on the Akabeko's rump came out about 23 x 16 mm, half its area (not "18 x 12")
    assert "18 x 12" not in source(AK) and "23 x 16 mm, half its area" in source(AK) and "23 x 16 mm" in source(SH)
    # the flat brush of the morning measured about 9 mm across its bristles (not "12 mm")
    for mod in (RD, RB, SH, AK, KM):
        assert "12 mm flat" not in source(mod) + mod.__doc__ and "12 mm brush" not in source(mod) + mod.__doc__, mod.__name__
    assert "bristles about 9 mm wide" in RD.__doc__ and "bristles about 9 mm wide" in RB.__doc__ and RD.BRUSH_WIDE_MM == 9.0
    # the next Kaminarimon is paintable, and its text says how the far light gets its press
    assert "NOT YET PAINTABLE" not in KM.__doc__ and "fd paint run PLAN --layer 0 --lower-mm 1.5" in KM.__doc__
    # the time of the statistics is said to be in-sample where it is given
    assert "IN-SAMPLE" in source(RB) and "leave_one_out" in RB.stats.__doc__ + RB.__doc__
    # what these texts name of the executor exists there under that name (they are written by other hands)
    import re
    import brushwork
    import paintskill
    named = {(mod, name) for m in (RD, RB, SH, AK, KM) for mod, name in re.findall(r"\b(paintskill|brushwork)\.([A-Za-z_]\w*)", source(m))}
    named.discard(("paintskill", "py"))
    assert ("paintskill", "plan_problems") in named and ("brushwork", "MAX_PRESS_M") in named and len(named) >= 10, named
    gone = sorted(f"{mod}.{name}" for mod, name in named if not hasattr({"paintskill": paintskill, "brushwork": brushwork}[mod], name))
    assert not gone, gone


# ----------------------------------------------------------------------------- flat brush
FLAT_PAN = (131.0, -107.2)             # the pan axis of the morning's canvas


def old_stroke_mask(pts, pan_axis, shape, px_per_mm, wide_mm=9.0, thin_mm=2.5, press=1.0):
    """render.stroke_mask as it was on 2026-11-13 14:00, before the brush block (copied from that version)."""
    mask = np.zeros(shape[:2], np.uint8)
    w = wide_mm * float(np.clip(0.6 + 0.4 * press, 0.7, 1.15))
    half = max(0.2, (w - thin_mm) / 2.0)
    s = RD.resample(pts, 0.5)
    for a, b in zip(s[:-1], s[1:]):
        wa, wb = RD.wide_axis(a, pan_axis) * half, RD.wide_axis(b, pan_axis) * half
        cv2.fillConvexPoly(mask, np.round(np.array([a - wa, a + wa, b + wb, b - wb]) * px_per_mm).astype(np.int32), 1)
    k = max(1, int(round(thin_mm * px_per_mm)) | 1)
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))


def old_render(plan, px_per_mm=4.0):
    """render.render as it was that day (the flat brush, 9 mm, a load of 180 mm fading to 0.55)."""
    w_mm, h_mm = plan["canvas_mm"]
    hh, ww = int(round(h_mm * px_per_mm)), int(round(w_mm * px_per_mm))
    img = np.empty((hh, ww, 3), np.float32)
    img[:] = np.array((244, 242, 236)[::-1], np.float32) / 255.0
    for layer in plan["layers"]:
        paint = plan["paints"][layer["paint"]]
        col = np.array(paint["rgb"][::-1], np.float32) / 255.0
        used = 0.0
        for st in layer["strokes"]:
            ln = RD.length_mm(st["pts"])
            if used + ln > 180.0 and used > 0.0:
                used = 0.0
            m = old_stroke_mask(st["pts"], plan["pan_axis_mm"], (hh, ww), px_per_mm, press=float(st.get("press", 1.0)))
            a = float(paint.get("opacity", 0.85)) * (1.0 - (1.0 - 0.55) * min(1.0, (used + 0.5 * ln) / 180.0))
            img[m > 0] *= (1.0 - a * (1.0 - col))
            used += ln
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


def flat_width(mask, at_px, horizontal: bool, ppm=4.0) -> float:
    """Extent in mm of a mask through a pixel, along a row (horizontal) or a column."""
    line = mask[int(at_px[1])] if horizontal else mask[:, int(at_px[0])]
    return float(line.sum()) / ppm


def test_flat_band_widths_with_and_without_the_spread():
    shape, ppm = (612, 896), 4.0
    radial = [(131.0, 40.0), (131.0, 100.0)]                               # straight away from the robot
    side = [(100.0, 70.0), (160.0, 70.0)]                                  # sideways under the pan axis
    bare_r, bare_s = RD.stroke_mask(radial, FLAT_PAN, shape, ppm), RD.stroke_mask(side, FLAT_PAN, shape, ppm)
    assert abs(flat_width(bare_r, (524, 280), True) - 9.0) < 0.5           # the band of a radial stroke
    assert abs(flat_width(bare_s, (520, 280), False) - 2.5) < 0.5          # a sideways stroke: the brush edge-on
    s = RD.BRUSH_SPREAD_MM
    assert 1.5 <= s <= 2.0 and RD.FLAT_BRUSH == {"kind": "flat", "wide_mm": 9.0, "thin_mm": 2.5, "spread_mm": s, "load_mm": 180.0}
    wet_r = RD.stroke_mask(radial, FLAT_PAN, shape, ppm, spread_mm=s)
    wet_s = RD.stroke_mask(side, FLAT_PAN, shape, ppm, spread_mm=s)
    assert abs(flat_width(wet_r, (524, 280), True) - 12.5) < 0.5           # the 12.5 mm band of the real sheet
    assert abs(flat_width(wet_s, (520, 280), False) - 6.0) < 0.5
    assert abs((flat_width(wet_r, (524, 280), False) - flat_width(bare_r, (524, 280), False)) - 2 * s) < 0.5      # every side
    # between the two directions the band is between; more press, a little wider
    diag = RD.stroke_mask([(100.0, 40.0), (160.0, 100.0)], FLAT_PAN, shape, ppm)
    area = float(diag.sum()) / 16.0 / RD.length_mm([(100.0, 40.0), (160.0, 100.0)])
    assert 2.5 + 0.5 < area < 9.0 + 0.5, area
    assert RD.stroke_mask(radial, FLAT_PAN, shape, ppm, press=1.5).sum() > bare_r.sum() > RD.stroke_mask(radial, FLAT_PAN, shape, ppm, press=0.5).sum()


def test_flat_brush_block_and_dispatch():
    assert RD.brush_of({}) == RD.FLAT_BRUSH and RD.brush_of({"brush": None}) == RD.FLAT_BRUSH
    assert RD.brush_of({"brush": {"kind": "flat", "spread_mm": 0}})["spread_mm"] == 0.0
    assert RD.brush_of({"brush": {"kind": "flat", "spread_mm": 0}})["wide_mm"] == 9.0
    r = RD.brush_of(KM.build())
    assert r["kind"] == "round" and r["wide_mm"] == 2.7 and r["broad_mm"] == 5.0 and r["load_mm"] == 140.0
    # the block of a painting set-up names a round brush's widths under "modes"
    r = RD.brush_of({"brush": {"kind": "round", "load_mm": 120, "modes": {"fine": {"press": 0.6, "width_mm": 2.4}, "broad": {"press": 1.2, "width_mm": 4.6}}}})
    assert (r["wide_mm"], r["broad_mm"], r["load_mm"]) == (2.4, 4.6, 120.0)
    # a plan file of the evening that has no block (the touch-up as it was run) is a flat plan by the rule, and
    # still gives a picture
    touch = RD.load_plan(KM.RETRACE_FILE)
    assert "brush" not in touch and RD.brush_of(touch) == RD.FLAT_BRUSH and RD.render(touch).shape == (612, 896, 3)
    assert RD.plan_stats(touch)["strokes"] == 13 and RD.plan_stats(touch)["n_problems"] == 0
    # what the painting module returns for it says which brush painted it: the same strokes, drawn as that brush
    again = KM.retrace()
    assert again["brush"] == KM.build()["brush"] and {k: v for k, v in again.items() if k != "brush"} == touch
    assert RD.brush_of(again)["kind"] == "round" and np.array_equal(RD.render(again), RB.render(again))
    assert not np.array_equal(RD.render(again), RD.render(touch))
    try:
        RD.brush_of({"brush": {"kind": "fan"}})
    except ValueError as e:
        assert "fan" in str(e)
    else:
        raise AssertionError("an unknown kind of brush was accepted")
    assert RD.LOAD_MM == 180.0 and RD.length_mm([(0, 0), (3, 4), (3, 14)]) == 15.0
    path = os.path.join(_TMP, "plan.json")
    with open(path, "w") as f:
        json.dump(AK.build(), f)
    assert RD.load_plan(path) == json.loads(json.dumps(AK.build()))
    # a flat plan with a shorter load reloads more often
    plan = AK.build()
    short = json.loads(json.dumps(plan))
    short["brush"]["load_mm"] = 90.0
    for ly in short["layers"]:
        ly["strokes"] = [s for s in ly["strokes"] if RD.length_mm(s["pts"]) <= 90.0]
    same = json.loads(json.dumps(short))
    same["brush"]["load_mm"] = 180.0
    assert RD.plan_stats(short)["loads"] > RD.plan_stats(same)["loads"] and RD.plan_stats(short)["n_problems"] == 0


def test_flat_valid_centres_keep_the_contact_line_inside():
    ppm = 4.0
    c = SH.Canvas((224, 153), ppm)
    c.rounded_box(50, 35, 190, 115, 8)
    c.circle((120, 75), 14, on=False)
    ok = SH.valid_centres(c.mask, FLAT_PAN, ppm)
    assert ok.any() and not ok[~(c.mask > 0)].any()
    grown = cv2.dilate(c.mask, np.ones((3, 3), np.uint8))                  # one pixel of slack
    ys, xs = np.nonzero(ok)
    for i in range(0, len(xs), max(1, len(xs) // 400)):
        uv = np.array([(xs[i] + 0.5) / ppm, (ys[i] + 0.5) / ppm])
        dab = RD.stroke_mask([uv, uv + RD.radial(uv, FLAT_PAN) * 0.01], FLAT_PAN, c.mask.shape, ppm)       # the brush standing there
        assert not (dab & (1 - grown)).any(), uv
    # the contact line is 9 mm across the ray: no centre in a strip of 6 mm that runs along the ray, some in 12 mm
    for width, any_ok in ((6.0, False), (12.0, True)):
        s = SH.Canvas((224, 153), ppm)
        s.rounded_box(131 - width / 2, 30, 131 + width / 2, 120, 0)
        assert bool(SH.valid_centres(s.mask, FLAT_PAN, ppm).any()) == any_ok, width
    # ... and a strip of 6 mm across the ray has room for it (the brush is 2.5 mm thick)
    s = SH.Canvas((224, 153), ppm)
    s.rounded_box(100, 70, 162, 76, 0)
    assert SH.valid_centres(s.mask, FLAT_PAN, ppm).any()


def test_flat_contour_and_fill_cover_a_shape():
    ppm = 4.0
    c = SH.Canvas((224, 153), ppm)
    c.rounded_box(60, 40, 170, 110, 10)
    strokes = SH.contour_strokes(c.mask, FLAT_PAN, ppm, margin_mm=0.0) + SH.fill_strokes(c.mask, FLAT_PAN, ppm)
    assert len(strokes) >= 3 and all(RD.length_mm(s["pts"]) <= RD.LOAD_MM for s in strokes)
    painted = SH.painted_mask(strokes, FLAT_PAN, (224, 153), ppm)
    inside = (c.mask > 0).astype(np.uint8)
    assert float((painted & inside).sum()) / float(inside.sum()) > 0.97
    assert int((painted & (1 - cv2.dilate(inside, np.ones((5, 5), np.uint8)))).sum()) == 0      # nothing beyond half a millimetre
    plan = {"canvas_mm": [224, 153], "pan_axis_mm": list(FLAT_PAN), "paints": {"red": {"rgb": [214, 38, 44], "opacity": 0.9}},
            "layers": [{"name": "box", "paint": "red", "strokes": strokes}]}
    assert RD.plan_stats(plan)["n_problems"] == 0
    # on the real sheet the paint went further: with the spread the same strokes cross the edge by about it
    wet = SH.painted_mask(strokes, FLAT_PAN, (224, 153), ppm, spread_mm=RD.BRUSH_SPREAD_MM)
    over = cv2.distanceTransform(1 - inside, cv2.DIST_L2, 5)[wet > 0].max() / ppm
    assert 1.0 < over < 3.0, over
    # the fill alone leaves what the brush cannot reach without crossing the edge
    fill_only = SH.painted_mask(SH.fill_strokes(c.mask, FLAT_PAN, ppm), FLAT_PAN, (224, 153), ppm)
    assert float((fill_only & inside).sum()) < float((painted & inside).sum())


def test_flat_old_plans_render_as_before_but_for_the_spread():
    shape, ppm = (612, 896), 4.0
    strokes = [{"pts": [[131.0, 40.0], [131.0, 100.0]], "press": 1.0}, {"pts": [[40.0, 60.0], [100.0, 62.0], [100.0, 110.0]], "press": 0.5},
               {"pts": [[150.0, 30.0], [200.0, 120.0], [160.0, 120.0], [150.0, 60.0]], "press": 1.5}, {"pts": [[20.0, 130.0], [110.0, 130.0]], "press": 1.1}]
    for st in strokes:      # the stroke itself is what it was
        assert np.array_equal(RD.stroke_mask(st["pts"], FLAT_PAN, shape, ppm, press=st["press"]),
                              old_stroke_mask(st["pts"], FLAT_PAN, shape, ppm, press=st["press"]))
    old = {"canvas_mm": [224, 153], "pan_axis_mm": list(FLAT_PAN), "paints": {"red": {"rgb": [214, 38, 44], "opacity": 0.9}, "black": {"rgb": [28, 28, 34]}},
           "layers": [{"name": "a", "paint": "red", "strokes": strokes}, {"name": "b", "paint": "black", "strokes": strokes[:1]}]}
    before = old_render(old)
    bare = dict(old, brush={"kind": "flat", "spread_mm": 0.0})
    assert np.array_equal(RD.render(bare), before)                         # with the spread set to 0: pixel for pixel
    now = RD.render(old)                                                   # no block: the flat brush of the morning
    assert np.array_equal(now, RD.render(dict(old, brush=dict(RD.FLAT_BRUSH))))
    assert now.shape == before.shape == (612, 896, 3)
    paper = np.array((244, 242, 236)[::-1], np.uint8)
    was, grown = (before != paper).any(axis=2).astype(np.uint8), (now != paper).any(axis=2).astype(np.uint8)
    assert not (was & (1 - grown)).any()                                   # nothing that was painted is bare now
    s = RD.BRUSH_SPREAD_MM

    def disc(mm):
        k = 2 * int(round(mm * ppm)) + 1
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    assert not (grown & (1 - cv2.dilate(was, disc(s + 0.5)))).any()        # and the paint is the stated spread further out:
    assert not (cv2.dilate(was, disc(s - 0.5)) & (1 - grown)).any()        # not more than it, not less
    assert RD.plan_stats(old) == RD.plan_stats(bare)                       # the numbers do not know the spread
    stats = RD.plan_stats(old)
    assert stats["strokes"] == 5 and stats["n_problems"] == 1 and "longer than one load" in stats["problems"][0]   # (the 190 mm polyline)
    for key in ("strokes", "travel_mm", "loads", "layers", "minutes", "problems", "n_problems"):
        assert key in stats
    _img, cover = RD.render(old, return_coverage=True)
    assert set(cover) == {"red", "black"} and cover["red"].shape == (612, 896) and float(cover["red"].max()) > 0.5
    assert np.array_equal(RD.render(old, upto_layer=0), RD.render(dict(old, layers=old["layers"][:1])))


def test_flat_preview_of_the_morning_is_reproduced():
    plan_file, png = os.path.join(RUNS, "akabeko_plan.json"), os.path.join(RUNS, "akabeko_preview.png")
    if not (os.path.exists(plan_file) and os.path.exists(png)):
        raise Skip("the plan and the preview of 2026-11-13 are not on this computer (runs/20261113_paint/)")
    plan = RD.load_plan(plan_file)
    assert "brush" not in plan
    assert np.array_equal(RD.render(dict(plan, brush={"kind": "flat", "spread_mm": 0.0})), cv2.imread(png))
    assert int((RD.render(plan) != cv2.imread(png)).any(axis=2).sum()) > 20000


def test_flat_design_report_flags_a_small_hole_and_passes_a_large_one():
    ppm = 4.0

    def box_with_hole(d):
        c = SH.Canvas((224, 153), ppm)
        c.rounded_box(40, 30, 180, 120, 5)
        c.circle((110, 75), d / 2.0, on=False)
        return SH.design_report(c.mask, FLAT_PAN, ppm)
    small, large = box_with_hole(10.0), box_with_hole(30.0)
    assert not small["ok"] and small["closing"] == 1 and len(small["features"]) == 1
    f = small["features"][0]
    assert f["kind"] == "hole" and f["closes"] and abs(f["narrow_mm"] - 10.0) < 0.6 and f["left_part"] < 0.1 and np.allclose(f["at_mm"], (110, 75), atol=0.5)
    assert large["ok"] and large["closing"] == 0 and len(large["features"]) == 1
    f = large["features"][0]
    # across the ray it loses 1.5 mm a side, along it 5 mm a side (the Akabeko's disc)
    assert not f["closes"] and abs(f["size_mm"][0] - 30.0) < 0.6 and abs(f["left_mm"][0] - 27.0) < 1.0 and abs(f["left_mm"][1] - 20.0) < 1.0, f
    assert 0.5 < f["left_part"] < 0.7
    assert SH.HOLE_LOSS_ACROSS_MM == 1.5 and SH.HOLE_LOSS_ALONG_MM == 5.0 and SH.MIN_LEFT_MM == 5.0
    # where a hole just stays open: left = its size less 10 mm along the ray, open from 5 mm of it on
    assert box_with_hole(14.0)["features"][0]["closes"] and not box_with_hole(16.5)["features"][0]["closes"]
    # a shape without white in it has nothing to report; the rounding of an inner corner is not a notch
    c = SH.Canvas((224, 153), ppm)
    c.polygon([(20, 30), (200, 30), (200, 60), (60, 60), (60, 120), (20, 120)])
    assert SH.design_report(c.mask, FLAT_PAN, ppm) == {"ok": True, "closing": 0, "features": []}
    # notches: a white line of 2.5 mm between two parts closes, a gap of 12 mm across the rays stays
    for gap, closes in ((2.5, True), (12.0, False)):
        c = SH.Canvas((224, 153), ppm)
        c.rounded_box(30, 30, 125, 120, 3)
        c.rounded_box(125 + gap, 30, 215, 120, 3)
        rep = SH.design_report(c.mask, FLAT_PAN, ppm)
        notch = [f for f in rep["features"] if f["kind"] == "notch"]
        assert len(notch) == 1 and notch[0]["closes"] == closes and abs(notch[0]["narrow_mm"] - gap) < 0.6, (gap, rep)
    # the same gap ALONG the rays (a horizontal slit under the robot) loses 5 mm a side and closes
    c = SH.Canvas((224, 153), ppm)
    c.rounded_box(60, 30, 200, 70, 3)
    c.rounded_box(60, 82, 200, 125, 3)
    rep = SH.design_report(c.mask, FLAT_PAN, ppm)
    assert [f["closes"] for f in rep["features"] if f["kind"] == "notch"] == [True], rep
    # other losses: a brush that takes nothing leaves everything
    c = SH.Canvas((224, 153), ppm)
    c.rounded_box(40, 30, 180, 120, 5)
    c.circle((110, 75), 5.0, on=False)
    gentle = SH.design_report(c.mask, FLAT_PAN, ppm, across_mm=0.5, along_mm=0.5)
    assert gentle["ok"] and abs(gentle["features"][0]["left_mm"][0] - 9.0) < 0.8


def test_akabeko_has_no_problems_and_its_white_is_flagged_as_it_came_out():
    plan = AK.build()
    assert RD.plan_stats(plan)["n_problems"] == 0 and RD.plan_stats(plan)["strokes"] == 34
    assert plan["brush"] == RD.FLAT_BRUSH and plan["brush"] is not RD.FLAT_BRUSH
    # it was not redesigned: the strokes are those the executor recorded as painted on 2026-11-13
    assert tuple((digest(ly["strokes"]), len(ly["strokes"]), ly["paint"]) for ly in plan["layers"]) == AKABEKO
    rep = AK.report()
    at = lambda u, v: min(rep["features"], key=lambda f: math.hypot(f["at_mm"][0] - u, f["at_mm"][1] - v))      # noqa: E731
    disc, eye, flank = at(*AK.RUMP_SPOT[0]), at(*AK.EYE[0]), at(*AK.FLANK_SPOT[0])
    # the 26 mm disc on the rump survived at about 23 x 16 mm, half its area
    assert disc["kind"] == "hole" and not disc["closes"] and abs(disc["narrow_mm"] - 26.0) < 1.0
    assert abs(disc["left_mm"][0] - 23.0) < 1.5 and abs(disc["left_mm"][1] - 16.0) < 1.5 and 0.4 < disc["left_part"] < 0.65, disc
    # the eye (17 x 13 mm) closed almost completely
    assert eye["kind"] == "hole" and eye["closes"] and eye["left_part"] < 0.25, eye
    # the 18 mm disc on the flank stayed open, the two stripes on the back and the crescent on the cheek did not,
    # the gap for the collar did
    assert flank["kind"] == "hole" and not flank["closes"]
    stripes = [at(98, 50), at(106, 50)]
    assert stripes[0] is not stripes[1] and all(f["kind"] == "notch" and f["closes"] for f in stripes), stripes
    cheek, gap = at(172, 85), at(133.5, 70)
    assert cheek["kind"] == "notch" and cheek["closes"] and gap["kind"] == "notch" and not gap["closes"], (cheek, gap)
    assert not rep["ok"] and rep["closing"] == 4 and len(rep["features"]) == 7, rep
    out = os.path.join(_TMP, "akabeko")
    assert AK.main(out) == plan
    with open(os.path.join(out, "akabeko_plan.json")) as f:
        assert json.load(f) == json.loads(json.dumps(plan))
    assert np.array_equal(cv2.imread(os.path.join(out, "akabeko_preview.png")), RD.render(plan))


def synthetic_photo(mask, shift_mm, ppm, rgb, noise=4.0, seed=3):
    """A top view of a sheet on which `mask` was painted in one colour, moved by shift_mm (camera noise, soft edges)."""
    rng = np.random.default_rng(seed)
    M = np.float32([[1, 0, shift_mm[0] * ppm], [0, 1, shift_mm[1] * ppm]])
    m = cv2.warpAffine(mask.astype(np.float32), M, (mask.shape[1], mask.shape[0]))
    m = cv2.GaussianBlur(m, (0, 0), 0.8)[..., None]
    paper = np.array([190.0, 184.0, 180.0], np.float32)
    img = paper * (1.0 - m) + np.array(rgb[::-1], np.float32) * m
    return np.clip(img + rng.normal(0.0, noise, img.shape), 0, 255).astype(np.uint8)


def test_fit_to_photo_recovers_a_known_shift_and_spread():
    plan = AK.build()
    ppm, margin = 4.0, 12.0
    bare = RD.layer_mask(plan, 0, ppm, margin)
    assert bare.shape == (708, 992) and bare.sum() > 100000
    k = int(round(1.5 * ppm))
    grown = cv2.dilate(bare, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    fit = RD.fit_to_photo(plan, synthetic_photo(grown, (-5.0, 3.0), ppm, (200, 30, 40)))
    assert np.allclose(fit["shift_mm"], (-5.0, 3.0), atol=0.26) and abs(fit["spread_mm"] - 1.5) <= 0.25, fit
    assert fit["iou"] > 0.95 and fit["iou_unfitted"] < fit["iou"] - 0.02 and fit["iou_as_planned"] < fit["iou_unfitted"] - 0.05, fit
    assert fit["px_per_mm"] == 4.0 and len(fit["scan"]) == 12
    # a brush that paints exactly what is modelled: no spread, the shift alone
    fit = RD.fit_to_photo(plan, synthetic_photo(bare, (4.0, -6.0), ppm, (200, 30, 40), seed=5))
    assert np.allclose(fit["shift_mm"], (4.0, -6.0), atol=0.26) and fit["spread_mm"] <= 0.25 and fit["iou"] > 0.95, fit
    assert abs(fit["iou_unfitted"] - fit["iou"]) < 0.03
    # the second layer among the first: only the pixels of its paint
    both = synthetic_photo(grown, (-5.0, 3.0), ppm, (200, 30, 40)).astype(np.float32)
    black = cv2.warpAffine(RD.layer_mask(plan, 1, ppm, margin).astype(np.float32), np.float32([[1, 0, -20], [0, 1, 12]]), (992, 708))[..., None]
    both = np.clip(both * (1.0 - black) + np.array([34.0, 28.0, 28.0], np.float32) * black, 0, 255).astype(np.uint8)
    fit = RD.fit_to_photo(plan, both, layer=1, paint_only=True)
    assert np.allclose(fit["shift_mm"], (-5.0, 3.0), atol=0.6) and fit["iou"] > 0.8, fit
    # a round plan: its layers as designed
    rp = KM.build()
    red = RD.layer_mask(rp, 1, ppm, margin)
    fit = RD.fit_to_photo(rp, synthetic_photo(red, RB.LANDING_MM, ppm, (226, 52, 38), seed=7), layer=1)
    assert np.allclose(fit["shift_mm"], RB.LANDING_MM, atol=0.26) and fit["spread_mm"] <= 0.25, fit
    # a picture that is not a top view of this canvas is refused
    try:
        RD.fit_to_photo(plan, np.zeros((600, 992, 3), np.uint8))
    except ValueError as e:
        assert "top view" in str(e)
    else:
        raise AssertionError("a picture of the wrong size was accepted")


def test_fit_to_photo_gives_the_numbers_of_the_morning():
    plan_file = os.path.join(RUNS, "akabeko_plan.json")
    if not (os.path.exists(STAGE_RED) and os.path.exists(plan_file)):
        raise Skip("the photo of the red layer (video/work_paint/assets/stage_red.jpg) or the plan of that morning "
                   "(runs/20261113_paint/akabeko_plan.json) is not on this computer")
    fit = RD.fit_to_photo(RD.load_plan(plan_file), cv2.imread(STAGE_RED))
    # "the red matched best grown by 1.5 to 2.0 mm on every side and shifted by about (-4.5, -8.5) mm
    #  (IoU 0.834 against 0.794 without)"
    assert 1.5 <= fit["spread_mm"] <= 2.0, fit
    assert abs(fit["shift_mm"][0] + 4.5) <= 0.5 and abs(fit["shift_mm"][1] + 8.5) <= 0.5, fit
    assert abs(fit["iou_unfitted"] - 0.794) < 0.005 and abs(fit["iou"] - 0.834) < 0.012 and fit["iou_as_planned"] < 0.65, fit
    assert abs(fit["spread_mm"] - RD.BRUSH_SPREAD_MM) <= 0.25              # the constant of the model is this measurement


def test_zz_the_rig_files_are_what_they_were():
    assert _rig_files() == _RIG_BEFORE, "a test created, changed or removed a set-up, state or config file of the rig"
    for var in ("FD_TOOL", "FD_PAINT_SETUP", "FD_PAINT_STATE", "FD_PAINT_OUT", "FD_RUNS", "FD_CAMERA_MODEL", "TAZUNA_HW_LOCK"):
        assert os.environ[var].startswith(_TMP), var                       # every file a command writes is a scratch file
    assert os.path.exists(LOCK_PATH), "the scratch hardware lock of this process is gone"
    assert _TRIPPED == [], _TRIPPED                                        # nothing reached for the servo bus or the camera


if __name__ == "__main__":
    import shutil
    import traceback
    only = sys.argv[1:]
    fails = skipped = 0
    n = 0
    t_all = time.time()
    try:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn) and (not only or any(o in name for o in only)):
                n += 1
                t0 = time.time()
                try:
                    with contextlib.redirect_stdout(io.StringIO()):       # (main() of a painting prints its statistics)
                        fn()
                    print(f"PASS {name} ({time.time() - t0:.1f} s)")
                except Skip as e:
                    skipped += 1
                    print(f"SKIP {name}: {e}")
                except Exception:
                    fails += 1
                    print("FAIL", name)
                    traceback.print_exc()
    finally:
        shutil.rmtree(_TMP, ignore_errors=True)
    print(f"tests: {n}  passed: {n - fails - skipped}  skipped: {skipped}  failures: {fails}  ({time.time() - t_all:.0f} s)")
    sys.exit(1 if fails else 0)
