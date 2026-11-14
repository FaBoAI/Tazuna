"""`fd paint ...`: painting with the brush that is taped to the gripper (paintskill.py, brushwork.py).

  fd paint status                      the set-up, the brush, what the brush carries (files only); `next`: the first
                                       step that is still missing (jaw, tip, brush block, contact plane)
  fd paint brush [--kind round ...]    which brush is on: show it, or store it in the set-up (file only)
  fd paint brush --carries none|PAINT  say what the brush carries (a brush rinsed or swapped by hand; file only)
  fd paint check                       no motion: is every well, dish, the blotting strip and the canvas in reach?
  fd paint look                        to the photo pose, one photo
  fd paint locate [--dry]              photo; find the paint blobs again (the palette gets pushed or turned); stored
                                       only when the photo shows the palette and the wells agree on ONE move of it; no
                                       well is put where the photo shows no paint, or paint of another colour than the
                                       well showed when it was last written (it keeps its place, "located": false, and
                                       nothing loads from it); the places before are kept in the file
  fd paint test --do ACTION ...        test strokes on scrap paper (`fd paint test --help`: how a new brush is measured);
                                       on the canvas of the set-up only with --scrap-laid, after a look at the sheet
  fd paint contact --scrap-laid [--save]   measure the paper height: ladders of strokes on a SCRAP SHEET laid over
                                       the canvas, two photos, the plane; a record beside the photos says with which brush
  fd paint contact --read B.jpg A.jpg [--save] [--dry]   no arm: read two saved photos again (brush state: their record)
  fd paint contact --adopt             no motion: record the brush as it is for the plane the set-up already has
  fd paint preview PLAN [--out FILE]   no motion: the simulated picture, the time it takes, every check of `run --dry`
  fd paint run PLAN --layer N [--start K] [--max-min 8] [--lower-mm D] [--dry]   paint one layer of a plan (resumable)
  fd paint rinse | rest                rest: fold the arm away off the picture (an arm that already lies folded and limp
                                       there is left as it lies; a tip that was never measured: the highest fold)

Every command answers with ONE result (a dict; `fd` prints it as JSON): ok, stage, reason and next (when it did
not do its job: `next` is the next command, one line), say (Japanese, for the operator), and for everything
that moves "brush": where the brush was left (height_mm, where, torque). What cannot be done is refused with a
text before anything moves; nothing here ends in a traceback. --dry moves nothing.

What is refused since the review of 2026-11-13, and the word that says "I have looked":
- every command that moves the brush (all but `rest`), while the tool file does not say that the tip was
  MEASURED (stage "tool", next: `fd calib-air --save`, or what `fd paint status` names for the jaw as it stands);
  --tip-looked-at after looking at a photo of gripper and brush;
- `fd paint contact` without --scrap-laid, and with it when its first photo shows something else than bare
  paper where a ladder would be painted (stage "contact": lay a scrap sheet);
- a contact rung more than 6 mm below the paper height the set-up already has (--distrust-plane when that
  plane is known to be wrong); press and --lower-mm together beyond 7.5 mm;
- test strokes and ladders within 4 cm of a well or 8 cm of a dish (--near-palette when that is meant);
- test strokes and ladders ON THE CANVAS of the set-up without --scrap-laid, and with it when the first photo
  shows something else than bare paper where they would land (stage "contact", as `fd paint contact`);
- loading a paint whose well the last `fd paint locate` did not find (stage "wells");
- strokes on a contact plane that is recorded for another brush state, or that has no record of one at all
  (`fd paint contact --adopt`, or measure it).

What the third review changed in the words (2026-11-14):
- a command in backticks, or at the beginning of a `next`, is ONE command that works as it is typed. Where it
  needs a value this program does not know (a plan's file, a folder of photos, a place) the step stands in
  words; where the value is known (the plan that was given, a pan angle) the whole command is printed;
- the options of the sub-commands are typed in full: `--scrap` is not --scrap-laid, `--tip` not
  --tip-looked-at (they are the operator's word); the refusal (stage "options") names the option;
- the continuation `fd paint run` prints keeps the --stop of the call; a plane that was ADOPTED for another
  brush state names `fd paint contact --adopt` beside the measurement; the next of an ok `fd paint brush` is
  `fd paint status`; a `fd paint rest` that cannot fold the arm with the tool at all names the way out
  (support the arm, take the tool off, `fd tool detach`), not itself.

After a command that moves, the arm stands at the photo pose with the torque ON (so that the brush does not
sink onto the paper); `fd paint rest` folds it away where neither the brush nor the place it would sink to is
over the picture, and only there switches the torque off. An interrupt (Ctrl-C, a cancelled or timed-out
call: SIGTERM is turned into one), a refused or a stopped move end the same way: brush lifted, progress
stored, the result says where the brush is. When the servos themselves stop answering in the middle of a
command (stage "hardware"), nothing can be lifted or read: the result says so ("brush": where "unknown"), and
what was painted until then is stored all the same.

Scratch files for tests and experiments: $FD_PAINT_SETUP, $FD_PAINT_STATE, $FD_PAINT_OUT (the photo folder,
else runs/<date>_paint), $FD_TOOL. A simulated command (`fd --sim paint ...`) keeps its state file and its
pictures OUTSIDE the tree unless these variables say otherwise (fd.sim_state_dir()), and never writes the
set-up of the real rig.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))      # tools/: so that this file also runs alone
import tazuna_paths  # noqa: E402,F401

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import shlex  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

import brushwork as BW  # noqa: E402
import kinematics as K  # noqa: E402
import paintskill as PS  # noqa: E402
import tooling as TL  # noqa: E402
from hwlock import HardwareLocked  # noqa: E402
from robot import Arm, MotionAborted, SafetyError, load_config  # noqa: E402
from servo_bus import BusError  # noqa: E402

SIM_JA = "（シミュレーションです。実機の結果ではありません。）"
CMD_JA = {"status": "状態の確認", "brush": "筆の設定", "check": "届く範囲の確認", "look": "撮影", "locate": "絵の具の位置合わせ",
          "test": "試し描き", "contact": "紙の高さの測定", "preview": "プレビュー", "run": "描画", "rinse": "筆洗い", "rest": "収納"}
INTERRUPTED_NEXT = ("the command was interrupted; the brush was lifted when that was possible (see \"brush\": where / "
                    "height_mm). `fd paint rest` folds the arm away; the same command goes on where it stopped")
STOPPED_NEXT = ("the reason says what stopped it and \"brush\" where the brush is; `fd paint look` takes a photo, "
                "`fd paint rest` folds the arm away")
BUS_NEXT = ("the servos stopped answering in the middle of the command: LOOK at the arm (it holds where it stopped, or "
            "hangs with the torque off; a wet brush may be on the paper) and at its cable and supply. When `fd state` "
            "answers again: `fd paint rest`; a layer that was being painted goes on after its last finished stroke when its "
            "command (fd paint run with the same plan and layer) is given again")
LOCKED_JA = "実機はロック中です（サーボもカメラも開いていません）。シミュレーション（fd --sim paint status など、fd のすぐ後ろに --sim）は使えます。"
SIM_NEXT = "`fd --sim paint status` (every paint command works on the simulated arm when --sim stands right after fd)"
MOVES = ("look", "locate", "rinse", "test", "contact", "run", "rest")       # the commands that send the arm somewhere
TIP_WORD = "--tip-looked-at"
CONTACT_NEXT = PS.CONTACT_NEXT
JAW_MOUNT_NEXT = ("fd tool jaw mount moving_outer   (after looking at a photo of gripper and brush: the brush is taped to the "
                  "outer face of the moving jaw; then `fd tool jaw close` closes the jaw on the fixed jaw and holds it)")
JAW_CLOSE_NEXT = "fd tool jaw close   (closes the jaw on the fixed jaw and holds it with a light squeeze: the brush is rigid then)"
JAW_ELSEWHERE_NEXT = ("`fd tool jaw status` shows where the jaw stands and what the tool file has for it, and names the command "
                      "that closes or holds it again; after the jaw moved, the tip is measured again (`fd calib-air --save`)")
TIP_NEXT = PS.TIP_NEXT
BRUSH_NEXT = ("fd paint brush --kind round --fine 0.6:2.7 --broad 1.2:5.0 --load-mm 140   (say which brush is on, with ITS "
              "numbers; a flat one: `fd paint brush --kind flat --wide-mm 9 --load-mm 180`)")
# What comes after the brush is ready: the plan's file is a value this program does not know, so the two commands
# are named in words (`fd paint preview PLAN` typed as it was printed is refused: third review, T12)
PLAN_NEXT = ("then, with the file of a plan: fd paint preview and that file (picture, minutes, every check), and fd "
             "paint run with that file, --layer 0 and --dry")
TOOL_FILE_NEXT = ("repair the tool file (`fd tool status` names it), or, after taking the tool off by hand, `fd tool detach`; "
                  "then register what is on the gripper (`fd tool attach --help` shows how)")
WELLS_LOST = ("wells not located: {names} (the last `fd paint locate` found no paint where the palette's move puts them; they "
              "keep the place they had, and no paint is loaded from them: `fd paint locate` again, or write their \"xy\" "
              "into the set-up by hand and take \"located\": false out)")
NEW_BRUSH_HELP = """\
actions of --do, done in the order given:
  rinse                      wash the brush                    load:PAINT     put PAINT on the brush by its recipe
  water[:clean|rinse]        dip into a dish                   blot[:N]       press it out on the blotting strip
  stroke:x,y                 one pull towards the robot (--ladder-len long) with what is on the brush
  line:x0,y0:x1,y1[:press]   one stroke in ANY direction with what is on the brush (press: factor of press_mm)
  swatch:PAINT:x,y           load PAINT, then one pull         ladder:x,y     pulls at falling heights (--zs)
x, y: metres in the robot's frame. Put test strokes on scrap paper. Strokes and ladders ON THE CANVAS of the
set-up (where the picture is; `fd paint status` shows it) are refused unless --scrap-laid says that a scrap sheet
lies over it, and the sheet is then looked at first: paint under a stroke is refused. Strokes within 4 cm of a
well or 8 cm of a dish of the set-up are refused unless --near-palette says they are meant to go there. Strokes
(not ladders) need a contact plane that is recorded for the brush as it is (step 4).

how a new brush is measured (2026-11-13: a thin round brush took three mounts and two hours without this order):
  1. the tool:   fd tool attach ...; look at a photo of gripper and brush before the first move. A brush on the
                 moving jaw must be squeezed shut:  fd tool jaw mount moving_outer ;  fd tool jaw close
                 then  fd calib-air --save   (tip-only when the camera stood still; only --save writes the record
                 that the tip was measured). Photos that were taken BEFORE the jaw was held are stored before it is
                 closed:  fd calib-air --from DIR --save  first. No painting command moves a brush whose tip was
                 not measured (the word for "I have looked at the photo": --tip-looked-at)
  2. the kind:   fd paint brush --kind round --fine 0.6:2.7 --broad 1.2:5.0 --load-mm 140     which brush is on (its
                 kind now; its numbers are corrected at step 7). Without it the set-up means the flat brush.
  3. the reach:  fd paint check                      every well, dish and the canvas, before any paint is wasted
  4. the paper:  LAY A SCRAP SHEET OVER THE CANVAS;  fd paint contact --scrap-laid --paint red   (five ladders, read
                 from two photos; look at the picture it writes), then  fd paint contact --read B.jpg A.jpg --save
                 or at once  fd paint contact --scrap-laid --paint red --save
  5. the line:   fd paint test --do load:red --do line:0.325,-0.06:0.325,-0.03:0.6 --do line:0.32,-0.02:0.345,-0.02:0.6
                 (sideways and along the robot's direction, at the fine press; these places are on the scrap strip
                 beyond the far edge of the sheet of 2026-11-13, OUTSIDE its canvas: put yours where scrap paper
                 lies; on the canvas only with a scrap sheet over it and --scrap-laid), the same with :1.2 for the
                 band; measure the widths in the photo. Between 0.6 and 1.2 the width of a round brush jumps: two
                 modes.
  6. the load:   fd paint test --do load:red --do line:...:1.2 --do line:...:1.2 ...   until the paint fades;
                 the length that still covers, times width / fine width, is load_mm (mm of FINE line)
  7. store it:   fd paint brush --kind round --fine 0.6:2.7 --broad 1.2:5.0 --load-mm 140     with the measured numbers
  8. a plan:     fd paint preview PLAN.json ;  fd paint run PLAN.json --layer 0 --dry ;  then without --dry
`fd paint status` names the first of these steps that is still missing.
"""


class _BadArgs(Exception):
    """The command line cannot be read (instead of argparse's exit: every answer of `fd paint` is a result)."""


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise _BadArgs(f"{self.prog}: {message}")


class _SubParser(_Parser):
    """The parser of one sub-command: its options are typed in full. argparse completes an abbreviated option by
    itself: `--scrap` was --scrap-laid and `--tip` was --tip-looked-at, and both are the operator's WORD for
    something the program cannot see (a scrap sheet lies over the picture; the photo of gripper and brush was
    looked at). Half a word is none (third review, T17)."""

    def __init__(self, *args, **kw):
        kw.setdefault("allow_abbrev", False)
        super().__init__(*args, **kw)


def _xy(text, what: str = "x,y") -> tuple:
    try:
        a, b = (float(v) for v in str(text).split(","))
        if not (math.isfinite(a) and math.isfinite(b)):
            raise ValueError
    except ValueError:
        raise PS.Refusal("options", f"{what} {text!r}: two numbers of metres, like 0.25,-0.03", "`fd paint test --help`") from None
    return a, b


def _tip_word(s):
    """The option of every command that moves the brush: the operator's word for a tip that was not measured."""
    s.add_argument(TIP_WORD, dest="tip_looked_at", action="store_true",
                   help="move although the tool file does not say that the tip was measured: you have LOOKED at a photo "
                        "in which gripper and brush are both visible, and the registered tip is where the brush tip is "
                        "(on 2026-11-13 a first guess was 9 cm short); `fd calib-air --save` measures it instead")


def build_parser():
    # (No --sim here: it is fd's own option and stands BEFORE the command, `fd --sim paint status`; main() reads
    # the "--sim" fd hands over in front of the words. As an option of this parser it was shown in the usage after
    # `paint`, where fd refuses it, and argparse took `--si` and `--s` for it, which got past that refusal.)
    p = _Parser(prog="fd [--sim] paint", description="watercolour with the brush that is taped to the gripper "
                                                      "(--sim: fd's own option, before the command: the simulated arm, no camera)",
                allow_abbrev=False)
    sub = p.add_subparsers(dest="cmd", required=True, parser_class=_SubParser)        # (no abbreviated options)
    sub.add_parser("status", help="the set-up, the brush, what the brush carries (files only); next: the first step "
                                  "that is still missing")
    s = sub.add_parser("brush", help="which brush is on the gripper: show it, or store it in the set-up; what it "
                                     "carries (files only)")
    s.add_argument("--carries", metavar="none|PAINT", default=None,
                   help="write what the brush carries into the state file: `none` for a brush that was rinsed or "
                        "swapped by hand, or a paint of the set-up (how much of it is then unknown: it is reloaded "
                        "before the next stroke)")
    s.add_argument("--kind", choices=("round", "flat"))
    s.add_argument("--fine", metavar="PRESS:WIDTH", help="round: the line mode, e.g. 0.6:2.7 (press factor : width in mm)")
    s.add_argument("--broad", metavar="PRESS:WIDTH", help="round: the band mode, e.g. 1.2:5.0")
    s.add_argument("--wide-mm", dest="wide_mm", type=float, help="flat: band of a stroke along the robot's direction")
    s.add_argument("--thin-mm", dest="thin_mm", type=float, help="flat: width of a sideways stroke")
    s.add_argument("--load-mm", dest="load_mm", type=float,
                   help="what one load of paint lasts: mm of FINE line (round), mm of travel (flat)")
    s.add_argument("--clear", action="store_true", help="take the block out: the flat brush of the morning of 2026-11-13")
    s.add_argument("--dry", action="store_true", help="show the block; write nothing")
    sub.add_parser("check", help="no motion: are the wells, the dish(es), the blotting strip and the canvas in reach?")
    s = sub.add_parser("look", help="to the photo pose, one photo")
    _tip_word(s)
    s = sub.add_parser("locate", help="photo; find the paint blobs again (the palette gets pushed) and store where they "
                                      "are: only when most wells are found by a blob of their own and those agree on "
                                      "one move of the palette; a well without paint in the photo, or with paint of "
                                      "another colour than it showed before (\"hsv\"), keeps its place and is marked not "
                                      "located; the places before are kept in the file (\"previous_xy\")")
    s.add_argument("--dry", action="store_true", help="a photo from where the arm stands; show what would change; "
                                                      "nothing moves, nothing is written")
    _tip_word(s)
    s = sub.add_parser("rinse", help="wash the brush")
    s.add_argument("--blot-first", dest="blot_first", type=int, default=0, metavar="N",
                   help="press the brush out N times on the blotting strip before it goes into the water (keeps ONE "
                        "dish clean longer)")
    _tip_word(s)
    s = sub.add_parser("rest", help="fold the arm away, brush off the picture; the torque goes off only there")
    s.add_argument("--keep-torque", action="store_true")
    s.add_argument("--pan", type=float, default=None, help="pan angle of the fold (default: chosen from the set-up)")
    s.add_argument("--clearance", type=float, default=None, help="brush tip above the table when folded, m (default 0.04)")
    s.add_argument("--dry", action="store_true", help="the pose and the way to it; nothing moves")
    s = sub.add_parser("test", help="test strokes on scrap paper", epilog=NEW_BRUSH_HELP,
                       formatter_class=argparse.RawDescriptionHelpFormatter)
    s.add_argument("--swatch", action="append", default=[], metavar="PAINT:x,y",
                   help="load PAINT and pull one stroke towards the robot that starts at x,y (base frame, metres)")
    s.add_argument("--ladder", action="append", default=[], metavar="x,y",
                   help="with what is on the brush: short strokes at falling heights, the first starts at x,y, the "
                        "next ones 12 mm further along +y")
    s.add_argument("--do", action="append", default=[], metavar="ACTION", help="actions in the order given (see below)")
    s.add_argument("--zs", default="2.0,0.5,-1.0,-2.5", help="ladder heights in mm relative to the contact height of the set-up")
    s.add_argument("--len", type=float, default=0.028, help="swatch length (m)")
    s.add_argument("--ladder-len", dest="ladder_len", type=float, default=0.016)
    s.add_argument("--press", type=float, default=1.0)
    s.add_argument("--lower-mm", dest="lower_mm", type=float, default=0.0,
                   help="strokes (not ladders) run with the contact plane this much lower than the set-up has it (0 .. 2; "
                        "press and lowering together at most 7.5 mm below first touch)")
    s.add_argument("--near-palette", dest="near_palette", action="store_true",
                   help="strokes and ladders MAY come within 4 cm of a well and 8 cm of a dish of the set-up (they are "
                        "refused there otherwise: one wrong sign drags the pressed brush through the palette)")
    s.add_argument("--scrap-laid", dest="scrap_laid", action="store_true",
                   help="your word that a scrap sheet lies over the canvas: needed for every stroke and ladder that "
                        "lies on the canvas of the set-up, where the picture is (the sheet is looked at first: paint "
                        "under a stroke is refused). Strokes outside the canvas need no word")
    s.add_argument("--no-photo", action="store_true")
    s.add_argument("--dry", action="store_true", help="read the actions and check their reach; nothing moves")
    _tip_word(s)
    s = sub.add_parser("contact", help="measure the height of the paper: ladders of strokes on a scrap sheet, two "
                                       "photos, the plane",
                       description="LAY A SCRAP SHEET OVER THE CANVAS, then say so with --scrap-laid. Ladders of short "
                                   "pull strokes (each rung 10 mm towards the robot, the next 12 mm further along +y, "
                                   "at falling plan heights) are painted at five places, a photo is taken before and "
                                   "after, and the marks are READ from the photos: the highest rung that left a mark, "
                                   "and how much of one, is the height of first touch there. The plane through the "
                                   "places is the contact plane. The first photo is looked at before any paint is "
                                   "loaded: a picture under the ladders is refused. A record beside the photos keeps "
                                   "the brush state the ladders were painted with.")
    s.add_argument("--scrap-laid", dest="scrap_laid", action="store_true",
                   help="your word that a scrap sheet lies over the canvas (a blank painting sheet cannot be told from "
                        "scrap in a photo); without it nothing is painted")
    s.add_argument("--at", action="append", default=None, metavar="canvas|x,y",
                   help="where: 'canvas' (default) = the four corners of the canvas, 15 mm inside, and its centre; "
                        "x,y = a ladder that STARTS there (repeatable)")
    s.add_argument("--zs", default=None, help="plan heights of the rungs in mm (base frame z, NOT relative to the "
                                              "stored contact height), falling (default \"12,9,6,3\"); no rung more than "
                                              "6 mm below the paper height the set-up already has")
    s.add_argument("--distrust-plane", dest="distrust_plane", action="store_true",
                   help="the contact plane of the set-up is known to be wrong (a new table, a pad under the sheet): rungs "
                        "may go deeper below it than 6 mm (never below the limit of brush work under the table)")
    s.add_argument("--near-palette", dest="near_palette", action="store_true",
                   help="a ladder MAY stand within 4 cm of a well or 8 cm of a dish of the set-up")
    s.add_argument("--paint", default=None, help="paint for the marks (default: what is on the brush, else the first "
                                                 "paint of the set-up; a strong colour reads best)")
    s.add_argument("--ladder-len", dest="ladder_len", type=float, default=None, help="length of a rung in m (default 0.010)")
    s.add_argument("--read", nargs=2, metavar=("BEFORE.jpg", "AFTER.jpg"),
                   help="no arm: read two saved photos again. Places, heights and the brush state come from the record "
                        "beside BEFORE.jpg (BEFORE.json); photos without one need --at / --zs as when they were taken, "
                        "and cannot be stored")
    s.add_argument("--save", action="store_true",
                   help="write the plane into the set-up, with the brush state the ladders were painted with (needs 4 "
                        "places read and residuals within 1.5 mm; with --read: the record beside the photos, and the "
                        "registered tool must still be that brush)")
    s.add_argument("--adopt", action="store_true",
                   help="no motion, nothing measured: record the brush AS IT IS NOW as the brush state the stored plane "
                        "holds for. Your word that the plane was measured with exactly this brush (for a plane from "
                        "before such records were kept); refused when the plane has a measured record of another brush")
    s.add_argument("--dry", action="store_true", help="the places and their reach; nothing moves, nothing is written "
                                                      "(with --read / --adopt too)")
    _tip_word(s)
    s = sub.add_parser("preview", help="no motion: the simulated picture, statistics, minutes, the checks of run --dry")
    s.add_argument("plan")
    s.add_argument("--out", default=None, metavar="FILE", help="where the picture goes (default: the photo folder)")
    s = sub.add_parser("run", help="paint one layer of a plan")
    s.add_argument("plan")
    s.add_argument("--layer", required=True, help="layer index or name")
    s.add_argument("--start", type=int, default=None, help="first stroke (default: where the last run of this layer stopped)")
    s.add_argument("--stop", type=int, default=None)
    s.add_argument("--max-min", dest="max_min", type=float, default=8.0)
    s.add_argument("--reload-mm", dest="reload_mm", type=float, default=None,
                   help="reload after this much plain TRAVEL, whatever the brush of the set-up says (short: every "
                        "stroke gets fresh paint)")
    s.add_argument("--lower-mm", dest="lower_mm", type=float, default=0.0,
                   help="paint this layer with the contact plane so much lower than it was measured (0 .. 2 mm): for "
                        "bands that come out thin where the sheet lies lower than the plane (2026-11-13: 1.5 at the far "
                        "middle); the written lines were run without. Press and lowering together: at most 7.5 mm "
                        "below first touch")
    s.add_argument("--dry", action="store_true", help="check the plan against the set-up and the arm's reach, say how "
                                                      "long it takes; nothing moves")
    s.add_argument("--no-photo", action="store_true")
    _tip_word(s)
    return p


# ----------------------------------------------------------------------------- arm, camera, folders
def _make_arm(sim: bool, tool="sim"):
    """The arm of a command. sim: a simulated bus, in the tool's rest pose, carrying the simulated tool when one
    was attached with `fd --sim tool attach`, else the real one (read only)."""
    if sim:
        from sim_bus import SimBus
        arm = Arm(bus=SimBus(realtime=False), cfg=load_config(), tool=TL.active(sim=True) if tool == "sim" else tool)
        arm.open()
        TL.sim_start(arm)
        return arm
    return Arm().open()


def _open_arm(sim: bool):
    """The arm of a command that moves it. An arm that cannot be opened (the hardware lock, no answer, no port)
    is a refusal: nothing was opened, nothing moved. (A bus error LATER is another matter: see main.)"""
    try:
        return _make_arm(sim)
    except (BusError, HardwareLocked, OSError) as e:
        if sim:
            raise
        if "hardware is locked" in str(e):
            raise PS.Refusal("hardware", f"{type(e).__name__}: {e}", "`fd hw status` says why it is locked; " + SIM_NEXT,
                             say=LOCKED_JA) from e
        raise PS.Refusal("hardware", f"the arm could not be opened: {type(e).__name__}: {e}",
                         "is the arm switched on and connected? `fd state` asks it; without it: " + SIM_NEXT) from e


def _close(arm):
    try:
        arm.close()
    except Exception as e:      # a bus that is already gone: the result of the command matters more
        _log(f"closing the arm failed: {type(e).__name__}: {e}")


def _kin_arm(sim: bool) -> tuple:
    """(arm, where its joint limits come from) for the commands that only compute. The real arm is asked for
    its limits (nothing moves); when it cannot be opened (the hardware lock, the arm switched off) the simulated
    arm stands in, carrying the REAL tool, and the result says so."""
    if sim:
        return _make_arm(True), "the simulated arm"
    try:
        return _make_arm(False), "the arm"
    except (BusError, HardwareLocked, SafetyError, OSError) as e:
        why = "the hardware is locked" if "hardware is locked" in str(e) else f"the arm could not be opened ({str(e)[:80]})"
        return _make_arm(True, tool=TL.active(sim=False)), f"the simulated arm ({why})"


def _camera(sim: bool):
    """The photo function of a command (paper exposure), None for the simulated arm (no camera is opened)."""
    if sim:
        return None
    import camera as C
    return lambda: C.grab(exposure="paper")


def _cam_model():
    from camera import CameraModel
    try:
        return CameraModel.load()
    except (OSError, ValueError, KeyError) as e:
        raise PS.Refusal("camera", f"the camera model cannot be read ({e})", "`fd calib-air` fits it") from e


def _camera_locked() -> bool:
    import hwlock
    return hwlock.info() is not None


def _out_dir(sim: bool = False) -> str:
    """Where photos and pictures go: $FD_PAINT_OUT; else runs/<date>_paint, and for a simulated command a
    folder OUTSIDE the tree (the pictures of `fd --sim paint preview` used to land among the real photos)."""
    env = os.environ.get("FD_PAINT_OUT")
    if env:
        return env
    if sim:
        return os.path.join(PS.sim_dir(), "paint_out")
    return os.path.join(tazuna_paths.ROOT, "runs", time.strftime("%Y%m%d") + "_paint")


def _tool_file_refusal(cmd, e) -> dict | None:
    """The refusal for a tool file that cannot be used (tooling.load's SafetyError), with a next of its own;
    None for any other SafetyError."""
    text = str(e)
    if "tool file" not in text or "cannot be used" not in text:
        return None
    res = _refusal(cmd, "tool", text, TOOL_FILE_NEXT)
    res["say"] = (f"{CMD_JA.get(cmd, 'コマンド')}は実行しませんでした。ツールのファイルが読めないので、グリッパーに何が付いているか"
                  "分かりません。ファイルを直すか、道具を手で外してから fd tool detach を実行してください。アームは動かしていません。")
    return res


def _registered_tool(sim: bool):
    """The tool a command will work with, read from its file before anything is opened: (tool, its file).
    Refusal (stage "tool") when none is registered."""
    tool = TL.active(sim=sim)
    PS.require_tool(tool)
    return tool, TL.source(sim)


def _log(*m):
    print(*m, file=sys.stderr, flush=True)


def _term_as_interrupt():
    """A cancelled or timed-out call ends this process with SIGTERM: turned into KeyboardInterrupt, so that the
    brush is lifted and the result is still printed (as `fd draw` does)."""
    import signal
    import threading
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, signal.default_int_handler)


def _refusal(cmd, stage: str, reason: str, next_: str = "", **more) -> dict:
    res = {"ok": False, "cmd": cmd, "stage": stage, "reason": reason,
           "next": next_ or "`fd paint --help` lists the commands; `fd paint status` shows the set-up",
           "say": f"{CMD_JA.get(cmd, 'コマンド')}は実行しませんでした（{reason}）。アームは動かしていません。"}
    res.update(more)
    return res


def _may_write_setup(a, what: str):
    """A simulated command does not change the set-up of the real rig (as `fd --sim config --set` is refused)."""
    if a.sim and not os.environ.get("FD_PAINT_SETUP"):
        raise PS.Refusal("options", f"fd --sim paint {what}: a simulated command would write the painting set-up of the "
                                    f"REAL rig ({PS.setup_path()})",
                         "the same command without --sim (it then writes that set-up), or with --dry to see what it would write")


def _load_plan(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            plan = json.load(f)
    except (OSError, ValueError) as e:
        raise PS.Refusal("plan", f"the plan {path} cannot be read ({e})",
                         "give the path of a plan file (JSON: canvas_mm, pan_axis_mm, layers; tools/paint/render.py)") from e
    if not (isinstance(plan, dict) and isinstance(plan.get("layers"), list) and plan["layers"]):
        raise PS.Refusal("plan", f"{path} is not a painting plan (no \"layers\")",
                         "give the path of a plan file (JSON: canvas_mm, pan_axis_mm, layers; tools/paint/render.py)")
    return plan


def _layer_index(plan: dict, text, path: str = "PLAN") -> int:
    names = [str(ly.get("name")) if isinstance(ly, dict) else "?" for ly in plan["layers"]]
    t = str(text)
    if t.lstrip("-").isdigit() and 0 <= int(t) < len(names):
        return int(t)
    if t in names:
        return names.index(t)
    raise PS.Refusal("plan", f"the plan has no layer {t!r}: its layers are " + ", ".join(f"{i} {n!r}" for i, n in enumerate(names)),
                     "the same command with one of these after --layer (its index or its name); `fd paint preview "
                     f"{shlex.quote(path)}` lists them")


def _abbreviated(argv) -> list:
    """[(a word of argv as it was typed, the options of its sub-command that begin with it)]: what argparse
    would have completed by itself before the sub-parsers stopped taking abbreviations (_SubParser)."""
    sub = next((act for act in build_parser()._actions if isinstance(getattr(act, "choices", None), dict)), None)
    cmd = next((w for w in argv if sub is not None and w in sub.choices), None)
    if cmd is None:
        return []
    options = [o for act in sub.choices[cmd]._actions for o in act.option_strings if o.startswith("--")]
    out = []
    for w in argv:
        word = str(w).split("=", 1)[0]
        if word.startswith("--") and len(word) > 2 and word not in options:
            could = [o for o in options if o.startswith(word)]
            if could:
                out.append((word, could))
    return out


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


# ----------------------------------------------------------------------------- the frame of a command that moves
class _Sess:
    """Arm, painter and state of one command that moves."""

    def __init__(self, a):
        self.a = a
        self.setup = PS.load_setup()
        # Everything a file can refuse is refused before the arm is opened: a state file that cannot be used
        # (it holds the progress of every layer), no brush, a brush whose tip was never measured.
        self.state = PS.load_state(a.sim)
        tool, _file = _registered_tool(a.sim)
        # (`locate --dry` takes its photo from where the arm stands: nothing moves, so nothing depends on the tip)
        looked = bool(getattr(a, "tip_looked_at", False)) or (a.cmd == "locate" and getattr(a, "dry", False))
        if not looked:
            PS.require_measured_tip(tool)
        _term_as_interrupt()
        self.arm = _open_arm(a.sim)
        try:
            self.photo = _camera(a.sim)
            self.pt = PS.Painter(self.arm, self.setup, photo=self.photo, out_dir=_out_dir(a.sim), log=_log,
                                 unmeasured_tip_ok=looked)
        except BaseException:
            _close(self.arm)
            raise
        self.res = {"ok": True, "cmd": a.cmd, "stage": a.cmd}
        self.notes = []              # warnings of the frame itself (they go in front of the body's)
        if not self.pt.tip_measured and getattr(a, "tip_looked_at", False):
            self.notes.append(f"the tip of tool '{self.arm.tool.name}' was not measured: this command moves on your word "
                              f"({TIP_WORD}) that the registered tip is where the brush tip is; `fd calib-air --save` measures "
                              "and records it")
        self.pt.paint, note = PS.carried_paint(self.state, self.arm.tool)
        self.pt.unknown = bool(note)        # "not known" is not "nothing": washed before its first load (Painter.load)
        if note:
            self.notes.append(note)
        self.pt.blot_k = int(self.state.get("blot_k", 0) or 0)
        self.say = ""
        self.refused_say = ""        # the operator's sentence for a refusal that has more to say than its reason
        self.skip_photo = False      # no photo after this command (it refused after its own look: nothing new to see)
        self.after = None            # what is done after the brush is up and the photo is taken
        self.dirty = False           # the state file must be written although nothing moved
        self.count_time = False      # a real `run`: its seconds go into the time model

    def save_state(self):
        pt, st = self.pt, self.state
        if not (pt.br.moved or self.dirty):
            return                   # a refusal or a dry run leaves the state file as it was
        st.update(paint=pt.paint, since_load_mm=round(min(pt.since_load, 1e6), 1), used_mm=round(min(pt.used, 1e6), 1),
                  blot_k=int(pt.blot_k), at=time.strftime("%Y-%m-%dT%H:%M:%S"), tool=PS.tool_id(self.arm.tool))
        if pt.unknown:              # moved, but neither washed nor loaded: what it carries is still not known
            st["paint_unknown"] = True
        else:
            st.pop("paint_unknown", None)
        PS.save_state(st, self.a.sim)


def _way_out(sess) -> dict:
    """The end of every command that moves: brush up, to the photo pose; the report of where the brush is. Never
    raises: a second interrupt and servos that do not answer are part of the report (and of the result)."""
    pt, res = sess.pt, sess.res
    try:
        try:
            return pt.br.retreat(sess.setup["photo_q_deg"])
        except KeyboardInterrupt:       # interrupted again, while the brush was being lifted: say where it is now
            res.update(ok=False, interrupted=True)
            res.setdefault("reason", "interrupted")
            return pt.br.where({"lift_error": "interrupted again while the brush was being lifted or taken away"})
    except BusError as e:               # nothing can be moved and nothing read: the result must not pretend otherwise
        if res["ok"]:
            res.update(ok=False, stage="hardware", reason=f"BusError: {e}")
        res["next"] = BUS_NEXT
        return pt.br.lost(e)


def _moving(a, body, photo_after: bool = False) -> dict:
    """Run body(sess) with everything a command that moves owes the operator: whatever happens in it (a
    refusal, a stopped move, an interrupt, a bug), the brush is lifted and taken to the photo pose, the state
    is stored and the result says where the brush is (or, when the servos do not answer, that it is not known)."""
    sess = _Sess(a)
    pt, res = sess.pt, sess.res
    t0 = pt.clock()
    try:
        try:
            body(sess)
        except PS.Refusal as e:
            detail = dict(e.detail)
            ja = detail.pop("say", None)        # (a refusal with a sentence of its own for the operator)
            if ja and not sess.refused_say:
                sess.refused_say = ja.replace("アームは動かしていません。", "")       # (where the arm is: said at the end)
            res.update(ok=False, stage=e.stage, reason=str(e), next=e.next, **detail)
        except BW.JawRefusal as e:
            # the jaw does not stand where the tool file has it: a refusal with its own way out (the general hint
            # sent the operator from status to check to run --dry and back to this refusal)
            res.update(ok=False, stage="tool", reason=str(e), next=JAW_ELSEWHERE_NEXT)
            sess.refused_say = (f"{CMD_JA[a.cmd]}は実行しませんでした。爪（グリッパー）が、ツールのファイルに記録された位置に"
                                "ありません。筆は、筆先を測ったときの場所からずれています。fd tool jaw status で爪の状態を確認して"
                                "ください（閉じ直したあとは、筆先の測り直しが必要です）。")
        except (SafetyError, MotionAborted) as e:
            res.update(ok=False, stage=a.cmd, reason=f"{type(e).__name__}: {e}")
        except KeyboardInterrupt:
            res.update(ok=False, stage=a.cmd, reason="interrupted", interrupted=True)
        except BusError as e:       # a servo did not answer (a cable, the supply): the way out tells whether they do now
            res.update(ok=False, stage="hardware", reason=f"BusError: {e}")
        except Exception as e:      # a bug here: the brush is lifted all the same, and the result says what it was
            import traceback
            res.update(ok=False, stage="internal", reason=f"internal error: {type(e).__name__}: {e}",
                       trace=traceback.format_exc().strip().splitlines()[-6:])
        try:
            rep = res["brush"] = _way_out(sess)
            if photo_after and pt.br.moved and rep["where"] == "photo pose" and sess.photo is not None \
                    and not getattr(a, "no_photo", False) and not sess.skip_photo:
                try:
                    res["photo"] = pt.look(a.cmd)
                except (SafetyError, MotionAborted) as e:
                    res["photo_error"] = str(e)
            if sess.after is not None and res["ok"]:
                try:
                    sess.after(sess)
                except PS.Refusal as e:
                    res.update(ok=False, stage=e.stage, reason=str(e), next=e.next, **e.detail)
                except Exception as e:      # (the brush is up and the arm holds: what remains is to say what broke)
                    import traceback
                    res.update(ok=False, stage="internal", reason=f"internal error: {type(e).__name__}: {e}",
                               trace=traceback.format_exc().strip().splitlines()[-6:])
        finally:
            seconds = round(float(pt.clock() - t0), 1)
            if sess.count_time and not a.sim:
                # only a run that ended ok feeds the time means (a stopped one would book the seconds of the load
                # that failed and of the way out as overhead per stroke: PS.note_run)
                PS.note_run(sess.state, res, seconds, pt.spent, ok=bool(res.get("ok")))
            try:
                sess.save_state()
            except (OSError, PS.Refusal) as e:      # (a full disk): the brush did its work, but the file does not know it
                res["state_file_error"] = f"{type(e).__name__}: {e}"
        if pt.br.moved:
            try:
                st = sess.arm.state()
                res.update(temps=st.get("temp_c"), volt=st.get("volt"))
            except Exception as e:          # the servos' temperatures are a courtesy, not worth the result
                res["state_error"] = f"{type(e).__name__}: {e}"
        res.update(loads_total=pt.loads, seconds=seconds)
        if pt.br.hop_retries:
            res["hop_retries"] = pt.br.hop_retries      # far moves the height monitor stopped, done again at half speed
        if isinstance(pt.br.jaw, dict) and pt.br.jaw.get("held"):
            res["jaw"] = pt.br.jaw          # the squeeze of a brush on the moving jaw was put back
        if pt.washed_unknown:
            res["washed_first"] = "what the brush carried was not known: it was washed before its first load"
        notes = list(sess.notes)
        if pt.br.jaw_note:                  # the jaw is shut, and no hold is recorded that could be put back
            notes.append(pt.br.jaw_note)
            res["jaw"] = {"held": False, "why": "no jaw hold is recorded although the jaw is shut"}
        if notes:
            res["warnings"] = notes + [w for w in res.get("warnings", []) if w not in notes]
        rep = res["brush"]
        if res["ok"]:
            say = sess.say or f"{CMD_JA[a.cmd]}が終わりました。"
            if pt.br.jaw_note:
                say += "注意: 爪は閉じていますが、筆を固定する保持（スクイーズ）の記録がありません（warnings）。"
        elif res.get("interrupted"):
            res.setdefault("next", INTERRUPTED_NEXT)
            say = f"{CMD_JA[a.cmd]}が途中で中断されました。"
        elif rep["where"] == "not moved":
            res.setdefault("next", "`fd paint status` shows the set-up; `fd paint check` what is in reach")
            say = sess.refused_say or f"{CMD_JA[a.cmd]}は実行しませんでした（{res.get('reason')}）。"
        else:
            res.setdefault("next", STOPPED_NEXT)
            say = sess.refused_say or sess.say or f"{CMD_JA[a.cmd]}の途中で止まりました（{res.get('reason')}）。"
        if pt.washed_unknown:
            say += "筆に何が付いているか分からなかったので、最初の補充の前に洗いました。"
        if "lift_error" in rep:
            say += "筆を持ち上げられませんでした。"
        if "state_file_error" in res:
            k = res.get("next_stroke")
            note = (f"the state file {PS.state_path(a.sim)} could not be written ({res['state_file_error']}): what the brush "
                    "carries and how far a layer is painted are NOT stored")
            if res["ok"]:
                res.update(ok=False, stage="state", reason=note,
                           next="make room on the disk (`df -h`), then `fd paint status`"
                                + (f"; until then this layer goes on with `--start {k}` (the file has the old progress)" if k is not None else ""))
            else:
                res.setdefault("warnings", []).append(note + (f"; this layer stands at stroke {k} (`--start {k}`)" if k is not None else ""))
            say += "状態ファイルに書き込めませんでした（進み具合は保存されていません）。"
        res["say"] = say + BW.where_ja(rep) + (SIM_JA if a.sim else "")
        if a.sim:
            res["sim"] = True
        return res
    finally:
        _close(sess.arm)


# ----------------------------------------------------------------------------- status, brush
STEP_JA = {"tool": "ツールの登録（fd tool attach）", "jaw": "爪を閉じて保持する（fd tool jaw mount / close）",
           "tip": "筆先の測定と記録（fd calib-air --save）", "brush": "筆の種類の登録（fd paint brush）",
           "contact": "紙の高さ（fd paint contact）",
           "tip first": "撮影済みの写真があれば、爪を保持する前に筆先を記録する（fd calib-air に --from と写真のフォルダ、--save を"
                        "付ける）。写真がなければ、爪を閉じて保持してから測定する"}


def _steps(tool, setup: dict, brush: dict, record: dict, cfg: dict) -> list:
    """What a brush needs before it paints, in the order it has to be done: [{"step", "done", "next", "why",
    "ja"}]. ONE order, for `fd paint status`, `check` and the refusals:
    - the jaw (a brush on the moving jaw is measured and painted with squeezed shut), then the tip
      (`fd calib-air --save`), then which brush it is (`fd paint brush`), then the paper height for that brush
      (`fd paint contact`). `fd paint status` used to send the operator to the contact plane whatever else was
      missing: measured with the brush loose, it was then accepted for the rigid one (review of 2026-11-13);
    - but while the jaw is shut WITHOUT a recorded hold and the tip is not measured (the rig's own tool file of
      2026-11-13), the tip comes FIRST: photos that were taken with the jaw as it stands are stored before the
      jaw is closed and held (`fd calib-air --from DIR --save`). With a hold on record they are photos of another
      jaw state: jaw first, as this list used to say, ended in a refusal at the third command, and the operator
      had to fly a new measurement on the real arm (second review). For a NEW capture the jaw is still closed
      first: the same `next` says so."""
    if tool is None:
        return [{"step": "tool", "done": False, "why": "no brush is registered", "ja": STEP_JA["tool"],
                 "next": "fd tool attach --help   (shows how the brush that is on the gripper is registered: its name, and where "
                         "its tip is by a photo of gripper and brush; a tip that is typed is a guess until `fd calib-air "
                         "--save` has measured it)"}]
    jaw = PS.jaw_hold_note(tool, cfg)
    measured = BW.tip_is_measured(tool)
    jaw_step = {"step": "jaw", "done": jaw is None, "why": jaw, "ja": STEP_JA["jaw"],
                "next": JAW_CLOSE_NEXT if getattr(tool, "on_jaw", None) == "moving_outer" else JAW_MOUNT_NEXT}
    tip_step = {"step": "tip", "done": measured, "next": PS.tip_next(tool, cfg), "ja": STEP_JA["tip"],
                "why": None if measured else "the tool file does not say that the tip was measured (a typed, guessed "
                                             "or estimated tip): no painting command moves this brush"}
    if jaw is not None and not measured:
        tip_step["ja"] = STEP_JA["tip first"]
        steps = [tip_step, jaw_step]
    else:
        steps = [jaw_step, tip_step]
    steps.append({"step": "brush", "done": bool(brush["stated"]), "next": BRUSH_NEXT, "ja": STEP_JA["brush"],
                  "why": None if brush["stated"] else "the set-up has no \"brush\" block: it is taken to be the flat brush "
                                                      "of the morning of 2026-11-13 (its band about 9 mm wide)"})
    if not record["recorded"]:
        why, nxt = record["reason"], PS.ADOPT_NEXT
    elif not record["ok"]:
        # (a record that was ADOPTED, PS.contact_record_state: the word may be given again, beside the measurement)
        why, nxt = record["reason"], record.get("next") or CONTACT_NEXT
    else:
        why, nxt = None, CONTACT_NEXT
    steps.append({"step": "contact", "done": bool(record["recorded"] and record["ok"]), "why": why, "next": nxt,
                  "ja": STEP_JA["contact"]})
    return steps


def cmd_status(a) -> dict:
    s = PS.load_setup()
    state = PS.load_state(a.sim)
    brush = PS.brush_of(s)
    tool = TL.active(sim=a.sim)
    cfg = load_config()
    rec = (s["contact"] or {}).get("measured")
    if tool is None:
        record = {"recorded": bool(rec), "ok": False, "why": ["no brush is registered"]}
    else:
        # (files only: the canvas pitch needs the arm's joint limits; `fd paint check` and `run` compare it too)
        record = PS.contact_record_state(s, tool, (rec or {}).get("pitch_deg", 0.0) if isinstance(rec, dict) else 0.0)
        record.pop("measured", None)
    steps = _steps(tool, s, brush, record, cfg)
    warnings = [f"{st['step']}: {st['why']}" for st in steps if not st["done"] and st["why"]]
    carried, note = PS.carried_paint(state, tool)
    if note:
        warnings.append(note)
    one = PS.one_dish(s)
    if one:
        warnings.append("ONE water dish: rinse and clean water are the same dish; change the water often")
    lost = PS.wells_not_located(s)
    if lost:
        warnings.append(WELLS_LOST.format(names=", ".join(lost)))
    if brush["kind"] == "round":
        f, b = brush["modes"]["fine"], brush["modes"]["broad"]
        brush_ja = f"丸筆（細線 {f['width_mm']:g} mm / 太線 {b['width_mm']:g} mm、1回の補充で細線 {brush['load_mm']:g} mm）"
    else:
        brush_ja = f"平筆（幅 {brush['wide_mm']:g} mm、1回の補充で {brush['load_mm']:g} mm）" + ("" if brush["stated"] else "＝既定")
    todo = [st for st in steps if not st["done"]]
    nxt = todo[0]["next"] if todo else "`fd paint check` (reach); " + PLAN_NEXT
    w, h = s["canvas"]["size_mm"]
    if note:
        carried_ja = ("筆に付いている絵の具は不明です（記録は別のツールのものです）。最初の補充の前に洗います。")
    else:
        carried_ja = f"筆には「{carried}」が付いています。" if carried else "筆には何も付いていない記録です。"
    res = {"ok": True, "cmd": "status", "stage": "status", "setup": PS.setup_path(), "state_file": PS.state_path(a.sim),
           "canvas": s["canvas"], "pan_axis_mm": PS.pan_axis_mm(s), "contact": s["contact"], "contact_record": record,
           "press_mm": s.get("press_mm", 2.0), "brush": brush, "one_dish": one,
           "water": {k: v for k, v in s["water"].items() if k in PS.DISHES}, "blot": s.get("blot"),
           "paints": sorted(s["paints"]), "wells": sorted(s["wells"]), "wells_not_located": lost,
           "tool": None if tool is None else {"name": tool.name, "tip_offset": tool.tip_offset, "gripper_raw": tool.gripper_raw,
                                              "pitch_deg": tool.pitch_deg, "tip_measured": BW.tip_is_measured(tool),
                                              "on_jaw": getattr(tool, "on_jaw", None),
                                              "gripper_goal_raw": getattr(tool, "gripper_goal_raw", None)},
           "carries": carried, "steps": [{k: st[k] for k in ("step", "done", "next")} for st in steps],
           "state": state, "times": PS.times_of(state), "warnings": warnings, "next": nxt,
           "say": (f"キャンバス {w:g}×{h:g} mm、絵の具 {len(s['paints'])} 種、筆は{brush_ja}"
                   + ("、水入れは1つ" if one else "、水入れは2つ") + "。" + carried_ja
                   + (f"次にすること: {todo[0]['ja']}（next）。" if todo else "")
                   + (f"注意 {len(warnings)} 件（warnings）。" if warnings else ""))}
    if a.sim:
        res["sim"] = True
    return res


def _mode(text, what: str):
    try:
        p, w = (float(v) for v in str(text).split(":"))
    except ValueError:
        raise PS.Refusal("options", f"{what} {text!r}: PRESS:WIDTH, like 0.6:2.7 (press factor : width in mm)",
                         "`fd paint brush --help`") from None
    return {"press": p, "width_mm": w}


def _brush_carries(a, s: dict) -> dict:
    """`fd paint brush --carries none` (or a paint's name): write what the brush carries into the state file (file only). For
    a brush that was rinsed at the tap or swapped: only `fd paint rinse`, a wash by the arm, cleared the record
    before, and a new dry brush was then loaded with the short reload recipe of the paint the old one carried."""
    word = str(a.carries)
    paint = None if word.lower() in ("none", "nothing", "clean") else word
    if paint is not None and paint not in s["paints"]:
        raise PS.Refusal("paint", f"--carries {word!r}: `none`, or a paint of the set-up (paints: {', '.join(sorted(s['paints']))})",
                         "`fd paint status` lists the paints of the set-up")
    tool, _file = _registered_tool(a.sim)
    state = PS.load_state(a.sim)
    was, _note = PS.carried_paint(state, tool)
    res = {"ok": True, "cmd": "brush", "stage": "brush", "state_file": PS.state_path(a.sim), "carries": paint, "carried": was,
           "tool": tool.name, "saved": False}
    if a.sim:
        res["sim"] = True           # (the simulated arm's own state file: never the real brush's)
    if a.dry:
        res.update(stage="planned", next="the same command without --dry",
                   say=f"筆の中身の記録を「{paint or 'なし'}」にする予定です（まだ保存していません）。")
        return res
    # how much of it is on the brush is not known: the next stroke of this paint gets a reload first
    state.update(paint=paint, since_load_mm=1e6, used_mm=1e6, at=time.strftime("%Y-%m-%dT%H:%M:%S"), tool=PS.tool_id(tool))
    state.pop("paint_unknown", None)
    PS.save_state(state, a.sim)
    res.update(saved=True, next="`fd paint status` shows it; a wash by the arm: `fd paint rinse`",
               say=(f"筆（ツール「{tool.name}」）には「{paint}」が付いていると記録しました（量は不明なので、次の線の前に補充します）。"
                    if paint else f"筆（ツール「{tool.name}」）には何も付いていないと記録しました。"))
    return res


def cmd_brush(a) -> dict:
    s = PS.load_setup()
    old = PS.brush_of(s)
    changing = bool(a.clear or a.kind or a.fine or a.broad or a.wide_mm is not None or a.thin_mm is not None
                    or a.load_mm is not None)
    if a.carries is not None:
        if changing:
            raise PS.Refusal("options", "--carries says what the brush carries (the state file); the brush block of the "
                                        "set-up (--kind, --fine, ...) is another command", "`fd paint brush --help`")
        return _brush_carries(a, s)
    res = {"ok": True, "cmd": "brush", "stage": "brush", "setup": PS.setup_path(), "saved": False}
    if not changing:
        res.update(brush=old, say=("セットアップに筆の記載はありません（午前の平筆として扱います）。" if not old["stated"] else
                                   f"セットアップの筆は {old['kind']}（1回の補充で {old['load_mm']:g} mm）です。"),
                   next="`fd paint brush --kind round --fine 0.6:2.7 --broad 1.2:5.0 --load-mm 140` stores a round brush")
        return res
    new = dict(s)
    if a.clear:
        new.pop("brush", None)
    else:
        kind = a.kind or old["kind"]
        if kind == "round":
            was = old if old["kind"] == "round" else {"modes": {"fine": {"press": 0.6, "width_mm": 2.7},
                                                                "broad": {"press": 1.2, "width_mm": 5.0}}, "load_mm": 140.0}
            new["brush"] = {"kind": "round",
                            "modes": {"fine": _mode(a.fine, "--fine") if a.fine else dict(was["modes"]["fine"]),
                                      "broad": _mode(a.broad, "--broad") if a.broad else dict(was["modes"]["broad"])},
                            "load_mm": float(a.load_mm) if a.load_mm is not None else was["load_mm"]}
            if a.wide_mm is not None or a.thin_mm is not None:
                raise PS.Refusal("options", "--wide-mm / --thin-mm describe a flat brush; a round one has --fine and --broad",
                                 "`fd paint brush --help`")
        else:
            was = old if old["kind"] == "flat" else PS.brush_of({})
            new["brush"] = {"kind": "flat", "wide_mm": float(a.wide_mm) if a.wide_mm is not None else was["wide_mm"],
                            "thin_mm": float(a.thin_mm) if a.thin_mm is not None else was["thin_mm"],
                            "load_mm": float(a.load_mm) if a.load_mm is not None else was["load_mm"]}
            if a.fine or a.broad:
                raise PS.Refusal("options", "--fine / --broad describe a round brush; a flat one has --wide-mm and --thin-mm",
                                 "`fd paint brush --help`")
    PS.check_setup(new, PS.setup_path())
    res["brush"] = PS.brush_of(new)
    if a.dry:
        res.update(stage="planned", say="筆の設定を確認しました（まだ保存していません）。")
        return res
    _may_write_setup(a, "brush")
    PS.save_setup(new)
    # (`fd paint status`: it names the first step that is still missing. The preview of a plan that stood here is
    # refused at stage "contact" while the plane has no record for this brush, as on the rig's file: third review, T16)
    res.update(saved=True, next="`fd paint status` names what this brush still needs before it paints, or the commands "
                                "for a plan when nothing is missing",
               say=("筆の記載を消しました（午前の平筆として扱います）。" if a.clear else
                    f"筆を {res['brush']['kind']} として保存しました（1回の補充で {res['brush']['load_mm']:g} mm）。"
                    "別の種類の筆で作ったプランは断られます。"))
    return res


# ----------------------------------------------------------------------------- look, locate, rinse
def _look_body(sess):
    sess.res["photo"] = sess.pt.look("look")
    sess.say = "撮影姿勢で写真を1枚撮りました。" if sess.res["photo"] else "撮影姿勢に移動しました（カメラがないので写真はありません）。"


def _locate_body(sess):
    a, pt, res = sess.a, sess.pt, sess.res
    if sess.photo is None:
        raise PS.Refusal("camera", "locate reads a photo: there is no camera with the simulated arm",
                         "`fd paint locate` on the real arm (--dry: a photo from where the arm stands, nothing written)")
    cam = _cam_model()
    if not a.dry:
        _may_write_setup(a, "locate")
    res["photo"] = pt.look("locate", move=not a.dry)
    if a.dry and float(np.max(np.abs(sess.arm.read_q() - np.radians(sess.setup["photo_q_deg"])))) > math.radians(3.0):
        res["note"] = "dry run: the photo was taken with the arm where it stands, not from the photo pose; it may hide wells"
    found = PS.relocate_wells(pt.last_photo, cam, sess.setup, pt.tz)
    res.update(found)
    res["saved"] = False
    n, total = found["found"], len(found["wells"])
    usable, why = PS.locate_usable(found)
    res["usable"] = usable
    if res["photo"]:
        res["picture"] = PS.locate_picture(pt.last_photo, cam, sess.setup, found, pt.tz,
                                           os.path.splitext(res["photo"])[0] + "_wells.jpg")
    if not usable and found.get("apart_mm"):
        # Nothing is stored: the wells that took a blob do not agree on one move of the palette. One of them took
        # paint that is not its own (magenta the edge of violet's, with the palette's record 8 deg and 17 mm off),
        # and the fit it bent put three more wells 12 to 20 mm beside the middle of their paint (2026-11-14).
        apart = "、".join(f"{name}（{d:g} mm）" for name, d in found["apart_mm"].items())
        sess.refused_say = ("絵の具の位置は保存しませんでした。見つけた絵の具どうしが、パレットの 1 つの動き（回転とずれ）に合いません"
                            f"（{apart}）。パレットを大きく動かした・回したあとでは、隣の絵の具を取り違えることがあります。パレットを"
                            "元の位置に戻してもらうか、セットアップの絵の具の位置（wells）を手で直してから、もう一度 fd paint locate を"
                            "実行してください。写真（picture）で、灰色の輪（セットアップの位置）と絵の具を見比べられます。"
                            "セットアップは元のままです。")
        raise PS.Refusal("locate", f"the wells were not stored: {why}. The set-up keeps the places it had",
                         "put the palette back where the set-up has it (the picture, \"picture\", shows where: grey rings = "
                         "the wells of the set-up, yellow dots = the paint that was found), or correct the wells in the "
                         "set-up by hand; then `fd paint locate` again")
    if not usable:
        # Nothing is stored: one photo without the palette, or with one stray blob, used to move every well.
        sess.refused_say = (f"絵の具の位置は保存しませんでした。写真で見つかった絵の具は {total} 個のうち {n} 個だけで、パレットとは"
                            "判断できません。写真（picture）を見て、パレット全体が写っているか確かめてください。セットアップは元のままです。")
        raise PS.Refusal("locate", f"the wells were not stored: {why}. The set-up keeps the places it had",
                         "look at the picture (\"picture\": grey rings = the wells of the set-up, yellow dots = what was "
                         "found): is the whole palette in the photo, and unshadowed? Then `fd paint locate` again; a "
                         "palette that was moved more than 3 cm needs its wells written into the set-up by hand")
    lost = list(found["not_located"])
    other = list(found.get("other_colour") or [])
    by_paint = [name for name, w in found["wells"].items() if w["by"] == "paint"]
    if other:
        # (a well is never written onto paint of another colour than the one it has on record)
        res.setdefault("warnings", []).append(
            "another colour: " + "; ".join(
                f"{name} would be written where the photo shows hsv {found['wells'][name]['other_colour']['seen']}, and has "
                f"{found['wells'][name]['other_colour']['on_record']} on record" for name in other)
            + ". Such a well is not written: it keeps its place and is not located. When the palette was turned or "
              "pushed far, put it back; when the well was refilled with another paint, take its \"hsv\" out of the "
              "set-up; then `fd paint locate` again")
    if lost:
        res.setdefault("warnings", []).append(
            f"not located: {', '.join(lost)}. The photo shows no paint "
            + ("of its colour " if other else "") + "where the palette's move puts "
            + ("this well: it keeps" if len(lost) == 1 else "these wells: they keep")
            + " the place the set-up had, marked \"located\": false, and no paint is loaded from "
            + ("it" if len(lost) == 1 else "them") + " (stage \"wells\") until `fd paint locate` finds "
            + ("it" if len(lost) == 1 else "them") + " or the place is written into the set-up by hand")
    if not a.dry:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        for name, w in found["wells"].items():
            well = sess.setup["wells"][name]
            well["previous_xy"] = [float(v) for v in well["xy"]]       # the place before this photo
            if w["located"]:
                well["xy"] = w["xy"]
                well.pop("located", None)
                well.pop("expected_xy", None)
                if w.get("hsv"):                # (the colour it shows there: the next photo holds the well to it)
                    well["hsv"] = [int(v) for v in w["hsv"]]
            else:                   # no paint in the photo where the palette's move puts it: it keeps its place,
                well["located"] = False         # and nothing is loaded from it (PS.require_wells)
                well["expected_xy"] = w["expected_xy"]      # (where the next photo looks for it)
        sess.setup["located"] = {"at": stamp, "photo": res["photo"], "shift_mm": found["shift_mm"],
                                 "turn_deg": found["turn_deg"], "found": n, "missing": list(found["missing"]),
                                 "not_located": lost,
                                 "note": "fd paint locate: \"previous_xy\" of every well is where it was before"}
        PS.save_setup(sess.setup)
        res["saved"] = True
    sess.say = (f"パレットの絵の具 {total} 個のうち {n} 個を写真で見つけました（ずれ {found['shift_mm'][0]:+.0f}, {found['shift_mm'][1]:+.0f} mm、"
                f"回転 {found['turn_deg']:+.1f} 度）。"
                + (f"{len(by_paint)} 個（{'、'.join(by_paint)}）は隣とつながって写っていたので、パレットの動きに合わせた位置に"
                   "絵の具があることを確かめて置きました。" if by_paint else "")
                + (f"{len(lost)} 個（{'、'.join(lost)}）は写真で確認できなかったので、元の位置のまま「位置未確認」にしました。"
                   "位置を確認するまで、その絵の具の補充は断ります。" if lost else "")
                + (f"このうち {len(other)} 個（{'、'.join(other)}）は、置く場所に、記録と違う色の絵の具が写っていました"
                   "（隣の絵の具の上に位置を書かないためです。パレットを大きく動かしたときは、元の位置に戻してください）。" if other else "")
                + ("セットアップに保存しました（元の位置は previous_xy に残してあります）。" if not a.dry else "保存はしていません。"))


def _rinse_body(sess):
    a, pt = sess.a, sess.pt
    if not 0 <= int(a.blot_first) <= 8:
        raise PS.Refusal("options", f"--blot-first {a.blot_first}: 0 .. 8", "`fd paint rinse --help`")
    if a.blot_first and not sess.setup.get("blot"):
        raise PS.Refusal("setup", "--blot-first needs a blotting strip (\"blot\") in the set-up", PS.SETUP_NEXT)
    if a.blot_first:
        pt.blot(int(a.blot_first))
    pt.rinse()
    sess.say = "筆を洗いました。" + ("（水入れは1つです。水が濁ったら替えてください。）" if PS.one_dish(sess.setup) else "")


# ----------------------------------------------------------------------------- test strokes
def _test_actions(a, setup: dict) -> list:
    """The actions of `fd paint test` in the order they are done, read and checked BEFORE anything moves:
    [(kind, {...}), ...]. Refusal for an unknown action, a paint the set-up does not have, a bad number."""
    paints = setup["paints"]
    help_ = "`fd paint test --help` lists the actions"

    def paint(name):
        if name not in paints:
            raise PS.Refusal("paint", f"paint {name!r} is not in the set-up (paints: {', '.join(sorted(paints))})",
                             "`fd paint status` lists the paints of the set-up")
        return name

    def num(text, what, lo, hi):
        try:
            v = float(text)
        except ValueError:
            v = float("nan")
        if not (math.isfinite(v) and lo <= v <= hi):
            raise PS.Refusal("options", f"{what} {text!r}: a number in {lo:g} .. {hi:g}", help_)
        return v

    def press(v, what):
        try:
            PS.press_m(setup, v)
        except SafetyError as e:
            raise PS.Refusal("options", f"{what}: {e}", help_) from None
        return v

    num(a.ladder_len, "--ladder-len", 0.003, 0.06)
    num(a.len, "--len", 0.003, 0.15)
    press(a.press, "--press")
    try:
        zs = [float(v) for v in str(a.zs).split(",")]
        if not (1 <= len(zs) <= 8 and all(math.isfinite(z) and -6.0 <= z <= 30.0 for z in zs)):
            raise ValueError
    except ValueError:
        raise PS.Refusal("options", f"--zs {a.zs!r}: 1 to 8 heights in mm relative to the contact height, -6 .. 30", help_) from None
    out = []
    for act in a.do:
        kind, _, rest = str(act).partition(":")
        if kind == "rinse" and not rest:
            out.append(("rinse", {"as": act}))
        elif kind == "load" and rest:
            out.append(("load", {"paint": paint(rest), "as": act}))
        elif kind == "water" and (rest or "clean") in PS.DISHES:
            out.append(("water", {"which": rest or "clean", "as": act}))
        elif kind == "blot":
            if not setup.get("blot"):
                raise PS.Refusal("setup", "the action blot needs a blotting strip (\"blot\") in the set-up", PS.SETUP_NEXT)
            out.append(("blot", {"n": int(num(rest or "2", "blot:N", 1, 8)), "as": act}))
        elif kind == "stroke" and rest:
            x, y = _xy(rest, "stroke:x,y")
            out.append(("line", {"pts": [(x, y), (x - a.ladder_len, y)], "press": a.press, "as": act}))
        elif kind == "line" and rest:
            parts = rest.split(":")
            if len(parts) not in (2, 3):
                raise PS.Refusal("options", f"{act!r}: line:x0,y0:x1,y1[:press]", help_)
            p0, p1 = _xy(parts[0], "line: start"), _xy(parts[1], "line: end")
            if math.hypot(p1[0] - p0[0], p1[1] - p0[1]) < 0.002:
                raise PS.Refusal("options", f"{act!r}: shorter than 2 mm", help_)
            out.append(("line", {"pts": [p0, p1], "as": act,
                                 "press": press(num(parts[2], "line: press", 0.0, 10.0), act) if len(parts) == 3 else a.press}))
        elif kind == "swatch" and ":" in rest:
            name, at = rest.split(":", 1)
            x, y = _xy(at, "swatch:PAINT:x,y")
            out.append(("swatch", {"paint": paint(name), "pts": [(x, y), (x - a.ladder_len, y)], "press": a.press, "as": act}))
        elif kind == "ladder" and rest:
            x, y = _xy(rest, "ladder:x,y")
            out.append(("ladder", {"at": (x, y), "zs": zs, "len": a.ladder_len, "as": act}))
        else:
            raise PS.Refusal("options", f"unknown test action {act!r}", help_)
    for spec in a.swatch:
        if ":" not in str(spec):
            raise PS.Refusal("options", f"--swatch {spec!r}: PAINT:x,y", help_)
        name, at = str(spec).split(":", 1)
        x, y = _xy(at, "--swatch PAINT:x,y")
        out.append(("swatch", {"paint": paint(name), "pts": [(x, y), (x - a.len, y)], "press": a.press, "as": f"swatch:{spec}"}))
    for spec in a.ladder:
        x, y = _xy(spec, "--ladder x,y")
        out.append(("ladder", {"at": (x, y), "zs": zs, "len": a.ladder_len, "as": f"ladder:{spec}"}))
    if not out:
        raise PS.Refusal("options", "nothing to do: give --do ACTION, --swatch or --ladder", help_)
    return out


def _stroke_reach(pt, pts_xy, z_of, press_m: float, want: float | None = None):
    """SafetyError when the arm cannot make this stroke (as BrushRun.stroke plans it); else the pitch it gets
    (the tool's working pitch, or `want`, or the nearest one the arm can hold at every point of it)."""
    br = pt.br
    pts = [np.array([x, y, z_of(x, y) - press_m]) for x, y in pts_xy]
    for p in pts:
        br.guard(float(p[2]), "stroke")
    top = max(z_of(x, y) for x, y in pts_xy) + br.o.hover_m
    hover, end = np.array([pts[0][0], pts[0][1], top]), np.array([pts[-1][0], pts[-1][1], top])
    above = np.array([hover[0], hover[1], br.tz + br.o.above_h])
    pitch = br.pitch_at([hover, *pts, end, above], want=want)
    TL.check_pose(pt.arm, pt.arm.tool, pt.arm.solve(above, pitch, br.roll), min_height=br.o.above_h - 0.004, what="above the stroke")
    return pitch


def _plane_for_test(pt, strokes: bool) -> list:
    """The contact plane and the test strokes of a command: the warnings for its result. Strokes (line, stroke,
    swatch) run at the plane minus a press, so they need what a layer needs: a plane that is recorded for the
    brush as it is (Refusal, stage "contact", otherwise: a plane of another brush state, or one that does not
    say whose it is, presses by a wrong amount, and one of a re-taped brush by 5 mm more or less). Ladders are
    not refused: their rungs are heights around the plane, and they are how a plane is looked for when there
    is none to trust; they get the warning."""
    warn = []
    try:
        cs = pt.contact_state()
    except SafetyError as e:
        warn.append(f"no single brush pitch reaches the whole canvas ({e}): `fd paint check`")
        cs = PS.contact_record_state(pt.s, pt.arm.tool, None)      # (all but the pitch)
    if not cs["ok"]:
        if strokes:
            raise PS.Refusal("contact", cs["reason"] + ". Test strokes with it would press the brush by a wrong amount "
                             "(ladders are not refused: fd paint test with --ladder and a place x,y runs at heights "
                             "around the plane)",
                             cs["next"], contact={k: cs[k] for k in ("recorded", "ok", "why")})
        warn.append(cs["reason"] + ": the heights of these ladders are off by an unknown amount ("
                    + ("`fd paint contact --adopt`, or measure the plane" if not cs["recorded"] else
                       "`fd paint contact --adopt` again when the plane was measured with the brush as it is now, or "
                       "measure the plane" if cs.get("adopted") else
                       "`fd paint contact --scrap-laid --save` measures the plane") + ")")
    return warn


def _test_body(sess):
    a, pt, res = sess.a, sess.pt, sess.res
    acts = _test_actions(a, sess.setup)
    res.update(swatches=[], ladders=[], did=[], actions=[k for k, _ in acts])
    for kind, d in acts:            # a paint whose well `fd paint locate` did not find: before anything moves
        if kind in ("load", "swatch"):
            PS.require_wells(sess.setup, d["paint"])
    lowered = pt.set_lower(a.lower_mm)
    if lowered:
        res["contact_lowered_mm"] = lowered
    for kind, d in acts:            # what the options ask for, before the reach: depth, and the palette
        if kind in ("line", "swatch"):
            try:
                pt.depth_m(d["press"])         # press and --lower-mm together (PS.MAX_DEPTH_M)
            except SafetyError as e:
                raise PS.Refusal("options", f"{d['as']}: {e}", "the same command with a smaller --lower-mm or press "
                                                               "(`fd paint test --help`)") from None
            lines = [d["pts"]]
        elif kind == "ladder":
            x, y = d["at"]
            lines = [[(x, y + PS.LADDER_GAP_M * k), (x - d["len"], y + PS.LADDER_GAP_M * k)] for k in range(len(d["zs"]))]
        else:
            continue
        if not a.near_palette:
            for pts in lines:
                why = PS.near_palette(sess.setup, pts)
                if why:
                    raise PS.Refusal("options", f"{d['as']}: {why}",
                                     "x, y are metres in the robot's frame (`fd paint check` lists where the wells and the "
                                     "dishes are); a stroke that is MEANT to go there: the same command with --near-palette")
    for kind, d in acts:            # the reach of every stroke, before the first one is made
        try:
            if kind in ("line", "swatch"):
                _stroke_reach(pt, d["pts"], pt.zc, PS.press_m(sess.setup, d["press"]))
            elif kind == "ladder":
                x, y = d["at"]
                for k, dz in enumerate(d["zs"]):
                    yy = y + PS.LADDER_GAP_M * k
                    _stroke_reach(pt, [(x, yy), (x - d["len"], yy)],
                                  lambda _x, _y, _z=PS.contact_z(sess.setup, x, yy) + dz / 1000.0: _z, 0.0)
        except SafetyError as e:
            raise PS.Refusal("reach", f"{d['as']}: {e}", "`fd paint check` shows what is in reach; x, y are metres in the "
                                                         "robot's frame") from e
    warn = _plane_for_test(pt, any(kind in ("line", "swatch") for kind, _d in acts))
    if warn:
        res["warnings"] = warn
    # What lies on the canvas of the set-up lies where the picture is: painted on a SCRAP sheet only, on the
    # operator's word, and after a look at the sheet (as `fd paint contact`). The example of this command's own
    # help, typed as printed with the finished Kaminarimon uncovered, painted two lines on it (second review of
    # 2026-11-13). Outside the canvas nothing changes.
    on = []                     # [(the action as typed, the table window its paint lands in)]
    for kind, d in acts:
        if kind in ("line", "swatch") and PS.on_canvas(sess.setup, d["pts"]):
            on.append((d["as"], PS.stroke_window(d["pts"])))
        elif kind == "ladder":
            x, y = d["at"]
            rungs = [{"y": y + PS.LADDER_GAP_M * k, "x0": x, "x1": x - d["len"]} for k in range(len(d["zs"]))]
            if any(PS.on_canvas(sess.setup, [(r["x0"], r["y"]), (r["x1"], r["y"])]) for r in rungs):
                on.append((d["as"], PS.ladder_window({"rungs": rungs})))
    if on:
        res["on_canvas"] = [name for name, _w in on]
    if a.dry:
        res.update(stage="planned", next="the same command without --dry")
        sess.say = f"試し描き {len(acts)} 手順を確認しました（届きます）。"
        if on and not a.scrap_laid:
            res["next"] = ("lay a scrap sheet over the canvas, then the same command with --scrap-laid and without --dry "
                           f"({len(on)} of its strokes lie on the canvas of the set-up, where the picture is)")
            res.setdefault("warnings", []).append(
                f"on the canvas of the set-up: {', '.join(res['on_canvas'])}. Without --dry this is refused unless "
                "--scrap-laid says that a scrap sheet lies there")
            sess.say += (f"そのうち {len(on)} 個（線・はしご）はキャンバス（絵のある場所）の上です。実行の前に捨て紙を1枚置いて、"
                         "--scrap-laid を付けてください。")
        return
    if on and not a.scrap_laid:
        sess.refused_say = (f"試し描きは実行しませんでした。{len(on)} 個の線・はしごがキャンバス（絵のある場所）の上です。そこに捨て紙を"
                            "1枚置いてから、--scrap-laid を付けて実行してください。")
        raise PS.Refusal("contact", f"{', '.join(res['on_canvas'])}: on the canvas of the set-up, where the picture is. A scrap "
                                    "sheet must lie there, and a photo cannot tell a blank painting sheet from scrap "
                                    "(--scrap-laid is your word that it does)",
                         "lay a scrap sheet over the canvas, then the same command with --scrap-laid (test strokes outside "
                         "the canvas need no word: `fd paint status` shows where the canvas lies)")
    if on:
        # the look at the sheet, before anything is loaded: bare paper where every one of them lands, or nothing
        res["photos"] = {"before": pt.look("test_before")}
        if pt.last_photo is None:
            res["sheet"] = None
            res.setdefault("warnings", []).append("no camera: the sheet was not looked at before the test strokes were painted")
        else:
            cam = _cam_model()
            rows = [{"name": name, **PS.bare_paper(pt.last_photo, cam, win, pt.tz)} for name, win in on]
            res["sheet"] = {"ok": all(r["bare"] for r in rows),
                            "places": [{k: r[k] for k in ("name", "bare", "not_paper_pct")} for r in rows]}
            if not res["sheet"]["ok"]:
                sess.refused_say = ("試し描きは中止しました。線を引く場所が白い紙ではありません（絵や古い線が見えています）。"
                                    "キャンバスに新しい捨て紙を1枚置くか、古い線から離れた場所を指定して、同じコマンドを実行して"
                                    "ください。何も描いていません。")
                sess.skip_photo = True
                raise PS.Refusal("contact", "the sheet is not bare paper where these test strokes would be painted: "
                                 + "; ".join(f"{r['name']}: {r['why']}" for r in rows if not r["bare"])
                                 + ". Nothing was painted and nothing loaded: lay a scrap sheet over the canvas",
                                 "lay a FRESH scrap sheet over the canvas (or give places away from the old marks), then "
                                 f"the same command again (the photo that was looked at: {res['photos']['before']})")
    t0 = pt.clock()
    for kind, d in acts:
        if kind == "rinse":
            pt.rinse()
        elif kind == "load":
            pt.load(d["paint"])
        elif kind == "water":
            pt.water(d["which"])
        elif kind == "blot":
            pt.blot(d["n"])
        elif kind == "line":
            r = pt.stroke_xy(d["pts"], d["press"])
            res["swatches"].append({"paint": pt.paint, "at": list(d["pts"][0]), "to": list(d["pts"][1]), "kind": "stroke",
                                    "press": d["press"], **r})
        elif kind == "swatch":
            ld = pt.load(d["paint"])
            r = pt.stroke_xy(d["pts"], d["press"])
            res["swatches"].append({"paint": d["paint"], "at": list(d["pts"][0]), "to": list(d["pts"][1]), "load": ld, **r})
        elif kind == "ladder":
            rungs = []
            res["ladders"].append({"at": list(d["at"]), "rungs": rungs, "paint": pt.paint})
            pt.lower_m, low = 0.0, pt.lower_m             # (a ladder's heights are relative to the plane AS MEASURED)
            try:
                pt.ladder(d["at"][0], d["at"][1], d["zs"], length=d["len"], relative=True, into=rungs)
            finally:
                pt.lower_m = low
        res["did"].append(d["as"])
        _log(f"{res['did'][-1]}: done  t={pt.clock() - t0:.0f}s")
    sess.say = f"試し描きを {len(res['did'])} 手順行いました。"


# ----------------------------------------------------------------------------- contact
DEFAULT_ZS = "12,9,6,3"
DEFAULT_LADDER_LEN = 0.010
SCRAP_NEXT = ("lay a scrap sheet over the whole canvas, then `fd paint contact --scrap-laid{more}` (the ladders are painted "
              "where the picture is; the first photo is looked at before any paint is loaded)")


# (in words: which two files these are is not known here; a `next` that begins with "fd " is a command to type)
READ_NEXT = ("give two photo files that can be read after --read: the photo taken before the ladders were painted, then "
             "the one taken after them (`fd paint contact --help`)")


def _at_list(a) -> list:
    out = []
    for item in (a.at or ["canvas"]):
        out.append("canvas" if str(item) == "canvas" else _xy(item, "--at"))
    return out


def _contact_plan(a, setup: dict, table_z: float, record: dict | None = None) -> list:
    """The ladders of a contact command: from its options, or (for --read) from the record beside the photos
    when the options say nothing about places, heights and length. Refusal (stage "options") for heights or a
    length that cannot be used."""
    if record is not None and a.at is None and a.zs is None and a.ladder_len is None:
        zs_in, length = record["zs_mm"], record["ladder_len_m"]
        at, names = [tuple(float(v) for v in p["at"]) for p in record["places"]], [p.get("name") for p in record["places"]]
    else:
        zs_in = str(a.zs if a.zs is not None else DEFAULT_ZS).split(",")
        length, at, names = (DEFAULT_LADDER_LEN if a.ladder_len is None else a.ladder_len), _at_list(a), None
    try:
        zs = PS.check_zs([float(v) for v in zs_in], table_z)
    except ValueError:
        raise PS.Refusal("options", f"--zs {a.zs!r}: plan heights in mm, like \"12,9,6,3\"", "`fd paint contact --help`") from None
    if not (math.isfinite(float(length)) and 0.005 <= float(length) <= 0.03):
        raise PS.Refusal("options", f"--ladder-len {length}: 0.005 .. 0.03 m", "`fd paint contact --help`")
    places = PS.contact_places(setup, at, zs, float(length))
    for p, name in zip(places, names or []):
        if isinstance(name, str) and name:
            p["name"] = name                     # ("near-left" ..., as when they were painted)
    return places


def _same_ladders(places: list, record: dict) -> bool:
    """Are these the ladders the record says were painted (starts within 0.5 mm, the same heights and length)?"""
    rp = record["places"]
    if len(rp) != len(places):
        return False
    for p, q in zip(places, rp):
        if max(abs(float(a_) - float(b_)) for a_, b_ in zip(p["at"], q["at"])) > 0.0005:
            return False
        zs = [r["plan_z_mm"] for r in p["rungs"]]
        if len(zs) != len(record["zs_mm"]) or max(abs(a_ - float(b_)) for a_, b_ in zip(zs, record["zs_mm"])) > 1e-6:
            return False
        if abs((p["rungs"][0]["x0"] - p["rungs"][0]["x1"]) - float(record["ladder_len_m"])) > 1e-6:
            return False
    return True


def _contact_report(res: dict, read: dict, picture, saved: bool, sim: bool) -> str:
    """Put a reading into a result; returns the Japanese sentence."""
    PS.strip_private(read)
    res.update(places=read["places"], read=read["read"], plane=read["plane"], usable=read["usable"], picture=picture,
               saved=saved)
    if read["warnings"]:
        res["warnings"] = list(res.get("warnings", [])) + read["warnings"]
    got = [p for p in read["places"] if p["read"]]
    touches = "、".join(f"{p['first_touch_mm']:.1f}" for p in got)
    if not read["usable"]:
        res.update(ok=False, stage="contact", reason=read["why"],
                   next="look at the picture (\"picture\"); lay a fresh scrap sheet and repeat `fd paint contact --scrap-laid` "
                        "(other --zs when the ladders were too high or too low)")
        return f"紙の高さを読み取れませんでした（{read['why']}）。"
    pl = read["plane"]
    say = f"紙の高さ（筆が最初に触れる高さ）を {len(got)} か所で読みました: {touches} mm。"
    if pl:
        say += (f"平面は中心で {pl['z0'] * 1000:.1f} mm、奥へ {pl['bx'] * 1000:+.1f} mm/m、横へ {pl['by'] * 1000:+.1f} mm/m、"
                f"ずれ最大 {pl['max_residual_mm']:.1f} mm。")
    say += "セットアップに保存しました。" if saved else "保存していません（保存は --save）。"
    return say


def _may_store(read: dict) -> None:
    """Refusal (stage "contact") when a reading is no measurement to store."""
    n = sum(1 for p in read["places"] if p["read"])
    if not read["usable"] or read["plane"] is None or n < PS.CONTACT_MIN_PLACES:
        raise PS.Refusal("contact", "the plane is not stored: " + (read["why"] or f"only {n} place(s) were read, "
                         f"{PS.CONTACT_MIN_PLACES} are needed (a plane through fewer has no residual to judge it by)"),
                         "lay a FRESH scrap sheet over the whole canvas, then `fd paint contact --scrap-laid --save` (the five "
                         "places of the canvas)")
    if read["plane"]["max_residual_mm"] > PS.CONTACT_MAX_RESIDUAL_MM:
        raise PS.Refusal("contact", f"the plane is not stored: a residual of {read['plane']['max_residual_mm']:.1f} mm",
                         "is the sheet flat, is the brush rigid? Then a FRESH scrap sheet over the whole canvas and `fd paint "
                         "contact --scrap-laid --save`")


def _contact_save(a, setup: dict, read: dict, record: dict, photos) -> None:
    """Write the plane into the set-up with the brush state it holds for; Refusal when it must not be stored."""
    _may_store(read)
    setup["contact"] = PS.contact_block(read["plane"], record, read, photos, old=setup.get("contact"))
    PS.save_setup(setup)


def _contact_read(a) -> dict:
    """`fd paint contact --read BEFORE AFTER`: no arm; the same reading from two saved photos. The ladders (where,
    at which heights) and the brush state they were painted with come from the record beside BEFORE; --save
    stores the plane for THAT brush state, and only while the registered tool still is that brush. --dry
    reads and reports and writes nothing (no picture, no set-up)."""
    import cv2
    setup = PS.load_setup()
    cfg = load_config()
    table_z = float(cfg.get("table_z", 0.0))
    rec = PS.load_contact_record(a.read[0])
    places = _contact_plan(a, setup, table_z, rec)
    res = {"ok": True, "cmd": "contact", "stage": "contact", "photos": list(a.read),
           "record": None if rec is None else rec["path"]}
    warnings = []
    as_painted = True
    if rec is None:
        warnings.append(f"no record beside the photos ({PS.contact_record_path(a.read[0])}): the ladders are taken from "
                        "--at / --zs, and which brush painted them is not known (the plane cannot be stored)")
    else:
        res.update(painted_with=rec["brush"], painted_at=rec.get("at"))
        as_painted = _same_ladders(places, rec)
        if not as_painted:
            warnings.append("--at / --zs / --ladder-len give other ladders than the record beside the photos says were "
                            f"painted ({rec['path']}): the photos are read as the options say (leave them away to read "
                            "them as painted)")
        unfinished = [p for p in rec["places"] if int(p.get("rungs_done", p.get("rungs", 0))) < int(p.get("rungs", 0))]
        if unfinished and as_painted:
            names = {str(p.get("name")) for p in unfinished}
            places = [p for p in places if p["name"] not in names]
            warnings.append("not read: " + ", ".join(sorted(names)) + " (the record says these ladders were not finished)")
            if not places:
                raise PS.Refusal("contact", "the record beside the photos says that no ladder was finished",
                                 SCRAP_NEXT.format(more=" --save" if a.save else ""))
    if a.save:
        # whether this plane may be stored at all is said before the photos are read
        if rec is None:
            raise PS.Refusal("contact", "the plane of these photos cannot be stored: there is no record of the brush state "
                             f"the ladders were painted with ({PS.contact_record_path(a.read[0])} is missing), and a plane "
                             "holds for one brush state only",
                             "the same command without --save shows the reading. For a plane that is in the set-up "
                             "already and was measured with the brush as it is: `fd paint contact --adopt`; else measure "
                             "again: " + CONTACT_NEXT)
        if not as_painted:
            raise PS.Refusal("options", "--save stores the plane of the ladders as they were painted: --at / --zs / "
                                        "--ladder-len differ from the record beside the photos",
                             f"fd paint contact --read {a.read[0]} {a.read[1]} --save   (without --at / --zs)")
        tool, _file = _registered_tool(a.sim)
        diff = PS.record_differs(rec, tool)
        if diff:
            raise PS.Refusal("contact", "the plane of these photos cannot be stored for the brush that is on: "
                             + "; ".join(diff) + ". The ladders were painted with another brush state",
                             CONTACT_NEXT, painted_with=rec["brush"])
        if not a.dry:
            _may_write_setup(a, shlex.join(["contact", "--read", a.read[0], a.read[1], "--save"]))
    imgs = []
    for path in a.read:
        img = cv2.imread(path)
        if img is None:
            raise PS.Refusal("options", f"the photo {path} cannot be read", READ_NEXT)
        imgs.append(img)
    if imgs[0].shape != imgs[1].shape:
        raise PS.Refusal("options", f"the two photos differ in size ({imgs[0].shape[1]}x{imgs[0].shape[0]} and "
                                    f"{imgs[1].shape[1]}x{imgs[1].shape[0]})", READ_NEXT)
    cam = _cam_model()
    read = PS.read_contact(imgs[0], imgs[1], cam, setup, places, table_z)
    picture = None
    if not a.dry:
        picture = PS.contact_picture(imgs[1], cam, setup, read, table_z,
                                     os.path.join(_out_dir(a.sim), f"{time.strftime('%H%M%S')}_contact_read.jpg"))
    if warnings:
        res["warnings"] = warnings
    saved = False
    if a.save and read["usable"]:
        _may_store(read)
        if not a.dry:
            # the brush state the plane holds for is the one of the RECORD (when the ladders were painted), not the
            # tool file of now; that the two are still the same brush was asked above
            _contact_save(a, setup, read, {**rec["brush"], "at": rec.get("at")}, a.read)
            saved = True
    say = _contact_report(res, read, picture, saved, a.sim)
    if a.dry:
        res["dry"] = True
        if res["ok"]:
            res.update(stage="planned", would_save=bool(a.save and read["usable"]),
                       next="the same command without --dry" + ("" if a.save else " (--save stores the plane)"))
        say = say.replace("保存していません（保存は --save）。", "何も書き込んでいません（--dry）。")
    elif a.save and not saved and res["ok"]:
        res.update(ok=False, reason="the plane was not stored")
    res["say"] = say + "アームは動かしていません。"
    if a.sim:
        res["sim"] = True
    return res


def _contact_adopt(a) -> dict:
    """`fd paint contact --adopt`: no motion, nothing measured. The plane of the set-up gets the brush state as it
    is NOW as what it holds for: the operator's word that it was measured with exactly this brush. For a plane
    from before such records were kept (the rig's set-up of 2026-11-13: measured that evening with the jaw
    squeezed shut, by hand-edited files). A plane without a record passed for every brush state; now no stroke
    is painted with it until it is adopted or measured. Refused for a plane whose record came from ladders and
    names another brush state: that is what the record is there to stop."""
    if a.read or a.save or a.at or a.zs is not None or a.paint or a.scrap_laid:
        raise PS.Refusal("options", "--adopt stands alone (with --dry to see what it would write): it measures nothing "
                                    "and reads no photo", "fd paint contact --adopt --dry")
    setup = PS.load_setup()
    arm, limits = _kin_arm(a.sim)
    try:
        pt = PS.Painter(arm, setup, unmeasured_tip_ok=True)
        try:
            pitch = pt.pitch_for_canvas()
        except SafetyError as e:
            raise PS.Refusal("reach", f"no single brush pitch reaches the whole canvas ({e})",
                             "`fd paint check` shows what is out of reach") from e
        tool = arm.tool
        record = PS.brush_state(tool, pitch)
        record["limits_from"] = limits
        old = setup["contact"]
        was = old.get("measured")
        res = {"ok": True, "cmd": "contact", "stage": "contact", "adopt": True, "saved": False, "tool": tool.name,
               "pitch": round(pitch, 1), "plane": {k: old[k] for k in ("z0", "x0", "y0", "bx", "by") if k in old},
               "brush_state": record, "limits_from": limits}
        if a.sim:
            res["sim"] = True
        if isinstance(was, dict):
            diff = PS.brush_differs(was, tool, pitch)
            if not diff:
                res.update(next="`fd paint check`; " + PLAN_NEXT,
                           say="紙の高さ（contact）は、すでに今の筆の状態で記録されています。何も書き換えていません。アームは動かしていません。")
                return res
            if not was.get("adopted"):
                raise PS.Refusal("contact", "the plane cannot be adopted: it has a MEASURED record, and that names another "
                                 "brush state (" + "; ".join(diff) + ")", CONTACT_NEXT)
            res["replaces"] = {k: was.get(k) for k in ("tool", "tip_offset", "gripper_raw", "gripper_goal_raw", "on_jaw", "at")}
        warnings = []
        if not BW.tip_is_measured(tool):
            warnings.append("the tip of this tool was not measured: when `fd calib-air --save` gives another tip, the plane has "
                            "to be measured (or adopted) again")
        note = PS.jaw_hold_note(tool, arm.cfg)
        if note:
            warnings.append(note + " Do that BEFORE adopting: the record keeps whether the jaw is held")
        if warnings:
            res["warnings"] = warnings
        if a.dry:
            res.update(stage="planned", next="fd paint contact --adopt",
                       say="今の筆の状態を、いまの紙の高さ（contact）の記録として書き込む予定です（まだ書いていません）。"
                           "アームは動かしていません。")
            return res
        _may_write_setup(a, "contact --adopt")
        block = PS.adopt_block(old, record)
        if isinstance(was, dict):
            block["measured"]["replaces"] = res["replaces"]
        setup["contact"] = block
        PS.save_setup(setup)
        res.update(saved=True, next="`fd paint check`; " + PLAN_NEXT,
                   say=(f"紙の高さ（contact）を、今の筆（ツール「{tool.name}」、先端 "
                        f"{[round(v * 1000, 1) for v in tool.tip_offset]} mm、角度 {pitch:.0f} 度）のものとして記録しました。"
                        "測定はしていません: この平面を今の筆の状態で測った、というあなたの確認の記録です。アームは動かしていません。"))
        return res
    finally:
        arm.close()


def _contact_body(sess):
    a, pt, res, setup = sess.a, sess.pt, sess.res, sess.setup
    places = _contact_plan(a, setup, pt.tz)
    zs = [r["plan_z_mm"] for r in places[0]["rungs"]]
    length = round(places[0]["rungs"][0]["x0"] - places[0]["rungs"][0]["x1"], 5)
    paint = a.paint or pt.paint or next(iter(setup["paints"]), None)
    if paint not in setup["paints"]:
        raise PS.Refusal("paint", f"paint {paint!r} is not in the set-up (paints: {', '.join(sorted(setup['paints']))})",
                         "`fd paint status` lists the paints of the set-up: give one of them after --paint")
    PS.require_wells(setup, paint)
    more = (" --save" if a.save else "") + (f" --paint {a.paint}" if a.paint else "")
    if a.save:
        _may_write_setup(a, "contact --save")
        if len(places) < PS.CONTACT_MIN_PLACES:
            raise PS.Refusal("options", f"--save needs at least {PS.CONTACT_MIN_PLACES} places (a plane through fewer has no "
                                        "residual to judge it by)",
                             "lay a scrap sheet over the whole canvas, then `fd paint contact --scrap-laid --save` (without "
                             "--at: the five places of the canvas)")
        if sess.photo is None:
            raise PS.Refusal("camera", "--save needs the two photos: there is no camera with the simulated arm",
                             "`fd paint contact --scrap-laid --save` on the real arm; `fd --sim paint contact --scrap-laid` "
                             "rehearses the moves")
    try:
        pitch = pt.pitch_for_canvas()
    except SafetyError as e:
        raise PS.Refusal("reach", f"no single brush pitch reaches the whole canvas ({e})", "`fd paint check` shows what is out of reach") from e
    warnings = []
    for p in places:
        if not p["inside_canvas"]:
            warnings.append(f"the ladder at {p['name']} lies outside the canvas: measure where the picture will be")
        for r in p["rungs"]:                 # every rung at the pitch the picture is painted with, before anything moves
            z = r["plan_z_mm"] / 1000.0
            try:
                got = _stroke_reach(pt, [(r["x0"], r["y"]), (r["x1"], r["y"])], lambda _x, _y, _z=z: _z, 0.0, want=pitch)
                for xx in (r["x0"], r["x1"]):
                    ko = PS._keepout_at(sess.arm.cfg, (xx, r["y"], z))
                    if ko and ko not in pt.br.o.ignore_keepouts:
                        raise SafetyError(f"({xx:.3f}, {r['y']:.3f}) lies inside the keep-out '{ko}'")
                if abs(got - pitch) > 0.05:
                    raise SafetyError(f"it can only be reached at pitch {got:.1f}, the canvas is painted at {pitch:.1f} "
                                      "(a contact plane holds for one pitch)")
            except SafetyError as e:
                raise PS.Refusal("reach", f"the ladder at {p['name']} cannot be painted: rung at ({r['x0']:.3f}, {r['y']:.3f}), "
                                          f"{r['plan_z_mm']:g} mm: {e}", "`fd paint check` shows what is in reach: give "
                                                                         "places in reach after --at (metres, as x,y)") from e
    # (what is out of reach was said first; now what is in reach but must not be painted like this)
    if a.distrust_plane:
        warnings.append("--distrust-plane: the rungs were not held against the paper height of the set-up (only against "
                        "the limit of brush work under the table)")
    else:
        PS.check_rungs(setup, places)            # no rung more than 6 mm below the paper the set-up already knows
    if not a.near_palette:
        for p in places:
            for r in p["rungs"]:
                why = PS.near_palette(setup, [(r["x0"], r["y"]), (r["x1"], r["y"])])
                if why:
                    raise PS.Refusal("options", f"the ladder at {p['name']}: {why}",
                                     "x, y are metres in the robot's frame (`fd paint check` lists where the wells and the "
                                     "dishes are); a ladder that is MEANT to stand there: the same command with --near-palette")
    res.update(paint=paint, pitch=round(pitch, 1), zs_mm=zs,
               ladders=[{"name": p["name"], "at": p["at"], "inside_canvas": p["inside_canvas"], "rungs_done": 0} for p in places])
    if warnings:
        res["warnings"] = warnings
    if a.dry:
        res.update(stage="planned", next=SCRAP_NEXT.format(more=more))
        sess.say = (f"紙の高さの測定を {len(places)} か所で計画しました（すべて届きます）。実行の前に、キャンバスの上に捨て紙を"
                    "1枚置いてください（--scrap-laid）。")
        return
    if not a.scrap_laid:
        on_canvas = any(PS.canvas_distance(setup, xx, r["y"]) < PS.REST_MARGIN_M for p in places for r in p["rungs"]
                        for xx in (r["x0"], r["x1"]))
        where = ("at the corners and the centre of the canvas, where the picture is" if a.at is None else
                 "on the canvas, where the picture is" if on_canvas else "on what lies at these places")
        sess.refused_say = ("紙の高さの測定は実行しませんでした。試し線は" + ("キャンバス（絵のある場所）" if on_canvas else "指定した場所")
                            + "に引きます。そこに捨て紙を1枚置いてから、--scrap-laid を付けて実行してください。")
        raise PS.Refusal("contact", f"the ladders are painted {where}: a scrap sheet must lie there, and a photo cannot tell "
                                    "a blank painting sheet from scrap (--scrap-laid is your word that it does)",
                         SCRAP_NEXT.format(more=more) if a.at is None and a.zs is None else
                         "lay a scrap sheet where the ladders go, then the same command with --scrap-laid")
    res["photos"] = {"before": pt.look("contact_before")}
    before = pt.last_photo
    if before is None:
        res["sheet"] = None
        res.setdefault("warnings", []).append("no camera: the sheet was not looked at before the ladders were painted")
    else:
        # the look at the sheet, before any paint is loaded: bare paper in every ladder window, or nothing is painted
        sheet = PS.sheet_check(before, _cam_model(), places, pt.tz)
        res["sheet"] = {"ok": sheet["ok"], "places": [{k: p[k] for k in ("name", "bare", "not_paper_pct")} for p in sheet["places"]]}
        if not sheet["ok"]:
            sess.refused_say = ("紙の高さの測定は中止しました。試し線を引く場所が白い紙ではありません（絵や古い線が見えています）。"
                                "キャンバス全体に新しい捨て紙を1枚置いてから、同じコマンドを実行してください。何も描いていません。")
            sess.skip_photo = True
            raise PS.Refusal("contact", sheet["why"] + ". Nothing was painted: lay a scrap sheet over the canvas",
                             "lay a FRESH scrap sheet over the whole canvas, then the same command again (the photo that was "
                             f"looked at: {res['photos']['before']})")
    brush = PS.brush_state(sess.arm.tool, pitch)
    painted = {"at": None}

    def record(after_photo=None):
        """The record beside the photos: which brush painted these ladders, where, at which heights."""
        if not res["photos"].get("before"):
            return None
        rows = [{"name": p["name"], "at": p["at"], "rungs": p["rungs"], "rungs_done": lad["rungs_done"]}
                for p, lad in zip(places, res["ladders"])]
        return PS.write_contact_record(res["photos"]["before"], brush, rows, zs, length, paint, setup,
                                       after_photo=after_photo, at=painted["at"], sim=a.sim)

    try:
        pt.load(paint)
        for p, lad in zip(places, res["ladders"]):
            done = []
            lad["rungs"] = done
            try:
                pt.ladder(p["at"][0], p["at"][1], [r["plan_z_mm"] for r in p["rungs"]], length=length,
                          relative=False, pitch=pitch, into=done)
            finally:
                lad["rungs_done"] = len(done)
            _log(f"ladder {p['name']}: done  t={pt.clock() - pt.t0:.0f}s")
    finally:
        painted["at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        if any(lad["rungs_done"] for lad in res["ladders"]):
            res["record"] = record()         # (also after a stop: what is on the sheet was painted with this brush)

    def after(sess_):
        res["photos"]["after"] = res.pop("photo", None)
        if sess_.photo is None:
            res.update(read=None, plane=None, saved=False)
            sess_.say = (f"紙の高さの測定用に {len(places)} か所へ試し線を引きました。カメラがないので、読み取りはしていません"
                         "（写真があれば fd paint contact --read で読めます）。")
            return
        # The command that reads the two photos again: {after} is the photo after the ladders. With the photo taken
        # it is a whole command ("reread"); without it the file is one this program does not know yet, and the
        # command is said in words (`... --read BEFORE AFTER.jpg` typed as it was printed is refused: T12)
        if res.get("record"):
            again = "fd paint contact --read {before} {after}" + (" --save" if a.save else "")
        else:           # (no record could be written: the ladders have to be named again)
            again = ("fd paint contact --read {before} {after} --zs " + ",".join(f"{z:g}" for z in zs)
                     + "".join(f" --at {p['at'][0]},{p['at'][1]}" for p in places))
        if before is None or pt.last_photo is None or pt.last_photo is before:
            words = again.format(before=shlex.quote(str(res["photos"]["before"])), after="(the new photo)")
            raise PS.Refusal("camera", "the ladders are painted, but the photo after them was not taken ("
                             + (res.get("photo_error") or "the arm did not come back to the photo pose") + ")",
                             "`fd paint look` takes it (its result names the file: \"photo\"); then the two photos are read "
                             f"with: {words}, the file of that new photo in the place of the words in brackets")
        if res.get("record"):
            res["record"] = record(res["photos"]["after"]) or res["record"]
        res["reread"] = again.format(before=shlex.quote(str(res["photos"]["before"])),
                                     after=shlex.quote(str(res["photos"]["after"])))
        cam = _cam_model()
        read = PS.read_contact(before, pt.last_photo, cam, setup, places, pt.tz)
        picture = PS.contact_picture(pt.last_photo, cam, setup, read, pt.tz,
                                     os.path.join(pt.out_dir or _out_dir(a.sim), f"{time.strftime('%H%M%S')}_contact_read.jpg"))
        saved = False
        if a.save and read["usable"]:
            _contact_save(a, setup, read, {**brush, "at": painted["at"]}, [res["photos"]["before"], res["photos"]["after"]])
            saved = True
        sess_.say = _contact_report(res, read, picture, saved, a.sim)
        if a.save and not saved and res["ok"]:
            res.update(ok=False, reason="the plane was not stored")

    sess.after = after


# ----------------------------------------------------------------------------- run
def _plan_runs(plan: dict, path: str) -> list | None:
    """The plan's OWN command for every layer, with the plan's path in it: its "run" list (written by the module
    that made the plan: the bands of the Kaminarimon with `--lower-mm 1.5`, its written lines with `--reload-mm
    110`). None when the plan has no such list, or one that does not fit its layers (it is then not trusted)."""
    runs = plan.get("run")
    if not (isinstance(runs, list) and len(runs) == len(plan["layers"]) and all(isinstance(c, str) for c in runs)):
        return None
    out = []
    for li, cmd in enumerate(runs):
        try:
            words = shlex.split(cmd)
        except ValueError:
            return None
        if words[:3] != ["fd", "paint", "run"] or words[4:6] != ["--layer", str(li)]:
            return None
        out.append(shlex.join(words[:3] + [path] + words[4:]))
    return out


def _layer_cmd(plan: dict, path: str, li: int) -> str:
    """The command that paints layer li: the plan's own when it has one (_plan_runs), else the plain one."""
    runs = _plan_runs(plan, path)
    return runs[li] if runs else shlex.join(["fd", "paint", "run", path, "--layer", str(li)])


def _plan_notes(res: dict, plan: dict, path: str, state: dict | None = None, li: int | None = None) -> None:
    """Into a result (preview, run --dry): the plan's own commands ("run"), what its maker left open ("open"),
    and, with the state, the layers (li: only that one) whose strokes are recorded as painted under ANOTHER plan
    file ("painted_elsewhere" and a warning: progress is kept per file, so a copy of a painted plan starts at 0)."""
    runs = _plan_runs(plan, path)
    if runs:
        res["run"] = runs
    if isinstance(plan.get("open"), list) and all(isinstance(t, str) for t in plan["open"]):
        # (the plan's maker writes PLAN where the file's name goes, as in its "run" list; here the file is known,
        # and a command in backticks is one that works as it is typed)
        res["open"] = [t.replace("`fd paint run PLAN ", f"`fd paint run {shlex.quote(path)} ") for t in plan["open"]]
    there = PS.painted_elsewhere(state, path, plan) if state else []
    if li is not None:
        there = [e for e in there if e["layer"] == plan["layers"][li].get("name")]
    if there:
        res["painted_elsewhere"] = there
        res.setdefault("warnings", []).append(
            "the same strokes are recorded as painted under another plan file: "
            + "; ".join(f"layer {e['layer']!r} {e['strokes']} of {e['of']} ({e['plan']})" for e in there[:4])
            + (f" and {len(there) - 4} more" if len(there) > 4 else "")
            + ". Progress is kept per plan file: this file starts at stroke 0. If the picture on the table is that "
              "one, paint from the file it was painted from")


def _run_body(sess):
    a, pt, res, state, setup = sess.a, sess.pt, sess.res, sess.state, sess.setup
    plan = _load_plan(a.plan)
    pt.check_plan(plan)
    li = _layer_index(plan, a.layer, a.plan)
    layer = plan["layers"][li]
    n = len(layer["strokes"])
    # (by the plan's REAL path, and under whatever spelling of it an earlier call stored: PS.progress_of)
    start = a.start if a.start is not None else PS.progress_of(state, a.plan, plan, li)
    stop = n if a.stop is None else int(a.stop)
    if not (0 <= start <= n and start <= stop <= n):
        raise PS.Refusal("options", f"--start {start} / --stop {stop}: the layer has strokes 0 .. {n - 1}", "`fd paint run --help`")
    if not (math.isfinite(float(a.max_min)) and 0.1 <= float(a.max_min) <= 60.0):
        raise PS.Refusal("options", f"--max-min {a.max_min}: 0.1 .. 60 minutes", "`fd paint run --help`")
    PS.reload_rule(pt.brush, a.reload_mm, setup["paints"].get(layer["paint"]))
    lowered = pt.set_lower(a.lower_mm)
    pt.check_depths(layer, start, stop)         # press and --lower-mm together (PS.MAX_DEPTH_M)
    try:
        pitch = pt.pitch_for_canvas()
    except SafetyError as e:
        raise PS.Refusal("reach", f"no single brush pitch reaches the whole canvas ({e})", "`fd paint check` shows what is out of reach") from e
    carried = PS.UNKNOWN_PAINT if pt.unknown else pt.paint      # (not known: the first load is preceded by a wash)
    budget = PS.layer_budget(setup, layer, carried, 1e9, start, stop, a.reload_mm)
    times = PS.times_of(state)
    minutes = round(PS.estimate_s(budget, times) / 60.0, 1)
    res.update(layer=layer["name"], paint=layer["paint"], start=start, of=n, pitch=round(pitch, 1),
               budget={k: budget[k] for k in ("strokes", "travel_mm", "loads", "blots", "washes")}, minutes=minutes,
               calls=max(1, int(math.ceil(minutes / float(a.max_min)))) if budget["strokes"] else 0,
               minutes_all=PS.plan_budget(setup, plan, state, carried, a.reload_mm)["minutes"], times=times)
    if lowered:
        res["contact_lowered_mm"] = lowered
    pt.require_plane("painting")                # recorded for the brush as it is: no record, no stroke
    PS.require_wells(setup, layer["paint"])     # its well was found by the last `fd paint locate`
    bad = pt.check_strokes(layer, start, stop)
    if bad:
        raise PS.Refusal("reach", f"{len(bad)} stroke(s) of layer {layer['name']!r} cannot be made: stroke {bad[0]['stroke']}: "
                                  f"{bad[0]['why']}", "`fd paint check` shows the reach; move the sheet or make the plan anew",
                         out_of_reach=bad[:10])
    # The same layer again, with what THIS call was given: a continuation without --reload-mm reloaded the written
    # lines every 140 mm instead of every 110 (3 loads of a layer instead of 8), one without --lower-mm painted the
    # rest of the bands thin (second review of 2026-11-13).
    whole = (shlex.join(["fd", "paint", "run", a.plan, "--layer", str(li)]) + (f" --lower-mm {lowered:g}" if lowered else "")
             + (f" --reload-mm {float(a.reload_mm):g}" if a.reload_mm is not None else "")
             + (f" --max-min {float(a.max_min):g}" if float(a.max_min) != 8.0 else ""))
    # ... and with its --stop: a call that was limited to stroke N and ended early (the time limit, a stop) named
    # the command WITHOUT it, which painted the rest of the layer, past N (third review: 15 strokes instead of 4).
    # `whole` is the layer to its end: named, in words, only once the call has reached its --stop.
    again = whole + (f" --stop {stop}" if a.stop is not None else "")
    own = _layer_cmd(plan, a.plan, li)          # what the plan itself says for this layer, when it says anything
    if _plan_runs(plan, a.plan) and shlex.split(own)[6:] != shlex.split(whole.split(" --max-min ")[0])[6:]:
        res.setdefault("warnings", []).append(f"the plan's own command for this layer is `{own}`: this call has other options")
    pt.br.check_jaw()           # (reads only; --dry too: a jaw that stands elsewhere than its record refuses the run)
    if a.dry:
        _plan_notes(res, plan, a.plan, state, li)
        # (round 3b, U4: with the --start of the call. Without it the printed command began at the saved progress, not
        # at the stroke this dry run planned from: strokes 0, 1, 2 were painted where 1 and 2 had been planned)
        res.update(stage="planned", next=again + (f" --start {a.start}" if a.start is not None else "") + "   (without --dry)")
        sess.say = (f"レイヤー「{layer['name']}」（{layer['paint']}）は {start}/{n} 本目から描けます。"
                    f"見積もりは約 {minutes:g} 分（補充 {budget['loads']} 回）、プラン全体で約 {res['minutes_all']:g} 分です。")
        return
    if start >= stop:
        res.update(done=bool(start >= n), next_stroke=start, strokes=0, travel_mm=0, loads=0, blots=0, washes=0,
                   next=(f"`{again} --start 0` paints it again" if stop >= n else
                         f"`{again} --start 0` paints strokes 0 to {stop - 1} again; `{whole}` paints the rest of the layer "
                         f"(this call was limited to stroke {stop} by --stop)")
                   + f"; `fd paint preview {shlex.quote(a.plan)}` lists the layers")
        sess.say = (f"レイヤー「{layer['name']}」はすでに {start}/{n} 本目まで描いてあります。何も描いていません。"
                    + ("" if stop >= n else f"（--stop {stop} までの指定です。）"))
        return
    sess.count_time = True
    res["next_stroke"] = start

    def painted(i):             # stroke by stroke: whatever ends this call, the state knows what is on the paper
        PS.set_progress(state, a.plan, plan, li, i)
        res["next_stroke"] = i
        sess.dirty = True
    r = pt.run_layer(plan, li, start=start, stop=a.stop, max_s=float(a.max_min) * 60.0, reload_mm=a.reload_mm,
                     on_stroke=painted)
    stage_warn = res.get("warnings")
    res.update(r)
    if stage_warn and "warnings" not in r:
        res["warnings"] = stage_warn
    PS.set_progress(state, a.plan, plan, li, r["next_stroke"])
    sess.dirty = True
    k = r["next_stroke"]
    if r["done"]:
        res["next"] = (f"`fd paint preview {shlex.quote(a.plan)}` lists the layers; the next one: "
                       f"`{_layer_cmd(plan, a.plan, li + 1)}`"
                       + (" (the plan's own command for it)" if _plan_runs(plan, a.plan) else "")) \
            if li + 1 < len(plan["layers"]) else "`fd paint rest` folds the arm away (the brush off the picture)"
        sess.say = f"レイヤー「{layer['name']}」を描き終えました（{r['strokes']} 本、補充 {r['loads']} 回）。"
    elif r["ok"] and k >= stop:
        # the call did what it was limited to (--stop): the same command again would paint nothing
        res["next"] = f"this call ended at its --stop {stop}; `{whole}` paints the rest of the layer, from stroke {k}"
        sess.say = (f"レイヤー「{layer['name']}」を {k}/{n} 本目まで描きました（--stop {stop} の指定どおりです）。"
                    "残りは next のコマンド（--stop なし）で描きます。")
    elif r["ok"]:
        res["next"] = f"`{again}` continues at stroke {k}" + (f" (and ends at stroke {stop}, as this call was told to)"
                                                               if stop < n else "")
        sess.say = (f"レイヤー「{layer['name']}」を {k}/{n} 本目まで描きました"
                    + ("（時間の上限で区切りました）" if r.get("time_limit") else "") + "。同じコマンドで続きを描きます"
                    + (f"（--stop {stop} まで）。" if stop < n else "。"))
    else:
        res["stage"] = "run"
        res["next"] = (INTERRUPTED_NEXT if r.get("interrupted") else STOPPED_NEXT) + f". `{again}` continues at stroke {k}" \
            + (" (that stroke was begun: its first part gets a second coat)" if pt.br.lowest_z is not None else "")
        sess.say = (f"レイヤー「{layer['name']}」は {k}/{n} 本目まで描いたところで"
                    + ("中断されました。" if r.get("interrupted") else f"止まりました（{r.get('reason')}）。"))


# ----------------------------------------------------------------------------- check, preview (no motion)
def _kin_painter(a, arm, setup: dict):
    """The painter of a command that only computes (check, preview): no tip gate (nothing moves)."""
    return PS.Painter(arm, setup, unmeasured_tip_ok=True)


def cmd_check(a) -> dict:
    setup = PS.load_setup()
    arm, limits = _kin_arm(a.sim)
    try:
        pt = _kin_painter(a, arm, setup)
        rep = PS.check_reach(pt)
        res = {"ok": rep["ok"], "cmd": "check", "stage": "check", "limits_from": limits, "tool": arm.tool.name,
               "pitch_canvas": rep["pitch_canvas"], "items": rep["items"], "problems": rep["problems"],
               "warnings": list(rep["warnings"])}
        if rep["pitch_canvas"] is not None:
            cs = pt.contact_state()
        else:                               # (no canvas pitch to compare: everything else of the record)
            cs = PS.contact_record_state(setup, arm.tool, None)
        res["contact_record"] = {k: cs[k] for k in ("recorded", "ok", "why")}
        # what the brush still needs before it paints, in the order of `fd paint status` (jaw, tip, brush, contact)
        steps = _steps(arm.tool, setup, PS.brush_of(setup), cs, arm.cfg)
        todo = [st for st in steps if not st["done"]]
        res["warnings"] += [f"{st['step']}: {st['why']}" for st in todo if st["why"] and (st["step"] != "contact" or cs["ok"])]
        if PS.one_dish(setup):
            res["warnings"].append("ONE water dish: rinse and clean water are the same dish")
        if PS.wells_not_located(setup):     # (their reach is that of the place they HAD)
            res["warnings"].append(WELLS_LOST.format(names=", ".join(PS.wells_not_located(setup))))
        bad = [it for it in rep["items"] if not it["ok"] and it["name"] in rep["problems"]]
        if bad:
            it = bad[0]
            what = {"well": f"the well {it['name']!r}", "dish": f"the {it['name']} dish", "blot": "the blotting strip",
                    "canvas": f"the {it['name']} of the canvas"}[it["kind"]]
            cause = it.get("cause", "reach")
            if cause == "keep-out":
                fix = f"move {what} out of the keep-out"
            elif cause == "too low":
                # (not a matter of distance: "move it nearer" was the advice for a strip that lay too low)
                fix = (f"{what} lies too low for the brush: raise it, or correct its height in the set-up "
                       + ("(\"blot.z\": the height of its top, in metres)" if it["kind"] == "blot" else
                          "(\"z_mm\": the brush tip above the table there)" if it["kind"] in ("well", "dish") else
                          "(the contact plane: `fd paint contact --scrap-laid --save` measures it, a scrap sheet laid "
                          "over the canvas first)"))
            elif it.get("reach_mm"):
                fix = (f"move {what} nearer to the robot: it is {it['r_mm']} mm from the pan axis, this brush reaches "
                       f"{it['reach_mm']} mm in that direction")
            else:
                fix = f"move {what} into reach (it is {it['r_mm']} mm from the pan axis)"
            if it["kind"] == "dish" and it["name"] == "clean":
                fix += ", or take \"clean\" out of \"water\" (ONE dish)"
            res.update(reason=f"{what} at ({it['xy'][0]:.3f}, {it['xy'][1]:.3f}) cannot be used: {it['why']}"
                              + (f" (and {len(bad) - 1} more: \"problems\")" if len(bad) > 1 else ""),
                       next=fix + "; put its place into the set-up; then `fd paint check` again")
            names = "、".join(i["name"] for i in bad)
            if cause == "reach":
                res["say"] = (f"届かない場所があります: {names}。" + (f"{bad[0]['name']} はロボットの旋回軸から {bad[0]['r_mm']} mm で、"
                              f"この筆が届くのは約 {bad[0]['reach_mm']} mm までです。" if bad[0].get("reach_mm") else "")
                              + "近づけてからセットアップを直してください。")
            else:
                res["say"] = (f"使えない場所があります: {names}（"
                              + ("立入禁止の範囲にあります" if cause == "keep-out" else "高さが低すぎます") + "）。セットアップを直してください。")
        elif not cs["ok"]:
            # (the next is the FIRST step that is missing, as `fd paint status` names it: a plane adopted before
            # the jaw is held or the tip measured is adopted for a brush state that then changes)
            first = todo[0]
            res.update(ok=False, stage="contact", reason=cs["reason"], next=first["next"])
            res["say"] = ("場所はすべて届きますが、紙の高さ（contact）は、今とは別の筆の状態のものとして記録（--adopt）されています。"
                          "その平面を今の筆のままで測ったのなら fd paint contact --adopt をもう一度、そうでなければ捨て紙を置いて "
                          "fd paint contact --scrap-laid --save で測り直してください。" if cs.get("adopted") else
                          "場所はすべて届きますが、紙の高さ（contact）は今の筆の状態とは別の状態で測ったものです。捨て紙を置いて "
                          "fd paint contact --scrap-laid --save で測り直してください。" if cs["recorded"] else
                          "場所はすべて届きますが、紙の高さ（contact）に、どの筆の状態で測ったかの記録がありません。"
                          "今の筆で測った平面なら fd paint contact --adopt、そうでなければ捨て紙を置いて測り直してください。")
            if first["step"] != "contact":
                res["say"] += f"その前にすること: {first['ja']}（next）。"
        else:
            res["next"] = todo[0]["next"] if todo else PLAN_NEXT
            res["say"] = (f"絵の具 {sum(1 for i in rep['items'] if i['kind'] == 'well')} か所、水入れ、キャンバスの四隅と中心に、"
                          f"すべて筆が届きます（キャンバスの筆の角度 {rep['pitch_canvas']:g} 度）。"
                          + (f"次にすること: {todo[0]['ja']}（next）。" if todo else "")
                          + (f"注意 {len(res['warnings'])} 件（warnings）。" if res["warnings"] else ""))
        res["say"] += "アームは動かしていません。" + ("" if limits == "the arm" else "（関節の範囲はシミュレーションの値です。）")
        if a.sim:
            res["sim"] = True
        return res
    finally:
        arm.close()


def cmd_preview(a) -> dict:
    import cv2
    setup = PS.load_setup()
    plan = _load_plan(a.plan)
    state = PS.load_state(a.sim)            # (a state file that cannot be used is refused before anything is written)
    res = {"ok": True, "cmd": "preview", "stage": "preview", "plan": os.path.abspath(a.plan)}
    problems = PS.plan_problems(plan, setup)
    structural = bool(problems) and "not a painting plan" in problems[0]
    try:
        img, renderer = PS.render_plan(plan)
    except Exception as e:              # a plan the simulations cannot draw (a stroke that is no stroke, a missing size)
        img, renderer = None, f"no picture ({type(e).__name__}: {e})"
    out = a.out or os.path.join(_out_dir(a.sim), f"{time.strftime('%H%M%S')}_preview.png")
    if img is not None:
        try:
            os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
            wrote = bool(cv2.imwrite(out, img))
        except (OSError, cv2.error):        # (a folder that cannot be made, a file type OpenCV does not write)
            wrote = False
        if not wrote:
            raise PS.Refusal("options", f"the picture cannot be written to {out}",
                             f"`fd paint preview {shlex.quote(a.plan)}` (without --out: the picture goes to the photo folder), or "
                             "--out with a file that ends in .png, in a folder that can be written")
    res.update(picture=out if img is not None else None, renderer=renderer)
    stats, source = PS.design_stats(plan)
    res.update(design_stats=stats, design_stats_from=source, brush=PS.brush_of(setup), plan_brush=plan.get("brush"))
    if not structural:
        _plan_notes(res, plan, a.plan, state)      # (the plan's own commands, what is open, what is painted elsewhere)
    layers = []
    tool_problem = cs_next = None
    arm = None
    if not structural:
        # The picture is made (and stays): what follows needs the registered tool. Without one, or with a tool file
        # that cannot be read, the answer used to be "the preview was not run" with the picture on the disk.
        try:
            arm, limits = _kin_arm(a.sim)
            pt = _kin_painter(a, arm, setup)
        except SafetyError as e:
            tool_problem = e
    if arm is not None:
        try:
            if tool_problem is None:
                res["limits_from"] = limits
                pitch = cs = None
                try:
                    pitch = pt.pitch_for_canvas()
                    cs = pt.contact_state()
                except SafetyError as e:
                    problems.append(f"no single brush pitch reaches the whole canvas ({e})")
                res["pitch"] = None if pitch is None else round(pitch, 1)
                if cs is not None:
                    res["contact_record"] = {k: cs[k] for k in ("recorded", "ok", "why")}
                    if not cs["ok"]:
                        problems.append(cs["reason"])
                        cs_next = cs["next"]
                try:
                    paint0, unknown0 = PS.carried_paint(state, arm.tool)
                    budget = PS.plan_budget(setup, plan, state, PS.UNKNOWN_PAINT if unknown0 else paint0)
                except Exception:
                    budget = None
                for li, layer in enumerate(plan["layers"]):
                    row = {"index": li, "name": layer.get("name") if isinstance(layer, dict) else None,
                           "paint": layer.get("paint") if isinstance(layer, dict) else None, "ok": True}
                    mine = [p for p in problems if isinstance(row["name"], str) and (p.startswith(f"{row['name']}[") or
                                                                                       p.startswith(f"layer {row['name']!r}"))]
                    if budget is not None:
                        row.update({k: budget["layers"][li][k] for k in ("strokes", "travel_mm", "loads", "blots", "washes", "minutes")})
                    if pitch is not None and not problems:
                        bad = pt.check_strokes(layer)
                        if bad:
                            mine.append(f"{len(bad)} stroke(s) out of reach: stroke {bad[0]['stroke']}: {bad[0]['why']}")
                            row["out_of_reach"] = bad[:10]
                    if mine:
                        row.update(ok=False, reason=mine[0])
                    layers.append(row)
                if budget is not None:
                    res.update(strokes=budget["strokes"], travel_mm=budget["travel_mm"], loads=budget["loads"],
                               blots=budget["blots"], washes=budget["washes"], minutes=budget["minutes"], times=budget["times"])
        finally:
            arm.close()
    res["layers"] = layers
    # the strokes over a fresh photo: only with the real camera (not simulated, not locked); nothing moves for it
    overlay, why = None, None
    if a.sim:
        why = "simulation: no camera"
    elif _camera_locked():
        why = "the hardware is locked: no photo"
    elif structural or img is None:
        why = "the plan cannot be drawn"
    else:
        try:
            photo = _camera(False)()
            cam = _cam_model()
            overlay = os.path.splitext(out)[0] + "_photo.jpg"
            if not cv2.imwrite(overlay, PS.overlay_plan(photo, cam, setup, plan), [cv2.IMWRITE_JPEG_QUALITY, 92]):
                overlay, why = None, f"the photo with the strokes could not be written to {overlay}"
        except Exception as e:              # the camera is not there, or busy: the simulated picture stands
            overlay, why = None, f"no photo ({type(e).__name__}: {str(e)[:120]})"
    res.update(overlay=overlay)
    if why:
        res["overlay_note"] = why + "; only the simulated picture was made"
    else:
        res["overlay_note"] = ("the photo was taken with the arm where it stands; the paint lands about 6 mm beside and 9 mm "
                               "nearer to the robot than the drawn lines (2026-11-13)")
    bad_layers = [r for r in layers if not r["ok"]]
    if problems or bad_layers:
        first = problems[0] if problems else bad_layers[0]["reason"]
        res.update(ok=False, stage="contact" if "contact plane" in first else "plan",
                   reason=first + (f" (and {len(problems) - 1} more)" if len(problems) > 1 else ""), problems=problems[:20],
                   next=(cs_next or CONTACT_NEXT) if "contact plane" in first else
                   "make the plan anew for this set-up (`fd paint status` shows canvas, pan axis and brush), or correct the set-up")
        res["say"] = (f"このプランは今のセットアップでは描けません（{first}）。"
                      + ("シミュレーションの絵は作りました。" if img is not None else ""))
    elif tool_problem is not None:
        # the picture stands; what needs the tool (reach, the contact plane, minutes) was not looked at
        bad_file = _tool_file_refusal("preview", tool_problem) is not None
        res.update(ok=False, stage="tool", reason=str(tool_problem), reach_checked=False,
                   next=TOOL_FILE_NEXT if bad_file else (getattr(tool_problem, "next", "") or
                                                         "`fd tool attach --help` shows how a brush is registered"))
        res["say"] = (("シミュレーションの絵は作りました（picture）。" if img is not None else "")
                      + ("ツールのファイルが読めない" if bad_file else "筆が登録されていない")
                      + "ので、筆が届くかどうか・紙の高さ・所要時間は確認していません。")
    else:
        res["next"] = f"{_layer_cmd(plan, a.plan, 0)} --dry"
        res["say"] = (f"プレビューを作りました: {len(plan['layers'])} レイヤー、{res.get('strokes', 0)} 本、"
                      f"見積もり約 {res.get('minutes', 0):g} 分（補充 {res.get('loads', 0)} 回、筆洗い {res.get('washes', 0)} 回）。")
    res["say"] += ("写真への重ね描きも作りました。" if overlay else "写真への重ね描きはありません（" +
                   ("シミュレーション" if a.sim else "カメラが使えません") + "）。") + "アームは動かしていません。"
    if a.sim:
        res["sim"] = True
    return res


# ----------------------------------------------------------------------------- rest
def cmd_rest(a) -> dict:
    setup = PS.load_setup()
    _term_as_interrupt()
    arm = _open_arm(a.sim)
    try:
        tool = arm.tool
        if tool is None:
            raise PS.Refusal("tool", "no brush is registered: there is nothing to keep off the picture",
                             "`fd park` folds the arm away")
        measured = BW.tip_is_measured(tool)
        q0 = arm.read_q()
        # An arm that already lies folded and limp with the brush clear of the canvas is left as it lies (unless
        # another pan, or a height, is asked for), as `fd tool rest` leaves one in the tool's own rest pose.
        same_pan = a.pan is None or abs(float(a.pan) - math.degrees(float(q0[0]))) < 3.0
        lying = PS.rest_as_it_lies(arm, setup, q0) if a.clearance is None and same_pan else None
        try:
            if lying is not None:
                choice = lying
            elif not measured and a.clearance is None:
                # A tip that was never measured: the fold is planned with the REGISTERED tip, so it is made as high as
                # this arm folds (no refusal: this is the command that puts the arm away), and the result says so.
                choice = PS.highest_rest(arm, setup, a.pan)
            else:
                choice = PS.rest_choice(arm, setup, a.pan, a.clearance)
        except PS.Refusal as e:
            if e.detail.get("cannot_fold") and a.clearance is None and "clearance_m" not in (setup.get("rest") or {}):
                # (the way out is not this command again: PS.rest_choice's next; the sentence to pass on says it too)
                e.detail["say"] = ("この道具を付けたままでは、アームを畳めません（どの旋回角でも収納姿勢を取れません）。アームは今の姿勢の"
                                   "ままです（トルクが入っていれば保持しています）。アームを手で支えて道具を外し、fd tool detach を"
                                   "実行してください。道具の関節制限は fd tool status で確認できます。アームは動かしていません。"
                                   + (SIM_JA if a.sim else ""))
            raise
        q_rest = choice.pop("q")
        res = {"ok": True, "cmd": "rest", "stage": "rest", "tool": tool.name, "pan_deg": choice["pan_deg"],
               "clearance_mm": round(choice["clearance_m"] * 1000.0), "chosen": choice["chosen"], "pose": choice}
        if not measured:
            res["tip_measured"] = False
            res["warnings"] = [
                f"the tip of tool '{tool.name}' was never measured: "
                + ("the arm was left as it lies, and where its brush really is, is not known" if lying is not None else
                   "this fold was planned with the REGISTERED tip"
                   + ("" if a.clearance is not None else f", as high as this arm folds ({res['clearance_mm']} mm above the table by that tip)"))
                + ". A tool that is longer than registered hangs lower than \"brush\" says (2026-11-13: a first guess was "
                  "9 cm short): LOOK at the arm"]
        there = float(np.max(np.abs(q0[:4] - q_rest[:4]))) < math.radians(3.0)
        try:
            if lying is not None:
                res.update(stage="planned" if a.dry else "rest", moved=False, torque="off",
                           note="the arm already lies folded with the torque off, the brush clear of the canvas: nothing moved")
            elif a.dry:
                if not there:
                    plan = TL.plan_free_move(arm, tool, q0, q_rest)
                    res.update(via=plan["via"], lift_first=bool(plan["lift"]),
                               planned_min_tip_height_mm=round(plan["min_height_m"] * 1000.0, 1))
                res.update(stage="planned", moved=False, torque="on" if arm.is_torqued() else "off",
                           next="the same command without --dry")
            else:
                if not arm.is_torqued() and not there:
                    # limp somewhere else (the tool's own rest pose is over the picture): tooling.rest would leave it
                    arm.torque_on()
                    arm._sleep(0.3)
                r = TL.rest(arm, clearance_m=choice["clearance_m"], pan_deg=choice["pan_deg"], torque_off=not a.keep_torque)
                res.update({k: r[k] for k in ("ok", "moved", "torque", "via", "duration_s", "min_tip_height_mm",
                                              "sag_after_release_mm", "reason", "note", "jaw") if k in r})
                if not r.get("ok", True):
                    # The `next` is written HERE: tooling.rest's own names `fd tool rest`, whose pose (pan 20) is over
                    # the picture, the very fold this command exists to avoid.
                    again = f"fd paint rest --pan {choice['pan_deg']:g}"
                    if r.get("torque") == "off":        # folded and released, and then it sank
                        res["next"] = ("support the arm by hand; `fd torque on` holds it where it is, then "
                                       f"`{again} --clearance {min(0.15, choice['clearance_m'] + 0.02):g}` (the same fold, the "
                                       "brush higher)")
                    else:                               # it did not get into the fold: the torque is on, the arm holds
                        res["next"] = (f"the torque is still ON and the arm holds; `fd state`, then `{again}` again (not "
                                       "fd tool rest: its fold is over the picture)")
        except (SafetyError, MotionAborted) as e:
            moved = bool(float(np.max(np.abs(arm.read_q() - q0))) > math.radians(0.5))
            res.update(ok=False, stage="move" if moved else "plan", reason=f"{type(e).__name__}: {e}", moved=moved,
                       torque="on" if arm.is_torqued() else "off",
                       next=("the arm stopped on the way and holds where it is: `fd state`, then `fd paint rest` again" if moved
                             else "nothing moved; `fd paint rest --dry` shows the pose, `fd tool status` the tool"))
        except KeyboardInterrupt:
            res.update(ok=False, stage="move", reason="interrupted", interrupted=True, moved=True,
                       torque="on" if arm.is_torqued() else "off",
                       next="the arm holds where it stopped (torque as \"torque\" says): `fd paint rest` again")
        q = arm.read_q()
        tl = K.tool_offset(*tool.tip_offset)
        p = K.fk(q, tl)[:3, 3]
        h = (float(p[2]) - TL.table_z(arm)) * 1000.0
        folded = float(np.max(np.abs(q[:4] - q_rest[:4]))) < math.radians(4.0)
        lands, low = PS.sink_of(arm, tool, q)
        res["brush"] = {"tip_xyz": [round(float(v), 4) for v in p], "height_mm": round(h, 1), "torque": "on" if arm.is_torqued() else "off",
                        "where": "rest" if folded else ("not moved" if not res.get("moved") else "stopped"),
                        "clear_of_canvas_mm": round(min(PS.canvas_distance(setup, float(p[0]), float(p[1])),
                                                        PS.canvas_distance(setup, *low)) * 1000.0, 1),
                        "would_sink_to": lands, "near": choice["near"]}
        near = f"（最寄りは {choice['near'][0]}、{choice['near'][1]} mm）" if choice.get("near") else ""
        if res["ok"] and not measured and res["stage"] != "planned":
            # (an arm that was left as it lies was not folded by this command: nothing was planned with any tip)
            res["next"] = ("LOOK at the arm: "
                           + ("it was left as it lies (nothing was planned or folded), and where the brush really is, is not "
                              f"known: the tip of tool '{tool.name}' was never measured" if lying is not None else
                              f"the fold was planned with the registered tip of tool '{tool.name}', which was never measured")
                           + " (a tool longer than registered hangs lower than \"brush\" says). `fd paint status` names "
                           "what the brush needs before it paints; after taking the brush off: `fd tool detach`")
        if lying is not None:
            # (the pan in words: the same command with "--pan P", typed as it was printed, is refused: 'P' is no angle)
            res.setdefault("next", "the same command without --dry leaves the arm as it lies too; with --pan and an angle "
                                   "in degrees (another pan than the arm lies at) it is folded there instead" if a.dry else
                           "`fd paint look` raises the arm to the photo pose again; after taking the brush off: `fd tool detach`")
            res["say"] = (f"アームはすでに畳んであります（トルクOFF）。筆先はキャンバスの {res['brush']['clear_of_canvas_mm']:.0f} mm 外、"
                          f"机の {h:.0f} mm 上です{near}。何も動かしていません。")
        elif res["stage"] == "planned":
            res["say"] = (f"収納の計画: 旋回 {choice['pan_deg']:g} 度で畳むと、筆先はキャンバスの {choice['clear_of_canvas_mm']:.0f} mm 外、"
                          f"机の {choice['tip_height_mm']:.0f} mm 上になります{near}。アームは動かしていません。")
        elif res["ok"]:
            res.setdefault("next", "`fd paint look` raises the arm to the photo pose again; after taking the brush off: `fd tool detach`")
            res["say"] = ((f"アームを畳みました（トルク{'OFF' if res.get('torque') == 'off' else 'ONのまま'}）。" if res.get("moved") else
                           f"アームはすでにその収納姿勢にあります（トルク{'OFF' if res.get('torque') == 'off' else 'ON'}）。何も動かしていません。")
                          + f"筆先はキャンバスの {res['brush']['clear_of_canvas_mm']:.0f} mm 外、机の {h:.0f} mm 上です{near}。"
                          "手首が下がっても絵には触れません。")
        elif res.get("interrupted"):
            res["say"] = f"収納が途中で中断されました。筆先は机の {h:.0f} mm 上、トルクは{res['torque'].upper()}です。"
        else:
            res["say"] = (f"収納できませんでした（{res.get('reason')}）。筆先は机の {h:.0f} mm 上、トルクは"
                          f"{res['brush']['torque'].upper()}です。")
        if not measured:
            res["say"] += ("注意: 筆先の位置は未測定です。"
                           + ("" if lying is not None else "畳み方は登録された先端の位置で計画しています"
                              + ("。" if a.clearance is not None else "（この腕で畳める一番高い姿勢）。"))
                           + "登録より長い道具は、これより低い所にあります。アームを目で確認してください。")
        if a.sim:
            res["say"] += SIM_JA
            res["sim"] = True
        return res
    finally:
        _close(arm)


def cmd_contact(a) -> dict:
    if a.adopt:
        return _contact_adopt(a)
    if a.read:
        return _contact_read(a)
    return _moving(a, _contact_body, photo_after=True)


COMMANDS = {
    "status": cmd_status, "brush": cmd_brush, "check": cmd_check, "preview": cmd_preview, "rest": cmd_rest,
    "contact": cmd_contact,
    "look": lambda a: _moving(a, _look_body),
    "locate": lambda a: _moving(a, _locate_body),
    "rinse": lambda a: _moving(a, _rinse_body),
    "test": lambda a: _moving(a, _test_body, photo_after=True),
    "run": lambda a: _moving(a, _run_body, photo_after=True),
}


def main(argv=None) -> dict:
    """One `fd paint ...` command; always returns a result (see the top of this file). The words are those
    after `fd paint` (fd.py hands them over untouched, "--sim" in front when fd was called with it); one leading
    "--" is dropped, as `fd calib-air` and `fd tool jaw` do."""
    argv = list(sys.argv[1:] if argv is None else argv)
    sim = argv[:1] == ["--sim"]                 # fd's own --sim, handed over in front of the words (never an option here)
    if sim:
        del argv[0]
    if argv[:1] == ["--"]:                      # (`fd paint -- status`, `fd --sim paint -- status`)
        del argv[0]
    try:
        a = build_parser().parse_args(argv)
        a.sim = sim
    except _BadArgs as e:
        if "--sim" in argv or any(w.startswith("--s") and "--sim".startswith(w) for w in argv):
            res = _refusal(None, "options", f"{e}. --sim is fd's own option: it stands before the command. Nothing moved",
                           "`fd --sim paint status` (the same words, --sim right after fd)")
            res["say"] = "--sim は fd のすぐ後ろに書いてください（fd --sim paint ...）。何も動かしていません。"
            return res
        short = _abbreviated(argv)
        if short:
            # an option cut short: not completed (T17). The declarations (--scrap-laid, --tip-looked-at) are the
            # operator's word, typed in full; so is every other option
            word, could = short[0]
            res = _refusal(None, "options", f"{e}. Options are typed in full: {word} is only the beginning of "
                                            + " / ".join(could) + ". Nothing moved",
                           "the same command with " + (could[0] if len(could) == 1 else "one of " + ", ".join(could))
                           + f" in the place of {word}")
            res["say"] = (f"オプションは省略できません（{word} ではなく {'、'.join(could)} と最後まで入力してください）。"
                          "何も動かしていません。")
            return res
        return _refusal(None, "options", str(e), "`fd paint --help` lists the commands; the options of one of them: its name "
                                                 "before --help, as in `fd paint run --help`")
    try:
        return COMMANDS[a.cmd](a)
    except PS.Refusal as e:
        res = _refusal(a.cmd, e.stage, str(e), e.next, **e.detail)
    except (BusError, HardwareLocked) as e:
        # (an arm that cannot be OPENED is a Refusal, see _open_arm: what arrives here happened after that)
        if "hardware is locked" in str(e):
            res = _refusal(a.cmd, "hardware", f"{type(e).__name__}: {e}",
                           "`fd hw status` says why it is locked; " + SIM_NEXT, say=LOCKED_JA)
        elif a.cmd in MOVES and not getattr(a, "dry", False):
            res = _refusal(a.cmd, "hardware", f"{type(e).__name__}: {e}", BUS_NEXT, brush={"where": "unknown", "torque": "unknown"})
            res["say"] = f"{CMD_JA[a.cmd]}の途中でサーボと通信できなくなりました（{e}）。" + BW.where_ja(res["brush"])
        else:
            res = _refusal(a.cmd, "hardware", f"{type(e).__name__}: {e}",
                           "is the arm switched on and connected? `fd state` asks it; without it: " + SIM_NEXT)
    except (SafetyError, MotionAborted) as e:
        # a tool file that cannot be used has a next of its own (the default one sent the operator to
        # `fd paint status`, which refuses the same file)
        res = _tool_file_refusal(a.cmd, e) or _refusal(a.cmd, "start", f"{type(e).__name__}: {e}")
    except KeyboardInterrupt:           # (outside the moves of a command: those end with the brush lifted, see _moving)
        res = _refusal(a.cmd, a.cmd, "interrupted", "`fd state` shows where the arm is; `fd paint rest` folds it away",
                       interrupted=True)
        res["say"] = f"{CMD_JA.get(a.cmd, 'コマンド')}が中断されました。アームの状態は fd state で確認してください。"
    except Exception as e:              # a bug must still reach the operator as a result
        import traceback
        res = _refusal(a.cmd, "internal", f"internal error: {type(e).__name__}: {e}",
                       trace=traceback.format_exc().strip().splitlines()[-6:])
        if a.cmd in MOVES and not getattr(a, "dry", False):     # (it may have happened after the arm had moved)
            res["next"] = "`fd state` shows where the arm is; `fd paint rest` folds it away"
            res["say"] = (f"{CMD_JA[a.cmd]}が内部エラーで止まりました（{type(e).__name__}: {e}）。"
                          "アームの状態は fd state で確認してください。")
    if a.sim:
        res["sim"] = True
    return res


if __name__ == "__main__":
    print(json.dumps(main(), ensure_ascii=False, default=_json_default))
