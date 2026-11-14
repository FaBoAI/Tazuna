"""`fd tool jaw` and its words: the jaw of a tool that is taped to the OUTER face of the moving jaw.

Why this exists (2026-11-13): a thin brush was taped to the moving jaw, and it painted in every direction only
after the jaw had been closed against the fixed jaw and held there with a light squeeze (tooling.py tells the
whole story and has the numbers). The gripper is locked while a tool is registered, so that was done by scratch
scripts that wrote gripper goals past the lock. These are the commands for it:

  fd tool jaw status                              where the jaw stands (raw, degrees), what the tool file records,
                                                  whether the squeeze is on; with the arm off or locked: the file only
  fd tool jaw mount fixed|moving_outer|between    record on which jaw the tool is taped (file only)
  fd tool jaw close [--to closed|DEG] [--squeeze TICKS] [--dry]
                                                  close the jaw: until it meets the fixed jaw, then hold it there
                                                  (the default), or to an angle
  fd tool jaw open --to DEG [--dry]               open it to an angle (0 = jaws parallel)
  fd tool jaw help | --help | -h                  the usage and every option (also behind a sub-command)

`close` and `open` are refused unless the tool is recorded as taped to the outer face of the moving jaw (`mount
moving_outer`): a tool held between the jaws would be crushed by closing them and dropped by opening them. The
jaw goes in small steps, the position is read after every step, and it stops where the jaw no longer follows
(tooling.move_jaw). The tool MOVES with the jaw: before the move, the tip the jaw model predicts must stay clear
of the table with the arm where it stands; afterwards the tool file has the new jaw position and holding goal,
that predicted tip (about 1 cm off on the real arm), no tip measurement and no pen calibration, and
`fd calib-air --tip-only --save` measures the tip again. `--dry` shows all of that and moves nothing. With --sim every
call is a new simulated arm folded in the tool's rest pose; `--at-ready` stands it at the stand-by pose instead,
where the real arm is after `fd ready`.

What the review of 2026-11-13 added:
- a tool file that cannot be read is answered by every sub-command as `status` always did (stage "tool");
- when the servo bus stops answering in the middle of a jaw move, the result says that the jaw's position is not
  known (stage "hardware"), and the tool file says the same (tooling.jaw_lost): it used to go on saying that
  the jaw was where it had been and that the tip was measured;
- `status` compares the jaw with the tool file for every tool that records a jaw position, held by a squeeze or
  not (tooling.jaw_tolerance): `fd ready` and brush work refuse a jaw that stands elsewhere, and this is the
  command their refusal names;
- a simulated command that stores the tool writes the SIMULATED tool file, which may be a copy of the real
  tool's record: the result says so ("simulated_tool", "sim_note", a sentence in "say");
- `--help` / `-h` anywhere among the words is answered with the usage (fd hands these words through since that
  day), and one leading `--` is dropped.

fd.py hands the words after `fd tool jaw` to main() ("--sim" in front when fd was called with --sim) and prints
the dict it returns as the JSON result: "ok", "stage", "reason", "next" (one line, the next command), "say" (for
the operator, in Japanese). Tests: tests/test_tooljaw.py (simulated arm; nothing here was run on the real arm).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import signal
import sys
import threading

USAGE = ("fd tool jaw status [--no-robot] | mount fixed|moving_outer|between [--dry] | "
         "close [--to closed|DEG] [--squeeze TICKS] [--step TICKS] [--lag TICKS] [--dry] [--no-robot] [--at-ready] | "
         "open --to DEG [--step TICKS] [--lag TICKS] [--dry] [--no-robot] [--at-ready]")
WHERE_JA = {"fixed": "固定爪に付いている", "moving_outer": "可動爪の外側に付いている", "between": "2 本の爪の間に挟んである"}
# (--save is what writes the record of the measured tip into the tool file; without it the painting commands go on
# refusing the tool)
NEXT_MEASURE = ("`fd calib-air --tip-only --save` measures the tool tip again and records it in the tool file (the camera "
                "must not have moved)")
# After a jaw command that left the tool where it was: what the operator does next, by what the tool file says.
NEXT_PAINT = "`fd paint status` says what the brush still needs before it paints, and names the next command"
NEXT_TIP = ("the tip of this tool is not on record as measured: `fd calib-air --tip-only --save` measures it and records "
            "it (`fd paint status` lists what else a brush needs)")
# The same with a painting set-up on record (_paint_setup): there `fd paint status` leads, and ONE order holds for
# every text. Round 3 (2026-11-14), copies of the rig's files (the jaw shut, nothing said of the jaw, the tip not on
# record as measured, the evening's photos of the tip in a folder): `fd tool jaw status` -> `fd tool jaw mount
# moving_outer` -> `fd tool jaw close` -> "`fd calib-air --tip-only --save` measures it" led jaw first to a new
# flight on the real arm, and the photos were named nowhere on that way; `fd paint status` names them FIRST,
# because a hold on record makes them photos of another jaw state.
FIRST_PAINT = ("the tip of this tool is not on record as measured: `fd paint status` names the next step FIRST (photos "
               "of the tip that were taken with the jaw as it stands now are stored before the jaw is closed and held; a "
               "hold on record makes them photos of another jaw state)")
NEXT_TIP_HELD = ("the tip of this tool is not on record as measured: `fd paint status` names the next step (a new "
                 "measurement, `fd calib-air --tip-only --save`; photos taken before the jaw was held are refused as they "
                 "are, and that refusal names the way back)")
FIRST_PAINT_JA = ("筆先はまだ測定の記録がありません。爪を閉じて保持する前に、先に fd paint status の next を確認してください"
                  "（いまの爪の位置で撮った筆先の写真があれば、保持の記録ができる前に保存します）。")
HELD_PAINT_JA = ("筆先はまだ測定の記録がありません。次の手順は fd paint status が示します（測り直しは fd calib-air --tip-only "
                 "--save。保持する前に撮った写真は、そのままでは保存を断られ、その結果の next が戻り方を示します）。")


def _cap(text: str) -> str:
    """The text with its first letter in upper case (a sentence that is put behind a full stop)."""
    return text[:1].upper() + text[1:]


def _paint_setup() -> str | None:
    """Path of the painting set-up that counts for the tool file in force, or None: the rule of
    fd._paint_setup_on_record, for the texts of this module (fd imports this module, not the other way round). No
    painting module, no set-up file, or a tool file named by $FD_TOOL while the set-up is the rig's own (a scratch
    tool is a test or an experiment, and the rig's set-up describes another table): None. Only with a set-up on
    record does `fd paint status` lead anywhere; without one it is refused (the set-up is written by hand first),
    and a pen, or a brush on a rig that has not painted yet, is not sent there."""
    try:
        import paintskill
    except Exception:           # the painting feature is not there, or does not load
        return None
    setup_path = getattr(paintskill, "setup_path", None)
    if not callable(setup_path):
        return None
    path = setup_path()
    if not os.path.isfile(path) or (os.environ.get("FD_TOOL") and not os.environ.get("FD_PAINT_SETUP")):
        return None
    return path


def _paint_first(tool) -> bool:
    """`fd paint status` comes before the next jaw command: the tool's tip is not on record as measured, and a
    painting set-up is on record."""
    return not tool.tip_measured and _paint_setup() is not None


class _Usage(Exception):
    """A command line this module cannot read (argparse would print and exit: here it becomes a result)."""


class _Parser(argparse.ArgumentParser):
    """The parser of these words and of every sub-command: a word it cannot read is a result (_Usage), and an
    option is taken only as it is spelled. argparse completes an abbreviation by itself: `close --sq 60` was read
    as --squeeze 60 and `--at` as --at-ready; a word the operator did not write out must not be guessed (round 3,
    2026-11-14: the declarations of the paint commands were taken from `--scrap` and `--tip` that way)."""

    def __init__(self, *a, **k):
        k.setdefault("allow_abbrev", False)
        super().__init__(*a, **k)

    def error(self, message):
        raise _Usage(message)


COMMANDS = ("status", "mount", "close", "open")


def build_parser() -> argparse.ArgumentParser:
    """The parser of the words after `fd tool jaw`. Its sub-parsers are kept as `.commands` {name: parser}: the
    help answer (_help) prints their option texts, which argparse itself never shows here (add_help=False: a
    help that prints and exits would leave fd without a result)."""
    p = _Parser(prog="fd tool jaw", add_help=False)
    sub = p.add_subparsers(dest="cmd", required=True, parser_class=_Parser)
    p.commands = {}
    s = p.commands["status"] = sub.add_parser("status", add_help=False)
    s.add_argument("--no-robot", action="store_true", help="do not read the arm: show what the tool file records")
    s = p.commands["mount"] = sub.add_parser("mount", add_help=False)
    s.add_argument("where", choices=("fixed", "moving_outer", "between"),
                   help="on which jaw the tool is taped (look at a photo of the gripper first)")
    s.add_argument("--dry", action="store_true", help="show what would be recorded; write nothing")
    for name in ("close", "open"):
        s = p.commands[name] = sub.add_parser(name, add_help=False)
        s.add_argument("--to", default="closed" if name == "close" else None, required=name == "open",
                       help="a jaw angle in degrees (0 = jaws parallel)" + (", or `closed`" if name == "close" else ""))
        if name == "close":
            s.add_argument("--squeeze", default=None, help="ticks the holding goal lies beyond where the jaw met the "
                                                           "fixed jaw (default 25; never more than config squeeze_ticks)")
        s.add_argument("--step", default=None, help="ticks per step, 1 .. 15 (default 15; one step every 0.12 s)")
        s.add_argument("--lag", default=None, help="ticks a standing jaw may be behind its goal, 10 .. 40 (default 30)")
        s.add_argument("--dry", action="store_true", help="show the plan and where the tool tip would end up; nothing moves")
        s.add_argument("--no-robot", action="store_true", help="with --dry: do not read the arm")
        s.add_argument("--at-ready", action="store_true",
                       help="with --sim: the simulated arm stands at the tool's stand-by pose, as after `fd ready` (every "
                            "simulated call starts folded in the rest pose, where a wide-open jaw has no room to close)")
    return p


def make_arm(sim: bool):
    """The arm a jaw command works on, opened (as fd.make_arm: the simulated arm carries the simulated tool and
    starts in the tool's rest pose with the jaw where the tool file has it). The tests put their own arm here."""
    import tooling
    from robot import Arm, load_config
    if sim:
        from sim_bus import SimBus
        arm = Arm(bus=SimBus(realtime=False), cfg=load_config(), tool=tooling.active(sim=True)).open()
        tooling.sim_start(arm)
        return arm
    return Arm().open()      # SafetyError without the rig's configuration; BusError when the hardware is locked or off


# ------------------------------------------------------------------ results
def _refused(stage: str, reason: str, next_: str, say: str, **more) -> dict:
    """The complete result of a refusal: the jaw did not move, nothing was written."""
    return {"ok": False, "error": "SafetyError", "stage": stage, "reason": reason, "message": reason, "next": next_,
            "say": say, **more}


def _no_tool() -> dict:
    # (words where the operator has to give a value, one complete command per pair of backticks: `fd tool attach
    # --name N --tip x,y,z` and `fd grip`, typed as they were printed, are refused)
    return _refused("tool", "no tool is attached: the tool jaw commands move the jaw of a tool taped to its outer face",
                    "fd tool attach with --name and the tip registers what is taped to the gripper (`fd tool attach --help` "
                    "lists the words: --tip in metres, or for a straight tool --along-fingers / --across-fingers and its "
                    "length); the bare gripper is moved with `fd grip 0.5` (half open; 0 = closed, 1 = open) and `fd grasp`",
                    "ツールが登録されていません。先に fd tool attach で登録してください（アームは動かしていません）。")


def _tool(sim: bool) -> tuple:
    """(the attached tool or None, None), or (None, the complete refusal) when the tool file cannot be used. A
    damaged file must stop the arm, never count as "no tool" (tooling.load), and every sub-command says so in
    the house shape: `mount`, `close` and `open` used to let the SafetyError through, and fd then printed a
    result without stage, next and say (review of 2026-11-13)."""
    import tooling as TL
    from robot import SafetyError
    try:
        return TL.active(sim), None
    except SafetyError as e:
        nxt = ("repair the file; `fd --sim tool detach` removes the SIMULATED tool file (on the real arm: take the "
               "tool off by hand and run `fd tool detach`)" if sim else
               "repair the file, or take the tool off by hand and run `fd tool detach`")
        return None, _refused("tool", str(e), nxt,
                              "ツールの記録ファイルが読めません。グリッパーに何が付いているか分からないため、爪は動かせません"
                              "（何も動かしていません）。", attached=None, file=TL.source(sim))


def _sim_state(sim: bool) -> dict | None:
    """tooling.sim_tool_state() for a simulated command (None without --sim): which tool file the simulation
    reads. Taken at the start of a command that may store the tool, so that a copy it then writes over is still
    reported as left behind."""
    if not sim:
        return None
    import tooling as TL
    return TL.sim_tool_state()


def _sim_said(res: dict, sim: bool, wrote: bool = False, was: dict | None = None) -> dict:
    """What a simulated command has to say about the tool file it works on, added to its result (nothing
    without --sim). "simulated_tool": the simulation reads a tool file of its own, not the real arm's. After a
    command that STORED the tool (wrote=True) that file is where it went, and when it is a copy of the real
    tool's record the simulation works on that copy from now on: "sim_note" and a sentence in "say" tell the
    operator, who would otherwise rehearse the next real command on a record the real arm does not have (review
    of 2026-11-13). "sim_copy_dropped": an older copy that is not read any more because the real tool was
    changed or replaced since (`was`: the state at the start of the command)."""
    st = _sim_state(sim)
    if st is None:
        return res
    res["simulated_tool"] = bool(st["simulated_tool"])
    dropped = st["dropped"] or (was or {}).get("dropped")
    if dropped:
        res["sim_copy_dropped"] = dropped
        res["say"] = (res.get("say") or "") + ("（以前のシミュレーション用のツール記録は、そのあと実機のツール記録が変わったため"
                                               "使っていません。実機の記録をもとにしています。）")
    if st["simulated_tool"]:
        res["sim_note"] = ("this command stored the tool in the SIMULATED tool file: " if wrote else "") + (st["note"] or "")
        if wrote:
            res["say"] = (res.get("say") or "") + (
                "（シミュレーション用のツール記録に保存しました。実機のツール記録は変えていません。実機の記録が変わるか "
                "fd --sim tool detach を実行するまで、fd --sim はこの記録を使います。）")
    return res


def _locked(tool, why: str) -> dict:
    """The refusal for a tool whose jaw must not be moved (tooling.jaw_locked says why)."""
    if tool.on_jaw is None:
        say = (f"ツール「{tool.name}」がどちらの爪に付いているか記録がないので、爪は動かしませんでした。爪の間に挟んだ道具は、"
               "閉じるとつぶれ、開くと落ちるためです。グリッパーの写真で確かめてから、fd tool jaw mount moving_outer"
               "（可動爪の外側）、fd tool jaw mount fixed（固定爪）、fd tool jaw mount between（爪の間）のどれかで記録してください。")
        nxt = ("look at a photo of the gripper (`fd snap`), then `fd tool jaw mount moving_outer` when the tool is taped to "
               "the outer face of the moving jaw (else `fd tool jaw mount fixed` or `fd tool jaw mount between`: the jaw "
               "then stays locked)")
    else:
        say = (f"ツール「{tool.name}」は{WHERE_JA[tool.on_jaw]}と記録されているので、爪は動かしませんでした"
               + ("（閉じるとつぶれ、開くと落ちます）。" if tool.on_jaw == "between" else "。")
               + "爪を動かせるのは、可動爪の外側にテープで留めた道具だけです。")
        nxt = ("the jaw stays locked for this tool. If the record is wrong (look at a photo of the gripper first): "
               "`fd tool jaw mount moving_outer`")
    return _refused("tool", why, nxt, say, tool=tool.name, on_jaw=tool.on_jaw)


def _options(e, **more) -> dict:
    return _refused("options", str(e),
                    "`fd tool jaw help` lists the options; `fd tool jaw status` shows where the jaw stands",
                    f"指定を受け付けられませんでした（{e}）。爪は動かしていません。", **more)


def _typed(a) -> str:
    """The words of this `close` / `open` as they were given, without --dry, --no-robot and --at-ready: for a next
    that names the same move again. `fd tool jaw open` without its --to is refused when it is typed, and a close
    named without the --squeeze it was planned with is another close (round 3: one complete command per pair of
    backticks, with the values the program knows)."""
    words = [a.cmd]
    if a.to is not None and not (a.cmd == "close" and str(a.to) == "closed"):
        words += ["--to", shlex.quote(str(a.to))]
    for flag in ("squeeze", "step", "lag"):
        if getattr(a, flag, None) is not None:
            words += [f"--{flag}", shlex.quote(str(getattr(a, flag)))]
    return " ".join(words)


def _too_low(e, cmd: str, dry: bool, sim: bool, record=None, **more) -> dict:
    """The tool would swing into the table (tooling.jaw_swing): the arm has to come up first. `fd ready` is what
    raises it, and `fd ready` refuses a jaw that does not stand where the tool file has it (tooling.jaw_in_place):
    when that is the case here, the hint names the way that is open, the jaw brought back to its recorded angle.
    cmd: the words of the move (_typed); record: (the tool, the configuration), for that comparison."""
    import tooling as TL
    tool, cfg = record if record is not None else (None, None)
    again = f"fd tool jaw {cmd}{' --dry' if dry else ''}"
    nxt = (f"with --sim every call starts folded in the rest pose: `fd --sim tool jaw {cmd}{' --dry' if dry else ''} "
           "--at-ready` rehearses it from the stand-by pose (on the real arm: `fd ready` first)" if sim else
           f"`fd ready` raises the arm; then `{again}` again")
    say = ("いまの姿勢で爪を動かすと、ツールの先端が机に近づきすぎます（爪は動かしていません）。先に fd ready でアームを"
           "上げてください。")
    jaw, tol = more.get("jaw"), TL.jaw_tolerance(tool)
    if not sim and jaw is not None and tol is not None and cfg is not None and abs(jaw["from_record_ticks"]) > tol:
        off = jaw["from_record_ticks"]
        lo, hi = TL.jaw_range_raw(cfg)
        if lo <= int(tool.gripper_raw) <= hi:        # the recorded position is one `--to DEG` can name
            back = f"fd tool jaw {'open' if off < 0 else 'close'} --to {TL.jaw_q_deg(cfg, tool.gripper_raw):.1f}"
            how_en = f"`{back}` brings the jaw back to the recorded angle first, where the tool file's tip holds"
            how_ja = f"先に {back} で爪を記録の角度に戻してください"
        else:                                        # recorded closed on the fixed jaw: fold the arm, the jaw moves there
            how_en = ("`fd tool rest` (with a painting set-up: `fd paint rest`) folds the arm away with the tool's tip 5 cm "
                      f"up and is not held up by the jaw; then `{again}` again")
            how_ja = f"先に fd tool rest（絵のセットアップがあるときは fd paint rest）で畳んでから、{again} をやり直してください"
        nxt += f" (`fd ready` refuses while the jaw stands {abs(off)} ticks from where the tool file has it: {how_en})"
        say += f"（爪が記録の位置からずれているあいだは fd ready も拒否されます。{how_ja}。）"
    return _refused("plan", str(e), nxt, say, **more)


def _recorded(tool, cfg) -> dict:
    """What the tool file records about the jaw and the tip."""
    import tooling as TL
    raw, goal = tool.gripper_raw, tool.gripper_goal_raw
    return {"gripper_raw": raw, "jaw_q_deg": None if raw is None else round(TL.jaw_q_deg(cfg, raw), 2),
            "gripper_goal_raw": goal, "squeeze_ticks": None if goal is None or raw is None else int(raw) - int(goal),
            "tip_offset_mm": [round(float(v) * 1000.0, 1) for v in tool.tip_offset],
            "tip_measured": bool(tool.tip_measured)}


def _tip_text(tip_mm) -> str:
    return "(" + ", ".join(f"{float(v):.0f}" for v in tip_mm) + ") mm"


# ------------------------------------------------------------------ the arm and the file
def _look(sim: bool, skip: bool = False, at_ready: bool = False) -> tuple:
    """(jaw state, the arm's joint pose, None), or (None, None, why the arm was not read). Reads only."""
    import tooling as TL
    from hwlock import HardwareLocked
    from robot import SafetyError
    from servo_bus import BusError
    if skip:
        return None, None, "--no-robot"
    try:
        arm = make_arm(sim)
    except (BusError, SafetyError, HardwareLocked, OSError) as e:
        return None, None, f"{type(e).__name__}: {e}"
    try:
        if at_ready:
            TL.sim_place(arm, TL.ready_q(arm, arm.tool))
        return TL.jaw_state(arm), arm.read_q(), None
    except (BusError, SafetyError) as e:
        return None, None, f"{type(e).__name__}: {e}"
    finally:
        arm.close()


def _jaw_or_none(arm):
    import tooling as TL
    from robot import SafetyError
    from servo_bus import BusError
    try:
        return TL.jaw_state(arm)
    except (BusError, SafetyError):
        return None


def _save(tool, sim: bool) -> str:
    """Store the tool; returns the file. A simulated command never writes into the tree: fd points the simulated
    tool file elsewhere before it calls this module; called another way, it must be told where ($FD_TOOL)."""
    import tazuna_paths
    import tooling as TL
    from robot import SafetyError
    path = TL.tool_path(sim)
    if sim and not os.environ.get("FD_TOOL") and (
            os.path.realpath(path) + os.sep).startswith(os.path.realpath(tazuna_paths.ROOT) + os.sep):
        raise SafetyError(f"the simulated tool file would be written into the project folder ({path}): run this through "
                          "fd (fd --sim tool jaw and the same words), or name the file with $FD_TOOL")
    TL.save(tool, sim)
    return path


# ------------------------------------------------------------------ status
def _status(a, sim: bool) -> dict:
    import tooling as TL
    from robot import load_config
    cfg = load_config()
    tool, bad = _tool(sim)
    if bad:
        return bad
    if tool is None:
        return _sim_said({"ok": True, "stage": "status", "attached": False, "tool": None, "file": TL.tool_path(sim),
                          "next": "nothing is attached: the bare gripper is moved with `fd grip 0.5` (half open; 0 = closed, "
                                  "1 = open) and `fd grasp`",
                          "say": "ツールは登録されていません（爪は fd grip / fd grasp で動かせます）。"}, sim)
    jaw, _q, note = _look(sim, a.no_robot)
    rec = _recorded(tool, cfg)
    res = {"ok": True, "stage": "status", "attached": True, "tool": tool.name, "on_jaw": tool.on_jaw,
           "may_move": TL.jaw_locked(tool) is None, "recorded": rec, "jaw": jaw, "file": TL.source(sim)}
    if jaw is None:
        res["jaw_note"] = f"the arm was not read ({note}): only what the tool file records is shown"
    # does the jaw stand where the tool file has it? (what `fd ready` and brush work ask before they move:
    # tooling.jaw_in_place; None: not compared: the arm was not read, or there is nothing to compare with)
    tol = TL.jaw_tolerance(tool)
    off = None if jaw is None else jaw["from_record_ticks"]
    res["in_place"] = None if tol is None or off is None else bool(abs(off) <= tol)
    where = "" if tool.on_jaw is None else f"ツール「{tool.name}」は{WHERE_JA[tool.on_jaw]}と記録されています。"
    moved_ja = ("" if res["in_place"] is not False else
                f"爪が記録の位置（raw {rec['gripper_raw']}）から {abs(off)} ticks ずれています（いま raw {jaw['pos_raw']}）。"
                "このままでは fd ready も筆の作業も拒否されます。")
    if res["in_place"] is False and rec["gripper_goal_raw"] is None:
        # a jaw that no squeeze pulls back and that stands elsewhere than its record: the tool is not where its tip
        # was measured (the held jaw has its own branch below)
        if tool.on_jaw == "moving_outer":
            res["next"] = ("the jaw is NOT where the tool file has it: `fd tool jaw close` closes it on the fixed jaw and holds "
                           "it (to an angle instead: `fd tool jaw help` lists the words), then " + NEXT_MEASURE)
            res["say"] = where + moved_ja + "fd tool jaw close で閉じ直し、fd calib-air --tip-only --save で先端を測り直してください。"
        elif tool.on_jaw == "between":
            res["next"] = ("the jaw is NOT where the tool file has it, so the jaws do not hold the tool as they did: set it "
                           "back by hand and register it again with its tip as it is then (fd tool attach with --replace, "
                           f"--name {shlex.quote(tool.name)} and the tip: `fd tool attach --help` lists the words); when the "
                           "record is wrong: `fd tool jaw mount moving_outer` / `fd tool jaw mount fixed`")
            res["say"] = where + moved_ja + "道具を手で挟み直して fd tool attach --replace で登録し直してください。"
        else:
            res["next"] = ("the jaw is NOT where the tool file has it, and the file does not say on which jaw the tool is "
                           "taped: look at a photo of the gripper, then `fd tool jaw mount moving_outer` and `fd tool jaw "
                           "close` (a tool on the outer face of the moving jaw), or `fd tool jaw mount fixed` (the moving "
                           "jaw then does not carry it)")
            res["say"] = (f"ツール「{tool.name}」がどちらの爪に付いているかは、まだ記録されていません。" + moved_ja
                          + "グリッパーの写真で確かめてから、可動爪の外側なら fd tool jaw mount moving_outer と fd tool jaw close、"
                          "固定爪なら fd tool jaw mount fixed を実行してください。")
    elif tool.on_jaw is None:
        # One command per pair of backticks. This read "fd tool jaw mount moving_outer|fixed|between" in one pair: typed as it was
        # printed, the shell ran `fd tool jaw mount moving_outer` and piped it into "fixed" (command not found), and
        # the tool file said moving_outer for a tool nobody had looked at (review of 2026-11-14).
        res["next"] = ("look at a photo of the gripper first, then record on which jaw the tool is taped: `fd tool jaw mount "
                       "moving_outer` (the outer face of the moving jaw: only then may the jaw be moved), `fd tool jaw mount "
                       "fixed` or `fd tool jaw mount between`")
        res["say"] = (f"ツール「{tool.name}」がどちらの爪に付いているかは、まだ記録されていません。グリッパーの写真で確かめてから、"
                      "fd tool jaw mount moving_outer（可動爪の外側）、fd tool jaw mount fixed（固定爪）、fd tool jaw mount between"
                      "（爪の間）のどれかで記録してください。")
        if _paint_first(tool):       # (FIRST_PAINT has the reason: one order for every text)
            res["next"] = _cap(FIRST_PAINT) + ". For the jaw itself: " + res["next"]
            res["say"] += FIRST_PAINT_JA
    elif tool.on_jaw != "moving_outer":
        # (`fd tool rest` is refused over the canvas of a painting set-up: there `fd paint rest` is the fold)
        fold = "fd paint rest" if _paint_setup() else "fd tool rest"
        res["next"] = f"the jaw stays locked for this tool (`{fold}` folds the arm away with it)"
        res["say"] = where + "爪は動かしません（グリッパーはロックされたままです）。"
    elif jaw is None:
        res["next"] = ("`fd tool jaw close --dry` shows what closing the jaw would do" if rec["gripper_goal_raw"] is None else
                       "the file records a jaw held closed with a squeeze; the next powered command puts the squeeze "
                       "back (`fd ready`, a painting command that moves, `fd tool jaw close`)")
        res["say"] = where + "アームを読めなかったので、記録だけを表示しています。"
        if _paint_first(tool):
            res["next"] = (_cap(FIRST_PAINT) + ". After that: " + res["next"] if rec["gripper_goal_raw"] is None else
                           res["next"] + ". " + _cap(NEXT_TIP_HELD))
            res["say"] += FIRST_PAINT_JA if rec["gripper_goal_raw"] is None else HELD_PAINT_JA
    elif rec["gripper_goal_raw"] is not None:
        off = jaw["from_record_ticks"]
        if abs(off) > TL.JAW_HOLD_TOL_TICKS:
            res["next"] = ("the jaw is NOT where the tool file has it: `fd tool jaw close` closes it on the fixed jaw again, "
                           "then " + NEXT_MEASURE)
            res["say"] = (where + f"爪が記録の位置（raw {rec['gripper_raw']}）から {abs(off)} ticks ずれています（いま raw "
                          f"{jaw['pos_raw']}）。fd tool jaw close で閉じ直し、fd calib-air --tip-only --save で先端を測り直してください。")
        elif jaw["squeeze_on"]:
            after = NEXT_PAINT if tool.tip_measured else _cap(NEXT_TIP_HELD if _paint_setup() else NEXT_TIP)
            # (words, not `fd tool jaw open --to DEG`: typed as it was printed, that is refused)
            res["next"] = (f"held: the tool has no play. {after}. Opening the jaw to an angle (the tip then has to be measured "
                           "again) is in `fd tool jaw help`")
            res["say"] = (where + f"爪は固定爪に押し当てて保持しています（raw {jaw['pos_raw']}、押し付け {jaw['squeeze_ticks']} "
                          f"ticks、負荷 {jaw['load']}、{jaw['temp_c']} ℃）。") + (HELD_PAINT_JA if _paint_first(tool) else "")
        else:
            res["next"] = ("the squeeze is off (the torque was off): the next powered command puts it back (`fd ready`, "
                           "a painting command that moves); `fd tool jaw close` does it now")
            res["say"] = (where + f"爪は記録どおりの位置（raw {jaw['pos_raw']}）にありますが、押し付けは切れています。"
                          "次にアームを動かすコマンドが自動で戻します。")
            if _paint_first(tool):
                res["next"] += ". " + _cap(NEXT_TIP_HELD)
                res["say"] += HELD_PAINT_JA
    else:
        res["next"] = ("`fd tool jaw close --dry` shows what closing does to the tool tip; `fd tool jaw close` closes the jaw "
                       "against the fixed jaw and holds it (the tool then has no play)")
        res["say"] = where + f"爪は {jaw['q_deg']:.1f} 度（raw {jaw['pos_raw']}）にあり、押し付けでは保持していません。"
        if _paint_first(tool):
            res["next"] = _cap(FIRST_PAINT) + ". After that: " + res["next"]
            res["say"] += FIRST_PAINT_JA
    return _sim_said(res, sim)


# ------------------------------------------------------------------ mount
def _mount(a, sim: bool) -> dict:
    import tooling as TL
    was = _sim_state(sim)
    tool, bad = _tool(sim)
    if bad:
        return bad
    if tool is None:
        return _no_tool()
    before = tool.on_jaw
    notes = []
    if before != a.where:
        tool.on_jaw = a.where
        if a.where != "moving_outer" and tool.gripper_goal_raw is not None:
            tool.gripper_goal_raw = None      # (the holding goal belongs to a tool on the moving jaw's outer face)
            notes.append("the recorded holding goal of the jaw was dropped: the gripper is locked for this tool again")
    saved = before != a.where and not a.dry
    path = _save(tool, sim) if saved else TL.tool_path(sim)
    res = {"ok": True, "stage": "planned" if a.dry else "mount", "tool": tool.name, "on_jaw": a.where, "before": before,
           "changed": before != a.where, "saved": saved, "may_move": a.where == "moving_outer", "notes": notes, "file": path}
    done = "記録する内容を確認しました（保存していません）" if a.dry else "記録しました"
    if a.where == "moving_outer":
        res["next"] = ("the jaw of this tool may now be moved, and NOTHING may be between the jaws: `fd tool jaw close --dry` "
                       "shows what closing does, `fd tool jaw close` closes the jaw on the fixed jaw and holds it")
        res["say"] = (f"ツール「{tool.name}」は可動爪の外側に付いている、と{done}。爪を動かすコマンド（fd tool jaw close / open）が"
                      "使えます。爪の間に何も挟まっていないことを写真で確かめてから使ってください。")
        if tool.gripper_goal_raw is None and _paint_first(tool):
            # (FIRST_PAINT has the reason: with the tip not on record, `fd paint status` comes before the jaw is closed)
            res["next"] = ("the jaw of this tool may now be moved, and NOTHING may be between the jaws. " + _cap(FIRST_PAINT)
                           + ". After that: `fd tool jaw close --dry` shows what closing does, `fd tool jaw close` closes the "
                             "jaw on the fixed jaw and holds it")
            res["say"] += FIRST_PAINT_JA
    else:
        res["next"] = "the jaw stays locked for this tool; `fd tool jaw status` shows the record"
        res["say"] = f"ツール「{tool.name}」は{WHERE_JA[a.where]}、と{done}。爪は動かしません。"
    return _sim_said(res, sim, wrote=saved, was=was)


# ------------------------------------------------------------------ close / open
def _move(a, sim: bool) -> dict:
    """`close` / `open`: the gate and the options from the tool file alone, then the dry run or the move."""
    import tooling as TL
    from robot import SafetyError, load_config
    tool, bad = _tool(sim)
    if bad:
        return bad
    if tool is None:
        return _no_tool()
    why = TL.jaw_locked(tool, "closing the jaw" if a.cmd == "close" else "opening the jaw")
    if why:
        return _locked(tool, why)
    cfg = load_config()
    base = {"tool": tool.name, "on_jaw": tool.on_jaw}
    try:
        if a.at_ready and not sim:
            to = "" if a.to in (None, "closed") else f" --to {shlex.quote(str(a.to))}"
            raise SafetyError(f"--at-ready places the SIMULATED arm (`fd --sim tool jaw {a.cmd}{to} --at-ready`); the real arm "
                              "is brought to its stand-by pose with `fd ready`. Nothing moved")
        plan = TL.jaw_plan(cfg, tool, a.cmd, a.to, getattr(a, "squeeze", None), a.step, a.lag)
    except SafetyError as e:
        return _options(e, **base)
    return _dry(a, sim, tool, cfg, plan, base) if a.dry else _run(a, sim, base)


def _dry(a, sim: bool, tool, cfg, plan: dict, base: dict) -> dict:
    """What the jaw move would do: the plan, where the jaw model has the tool tip afterwards, and (with the arm
    readable) whether the tool clears the table while it swings. Nothing moves, nothing is written."""
    import tooling as TL
    from robot import SafetyError
    jaw, q_arm, note = _look(sim, a.no_robot, a.at_ready)
    res = {"ok": True, "stage": "planned", "moved": False, "saved": False, **base, "recorded": _recorded(tool, cfg),
           "jaw": jaw}
    if jaw is None:
        res["jaw_note"] = (f"the arm was not read ({note}): the plan is from the tool file alone, and the swing of the tool "
                           "was not checked against the table")
    else:
        try:
            plan = TL.jaw_plan(cfg, tool, a.cmd, a.to, getattr(a, "squeeze", None), a.step, a.lag, pos_raw=jaw["pos_raw"])
        except SafetyError as e:
            return _options(e, **base, jaw=jaw)
        try:
            res["swing"] = TL.jaw_swing(cfg, tool, q_arm, jaw["pos_raw"], plan["expect_raw"])
        except SafetyError as e:
            return _too_low(e, _typed(a), True, sim, (tool, cfg), **base, jaw=jaw)
    est = TL.jaw_estimate(cfg, tool, plan["expect_raw"])
    stays = tool.gripper_raw is not None and abs(plan["expect_raw"] - int(tool.gripper_raw)) <= TL.JAW_SAME_TICKS
    res.update(plan=plan, estimate=est, tool_would_move=not stays)
    goal_ja = "固定爪に当たるまで" if plan["target_raw"] is None else f" {plan['to']:g} 度まで"
    verb_ja = "閉じる" if a.cmd == "close" else "開く"
    does = f"`fd tool jaw {_typed(a)}` does it"       # (the same move with the options it was planned with)
    res["next"] = does + "; afterwards " + NEXT_MEASURE
    if stays:            # the jaw stands where the tool file has it: nothing swings, only the hold changes
        res["next"] = (does + ": the jaw is expected to end where the tool file has it, so the tool does not move and its "
                       "tip and what was measured with it are kept" + ("; the squeeze is put on" if plan["squeeze_ticks"] else ""))
        res["say"] = (f"可動爪を{goal_ja}{verb_ja}計画を確認しました（何も動かしていません）。爪はすでに記録どおりの位置にあるので、"
                      "ツールは動かず、先端の値はそのまま使える見込みです" + ("（押し付けだけを掛け直します）。" if plan["squeeze_ticks"] else "。"))
    elif est is None:
        res["say"] = (f"可動爪を{goal_ja}{verb_ja}計画を確認しました（何も動かしていません）。ツールの記録に爪の位置がないため、"
                      "動かしたあとの先端位置は推定できません。動かしたあと fd calib-air --tip-only --save で測ってください。")
    else:
        res["say"] = (f"可動爪を{goal_ja}{verb_ja}計画を確認しました（何も動かしていません）。ツールは爪と一緒に動き、先端は"
                      f"グリッパー座標でおよそ {_tip_text(est['tip_offset_mm'])} に移ります（爪の回転からの推定で、実機では "
                      f"{est['expect_error_mm']:.0f} mm ほどずれました）。動かしたあと fd calib-air --tip-only --save で測り直してください。")
    return _sim_said(res, sim)


def _bus_gone(e, cmd: str, base: dict, tool=None, cfg=None, path=None, saved: bool = False) -> dict:
    """The servo bus stopped answering after the arm had been opened (a cable, the supply, a servo that reset).
    cmd: the words of the move (_typed), for the next that names it again.
    The complete result, stage "hardware". What is known about the jaw comes from tooling.move_jaw, on the
    exception (`jaw_left`): nothing commanded yet / the jaw read afterwards / its position NOT known. In the
    last case the tool file has been made to say so (tooling.jaw_lost, stored by _run): until 2026-11-13 it went
    on saying that the jaw was where it had been and that the tip was measured."""
    left = getattr(e, "jaw_left", None)
    res = {"ok": False, "error": type(e).__name__, "stage": "hardware", **base, "jaw": None, "file": path, "saved": saved,
           "message": str(e)}
    if tool is not None and cfg is not None:
        res["recorded"] = _recorded(tool, cfg)
    again = f"`fd tool jaw {cmd}` again"
    if left is None:
        res.update(position_known=None,
                   reason=f"the servo bus stopped answering before the jaw was commanded ({e}): this command did not "
                          "move the jaw, and the tool file is as it was",
                   next=f"`fd state` shows whether the servos answer again; then `fd tool jaw status`, and {again}",
                   say="サーボが応答しなくなりました（爪を動かす前です）。爪は動かしておらず、ツールの記録も変えていません。"
                       "fd state で応答を確かめてからやり直してください。")
    elif left["pos_raw"] is None:
        res.update(position_known=False,
                   reason=f"the servo bus stopped answering in the middle of the jaw move ({e}): where the jaw stands is NOT "
                          "known, and "
                          + ("nothing pushes on it (the gripper torque is off)" if left["released"] else
                             "the push on it could not be taken off (the servo keeps its last goal while it has power)")
                          + ". The tool file now says so: no jaw position, no holding goal, and the tip is not measured "
                          "any more" + ("" if saved else " (the tool file could NOT be written)"),
                   next="look at the gripper and at the servos' power; when `fd state` answers again: `fd tool jaw status`, "
                        "then `fd tool jaw close` closes the jaw again, then " + NEXT_MEASURE,
                   say="爪を動かしている途中でサーボが応答しなくなりました。爪がどこにあるか分かりません"
                       + ("。" if left["released"] else "（押し付けが残っているかもしれません）。")
                       + ("ツールの記録は「爪の位置は不明・先端は未測定」に書き換えました。" if saved else
                          "ツールの記録ファイルは書き換えられませんでした（爪の位置と先端の値は、もう正しくありません）。")
                       + "グリッパーとサーボの電源を確かめ、fd state で応答が戻ってから fd tool jaw status、fd tool jaw close、"
                         "fd calib-air --tip-only --save の順にやり直してください。")
    else:
        off = "the push was taken off" if left["released"] else "the push on it could NOT be taken off"
        res.update(position_known=True,
                   reason=f"the servo bus failed in the middle of the jaw move ({e}); afterwards the jaw was read at raw "
                          f"{left['pos_raw']}, and {off}. The tool file says where the jaw stands",
                   next=f"`fd state` shows whether the servos answer; `fd tool jaw status` where the jaw stands; then {again}, "
                        "then " + NEXT_MEASURE,
                   say=f"爪を動かしている途中でサーボとの通信が途切れました。爪は raw {left['pos_raw']} にあります"
                       + ("（押し付けは解除してあります）。" if left["released"] else "（押し付けを解除できませんでした）。")
                       + "fd state で応答を確かめ、fd tool jaw status で位置を見てからやり直してください。")
    return res


def _run(a, sim: bool, base: dict) -> dict:
    """The move itself, the bookkeeping, the tool file. A servo bus that stops answering once the arm is open
    is answered as a result (_bus_gone), with the tool file saying what is known about the jaw."""
    import tooling as TL
    from robot import MotionAborted, SafetyError
    from servo_bus import BusError
    squeeze = getattr(a, "squeeze", None)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, signal.default_int_handler)      # a cancelled call still takes the push off
    was = _sim_state(sim)
    arm = make_arm(sim)            # (the hardware lock, a bus that is off: BusError; fd turns it into the result)
    tool = cfg = None
    try:
        tool = arm.tool
        why = TL.jaw_locked(tool, "closing the jaw" if a.cmd == "close" else "opening the jaw")
        if why:                    # (the arm's own reading of the tool file)
            return _no_tool() if tool is None else _locked(tool, why)
        cfg = arm.cfg
        if a.at_ready:
            try:
                TL.sim_place(arm, TL.ready_q(arm, tool))
            except SafetyError as e:
                return _refused("plan", str(e), "`fd --sim ready` says the same; `fd tool status` shows the tool",
                                f"待機姿勢を作れませんでした（{e}）。爪は動かしていません。", **base)
        jaw0 = TL.jaw_state(arm)
        try:
            plan = TL.jaw_plan(cfg, tool, a.cmd, a.to, squeeze, a.step, a.lag, pos_raw=jaw0["pos_raw"])
        except SafetyError as e:
            return _options(e, **base, jaw=jaw0)
        try:
            TL.jaw_swing(cfg, tool, arm.read_q(), jaw0["pos_raw"], plan["expect_raw"])
        except SafetyError as e:
            return _too_low(e, _typed(a), False, sim, (tool, cfg), **base, jaw=jaw0)
        before = json.dumps(tool.to_dict(), sort_keys=True, default=str)
        path = TL.tool_path(sim)
        rep = cut_short = None
        saved = False
        try:
            rep = TL.move_jaw(arm, a.cmd, a.to, squeeze, a.step, a.lag)
        except SafetyError as e:   # a servo too hot, a supply too low, a speed tier that is not validated: nothing moved
            return _refused("jaw", str(e), "`fd state` shows the servos; `fd tool jaw status` where the jaw stands",
                            f"爪を動かせませんでした（{e}）。爪は動かしていません。", **base, jaw=_jaw_or_none(arm))
        except (MotionAborted, KeyboardInterrupt, BusError) as e:
            cut_short = e
        finally:
            # whatever happened, the file says where the jaw is now, or that nobody knows (move_jaw keeps the tool
            # in memory up to date, also on its way out of an error)
            if json.dumps(tool.to_dict(), sort_keys=True, default=str) != before:
                path = _save(tool, sim)
                saved = True
        if isinstance(cut_short, BusError):
            return _sim_said(_bus_gone(cut_short, _typed(a), base, tool, cfg, path, saved), sim, wrote=saved, was=was)
        res = {"ok": False, "stage": a.cmd, **base, "jaw": _jaw_or_none(arm), "file": path, "recorded": _recorded(tool, cfg)}
        if cut_short is not None:
            what = "interrupted" if isinstance(cut_short, KeyboardInterrupt) else f"{type(cut_short).__name__}: {cut_short}"
            left = getattr(cut_short, "jaw_left", None)        # (what tooling.move_jaw found on its way out)
            freed = left is None or left["released"]
            res.update(error=type(cut_short).__name__, interrupted=isinstance(cut_short, KeyboardInterrupt), saved=saved,
                       reason=f"the jaw move was cut short ({what}); "
                              + ("the push was taken off and the jaw is not held" if freed else
                                 "the push on the jaw could NOT be taken off (the gripper servo did not take the goal)"),
                       next="`fd tool jaw status` shows where the jaw stands; `fd tool jaw close` closes it again; then "
                            + NEXT_MEASURE,
                       say="爪の動作が途中で止まりました。" + ("押し付けは解除してあります。" if freed else "押し付けを解除できませんでした。")
                           + "fd tool jaw status で位置を確かめ、閉じ直したあと fd calib-air --tip-only --save で先端を測り直してください。")
            return _sim_said(res, sim, wrote=saved, was=was)
        rec = rep.pop("record")
        est = rec["estimate"]
        res.update(ok=bool(rep["ok"]), moved=rep["pos_raw"] != rep["from_raw"], tool_moved=rec["tool_moved"],
                   stopped_early=rep["stopped_early"], met_fixed_jaw=rep["met_fixed_jaw"], move=rep,
                   tip_offset_mm=[round(float(v) * 1000.0, 1) for v in tool.tip_offset],
                   tip_measured=bool(tool.tip_measured), estimate=est, dropped=rec["dropped"], saved=True)
        where = f"raw {rep['pos_raw']}、{rep['q_deg']:.1f} 度"
        figures = f"負荷 {rep['load']}、{rep['temp_c']} ℃"
        if rec["tool_moved"]:
            moved_en = ("the tool MOVED with the jaw, its tip is not measured any more: " + NEXT_MEASURE
                        + (f"; until then the tool file has the jaw model's estimate {_tip_text(est['tip_offset_mm'])} (the old "
                           f"tip turned {est['turned_deg']:g} deg about the jaw pivot), which was about "
                           f"{est['expect_error_mm']:.0f} mm off on the real arm" if est else
                           "; the tool file keeps the OLD tip offset, which is wrong now (it did not record where the jaw "
                           "stood for it)"))
            moved_ja = ("ツールは爪と一緒に動いたので、先端は未測定に戻しました。fd calib-air --tip-only --save で測り直してください"
                        + ("（いまの値は爪の回転からの推定で、実機では 1 cm ほどずれました）。" if est else "（先端の値は古いままです）。"))
        else:
            # (a finished jaw move names the command that comes next, by what the tool file says: the result used to
            # end with "... are kept", and the operator had to know that `fd paint status` goes on from there)
            # (the tip not on record, a painting set-up on record: `fd paint status` leads, by whether a hold is on
            # record now: FIRST_PAINT has the reason)
            led = None if not _paint_first(tool) else "held" if tool.gripper_goal_raw is not None else "first"
            moved_en = ("the jaw ended where the tool file had it: the tool did not move, its tip and what was measured "
                        "with it are kept. Next: " + (NEXT_PAINT if tool.tip_measured else
                                                      {None: NEXT_TIP, "held": NEXT_TIP_HELD, "first": FIRST_PAINT}[led]))
            moved_ja = ("爪の位置は記録と同じなので、ツール先端の値はそのまま使えます。次の手順は fd paint status が示します。"
                        if tool.tip_measured else
                        "爪の位置は記録と同じなので、ツール先端の値は変えていません。"
                        + {None: "筆先はまだ測定の記録がないので、fd calib-air --tip-only --save で測って記録してください。",
                           "held": HELD_PAINT_JA, "first": FIRST_PAINT_JA}[led])
        if rep["ok"] and rep["met_fixed_jaw"]:
            held = rep["squeeze_ticks"] > 0
            res["next"] = (("held against the fixed jaw; " if held else "closed on the fixed jaw, NOT held (squeeze 0); ")
                           + moved_en)
            res["say"] = ("可動爪を固定爪に当たるまで閉じ、"
                          + (f"軽く押し付けて保持しました（{where}、押し付け {rep['squeeze_ticks']} ticks、{figures}）。" if held else
                             f"押し付けずに止めました（{where}、{figures}）。") + moved_ja)
        elif rep["ok"]:
            res["next"] = (f"the jaw stands at {rep['q_deg']:.1f} deg, held by its position only (no squeeze: the tool has the "
                           "jaw's backlash); " + moved_en)
            res["say"] = (f"可動爪を {rep['q_deg']:.1f} 度（raw {rep['pos_raw']}）まで" + ("閉じました" if a.cmd == "close" else "開きました")
                          + f"（{figures}）。" + moved_ja)
        else:
            res["reason"] = rep["why"]
            res["next"] = (("look at the gripper (a photo: `fd snap`): something keeps the jaw from moving; `fd tool jaw "
                            "status` shows where it stands. " if rep["stopped_early"] else
                            "the fixed jaw was not found where the config has it (\"gripper_touch_raw\"): `fd tool jaw "
                            "status`, and look at the gripper. ") + moved_en[0].upper() + moved_en[1:])
            res["say"] = ((f"爪が途中（{where}）で止まりました。何かが当たっています。" if rep["stopped_early"] else
                           f"固定爪に当たる位置が見つかりませんでした（{where}）。") + "押し付けは解除してあります。写真で確かめてください。"
                          + (moved_ja if rec["tool_moved"] else ""))
        return _sim_said(res, sim, wrote=saved, was=was)
    except BusError as e:          # (before the jaw was commanded: the look at the jaw, the arm's pose)
        return _bus_gone(e, _typed(a), base, tool, cfg)
    finally:
        arm.close()


# ------------------------------------------------------------------ entry
def _help(words) -> dict:
    """The answer to `fd tool jaw`, `fd tool jaw help`, `fd tool jaw --help`, `fd tool jaw -h` and to a help word
    behind a sub-command (`fd tool jaw close --help`): the usage line, and the options with their texts ("help": what argparse
    would print, as lines; of the sub-command that was named, else of all four). Nothing is read from the arm or
    the tool file. fd hands `--help` and `-h` through since 2026-11-13 (its own parser used to stop at them with
    exit code 2 and no result), so they must be answered here, as a result."""
    parser = build_parser()
    named = [w for w in words if w in COMMANDS][:1]
    lines = []
    for name in named or COMMANDS:
        lines += [ln for ln in parser.commands[name].format_help().splitlines() if ln.strip()]
    return {"ok": True, "stage": "help", "usage": USAGE, "command": named[0] if named else None, "help": lines,
            "next": "`fd tool jaw status` shows where the jaw stands and what the tool file records",
            "say": "fd tool jaw の使い方: status（状態）/ mount（どの爪に付いているかの記録）/ close（閉じて保持）/ open（開く）。"
                   "何も動かしていません。"}


def main(argv=None) -> dict:
    argv = list(sys.argv[1:] if argv is None else argv)
    sim = bool(argv) and argv[0] == "--sim"
    words = argv[1:] if sim else argv
    if words[:1] == ["--"]:
        words = words[1:]          # (fd drops one itself; kept for a caller that hands the words on as they were typed)
    if "--sim" in words:
        # fd read this as a command for the REAL arm (its --sim stands before the command): never guess which was meant
        return _refused("options", "--sim must stand before the command (fd --sim tool jaw, then its words). Nothing moved",
                        "`fd --sim tool jaw " + " ".join(w for w in words if w != "--sim") + "`",
                        "--sim は fd のすぐ後ろに書いてください（fd --sim tool jaw ...）。何も動かしていません。")
    if not words or words[0] == "help" or any(w in ("--help", "-h") for w in words):
        return _help(words)
    try:
        a = build_parser().parse_args(words)
    except _Usage as e:
        # (what was typed is quoted, not put in backticks: backticks are for commands that work as they are typed)
        return _refused("options", f"\"fd tool jaw {' '.join(words)}\": {e}", "usage: " + USAGE,
                        f"コマンドを読めませんでした（{e}）。何も動かしていません。")
    if a.cmd == "status":
        return _status(a, sim)
    if a.cmd == "mount":
        return _mount(a, sim)
    return _move(a, sim)


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    import tazuna_paths  # noqa: F401
    _res = main()
    print(json.dumps(_res, ensure_ascii=False, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)))
    sys.exit(0 if _res.get("ok") else 1)
