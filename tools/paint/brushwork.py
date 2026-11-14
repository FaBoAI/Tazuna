"""Brush work: the moves of a paint brush that is taped to the gripper (a registered tool, tooling.Tool).

What a brush needs that a pen does not:
- it travels between the paper, the paint wells and the water dishes, over rims: every far move ("hop") is a
  checked free move (tooling.free_move) with the tip at the travel height, and the brush is first raised
  straight up to that height when it is lower;
- it is pressed INTO things on purpose (bristles bend): the paper, a blob of paint, the bottom of a dish. How
  far the modelled tip may go below a surface is capped here (MAX_PRESS_M), and no move goes below
  table_z - MAX_BELOW_TABLE_M whatever the caller computed;
- its pitch differs from place to place (far wells are reached with the brush leaning more): one operation
  (a dab into a well, one stroke) keeps ONE pitch that the arm can hold at every point of it (pitch_at).

Everything is planned and checked before it moves (Arm.plan_tip_path, Arm.run_path's own checks,
tooling.check_pose); a refusal is a SafetyError and leaves the arm where it stands.

What the evening of 2026-11-13 added (a thin round brush, 90 strokes and about 60 loads on the real arm):
- a far hop that the height monitor stops (the MEASURED tip dipped up to 45 mm under the planned 65 mm in long
  joint-space moves at 30 deg/s: tracking lag between shoulder and elbow, not sag: a static hold agreed with the
  plan within 3 mm at seven places) is tried ONCE more, from the travel height, at half speed (`hop_retries`
  counts them); a second stop ends the command. The hop speed is 20 deg/s since 18:15 of that day: no stop after;
- a brush taped to the MOVING jaw is rigid only while the jaw is squeezed shut against the fixed jaw: `power()`
  asks tooling.hold_jaw to put the squeeze back whenever the torque comes on (the gripper's goal is its own
  position then, and the brush would float by millimetres);
- every way out says where the brush is (`retreat` -> "where", "height_mm"), also after an interrupt; when the
  servos themselves stop answering, `lost` says that it is not known.

What the review of that day's update added (2026-11-13, late):
- a brush whose tip was never measured does not move here: the tool file must carry tip_measured (what
  `fd calib-air` records). That evening a first guess of the tip was 9 cm short of the real one; brush work
  lowers the tip to 2.5 mm above the table by that number. `unmeasured_tip_ok=True` is for what only computes,
  and for the operator who has looked at a photo of gripper and brush (`--tip-looked-at`);
- before the torque comes on, the jaw is compared with the tool file where tooling can do that
  (tooling.jaw_in_place: a jaw that stands elsewhere carries the brush tip elsewhere), and a jaw that is SHUT
  while the tool file records no hold is said (`jaw_note`): nothing can be between jaws that are shut, so the
  brush is on a jaw's outer face, and without the squeeze it has the play of that evening.

Units: metres, base frame; "h" = height above the table (cfg["table_z"]); speeds in mm/s of the tool tip.
Used by paintskill.py. Tests: tests/test_paint.py (simulated arm).
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

import kinematics as K
import tooling as TL
import tracking as TR
from robot import MotionAborted, SafetyError

MAX_PRESS_M = 0.006           # a stroke never presses the modelled tip further below its contact height
MAX_BELOW_TABLE_M = 0.006     # ... and nothing sends the modelled tip further below the table than this
MIN_TRAVEL_H = 0.035
MOTION_ERRORS = (SafetyError, MotionAborted)
# where the brush was left, as the operator hears it (retreat() -> "where")
WHERE_JA = {"not moved": "アームは動かしていません。",
            "photo pose": "アームは撮影姿勢（トルクON）です。",
            "lifted": "アームは筆を持ち上げた位置で止まっています（トルクON）。",
            "stopped": "アームはその場で止まっています（トルクON、筆の持ち上げに失敗）。",
            "limp": "トルクはOFFです（収納姿勢とは限りません: fd state で確認）。",
            "unknown": "アームの位置を読めません（サーボが応答しません）。その場で止まっているか、トルクが切れています: "
                       "目で確認してください。"}
JAW_NEXT = ("`fd tool jaw mount moving_outer` (after looking at a photo of gripper and brush), then `fd tool jaw close`: "
            "the jaw is closed on the fixed jaw and held with a light squeeze")


class JawRefusal(SafetyError):
    """Brush work was refused because the jaw does not stand where the tool file has it, or cannot be held there
    (what tooling.jaw_in_place / hold_jaw refuse): the tool is no longer where its tip was measured. A
    SafetyError of its own, so that a command can answer with the way out (`fd tool jaw status`) instead of its
    general hint, which led round in a circle (status ok, check ok, run --dry "planned", run refused)."""


def tip_is_measured(tool) -> bool:
    """Does the tool file say that the tip was measured (Tool.tip_measured, the record `fd calib-air` writes)?
    A typed tip, a first guess and the jaw model's estimate are not: the record is dropped whenever the tip
    offset changes (tooling.Tool)."""
    return bool(getattr(tool, "tip_measured", None))


def unmeasured_text(tool) -> str:
    return (f"the tip of tool '{getattr(tool, 'name', '?')}' was never measured (the tool file has no tip_measured: its "
            "tip offset is typed in, a first guess or an estimate). Brush work lowers the tip to a few millimetres "
            "above the table by that number, and on 2026-11-13 a first guess was 9 cm short of the real tip. "
            "Nothing moved")


def jaw_note(tool, touch_raw) -> str | None:
    """What has to be said about the jaw of this tool before it paints, or None: the tool file has the jaw SHUT
    (gripper_raw at or below where the fingertips touch) and records no hold (gripper_goal_raw), for a tool that
    is not recorded as taped to the fixed jaw or held between the jaws. Nothing can be between jaws that are
    shut: the brush is on the outer face of a jaw, and on the moving jaw it is rigid only while the jaw is
    squeezed against the fixed one (2026-11-13: without the squeeze pushed and sideways strokes skipped, and
    first touch was 4 mm higher). The rig's own file of that evening is like that: the squeeze was put on by a
    scratch script and stands only in its notes."""
    if tool is None or getattr(tool, "gripper_goal_raw", None) is not None:
        return None
    raw, on = getattr(tool, "gripper_raw", None), getattr(tool, "on_jaw", None)
    try:
        shut = raw is not None and touch_raw is not None and int(raw) <= int(touch_raw)
    except (TypeError, ValueError):
        shut = False
    if not shut or on in ("fixed", "between"):
        return None
    where = ("the tool file does not say on which jaw the brush is taped" if on is None else
             "the brush is recorded on the outer face of the moving jaw")
    return (f"the jaw is shut (raw {int(raw)}, the fingertips touch at {int(touch_raw)}) but no jaw hold is recorded: {where}, "
            "and nothing puts the squeeze back when the torque comes on. A brush on the moving jaw then has play "
            f"(2026-11-13: sideways and pushed strokes skipped, first touch 4 mm higher). {JAW_NEXT}")


@dataclass
class BrushOptions:
    # Tip above the table while travelling (palette rim ~15 mm, dish rim ~25 mm). On the real arm the MEASURED
    # tip ran 13 mm under the plan in a free move with the arm stretched out (2026-11-13: planned 45, seen 32):
    # the height must leave room for that.
    travel_h: float = 0.065
    sag_m: float = 0.03           # a free move is stopped when the measured tip is this far under the lower end height
    above_h: float = 0.030        # a far hop to a stroke start ends this high above the table
    hover_m: float = 0.010        # brush-up height above the contact height between strokes
    near_m: float = 0.07          # a next start nearer than this is reached at the hover height
    v_stroke: float = 20.0
    v_down: float = 15.0
    v_up: float = 30.0
    v_glide: float = 40.0
    v_stir: float = 25.0
    hop_deg_s: float = 20.0   # 30 until 2026-11-13 18:15: with the brush tip under the fingers the measured tip dipped 45 mm in long hops
    ignore_keepouts: tuple = ()   # keep-outs a TIP move may enter (e.g. ("cables",) when the paper covers that area)

    def check(self) -> "BrushOptions":
        def num(name, lo, hi):
            v = getattr(self, name)
            if not (isinstance(v, (int, float)) and math.isfinite(v) and lo <= v <= hi):
                raise SafetyError(f"brush option {name} = {v!r} is outside {lo} .. {hi}")
        num("travel_h", MIN_TRAVEL_H, 0.12)
        num("sag_m", 0.01, 0.04)
        num("above_h", 0.025, 0.08)
        num("hover_m", 0.004, 0.03)
        num("near_m", 0.0, 0.15)
        for name in ("v_stroke", "v_down", "v_up", "v_glide", "v_stir"):
            num(name, 1.0, 80.0)
        num("hop_deg_s", 5.0, 30.0)
        return self


class BrushRun:
    """One command's worth of brush moves on one arm. Create it, call the moves, call retreat() on the way out."""

    def __init__(self, arm, opts: BrushOptions | None = None, log=None, unmeasured_tip_ok: bool = False):
        if getattr(arm, "tool", None) is None:
            raise SafetyError("brush work needs a registered tool (`fd tool attach --help` shows how a brush is registered)")
        if not unmeasured_tip_ok and not tip_is_measured(arm.tool):
            raise SafetyError(unmeasured_text(arm.tool) + ". `fd calib-air --save` measures it and records that")
        self.arm, self.tool = arm, arm.tool
        self.o = (opts or BrushOptions()).check()
        self.tz = TL.table_z(arm)
        self.roll = float(arm._c("ready")["roll"])
        self.tl = K.tool_offset(*self.tool.tip_offset)
        self.floor = float(arm.cfg["floor_z"]) + float(arm.cfg["floor_clearance"])
        self.z_min = self.tz - MAX_BELOW_TABLE_M
        self.log = log or (lambda *a, **k: None)
        # seconds of the arm's own clock: simulated time on the simulated arm, so that time limits and the
        # time model can be tested without waiting
        self.clock = getattr(arm, "_clock", None) or time.perf_counter
        self.sp = None
        self.moved = False
        self.lowest_z = None
        self.path_m = 0.0            # tip travel in contact (strokes), for the paint budget
        self.moves = 0
        self.hop_retries = 0         # far moves that the height monitor stopped and that were done again, slower
        self.jaw = None              # what tooling.hold_jaw answered (a tool taped to the moving jaw), else None
        self.jaw_place = None        # what tooling.jaw_in_place answered before the torque came on, else None
        self.jaw_note = None         # the jaw is shut and no hold is recorded (jaw_note): said once the brush is powered
        self._jaw_asked = False

    # ------------------------------------------------------------------ where the brush is
    def tip(self, q=None) -> np.ndarray:
        q = self.arm.read_q() if q is None else q
        return K.fk(np.asarray(q, dtype=float), self.tl)[:3, 3].copy()

    def height(self, q=None) -> float:
        return float(self.tip(q)[2]) - self.tz

    def pitch_now(self, q=None) -> float:
        q = self.arm.read_q() if q is None else q
        return float(K.pose_summary(np.asarray(q, dtype=float), self.tl)["pitch_down_deg"])

    def pitch_at(self, pts, want: float | None = None) -> float:
        """One gripper pitch (deg below horizontal) the arm can hold with the tool tip at EVERY one of the points,
        as near to `want` (default: the tool's working pitch) as they allow. SafetyError when there is none."""
        lim = self.arm.q_limits()
        roll = math.radians(self.roll)
        pts = [np.asarray(p, dtype=float) for p in pts]
        pitch = float(self.tool.pitch_deg if want is None else want)
        for _ in range(6):
            changed = False
            for p in pts:
                if K.ik(p, pitch, roll, tool=self.tl, limits=lim)[1]["ok"]:
                    continue
                new, _ = K.best_pitch(p, want=pitch, wrist_roll=roll, tool=self.tl, limits=lim, lo=15.0)
                if new is None:
                    raise SafetyError(f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) is out of reach for the tip of tool "
                                      f"'{self.tool.name}'")
                pitch, changed = float(new), True
            if not changed:
                return pitch
        raise SafetyError("no single gripper pitch reaches every point of this move: "
                          + ", ".join(f"({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})" for p in pts[:4]))

    # ------------------------------------------------------------------ power
    def power(self):
        """Torque on (the arm holds where it is) and the speed tier's servo profile. When the torque had to be
        switched on, and once per command anyway, a tool on the moving jaw gets its squeeze back
        (tooling.hold_jaw, where this harness has it): after a torque-off the gripper's goal is where the jaw
        stands, and the brush has play (2026-11-13: sideways strokes skipped until the jaw was held shut).

        BEFORE the torque comes on (and once per command anyway) the jaw is compared with the tool file, where
        tooling can (tooling.jaw_in_place: reads only): a jaw that stands elsewhere than when the tip was
        measured is a refusal (JawRefusal, a SafetyError) with the arm exactly as it was found. A jaw that is shut although
        the tool file records no hold cannot be held by anything here: `jaw_note` then says so, and the
        command passes it on."""
        fresh = not self.arm.is_torqued()
        if fresh or not self._jaw_asked:
            self.check_jaw()            # (JawRefusal: nothing was switched on or written)
        if fresh:
            self.arm.torque_on()
            self.arm._sleep(0.3)
        if self.sp is None:
            self.sp = self.arm.apply_profile(None)
        if fresh or not self._jaw_asked:
            hold = getattr(TL, "hold_jaw", None)
            if hold is not None:
                try:
                    self.jaw = hold(self.arm, self.tool)
                except SafetyError as e:
                    # A jaw that cannot be held is a refusal of the command (a brush with play does not paint).
                    # The arm is left as it was found where that is safe: limp again when it rests (torque_off
                    # refuses anywhere else, and the arm then holds).
                    if fresh:
                        try:
                            self.arm.torque_off()
                        except MOTION_ERRORS:
                            pass
                    raise JawRefusal(str(e)) from e
            if isinstance(self.jaw, dict) and self.jaw.get("held"):
                self.jaw_note = None
            else:
                try:
                    touch = self.arm._c("gripper_touch_raw")
                except (KeyError, AttributeError):
                    touch = None
                self.jaw_note = jaw_note(self.tool, touch)
            self._jaw_asked = True
        self.moved = True

    def check_jaw(self):
        """Reads only: does the jaw stand where the tool file has it (tooling.jaw_in_place, where this harness has
        it)? JawRefusal when not. Asked by power() before the torque comes on, and by a command that plans
        without moving (`fd paint run` with --dry: it used to answer "planned" for a layer the run then refused)."""
        in_place = getattr(TL, "jaw_in_place", None)
        if in_place is not None:
            try:
                self.jaw_place = in_place(self.arm, self.tool)
            except SafetyError as e:
                raise JawRefusal(str(e)) from e
        return self.jaw_place

    # ------------------------------------------------------------------ the two kinds of move
    def guard(self, z: float, what: str):
        if not math.isfinite(z) or z < self.z_min - 1e-9:
            raise SafetyError(f"{what}: height {z * 1000:.1f} mm is below the limit of brush work "
                              f"({self.z_min * 1000:.1f} mm = {MAX_BELOW_TABLE_M * 1000:.0f} mm under the table); nothing moved")

    def line(self, pts, v_mm_s: float, pitch: float | None = None, settle=False, step: float = 0.003,
             what: str = "move") -> dict:
        """The tool tip along straight pieces through pts [(x, y, z), ...] (absolute, base frame) at one pitch
        (default: the pitch the gripper has now). Contact is allowed exactly when a point lies below the
        fingertip floor (the table), down to that point and never below z_min."""
        pts = [np.asarray(p, dtype=float) for p in pts]
        z_low = min(float(p[2]) for p in pts)
        self.guard(z_low, what)                 # before the torque comes on: a refusal leaves everything as it was
        self.power()
        q = self.arm.read_q()
        pitch = self.pitch_now(q) if pitch is None else float(pitch)
        p0 = self.tip(q)
        wps = self.arm.plan_tip_path([(p, pitch, self.roll) for p in pts], q_start=q, step_m=step)
        length = float(sum(np.linalg.norm(b - a) for a, b in zip([p0, *pts[:-1]], pts)))
        # (not faster than 25 deg/s at any joint: near the robot's base a few mm of the tip are many degrees of
        # the wrist, and a wrist that arrives 3 deg late ends the move as "target not reached")
        speed = min(TR.joint_speed_for(wps, length, float(v_mm_s) / 1000.0), 25.0)
        kw = {"settle": settle}
        if min(z_low, float(p0[2])) < self.floor + 0.0005:
            kw.update(allow_contact=True, contact_depth=max(0.0, self.floor - min(z_low, float(p0[2]))) + 0.0005)
        if self.o.ignore_keepouts:
            kw["ignore_keepouts"] = tuple(self.o.ignore_keepouts)
        r = self.arm.run_path(wps, speed, **kw)
        self.lowest_z = z_low if self.lowest_z is None else min(self.lowest_z, z_low)
        self.moves += 1
        return {"length_m": length, "duration_s": r.get("duration_s"), "pitch": pitch}

    def raise_to(self, h: float, v_mm_s: float | None = None, at_least: float = 0.03) -> bool:
        """Up to h above the table when the tip is lower. True when it moved. The ways, the first that the arm
        can do: straight up at the pitch it has; up in steps of 12 mm, the gripper turning to the nearest pitch
        it can hold at each step (near the robot's base the wrist is at its limit and the brush must lean back as
        it rises); up and away from the robot, 2 to 8 cm; the shoulder joint alone (what got the brush out on
        2026-11-13, when a stopped free move had left it 23 mm over the palette); up and away again, only to
        `at_least`; tooling.lift (whatever that harness-wide lift can still do from there).
        A way that ends lower than h but at least `at_least` above the table is accepted (a free move starts
        from there); SafetyError when the brush cannot even be brought that high."""
        q = self.arm.read_q()
        p = self.tip(q)
        if p[2] - self.tz >= h - 0.002:
            return False
        v = self.o.v_up if v_mm_s is None else v_mm_s
        try:
            self.line([np.array([p[0], p[1], self.tz + h])], v, what="raise")
            return True
        except SafetyError:
            pass
        moved = False
        try:
            while self.height() < h - 0.002:
                q = self.arm.read_q()
                p = self.tip(q)
                nxt = np.array([p[0], p[1], min(self.tz + h, p[2] + 0.012)])
                self.line([nxt], v, pitch=self.pitch_at([nxt], want=self.pitch_now(q)), what="raise (step)")
                moved = True
            return True
        except SafetyError:
            pass
        # up and away from the robot (close to the base the tip cannot rise where it stands: the wrist is at its limit)
        q = self.arm.read_q()
        p = self.tip(q)
        out = np.array([p[0] - K._X_PAN, p[1]])
        out = out / max(float(np.linalg.norm(out)), 1e-9)
        for d in (0.02, 0.04, 0.06, 0.08):
            top = np.array([p[0] + d * out[0], p[1] + d * out[1], self.tz + h])
            try:
                self.line([top], v, pitch=self.pitch_at([top], want=self.pitch_now(q)), what="raise (outwards)")
                return True
            except SafetyError:
                continue
        self.power()
        q = self.arm.read_q()
        for d in range(2, 31, 2):
            q_up = q.copy()
            q_up[1] -= math.radians(d)
            if self.height(q_up) >= h:
                try:
                    TL.check_pose(self.arm, self.tool, q_up, q_start=q, what="raise by the shoulder")
                    self.arm.run_path([q, q_up], 12.0, settle=0.6)
                    self.moves += 1
                    return True
                except SafetyError:
                    break
        if self.height() < at_least - 0.002:
            # up and away again, only as high as a free move needs to start from: a few centimetres from the pan
            # axis the arm cannot bring the tip to the travel height at all
            q = self.arm.read_q()
            p = self.tip(q)
            for d in (0.02, 0.04, 0.06, 0.08):
                top = np.array([p[0] + d * out[0], p[1] + d * out[1], self.tz + at_least + 0.005])
                try:
                    self.line([top], v, pitch=self.pitch_at([top], want=self.pitch_now(q)), what="raise (outwards, low)")
                    return True
                except SafetyError:
                    continue
        if self.height() < at_least - 0.002:
            TL.lift(self.arm, to_height=at_least + 0.005)      # (raises SafetyError / MotionAborted itself)
            moved = True
        if self.height() < at_least - 0.004:
            raise SafetyError(f"the brush is {self.height() * 1000:.0f} mm above the table and no way up to "
                              f"{at_least * 1000:.0f} mm was found; the arm holds where it is")
        return moved

    def _free(self, q_to, min_height, settle=0.5) -> dict:
        """One free move (tooling.free_move: joint space, checked every 1.5 deg, the measured tip height watched)
        with the second try of 2026-11-13: when the height monitor stops it (MotionAborted with the measured tip
        height in its info: the arm holds where it stopped, the brush is up, nothing was touched), the brush is
        raised to the travel height and the same move is tried ONCE more at half the speed. `min_height`: a
        number, or a function of nothing that gives it for the pose the arm stands in when the try starts. A
        second stop, and every other abort, ends the command as before."""
        speed = min(float(self.sp["transit"]), self.o.hop_deg_s)
        for attempt in (0, 1):
            limit = float(min_height() if callable(min_height) else min_height)
            try:
                return TL.free_move(self.arm, q_to, speed, settle=settle, min_height=limit)
            except MotionAborted as e:
                info = getattr(e, "info", None) or {}
                if attempt or info.get("reason") != "monitor" or "tool_tip_height_m" not in info:
                    raise
                self.hop_retries += 1
                self.log(f"free move stopped by the height monitor (tip {float(info['tool_tip_height_m']) * 1000:.0f} mm "
                         f"above the table, limit {limit * 1000:.0f} mm): raising to the travel height, once more at "
                         f"{speed / 2.0:.0f} deg/s")
                self.raise_to(self.o.travel_h)
                speed *= 0.5
        raise AssertionError("unreachable")       # pragma: no cover

    def hop(self, xy, h: float | None = None, pitch: float | None = None, also=()) -> float:
        """Far move: raise the brush to the travel height, then a free move (joint space, checked every 1.5 deg,
        the tip never lower than 1 cm under the lower of the two heights) to above xy at height h (default: the
        travel height). The pitch is one the arm can hold there and at the points `also` (the rest of the
        operation that follows). A move stopped by the height monitor is tried once more, slower (_free).
        Returns the pitch."""
        h = self.o.travel_h if h is None else float(h)
        if h < 0.025:
            raise SafetyError(f"a far move ends at least 25 mm above the table (asked: {h * 1000:.0f} mm)")
        target = np.array([float(xy[0]), float(xy[1]), self.tz + h])
        pitch = self.pitch_at([target, *also]) if pitch is None else float(pitch)
        q_to = self.arm.solve(target, pitch, self.roll)
        TL.check_pose(self.arm, self.tool, q_to, min_height=h - 0.004, what="hop target")
        self.power()
        self.raise_to(min(self.o.travel_h, max(h, MIN_TRAVEL_H)))
        self._free(q_to, lambda: max(TL.MIN_FREE_HEIGHT, min(self.height(), h) - self.o.sag_m))
        self.moves += 1
        return pitch

    # ------------------------------------------------------------------ operations
    def dab(self, xy, z: float, pattern=(), v_mm_s: float | None = None, dwell_s: float = 0.0, lift_h: float | None = None,
            what: str = "dab") -> dict:
        """Into a well or a dish and out again: above xy at the travel height, straight down to z (absolute),
        the pattern [(dx, dy, dz), ...] relative to (xy, z) one point after the other, back to (xy, z), straight
        up. Returns {"pitch", "z", "length_m"}."""
        x, y = float(xy[0]), float(xy[1])
        self.guard(z, what)
        base = np.array([x, y, float(z)])
        pts = [base + np.asarray(d, dtype=float) for d in pattern]
        for p in pts:
            self.guard(float(p[2]), what)
        top = np.array([x, y, self.tz + (self.o.travel_h if lift_h is None else float(lift_h))])
        pitch = self.hop((x, y), also=[base, *pts])
        self.line([base], self.o.v_down, pitch=pitch, what=f"{what}: down")
        length = 0.0
        if pts:
            r = self.line(pts + [base], self.o.v_stir if v_mm_s is None else v_mm_s, pitch=pitch, step=0.002, what=what)
            length = r["length_m"]
        if dwell_s > 0:
            self.arm._sleep(dwell_s)
        self.line([top], self.o.v_up, pitch=pitch, settle=0.25, what=f"{what}: up")
        return {"pitch": round(pitch, 1), "z": round(float(z), 4), "length_m": round(length, 4)}

    def stroke(self, pts_xy, z_contact, press_m: float, v_mm_s: float | None = None, pitch: float | None = None,
               lead_in: bool = True) -> dict:
        """One brush stroke on the paper: above its start, down to (contact height - press), along the points,
        up. pts_xy: [(x, y), ...]; z_contact: the absolute height at which the bristles first touch, a number or
        a function of (x, y); press_m: how far below that the modelled tip runs (0 .. MAX_PRESS_M).
        A start that is near and at the hover height is reached by a glide, a far one by a hop.
        Returns {"pitch", "length_m", "z_low", "seconds" (the whole stroke, by the arm's clock), "travel_s" (the
        part on the paper)}."""
        if not (math.isfinite(float(press_m)) and 0.0 <= float(press_m) <= MAX_PRESS_M + 1e-9):
            raise SafetyError(f"brush press {float(press_m) * 1000:.1f} mm is outside 0 .. {MAX_PRESS_M * 1000:.0f} mm; "
                              "nothing moved")
        zc = z_contact if callable(z_contact) else (lambda x, y, _z=float(z_contact): _z)
        pts = [np.array([float(x), float(y), float(zc(x, y)) - float(press_m)]) for x, y in pts_xy]
        if len(pts) < 2:
            raise SafetyError("a stroke needs at least two points")
        for p in pts:
            self.guard(float(p[2]), "stroke")
        z_top = max(float(zc(x, y)) for x, y in pts_xy)
        hover = np.array([pts[0][0], pts[0][1], z_top + self.o.hover_m])
        end_up = np.array([pts[-1][0], pts[-1][1], z_top + self.o.hover_m])
        t0 = self.clock()
        q = self.arm.read_q()
        p_now = self.tip(q)
        near = (float(np.hypot(p_now[0] - hover[0], p_now[1] - hover[1])) <= self.o.near_m
                and hover[2] - 0.002 <= p_now[2] <= self.tz + self.o.above_h + 0.006)
        if pitch is None:
            pitch = self.pitch_at([hover, *pts, end_up, np.array([hover[0], hover[1], self.tz + self.o.above_h])])
        if near and abs(self.pitch_now(q) - pitch) < 1.5:
            self.line([hover], self.o.v_glide, pitch=pitch, settle=0.2, what="glide")
        else:
            self.hop((hover[0], hover[1]), h=self.o.above_h, pitch=pitch)
            self.line([hover], self.o.v_down * 2.0, pitch=pitch, settle=0.2, what="to hover")
        self.line([pts[0]], self.o.v_down, pitch=pitch, what="stroke: down")
        t1 = self.clock()
        r = self.line(pts[1:], self.o.v_stroke if v_mm_s is None else v_mm_s, pitch=pitch, step=0.002, what="stroke")
        t2 = self.clock()
        self.line([end_up], self.o.v_up, pitch=pitch, settle=0.2, what="stroke: up")
        self.path_m += r["length_m"]
        return {"pitch": round(pitch, 1), "length_m": round(r["length_m"], 4), "z_low": round(min(float(p[2]) for p in pts), 4),
                "seconds": round(float(self.clock() - t0), 2), "travel_s": round(float(t2 - t1), 2)}

    # ------------------------------------------------------------------ the way out
    def goto_q(self, q_deg, settle=0.4):
        """Free move to a joint pose given in degrees (the photo pose), after raising the brush."""
        q_to = np.radians(np.asarray(q_deg, dtype=float))
        TL.check_pose(self.arm, self.tool, q_to, min_height=MIN_TRAVEL_H, what="pose")
        self.power()
        self.raise_to(self.o.travel_h)
        if float(np.max(np.abs(self.arm.read_q() - q_to))) > math.radians(0.7):
            self._free(q_to, TL.MIN_FREE_HEIGHT, settle=settle)
        self.arm._sleep(0.3)

    def where(self, rep: dict | None = None, at_pose: bool = False) -> dict:
        """Where the brush is, for a result: tip_xyz, height_mm (above the table), lifted (at least 2 cm up),
        torque, and "where": "not moved" | "photo pose" | "lifted" | "stopped" (the lift failed: see
        lift_error) | "limp" (torque off). Reads the encoders; moves nothing."""
        rep = dict(rep or {})
        p = self.tip()
        h = float(p[2]) - self.tz
        torque = bool(self.arm.is_torqued())
        if not self.moved:
            where = "not moved"
        elif not torque:
            where = "limp"
        elif "lift_error" in rep:
            where = "stopped"
        elif "retreat_error" in rep or not at_pose:
            where = "lifted"
        else:
            where = "photo pose"
        rep.update(tip_xyz=[round(float(v), 4) for v in p], height_mm=round(h * 1000.0, 1),
                   lifted=bool(h >= TL.MIN_FREE_HEIGHT), where=where, torque="on" if torque else "off")
        if self.lowest_z is not None:
            rep["lowest_plan_z_mm"] = round(self.lowest_z * 1000.0, 2)
        if self.hop_retries:
            rep["hop_retries"] = self.hop_retries
        return rep

    def lost(self, error) -> dict:
        """The report when the servos do not answer any more (a cable, the supply): nothing can be read, so
        "where" is "unknown" and "torque" too; "not moved" when this command had not sent the arm anywhere."""
        if not self.moved:
            return {"where": "not moved", "torque": "unknown"}
        rep = {"where": "unknown", "torque": "unknown", "lift_error": f"{type(error).__name__}: {error}"}
        if self.lowest_z is not None:
            rep["lowest_plan_z_mm"] = round(self.lowest_z * 1000.0, 2)
        if self.hop_retries:
            rep["hop_retries"] = self.hop_retries
        return rep

    def retreat(self, q_deg=None) -> dict:
        """Brush up, then to the given pose (or stay above where it is). Never raises for a motion that is
        refused or stopped: the report says what did not work and where the brush is (see where()). (An error
        of the servo bus itself is not a motion error: it is raised, and the caller reports lost().)"""
        rep = {}
        if self.moved and self.arm.is_torqued():
            try:
                self.raise_to(self.o.travel_h)
            except MOTION_ERRORS as e:
                rep["lift_error"] = f"{type(e).__name__}: {e}"
            if q_deg is not None and "lift_error" not in rep:
                try:
                    self.goto_q(q_deg)
                except MOTION_ERRORS as e:
                    rep["retreat_error"] = f"{type(e).__name__}: {e}"
        return self.where(rep, at_pose=q_deg is not None)


def where_ja(rep: dict) -> str:
    """One Japanese sentence for the operator from a retreat() / where() report."""
    tail = WHERE_JA.get(rep.get("where"), "")
    if rep.get("where") == "not moved" or "height_mm" not in rep:      # (no height: the servos did not answer)
        return tail
    return f"筆先は机の {float(rep['height_mm']):.0f} mm 上、" + tail
