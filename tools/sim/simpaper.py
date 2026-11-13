"""The simulated sheet of paper, pen and camera: what lets the pen pipeline run without hardware.

On 2026-11-11 the real arm drew a cat, and nearly everything that went wrong was something the
harness could not see in its own model: the pen tip was not where the model put it (up to 16.6 mm,
differently left and right), the paper was met at another height than planned and that height
changed across the sheet, the height wandered by millimetres while the arm crept, a pushed pen
skipped, and the locked "desk" exposure clipped the paper so that thin ink was invisible. SimPaper
holds exactly these as HIDDEN TRUTHS and shows the harness only what the real world showed it: the
joint encoders during a motion and a photo afterwards.

    paper = SimPaper(arm, cam, tip_error=(0, -0.015, 0.008), contact={...}, pos_error=shear(...),
                     wander_mm=1.5, seed=1)        # attaches itself to arm.run_path; or **night_truths()
    ... any code that moves the arm ...
    photo = paper.render("paper")                  # 1920x1080 BGR, as camera.grab would return

What is modelled: where the real tip is for a measured joint pose (true tool offset, an extra
position error that depends on the place, a smooth seeded height wander); ink only while that tip is
at or below the true contact surface, darker and wider the deeper it presses; dashes instead of a
line while the pen is pushed along its lean; ink that was on the sheet before; the two exposures.
What is NOT modelled: contact forces (the simulated pen sinks into the paper instead of bending the
arm: read max_depth_m), ink flow, the pen sliding in its tape, paper that moves, shadows, the arm's
real outline (a rough dark / pale silhouette at the last pose warns that a photo was taken with the
arm in the way). It is a test bench for logic, not evidence that a drawing will succeed.

Units: metres, base frame; photos BGR uint8. Tests: test_strokes.py.
"""

from __future__ import annotations

import inspect
import math

import cv2
import numpy as np

import camera
import kinematics as K

W, H = camera.W, camera.H
EXPOSURES = ("paper", "desk")
SKIPS = ("none", "tip-leading", "tip-trailing")   # which direction of travel loses ink (a hidden truth, SPEC 9.1)
# Bare paper with the "paper" exposure (110 / gain 0): 155..175 in the photos of 2026-11-11.
PAPER_BGR = (176.0, 171.0, 173.0)
# The desk (dark red wood) with each exposure, as measured in photos/225720 and overexposed/0637.
DESK_BGR = {"paper": (20.0, 18.0, 30.0), "desk": (125.0, 115.0, 160.0)}
# Light reaching the sensor relative to the paper exposure. With the desk exposure the paper is far
# beyond the clip level (255, 253, 255 in the real photo): whatever a thin line takes away, the
# pixel stays white. Blur happens BEFORE the clip, as in the lens, which is why thin ink vanishes.
EXPOSURE_GAIN = {"paper": 1.0, "desk": 6.9}
INK_REFLECTANCE = 0.012      # marker ink against the paper (line cores read 5..30 next to paper at 160)
GAMMA = 2.2
A4_M = (0.297, 0.210)
_SS = 3                      # supersampling of the ink raster (an 0.4 mm line is 1.4 px wide)
_SHIFT = 4                   # sub-pixel bits handed to cv2.fillConvexPoly
_DISC = np.stack([np.cos(np.arange(8) * math.pi / 4.0), np.sin(np.arange(8) * math.pi / 4.0)], axis=1)


def sheet_polygon(centre_xy=(0.225, 0.0), size_m=A4_M, rot_deg: float = 6.0) -> np.ndarray:
    """Corners of a sheet on the table: long side across the robot (along y), turned by rot_deg
    counter-clockwise seen from above. The default is about where the A4 sheet of 2026-11-11 lay
    (6 degrees askew, far edge 12 cm and near edge 33 cm in front of the base; read off photo 225720)."""
    th = math.radians(rot_deg)
    ex = np.array([math.cos(th), math.sin(th)])       # the sheet's short side (table x when not turned)
    ey = np.array([-math.sin(th), math.cos(th)])
    c = np.asarray(centre_xy, dtype=float)
    hx, hy = size_m[1] / 2.0, size_m[0] / 2.0
    return np.array([c - hx * ex - hy * ey, c - hx * ex + hy * ey, c + hx * ex + hy * ey, c + hx * ex - hy * ey])


def shear(centre_xy=(0.223, 0.003), kxy: float = -0.10, kyx: float = 0.0, offset=(0.0, 0.0)) -> dict:
    """A position error that depends on the place: error = offset + [[0, kxy], [kyx, 0]] (p - centre).
    kxy = -0.10 is the night's finding (d actual_x / d commanded_y = -0.104: ink landed 6 mm nearer
    the robot on the left of the box and 6 mm further on the right)."""
    return {"centre": [float(centre_xy[0]), float(centre_xy[1])], "matrix": [[0.0, float(kxy)], [float(kyx), 0.0]],
            "offset": [float(offset[0]), float(offset[1])]}


def night_truths(box_centre=(0.2575, 0.003), box_half_m: float = 0.0525, table_z: float = -0.0015) -> dict:
    """Keyword arguments for SimPaper that make a world like the evening of 2026-11-11 (the DRAW
    acceptance of SPEC section 7): the real pen tip 15 mm to the side of and 8 mm further out than
    the first model, a sheared position error of 10 %, a contact surface tilted by 1.5 mm across the
    drawing box, a height wander of 1.5 mm.

    The tool error is in gripper_frame_link axes: y is sideways (horizontal at wrist roll 0), z along
    the fingers; x lies in the arm's vertical plane. The real pen being LONGER than the model means
    that it touches the paper while the model still sees it 7.7 mm up (at gripper pitch 75): a first
    descent has to start high."""
    return {"tip_error": (0.0, -0.015, 0.008),
            "pos_error": shear(box_centre, kxy=-0.10),
            "contact": {"x0": float(box_centre[0]), "y0": float(box_centre[1]), "a": float(table_z), "bx": 0.0,
                        "by": 0.0015 / (2.0 * box_half_m)},
            "wander_mm": 1.5}


def _surface_fn(contact, table_z: float):
    """contact (None | height | (a, b, c) | {"x0", "y0", "a", "bx", "by"} | callable) -> z(x, y) that
    also takes arrays."""
    if contact is None:
        return lambda x, y: table_z + 0.0 * np.asarray(x, dtype=float)
    if callable(contact):
        def fn(x, y):
            if np.ndim(x) == 0:
                return float(contact(float(x), float(y)))
            return np.array([float(contact(float(a), float(b))) for a, b in zip(np.ravel(x), np.ravel(y))]).reshape(np.shape(x))
        return fn
    if isinstance(contact, dict):
        x0, y0 = float(contact.get("x0", 0.0)), float(contact.get("y0", 0.0))
        a, bx, by = float(contact["a"]), float(contact.get("bx", 0.0)), float(contact.get("by", 0.0))
        return lambda x, y: a + bx * (np.asarray(x, dtype=float) - x0) + by * (np.asarray(y, dtype=float) - y0)
    if np.ndim(contact) == 0:
        z0 = float(contact)
        return lambda x, y: z0 + 0.0 * np.asarray(x, dtype=float)
    a, b, c = (float(v) for v in contact)          # cfg["table_plane"]: z = a + b x + c y
    return lambda x, y: a + b * np.asarray(x, dtype=float) + c * np.asarray(y, dtype=float)


def _error_fn(pos_error):
    if pos_error is None:
        return None
    if callable(pos_error):
        return lambda x, y: tuple(float(v) for v in pos_error(x, y))
    c = np.asarray(pos_error.get("centre", (0.0, 0.0)), dtype=float)
    m = np.asarray(pos_error.get("matrix", np.zeros((2, 2))), dtype=float).reshape(2, 2)
    o = np.asarray(pos_error.get("offset", (0.0, 0.0)), dtype=float)

    def fn(x, y):
        d = o + m @ (np.array([x, y]) - c)
        return float(d[0]), float(d[1])

    return fn


class SimPaper:
    """A sheet on the table, a pen on the gripper and the fixed camera, with truths the harness
    is not told (see the module text).

    arm: the simulated robot.Arm whose motions are observed (attach=True wraps arm.run_path so that
        every motion feeds its measured joint poses; monitor() gives the callback for doing it by
        hand). None: feed() / feed_tip() only.
    cam: camera.CameraModel (None: loaded from camera_model.json / $FD_CAMERA_MODEL at the first photo).
    true_tip_offset: the REAL tool tip in gripper_frame_link axes (m); or tip_error: what to add to
        the offset the arm believes at this moment (arm.tip_tool()). Default: the model is right.
    contact: the real surface the pen meets, z(x, y): None = the table plane (cfg["table_z"]), a
        height, (a, b, c) for z = a + b x + c y, tooling's {"x0", "y0", "a", "bx", "by"}, or a callable.
    pos_error: extra error of the real tip position, (x, y) -> (dx, dy) or a dict as shear() makes.
    wander_mm: the real tip height wanders within +-wander_mm, smoothly in time (periods of 3..15 s,
        like the stick-slip creep that gave -3.9..+2.5 mm on the real arm); seeded.
    sheet: polygon of the paper on the table (default sheet_polygon(): the night's A4).
    pen_width_m: line width when pressed (0.001 = the marker, 0.0004 = the thin pen).
    pen_axis: unit vector of the pen in gripper_frame_link axes, pointing to the tip (default: along
        the true tip offset). Its horizontal part is the lean.
    skips: which direction of travel loses ink: "none" (default), "tip-leading" (dashes while the pen moves
        along its lean) or "tip-trailing" (dashes while it moves against it). A hidden truth like the others:
        nothing is known for certain about the real pen (SPEC 9.1). push_skips=True means "tip-leading".
    old_ink: polylines (table xy) that are on the sheet already, or dicts {"xy", "width_m", "darkness"}.
    noise / blur_px / shading: photo noise (grey levels, sd), lens blur (sigma, px), how much darker
        the far end of the sheet is. show_arm: draw the arm's rough silhouette at its last pose."""

    def __init__(self, arm=None, cam=None, *, true_tip_offset=None, tip_error=None, contact=None, pos_error=None,
                 wander_mm: float = 0.0, seed: int = 0, sheet=None, pen_width_m: float = 0.001, pen_axis=None,
                 push_skips: bool | None = None, old_ink=None, noise: float = 0.0, blur_px: float = 0.45,
                 shading: float = 0.08, show_arm: bool = True, attach: bool = True, skips: str | None = None):
        self.arm = arm
        self.cam = cam
        model = np.zeros(3) if arm is None else (np.linalg.inv(K._T_TOOL) @ arm.tip_tool())[:3, 3]
        if true_tip_offset is not None:
            self.true_tip_offset = np.asarray(true_tip_offset, dtype=float).reshape(3).copy()
        else:
            self.true_tip_offset = model + (np.zeros(3) if tip_error is None else np.asarray(tip_error, dtype=float))
        self.model_tip_offset = model.copy()
        self._tool_true = K.tool_offset(*self.true_tip_offset)
        if pen_axis is None:
            n = float(np.linalg.norm(self.true_tip_offset))
            pen_axis = self.true_tip_offset / n if n > 1e-6 else K._APPROACH_LOCAL
        self.pen_axis = np.asarray(pen_axis, dtype=float) / float(np.linalg.norm(pen_axis))
        self.table_z = float(arm.cfg.get("table_z", -0.0015)) if arm is not None else -0.0015
        self.contact = contact
        self._surface = _surface_fn(contact, self.table_z)
        self.pos_error = pos_error
        self._pos_error = _error_fn(pos_error)
        self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.wander_m = abs(float(wander_mm)) / 1000.0
        self._w_period = np.array([3.7, 6.1, 9.8, 15.3]) * rng.uniform(0.85, 1.15, 4)
        self._w_phase = rng.uniform(0.0, 2.0 * math.pi, 4)
        self._w_amp = np.array([1.0, 0.8, 0.6, 0.5])
        self._skip_rng = np.random.default_rng(self.seed + 7919)
        self._noise_rng = np.random.default_rng(self.seed + 104729)
        self.sheet = sheet_polygon() if sheet is None else np.asarray(sheet, dtype=float).reshape(-1, 2)
        self.pen_width_m = float(pen_width_m)
        self.full_depth_m = 0.001        # pressing depth at which the line has its full width and darkness
        # Which direction of travel loses ink is a HIDDEN TRUTH that a test sets (SPEC 9.1: the one real
        # drawing does not prove a rule either way), default none: skips = "none" | "tip-leading" (dashes
        # while the tip is pushed ahead of the tool, along its lean) | "tip-trailing" (dashes while it is
        # dragged behind). push_skips=True is the older spelling of "tip-leading".
        if skips is None:
            skips = "tip-leading" if push_skips else "none"
        if skips not in SKIPS:
            raise ValueError(f"skips {skips!r} is not one of {SKIPS}")
        self.skips = skips
        self.push_skips = skips == "tip-leading"
        self.push_cos = 0.5              # "along the lean" = moving within 60 degrees of it (a guess)
        self.min_lean = math.sin(math.radians(4.0))   # a pen more upright than this has no push side
        self.skip_on_m, self.skip_off_m = (0.0012, 0.0035), (0.0012, 0.0035)   # dash and gap lengths while pushed
        self.ink_step_m = 0.00025        # spacing of the stored ink points
        # The encoders step by 0.088 deg (0.4 mm at the tip) but the real joints do not jump: the real
        # pose is taken as the mean of the last smooth_n readings (100 ms), and the direction of travel
        # is read over heading_base_m, not from one control cycle to the next.
        self.smooth_n = 5
        self.heading_base_m = 0.001
        self._q_hist = []
        self._trail = []
        self.noise, self.blur_px, self.shading, self.show_arm = float(noise), float(blur_px), float(shading), bool(show_arm)
        self._lines = []                 # [{"old": bool, "pts": [[x, y, width, darkness, depth, t], ...]}]
        self._cur = None                 # the line the pen is drawing now
        self._pending = None             # latest sample of that line that is not stored yet
        self._contact_prev = None        # previous position at which the pen was on the paper
        self._skip = None                # [inked, metres left] while the pen is pushed
        self._time = 0.0
        self._version = 0
        self._cache = {}
        self._last_q = None
        self.max_depth_m = 0.0           # deepest the true tip has been below the contact surface
        self.samples = 0                 # poses fed so far
        self._orig_run_path = None
        for item in (old_ink or []):
            if isinstance(item, dict):
                self.add_ink(item["xy"], item.get("width_m"), item.get("darkness", 1.0))
            else:
                self.add_ink(item)
        if arm is not None and attach:
            self.attach()

    # ------------------------------------------------------------ hidden truths (tests may read them)
    def surface_z(self, x, y):
        """Height of the real contact surface at (x, y)."""
        z = self._surface(x, y)
        return float(z) if np.ndim(z) == 0 else z

    def wander(self, t: float) -> float:
        """Height error of the real tip at time t (m): smooth, bounded by wander_mm, set by the seed."""
        if self.wander_m <= 0.0:
            return 0.0
        return float(self.wander_m * np.sum(self._w_amp * np.sin(2.0 * math.pi * t / self._w_period + self._w_phase))
                     / self._w_amp.sum())

    def _true(self, q, t: float):
        tf = K.fk(np.asarray(q, dtype=float), self._tool_true)
        p = tf[:3, 3].copy()
        if self._pos_error is not None:
            dx, dy = self._pos_error(float(p[0]), float(p[1]))
            p[0] += dx
            p[1] += dy
        p[2] += self.wander(t)
        return p, (tf[:3, :3] @ self.pen_axis)[:2]

    def true_tip(self, q, t: float | None = None) -> np.ndarray:
        """Where the real tool tip is for joint pose q (at time t; default: now)."""
        return self._true(q, self._time if t is None else float(t))[0]

    def lean_xy(self, q) -> np.ndarray:
        """Horizontal part of the pen's direction (towards its tip) at pose q; its length is the sine
        of the lean angle."""
        return self._true(q, self._time)[1]

    def truth(self) -> dict:
        """The hidden truths, for test reports."""
        return {"true_tip_offset": self.true_tip_offset.tolist(), "model_tip_offset": self.model_tip_offset.tolist(),
                "contact": self.contact if not callable(self.contact) else "callable", "table_z": self.table_z,
                "pos_error": self.pos_error if not callable(self.pos_error) else "callable",
                "wander_mm": self.wander_m * 1000.0, "seed": self.seed, "pen_width_mm": self.pen_width_m * 1000.0,
                "skips": self.skips}

    # ------------------------------------------------------------ watching the arm
    def monitor(self, inner=None):
        """A callback for Arm.run_path(monitor=...): monitor(t, q_plan, q_meas) feeds the MEASURED
        pose of every control cycle to the paper and returns what `inner` (another monitor, or None)
        returns, so it never stops a motion by itself."""
        state = {"last": None, "base": self._time}

        def cb(t, q_plan, q_meas):
            if state["last"] is not None and t < state["last"]:      # the same callback on a new motion
                state["base"] = self._time
            state["last"] = t
            self.feed(q_meas, state["base"] + float(t))
            return bool(inner(t, q_plan, q_meas)) if inner is not None else False

        return cb

    def attach(self, arm=None):
        """Watch every Arm.run_path of this arm from now on: the arm's method is wrapped on the
        instance so that the caller's own monitor (tracking.record, a skill) still runs. Motions
        that do not go through run_path (probe_down, bench_motion) are not seen."""
        arm = self.arm if arm is None else arm
        if arm is None:
            raise ValueError("no arm to attach to")
        if getattr(arm, "_simpaper", None) is self:
            return self
        if getattr(arm, "_simpaper", None) is not None:
            arm._simpaper.detach()
        orig = arm.run_path
        sig = inspect.signature(orig)
        paper = self

        def run_path(*args, **kw):
            ba = sig.bind(*args, **kw)
            ba.arguments["monitor"] = paper.monitor(ba.arguments.get("monitor"))
            return orig(*ba.args, **ba.kwargs)

        run_path.__doc__ = orig.__doc__
        arm.run_path = run_path
        arm._simpaper = self
        self.arm, self._orig_run_path = arm, orig
        return self

    def detach(self):
        """Stop watching the arm (its run_path is the class's again)."""
        arm = self.arm
        if arm is not None and getattr(arm, "_simpaper", None) is self:
            arm.__dict__.pop("run_path", None)
            arm.__dict__.pop("_simpaper", None)
        self._orig_run_path = None
        return self

    def feed(self, q_meas, t: float | None = None) -> bool:
        """One measured joint pose (rad, 5) at time t (seconds; None: 20 ms after the last). Returns
        True when the pen laid ink there."""
        t = self._tick(t)
        q = np.asarray(q_meas, dtype=float).reshape(5)
        if self._q_hist and float(np.max(np.abs(q - self._q_hist[-1]))) > math.radians(3.0):
            self._q_hist = []                             # no arm moves 3 deg in one cycle: a new start pose
        self._q_hist = (self._q_hist + [q.copy()])[-max(1, int(self.smooth_n)):]
        self._last_q = np.mean(self._q_hist, axis=0)
        p, lean = self._true(self._last_q, t)
        return self.feed_tip(p, t, lean)

    def feed_tip(self, xyz, t: float | None = None, lean=None) -> bool:
        """One REAL tip position (what feed() derives from a pose): ink is laid when it is at or
        below the contact surface. lean: horizontal direction the pen tip points to (None or shorter
        than sin(4 deg): no push side)."""
        t = self._tick(t)
        self.samples += 1
        x, y, z = (float(v) for v in np.asarray(xyz, dtype=float).reshape(3))
        depth = self.surface_z(x, y) - z
        if depth < 0.0:                                   # in the air: the line (if any) ends
            self._close()
            self._contact_prev, self._skip, self._trail = None, None, []
            return False
        self.max_depth_m = max(self.max_depth_m, depth)
        p = np.array([x, y])
        prev, self._contact_prev = self._contact_prev, p
        inked = True if self._skip is None else bool(self._skip[0])
        self._trail = (self._trail + [p])[-400:]
        if prev is not None and self.skips != "none":
            dist = float(np.hypot(p[0] - prev[0], p[1] - prev[1]))
            head, i = p - self._trail[0], 0
            for i in range(len(self._trail) - 2, -1, -1):           # the newest point a baseline away
                head = p - self._trail[i]
                if float(np.hypot(head[0], head[1])) >= self.heading_base_m:
                    break
            self._trail = self._trail[i:]
            hn = float(np.hypot(head[0], head[1]))
            if dist > 1e-7 and hn > 2e-4:                           # else: too little travel to tell a direction
                ln = 0.0 if lean is None else float(np.hypot(lean[0], lean[1]))
                sgn = 1.0 if self.skips == "tip-leading" else -1.0
                if ln > self.min_lean and sgn * float(head[0] * lean[0] + head[1] * lean[1]) / (hn * ln) > self.push_cos:
                    inked = self._skip_step(dist)
                else:
                    self._skip, inked = None, True
        if not inked:
            self._close()
            return False
        self._add(x, y, depth, t)
        return True

    def pen_up(self):
        """End the current line (a motion ended while the pen was down and the next one is elsewhere)."""
        self._close()
        self._contact_prev, self._skip, self._trail = None, None, []

    def _tick(self, t) -> float:
        self._time = self._time + 0.02 if t is None else float(t)
        return self._time

    def _skip_step(self, dist: float) -> bool:
        """The pen travels the way this world's hidden truth (`skips`) makes it lose ink: it jumps and
        lands, leaving dashes of a few mm (as seen on 2026-11-11; which way of travel did it there is
        not settled). The state is [inked, metres left of this dash or gap]; such a run starts inked."""
        if self._skip is None:
            self._skip = [True, float(self._skip_rng.uniform(*self.skip_on_m))]
        self._skip[1] -= dist
        if self._skip[1] <= 0.0:
            on = not self._skip[0]
            self._skip = [on, float(self._skip_rng.uniform(*(self.skip_on_m if on else self.skip_off_m)))]
        return bool(self._skip[0])

    def _look(self, depth: float):
        """(width, darkness) of the line for a pressing depth: a touch gives a thin, pale line."""
        k = min(1.0, max(0.0, depth) / self.full_depth_m)
        squash = 1.0 + 0.15 * min(1.0, max(0.0, depth - self.full_depth_m) / 0.002)
        return self.pen_width_m * (0.55 + 0.45 * k) * squash, 0.45 + 0.55 * k

    def _add(self, x, y, depth, t):
        w, d = self._look(depth)
        pt = [x, y, w, d, depth, t]
        self._version += 1
        if self._cur is None:
            self._cur = {"old": False, "pts": [pt]}
            self._lines.append(self._cur)
            self._pending = None
            return
        last = self._cur["pts"][-1]
        if math.hypot(x - last[0], y - last[1]) >= self.ink_step_m:
            self._cur["pts"].append(pt)
            self._pending = None
        else:
            if depth > last[4]:                           # pressing harder on the spot darkens it
                last[2], last[3], last[4] = w, d, depth
            self._pending = pt

    def _close(self):
        tail = self._tail()
        if tail is not None:
            self._cur["pts"].append(tail)
            self._version += 1
        self._cur, self._pending = None, None

    def _tail(self):
        """The newest sample of the line being drawn, when it is not stored yet and is another place
        than the last stored point."""
        if self._cur is None or self._pending is None:
            return None
        last = self._cur["pts"][-1]
        return self._pending if math.hypot(self._pending[0] - last[0], self._pending[1] - last[1]) > 1e-6 else None

    # ------------------------------------------------------------ ink on the sheet
    def add_ink(self, polyline_xy, width_m: float | None = None, darkness: float = 1.0, old: bool = True):
        """Put a line on the sheet without the arm (ink from before, or a test's own marks)."""
        p = np.asarray(polyline_xy, dtype=float).reshape(-1, 2)
        w = self.pen_width_m if width_m is None else float(width_m)
        self._lines.append({"old": bool(old), "pts": [[float(x), float(y), w, float(darkness), self.full_depth_m, self._time]
                                                      for x, y in p]})
        self._version += 1
        return self

    def ink(self, new_only: bool = False) -> list:
        """Where ink really is: [{"xy": (N, 2), "width_m", "darkness", "depth_m", "t" (each (N,)),
        "old": bool}, ...], one entry per unbroken line."""
        out = []
        for ln in self._lines:
            if new_only and ln["old"]:
                continue
            tail = self._tail() if ln is self._cur else None
            a = np.array(ln["pts"] + ([tail] if tail is not None else []), dtype=float)
            out.append({"xy": a[:, :2].copy(), "width_m": a[:, 2].copy(), "darkness": a[:, 3].copy(),
                        "depth_m": a[:, 4].copy(), "t": a[:, 5].copy(), "old": ln["old"]})
        return out

    def ink_length(self, new_only: bool = True) -> float:
        """Metres of line on the sheet (new_only: without the ink that was there before)."""
        return float(sum(np.linalg.norm(np.diff(ln["xy"], axis=0), axis=1).sum() for ln in self.ink(new_only)))

    def clear(self, old_too: bool = False):
        """A fresh sheet (old_too) or the same sheet without what the arm drew."""
        self._close()
        self._lines = [] if old_too else [ln for ln in self._lines if ln["old"]]
        self._contact_prev, self._skip, self._trail = None, None, []
        self.max_depth_m = 0.0
        self._version += 1
        return self

    def coverage(self, polylines_xy, tol_m: float = 0.004, step_m: float = 0.001, new_only: bool = True) -> dict:
        """The truth a detector is compared with: how much of the wanted lines has real ink within
        tol_m. {"drawn_fraction", "offset_mm": {"mean", "p95"} (distance from the wanted line to the
        nearest ink, over the drawn part), "samples"}."""
        want = []
        for pl in polylines_xy:
            p = np.asarray(pl, dtype=float).reshape(-1, 2)
            cum = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
            s = np.linspace(0.0, cum[-1], max(2, int(math.ceil(cum[-1] / step_m)) + 1))
            want.append(np.stack([np.interp(s, cum, p[:, 0]), np.interp(s, cum, p[:, 1])], axis=1))
        want = np.concatenate(want) if want else np.zeros((0, 2))
        ink = [ln["xy"] for ln in self.ink(new_only)]
        if not len(want) or not ink:
            return {"drawn_fraction": 0.0, "offset_mm": {"mean": None, "p95": None}, "samples": int(len(want))}
        pts = np.concatenate(ink)
        d = np.concatenate([np.sqrt(((want[i:i + 256, None, :] - pts[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
                            for i in range(0, len(want), 256)])
        ok = d <= tol_m
        off = d[ok] * 1000.0
        return {"drawn_fraction": float(ok.mean()),
                "offset_mm": {"mean": float(off.mean()) if len(off) else None,
                              "p95": float(np.percentile(off, 95)) if len(off) else None},
                "samples": int(len(want))}

    def state(self) -> dict:
        """The ink as plain data (json), so that separate `fd --sim` runs can share one sheet."""
        self._close()
        return {"time": self._time, "max_depth_m": self.max_depth_m,
                "lines": [{"old": ln["old"], "pts": [[float(v) for v in pt] for pt in ln["pts"]]} for ln in self._lines]}

    def load_state(self, state: dict):
        self._close()
        self._lines = [{"old": bool(ln["old"]), "pts": [list(pt) for pt in ln["pts"]]} for ln in state.get("lines", [])]
        self._time = float(state.get("time", 0.0))
        self.max_depth_m = float(state.get("max_depth_m", 0.0))
        self._contact_prev, self._skip, self._trail = None, None, []
        self._version += 1
        return self

    # ------------------------------------------------------------ the camera
    def _camera(self):
        if self.cam is None:
            self.cam = camera.CameraModel.load()
        return self.cam

    def _project(self, xy: np.ndarray) -> np.ndarray:
        """Pixels of points on the real surface."""
        xy = np.asarray(xy, dtype=float).reshape(-1, 2)
        z = np.asarray(self._surface(xy[:, 0], xy[:, 1]), dtype=float).reshape(-1)
        return self._camera().project(np.column_stack([xy, z]))

    def _sheet_mask(self) -> np.ndarray:
        key = ("sheet", id(self._camera()), self.sheet.tobytes())
        if self._cache.get("sheet_key") != key:
            m = np.zeros((H, W), np.uint8)
            uv = self._project(self.sheet)
            cv2.fillPoly(m, [np.round(uv * (1 << _SHIFT)).astype(np.int32)], 255, lineType=cv2.LINE_AA, shift=_SHIFT)
            self._cache["sheet_key"], self._cache["sheet"] = key, m.astype(np.float32) / 255.0
        return self._cache["sheet"]

    def _ink_cover(self) -> np.ndarray:
        """(H, W) float32: share of each pixel covered by ink, times the ink's darkness."""
        key = (self._version, id(self._camera()), self._pending is not None)
        if self._cache.get("ink_key") == key:
            return self._cache["ink"]
        polys, dark = [], []
        for ln in self.ink():
            p, w, d = ln["xy"], ln["width_m"], ln["darkness"]
            polys.append((p[:, None, :] + 0.5 * w[:, None, None] * _DISC[None, :, :]).reshape(-1, 2))    # a dot per point
            dark.append((np.repeat(d, 8), 8))
            if len(p) > 1:
                seg = p[1:] - p[:-1]
                n = np.stack([-seg[:, 1], seg[:, 0]], axis=1) / np.maximum(np.linalg.norm(seg, axis=1), 1e-12)[:, None]
                a, b = 0.5 * w[:-1, None] * n, 0.5 * w[1:, None] * n
                polys.append(np.stack([p[:-1] + a, p[1:] + b, p[1:] - b, p[:-1] - a], axis=1).reshape(-1, 2))
                dark.append((np.repeat(0.5 * (d[:-1] + d[1:]), 4), 4))
        cover = np.zeros((H, W), np.float32)
        if polys:
            uv = self._project(np.concatenate(polys))
            x0, y0 = max(0, int(math.floor(uv[:, 0].min())) - 2), max(0, int(math.floor(uv[:, 1].min())) - 2)
            x1, y1 = min(W, int(math.ceil(uv[:, 0].max())) + 3), min(H, int(math.ceil(uv[:, 1].max())) + 3)
            if x1 > x0 and y1 > y0:
                canvas = np.zeros(((y1 - y0) * _SS, (x1 - x0) * _SS), np.uint8)
                # pixel (i) spans i - 0.5 .. i + 0.5; its sub-pixel k is centred at i - 0.5 + (k + 0.5) / SS
                fx = np.round(((uv - (x0, y0) + 0.5) * _SS - 0.5) * (1 << _SHIFT)).astype(np.int32)
                items, i = [], 0
                for vals, k in dark:
                    n = len(vals) // k
                    items.extend((float(vals[j * k]), i + j * k, k) for j in range(n))
                    i += len(vals)
                for dk, i0, k in sorted(items):            # darker ink is drawn last: it stays on top
                    cv2.fillConvexPoly(canvas, fx[i0:i0 + k], int(round(255 * min(1.0, dk))), lineType=cv2.LINE_8, shift=_SHIFT)
                cover[y0:y1, x0:x1] = cv2.resize(canvas, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        self._cache["ink_key"], self._cache["ink"] = key, cover
        return cover

    def _draw_arm(self, img: np.ndarray, exposure: str):
        """A rough silhouette of the arm and the pen at the last pose seen: pale links, dark pen. It
        is there so that a photo taken with the arm over the drawing does not look clean."""
        cam, q = self._camera(), self._last_q
        fr = K.link_frames(q)
        tip, lean = self._true(q, self._time)
        axis = K.fk(q, self._tool_true)[:3, :3] @ self.pen_axis
        chain = [fr[1][:3, 3], fr[2][:3, 3], fr[3][:3, 3], K.fk(q)[:3, 3]]
        gain = EXPOSURE_GAIN[exposure]

        def line(a, b, width_m, bgr):
            uv = cam.project([a, b])
            depth = float((cam.R @ ((np.asarray(a) + np.asarray(b)) / 2.0) + cam.tvec)[2])
            if depth < 0.05 or not np.all(np.isfinite(uv)) or np.abs(uv).max() > 20000:
                return
            col = tuple(float(min(1.5, (c / 255.0) ** GAMMA * gain)) for c in bgr)
            cv2.line(img, (int(round(uv[0, 0])), int(round(uv[0, 1]))), (int(round(uv[1, 0])), int(round(uv[1, 1]))),
                     col, thickness=max(2, int(round(cam.f * width_m / depth))), lineType=cv2.LINE_AA)

        for a, b in zip(chain[:-1], chain[1:]):
            line(a, b, 0.045, (150.0, 150.0, 150.0))
        line(tip - 0.10 * axis, tip, 0.009, (28.0, 28.0, 30.0))

    def render(self, exposure: str = "paper", noise: float | None = None, blur_px: float | None = None,
               show_arm: bool | None = None) -> np.ndarray:
        """The photo the fixed camera would take now: (1080, 1920, 3) BGR uint8.

        exposure "paper" (110 / gain 0): paper about 175, nothing clipped, ink shows. "desk" (the
        locked 333 / 29): the desk is well exposed and the paper is clipped to 255; a thin line is
        gone, a bold one is left as a narrow grey trace. Light is added up in linear terms, blurred,
        clipped and only then turned into pixel values, as in the real camera."""
        if exposure not in EXPOSURES:
            raise ValueError(f"exposure {exposure!r} is not one of {EXPOSURES}")
        gain = EXPOSURE_GAIN[exposure]
        paper = (np.array(PAPER_BGR, np.float32) / 255.0) ** GAMMA * gain
        desk = (np.array(DESK_BGR[exposure], np.float32) / 255.0) ** GAMMA
        sheet = self._sheet_mask()
        lit = sheet * (1.0 - self._ink_cover() * (1.0 - INK_REFLECTANCE))
        if self.shading:
            lit = lit * (1.0 - self.shading * (H - 1 - np.arange(H, dtype=np.float32)) / (H - 1))[:, None]
        img = lit[:, :, None] * paper + (1.0 - sheet)[:, :, None] * desk
        if (self.show_arm if show_arm is None else show_arm) and self._last_q is not None:
            self._draw_arm(img, exposure)
        sigma = self.blur_px if blur_px is None else float(blur_px)
        if sigma > 0.0:
            img = cv2.GaussianBlur(img, (0, 0), sigma)
        img = np.clip(img, 0.0, 1.0) ** (1.0 / GAMMA) * 255.0
        sd = self.noise if noise is None else float(noise)
        if sd > 0.0:
            img = img + self._noise_rng.normal(0.0, sd, img.shape).astype(np.float32)
        return np.clip(img + 0.5, 0.0, 255.0).astype(np.uint8)

    def snap(self, path: str | None = None, exposure: str = "paper", **kw) -> np.ndarray:
        """render(), and the photo written to `path` (jpeg quality 95, as fd snap) when given."""
        img = self.render(exposure, **kw)
        if path:
            cv2.imwrite(path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        return img
