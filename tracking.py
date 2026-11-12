"""Tracking measurement: how well does the tool tip follow a planned slow path?

Why this exists: on 2026-11-11 a pen was drawn with at 6 mm/s and the lines came out stair-stepped
and partly dashed. The arm's joints stick and slip at that speed (measured, see README / DESIGN):
a joint stands still until the standing command is about 16 encoder ticks (lift / elbow) or 7 ticks
(pan / wrist) away, then jumps 6-10 ticks. This module records what the arm did during a motion
(`record`, through run_path's monitor callback), puts numbers on it (`Trace.analyse`, the figures
of the night's prototype tools/pen_draw/track_test.py, so old and new traces compare) and says
WHAT the error is (`Trace.breakdown`: along the path, across it, in height, per joint when the
joint angles are known, spectrum, stick-slip signature).

Pure measurement: nothing here writes to the servo bus except through Arm.run_path. Two kinds of
Trace exist: recorded here (joint angles of every control cycle; tip positions follow by FK for any
tool) and loaded from the Cartesian files of the night (t, planned xyz, measured xyz; no joints).
`joints_from_tips` recovers the encoder ticks behind such a file exactly (the measured tip is the
FK of whole ticks), which is how the per-joint numbers of 2026-11-11 were obtained.

Units: seconds, metres, radians inside; millimetres and degrees in the result dictionaries.
"""

from __future__ import annotations

import math

import numpy as np

import kinematics as K

ARM = K.ARM_JOINTS
TICKS_PER_RAD = 4096.0 / (2.0 * math.pi)
SHAPES = ("circle", "square", "line")


class Trace:
    """One recorded motion. `t` (N,), then either joint angles `q_plan` / `q_meas` (N, 5; radians)
    or tip positions `tip_plan` / `tip_meas` (N, 3; metres), or both. `meta` is free text / numbers
    about how it was taken; `result` is run_path's result when the motion completed."""

    def __init__(self, t, q_plan=None, q_meas=None, tip_plan=None, tip_meas=None, meta=None):
        self.t = np.asarray(t, dtype=float).reshape(-1)
        n = len(self.t)

        def arr(v, width):
            if v is None:
                return None
            v = np.asarray(v, dtype=float).reshape(-1, width)
            if len(v) != n:
                raise ValueError(f"trace arrays differ in length ({len(v)} vs {n} time stamps)")
            return v

        self.q_plan, self.q_meas = arr(q_plan, 5), arr(q_meas, 5)
        self.tip_plan, self.tip_meas = arr(tip_plan, 3), arr(tip_meas, 3)
        if (self.q_plan is None or self.q_meas is None) and (self.tip_plan is None or self.tip_meas is None):
            raise ValueError("a Trace needs joint angles (q_plan, q_meas) or tip positions (tip_plan, tip_meas)")
        self.meta = dict(meta or {})
        self.result = None

    def __len__(self):
        return len(self.t)

    @property
    def has_joints(self) -> bool:
        return self.q_plan is not None and self.q_meas is not None

    # ------------------------------------------------------------ files
    @classmethod
    def from_tip_array(cls, a, meta=None) -> "Trace":
        """(N, 7) array: t, planned x y z, measured x y z (the files of tools/pen_draw/track_test.py)."""
        a = np.asarray(a, dtype=float)
        if a.ndim != 2 or a.shape[1] != 7:
            raise ValueError(f"a Cartesian trace has 7 columns (t, plan xyz, measured xyz), got shape {a.shape}")
        return cls(a[:, 0], tip_plan=a[:, 1:4], tip_meas=a[:, 4:7], meta=meta)

    @classmethod
    def load(cls, path) -> "Trace":
        """A trace file: .npz written by save() (joints, and tips if they were stored), or a .npy
        Cartesian trace (N, 7) as saved on 2026-11-11."""
        path = str(path)
        if path.endswith(".npz"):
            with np.load(path, allow_pickle=False) as z:
                get = lambda k: z[k] if k in z.files else None  # noqa: E731
                meta = {}
                if "meta_json" in z.files:
                    import json
                    meta = json.loads(str(z["meta_json"]))
                tr = cls(z["t"], get("q_plan"), get("q_meas"), get("tip_plan"), get("tip_meas"), meta)
            return tr
        tr = cls.from_tip_array(np.load(path, allow_pickle=False))
        tr.meta["file"] = path
        return tr

    def save(self, path, tool=None) -> str:
        """Write a compressed .npz (about 60 bytes per control cycle). With `tool` (4x4, e.g.
        arm.tip_tool()) the tip positions are stored too, so the file can be read without knowing
        which tool was on."""
        import json
        path = str(path)
        if not path.endswith(".npz"):
            path += ".npz"
        data = {"t": self.t, "meta_json": np.array(json.dumps(self.meta, default=str))}
        if self.has_joints:
            data.update(q_plan=self.q_plan, q_meas=self.q_meas)
        if tool is not None or self.tip_plan is not None:
            pp, pm = self.tip(tool)
            data.update(tip_plan=pp, tip_meas=pm)
        np.savez_compressed(path, **data)
        return path

    # ------------------------------------------------------------ geometry
    def tip(self, tool=None):
        """(planned xyz, measured xyz), each (N, 3) metres: FK of the joint angles with `tool`
        (4x4 tool frame, e.g. arm.tip_tool(), or a tooling.Tool; None = gripper_frame_link). A
        trace that only has stored tip positions returns those (`tool` cannot change them)."""
        if self.has_joints and (tool is not None or self.tip_plan is None):
            return fk_points(self.q_plan, tool), fk_points(self.q_meas, tool)
        if self.tip_plan is None:
            raise ValueError("this trace has no tip positions and no joint angles")
        return self.tip_plan, self.tip_meas

    # ------------------------------------------------------------ the figures of the night
    def analyse(self, tool=None) -> dict:
        """Tracking figures, computed exactly as tools/pen_draw/track_test.py did on the real arm
        (so they compare with testdata/pen_20261111/manifest.json):

        path_err_mm   distance of every measured tip position to the planned path on the table
                      plane (x, y only; nearest planned sample): mean / p95 / max
        z_err_mm      measured minus planned height: min / max (the "height wander")
        lag_s         time shift (0 .. 1.55 s) that best aligns measured with planned (x, y)
        err_after_lag_mm  mean x, y error at that shift (middle 20 .. 90 % of the motion)
        end_err_mm    3-D distance between the last measured and the last planned position
        duration_s    time stamp of the last sample; n = samples; rate_hz = control cycles per second
        """
        pp, pm = self.tip(tool)
        t = self.t
        n = len(t)
        if n < 2:
            raise ValueError("a trace needs at least two samples")
        d = _nearest_xy(pm[:, :2], pp[:, :2])
        best = (float("inf"), 0.0)
        a, b = int(n * 0.2), int(n * 0.9)
        for lag in np.arange(0.0, 1.6, 0.05):
            idx = np.clip(np.searchsorted(t, t - lag), 0, n - 1)
            e = np.linalg.norm(pm[:, :2] - pp[idx, :2], axis=1)[a:b]
            e = float(e.mean()) if len(e) else float("inf")
            if e < best[0]:
                best = (e, float(lag))
        dz = pm[:, 2] - pp[:, 2]
        dur = float(t[-1])
        return {
            "path_err_mm": {"mean": round(float(d.mean()) * 1000.0, 2), "p95": round(float(np.percentile(d, 95)) * 1000.0, 2),
                            "max": round(float(d.max()) * 1000.0, 2)},
            "z_err_mm": {"min": round(float(dz.min()) * 1000.0, 2), "max": round(float(dz.max()) * 1000.0, 2)},
            "lag_s": round(best[1], 2),
            "err_after_lag_mm": round(best[0] * 1000.0, 2) if math.isfinite(best[0]) else None,
            "end_err_mm": round(float(np.linalg.norm(pm[-1] - pp[-1])) * 1000.0, 2),
            "duration_s": round(dur, 2),
            "n": int(n),
            "rate_hz": round((n - 1) / dur, 1) if dur > 0 else None,
        }

    # ------------------------------------------------------------ what the error is
    def breakdown(self, tool=None, cfg=None) -> dict:
        """What the tracking error consists of.

        "tool_frame_mm": the error (measured minus planned, same time stamp) split into the
            direction the tool is moving in on the table ("along": + = ahead of the plan),
            sideways to it ("across": + = to the left of the direction of travel) and "height";
            each with mean / rms / min / max over the part of the motion where the plan moves.
        "spectrum": per component the frequency with the most power (Hz, mean removed) and the
            share of the power above 1 Hz (a stick-slip cycle shows as a peak of 1.5 - 2.5 Hz;
            a slow drift as a share near 0).
        "stick_slip": the measured tip stands still (< 0.5 mm/s) for "stall_frac" of the time the
            plan moves, and moves in bursts: "burst_ratio" = 95th percentile of the measured tip
            speed / median planned speed (about 1 for smooth tracking; 4 - 9 on the real arm at
            6 mm/s, where one encoder tick is 0.3 - 0.4 mm of tip travel).
        "joints" (only with joint angles; cfg gives ticks): see joint_stats().
        """
        pp, pm = self.tip(tool)
        t = self.t
        n = len(t)
        e = pm - pp
        dt = np.diff(t)
        dt = np.where(dt > 1e-6, dt, 1e-6)
        vp = np.diff(pp[:, :2], axis=0) / dt[:, None]
        sp = np.linalg.norm(vp, axis=1)
        moving = np.concatenate([sp > 5e-4, [False]])  # the plan moves faster than 0.5 mm/s
        # direction of travel: planned velocity smoothed over about 0.25 s, held where the plan rests
        k = max(1, int(round(0.25 / float(np.median(dt)))))
        ker = np.ones(k) / k
        vs = np.stack([np.convolve(vp[:, i], ker, mode="same") for i in range(2)], axis=1)
        tan = np.zeros((n, 2))
        last = np.array([1.0, 0.0])
        for i in range(n):
            v = vs[min(i, n - 2)]
            s = float(np.hypot(v[0], v[1]))
            if s > 2e-4:
                last = v / s
            tan[i] = last
        along = e[:, 0] * tan[:, 0] + e[:, 1] * tan[:, 1]
        across = -e[:, 0] * tan[:, 1] + e[:, 1] * tan[:, 0]
        comp = {"along": along, "across": across, "height": e[:, 2]}
        sel = moving if moving.sum() >= 8 else np.ones(n, dtype=bool)

        def stat(v):
            v = v[sel] * 1000.0
            return {"mean": round(float(v.mean()), 2), "rms": round(float(np.sqrt((v * v).mean())), 2),
                    "min": round(float(v.min()), 2), "max": round(float(v.max()), 2)}

        out = {"tool_frame_mm": {k_: stat(v) for k_, v in comp.items()},
               "spectrum": {k_: _spectrum(t[sel], v[sel]) for k_, v in comp.items()}}
        vm = np.linalg.norm(np.diff(pm[:, :2], axis=0), axis=1) / dt
        mv = moving[:-1]
        if mv.sum() >= 8:
            out["stick_slip"] = {"stall_frac": round(float(np.mean(vm[mv] < 5e-4)), 2),
                                 "burst_ratio": round(float(np.percentile(vm[mv], 95) / max(float(np.median(sp[mv])), 1e-6)), 1),
                                 "plan_speed_mm_s": round(float(np.median(sp[mv])) * 1000.0, 2)}
        else:
            out["stick_slip"] = None
        if self.has_joints:
            out["joints"] = self.joint_stats(cfg, tool)
        return out

    def joint_stats(self, cfg=None, tool=None) -> dict:
        """Per joint: tracking error (planned minus measured; rms / min / max in degrees and, with
        cfg, the rms in encoder ticks), how much tip error it causes ("tip_rms_mm": rms of the
        joint error times its lever at the tool tip) and its slips: a slip is a move of 3 ticks
        or more that follows 0.12 s or more of standing still ("slips", "slip_ticks" median,
        "stuck_s" median, "err_at_start_ticks": median |error| when the joint let go). Needs joint
        angles; ticks need cfg ("zero_ticks", "sign")."""
        if not self.has_joints:
            raise ValueError("this trace has no joint angles (see joints_from_tips for the files of 2026-11-11)")
        t = self.t
        n = len(t)
        e = self.q_plan - self.q_meas
        dq = np.abs(np.diff(self.q_plan, axis=0)).sum(axis=0)
        sel = np.ones(n, dtype=bool)
        lever = np.zeros(5)
        tool = tool_frame(tool)
        q_mid = self.q_meas[n // 2]
        for i in range(5):
            d = np.zeros(5)
            d[i] = 1e-5
            lever[i] = float(np.linalg.norm(K.fk(q_mid + d, tool)[:3, 3] - K.fk(q_mid - d, tool)[:3, 3])) / 2e-5
        out = {}
        for i, j in enumerate(ARM):
            ed = np.degrees(e[sel, i])
            rec = {"err_rms_deg": round(float(np.sqrt((ed * ed).mean())), 3), "err_min_deg": round(float(ed.min()), 2),
                   "err_max_deg": round(float(ed.max()), 2), "travel_deg": round(float(math.degrees(dq[i])), 1),
                   "tip_rms_mm": round(float(np.sqrt((e[sel, i] ** 2).mean())) * lever[i] * 1000.0, 2),
                   "mm_per_tick": round(lever[i] / TICKS_PER_RAD * 1000.0, 2)}
            if cfg is not None:
                z, s = float(cfg["zero_ticks"][j]), float(cfg["sign"][j])
                plan_tk = z + s * self.q_plan[:, i] * TICKS_PER_RAD
                meas_tk = np.round(z + s * self.q_meas[:, i] * TICKS_PER_RAD)
                rec["err_rms_ticks"] = round(rec["err_rms_deg"] * 4096.0 / 360.0, 1)
                ev = slip_events(t, plan_tk, meas_tk)
                rec["slips"] = len(ev)
                if ev:
                    a = np.array(ev)
                    rec.update(slip_ticks=round(float(np.median(a[:, 4])), 1), stuck_s=round(float(np.median(a[:, 5])), 2),
                               err_at_start_ticks=round(float(np.median(np.abs(a[:, 2]))), 1))
            out[j] = rec
        return out


# ---------------------------------------------------------------- helpers
def tool_frame(tool=None) -> np.ndarray:
    """The 4x4 frame every `tool` argument of this module means: None = gripper_frame_link, a 4x4
    array (arm.tip_tool()), or an object with a `tip_offset` (a tooling.Tool, which is what `tool`
    means everywhere else in the harness; tooling is not imported here)."""
    if tool is None:
        return K._T_TOOL
    off = getattr(tool, "tip_offset", None)
    if off is not None:
        return K.tool_offset(*(float(v) for v in off))
    t = np.asarray(tool, dtype=float)
    if t.shape != (4, 4):
        raise ValueError(f"tool must be None, a 4x4 frame (arm.tip_tool()) or a tooling.Tool, got shape {t.shape}")
    return t


def fk_points(q, tool=None) -> np.ndarray:
    """Tool-tip positions (N, 3) for joint angles (N, 5): the same chain as kinematics.fk,
    vectorised (a trace has thousands of poses). tool: see tool_frame()."""
    q = np.asarray(q, dtype=float).reshape(-1, 5)
    n = len(q)
    t = np.broadcast_to(np.eye(4), (n, 4, 4)).copy()
    for i in range(5):
        c, s = np.cos(q[:, i]), np.sin(q[:, i])
        r = np.zeros((n, 4, 4))
        r[:, 0, 0], r[:, 0, 1], r[:, 1, 0], r[:, 1, 1], r[:, 2, 2], r[:, 3, 3] = c, -s, s, c, 1.0, 1.0
        t = t @ K._FIXED[i] @ r
    return (t @ tool_frame(tool))[:, :3, 3]


def _nearest_xy(pts, path, chunk: int = 2000) -> np.ndarray:
    """Distance of every point to the nearest sample of `path` (both (N, 2)); chunked so that a
    drawing of several minutes does not need an N x N matrix at once."""
    out = np.empty(len(pts))
    for i in range(0, len(pts), chunk):
        d = pts[i:i + chunk, None, :] - path[None, :, :]
        out[i:i + chunk] = np.sqrt((d * d).sum(axis=2)).min(axis=1)
    return out


def _spectrum(t, v) -> dict:
    """Dominant frequency (Hz) and share of the power above 1 Hz of a signal sampled at (nearly)
    even times; None values when it is too short."""
    if len(v) < 32:
        return {"peak_hz": None, "above_1hz": None}
    n = len(v)
    tt = np.linspace(float(t[0]), float(t[-1]), n)
    x = np.interp(tt, t, v)
    x = x - x.mean()
    p = np.abs(np.fft.rfft(x * np.hanning(n))) ** 2
    f = np.fft.rfftfreq(n, d=(tt[-1] - tt[0]) / (n - 1))
    if p[1:].sum() <= 0.0:
        return {"peak_hz": None, "above_1hz": None}
    return {"peak_hz": round(float(f[1:][int(np.argmax(p[1:]))]), 2), "above_1hz": round(float(p[f > 1.0].sum() / p[1:].sum()), 2)}


def slip_events(t, plan_ticks, meas_ticks, min_stuck_s: float = 0.12, min_slip: int = 3) -> list:
    """Slips of one joint: the measured position stood still for min_stuck_s or longer and then
    moved min_slip ticks or more in one go (pauses of up to two samples inside a slip are allowed).
    Returns [(t_start, direction, error at start, error at end, slip ticks, stuck seconds, slip
    seconds), ...] with error = planned - measured in ticks."""
    t = np.asarray(t, dtype=float)
    m = np.asarray(meas_ticks, dtype=float)
    p = np.asarray(plan_ticks, dtype=float)
    out, n, i, last = [], len(t), 0, 0
    while i < n - 1:
        if m[i + 1] == m[i]:
            i += 1
            continue
        stuck = t[i] - t[last]
        d = math.copysign(1.0, m[i + 1] - m[i])
        k, gap, end = i + 1, 0, i + 1
        while k < n - 1 and gap <= 2:
            step = m[k + 1] - m[k]
            if step * d > 0:
                end, gap = k + 1, 0
            elif step != 0:
                break
            else:
                gap += 1
            k += 1
        slip = abs(m[end] - m[i])
        if stuck >= min_stuck_s and slip >= min_slip:
            out.append((float(t[i]), d, float(p[i] - m[i]), float(p[end] - m[end]), float(slip), float(stuck),
                        float(t[end] - t[i])))
        last = i = end
    return out


# ---------------------------------------------------------------- recording
def record(arm, waypoints, speed_deg_s, **run_path_kw) -> Trace:
    """Run arm.run_path(waypoints, speed_deg_s, **run_path_kw) and record the planned and the
    measured joint angles of every control cycle through run_path's monitor callback.

    A `monitor` given in run_path_kw is still called (and can still stop the motion). Returns the
    Trace; `trace.result` is run_path's result. When the motion raises (SafetyError before it
    starts, MotionAborted during it), the exception is passed on with the part recorded so far
    attached as `exc.trace` (None when nothing was recorded)."""
    user = run_path_kw.pop("monitor", None)
    ts, qp, qm = [], [], []

    def monitor(t, q_plan, q_meas):
        ts.append(float(t))
        qp.append(np.array(q_plan, dtype=float))
        qm.append(np.array(q_meas, dtype=float))
        return bool(user(t, q_plan, q_meas)) if user is not None else False

    def build():
        return Trace(ts, qp, qm, meta={"speed_deg_s": float(speed_deg_s),
                                       "fine": bool(run_path_kw.get("fine", False))}) if len(ts) >= 2 else None

    try:
        res = arm.run_path(waypoints, speed_deg_s, monitor=monitor, **run_path_kw)
    except Exception as e:
        try:
            e.trace = build()
        except Exception:
            e.trace = None
        raise
    tr = build()
    if tr is None:
        raise ValueError("the motion had fewer than two control cycles; nothing to analyse")
    tr.result = res
    return tr


def fine_accepted(cfg: dict, min_runs: int = 2) -> bool:
    """Has the fine mode passed on the REAL arm? True when config.json holds at least min_runs
    comparisons recorded by `fd track-test --fine --repeat 3 --accept` (cfg["fine_motion"]["accepted"];
    a simulated run is never recorded there). This, not a constant in the code, is what may make
    fine=True the default of a drawing: callers pass run_path(..., fine=fine_accepted(arm.cfg))."""
    acc = (cfg.get("fine_motion") or {}).get("accepted") or []
    return sum(1 for r in acc if isinstance(r, dict) and (r.get("verdict") or {}).get("passed") is True) >= int(min_runs)


def shape_points(shape: str, size_m: float, centre_xy, z: float) -> list:
    """Corner points [(x, y, z), ...] of a test figure in the air at height z (base frame):
    "circle" (diameter size_m, 40 chords, starts at +x), "square" (side size_m, starts at the
    corner nearest to the robot on its right, first side towards +y), "line" (length size_m along
    y, there and back). The first point is where the figure starts and ends."""
    cx, cy = float(centre_xy[0]), float(centre_xy[1])
    h = 0.5 * float(size_m)
    if shape == "circle":
        pts = [(cx + h * math.cos(a), cy + h * math.sin(a), z) for a in np.linspace(0.0, 2.0 * math.pi, 41)]
        return pts[:-1] + [pts[0]]  # closed exactly (sin(2 pi) is not quite 0)
    if shape == "square":
        return [(cx - h, cy - h, z), (cx - h, cy + h, z), (cx + h, cy + h, z), (cx + h, cy - h, z), (cx - h, cy - h, z)]
    if shape == "line":
        return [(cx, cy - h, z), (cx, cy + h, z), (cx, cy - h, z)]
    raise ValueError(f"unknown shape {shape!r} (known: {', '.join(SHAPES)})")


def joint_speed_for(wps, length_m: float, v_m_s: float, lo: float = 1.5, hi: float = 35.0) -> float:
    """Joint speed (deg/s of the fastest joint, what run_path takes) that moves the tool tip along
    a path of length_m at about v_m_s: 1.3 x the largest joint travel divided by the time the tip
    should take (the factor the prototype used: the joint that travels most is not the fastest one
    everywhere). Clamped to lo .. hi."""
    dq = float(np.degrees(np.abs(np.diff(np.asarray(wps, dtype=float), axis=0)).sum(axis=0).max()))
    return float(min(hi, max(lo, 1.3 * dq * v_m_s / max(float(length_m), 1e-4))))


# ---------------------------------------------------------------- joints behind a Cartesian trace
def joints_from_tips(trace: Trace, cfg: dict, tip_offset, pitch_deg: float, roll_tick: float | None = None,
                     first_range=(2, 22, 22, 22, 2)):
    """Encoder ticks (N, 5) behind a Cartesian trace whose measured tip positions are the FK of
    whole encoder ticks (the files of 2026-11-11), or None when the first sample does not fit
    (wrong tip_offset / calibration).

    The first sample is searched around the IK solution at pitch_deg (within first_range ticks per
    joint), every later one within 3 ticks of the one before (6, and 3 on the roll, when that
    fails). A sample counts as found when FK reproduces the saved position to 0.1 um; the search is
    unambiguous because three coordinates to 1e-7 m pin four small integers. cfg supplies
    "zero_ticks" / "sign"; tip_offset is the tool tip the trace was recorded with.
    Returns (ticks, n_bad) with n_bad = samples that could not be matched (kept at the previous)."""
    if trace.tip_meas is None:
        raise ValueError("the trace has no stored tip positions")
    tool = K.tool_offset(*tip_offset)
    z = np.array([float(cfg["zero_ticks"][j]) for j in ARM])
    s = np.array([float(cfg["sign"][j]) for j in ARM])

    def search(p, centre, rng):
        grids = [np.arange(c - r, c + r + 1) for c, r in zip(centre, rng)]
        cand = np.array(np.meshgrid(*grids, indexing="ij")).reshape(5, -1).T.astype(float)
        d = np.linalg.norm(fk_points(s * (cand - z) / TICKS_PER_RAD, tool) - p, axis=1)
        i = int(np.argmin(d))
        return cand[i], float(d[i])

    pm = trace.tip_meas
    rt = round(z[4]) if roll_tick is None else float(roll_tick)
    q0, _info = K.ik(pm[0], pitch_deg, s[4] * (rt - z[4]) / TICKS_PER_RAD, tool=tool)
    tk, d = search(pm[0], np.round(z + s * q0 * TICKS_PER_RAD), first_range)
    if d > 1e-7:
        return None
    out, bad = [tk], 0
    for p in pm[1:]:
        tk2, d = search(p, out[-1], (3, 3, 3, 3, 0))
        if d > 1e-7:
            tk2, d = search(p, out[-1], (6, 6, 6, 6, 3))
            if d > 1e-7:
                tk2, bad = out[-1], bad + 1
        out.append(tk2)
    return np.array(out), bad
