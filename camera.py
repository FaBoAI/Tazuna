"""Fixed webcam (Logitech C920) capture with locked focus / exposure / white balance,
plus the pinhole camera model used to relate pixels to the robot base frame."""

from __future__ import annotations

import json
import os
import subprocess
import time

import cv2
import numpy as np

DEV = "/dev/video0"
W, H = 1920, 1080
# The calibration data of this rig cannot identify the focal length (review finding VM-4: the
# reprojection error is flat from f = 1400 to 2000), so it is pinned to the C920's nominal value:
# fit_camera(xyz, uv, f_range=(F_PINNED, F_PINNED)).
F_PINNED = 1430.0
HERE = os.path.dirname(os.path.abspath(__file__))
CAM_MODEL_PATH = os.path.join(HERE, "camera_model.json")

# Order matters: the manual values only take effect after the auto modes are off.
LOCKED = [
    ("focus_automatic_continuous", 0),
    ("focus_absolute", 20),
    ("white_balance_automatic", 0),
    ("white_balance_temperature", 4100),
    ("auto_exposure", 1),
    ("exposure_dynamic_framerate", 0),
    ("exposure_time_absolute", 333),
    ("gain", 29),
]
AUTO = [
    ("focus_automatic_continuous", 1),
    ("white_balance_automatic", 1),
    ("auto_exposure", 3),
    ("exposure_dynamic_framerate", 1),
]

# Exposure by scene. "desk" is what LOCKED sets (tuned for the dark desk: the colour thresholds and the scene
# perception assume it). On white paper it clips to 255 and a thin ink line disappears (2026-11-11: a detector
# answered "no ink" on such photos and the pen was pressed deeper). "paper" is the short exposure with which the
# paper reads about 175 and ink shows. Focus and white balance are the same for both (LOCKED).
# Order matters inside a preset: manual mode first, then the values.
_EXPOSURE_KEYS = ("auto_exposure", "exposure_dynamic_framerate", "exposure_time_absolute", "gain")
EXPOSURES = {
    "desk": {k: v for k, v in LOCKED if k in _EXPOSURE_KEYS},
    "paper": {"auto_exposure": 1, "exposure_dynamic_framerate": 0, "exposure_time_absolute": 110, "gain": 0},
}
EXPOSURE_WARM_FRAMES = 10   # frames dropped after a change of exposure (the first ones still have the old one)


def _ctl(pairs):
    for k, v in pairs:
        subprocess.run(["v4l2-ctl", "-d", DEV, "-c", f"{k}={v}"], capture_output=True, check=False)


def lock_settings():
    _ctl(LOCKED)


def restore_auto():
    _ctl(AUTO)


def set_exposure(name: str) -> None:
    """Write the exposure controls of a preset (EXPOSURES: "desk" | "paper"). Focus and white balance are not
    touched: lock_settings() sets those, and stays what it was (= "desk"). Whoever switches to "paper" puts
    "desk" back afterwards (grab(exposure=...) does): everything else in the harness assumes the desk values."""
    if name not in EXPOSURES:
        raise ValueError(f"unknown exposure {name!r}: one of {', '.join(EXPOSURES)}")
    _ctl(list(EXPOSURES[name].items()))


def exposure_name(vals: dict | None = None) -> str | None:
    """Name of the preset the camera is set to now ("desk" | "paper"), "auto" when the exposure is automatic,
    None when the camera cannot be asked or the values are no preset."""
    vals = read_settings() if vals is None else vals
    if not vals:
        return None
    if vals.get("auto_exposure") == dict(AUTO)["auto_exposure"]:
        return "auto"
    for name, want in EXPOSURES.items():
        if all(k in vals and abs(vals[k] - v) <= READBACK_TOLERANCE.get(k, 0) for k, v in want.items()):
            return name
    return None


def read_settings() -> dict:
    """Current values of the controls in LOCKED (name -> int); {} when v4l2-ctl cannot be asked."""
    names = ",".join(k for k, _ in LOCKED)
    try:
        out = subprocess.run(["v4l2-ctl", "-d", DEV, "-C", names], capture_output=True, check=False, text=True,
                             timeout=5.0).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    vals = {}
    for line in out.splitlines():
        name, _, rest = line.partition(":")
        try:
            vals[name.strip()] = int(rest.split()[0])
        except (ValueError, IndexError):
            pass
    return vals


# The camera reports gain and exposure time rounded to its own steps while it is streaming
# (28 / 332 for the requested 29 / 333, measured on this C920): such a difference is not a change.
READBACK_TOLERANCE = {"gain": 2, "exposure_time_absolute": 2}


def settings_locked() -> bool | None:
    """True when lock_settings() would change nothing (every control already has its LOCKED value),
    False when at least one differs (camera in automatic mode, or a value was changed), None when
    the camera cannot be asked. After a change from automatic to fixed focus / exposure the first
    frames are still adapting; fastcam.FastCamera drops them."""
    vals = read_settings()
    if not vals:
        return None
    return all(k in vals and abs(vals[k] - v) <= READBACK_TOLERANCE.get(k, 0) for k, v in LOCKED)


def auto_modes_on() -> list | None:
    """The automatic modes (focus, white balance, exposure) that are still switched on, as
    "name=value" strings; [] when the camera is fully manual as lock_settings() leaves it, None
    when the camera cannot be asked. lock_settings() itself does not report a control that could
    not be set, so this is the check that the lock really took effect."""
    vals = read_settings()
    if not vals:
        return None
    want = dict(LOCKED)
    return [f"{k}={vals.get(k, 'missing')}" for k, _ in AUTO if vals.get(k) != want[k]]


def grab(n_avg: int = 4, warm: int = 5, lock: bool = True, exposure: str | None = None) -> np.ndarray:
    """Return one BGR frame (the mean of n_avg consecutive frames to suppress noise).

    exposure None: as always (the locked "desk" values when lock is True). exposure "paper" (or "desk"): that
    preset is written after the lock, at least EXPOSURE_WARM_FRAMES frames are dropped, and the desk values
    are put back before this returns, also when the camera fails."""
    if exposure is not None and exposure not in EXPOSURES:
        raise ValueError(f"unknown exposure {exposure!r}: one of {', '.join(EXPOSURES)}")
    if lock:
        lock_settings()
    if exposure is None:
        return _grab_frames(n_avg, warm)
    try:
        set_exposure(exposure)
        return _grab_frames(n_avg, max(warm, EXPOSURE_WARM_FRAMES))
    finally:
        if exposure != "desk":
            set_exposure("desk")


def _grab_frames(n_avg: int, warm: int) -> np.ndarray:
    cap = cv2.VideoCapture(DEV, cv2.CAP_V4L2)
    try:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {DEV}")
        for _ in range(warm):
            cap.read()
        acc = None
        got = 0
        t0 = time.time()
        while got < n_avg and time.time() - t0 < 5.0:
            ok, f = cap.read()
            if not ok or f is None:
                continue
            acc = f.astype(np.float32) if acc is None else acc + f.astype(np.float32)
            got += 1
        if got == 0:
            raise RuntimeError("camera returned no frames")
        return np.clip(acc / got + 0.5, 0, 255).astype(np.uint8)
    finally:
        cap.release()


# --------------------------------------------------------------------------- model
class CameraModel:
    """Pinhole model: pixel = K [R|t] X with X in the robot base frame (metres)."""

    def __init__(self, f: float, cx: float, cy: float, rvec, tvec, k1: float = 0.0, rms: float | None = None,
                 n: int | None = None):
        self.f, self.cx, self.cy, self.k1 = float(f), float(cx), float(cy), float(k1)
        self.rvec = np.asarray(rvec, dtype=float).reshape(3)
        self.tvec = np.asarray(tvec, dtype=float).reshape(3)
        self.R, _ = cv2.Rodrigues(self.rvec)
        self.rms, self.n = rms, n

    @property
    def K(self) -> np.ndarray:
        return np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])

    @property
    def dist(self) -> np.ndarray:
        return np.array([self.k1, 0.0, 0.0, 0.0, 0.0])

    @property
    def center(self) -> np.ndarray:
        """Camera position in the base frame."""
        return -self.R.T @ self.tvec

    def project(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 1, 3)
        uv, _ = cv2.projectPoints(pts, self.rvec, self.tvec, self.K, self.dist)
        return uv.reshape(-1, 2)

    def ray(self, uv) -> tuple[np.ndarray, np.ndarray]:
        """Origin and unit direction (base frame) of the viewing ray through pixel uv."""
        p = cv2.undistortPoints(np.asarray(uv, dtype=float).reshape(1, 1, 2), self.K, self.dist).reshape(2)
        d = self.R.T @ np.array([p[0], p[1], 1.0])
        return self.center, d / np.linalg.norm(d)

    def to_plane(self, uv, z: float) -> np.ndarray:
        """Intersection of the pixel ray with the horizontal plane at height z."""
        o, d = self.ray(uv)
        if abs(d[2]) < 1e-9:
            raise ValueError("ray parallel to the plane")
        s = (z - o[2]) / d[2]
        if s <= 0:
            raise ValueError("plane is behind the camera for this pixel")
        return o + s * d

    def to_table(self, uv, plane) -> np.ndarray:
        """Intersection of the pixel ray with the plane z = a + b*x + c*y (plane = (a, b, c), e.g.
        cfg["table_plane"]); to_plane() is the special case b = c = 0."""
        a, b, c = (float(v) for v in plane)
        o, d = self.ray(uv)
        den = d[2] - b * d[0] - c * d[1]
        if abs(den) < 1e-9:
            raise ValueError("ray parallel to the plane")
        s = (a + b * o[0] + c * o[1] - o[2]) / den
        if s <= 0:
            raise ValueError("plane is behind the camera for this pixel")
        return o + s * d

    def to_dict(self) -> dict:
        return {"f": self.f, "cx": self.cx, "cy": self.cy, "k1": self.k1, "rvec": self.rvec.tolist(),
                "tvec": self.tvec.tolist(), "rms_px": self.rms, "n_points": self.n,
                "center_base_m": self.center.tolist(), "image_size": [W, H]}

    def save(self, path: str = CAM_MODEL_PATH):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path: str | None = None) -> "CameraModel":
        """camera_model.json, or $FD_CAMERA_MODEL when that is set (see robot.load_config)."""
        path = path or os.environ.get("FD_CAMERA_MODEL") or CAM_MODEL_PATH
        with open(path) as fh:
            d = json.load(fh)
        return cls(d["f"], d["cx"], d["cy"], d["rvec"], d["tvec"], d.get("k1", 0.0), d.get("rms_px"), d.get("n_points"))


def _roundtrip_error(model: "CameraModel", n: int = 9) -> float:
    """Largest pixel error of pixel -> ray -> pixel over an n x n grid of the image (a distortion
    coefficient that cannot be inverted at the image corners shows up here)."""
    worst = 0.0
    for u in np.linspace(0.0, W - 1.0, n):
        for v in np.linspace(0.0, H - 1.0, n):
            o, d = model.ray((u, v))
            back = model.project([o + d])[0]
            worst = max(worst, float(np.hypot(back[0] - u, back[1] - v)))
    return worst


def fit_camera(xyz, uv, f_range=(1150.0, 1650.0), f_step: float = 10.0, fit_k1: bool = False) -> CameraModel:
    """Fit camera pose + focal length from 3D (base frame) <-> pixel correspondences.
    The principal point is fixed at the image centre. The focal length is found by a
    1-D search, each candidate solved with PnP; needs >= 6 well-spread, non-coplanar points.

    f_range = (f, f) PINS the focal length (no search, no refinement around it). The search
    never leaves f_range. The returned model carries diagnostics as attributes:
      f_at_bound   True when the best focal length sits at an end of a real search range
                   (the data did not identify it: pin it instead)
      rms_profile  [(f, rms_px), ...] of the coarse search
      k1_note      why a requested k1 was or was not accepted
    The in-sample rms understates the predictive error (about 2x on this rig): see leave_one_out()."""
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    if len(xyz) != len(uv):
        raise ValueError(f"{len(xyz)} points but {len(uv)} pixels")
    if len(xyz) < 6:
        raise ValueError("need at least 6 correspondences")
    f_lo, f_hi = float(f_range[0]), float(f_range[1])
    if not (f_lo > 0 and f_hi >= f_lo):
        raise ValueError(f"bad f_range {f_range}")
    if not f_step > 0:
        raise ValueError("f_step must be positive")
    cx, cy = W / 2.0, H / 2.0
    dist0 = np.zeros(5)

    def solve(f, init=None):
        Kf = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1.0]])
        try:
            if init is None:
                ok, rvec, tvec = cv2.solvePnP(xyz, uv, Kf, dist0, flags=cv2.SOLVEPNP_EPNP)
                if not ok:
                    return None
            else:
                rvec, tvec = init
            ok, rvec, tvec = cv2.solvePnP(xyz, uv, Kf, dist0, rvec.copy(), tvec.copy(), useExtrinsicGuess=True,
                                          flags=cv2.SOLVEPNP_ITERATIVE)
        except cv2.error:
            return None
        if not ok:
            return None
        proj, _ = cv2.projectPoints(xyz.reshape(-1, 1, 3), rvec, tvec, Kf, dist0)
        err = proj.reshape(-1, 2) - uv
        rms = float(np.sqrt(np.mean(np.sum(err * err, axis=1))))
        if not np.isfinite(rms):
            return None
        # the camera must look at the points (positive depth)
        R, _ = cv2.Rodrigues(rvec)
        depth = (R @ xyz.T + tvec.reshape(3, 1))[2]
        if np.any(depth <= 0):
            return None
        return rms, rvec, tvec

    pinned = f_hi - f_lo < 1e-9
    best = None
    profile = []
    f = f_lo
    while f <= f_hi + 1e-9:
        r = solve(f)
        if r is not None:
            profile.append((float(f), r[0]))
            if best is None or r[0] < best[0]:
                best = (r[0], float(f), r[1], r[2])
        f += f_step
    if best is None:
        raise RuntimeError("camera fit failed")
    if not pinned:
        # refine f around the best coarse value, never outside the requested range (finding VM-4)
        rms, f0, rvec, tvec = best
        for f in np.arange(max(f_lo, f0 - f_step), min(f_hi, f0 + f_step) + 1e-9, 1.0):
            r = solve(float(f), init=(rvec, tvec))
            if r is not None and r[0] < best[0]:
                best = (r[0], float(f), r[1], r[2])
    rms, f, rvec, tvec = best
    model = CameraModel(f, cx, cy, rvec, tvec, 0.0, rms, len(xyz))
    model.f_at_bound = bool(not pinned and (f <= f_lo + 1e-6 or f >= f_hi - 1e-6))
    model.rms_profile = profile
    model.k1_note = "not requested"
    if fit_k1:
        # A distortion term almost always lowers the in-sample rms; it is accepted only when it is
        # supported by the data and stays invertible over the whole image (finding VM-6).
        if len(xyz) < 25:
            model.k1_note = f"rejected: {len(xyz)} points, at least 25 are needed"
        else:
            Kf = model.K.copy()
            flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
                     | cv2.CALIB_FIX_FOCAL_LENGTH | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
            try:
                rms2, K2, d2, rv, tv = cv2.calibrateCamera([xyz.astype(np.float32)], [uv.astype(np.float32)],
                                                           (W, H), Kf, np.zeros(5), flags=flags)
                k1 = float(d2.ravel()[0])
                cand = CameraModel(f, cx, cy, rv[0], tv[0], k1, float(rms2), len(xyz))
                if not rms2 < 0.8 * rms:
                    model.k1_note = f"rejected: rms only {rms:.2f} -> {rms2:.2f} px (needs a 20 % improvement)"
                elif abs(k1) > 0.15:
                    model.k1_note = f"rejected: k1 = {k1:+.3f} is outside +-0.15"
                else:
                    rt = _roundtrip_error(cand)
                    if rt > 0.5:
                        model.k1_note = f"rejected: k1 = {k1:+.3f} cannot be inverted (round trip {rt:.1f} px)"
                    else:
                        cand.f_at_bound, cand.rms_profile = model.f_at_bound, profile
                        cand.k1_note = f"accepted: k1 = {k1:+.4f}, rms {rms:.2f} -> {rms2:.2f} px"
                        model = cand
            except cv2.error as e:
                model.k1_note = f"rejected: distortion fit failed ({e})"
    return model


def leave_one_out(xyz, uv, **fit_kwargs) -> dict:
    """Predictive error of fit_camera on this data: each point is left out in turn, the camera is
    fitted to the others and the left-out point is predicted.
    Returns {"px": [...], "mm": [...], "rms_px", "max_px", "rms_mm", "max_mm", "f_min", "f_max"}; mm is
    the horizontal distance between the true point and where its pixel lands on the horizontal plane
    through it (what an open-loop move to a place seen in the image would be off by)."""
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    uv = np.asarray(uv, dtype=float).reshape(-1, 2)
    n = len(xyz)
    if n < 7:
        raise ValueError("leave-one-out needs at least 7 correspondences")
    px, mm, fs = [], [], []
    for i in range(n):
        keep = np.arange(n) != i
        m = fit_camera(xyz[keep], uv[keep], **fit_kwargs)
        fs.append(m.f)
        px.append(float(np.linalg.norm(m.project(xyz[i:i + 1])[0] - uv[i])))
        try:
            q = m.to_plane(uv[i], xyz[i, 2])
            mm.append(float(np.linalg.norm(q[:2] - xyz[i, :2]) * 1000.0))
        except ValueError:
            mm.append(float("inf"))
    px_a, mm_a = np.array(px), np.array(mm)
    return {"px": px, "mm": mm, "rms_px": float(np.sqrt(np.mean(px_a ** 2))), "max_px": float(px_a.max()),
            "rms_mm": float(np.sqrt(np.mean(mm_a ** 2))), "max_mm": float(mm_a.max()),
            "f_min": float(min(fs)), "f_max": float(max(fs))}


def check_model(model: CameraModel, xyz=None, rms_max_px: float = 20.0) -> list:
    """Reasons why a fitted model should NOT be trusted (empty list = plausible): large residual,
    focal length at the search bound, camera below / behind the calibration points, a distortion
    that cannot be inverted."""
    problems = []
    if model.rms is None or not np.isfinite(model.rms) or model.rms > rms_max_px:
        problems.append(f"rms {model.rms} px is above {rms_max_px:g} px")
    if getattr(model, "f_at_bound", False):
        problems.append(f"focal length {model.f:g} is at the edge of the search range (not identified: pin it)")
    c = model.center
    if xyz is not None:
        xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
        if c[2] < float(xyz[:, 2].max()) + 0.05:
            problems.append("camera centre is not above the calibration points")
        if np.any((xyz @ model.R.T + model.tvec)[:, 2] <= 0):
            problems.append("some calibration points are behind the camera")
    if model.k1 and _roundtrip_error(model) > 0.5:
        problems.append(f"k1 = {model.k1:+.3f} cannot be inverted at the image corners")
    return problems


def residuals(model: CameraModel, xyz, uv) -> np.ndarray:
    return np.linalg.norm(model.project(xyz) - np.asarray(uv, dtype=float).reshape(-1, 2), axis=1)


def residual_vectors(model: CameraModel, xyz, uv) -> np.ndarray:
    """Signed (du, dv) = projected minus observed pixel for every correspondence (N x 2)."""
    return model.project(xyz) - np.asarray(uv, dtype=float).reshape(-1, 2)
