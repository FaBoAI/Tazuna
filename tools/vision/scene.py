"""Scene understanding from ONE fixed, calibrated camera (numpy + OpenCV only).

    perceive(img, cam, cfg, arm_q, jaw_q_deg)  -> fdtypes.Scene   (boxes, cylinders, containers)
    annotate(img, scene, cam)                  -> picture for a human / LLM glance
    find_object(scene, query)                  -> SceneObject (LookupError with the reason otherwise)
    object_present(img, cam, cfg, obj, arm_q)  -> bool            (object_presence() gives the details)
    objects_in_container(img, cam, cfg, box)   -> colour blobs inside the rim
    detect_jaws(img, ref, cam, arm_q, jaw, cfg)-> where the two jaw noses really are vs. kinematics

How it works (and what it assumes)
----------------------------------
* The table is the plane z = cfg["table_z"]. It is glossy dark-red wood, so "table" is a FAMILY of
  colours (red hue; the more mirror reflection, the brighter and less saturated). A pixel is an object
  pixel when it is outside that family. All brightness limits are RELATIVE to the table brightness
  measured in the same frame, so a global exposure / light change does not move anything (review
  finding VM-1). Known limit: things whose colour lies INSIDE the table family (brown, dark red,
  pinkish grey) are not seen, and a red object whose side faces are no brighter than the table
  shows only its top (it is then reported as flat / low confidence, never as a normal object).
* An object standing on the table shows its top face and one or two side faces. The top face is found
  as the bright part of the blob; the height follows from the vertical offset between the top face's
  near edges and the blob's near (ground contact) edges; with the height known the top face is
  back-projected to its own plane, where a box top is a true rectangle (minimum-area rectangle) and
  a cylinder top a true circle. Nothing is "fitted until the outline matches" with free width/height,
  which is what made v1 biased (finding VM-2). Dimensions are normalised: length >= width and yaw is
  the direction of the long axis.
* A container is a large dark box top (dark interior plus its rim; lit outer walls do not count). Its
  outline at rim height gives centre, size and orientation. What makes it a container is what a dark
  open box always shows, however it is turned and whatever lies in it: its rim along all four sides
  (each side is located from its own outline pixels, _outline_sides) and no table inside the rim; how
  much of its floor is visible does not matter. A corner behind the gripper is rebuilt from the two
  rim edges that meet there (and must agree with the other three corners); pieces of the box on both
  sides of the arm are put together. Its rim height is MEASURED: on the far inner wall (rim edge row
  versus wall/floor step row, finding VM-0) plus the floor thickness, which is assumed
  (cfg["scene"]["container_floor_mm"], default 5 mm) unless the foot of the near outer wall is in view:
  then rim edge and foot give the rim height above the table directly (_outer_wall_height) and the
  floor thickness follows. A thickness that the user has put into the config counts as known and is
  used in every view; where the image contradicts it by more than 1.5 mm the note says so (the
  camera calibration or that value is wrong). The notes on the object say which of these was used.
* Things seen through the opening are reported with in_container set; they are never graspable().
  Upright round things (cylinder, disc, cap) are found from the arcs of the blob's upper outline
  (touching things of one colour give one arc each): on the plane of the object's top the arc is a
  true circle, which gives centre and radius consistently with the height used (_round_tops). The
  height comes from the object's own base where that is seen on the floor, for an open top from the
  depth of its cavity, and is otherwise assumed (low confidence, said in the notes). Box-shaped
  things get the size of their top face; everything else a rough footprint with an ASSUMED height.
  Confidence above 0.5 means: measured, and standing on a floor whose height was measured too.
* Open tops (a cup, a bottle cap on its closed side; fdtypes.SceneObject.hollow, inner_radius, wall,
  z_inner_bottom): inside the rim circle the far inner wall is darker than the inner bottom, from the
  rim down to a step at the depth of the cavity (_cavity). hollow is True only on that evidence and
  False only for a top that is evenly lit, neither dark nor over-exposed, and shows nothing of it;
  else it stays None (not judged: a white top that is over-exposed all over cannot show a cavity)
  and the notes of a round thing say why. z_inner_bottom is the level the inner wall was SEEN down
  to (rim minus the visible depth): what stands on the inner bottom next to the wall is not known.
  It is given only when the object's height is measured (its own base in view); with the base
  hidden only the depth below the rim is known (in the notes) and z_inner_bottom stays None.
* The robot is removed with capsules around the links (joint angles through kinematics.link_frames),
  cables by their thinness, glare and mirror reflections by the table-colour family.
* detect_jaws slides each finger's outline (from the official meshes) over the pixels that are pale
  and new with respect to a reference frame, and converts the pixel shift of the nose into a
  base-frame xy offset. A nose that is hidden or merges with something white is refused.

Units: metres / radians in the robot base frame; pixels are full-resolution (1920 x 1080) pixels.
"""

from __future__ import annotations

if __name__ == "__main__":  # run as a script: put tools/ and its feature folders on the import path first
    import os as _os
    import sys as _sys

    _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
    import tazuna_paths  # noqa: F401

import math
import time

import cv2
import numpy as np
import numpy.ma  # noqa: F401  (np.median / np.percentile import it lazily: 20 ms inside the first frame otherwise)

import gripper_model as GM
import kinematics as K
from fdtypes import Scene, SceneObject

# --------------------------------------------------------------------------- parameters
DEFAULTS = {
    # table area (base frame, metres) in which objects are reported: x_min, x_max, y_min, y_max
    "workspace": [0.09, 0.45, -0.24, 0.24],
    "obj_min_mm": 12.0,          # smaller blobs are noise
    "obj_max_mm": 80.0,          # larger loose blobs are reported as warnings, not as objects
    "container_min_mm": 65.0,    # a dark box top at least this long AND wide is a container
    "container_floor_mm": 5.0,   # assumed thickness of a container's floor (cannot be seen)
    "container_wall_mm": 2.5,    # assumed wall thickness (opening = outer rim - 2 walls)
    "container_height_mm": 30.0,  # used ONLY when the rim height cannot be measured (noted on the object)
    "arm_margin_mm": 12.0,       # the arm capsules are grown by this much
    "min_table_fraction": 0.12,  # of the workspace pixels; below this the frame is refused
    "table_v_min": 70.0,         # table brightness (0..255) outside this range -> warning (saved frames: 129..140)
    "table_v_max": 172.0,
    "detect_scale": 0.5,         # detection runs on a half-resolution copy
    # table colour family (OpenCV HSV, hue in 0..179 re-centred on red so that red = 0)
    "table_hue": [-26, 10],
    # rows: saturation from (inclusive), V/Vtable min, V/Vtable max ; a pixel below the first row's
    # saturation or above TABLE_S_MAX is never table
    # (glare tops out at V/Vtable 1.5 with S >= 21; the shaded side of a white object is S 10..21 at
    # V/Vtable 1.6..1.75 and must stay outside)
    "table_band": [[14, 1.10, 1.52], [28, 0.78, 1.60], [60, 0.55, 1.50], [100, 0.42, 1.45],
                   [130, 0.30, 1.30], [175, 0.30, 1.10]],
    "table_s_max": 200,
    "dark_v": 0.66, "dark_s": 90, "black_v": 0.30,   # container interior: V/Vtable < dark_v and S < dark_s
    # things inside a container: strongly coloured (S >= content_s, V/Vtable >= content_v_color) or light
    "content_s": 100, "content_v_color": 0.60, "content_v_light": 0.95,
}

COLOR_NAMES = ("red", "blue", "green", "yellow", "white", "ivory", "black", "other")
_SQ3 = np.ones((3, 3), np.uint8)


def _params(cfg: dict | None) -> dict:
    p = dict(DEFAULTS)
    user = (cfg or {}).get("scene") or {}
    unknown = [k for k in user if k not in DEFAULTS]
    if unknown:
        raise ValueError(f"cfg['scene'] has unknown keys {unknown}; known keys: {sorted(DEFAULTS)}")
    p.update(user)
    p["container_floor_given"] = "container_floor_mm" in user   # the user's own value (not the default guess)
    return p


def _ell(n: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (n, n))


def _bbox_pad(mask: np.ndarray, pad: int):
    """Slices of the bounding box of a mask's non-zero pixels, grown by pad (None: the mask is empty)."""
    x, y, w, h = cv2.boundingRect(mask)
    if w == 0 or h == 0:
        return None
    return (slice(max(0, y - pad), min(mask.shape[0], y + h + pad)),
            slice(max(0, x - pad), min(mask.shape[1], x + w + pad)))


def _dilate_ell(mask: np.ndarray, n: int) -> np.ndarray:
    """cv2.dilate(mask, _ell(n)), worked out only around the non-zero pixels (the same result, faster)."""
    out = np.zeros_like(mask)
    sl = _bbox_pad(mask, n // 2 + 1)
    if sl is not None:
        out[sl] = cv2.dilate(mask[sl], _ell(n))
    return out


def _open_ell(mask: np.ndarray, n: int) -> np.ndarray:
    """cv2.morphologyEx(mask, cv2.MORPH_OPEN, _ell(n)), worked out only around the non-zero pixels."""
    out = np.zeros_like(mask)
    sl = _bbox_pad(mask, n // 2 + 1)
    if sl is not None:
        out[sl] = cv2.morphologyEx(mask[sl], cv2.MORPH_OPEN, _ell(n))
    return out


def _min_channel(bgr: np.ndarray) -> np.ndarray:
    """The smallest colour channel of every pixel (bgr.min(axis=2), several times faster)."""
    b, g, r = cv2.split(np.ascontiguousarray(bgr))
    return cv2.min(cv2.min(b, g), r)


def _max_channel(bgr: np.ndarray) -> np.ndarray:
    """The largest colour channel of every pixel (bgr.max(axis=2), several times faster)."""
    b, g, r = cv2.split(np.ascontiguousarray(bgr))
    return cv2.max(cv2.max(b, g), r)


# --------------------------------------------------------------------------- camera helpers
class _FastCam:
    """numpy projection / back-projection of a camera.CameraModel (same maths, vectorised)."""

    def __init__(self, cam):
        self.cam = cam
        self.R = np.asarray(cam.R, dtype=float)
        self.t = np.asarray(cam.tvec, dtype=float).reshape(3)
        self.f, self.cx, self.cy, self.k1 = float(cam.f), float(cam.cx), float(cam.cy), float(cam.k1)
        self.C = -self.R.T @ self.t
        kmat = np.array([[self.f, 0.0, self.cx], [0.0, self.f, self.cy], [0.0, 0.0, 1.0]])
        self._PA = (kmat @ self.R).T.copy()      # uvw = X @ _PA + _Pb
        self._Pb = kmat @ self.t

    def project(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        if not self.k1:
            uvw = pts @ self._PA + self._Pb
            w = uvw[:, 2:3]
            return uvw[:, :2] / np.where(np.abs(w) < 1e-9, 1e-9, w)
        pc = pts @ self.R.T + self.t
        z = np.where(np.abs(pc[:, 2]) < 1e-9, 1e-9, pc[:, 2])
        x, y = pc[:, 0] / z, pc[:, 1] / z
        k = 1.0 + self.k1 * (x * x + y * y)          # same radial term as cv2.projectPoints
        return np.stack([self.f * x * k + self.cx, self.f * y * k + self.cy], axis=1)

    def depth(self, pts) -> np.ndarray:
        pts = np.asarray(pts, dtype=float).reshape(-1, 3)
        return (pts @ self.R.T + self.t)[:, 2]

    def rays(self, uv) -> np.ndarray:
        uv = np.asarray(uv, dtype=float).reshape(-1, 2)
        if self.k1:
            p = cv2.undistortPoints(uv.reshape(-1, 1, 2), self.cam.K, self.cam.dist).reshape(-1, 2)
        else:
            p = (uv - (self.cx, self.cy)) / self.f
        return np.column_stack([p, np.ones(len(p))]) @ self.R  # rows are R^T d

    def to_plane(self, uv, z: float) -> np.ndarray:
        """Intersections (N x 3) of the pixel rays with the plane at height z (NaN where there is none)."""
        d = self.rays(uv)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (z - self.C[2]) / d[:, 2]
        s = np.where(s > 0, s, np.nan)
        return self.C + s[:, None] * d

    def px_per_m(self, p3) -> float:
        """Image pixels per metre for a small horizontal step across the viewing direction at p3."""
        p3 = np.asarray(p3, dtype=float)
        away = p3[:2] - self.C[:2]
        n = np.linalg.norm(away)
        side = np.array([-away[1], away[0], 0.0]) / (n if n > 1e-9 else 1.0)
        a, b = self.project(np.array([p3 - 0.005 * side, p3 + 0.005 * side]))
        return float(np.linalg.norm(a - b) / 0.01)


def _check_image(img, cam):
    if img is None or getattr(img, "ndim", 0) != 3 or img.shape[2] != 3:
        raise ValueError("scene: expected a BGR image (H x W x 3)")
    h, w = img.shape[:2]
    if abs(w - 2.0 * cam.cx) > 2.0 or abs(h - 2.0 * cam.cy) > 2.0:
        raise ValueError(f"scene: image is {w}x{h} but the camera model was calibrated for "
                         f"{int(round(2 * cam.cx))}x{int(round(2 * cam.cy))}")


# --------------------------------------------------------------------------- table colour family
_HUE_RED = np.where(np.arange(180) > 90, np.arange(180) - 180, np.arange(180)).astype(np.int16)


def _table_lut(vt: float, p: dict) -> np.ndarray:
    """256 x 256 boolean table indexed [S, V]: True where (S, V) belongs to the table family."""
    s = np.arange(256)[:, None]
    v = np.arange(256, dtype=np.float32)[None, :] / max(vt, 1.0)
    lut = np.zeros((256, 256), bool)
    band = p["table_band"]
    for i, (s0, lo, hi) in enumerate(band):
        s1 = band[i + 1][0] if i + 1 < len(band) else p["table_s_max"] + 1
        lut |= (s >= s0) & (s < s1) & (v >= lo) & (v <= hi)
    return lut


def _hue_ok(h: np.ndarray, p: dict) -> np.ndarray:
    hr = _HUE_RED[h]
    return (hr >= p["table_hue"][0]) & (hr <= p["table_hue"][1])


def _planes(hsv) -> tuple:
    """(H, S, V) as separate contiguous planes (an HSV image, or such a tuple already)."""
    return hsv if isinstance(hsv, tuple) else tuple(cv2.split(np.ascontiguousarray(hsv)))


_RED_WOOD_HUE = np.zeros(256, np.uint8)
_RED_WOOD_HUE[:10] = 255
_RED_WOOD_HUE[160:] = 255


def _table_reference(hsv, where: np.ndarray) -> tuple[float, float]:
    """Median V of clearly table-coloured pixels inside `where`, and their share of `where`.
    hsv: an HSV image or its three planes (_planes)."""
    # red wood: hue within -20..+9 of red (the lower side leaves room for bluer daylight under the
    # locked white balance), medium saturation
    h, s, v = _planes(hsv)
    wm = np.ascontiguousarray(where).view(np.uint8) if where.dtype == np.bool_ else where
    n_where = int(cv2.countNonZero(wm))
    if n_where == 0:
        return 0.0, 0.0
    sel = cv2.bitwise_and(cv2.LUT(h, _RED_WOOD_HUE), cv2.inRange(s, 80, 170), mask=wm)
    hist = cv2.calcHist([v], [0], sel, [256], [0, 256]).ravel().astype(np.int64)
    n = int(hist.sum())
    if n == 0:
        return 0.0, 0.0
    cum = np.cumsum(hist)                                       # the median as numpy defines it, from the histogram
    lo, hi = int(np.searchsorted(cum, (n - 1) // 2 + 1)), int(np.searchsorted(cum, n // 2 + 1))
    return 0.5 * (lo + hi), n / n_where


_TABLE_TABS: dict = {}


def _table_tabs(vt: float, p: dict):
    """The table colour family as three 256-entry tables: for every S the interval [lo, hi] of V that
    belongs to it, and the hues that do (255 / 0). (None, None, None) if the family is not one
    V-interval per S (then the full table has to be looked up pixel by pixel)."""
    key = (float(vt), repr(p["table_band"]), float(p["table_s_max"]), tuple(p["table_hue"]))
    tabs = _TABLE_TABS.get(key)
    if tabs is None:
        lut = _table_lut(vt, p)
        rows = lut.any(axis=1)
        lo = np.where(rows, lut.argmax(axis=1), 1).astype(np.uint8)
        hi = np.where(rows, 255 - lut[:, ::-1].argmax(axis=1), 0).astype(np.uint8)
        v = np.arange(256)[None, :]
        if np.array_equal(lut, (v >= lo[:, None]) & (v <= hi[:, None])):
            hue = np.zeros(256, np.uint8)
            hue[:180] = np.where(_hue_ok(np.arange(180), p), 255, 0)
            tabs = (lo, hi, hue)
        else:
            tabs = (None, None, None)
        if len(_TABLE_TABS) > 32:
            _TABLE_TABS.clear()
        _TABLE_TABS[key] = tabs
    return tabs


def _nontable(hsv, vt: float, p: dict) -> np.ndarray:
    """True where a pixel is not of the table colour family. hsv: an HSV image or its three planes."""
    h, s, v = _planes(hsv)
    lo, hi, hue = _table_tabs(vt, p)
    if lo is None:
        return ~(_table_lut(vt, p)[s, v] & _hue_ok(h, p))
    tab = cv2.bitwise_and(cv2.compare(v, cv2.LUT(s, lo), cv2.CMP_GE), cv2.compare(v, cv2.LUT(s, hi), cv2.CMP_LE))
    return cv2.bitwise_and(tab, cv2.LUT(h, hue)) == 0


def _dark(hsv, vt: float, p: dict) -> np.ndarray:
    """True where a pixel is as dark as the inside of a container. hsv: an HSV image or its three planes."""
    # ((V < dark_v * vt) & (S < dark_s)) | (V < black_v * vt), on whole numbers: x < t  <=>  x <= ceil(t) - 1
    _, s, v = _planes(hsv)
    v1, s1, v2 = math.ceil(p["dark_v"] * vt) - 1, math.ceil(p["dark_s"]) - 1, math.ceil(p["black_v"] * vt) - 1
    return cv2.bitwise_or(cv2.bitwise_and(cv2.inRange(v, 0, v1), cv2.inRange(s, 0, s1)), cv2.inRange(v, 0, v2)) > 0


# --------------------------------------------------------------------------- colour naming
def color_name(bgr_pixels: np.ndarray, vt: float) -> tuple[str, str]:
    """Name the colour of an object from its pixels (N x 3, BGR). Returns (name, detail).

    name is one of COLOR_NAMES; detail is a finer description (e.g. "orange") for the notes.
    The hue is taken from the more saturated 40 % of the pixels: lit faces of light objects clip to
    pure white in the camera and carry no colour, the shaded faces do (an ivory part is white on
    top and yellowish on its side). Brightness is judged relative to the table brightness vt, so
    the name does not change with the exposure."""
    px = np.asarray(bgr_pixels, dtype=np.uint8).reshape(-1, 1, 3)
    if len(px) == 0:
        return "other", "no pixels"
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)
    v_rel = float(np.percentile(hsv[:, 2], 80)) / max(vt, 1.0)     # brightness of the lit part
    s_all = float(np.median(hsv[:, 1]))
    order = np.argsort(hsv[:, 1])
    sat = hsv[order[int(0.6 * len(order)):]]
    s_hi = float(np.median(sat[:, 1]))
    ang = sat[:, 0] * (math.pi / 90.0)
    hue = (math.degrees(math.atan2(float(np.sin(ang).mean()), float(np.cos(ang).mean()))) % 360.0) / 2.0  # 0..180
    detail = f"H{hue:.0f} S{s_hi:.0f} V{v_rel:.2f}xTable"
    if v_rel < 0.62 and s_all < 110:
        return "black", detail
    if s_hi < 24:
        if v_rel < 1.15:
            return "other", "gray " + detail
        return "white", detail
    if s_hi < 100 and v_rel >= 1.15:
        if 10 <= hue <= 40:
            return "ivory", detail
        if s_hi < 45:
            return "white", "tinted " + detail
    if hue <= 8 or hue >= 168:
        return "red", detail
    if hue < 19:
        return "other", "orange " + detail
    if hue < 36:
        return "yellow", detail
    if hue < 88:
        return "green", detail
    if hue < 132:
        return "blue", detail
    if hue < 150:
        return "other", "purple " + detail
    return "other", "pink " + detail


# --------------------------------------------------------------------------- robot mask
def _gf_point(T_gf: np.ndarray, x_mm: float, y_mm: float, z_mm: float) -> np.ndarray:
    """Base-frame point of a gripper_frame_link coordinate given in millimetres."""
    return T_gf[:3, 3] + T_gf[:3, :3] @ (np.array([x_mm, y_mm, z_mm]) / 1000.0)


def nose_points_gf(jaw_q_deg: float) -> dict:
    """Distal ends (noses) of both jaws in gripper_frame_link coordinates, millimetres.
    Uses gripper_model.nose_points when the KIN owner has added it, else the same closed form
    (fixed nose (2.0, 0, 6.298); moving nose = the flat end centred at (-17.9, 7.27) for q = 0)."""
    fn = getattr(GM, "nose_points", None)
    if fn is not None:
        try:
            d = fn(float(jaw_q_deg))
            return {"fixed": tuple(float(v) for v in d["fixed"]), "moving": tuple(float(v) for v in d["moving"])}
        except Exception:
            pass
    mx, mz = GM.jaw_point(float(jaw_q_deg), (-17.9, 7.27))
    return {"fixed": (2.0, 0.0, GM.FIXED_TIP_Z), "moving": (mx, 0.124, mz)}


def arm_capsules(arm_q, jaw_q_deg: float | None = None) -> list:
    """Capsules [(p0, p1, r0, r1), ...] (base frame, metres) that enclose the arm and gripper."""
    q = np.asarray(arm_q, dtype=float).reshape(5)
    j = [f[:3, 3] for f in K.link_frames(q)]
    T = K.fk(q)  # gripper_frame_link
    caps = [                                     # (the fixed base is covered by the keep-out box in arm_mask)
        (j[0], j[1], 0.058, 0.055),                                                 # shoulder
        (j[1], j[2], 0.047, 0.047),                                                 # upper arm
        (j[2], j[3], 0.045, 0.045),                                                 # forearm
        (j[3], j[4], 0.045, 0.045),                                                 # wrist
        (j[4], _gf_point(T, -5.0, 0.0, -58.0), 0.047, 0.042),                       # gripper body + servo
        (_gf_point(T, 10.0, 0.0, -61.0), _gf_point(T, 2.0, 0.0, 6.3), 0.020, 0.005),  # fixed finger
    ]
    pivot = _gf_point(T, GM.PIVOT_X, 0.0, GM.PIVOT_Z)
    if jaw_q_deg is None:
        angles = np.arange(-12.0, 61.0, 8.0)  # unknown opening: cover the usual range
    else:
        angles = (jaw_q_deg - 5.0, jaw_q_deg, jaw_q_deg + 5.0)
    for a in angles:
        mx, mz = GM.jaw_point(float(a), (-19.0, 7.3))
        caps.append((pivot, _gf_point(T, mx, 0.0, mz), 0.026, 0.005))
    return caps


def arm_mask(shape_hw, cam, arm_q, jaw_q_deg: float | None = None, scale: float = 1.0,
             margin_m: float = 0.012, keepout: list | None = None) -> np.ndarray:
    """uint8 mask (255 = robot) of size shape_hw for an image that is `scale` times the full frame."""
    return _arm_masks(shape_hw, cam, arm_q, jaw_q_deg, scale, (margin_m,), keepout)[0]


def _arm_masks(shape_hw, cam, arm_q, jaw_q_deg, scale: float, margins, keepout: list | None = None) -> list:
    """arm_mask for several safety margins at once (the geometry is worked out once)."""
    fc = cam if isinstance(cam, _FastCam) else _FastCam(cam)
    masks = [np.zeros(shape_hw[:2], np.uint8) for _ in margins]
    for p0, p1, r0, r1 in arm_capsules(arm_q, jaw_q_deg):
        n = max(2, int(np.linalg.norm(p1 - p0) / 0.012) + 1)
        tt = np.linspace(0.0, 1.0, n)
        pts = p0[None, :] + tt[:, None] * (p1 - p0)[None, :]
        z = fc.depth(pts)
        if np.any(z < 0.05):
            continue
        uv = (fc.project(pts) * scale).tolist()
        ok = [math.isfinite(u) and math.isfinite(v) and abs(u) <= 1e5 and abs(v) <= 1e5 for u, v in uv]
        cs = [(int(round(u)), int(round(v))) if good else None for (u, v), good in zip(uv, ok)]
        for m, margin_m in zip(masks, margins):
            rpx = ((r0 + tt * (r1 - r0) + margin_m) * fc.f / z * scale).tolist()
            prev = None
            for c, r in zip(cs, rpx):
                if c is None:
                    prev = None
                    continue
                cv2.circle(m, c, max(1, int(round(r))), 255, -1)
                if prev is not None:
                    cv2.line(m, prev[0], c, 255, max(1, int(round(2 * min(r, prev[1])))))
                prev = (c, r)
    # fixed base: the keep-out box around the base column, as a filled hull
    kb = keepout or [-0.12, 0.085, -0.075, 0.075, -0.05, 0.13]
    xs, ys, zs = (kb[0], kb[1]), (kb[2] - 0.01, kb[3] + 0.01), (max(kb[4], -0.03), kb[5])
    corners = np.array([[x, y, z] for x in xs for y in ys for z in zs])
    if np.all(fc.depth(corners) > 0.05):
        hull = cv2.convexHull((fc.project(corners) * scale).astype(np.float32)).astype(np.int32)
        for m in masks:
            cv2.fillConvexPoly(m, hull, 255)
    return masks


# --------------------------------------------------------------------------- geometry of primitives
def box_corners(x, y, yaw, length, width, height, z0) -> np.ndarray:
    """8 corners; 0-3 bottom, 4-7 top; length along yaw (same convention as vision.box_corners)."""
    c, s = math.cos(yaw), math.sin(yaw)
    ex, ey = np.array([c, s, 0.0]), np.array([-s, c, 0.0])
    base = np.array([x, y, z0])
    out = []
    for dz in (0.0, height):
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            out.append(base + sx * length / 2 * ex + sy * width / 2 * ey + np.array([0.0, 0.0, dz]))
    return np.array(out)


_RING = np.linspace(0.0, 2.0 * math.pi, 28, endpoint=False)
_RING_CS = np.stack([np.cos(_RING), np.sin(_RING)], axis=1)


def cyl_points(x, y, r, h, z0, top_only: bool = False) -> np.ndarray:
    """Points on the top circle (and the base circle unless top_only) of an upright cylinder."""
    n = len(_RING_CS)
    out = np.empty((n if top_only else 2 * n, 3))
    out[:n, 0] = x + r * _RING_CS[:, 0]
    out[:n, 1] = y + r * _RING_CS[:, 1]
    out[:n, 2] = z0 + h
    if not top_only:
        out[n:, :2] = out[:n, :2]
        out[n:, 2] = z0
    return out


def _hull_mask(fc: _FastCam, pts3, shape_hw, off, out: np.ndarray | None = None) -> np.ndarray:
    """Filled convex hull of projected 3-D points, drawn into a crop whose origin is `off`
    (`out` = a buffer of that shape to reuse)."""
    uv = fc.project(pts3) - off
    if out is None:
        m = np.zeros(shape_hw, np.uint8)
    else:
        m = out
        m.fill(0)
    q = uv * 8.0
    if not (np.abs(q).max() < 8e5):       # also catches NaN
        return m
    hull = cv2.convexHull(np.rint(q).astype(np.int32))
    cv2.fillConvexPoly(m, hull, 255, lineType=cv2.LINE_8, shift=3)
    return m


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = cv2.countNonZero(cv2.bitwise_and(a, b))
    union = cv2.countNonZero(cv2.bitwise_or(a, b))
    return inter / union if union else 0.0


def _nelder_mead(fn, x0, steps, iters: int = 160, tol: float = 2e-4):
    """Small Nelder-Mead (same algorithm as vision.nelder_mead, kept local to avoid an import cycle)."""
    n = len(x0)
    sim = [np.array(x0, dtype=float)]
    for i in range(n):
        q = np.array(x0, dtype=float)
        q[i] += steps[i]
        sim.append(q)
    vals = [fn(q) for q in sim]
    for _ in range(iters):
        order = np.argsort(vals)
        sim = [sim[i] for i in order]
        vals = [vals[i] for i in order]
        if abs(vals[-1] - vals[0]) < tol:
            break
        cen = np.mean(sim[:-1], axis=0)
        xr = cen + (cen - sim[-1])
        fr = fn(xr)
        if fr < vals[0]:
            xe = cen + 2.0 * (cen - sim[-1])
            fe = fn(xe)
            sim[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            sim[-1], vals[-1] = xr, fr
        else:
            xc = cen + 0.5 * (sim[-1] - cen)
            fc_ = fn(xc)
            if fc_ < vals[-1]:
                sim[-1], vals[-1] = xc, fc_
            else:
                for i in range(1, n + 1):
                    sim[i] = sim[0] + 0.5 * (sim[i] - sim[0])
                    vals[i] = fn(sim[i])
    i = int(np.argmin(vals))
    return sim[i], vals[i]


def _fill(mask: np.ndarray) -> np.ndarray:
    """Largest outer contour of a binary mask, filled (removes holes and small satellites)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = np.zeros_like(mask)
    if cnts:
        cv2.drawContours(out, [max(cnts, key=cv2.contourArea)], -1, 255, -1)
    return out


def _low_edge(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per column: (has_any, row of the lowest set pixel)."""
    any_ = mask.any(axis=0)
    rows = mask.shape[0] - 1 - np.argmax(mask[::-1, :] > 0, axis=0)
    return any_, rows


def _top_edge(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    any_ = mask.any(axis=0)
    rows = np.argmax(mask > 0, axis=0)
    return any_, rows


def _edges_lr(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per row: (has_any, leftmost set column, rightmost set column)."""
    any_ = mask.any(axis=1)
    left = np.argmax(mask > 0, axis=1)
    right = mask.shape[1] - 1 - np.argmax(mask[:, ::-1] > 0, axis=1)
    return any_, left, right


def _med(a) -> float:
    """Median without numpy's NaN bookkeeping (fast; the callers pass finite values)."""
    a = np.asarray(a, dtype=float).ravel()
    n = len(a)
    if n == 0:
        return float("nan")
    k = n // 2
    part = np.partition(a, k)
    return float(part[k]) if n % 2 else 0.5 * (float(part[k]) + float(np.max(part[:k])))


def _tmean(a) -> float:
    """Mean of the middle half of the values (robust like a median, but not quantised to pixels)."""
    a = np.sort(np.asarray(a, dtype=float).ravel())
    n = len(a)
    if n == 0:
        return float("nan")
    if n < 4:
        return float(a.mean())
    return float(a[n // 4: n - n // 4].mean())


# --------------------------------------------------------------------------- one object: top face, fits
def _split_top(crop_bgr: np.ndarray, M: np.ndarray) -> tuple[np.ndarray | None, dict]:
    """Top face of an object blob = its bright part (the top is lit from above; side faces are darker
    and more saturated). Uses the minimum colour channel (the 'white' share of a pixel), split by Otsu
    on the blob interior. Returns (mask or None when there is no clear two-face structure, info)."""
    info = {"valid": False}
    area_m = cv2.countNonZero(M)
    if area_m < 200:
        info["why"] = "blob too small"
        return None, info
    minch = _min_channel(crop_bgr)
    inner = cv2.erode(M, _ell(7))  # boundary pixels are mixtures with the table
    vals = minch[inner > 0]
    if len(vals) < 100:
        info["why"] = "blob too thin"
        return None, info
    thr, _ = cv2.threshold(vals.reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lo, hi = vals[vals <= thr], vals[vals > thr]
    if len(lo) == 0 or len(hi) == 0:
        info["why"] = "uniform brightness"
        return None, info
    m0, m1 = float(lo.mean()), float(hi.mean())
    info.update(thr=float(thr), m_dark=m0, m_bright=m1)
    if (m1 - m0) < 0.085 * max(m1, 1.0) or (m1 - m0) < 6.0:
        info["why"] = "faces not distinguishable (saturated or uniform)"
        return None, info
    T = ((M > 0) & (minch > thr)).astype(np.uint8) * 255
    T = cv2.morphologyEx(T, cv2.MORPH_OPEN, _ell(5))
    T = _fill(T)
    T = cv2.morphologyEx(T, cv2.MORPH_CLOSE, _ell(5))
    area_t = cv2.countNonZero(T)
    frac = area_t / float(area_m)
    info["frac"] = frac
    if not (0.18 <= frac <= 0.86):
        info["why"] = f"bright part is {frac:.0%} of the blob"
        return None, info
    ym = np.nonzero(M)[0].mean()
    yt = np.nonzero(T)[0].mean()
    if yt > ym - 1.0:
        info["why"] = "bright part is not the upper part"
        return None, info
    info["valid"] = True
    return T, info


def _bisect(fn, lo: float, hi: float, iters: int = 22) -> float:
    """Root of a decreasing function fn on [lo, hi] (clamped to the ends when there is no sign change)."""
    flo, fhi = fn(lo), fn(hi)
    if not (flo > 0.0):
        return lo
    if not (fhi < 0.0):
        return hi
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if fn(mid) > 0.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _fit_cylinder(fc: _FastCam, M: np.ndarray, off, z0: float) -> dict | None:
    """Upright cylinder standing on z0 whose outline matches the blob M (crop with origin off).

    The outline of a cylinder shows all four unknowns separately: the two straight sides give the
    diameter and the sideways position, the lower arc the distance, the upper arc the height. Each
    is measured as a robust mean over many boundary pixels (so a reflection tab or a stray pixel
    cannot pull it) and applied as a correction; three or four rounds converge."""
    ys, xs = np.nonzero(M)
    if len(xs) < 150:
        return None
    u0, u1, v0, v1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
    sel = ys >= v1 - 2
    ub = float(np.median(xs[sel]))
    low = fc.to_plane([[ub + off[0], v1 + 0.5 + off[1]]], z0)[0]
    if not np.all(np.isfinite(low)):
        return None
    away = low[:2] - fc.C[:2]
    away = away / max(np.linalg.norm(away), 1e-9)
    side = np.array([-away[1], away[0]])
    r = 0.5 * (u1 - u0 + 1.0) / fc.px_per_m(low)
    c = low[:2] + away * r
    far = np.array([c[0] + away[0] * r, c[1] + away[1] * r, 0.0])

    def top_row(hh):  # image row of the far point of the top circle minus the observed top row
        far[2] = z0 + hh
        return float(fc.project(far[None, :])[0, 1] - off[1] - v0)

    h = _bisect(top_row, 0.002, 0.15, iters=14)
    # tight window of the blob (plus a margin): every rasterisation is cheap
    pad = 14
    x_lo, y_lo = max(0, int(u0) - pad), max(0, int(v0) - pad)
    x_hi, y_hi = min(M.shape[1], int(u1) + pad + 1), min(M.shape[0], int(v1) + pad + 1)
    Mw = np.ascontiguousarray(M[y_lo:y_hi, x_lo:x_hi])
    off_w = np.array([off[0] + x_lo, off[1] + y_lo], dtype=float)
    buf = np.zeros_like(Mw)
    o_has_c, o_low = _low_edge(Mw)
    _, o_top = _top_edge(Mw)
    o_has_r, o_left, o_right = _edges_lr(Mw)
    cols = np.nonzero(o_has_c)[0]
    k = max(1, int(0.2 * len(cols)))
    cols = cols[k:-k] if len(cols) > 2 * k + 4 else cols

    def raster(x, y, rr, hh):
        return _hull_mask(fc, cyl_points(x, y, rr, hh, z0), None, off_w, out=buf)

    x, y = float(c[0]), float(c[1])
    worst = float("inf")
    for _ in range(5):
        p0 = np.array([x, y, z0 + 0.5 * h])
        # local image scale: pixels per metre sideways / away from the camera (on the table) / upwards
        pr = fc.project(np.array([p0, p0 + [side[0] * 0.002, side[1] * 0.002, 0.0],
                                  [x, y, z0], [x + away[0] * 0.002, y + away[1] * 0.002, z0],
                                  [x, y, z0 + h], [x, y, z0 + h + 0.002]]))
        du_side = (pr[1, 0] - pr[0, 0]) / 0.002
        dv_away = (pr[3, 1] - pr[2, 1]) / 0.002
        dv_up = (pr[5, 1] - pr[4, 1]) / 0.002
        if abs(du_side) < 50 or abs(dv_away) < 50 or abs(dv_up) < 50:
            return None
        m = raster(x, y, r, h)
        m_has_r, m_left, m_right = _edges_lr(m)
        # rows where the outline is the two straight sides: between the two circle centres
        vb, vt_ = pr[2, 1] - off_w[1], pr[4, 1] - off_w[1]
        ra, rb = int(math.ceil(min(vb, vt_))) + 1, int(math.floor(max(vb, vt_)))
        if rb - ra < 4:
            mid = 0.5 * (vb + vt_)
            ra, rb = int(mid) - 3, int(mid) + 4
        rows = np.arange(max(ra, 0), min(rb, Mw.shape[0]))
        ok = o_has_r[rows] & m_has_r[rows]
        if np.count_nonzero(ok) < 3:
            return None
        rows = rows[ok]
        dl = _tmean(o_left[rows] - m_left[rows])
        dr = _tmean(o_right[rows] - m_right[rows])
        r_new = r + 0.5 * (dr - dl) / abs(du_side)
        shift = 0.5 * (dl + dr) / du_side
        x, y, r = x + side[0] * shift, y + side[1] * shift, max(r_new, 0.003)
        m = raster(x, y, r, h)
        m_has_c, m_low = _low_edge(m)
        okc = m_has_c[cols]
        if np.count_nonzero(okc) < 5:
            return None
        db = _tmean(o_low[cols][okc] - m_low[cols][okc])
        x, y = x + away[0] * db / dv_away, y + away[1] * db / dv_away
        m = raster(x, y, r, h)
        m_has_c, m_top = _top_edge(m)
        okc = m_has_c[cols]
        if np.count_nonzero(okc) < 5:
            return None
        dt = _tmean(o_top[cols][okc] - m_top[cols][okc])
        h = max(h + dt / dv_up, 0.002)
        worst = max(abs(dl), abs(dr), abs(db), abs(dt))
        if worst < 0.25:
            break
    m = raster(x, y, r, h)
    return {"x": x, "y": y, "r": float(r), "h": float(h), "iou": _iou(m, Mw), "resid_px": float(worst)}


def _robust_rect(xy: np.ndarray) -> dict | None:
    """Rectangle around 2-D boundary points (metres): orientation from the minimum-area rectangle,
    extents from the 3rd / 97th percentiles along its axes so that a few stray points do not inflate it.
    Returns centre, long-axis yaw in [0, pi), length >= width."""
    if len(xy) < 8 or not np.all(np.isfinite(xy)):
        return None
    (cx, cy), (w, h), ang = cv2.minAreaRect((xy * 1000.0).astype(np.float32))
    if w < 1e-6 or h < 1e-6:
        return None
    a = math.radians(ang)
    ex, ey = np.array([math.cos(a), math.sin(a)]), np.array([-math.sin(a), math.cos(a)])
    d = xy - np.array([cx, cy]) / 1000.0
    px, py = d @ ex, d @ ey
    x_lo, x_hi = np.percentile(px, [3, 97])
    y_lo, y_hi = np.percentile(py, [3, 97])
    c = np.array([cx, cy]) / 1000.0 + ex * 0.5 * (x_lo + x_hi) + ey * 0.5 * (y_lo + y_hi)
    lx, ly = float(x_hi - x_lo), float(y_hi - y_lo)
    if lx >= ly:
        length, width, yaw = lx, ly, a
    else:
        length, width, yaw = ly, lx, a + math.pi / 2.0
    return {"x": float(c[0]), "y": float(c[1]), "yaw": float(yaw % math.pi), "length": length, "width": width}


def _contour_pts(mask: np.ndarray, off) -> np.ndarray:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.zeros((0, 2))
    c = max(cnts, key=cv2.contourArea).reshape(-1, 2).astype(float)
    return c + np.asarray(off, dtype=float)


def _height_from_faces(fc: _FastCam, M: np.ndarray, T: np.ndarray, off, z0: float) -> tuple[float, float, int] | None:
    """Height of a prism from the vertical offset between the near edges of its top face (lower edge
    of T) and the ground-contact edges straight below them (lower edge of M).
    Returns (height, spread_px, n_columns) or None."""
    has_t, low_t = _low_edge(T)
    has_m, low_m = _low_edge(M)
    cols = np.nonzero(has_t)[0]
    if len(cols) < 12:
        return None
    trim = max(2, int(0.10 * len(cols)))
    cols = cols[trim:-trim]
    uv_t = np.column_stack([cols + off[0], low_t[cols] + 0.5 + off[1]]).astype(float)
    d = fc.rays(uv_t)                      # the top-edge point is C + s*d with s fixed by its height
    if np.any(d[:, 2] >= -1e-9):
        return None
    w = M.shape[1]
    off_a = np.asarray(off, dtype=float)
    base = np.empty((len(cols), 3))
    base[:, 2] = z0

    def resid(h):
        s = (z0 + h - fc.C[2]) / d[:, 2]
        base[:, 0] = fc.C[0] + s * d[:, 0]
        base[:, 1] = fc.C[1] + s * d[:, 1]
        uv_b = fc.project(base) - off_a     # the point of the ground edge straight below
        cu = np.round(uv_b[:, 0]).astype(int)
        ok = (cu >= 0) & (cu < w)
        cu = np.clip(cu, 0, w - 1)
        ok &= has_m[cu]
        return (low_m[cu] + 0.5 - uv_b[:, 1])[ok]

    def med(h):
        r = resid(h)
        return _med(r) if len(r) else -1.0

    h = _bisect(med, 0.002, 0.15, iters=14)
    r = resid(h)
    if len(r) < 8:
        return None
    spread = _med(np.abs(r - _med(r)))
    return float(h), spread, int(len(r))


def _fit_box_top(fc: _FastCam, M: np.ndarray, T: np.ndarray, off, z0: float) -> dict | None:
    """Box standing on z0 from its top face T and its whole blob M (see the module docstring)."""
    hh = _height_from_faces(fc, M, T, off, z0)
    if hh is None:
        return None
    h, spread, n = hh
    xy = fc.to_plane(_contour_pts(T, off) + 0.0, z0 + h)[:, :2]
    rect = _robust_rect(xy)
    if rect is None:
        return None
    corners = box_corners(rect["x"], rect["y"], rect["yaw"], rect["length"], rect["width"], h, z0)
    sil = _hull_mask(fc, corners, M.shape, off)
    top = _hull_mask(fc, corners[4:], M.shape, off)
    rect.update(h=h, iou=_iou(sil, M), iou_top=_iou(top, T), h_spread_px=spread, h_cols=n)
    return rect


def _fit_box_sil(fc: _FastCam, M: np.ndarray, off, z0: float) -> dict | None:
    """Box from the outline only (used when the faces cannot be told apart). Width and height trade
    off against each other in this fit, so the caller must treat them as uncertain."""
    ys, xs = np.nonzero(M)
    if len(xs) < 150:
        return None
    pad = 14
    x_lo, y_lo = max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad)
    x_hi, y_hi = min(M.shape[1], int(xs.max()) + pad + 1), min(M.shape[0], int(ys.max()) + pad + 1)
    Mw = np.ascontiguousarray(M[y_lo:y_hi, x_lo:x_hi])
    off_w = np.array([off[0] + x_lo, off[1] + y_lo], dtype=float)
    cnt = _contour_pts(Mw, off_w)
    if len(cnt) < 20:
        return None
    hull = cv2.convexHull(cnt.astype(np.float32)).reshape(-1, 1, 2)
    ap = cv2.approxPolyDP(hull, 0.02 * cv2.arcLength(hull, True), True).reshape(-1, 2)
    if len(ap) < 3:
        return None
    low = ap[np.argsort(-ap[:, 1])[:2]]
    a3, b3 = fc.to_plane(low, z0)
    if not (np.all(np.isfinite(a3)) and np.all(np.isfinite(b3))):
        return None
    mid = 0.5 * (a3 + b3)
    edge = max(float(np.linalg.norm(a3 - b3)), 0.012)
    edge_yaw = math.atan2(b3[1] - a3[1], b3[0] - a3[0])
    away = mid[:2] - fc.C[:2]
    away = away / max(np.linalg.norm(away), 1e-9)
    area_m = cv2.countNonZero(Mw)
    buf, tmp = np.zeros_like(Mw), np.zeros_like(Mw)

    def cost(q):
        if q[3] < 0.008 or q[4] < 0.008 or q[5] < 0.004:
            return 1.0
        m = _hull_mask(fc, box_corners(q[0], q[1], q[2], q[3], q[4], q[5], z0), None, off_w, out=buf)
        inter = cv2.countNonZero(cv2.bitwise_and(m, Mw, dst=tmp))
        union = cv2.countNonZero(m) + area_m - inter
        return 1.0 - inter / union if union else 1.0

    guess = min(edge, 0.04)
    best = None
    for yaw0, l0, w0, depth in ((edge_yaw, edge, 0.7 * guess, 0.7 * guess), (edge_yaw + math.pi / 2, guess, edge, guess)):
        init = [mid[0] + away[0] * depth / 2, mid[1] + away[1] * depth / 2, yaw0, l0, w0, 0.7 * guess]
        q, c = _nelder_mead(cost, init, [0.006, 0.006, 0.12, 0.006, 0.006, 0.006], iters=70, tol=5e-4)
        if best is None or c < best[1]:
            best = (q, c)
    q, c = _nelder_mead(cost, best[0], [0.002, 0.002, 0.04, 0.002, 0.002, 0.002], iters=70, tol=2e-4)
    length, width, yaw = float(q[3]), float(q[4]), float(q[2])
    if width > length:
        length, width, yaw = width, length, yaw + math.pi / 2.0
    return {"x": float(q[0]), "y": float(q[1]), "yaw": yaw % math.pi, "length": length, "width": width,
            "h": float(q[5]), "iou": 1.0 - float(c)}


def _rough_extent(fc: _FastCam, M: np.ndarray, off, z0: float) -> dict | None:
    """Upright bounding cylinder of a blob that could not be modelled (no fitting, no claims)."""
    ys, xs = np.nonzero(M)
    if len(xs) == 0:
        return None
    u0, u1, v0, v1 = float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max())
    low = fc.to_plane([[0.5 * (u0 + u1) + off[0], v1 + 0.5 + off[1]]], z0)[0]
    if not np.all(np.isfinite(low)):
        return None
    away = low[:2] - fc.C[:2]
    away = away / max(np.linalg.norm(away), 1e-9)
    r = 0.5 * (u1 - u0 + 1.0) / fc.px_per_m(low)
    c = low[:2] + away * r
    far = np.array([c[0] + away[0] * r, c[1] + away[1] * r, 0.0])

    def top_row(h):
        far[2] = z0 + h
        return float(fc.project(far[None, :])[0, 1] - off[1] - v0)

    h = _bisect(top_row, 0.002, 0.15)
    return {"x": float(c[0]), "y": float(c[1]), "r": float(r), "h": float(h)}


def _poly_ccw(pts) -> list:
    pts = [(float(p[0]), float(p[1])) for p in pts]
    area = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1] for i in range(len(pts)))
    return pts if area >= 0 else pts[::-1]


def _circle_poly(x, y, r, n: int = 16) -> list:
    return [(x + r * math.cos(2 * math.pi * i / n), y + r * math.sin(2 * math.pi * i / n)) for i in range(n)]


_HUE_GROUP_NAMES = ((-11, 10, "red"), (10, 38, "yellow/orange"), (38, 88, "green"), (88, 132, "blue"),
                    (132, 169, "purple/pink"))


def _hue_groups(crop_bgr: np.ndarray, mask: np.ndarray) -> list:
    """Names of the clearly different hues that each cover at least 20 % of the strongly coloured
    pixels of a blob (one object has one hue; two touching objects of different colour have two)."""
    px = crop_bgr[mask > 0]
    if len(px) < 200:
        return []
    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    strong = hsv[hsv[:, 1] >= 110]
    if len(strong) < 0.3 * len(px):
        return []
    h = strong[:, 0].astype(np.int16)
    out = []
    for lo, hi, name in _HUE_GROUP_NAMES:
        if name == "red":
            share = np.mean((h <= hi) | (h >= 180 + lo))
        else:
            share = np.mean((h > lo) & (h <= hi))
        if share >= 0.2:
            out.append(name)
    return out


def _model_blob(fc: _FastCam, crop_bgr: np.ndarray, M: np.ndarray, off, z_bottom: float, vt: float,
                outline_fit: bool = True) -> dict:
    """Decide what the blob M is (box / cylinder / unknown) and measure it.
    Returns a dict with kind, center, yaw, size, footprint, confidence, notes, color, pixel_bbox.
    outline_fit=False skips the (slow) outline-only box search for blobs without visible faces."""
    notes = []
    ys, xs = np.nonzero(M)
    bbox = (int(xs.min() + off[0]), int(ys.min() + off[1]), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    core = cv2.erode(M, _ell(7))       # boundary pixels are mixed with the table colour
    if cv2.countNonZero(core) < 60:
        core = M
    color, detail = color_name(crop_bgr[core > 0], vt)
    if color == "other":
        notes.append("colour: " + detail)
    out = {"pixel_bbox": bbox, "color": color, "color_detail": detail, "notes": notes}
    T, tinfo = _split_top(crop_bgr, M)
    box = _fit_box_top(fc, M, T, off, z_bottom) if T is not None else None
    cyl = None
    # A box whose top face is matched this well cannot be a cylinder (a circle fills at most
    # pi/4 = 0.79 of its bounding rectangle), so the cylinder fit is skipped.
    sure_box = box is not None and box["iou_top"] >= 0.90 and box["iou"] >= 0.88
    if not sure_box:
        cyl = _fit_cylinder(fc, M, off, z_bottom)
    kind = None
    if sure_box:
        kind = "box"
    elif cyl is not None and cyl["iou"] >= 0.90 and (box is None or cyl["iou"] >= box["iou"] + 0.03):
        kind = "cylinder"
    elif box is not None and box["iou"] >= 0.84 and box["iou_top"] >= 0.84 and (cyl is None or box["iou"] >= cyl["iou"] + 0.03):
        kind = "box"
    outline_only = None
    if kind is None and T is None:
        # no visible face structure and not a clean cylinder: a box from the outline only
        bs = _fit_box_sil(fc, M, off, z_bottom) if outline_fit else None
        if bs is not None and bs["iou"] >= 0.90 and (cyl is None or bs["iou"] >= cyl["iou"] + 0.015):
            outline_only = bs
        elif cyl is not None and cyl["iou"] >= 0.88:
            kind = "cylinder"
    out["fits"] = {"cyl": cyl, "box": box, "top": tinfo, "outline_box": outline_only}
    flat_note = ("only {:.0f} mm high: too flat to grasp. If this is really the top of a taller object whose "
                 "sides have the table's colour, its true position is farther from the camera")
    if kind == "box":
        s = min(box["iou"], box["iou_top"])
        conf = float(np.clip((s - 0.78) / 0.14, 0.05, 1.0))
        if box.get("h_spread_px", 0.0) > 5.0:
            conf = min(conf, 0.5)
            notes.append(f"height uncertain (edge offsets scatter {box['h_spread_px']:.0f} px)")
        if box["h"] < 0.005:
            conf = min(conf, 0.3)
            notes.append(flat_note.format(box["h"] * 1000))
        corners = box_corners(box["x"], box["y"], box["yaw"], box["length"], box["width"], 0.0, 0.0)[:4, :2]
        out.update(kind="box", center=(box["x"], box["y"]), yaw=box["yaw"],
                   size=(box["length"], box["width"], box["h"]), footprint=_poly_ccw(corners), confidence=conf)
        return out
    if kind == "cylinder":
        conf = float(np.clip((cyl["iou"] - 0.86) / 0.08, 0.05, 1.0))
        if cyl["h"] < 0.005:
            conf = min(conf, 0.3)
            notes.append(flat_note.format(cyl["h"] * 1000))
        out.update(kind="cylinder", center=(cyl["x"], cyl["y"]), yaw=0.0,
                   size=(2 * cyl["r"], 2 * cyl["r"], cyl["h"]),
                   footprint=_circle_poly(cyl["x"], cyl["y"], cyl["r"]), confidence=conf)
        return out
    if outline_only is not None:
        # Not reported as a box: without distinguishable faces the height (and with it the distance)
        # is a guess. The numbers are the best outline match and must not be used for a grasp.
        bs = outline_only
        notes.append("its faces cannot be told apart (uniform or clipped brightness): the reported box is only the "
                     f"best match to the outline ({bs['iou']:.2f}); height and distance are unreliable")
        hues = _hue_groups(crop_bgr, core)
        if len(hues) >= 2:
            notes.append("looks like several things of different colour touching each other in the image ("
                         + ", ".join(hues) + "): separate them, or remove the nearer one and look again")
        corners = box_corners(bs["x"], bs["y"], bs["yaw"], bs["length"], bs["width"], 0.0, 0.0)[:4, :2]
        out.update(kind="unknown", center=(bs["x"], bs["y"]), yaw=bs["yaw"],
                   size=(bs["length"], bs["width"], bs["h"]), footprint=_poly_ccw(corners), confidence=0.25)
        return out
    ext = _rough_extent(fc, M, off, z_bottom)
    best = max([f["iou"] for f in (cyl, box) if f is not None] + [0.0])
    notes.append(f"shape not recognised as an upright box or cylinder (best outline match {best:.2f}); "
                 "the reported extent is a rough upright bounding cylinder")
    hues = _hue_groups(crop_bgr, core)
    if len(hues) >= 2:
        notes.append("looks like several things of different colour touching each other in the image ("
                     + ", ".join(hues) + "): separate them, or remove the nearer one and look again")
    if ext is None:
        ext = {"x": float("nan"), "y": float("nan"), "r": 0.0, "h": 0.0}
    out.update(kind="unknown", center=(ext["x"], ext["y"]), yaw=0.0, size=(2 * ext["r"], 2 * ext["r"], ext["h"]),
               footprint=_circle_poly(ext["x"], ext["y"], ext["r"]), confidence=0.2)
    return out


# --------------------------------------------------------------------------- frame-level segmentation
def _workspace_mask(fc: _FastCam, p: dict, z0: float, shape_hw, scale: float) -> np.ndarray:
    """Image region (uint8) in which objects standing inside the workspace rectangle can appear."""
    x0, x1, y0, y1 = p["workspace"]
    x1 = min(x1, float(fc.C[0]) - 0.10) if fc.C[0] > x0 + 0.2 else x1
    pts = np.array([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z0 + 0.09)])
    m = np.zeros(shape_hw[:2], np.uint8)
    if np.any(fc.depth(pts) < 0.03):
        m[:] = 255            # unusual camera placement: do not restrict
        return m
    uv = fc.project(pts) * scale
    uv = np.clip(uv, -4.0 * max(shape_hw), 4.0 * max(shape_hw))
    hull = cv2.convexHull(uv.astype(np.float32)).astype(np.int32)
    cv2.fillConvexPoly(m, hull, 255)
    return m


def _segment(img: np.ndarray, fc: _FastCam, p: dict, z0: float, cfg: dict, arm_q, jaw_q_deg) -> dict:
    """Half-resolution products shared by everything else."""
    s = float(p["detect_scale"])
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    planes = _planes(hsv)
    roi = _workspace_mask(fc, p, z0, small.shape, s)
    if arm_q is not None:
        ko = (cfg.get("keepouts") or {}).get("base_column")
        # with the safety margin, and without it: what the arm really covers (used for the box outline)
        arm, arm_tight = _arm_masks(small.shape, fc, arm_q, jaw_q_deg, s, (p["arm_margin_mm"] / 1000.0, 0.003), ko)
    else:
        arm = np.zeros(small.shape[:2], np.uint8)
        arm_tight = arm
    free = (roi > 0) & (arm == 0)
    vt, frac = _table_reference(planes, free)
    ctx = {"scale": s, "small": small, "hsv": hsv, "roi": roi, "arm": arm, "arm_tight": arm_tight, "vt": vt,
           "table_fraction": frac}
    if vt <= 0:
        return ctx
    nt_all = _nontable(planes, vt, p)
    dark_all = _dark(planes, vt, p)
    free_tight = (roi > 0) & (arm_tight == 0)
    # for containers: not table, not brighter than a box's rim or walls (the white arm in front of a
    # box is excluded by its brightness, so only the tight arm mask is needed there)
    boxish = nt_all & free_tight & (planes[2] < 0.85 * vt)
    ctx.update(nontable=(nt_all & free).astype(np.uint8) * 255, dark=(dark_all & free_tight).astype(np.uint8) * 255,
               boxish=boxish.astype(np.uint8) * 255)
    return ctx


def _full_pts(pts_small: np.ndarray, s: float) -> np.ndarray:
    """Pixel-centre coordinates of a reduced image -> full-resolution coordinates."""
    return (np.asarray(pts_small, dtype=float) + 0.5) / s - 0.5


def _densify(poly: np.ndarray, step: float = 3.0) -> np.ndarray:
    out = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        k = max(1, int(np.linalg.norm(b - a) / step))
        for j in range(k):
            out.append(a + (b - a) * (j / k))
    return np.array(out)


# --------------------------------------------------------------------------- containers
def _rim_height(img: np.ndarray, fc: _FastCam, hull: np.ndarray, vt: float, z0: float, p: dict,
                arm_s: np.ndarray | None = None, scale: float = 1.0, fracs=None) -> dict:
    """Rim height of an open box from its far inner wall (finding VM-0).

    hull: outline (N x 2, full-res pixels) of the box top; arm_s: robot mask at `scale` (column bands
    in which the arm stands in front of the wall are skipped). In each of several column bands the row
    where the dark inner wall starts (inner rim edge, height H) and the row of the wall/floor step
    (height = floor thickness) are located; both lie on the same vertical wall, so the two viewing
    rays give the wall's inner height directly. H = inner height + assumed floor thickness.
    fracs: where the bands are, as fractions of the outline's width (default: 11 bands from 0.2 to 0.8)."""
    t_floor = p["container_floor_mm"] / 1000.0
    out = {"measured": False, "bands": [], "floor_m": t_floor}
    u_lo, u_hi = float(hull[:, 0].min()), float(hull[:, 0].max())
    v_lo, v_hi = float(hull[:, 1].min()), float(hull[:, 1].max())
    height_px = v_hi - v_lo
    if u_hi - u_lo < 60 or height_px < 60:
        out["why"] = "opening too small in the image"
        return out
    x0, x1 = int(max(0, u_lo)), int(min(img.shape[1], u_hi + 1))
    y0, y1 = int(max(0, v_lo - 12)), int(min(img.shape[0], v_lo + 0.8 * height_px))
    hsv = cv2.cvtColor(img[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    val = hsv[..., 2].astype(np.float32)
    sat = hsv[..., 1].astype(np.float32)
    inside = np.zeros(val.shape, np.uint8)
    cv2.fillConvexPoly(inside, cv2.convexHull(np.round(hull - (x0, y0)).astype(np.int32)), 255)
    hs = []
    n_solid = 0
    half = 8
    thr_step = max(1.5, 0.012 * vt)      # brightness levels per row in the smoothed profile
    # with bands of the caller's choosing (a box turned on the table, seen past the arm): not where the
    # outline is low, i.e. next to its left and right corners, where the wall does not reach the floor
    min_rows = 40 if fracs is None else max(40, int(0.55 * cv2.countNonZero(inside.max(axis=1))))
    for frac in (np.linspace(0.2, 0.8, 11) if fracs is None else fracs):
        uc = int(round(u_lo + frac * (u_hi - u_lo))) - x0
        if uc - half < 0 or uc + half >= val.shape[1]:
            continue
        rows_in = np.nonzero(inside[:, uc])[0]
        if len(rows_in) < min_rows:
            continue
        prof = val[:, uc - half:uc + half].mean(axis=1)
        sprof = sat[:, uc - half:uc + half].mean(axis=1)
        sm = cv2.GaussianBlur(prof.reshape(-1, 1), (1, 5), 0).ravel()
        band = {"u": float(uc + x0)}
        out["bands"].append(band)
        # inner rim edge: where the profile falls below the dark level just under the box outline
        a = max(0, int(rows_in[0]) - 4)
        below = np.nonzero(sm[a:a + 30] < 0.45 * vt)[0]
        if len(below) == 0:
            band["rejected"] = "no dark wall under the rim"
            continue
        r0 = a + int(below[0])
        # wall/floor step: the FIRST sharp brightness change further down. Sharp, because the wall
        # itself shades gradually; first, because a later (stronger) edge is an object on the floor.
        d = np.diff(sm)
        lo, hi = r0 + 10, min(len(d) - 12, r0 + int(0.62 * height_px))
        if hi - lo < 6:
            band["rejected"] = "wall too short"
            continue
        hit = np.nonzero(np.abs(d[lo:hi]) >= thr_step)[0]
        if len(hit) == 0:
            band["rejected"] = "no wall/floor step"
            continue
        ra = lo + int(hit[0])
        r1 = ra + int(np.argmax(np.abs(d[ra:ra + 6]))) + 1
        wall = float(prof[max(r0 + 3, r1 - 12):r1 - 2].mean()) if r1 - 2 > r0 + 3 else float(prof[r0 + 3])
        floor = float(prof[r1 + 3:r1 + 12].mean())
        floor_s = float(sprof[r1 + 3:r1 + 12].mean())
        band.update(r_rim=r0 + y0, r_step=r1 + y0, wall_v=wall, floor_v=floor)
        if arm_s is not None:
            ca, cb = int((uc - half + x0) * scale), int((uc + half + x0) * scale) + 1
            ra_, rb_ = int((r0 - 3 + y0) * scale), int((r1 + 12 + y0) * scale) + 1
            if np.any(arm_s[max(0, ra_):rb_, max(0, ca):cb] > 0):
                band["rejected"] = "the arm stands in front of the wall here"
                continue
        if floor > 0.80 * vt or floor_s > 125:
            band["rejected"] = "something stands in front of the wall here"
            continue
        if floor < wall:
            # brighter above, darker below = the lit top face of a SOLID dark block, not a cavity
            band["rejected"] = "brighter above the step than below"
            n_solid += 1
            continue
        # both pixels are on the inner far wall: same horizontal position along the view direction
        pb = fc.to_plane([[band["u"], r1 + y0 + 0.5]], z0 + t_floor)[0]
        dtop = fc.rays([[band["u"], r0 + y0 + 0.5]])[0]
        if not np.all(np.isfinite(pb)):
            band["rejected"] = "no ray/floor intersection"
            continue
        e = pb[:2] - fc.C[:2]
        e = e / max(np.linalg.norm(e), 1e-9)
        den = float(e @ dtop[:2])
        if abs(den) < 1e-9:
            band["rejected"] = "degenerate ray"
            continue
        h_band = float(fc.C[2] + float(e @ (pb[:2] - fc.C[:2])) / den * dtop[2] - z0)
        band["H"] = h_band
        if 0.006 <= h_band - t_floor <= 0.15:
            hs.append(h_band)
        else:
            band["rejected"] = "implausible height"
    out["solid"] = n_solid >= 3 and n_solid > len(hs)
    if len(hs) < 3:
        out["why"] = f"the wall/floor step was found in only {len(hs)} of {len(out['bands'])} column bands"
        return out
    # Something dark standing against the wall can only make the visible wall SHORTER, so among the
    # groups of bands that agree with each other (within 2 mm) the tallest one is the wall itself.
    hs = np.sort(np.array(hs))[::-1]
    inl = None
    for h_top in hs:
        grp = hs[(hs <= h_top) & (hs >= h_top - 0.002)]
        if len(grp) >= 3:
            inl = grp
            break
    if inl is None:
        out["why"] = f"the column bands disagree (no 3 of {len(hs)} within 2 mm of each other)"
        return out
    # ... and for the same reason two bands that agree with each other (within 1 mm) on a clearly taller
    # wall are the wall, and the larger group below them is something standing against it. Two bands
    # are thin evidence: the result is then marked as not consistent (lower confidence).
    taller = hs[hs > inl.max() + 0.002]
    for h_top in taller:
        grp = taller[(taller <= h_top) & (taller >= h_top - 0.001)]
        if len(grp) >= 2:
            inl = grp
            break
    out.update(H=float(inl.mean()), spread_m=float(inl.max() - inl.min()), n=int(len(inl)), n_bands=int(len(hs)),
               consistent=bool(len(inl) >= 3 and len(inl) >= 0.6 * len(hs)))
    out["measured"] = True
    return out


def _outer_wall_height(img: np.ndarray, fc: _FastCam, quad: np.ndarray, vt: float, z0: float, h_guess: float,
                       arm_s: np.ndarray | None = None, scale: float = 1.0, wide: bool = False) -> dict:
    """Rim height of a box from its near OUTER walls: the distance in the image between the outer rim
    edge and the foot of the wall on the table, straight below it.

    quad: the rim outline (4 x 2, full-res pixels, in order); h_guess: where to look for the foot
    (the search covers 0.45 .. 1.7 times that height, 0.3 .. 3 times with wide=True). For several
    places along every side that faces the camera, the pixels below the rim are followed along the
    image of a vertical line: the wall is dull (little colour), the table strongly coloured, and the
    step between them is the foot. Rim pixel and foot pixel lie on one vertical, so their two viewing
    rays give the height above the table directly; no thickness has to be assumed.
    Measured only when the places agree: {"measured", "H", "n", "spread_m"} or {"measured": False, "why"}."""
    out = {"measured": False, "n_tried": 0}
    h_img, w_img = img.shape[:2]
    plane = fc.to_plane(quad, z0 + h_guess)
    if not np.all(np.isfinite(plane)):
        out["why"] = "the rim cannot be placed on its plane"
        return out
    ctr = plane[:, :2].mean(axis=0)
    reach = 3.0 if wide else 1.7
    hs = []
    for i in range(4):
        a, b = plane[i, :2], plane[(i + 1) % 4, :2]
        mid = 0.5 * (a + b)
        e = b - a
        nout = np.array([e[1], -e[0]])
        nout = nout / max(float(np.linalg.norm(nout)), 1e-9)
        if float(nout @ (mid - ctr)) < 0.0:
            nout = -nout
        tocam = fc.C[:2] - mid
        if float(nout @ tocam) <= 0.2 * float(np.linalg.norm(tocam)):
            continue                           # this wall faces away from the camera, or is seen edge-on
        fs = np.linspace(0.12, 0.88, 9)
        prs = quad[i][None, :] + fs[:, None] * (quad[(i + 1) % 4] - quad[i])[None, :]
        xrs = fc.to_plane(prs, z0 + h_guess)                   # the rim points seen at these pixels
        feet = xrs.copy()
        feet[:, 2] = z0
        pbs = fc.project(np.nan_to_num(feet))
        dtops = fc.rays(prs)
        for pr, xr, pb, dtop in zip(prs, xrs, pbs, dtops):
            out["n_tried"] += 1
            if not np.all(np.isfinite(xr)):
                continue
            len0 = float(np.linalg.norm(pb - pr))
            if len0 < 20.0:
                continue
            dv = (pb - pr) / len0
            ls = np.arange(3.0, reach * len0)
            pts = pr[None, :] + ls[:, None] * dv[None, :]
            xi, yi = np.rint(pts[:, 0]).astype(int), np.rint(pts[:, 1]).astype(int)
            ok = (xi >= 2) & (xi < w_img - 2) & (yi >= 0) & (yi < h_img)
            n_ok = len(ok) if bool(ok.all()) else int(np.argmin(ok))
            if n_ok < 1.25 * len0:
                continue                       # the foot of the wall is outside the image
            xi, yi, ls = xi[:n_ok], yi[:n_ok], ls[:n_ok]
            if arm_s is not None and np.any(arm_s[np.clip((yi * scale).astype(int), 0, arm_s.shape[0] - 1),
                                                  np.clip((xi * scale).astype(int), 0, arm_s.shape[1] - 1)] > 0):
                continue
            px = (img[yi, xi - 2].astype(np.float32) + img[yi, xi] + img[yi, xi + 2]) / 3.0
            mx, mn = px.max(axis=1), px.min(axis=1)
            sat = 255.0 * (mx - mn) / np.maximum(mx, 1.0)
            k1 = max(8, int(0.35 * len0))
            s_wall, v_wall = _med(sat[3:k1]), _med(mx[3:k1])
            s_tab = _med(sat[-max(6, int(0.2 * len0)):])
            if s_wall > 85.0 or v_wall > 1.35 * vt or s_tab < s_wall + 30.0:
                continue                       # not a dull wall above a coloured table
            thr = s_wall + 0.5 * (s_tab - s_wall)
            start = int((0.3 if wide else 0.45) * len0)
            above = sat >= thr
            if len(above) < start + 6:
                continue
            run = above[start:-3] & above[start + 1:-2] & above[start + 2:-1] & above[start + 3:]
            hit = np.nonzero(run)[0]
            if len(hit) == 0:
                continue
            j = start + int(hit[0])
            lc = float(ls[j])
            if j > 0 and sat[j] > sat[j - 1]:
                lc += float(np.clip((thr - sat[j - 1]) / (sat[j] - sat[j - 1]), 0.0, 1.0)) - 1.0
            xc = fc.to_plane([pr + lc * dv], z0)[0]
            if not np.all(np.isfinite(xc)):
                continue
            ev = xc[:2] - fc.C[:2]
            dist = float(np.linalg.norm(ev))
            den = float((ev / max(dist, 1e-9)) @ dtop[:2])
            if abs(den) < 1e-9:
                continue
            h = float(fc.C[2] + dist / den * dtop[2] - z0)
            if 0.006 <= h <= 0.25:
                hs.append(h)
    out["hs"] = hs
    if len(hs) < 6:
        out["why"] = f"the foot of the near outer wall was found in only {len(hs)} of {out['n_tried']} places"
        return out
    hs = np.array(hs)
    inl = hs[np.abs(hs - _med(hs)) <= 0.0012]
    if len(inl) < 6 or len(inl) < 0.7 * len(hs):
        out["why"] = f"the places along the near outer wall disagree ({len(inl)} of {len(hs)} within 1.2 mm)"
        return out
    out.update(measured=True, H=float(inl.mean()), n=int(len(inl)), spread_m=float(inl.max() - inl.min()))
    return out


def _max_quad(hull: np.ndarray) -> np.ndarray | None:
    """The four vertices of a convex polygon that span the largest quadrilateral (in polygon order)."""
    pts = hull.astype(np.float32).reshape(-1, 1, 2)
    eps = 1.5
    ap = pts
    while len(ap) > 12 and eps < 40:
        ap = cv2.approxPolyDP(pts, eps, True)
        eps *= 1.5
    ap = ap.reshape(-1, 2).astype(float)
    n = len(ap)
    if n < 4:
        return None
    best, best_area = None, -1.0
    x, y = ap[:, 0], ap[:, 1]
    for i in range(n - 3):
        for j in range(i + 1, n - 2):
            for k in range(j + 1, n - 1):
                for m in range(k + 1, n):
                    area = abs((x[i] * y[j] - x[j] * y[i]) + (x[j] * y[k] - x[k] * y[j])
                               + (x[k] * y[m] - x[m] * y[k]) + (x[m] * y[i] - x[i] * y[m]))
                    if area > best_area:
                        best_area, best = area, (i, j, k, m)
    return ap[list(best)]


def _rect_from_quad(pts: np.ndarray) -> dict | None:
    """Rectangle (centre, long-axis yaw, length >= width) from four corner points given in order,
    plus how far the quadrilateral is from a rectangle."""
    e = np.roll(pts, -1, axis=0) - pts
    n = np.linalg.norm(e, axis=1)
    if np.any(n < 1e-6):
        return None
    a, b = 0.5 * (n[0] + n[2]), 0.5 * (n[1] + n[3])
    da, db = (e[0] - e[2]) / 2.0, (e[1] - e[3]) / 2.0
    cosang = max(abs(float(e[i] @ e[(i + 1) % 4]) / (n[i] * n[(i + 1) % 4])) for i in range(4))
    ratio = max(max(n[0], n[2]) / min(n[0], n[2]), max(n[1], n[3]) / min(n[1], n[3]))
    c = pts.mean(axis=0)
    if a >= b:
        length, width, d = a, b, da
    else:
        length, width, d = b, a, db
    return {"x": float(c[0]), "y": float(c[1]), "yaw": float(math.atan2(d[1], d[0]) % math.pi),
            "length": float(length), "width": float(width), "skew_cos": float(cosang), "side_ratio": float(ratio)}


def _line_cross(p1: np.ndarray, d1: np.ndarray, p2: np.ndarray, d2: np.ndarray) -> np.ndarray | None:
    """Intersection of two lines given as point + unit direction (None when they are nearly parallel)."""
    cross = float(d1[0] * d2[1] - d1[1] * d2[0])
    if abs(cross) < 0.17:                  # less than about 10 degrees apart
        return None
    r = p2 - p1
    return p1 + (float(r[0] * d2[1] - r[1] * d2[0]) / cross) * d1


def _outline_sides(cnt: np.ndarray, hull: np.ndarray, blocked: np.ndarray, min_len: float = 12.0) -> dict | None:
    """The four straight sides of a convex outline that may be partly hidden (the rim of a box).

    cnt: outline pixels (N x 2, coordinates of the reduced image) of a region whose true outline is a
    convex quadrilateral; hull: their convex hull (M x 2); blocked: uint8 mask of the reduced image,
    non-zero where an outline pixel is not the region's own edge (robot arm, border of the image or
    of the workspace).

    Every edge of the simplified hull is checked for outline pixels that run along it and are not
    blocked: a real side has them, a chord that the hull spans across a hidden corner or a notch has
    not. A line is fitted to the pixels of each real side and neighbouring lines are intersected,
    which gives every corner, also one that cannot be seen.

    Returns None when four sides cannot be established, else a dict: "corners" (4 x 2; side i runs
    from corner i to corner i + 1), "corner_seen" (the corner lies on the visible outline),
    "corner_blocked" (it lies in the blocked mask), "seen" (fraction of each side supported by its own
    outline pixels), "hidden" (fraction of each side inside the blocked mask), "length" (pixels),
    "chord" (length of the longest hull edge that the outline does not follow, pixels)."""
    hull = np.asarray(hull, dtype=np.float32).reshape(-1, 2)
    pts = np.asarray(cnt, dtype=float).reshape(-1, 2)
    if len(hull) < 4 or len(pts) < 40:
        return None
    poly = cv2.approxPolyDP(hull.reshape(-1, 1, 2), 2.0, True).reshape(-1, 2).astype(float)
    k = len(poly)
    if k < 4 or k > 14:
        return None
    hh, ww = blocked.shape
    xi = np.clip(np.rint(pts[:, 0]).astype(int), 0, ww - 1)
    yi = np.clip(np.rint(pts[:, 1]).astype(int), 0, hh - 1)
    free = blocked[yi, xi] == 0
    cand = []
    chord = 0.0
    for i in range(k):
        a, b = poly[i], poly[(i + 1) % k]
        length = float(np.linalg.norm(b - a))
        if length < 1e-6:
            continue
        d = (b - a) / length
        rel = pts - a
        t = rel @ d
        off = rel[:, 0] * d[1] - rel[:, 1] * d[0]
        sel = (np.abs(off) <= 2.5) & (t >= 0.0) & (t <= length) & free
        seen_len = min(length, 2.0 * len(np.unique((t[sel] // 2.0).astype(int))))
        if seen_len >= max(min_len, 0.25 * length):
            cand.append({"d": d, "sel": sel, "seen_len": seen_len})
        else:
            chord = max(chord, length)         # a hull edge the outline does not follow: something is cut off there

    def continues(u, v):                   # two hull edges that are one side (less than 8 degrees apart)
        return abs(float(u["d"][0] * v["d"][1] - u["d"][1] * v["d"][0])) < 0.14 and float(u["d"] @ v["d"]) > 0.0

    sides = []
    for c in cand:
        if sides and continues(sides[-1], c):
            sides[-1]["sel"] = sides[-1]["sel"] | c["sel"]
            sides[-1]["seen_len"] += c["seen_len"]
        else:
            sides.append(dict(c))
    if len(sides) > 1 and continues(sides[-1], sides[0]):
        sides[0]["sel"] = sides[0]["sel"] | sides[-1]["sel"]
        sides[0]["seen_len"] += sides[-1]["seen_len"]
        sides.pop()
    if len(sides) > 4:                     # short extra edges (a notch, a rounded corner): the four best supported
        keep = sorted(sorted(range(len(sides)), key=lambda j: -sides[j]["seen_len"])[:4])
        sides = [sides[j] for j in keep]
    if len(sides) != 4:
        return None
    lines = []
    for m in sides:
        q = pts[m["sel"]]
        t = q @ m["d"]
        core = q[(t >= float(t.min()) + 5.0) & (t <= float(t.max()) - 5.0)]     # corners are rounded: leave them out
        if len(core) >= 12:
            q = core
        vx, vy, x0, y0 = (float(v) for v in cv2.fitLine(q.astype(np.float32), cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())
        d = np.array([vx, vy])
        lines.append((np.array([x0, y0]), d if float(d @ m["d"]) >= 0.0 else -d))
    corners = []
    for i in range(4):
        c = _line_cross(lines[i - 1][0], lines[i - 1][1], lines[i][0], lines[i][1])
        if c is None or not np.all(np.isfinite(c)):
            return None
        corners.append(c)
    corners = np.array(corners)
    e = np.roll(corners, -1, axis=0) - corners
    e1 = np.roll(e, -1, axis=0)
    turn = e[:, 0] * e1[:, 1] - e[:, 1] * e1[:, 0]
    if not (np.all(turn > 0) or np.all(turn < 0)):
        return None                        # not convex
    area_q = 0.5 * abs(float(np.dot(corners[:, 0], np.roll(corners[:, 1], -1)) - np.dot(corners[:, 1], np.roll(corners[:, 0], -1))))
    area_h = float(cv2.contourArea(hull))
    if not (0.9 * area_h <= area_q <= 1.6 * area_h):
        return None                        # the sides do not enclose what is seen, or run far beyond it
    hull_i = hull.reshape(-1, 1, 2)
    seen_c, blocked_c = [], []
    for c in corners:
        seen_c.append(bool(cv2.pointPolygonTest(hull_i, (float(c[0]), float(c[1])), True) >= -3.5))
        cx, cy = int(round(float(c[0]))), int(round(float(c[1])))
        blocked_c.append(bool(not (0 <= cx < ww and 0 <= cy < hh) or blocked[cy, cx] > 0))
    mark = np.zeros((hh, ww), np.uint8)
    any_sel = sides[0]["sel"] | sides[1]["sel"] | sides[2]["sel"] | sides[3]["sel"]
    mark[yi[any_sel], xi[any_sel]] = 255
    x0, y0 = max(0, int(xi[any_sel].min()) - 3), max(0, int(yi[any_sel].min()) - 3)
    x1, y1 = min(ww, int(xi[any_sel].max()) + 4), min(hh, int(yi[any_sel].max()) + 4)
    mark[y0:y1, x0:x1] = cv2.dilate(mark[y0:y1, x0:x1], np.ones((5, 5), np.uint8))
    seen_f, hid_f, lens = [], [], []
    for i in range(4):
        a, b = corners[i], corners[(i + 1) % 4]
        length = float(np.linalg.norm(b - a))
        n = max(2, int(length / 2.0))
        sp = a[None, :] + np.linspace(0.0, 1.0, n)[:, None] * (b - a)[None, :]
        sx, sy = np.rint(sp[:, 0]).astype(int), np.rint(sp[:, 1]).astype(int)
        inside = (sx >= 0) & (sx < ww) & (sy >= 0) & (sy < hh)
        sxc, syc = np.clip(sx, 0, ww - 1), np.clip(sy, 0, hh - 1)
        seen = inside & (mark[syc, sxc] > 0)
        hid = ~seen & (~inside | (blocked[syc, sxc] > 0))
        seen_f.append(float(np.mean(seen)))
        hid_f.append(float(np.mean(hid)))
        lens.append(length)
    return {"corners": corners, "corner_seen": seen_c, "corner_blocked": blocked_c, "seen": seen_f, "hidden": hid_f,
            "length": lens, "chord": chord}


def _arm_split_groups(lab: np.ndarray, stats: np.ndarray, dark_count: np.ndarray, arm: np.ndarray) -> dict:
    """Dark regions that the robot arm may have cut apart: {label: [other labels to try together with it]}.
    Two pieces are candidates when both border the arm mask and lie within about 3 cm of each other in
    the image (the gripper is narrower than that); whether they really form one box is decided by
    the outline fitted to them."""
    ring = cv2.bitwise_and(_dilate_ell(arm, 7), cv2.bitwise_not(arm))
    touch = np.bincount(lab[ring > 0], minlength=len(dark_count))
    ids = [j for j in range(1, len(dark_count)) if touch[j] >= 8 and dark_count[j] >= 300]
    out = {}
    gap = 60
    for i in ids:
        xi, yi, wi, hi = (int(v) for v in stats[i, :4])
        mates = []
        for j in ids:
            if j == i:
                continue
            xj, yj, wj, hj = (int(v) for v in stats[j, :4])
            if xj <= xi + wi + gap and xi <= xj + wj + gap and yj <= yi + hi + gap and yi <= yj + hj + gap:
                mates.append(j)
        if mates:
            out[i] = mates
    return out


def _find_containers(img: np.ndarray, fc: _FastCam, ctx: dict, p: dict, z0: float, warnings: list) -> list:
    """Open boxes: large dark openings. Returns dicts ready to become SceneObjects (+ image polygons).

    What is used is what a dark open box always shows, whatever lies in it and however it is turned:
    its dark rim and inner walls. The outline of that dark region is a quadrilateral (the outer edge
    of the rim); its four sides are located separately (_outline_sides), so a corner behind the
    gripper is rebuilt from the two sides that meet there, and the box is accepted when its rim is
    seen along every side, not when enough of its floor is visible."""
    s = ctx["scale"]
    vt = ctx["vt"]
    dark = _open_ell(ctx["dark"], 5)
    # A box is one connected "not table" region (interior, rim, contents). Things lying in it can cut
    # its dark interior into several pieces, so the dark pixels are grouped by the not-table region
    # they belong to instead of being taken piece by piece.
    nt_open = cv2.bitwise_or(cv2.morphologyEx(ctx["boxish"], cv2.MORPH_OPEN, _SQ3), dark)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(nt_open, connectivity=8)
    dark_count = np.bincount(lab[dark > 0], minlength=n)
    hh, ww = dark.shape
    has_arm = bool(np.any(ctx["arm_tight"]))
    # where the outline of a region is not its own edge: the robot, the limits of the workspace and of the image
    blocked = ((ctx["arm_tight"] > 0) | (ctx["roi"] == 0)).astype(np.uint8) * 255
    blocked[0, :] = blocked[-1, :] = 255
    blocked[:, 0] = blocked[:, -1] = 255
    blocked = cv2.dilate(blocked, _ell(9))
    mates = _arm_split_groups(lab, stats, dark_count, ctx["arm_tight"]) if has_arm else {}
    table_like = ((ctx["nontable"] == 0) & (ctx["roi"] > 0) & (ctx["arm"] == 0)).astype(np.uint8)
    found = []
    wall = p["container_wall_mm"] / 1000.0
    h_default = p["container_height_mm"] / 1000.0
    c_min = p["container_min_mm"] / 1000.0
    x_lo, x_hi, y_lo, y_hi = p["workspace"]

    def build(quad, rebuilt, from_sides, sides, at_arm, n_arm, fill, hull, dark_i, bbox, labels):
        """A container from one outline quadrilateral (full-res pixels) of a candidate region:
        a dict, a warning text, or None."""
        x, y, w, h = bbox
        r0 = _rect_from_quad(fc.to_plane(quad, z0 + h_default)[:, :2])
        if r0 is None or not np.isfinite(r0["length"]) or r0["width"] < 0.7 * c_min or r0["length"] < c_min:
            return None
        if not (x_lo <= r0["x"] <= x_hi and y_lo <= r0["y"] <= y_hi):
            return None
        where = f"near x={r0['x']:.2f} y={r0['y']:+.2f}"
        notes = []
        conf_cap = 1.0
        if sides is None and fill < 0.40:
            # the sides could not be told apart: the rule that held before (a mostly visible interior)
            if n_arm >= 40:
                return (f"a dark box-like region {where} is partly hidden by the robot arm (fewer than four of its "
                        "sides can be made out) and is not reported as a container: move the arm away and look again")
            return None
        if at_arm and not from_sides:
            return (f"a dark box-like region {where} has {at_arm} corner(s) hidden by the robot arm and "
                    "is not reported as a container: move the arm away and look again")
        if sides is not None:
            # A box shows its rim along every side (except where the robot is in front of it) and
            # never the table inside the rim; how much of its floor is covered does not matter.
            if fill < 0.10:
                return None
            lens = np.array(sides["length"])
            seen, hid = np.array(sides["seen"]), np.array(sides["hidden"])
            open_len = float(((1.0 - hid) * lens).sum())
            if open_len < 0.35 * float(lens.sum()) or float((seen * lens).sum()) < 0.55 * open_len:
                return None
            qm = np.zeros((hh, ww), np.uint8)
            cv2.fillConvexPoly(qm, np.rint((quad + 0.5) * s - 0.5).astype(np.int32), 1)
            qm = cv2.erode(qm, _ell(9))
            n_in = cv2.countNonZero(qm)
            if n_in and cv2.countNonZero(cv2.bitwise_and(qm, table_like)) > 0.25 * n_in:
                return None     # the table shows through: a ring or a frame, not a box
            if rebuilt:
                by_arm = all(sides["corner_blocked"][k] for k in rebuilt)
                cause = "hidden by the robot arm" if by_arm else "not visible (covered by something)"
                support = min(min(seen[k - 1], seen[k]) for k in rebuilt)
                short = min(min(seen[k - 1] * lens[k - 1], seen[k] * lens[k]) for k in rebuilt)
                if len(rebuilt) > 1 or support < 0.2 or short < 25.0:
                    return (f"a dark box-like region {where} has {len(rebuilt)} corner(s) {cause} and too little "
                            "of the rim next to them is seen to rebuild them: it is not reported as a container"
                            + (": move the arm away and look again" if by_arm else ""))
                k = rebuilt[0]
                notes.append(f"one corner of the rim is {cause}: it was rebuilt from the two rim edges that meet "
                             f"there (seen over {100 * seen[k - 1]:.0f} % and {100 * seen[k]:.0f} % of their length); "
                             "centre and size are less certain")
                conf_cap = 0.7 if support >= 0.5 else (0.6 if support >= 0.3 else 0.5)
            weakest = int(np.argmin(seen + hid))
            if seen[weakest] + hid[weakest] < 0.5:
                notes.append(f"one rim edge is seen over only {100 * seen[weakest]:.0f} % of its length (something "
                             "bright lies against it?)")
                conf_cap = min(conf_cap, 0.7)
        rim = _rim_height(img, fc, quad if rebuilt else hull, vt, z0, p, arm_s=ctx["arm"], scale=s)
        if rim.get("solid"):
            return None         # a solid dark block (lit top above a darker side): a loose object
        second = False
        if not rim["measured"] and sides is not None:
            # Too few of the 11 bands show the wall (things stand against it, or the arm is above it):
            # look again in narrower steps and right up to the arm (its mask without the safety margin).
            again = _rim_height(img, fc, quad if rebuilt else hull, vt, z0, p, arm_s=ctx["arm_tight"], scale=s,
                                fracs=np.linspace(0.06, 0.94, 25))
            if again["measured"] and not again.get("solid"):
                rim, second = again, True
        # The near outer wall, where its foot on the table is in view: rim height above the table with
        # nothing assumed (None when the box stands too close to the lower image border, as it usually does).
        outer = {"measured": False}
        if sides is not None:
            try:
                outer = _outer_wall_height(img, fc, quad, vt, z0, rim["H"] if rim["measured"] else h_default,
                                           arm_s=ctx["arm"], scale=s, wide=not rim["measured"])
            except Exception as e:             # an extra measurement: never a reason to lose the container
                outer = {"measured": False, "why": f"{type(e).__name__}: {e}"}
        floor_m = rim["floor_m"]
        given = bool(p.get("container_floor_given"))            # a floor thickness the user put into the config
        floor_checked = given                                   # ... counts as known unless the image contradicts it
        how_floor = "as given in scene.container_floor_mm" if given else "ASSUMED"
        head = []
        if rim["measured"]:
            H = rim["H"]
            wall_in = H - rim["floor_m"]
            seen_in = (f"inner wall {1000 * wall_in:.1f} mm measured on the far inner wall ({rim['n']} column bands, "
                       f"spread {rim['spread_m'] * 1000:.1f} mm)")
            conf = 0.9
            seen_floor = outer["H"] - wall_in if outer["measured"] else None
            if seen_floor is not None and 0.0005 <= seen_floor <= 0.012 \
                    and not (given and abs(seen_floor - rim["floor_m"]) > 0.0015):
                H, floor_m, floor_checked = outer["H"], seen_floor, True
                head.append(f"rim height {H * 1000:.1f} mm measured on the near outer wall (rim edge to its foot on the "
                            f"table, {outer['n']} places, spread {outer['spread_m'] * 1000:.1f} mm); {seen_in}: the floor "
                            f"is {floor_m * 1000:.1f} mm thick")
                if abs(floor_m - rim["floor_m"]) > 0.0015:
                    head.append(f"(where the foot of the outer wall is not in view the floor is ASSUMED "
                                f"{rim['floor_m'] * 1000:.0f} mm thick and the same box comes out "
                                f"{abs(floor_m - rim['floor_m']) * 1000:.1f} mm "
                                f"{'higher' if floor_m < rim['floor_m'] else 'lower'}: measure the floor with a ruler and put "
                                "it into scene.container_floor_mm; a value given there is used in every view and the image "
                                "is checked against it)")
            else:
                head.append(f"rim height {H * 1000:.1f} mm = {seen_in} + floor thickness {rim['floor_m'] * 1000:.0f} mm "
                            + how_floor)
                if outer["measured"]:
                    floor_checked = False
                    head.append(f"the near outer wall gives a rim height of {outer['H'] * 1000:.1f} mm, which does not "
                                "fit the inner wall" + (f" plus the floor thickness given in the config (the image says "
                                                        f"{seen_floor * 1000:.1f} mm): the camera calibration or that value "
                                                        "is wrong" if given else ": one of the two is wrong"))
                    conf = 0.6
            if second:
                head.append("most of the far inner wall is hidden (by the robot arm or by things standing against "
                            "it): its height comes from the short pieces that are left")
                conf = min(conf, 0.8)
            if not rim["consistent"]:
                head.append(f"only {rim['n']} of {rim['n_bands']} column bands show this wall height (something dark "
                            "may stand against the far wall): the rim height is less certain")
                conf = min(conf, 0.6 if not floor_checked else 0.8)
        elif outer["measured"]:
            H = outer["H"]
            head.append(f"rim height {H * 1000:.1f} mm measured on the near outer wall (rim edge to its foot on the table, "
                        f"{outer['n']} places, spread {outer['spread_m'] * 1000:.1f} mm); the far inner wall could not be "
                        f"measured ({rim.get('why', 'unknown reason')}): floor thickness {rim['floor_m'] * 1000:.0f} mm "
                        + how_floor)
            conf = 0.8
        else:
            H = h_default
            head.append(f"rim height NOT measured ({rim.get('why', 'unknown reason')}); "
                        f"{H * 1000:.0f} mm assumed: touch the rim to confirm before releasing over it")
            conf = 0.4
        notes[0:0] = head
        conf = min(conf, conf_cap)
        on_plane = fc.to_plane(quad, z0 + H)[:, :2]
        rect = _rect_from_quad(on_plane)
        if rebuilt and rect is not None:
            # a rebuilt corner must be where the other three put the fourth corner of a rectangle
            k = rebuilt[0]
            fourth = on_plane[k - 1] + on_plane[(k + 1) % 4] - on_plane[(k + 2) % 4]
            off = float(np.linalg.norm(fourth - on_plane[k]))
            if off > 0.012:
                return (f"a dark box-like region {where} has a corner that cannot be seen, and the corner rebuilt "
                        f"from its rim edges is {off * 1000:.0f} mm away from where the other three corners put it: "
                        "it is not reported as a container")
        if rect is None or rect["skew_cos"] > 0.22 or rect["side_ratio"] > 1.2:
            alt = _robust_rect(fc.to_plane(_densify(hull), z0 + H)[:, :2])
            if alt is None or not rim["measured"] or rim["n"] < 5 or rebuilt:
                if x_lo + 0.04 <= r0["x"] <= x_hi - 0.04 and y_lo + 0.04 <= r0["y"] <= y_hi - 0.04:
                    return (f"a dark region {where} is neither a clean rectangular opening nor has a clearly "
                            "measurable inner wall: not taken as a container")
                return None
            notes.append("the outline of the box top is not a clean rectangle (partly hidden?): size and "
                         "centre are less certain")
            conf = min(conf, 0.5)
            rect = alt
        if not (x_lo <= rect["x"] <= x_hi and y_lo <= rect["y"] <= y_hi):
            return None
        if rect["width"] < c_min or rect["length"] < c_min:
            return None         # too small to be a container at its real height: a loose dark object
        if y + h >= hh - 1:
            notes.append("the box top is cut by the lower image border: its near edge is uncertain")
            conf = min(conf, 0.5)
        length, width = rect["length"], rect["width"]
        in_l, in_w = max(length - 2 * wall, 0.01), max(width - 2 * wall, 0.01)
        inner = {"center": (rect["x"], rect["y"]), "yaw": rect["yaw"], "size": (in_l, in_w),
                 "floor_z": z0 + floor_m, "rim_z": z0 + H}
        corners = box_corners(rect["x"], rect["y"], rect["yaw"], length, width, H, z0)
        uv = fc.project(corners)
        bx0, by0 = max(0, int(uv[:, 0].min())), max(0, int(uv[:, 1].min()))
        bx1, by1 = min(img.shape[1] - 1, int(uv[:, 0].max())), min(img.shape[0] - 1, int(uv[:, 1].max()))
        inner_uv = fc.project(box_corners(rect["x"], rect["y"], rect["yaw"], in_l, in_w, 0.0, z0 + H)[:4])
        color, detail = color_name(ctx["small"][y:y + h, x:x + w][dark_i[y:y + h, x:x + w] > 0], vt)
        if color != "black":
            notes.append("colour: " + detail)
        notes.append(f"opening = outer rim - 2 x {wall * 1000:.1f} mm wall (assumed)")
        return {"kind": "container", "color": color, "center": (rect["x"], rect["y"]), "yaw": rect["yaw"],
                "size": (length, width, H), "z_bottom": z0, "footprint": _poly_ccw(corners[:4, :2]),
                "pixel_bbox": (bx0, by0, bx1 - bx0 + 1, by1 - by0 + 1), "confidence": conf, "inner": inner,
                "notes": notes, "outer_uv": uv, "inner_uv": inner_uv, "rim": rim, "quad_uv": quad,
                "sides": sides, "labels": list(labels), "outer": outer, "floor_checked": floor_checked}


    def attempt(labels: list):
        """One candidate region (one label, or several pieces around the arm).
        Returns a container dict, a warning text, or None."""
        sel = (lab == labels[0]) if len(labels) == 1 else np.isin(lab, labels)
        dark_i = (sel & (dark > 0)).astype(np.uint8)
        area = int(cv2.countNonZero(dark_i))
        if area < 1500:
            return None
        x, y, w, h = cv2.boundingRect(dark_i)
        if w < 30 or h < 30:
            return None
        if x <= 0 or x + w >= ww or y <= 0:
            return None         # cut by the left / right / upper image border: outside the usable view
        # The outline wanted is the OUTER edge of the rim. The rim's top surface is dark gray rather
        # than black, and an inner wall next to a bright object can be lighter than "dark", so the box
        # top is everything dim and not table-coloured of this region within a few millimetres of the
        # bounding box of its dark pixels (the limit keeps a lit outer wall out).
        pad = 8
        xa, ya, xb, yb = max(0, x - pad), max(0, y - pad), min(ww, x + w + pad), min(hh, y + h + pad)
        comp = sel[ya:yb, xa:xb].astype(np.uint8) * 255
        comp = cv2.morphologyEx(comp, cv2.MORPH_CLOSE, _ell(7))
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if len(labels) == 1:
            cnt = max(cnts, key=cv2.contourArea).reshape(-1, 2) + (xa, ya)
        else:
            cnt = np.vstack([c.reshape(-1, 2) for c in cnts if cv2.contourArea(c) >= 40.0]) + (xa, ya)
        hull_s = cv2.convexHull(cnt.astype(np.int32)).reshape(-1, 2)
        if len(hull_s) < 4:
            return None
        hull = _full_pts(hull_s, s)
        quad = _max_quad(hull)
        if quad is None:
            return None
        # The sides of the outline, each from its own pixels (_outline_sides). The corners of the convex
        # outline are used as they are (as before) unless one of them stands at the robot arm or the
        # outline has a cut; then the corners are the intersections of the sides, which also gives a
        # corner that cannot be seen. If that does not make a box, the convex outline is tried after all.
        sides = _outline_sides(cnt, hull_s, blocked)
        at_arm = 0
        if has_arm:
            for cu, cv_ in (quad + 0.5) * s - 0.5:
                xi, yi = int(round(cu)), int(round(cv_))
                if np.any(ctx["arm_tight"][max(0, yi - 4):yi + 5, max(0, xi - 4):xi + 5] > 0):
                    at_arm += 1
        fill = area / max(cv2.contourArea(hull_s.astype(np.float32)), 1.0)
        trials = []
        if sides is not None and (at_arm or sides["chord"] >= 14.0):
            trials.append((_full_pts(sides["corners"], s), [k for k in range(4) if not sides["corner_seen"][k]], True))
        trials.append((quad, [], False))
        why = None
        n_arm = int(np.count_nonzero(ctx["arm"][cnt[:, 1], cnt[:, 0]])) if has_arm else 0     # outline pixels at the arm
        for quad_t, rebuilt_t, from_sides in trials:
            res = build(quad_t, rebuilt_t, from_sides, sides, at_arm, n_arm, fill, hull, dark_i, (x, y, w, h), labels)
            if isinstance(res, dict):
                return res
            if why is None:
                why = res
        return why

    used = set()
    order = sorted(range(1, n), key=lambda j: -int(dark_count[j]))
    for i in order:
        if i in used or int(dark_count[i]) < 300:
            continue
        res = None
        if i in mates:
            group = [i] + [j for j in mates[i] if j not in used]
            if len(group) > 1 and sum(int(dark_count[j]) for j in group) >= 1500:
                res = attempt(group)
                if isinstance(res, dict):
                    used.update(group)
                    res["notes"].append("the robot arm stands in front of it and divides its image")
                    res["confidence"] = min(res["confidence"], 0.7)
        if not isinstance(res, dict):
            if int(dark_count[i]) < 1500:
                continue
            res = attempt([i])
        if isinstance(res, dict):
            found.append(res)
        elif isinstance(res, str):
            warnings.append(res)
    return found


def _content_classes(hsv: np.ndarray, vt: float, p: dict) -> np.ndarray:
    """Per pixel inside a container: 0 = container itself (dark), 1 = light (white / ivory / gray),
    2.. = hue classes. Used to separate things that touch each other in the image."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2].astype(np.float32)
    hr = _HUE_RED[h]
    chroma = (s >= p["content_s"]) & (v >= p["content_v_color"] * vt)
    cls = np.zeros(h.shape, np.uint8)
    cls[(v >= p["content_v_light"] * vt) & ~chroma] = 1
    hue_cls = np.full(h.shape, 2, np.uint8)             # red
    hue_cls[(hr > 10) & (hr <= 38)] = 3                 # orange / yellow
    hue_cls[(hr > 38) & (hr <= 88)] = 4                 # green
    hue_cls[(hr > 88) | (hr < -48)] = 5                 # blue  (hue 88..132 is hr > 88 or hr < -48)
    hue_cls[(hr >= -48) & (hr < -11)] = 6               # purple / pink
    cls[chroma] = hue_cls[chroma]
    return cls


def _content_blobs(img: np.ndarray, inner_uv: np.ndarray, vt: float, p: dict,
                   arm_full: np.ndarray | None = None) -> list:
    """Coloured / bright blobs seen through the opening of a container (image polygon inner_uv).
    Returns [{"mask", "off", "crop", "area_px", "near_edge", "near_arm"}], largest first. near_edge is
    True when the blob reaches the lower edge of the opening, i.e. its lower part is hidden by the
    near wall; near_arm when it lies next to the masked robot arm (partly hidden by the arm, or
    something the gripper still holds: one view cannot tell these apart)."""
    h, w = img.shape[:2]
    x0, y0 = max(0, int(np.floor(inner_uv[:, 0].min()))), max(0, int(np.floor(inner_uv[:, 1].min())))
    x1, y1 = min(w, int(np.ceil(inner_uv[:, 0].max())) + 1), min(h, int(np.ceil(inner_uv[:, 1].max())) + 1)
    if x1 - x0 < 20 or y1 - y0 < 20:
        return []
    crop = img[y0:y1, x0:x1]
    region = np.zeros(crop.shape[:2], np.uint8)
    cv2.fillConvexPoly(region, cv2.convexHull(np.round(inner_uv - (x0, y0)).astype(np.int32)), 255)
    region = cv2.erode(region, _ell(9), borderValue=0)   # keep clear of the rim's lighter top surface
    arm_c = None
    if arm_full is not None:
        arm_c = arm_full[y0:y1, x0:x1]
        region[arm_c > 0] = 0
    has_c, low_region = _low_edge(region)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    cls = _content_classes(hsv, vt, p)
    cls[region == 0] = 0
    out = []
    for c in range(1, 7):
        m = (cls == c).astype(np.uint8) * 255
        if cv2.countNonZero(m) < 600:
            continue
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, _ell(5))
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ell(9))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            bx, by, bw, bh, area = (int(v) for v in stats[i])
            if area < 800 or bw < 30 or bh < 20:
                continue
            pad = 6
            cx0, cy0 = max(0, bx - pad), max(0, by - pad)
            cx1, cy1 = min(m.shape[1], bx + bw + pad), min(m.shape[0], by + bh + pad)
            sub = _fill((lab[cy0:cy1, cx0:cx1] == i).astype(np.uint8) * 255)
            cols = np.arange(bx, bx + bw)
            _, low_b = _low_edge(m[:, bx:bx + bw] & (lab[:, bx:bx + bw] == i).astype(np.uint8) * 255)
            gap = low_region[cols][has_c[cols]] - low_b[has_c[cols]]
            near = bool(len(gap) and np.percentile(gap, 20) <= 4)
            near_arm = False
            if arm_c is not None:
                grown = cv2.dilate(sub, _ell(13))       # 6 px around the blob (its crop is padded by 6)
                near_arm = bool(np.any(arm_c[cy0:cy1, cx0:cx1][grown > 0] > 0))
            out.append({"mask": sub, "off": (x0 + cx0, y0 + cy0), "crop": crop[cy0:cy1, cx0:cx1],
                        "area_px": area, "near_edge": near, "near_arm": near_arm,
                        # what lies around it (same window): the opening without the arm, and the colour classes
                        "region": region[cy0:cy1, cx0:cx1], "cls": cls[cy0:cy1, cx0:cx1]})
    out.sort(key=lambda b: -b["area_px"])
    return out


def _model_content(fc: _FastCam, blob: dict, inner: dict, vt: float) -> dict:
    """Rough description of something lying inside a container. Only its top is seen reliably, so
    the footprint is the visible blob back-projected to an ASSUMED top height (70 % of the inner
    depth, capped at 25 mm); nothing here is accurate enough to grasp from."""
    m, off, crop = blob["mask"], blob["off"], blob["crop"]
    ys, xs = np.nonzero(m)
    bbox = (int(xs.min() + off[0]), int(ys.min() + off[1]), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1))
    core = cv2.erode(m, _ell(7))
    if cv2.countNonZero(core) < 60:
        core = m
    color, detail = color_name(crop[core > 0], vt)
    notes = []
    if color == "other":
        notes.append("colour: " + detail)
    depth = max(inner["rim_z"] - inner["floor_z"], 0.005)
    h_top = min(0.7 * depth, 0.025)
    cnt = _contour_pts(m, off)
    xy = fc.to_plane(cnt, inner["floor_z"] + h_top)[:, :2]
    rect = _robust_rect(xy)
    kind, yaw = "unknown", 0.0
    if rect is None:
        c = np.nanmean(xy, axis=0)
        rect = {"x": float(c[0]), "y": float(c[1]), "yaw": 0.0, "length": 0.02, "width": 0.02}
    area = 0.5 * abs(float(np.dot(xy[:, 0], np.roll(xy[:, 1], -1)) - np.dot(xy[:, 1], np.roll(xy[:, 0], -1))))
    fill_rect = area / max(rect["length"] * rect["width"], 1e-9)
    if not blob["near_edge"]:
        if fill_rect >= 0.90:
            kind, yaw = "box", rect["yaw"]
        elif 0.70 <= fill_rect <= 0.84 and rect["width"] >= 0.8 * rect["length"]:
            kind = "cylinder"                         # a disc fills pi/4 of its bounding square
    # keep the estimate inside the opening
    ca, sa = math.cos(inner["yaw"]), math.sin(inner["yaw"])
    d = np.array([rect["x"], rect["y"]]) - np.array(inner["center"])
    along = float(np.clip(d[0] * ca + d[1] * sa, -inner["size"][0] / 2, inner["size"][0] / 2))
    across = float(np.clip(-d[0] * sa + d[1] * ca, -inner["size"][1] / 2, inner["size"][1] / 2))
    cx = inner["center"][0] + along * ca - across * sa
    cy = inner["center"][1] + along * sa + across * ca
    length, width = min(rect["length"], inner["size"][0]), min(rect["width"], inner["size"][1])
    if kind == "cylinder":
        dia = 0.5 * (length + width)
        size, foot = (dia, dia, h_top), _circle_poly(cx, cy, dia / 2)
    else:
        size = (length, width, h_top)
        foot = _poly_ccw(box_corners(cx, cy, rect["yaw"], length, width, 0.0, 0.0)[:4, :2])
        yaw = rect["yaw"]
    notes.append(f"height {h_top * 1000:.0f} mm is ASSUMED (only the top is visible inside a container)")
    if blob["near_edge"]:
        notes.append("partly hidden behind the container's near wall")
    near_arm = bool(blob.get("near_arm", False))
    if near_arm:
        notes.append("next to the robot arm in the image: partly hidden by the arm, or still held by the gripper "
                     "(one view cannot tell)")
    return {"kind": kind, "color": color, "center": (cx, cy), "yaw": yaw, "size": size, "footprint": foot,
            "pixel_bbox": bbox, "confidence": 0.3 if kind != "unknown" else 0.2, "notes": notes,
            "area_px": int(blob["area_px"]), "near_arm": near_arm}


# --------------------------------------------------------------------------- upright round things, open tops
_BOTTOM_ALLOWANCE_M = 0.0025    # height of such an object minus the visible depth of its cavity (bottom + fillet), assumed


def _circle_lsq(xy: np.ndarray) -> tuple | None:
    """Algebraic least-squares circle through 2-D points: (cx, cy, r) or None."""
    if len(xy) < 6:
        return None
    m = xy.mean(axis=0)
    q = (xy - m) * 100.0                    # centimetres: keeps the normal equations well conditioned
    x, y = q[:, 0], q[:, 1]
    w = x * x + y * y
    sxx, syy, sxy = float(x @ x), float(y @ y), float(x @ y)
    det = sxx * syy - sxy * sxy             # (the sums of x and of y are zero)
    if not (abs(det) > 1e-12):
        return None
    bx, by = 0.5 * float(x @ w), 0.5 * float(y @ w)
    cx, cy = (bx * syy - by * sxy) / det, (by * sxx - bx * sxy) / det
    r2 = float(w.mean()) + cx * cx + cy * cy
    if not (r2 > 0.0 and math.isfinite(r2)):
        return None
    return cx / 100.0 + float(m[0]), cy / 100.0 + float(m[1]), math.sqrt(r2) / 100.0


def _fit_top_circle(fc: _FastCam, auv: np.ndarray, z_top: float, keep: np.ndarray | None = None,
                    rays: np.ndarray | None = None) -> dict | None:
    """Horizontal circle at height z_top whose far half is the arc auv (N x 2 image points, the upper
    outline of something seen from the side and above).

    The points are put on the plane z = z_top, where the rim of an upright round thing is a true
    circle; a tilted rim, the top of a box or two things in one arc are not, and leave large
    residuals. Only the far half counts (nearer the camera the outline runs down the object's side).
    keep: use exactly these points (no selection); rays: fc.rays(auv), when the caller fits the same
    arc at several heights. Returns x, y, r, rms (m), cover_deg (angle of arc seen from the centre),
    frac (share of the far-half points that fit), n, keep; or None."""
    if rays is None:
        rays = fc.rays(auv)
    with np.errstate(divide="ignore", invalid="ignore"):
        sc = (z_top - fc.C[2]) / rays[:, 2]
    if not (np.all(np.isfinite(sc)) and np.all(sc > 0)):
        return None
    pl = fc.C[:2] + sc[:, None] * rays[:, :2]
    if keep is not None:
        c = _circle_lsq(pl[keep])
        if c is None:
            return None
        res = np.hypot(pl[keep, 0] - c[0], pl[keep, 1] - c[1]) - c[2]
        return {"x": c[0], "y": c[1], "r": c[2], "rms": float(np.sqrt(np.mean(res * res))), "keep": keep,
                "n": int(np.count_nonzero(keep))}
    width = float(auv[:, 0].max() - auv[:, 0].min())
    sel = auv[:, 1] <= auv[:, 1].min() + 0.3 * width           # the top of the arc, for a start
    far = sel
    c = None
    for tol in (0.0010, 0.0006, 0.00045, 0.00045):
        if np.count_nonzero(sel) < 12:
            return None
        c = _circle_lsq(pl[sel])
        if c is None:
            return None
        rel = pl - np.array(c[:2])
        n = fc.C[:2] - np.array(c[:2])
        n = n / max(float(np.linalg.norm(n)), 1e-9)
        eta = rel @ n                                          # towards the camera
        far = eta <= 0.15 * c[2]
        new = far & (np.abs(np.hypot(rel[:, 0], rel[:, 1]) - c[2]) <= tol)
        if np.array_equal(new, sel):
            break
        sel = new
    if np.count_nonzero(sel) < 12:
        return None
    c = _circle_lsq(pl[sel])
    if c is None:
        return None
    rel = pl[sel] - np.array(c[:2])
    n = fc.C[:2] - np.array(c[:2])
    n = n / max(float(np.linalg.norm(n)), 1e-9)
    ang = np.degrees(np.arctan2(rel[:, 0] * (-n[1]) + rel[:, 1] * n[0], -(rel @ n)))
    lo, hi = np.percentile(ang, [2, 98])
    res = np.hypot(rel[:, 0], rel[:, 1]) - c[2]
    return {"x": c[0], "y": c[1], "r": c[2], "rms": float(np.sqrt(np.mean(res * res))), "cover_deg": float(hi - lo),
            "frac": float(np.count_nonzero(sel)) / max(int(np.count_nonzero(far)), 1), "n": int(np.count_nonzero(sel)),
            "keep": sel}


def _outline_paths(mask: np.ndarray) -> tuple[list, np.ndarray | None]:
    """Upper outline of a blob cut at its concave corners (a list of K x 2 pixel arrays, each one arc,
    left to right) and its lower outline (one array). Two round things side by side in one blob give
    two arcs; a box gives one arc that is not round."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return [], None
    c = max(cnts, key=cv2.contourArea)
    pts = c.reshape(-1, 2)
    n = len(pts)
    if n < 40:
        return [], None
    il = int(np.lexsort((pts[:, 1], pts[:, 0]))[0])            # leftmost (the upper one of them)
    ir = int(np.lexsort((pts[:, 1], -pts[:, 0]))[0])           # rightmost (the upper one of them)

    def path(i0, i1):
        return np.arange(i0, i1 + 1) if i0 <= i1 else np.concatenate([np.arange(i0, n), np.arange(0, i1 + 1)])

    p1, p2 = path(il, ir), path(ir, il)
    if pts[p1, 1].mean() < pts[p2, 1].mean():
        up, low = p1, p2[::-1]
    else:
        up, low = p2[::-1], p1
    on_up = np.zeros(n, bool)
    on_up[up] = True
    cuts = []
    try:
        defects = cv2.convexityDefects(c, cv2.convexHull(c, returnPoints=False))
    except cv2.error:
        defects = None
    if defects is not None:
        for _, _, f, d in defects.reshape(-1, 4):
            if d >= 4 * 256 and on_up[f]:                     # a pocket at least 4 px deep in the upper outline
                cuts.append(int(f))
    pos = {int(i): k for k, i in enumerate(up)}
    arcs, start = [], 0
    for cp in sorted(pos[f] for f in cuts) + [len(up)]:
        if cp - start >= 24:
            arcs.append(pts[up[start:cp]])
        start = cp + 1
    return arcs, pts[low]


def _cavity(img: np.ndarray, fc: _FastCam, x: float, y: float, r: float, z_top: float, valid=None) -> dict:
    """Is the top of an upright round object open (a cup, a bottle cap standing on its closed side)?

    Looking into a cavity from above and from the side, the camera sees the rim, below it the far
    inner wall, and where that wall ends the inner bottom. Every pixel inside the rim circle is given
    the depth t at which a cylindrical inner wall would be seen there; the brightness (the smallest
    colour channel, which still varies where a white part is over-exposed) averaged over pixels of
    equal t is flat for a solid top and shows a darker wall followed by a step up to the bottom for a
    cavity. The step must be there on the left and on the right half, the wall must be darker than
    the bottom as a whole, and the rim above it must not be darker than the wall (a shadow falling
    across a flat top fails that).

    valid: (mask, (x0, y0)) = pixels that belong to the object (None: all).
    Returns {"hollow": bool, "depth": m | None (rim to the visible end of the wall), "wall": m | None
    (width of the bright rim, an upper bound of the wall thickness where the white is over-exposed),
    "contrast": brightness levels, "bottom_frac": share of the top's pixels that show the inner bottom (set
    when the brightness pattern is that of a cavity), "clipped": bool, "faint": bool (everything fits a cavity, but the
    step is weaker than the 3 levels asked for: hollow stays False), "judged": bool (the answer can be
    relied on: a cavity is seen, or the top is evenly lit, neither dark nor over-exposed, and shows no
    step at all; False = it could not be told), "why": str}."""
    out = {"hollow": False, "depth": None, "wall": None, "contrast": 0.0, "clipped": False, "faint": False,
           "judged": False, "why": ""}
    uv = fc.project(cyl_points(x, y, r, 0.0, z_top, top_only=True))
    h_img, w_img = img.shape[:2]
    x0, y0 = max(0, int(np.floor(uv[:, 0].min())) - 2), max(0, int(np.floor(uv[:, 1].min())) - 2)
    x1, y1 = min(w_img, int(np.ceil(uv[:, 0].max())) + 3), min(h_img, int(np.ceil(uv[:, 1].max())) + 3)
    if x1 - x0 < 30 or y1 - y0 < 16:
        out["why"] = "too small in the image"
        return out
    sub = 2 if (x1 - x0) * (y1 - y0) > 5000 else 1          # every second pixel is plenty for a large top
    minc = _min_channel(img[y0:y1:sub, x0:x1:sub]).astype(np.float32).ravel()
    uu, vv = np.meshgrid(np.arange(x0, x1, sub, dtype=float), np.arange(y0, y1, sub, dtype=float))
    q = fc.to_plane(np.column_stack([uu.ravel(), vv.ravel()]), z_top)[:, :2] - np.array([x, y])
    ok = np.isfinite(q[:, 0])
    if valid is not None:
        vm, voff = valid
        cx = (uu.ravel() - voff[0]).astype(int)
        cy = (vv.ravel() - voff[1]).astype(int)
        inm = (cx >= 0) & (cy >= 0) & (cx < vm.shape[1]) & (cy < vm.shape[0])
        okv = np.zeros(len(cx), bool)
        okv[inm] = vm[cy[inm], cx[inm]] > 0
        ok &= okv
    nvec = fc.C[:2] - np.array([x, y])
    dist = float(np.linalg.norm(nvec))
    nvec = nvec / max(dist, 1e-9)
    eta = np.where(ok, q[:, 0] * nvec[0] + q[:, 1] * nvec[1], 0.0)        # towards the camera
    xi = np.where(ok, q[:, 0] * (-nvec[1]) + q[:, 1] * nvec[0], 0.0)
    rho = np.hypot(xi, eta)
    ri = r - 0.002                                              # a typical wall; only places the pixels
    delta = float(fc.C[2] - z_top)
    if ri < 0.004 or delta < 0.05:
        out["why"] = "too small"
        return out
    # depth t at which the far half of an inner wall of radius ri is seen at a pixel: the wall circle at
    # depth t appears on the rim plane shrunk by (1 - k) and moved k * dist towards the camera, k = t / (delta + t)
    a_ = dist * dist - ri * ri
    b_ = -2.0 * (eta * dist - ri * ri)
    c_ = xi * xi + eta * eta - ri * ri
    k = (-b_ + np.sqrt(np.maximum(b_ * b_ - 4.0 * a_ * c_, 0.0))) / (2.0 * a_)
    t = k * delta / np.maximum(1.0 - k, 1e-6)
    step = 0.0005
    nb = int(min(0.04, 2.4 * r) / step)
    inside = ok & (rho < ri - 0.0005)
    idx = np.where(inside, np.clip(t / step, -1, nb), -1).astype(int)
    use = inside & (idx >= 0) & (idx < nb)
    if np.count_nonzero(use) < 300 // (sub * sub):
        out["why"] = "too little of the top is visible"
        return out
    need = max(2, 8 // (sub * sub))                             # pixels a depth bin must have

    def profile(sel):
        cnt = np.bincount(idx[sel], minlength=nb)[:nb]
        sm = np.bincount(idx[sel], weights=minc[sel], minlength=nb)[:nb]
        pr = np.full(nb, np.nan)
        good = cnt >= need
        pr[good] = sm[good] / cnt[good]
        return pr

    def steps(pr, before=6, after=6):
        """For every bin j: mean of the `after` bins from j minus mean of the `before` bins up to j
        (NaN where fewer than 4 bins on either side have data)."""
        ok_ = np.isfinite(pr)
        cs = np.concatenate([[0.0], np.cumsum(np.where(ok_, pr, 0.0))])
        cn = np.concatenate([[0], np.cumsum(ok_)])
        j = np.arange(nb)
        a0, b1 = np.maximum(j - before, 0), np.minimum(j + after, nb)
        n_lo, n_hi = cn[j] - cn[a0], cn[b1] - cn[j]
        with np.errstate(divide="ignore", invalid="ignore"):
            st = (cs[b1] - cs[j]) / n_hi - (cs[j] - cs[a0]) / n_lo
        st[(n_lo < 4) | (n_hi < 4)] = np.nan
        return st

    prof = profile(use)
    st = steps(prof)
    st[:8] = np.nan                                             # a cavity at least 4 mm deep
    st[nb - 4:] = np.nan
    best, jb = 0.0, None
    if np.any(np.isfinite(st)):
        jb = int(np.nanargmax(st))
        best = float(st[jb])
        if not best > 0.0:
            best, jb = 0.0, None
    out["contrast"] = round(best, 2)
    seen_p = prof[np.isfinite(prof)]
    level = float(np.percentile(seen_p, 90)) if len(seen_p) else 0.0
    out["clipped"] = bool(level >= 250.0)
    if jb is None or best < 1.5:
        out["judged"] = bool(60.0 <= level < 250.0)             # a dark or an over-exposed top would not show a cavity
        out["why"] = ("no darker inner wall with a step to a brighter bottom is seen inside the rim"
                      + (" (the top is over-exposed white: a shallow or very evenly lit cavity would not show)"
                         if out["clipped"] else (" (the top is dark: a cavity would hardly show)" if level < 60.0 else "")))
        return out
    wall_mean = float(np.nanmean(prof[2:jb]))
    floor_mean = float(np.nanmean(prof[jb:jb + 10]))
    if floor_mean - wall_mean < 0.5 * best:
        out["why"] = "a brightness step inside the rim, but the part above it is not darker as a whole (a mark, a shadow edge)"
        return out
    first = prof[1:6]                                           # 0.5 .. 2.5 mm below the rim
    if not np.any(np.isfinite(first)) or float(np.nanmin(first)) > floor_mean - 0.5 * best:
        # an inner wall begins at the rim; a hub, a boss or a mark in the middle of a flat top does not
        out["why"] = ("the darker part does not begin at the rim (something raised or dark in the middle of a flat top, "
                      "not an inner wall)")
        return out
    after = prof[jb:jb + 10]
    after = after[np.isfinite(after)]
    if len(after) >= 4 and float(after.std()) > 0.4 * best:
        out["why"] = "what lies beyond the brightness step is not an evenly lit bottom (a pattern on a flat top)"
        return out
    # the end of an inner wall runs all the way round: every quarter of the top must show the step
    ang = np.degrees(np.arctan2(xi, -eta))                      # 0 = straight away from the camera
    parts = []
    for a0, a1 in ((-180.0, -45.0), (-45.0, 0.0), (0.0, 45.0), (45.0, 180.0)):
        sel = use & (ang >= a0) & (ang < a1)
        if np.count_nonzero(sel) < 80 // (sub * sub):
            continue
        sq = steps(profile(sel))[max(8, jb - 3):min(nb - 4, jb + 4)]
        if np.any(np.isfinite(sq)):
            parts.append(float(np.nanmax(sq)))
    if len(parts) < 2 or min(parts) < 0.35 * best:
        out["why"] = "the brightness step is on one side of the top only (a shadow edge or a mark, not the end of an inner wall)"
        return out
    ring = ok & (rho >= r - 0.00175) & (rho <= r - 0.0006) & (eta < -0.3 * rho)
    if np.count_nonzero(ring) >= 6 and float(minc[ring].mean()) < wall_mean + 0.2 * best:
        out["why"] = "the rim is not brighter than what lies inside it (a shadow across a flat top)"
        return out
    # the inner bottom must be seen over a real part of the top: a bright sliver at the near edge is the edge of
    # a flat top catching the light (or the bottom of a cavity too deep to be looked into), not an inner bottom
    out["bottom_frac"] = float(np.count_nonzero(use & (idx >= jb))) / float(np.count_nonzero(use))
    if out["bottom_frac"] < 0.12:
        out["why"] = (f"the brighter part beyond the brightness step covers only {100.0 * out['bottom_frac']:.0f} % of the "
                      "top (a highlight at the near edge of a flat top, or a cavity too deep for its bottom to be seen)")
        return out
    if best < 3.0:
        out.update(faint=True, why=f"a darker inner wall with a step to a brighter bottom shows, but only by {best:.1f} "
                                   "brightness levels (3 are asked for)")
        return out
    out.update(hollow=True, judged=True, depth=float(jb * step), why="")
    # width of the bright rim, seen from outside inwards on the far half
    sector = ok & (eta < 0.2 * rho)
    lv, edges = [], np.arange(0.0006, 0.0046, 0.00025)
    for e0 in edges:
        sel = sector & (rho <= r - e0) & (rho > r - e0 - 0.00025)
        lv.append(float(minc[sel].mean()) if np.count_nonzero(sel) >= 3 else float("nan"))
    lv = np.array(lv)
    if np.all(np.isfinite(lv[:3])):
        drop = float(np.mean(lv[:3])) - max(1.5, 0.15 * best)
        for i in range(2, len(lv) - 1):
            if lv[i] <= drop and lv[i + 1] <= drop:
                out["wall"] = float(np.clip(edges[i] + 0.000125, 0.0008, 0.0045))
                break
    return out


def _round_tops(img: np.ndarray, fc: _FastCam, blob: dict, inner: dict, vt: float, h_guess: float,
                floor_checked: bool) -> tuple[list, list]:
    """Upright round things (cylinder, disc, cap) in one colour blob seen through the opening of a
    container, each from an arc of the blob's upper outline, and what is left of the blob.

    For every arc that is the far half of a horizontal circle (see _fit_top_circle):
    * the height comes from the lower outline, where that is the thing's own base on the dark floor
      (the circle on the top plane and the base circle on the floor must be one cylinder);
    * else, for a top that is open (_cavity), from the depth of the cavity plus a bottom as thick as
      a typical wall: a lower bound, and it is said so;
    * else it is assumed, as for everything else in a container, and the confidence stays low.
    The centre and the radius are the circle on the plane of that height, so all three are consistent
    with the camera model.
    Returns (objects as dicts like _model_content's, blobs left over for _model_content)."""
    m, off = blob["mask"], np.array(blob["off"], dtype=float)
    region, cls = blob.get("region"), blob.get("cls")
    hh, ww = m.shape
    z_floor = float(inner["floor_z"])
    arcs, low = _outline_paths(m)
    cands = []
    for arc in arcs:
        if region is not None:                                  # not where the arm or the rim of the opening cuts the blob
            arc = arc[region[np.clip(arc[:, 1] - 3, 0, hh - 1), arc[:, 0]] > 0]
        if len(arc) < 24:
            continue
        auv = arc + off
        rays = fc.rays(auv)
        fit = _fit_top_circle(fc, auv, z_floor + h_guess, rays=rays)
        if fit is None or fit["n"] < 24 or fit["rms"] > 0.00035 or fit["cover_deg"] < 100.0 or fit["frac"] < 0.6 \
                or (fit["frac"] < 0.7 and fit["cover_deg"] < 150.0) or not (0.005 <= fit["r"] <= 0.045):
            continue
        if fit["r"] < 0.009 and (fit["n"] < 40 or fit["frac"] < 0.8):
            continue                                            # a short round stretch of something that is not round
        cands.append({"auv": auv, "fit": fit, "rays": rays})
    if not cands:
        return [], []
    ys, xs = np.nonzero(m)
    claimed = np.zeros_like(m)
    objs = []
    for cd in cands:
        auv, fit, rays = cd["auv"], cd["fit"], cd["rays"]
        keep = fit["keep"]
        notes = []
        # ---- the base: lower outline pixels of the blob under this top that border the dark floor
        ring_uv = fc.project(cyl_points(fit["x"], fit["y"], fit["r"], 0.0, z_floor + h_guess, top_only=True)) - off
        u_lo, u_hi = float(ring_uv[:, 0].min()), float(ring_uv[:, 0].max())
        base_uv, h, base_seen = None, None, False
        if low is not None and len(low) and region is not None and cls is not None:
            cols = (low[:, 0] >= u_lo + 2) & (low[:, 0] <= u_hi - 2)
            lx, ly = low[cols, 0], low[cols, 1]
            if len(lx) >= 12:
                # The colour class ends where the object's side gets dim near its foot. Follow each column
                # down to where the dark floor really begins; the base is seen only where it ends on that
                # floor (not on another thing, not at the edge of the opening).
                reach = 14
                fx = np.clip(lx + int(off[0]), 0, img.shape[1] - 1)
                fy = ly[:, None] + int(off[1]) + np.arange(1, reach + 1)[None, :]
                val = img[np.clip(fy, 0, img.shape[0] - 1), fx[:, None]].max(axis=2)
                dim = (val < 0.6 * vt) | (fy >= img.shape[0])
                ext = np.where(dim.any(axis=1), np.argmax(dim, axis=1), reach)        # rows the object goes on for
                cy = np.clip(ly[:, None] + np.arange(1, reach + 1)[None, :], 0, hh - 1)
                other = (cls[cy, lx[:, None]] != 0) & (np.arange(reach)[None, :] < ext[:, None]) & \
                        (ly[:, None] + np.arange(1, reach + 1)[None, :] < hh)
                end = np.clip(ly + ext + 1, 0, hh - 1)
                free = (ext < reach) & ~other.any(axis=1) & (ly + ext + 1 < hh) & (region[end, lx] > 0)
                if np.count_nonzero(free) >= 12 and (lx[free].max() - lx[free].min()) >= 0.3 * (u_hi - u_lo):
                    base_uv = np.column_stack([lx[free], ly[free] + ext[free]]) + off + np.array([0.0, 0.5])
        if base_uv is not None:
            q_base = fc.to_plane(base_uv, z_floor)[:, :2]
            rk = rays[keep]

            def miss(hh_, full=False):
                # how far outside the base circle the seen base points lie when the object is hh_ high
                # (falls as hh_ grows: the circle on the top plane moves towards the camera)
                c = _circle_lsq(fc.C[:2] + ((z_floor + hh_ - fc.C[2]) / rk[:, 2])[:, None] * rk[:, :2])
                if c is None:
                    return -1.0, None, None
                rel = q_base - np.array(c[:2])
                n = fc.C[:2] - np.array(c[:2])
                n = n / max(float(np.linalg.norm(n)), 1e-9)
                near = (rel @ n) > 0.25 * c[2]                   # the near arc of the base, not its sides
                if np.count_nonzero(near) < 8:
                    return -1.0, None, None                      # the circle has passed the points: too high
                d = np.hypot(rel[near, 0], rel[near, 1]) - c[2]
                return _med(d), d, (_fit_top_circle(fc, auv, z_floor + hh_, keep=keep, rays=rays) if full else None)

            hb = _bisect(lambda v: miss(v)[0], 0.003, 0.08, iters=12)
            _, d, f = miss(hb, full=True)
            if d is not None and f is not None and 0.0035 < hb < 0.078 and _med(np.abs(d - _med(d))) <= 0.0007 \
                    and float(np.mean(np.abs(d) <= 0.0012)) >= 0.6:
                sil = _hull_mask(fc, cyl_points(f["x"], f["y"], f["r"], hb, z_floor), m.shape, off)
                sil[(m == 0) & (cls != 0)] = 0                  # where something of another colour stands in front
                col = np.zeros_like(m)
                col[:, max(0, int(u_lo) - 2):int(u_hi) + 3] = 255
                own = cv2.bitwise_and(m, col)
                if _iou(sil, own) >= 0.8:
                    h, base_seen, fit = hb, True, dict(f, cover_deg=fit["cover_deg"], frac=fit["frac"])
        # ---- an open top
        h_now = h if base_seen else h_guess
        try:
            cav = _cavity(img, fc, fit["x"], fit["y"], fit["r"], z_floor + h_now, valid=(m, blob["off"]))
        except Exception as e:
            cav = {"hollow": False, "faint": False, "judged": False, "why": f"{type(e).__name__}: {e}"}
        doubt = None
        if cav["hollow"] and base_seen and not (0.35 * h <= cav["depth"] <= h - 0.0003):
            # a "cavity" deeper than the thing is high (or a mere dent): the pattern in its top is not understood
            doubt = (f"inside the rim there is the pattern of a cavity {cav['depth'] * 1000:.0f} mm deep, which does not "
                     f"fit its height of {h * 1000:.0f} mm")
            cav = dict(cav, hollow=False, judged=False, faint=False, why=doubt)
        if cav["hollow"] and not base_seen:
            # the height from the cavity; the circle again on the plane of that height (the depth read in
            # the image changes by well under a millimetre with it)
            h_now = cav["depth"] + _BOTTOM_ALLOWANCE_M
            f = _fit_top_circle(fc, auv, z_floor + h_now, keep=keep, rays=rays)
            if f is not None:
                fit = dict(f, cover_deg=fit["cover_deg"], frac=fit["frac"])
        quality = (float(np.clip((0.0004 - fit["rms"]) / 0.00025, 0.0, 1.0))
                   * float(np.clip((fit["cover_deg"] - 80.0) / 60.0, 0.3, 1.0)))
        if base_seen:
            conf = 0.6 + 0.3 * quality
            notes.append(f"upright and round: rim circle fitted to {fit['cover_deg']:.0f} deg of arc ({fit['rms'] * 1000:.2f} mm "
                         "rms), height from its own base on the container floor")
        elif cav["hollow"]:
            h = h_now
            conf = 0.5 + 0.2 * quality
            notes.append(f"upright and round: rim circle fitted to {fit['cover_deg']:.0f} deg of arc ({fit['rms'] * 1000:.2f} mm "
                         "rms); its base is hidden, so the height is the visible depth of the cavity "
                         f"({cav['depth'] * 1000:.0f} mm) + {_BOTTOM_ALLOWANCE_M * 1000:.1f} mm ASSUMED for its bottom: a lower "
                         "bound, about 2 mm short for a cap with a liner or a thick bottom (per mm of height the centre "
                         "moves 1.3 mm along the viewing direction)")
        else:
            h = h_guess
            conf = 0.3
            notes.append(f"round top (rim circle fitted to {fit['cover_deg']:.0f} deg of arc, {fit['rms'] * 1000:.2f} mm rms) "
                         f"whose base is hidden (it stands behind or lies on something): height {h * 1000:.0f} mm ASSUMED, "
                         "the diameter changes by 0.4 % and the centre by 1.3 mm per mm of error")
        res = {"kind": "cylinder", "center": (fit["x"], fit["y"]), "yaw": 0.0, "size": (2.0 * fit["r"], 2.0 * fit["r"], h),
               "footprint": _circle_poly(fit["x"], fit["y"], fit["r"]), "confidence": conf, "notes": notes,
               "near_arm": bool(blob.get("near_arm", False)), "base_seen": base_seen,
               "hollow": True if cav["hollow"] else (False if cav.get("judged") else None),
               "inner_radius": None, "wall": None, "z_inner_bottom": None, "_fit": fit, "_arc": (auv, keep, rays)}
        if cav.get("faint"):
            notes.append(f"it may be open at the top: inside the rim there is the pattern of a cavity, but too faint to "
                         f"count ({cav['contrast']:.1f} brightness levels, 3 are asked for)")
        if cav["hollow"]:
            # as far down as the inner wall is seen; above its base only when the height is measured (with the
            # base hidden the height itself rests on an assumed bottom, so nothing would be said by it)
            z_ib = max(z_floor + h - cav["depth"], z_floor + 0.0005) if base_seen else None
            res.update(wall=cav["wall"], z_inner_bottom=(float(z_ib) if base_seen else None),
                       inner_radius=(float(fit["r"] - cav["wall"]) if cav["wall"] else None))
            notes.append(_hollow_note(cav, (z_ib - z_floor) if base_seen else None))
        elif res["hollow"] is None and cav.get("why") and not cav.get("faint"):
            notes.append("open or closed top: not determined (" + cav["why"] + ")")
            if doubt:
                notes.append("what its top shows does not fit the height found from its base: less certain")
                res["confidence"] = min(res["confidence"], 0.5)
        # what part of the blob it is: its outline as a cylinder, or (base hidden) its top and what is under it
        top_sil = _hull_mask(fc, cyl_points(fit["x"], fit["y"], fit["r"], 0.0, z_floor + h, top_only=True), m.shape, off)
        if base_seen:
            part = _hull_mask(fc, cyl_points(fit["x"], fit["y"], fit["r"], h, z_floor), m.shape, off)
        else:
            part = top_sil.copy()
            tys, txs = np.nonzero(top_sil)
            if len(txs):
                part[int(tys.max()):, int(txs.min()):int(txs.max()) + 1] = 255
        res["_part"], res["_top"] = cv2.bitwise_and(m, cv2.dilate(part, _SQ3)), top_sil
        # where its colour shows: the top (often over-exposed) and the side face straight below it
        tys = np.nonzero(top_sil.any(axis=1))[0]
        side = top_sil.copy()
        if len(tys):
            drop = max(4, int(0.6 * (tys.max() - tys.min() + 1)))
            side[drop:] = cv2.bitwise_or(side[drop:], top_sil[:-drop])
        res["_colour"] = cv2.bitwise_and(m, side)
        res["_depth"] = float(np.linalg.norm(fc.C[:2] - np.array([fit["x"], fit["y"]])))
        objs.append(res)
    # nearer things hide farther ones: where two parts overlap the nearer one owns the pixels
    objs.sort(key=lambda o: o["_depth"])
    for o in objs:
        o["_part"] = cv2.bitwise_and(o["_part"], cv2.bitwise_not(claimed))
        claimed = cv2.bitwise_or(claimed, o["_part"])
    rest = cv2.morphologyEx(cv2.bitwise_and(m, cv2.bitwise_not(claimed)), cv2.MORPH_OPEN, _ell(5))
    leftovers = []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(rest, connectivity=8)
    for i in range(1, n):
        bx, by, bw, bh, area = (int(v) for v in stats[i])
        if area < 800 or bw < 30 or bh < 20:
            continue
        sub = (lab == i).astype(np.uint8) * 255
        near = False
        if region is not None:
            has_c, low_r = _low_edge(sub)
            cols = np.nonzero(has_c)[0]
            near = bool(np.mean(region[np.clip(low_r[cols] + 5, 0, hh - 1), cols] == 0) >= 0.2)
        leftovers.append({"mask": sub, "off": blob["off"], "crop": blob["crop"], "area_px": area, "near_edge": near,
                          "near_arm": bool(blob.get("near_arm", False)), "region": region, "cls": cls})
    # a hidden-base round thing with something else under it keeps only its top (the rest is that other thing)
    for o in objs:
        if not o["base_seen"] and leftovers:
            o["_part"] = cv2.bitwise_and(o["_part"], cv2.dilate(o["_top"], _ell(9)))
    # Something of the same colour sticks out from under a round thing: it may LIE ON that thing (a
    # disc on a block looks exactly like a short cylinder standing behind the block). One view cannot
    # tell, so its height above the floor is not trusted: it is put as high as lying on a thing of
    # the usual assumed height would put it (never above the rim), which is the careful choice for
    # everything that has to pass over it. An open top whose cavity depth fits its height keeps its place.
    rim_h = max(float(inner["rim_z"]) - z_floor, 0.005)
    for o in objs:
        if not leftovers or (o["hollow"] and o["base_seen"]):
            continue
        if o["hollow"]:
            continue                                            # height from its cavity: independent of what is under it
        grown = cv2.dilate(o["_part"], _ell(11))
        ys_ = np.nonzero(o["_top"].any(axis=1))[0]
        if not len(ys_):
            continue
        for lf in leftovers:
            lower = lf["mask"].copy()
            lower[:int(ys_.mean())] = 0                         # only what lies below its top counts
            if cv2.countNonZero(cv2.bitwise_and(grown, lower)) < 30:
                continue
            thick = o["size"][2] if o["base_seen"] else None
            h_new = min(rim_h, h_guess + (thick if thick is not None else 0.008))
            auv, keep, rays = o["_arc"]
            f = _fit_top_circle(fc, auv, z_floor + h_new, keep=keep, rays=rays)
            if f is not None:
                o["_fit"] = dict(f, cover_deg=o["_fit"]["cover_deg"], frac=o["_fit"]["frac"])
                o["center"] = (f["x"], f["y"])
                o["size"] = (2.0 * f["r"], 2.0 * f["r"], h_new)
                o["footprint"] = _circle_poly(f["x"], f["y"], f["r"])
            else:
                o["size"] = (o["size"][0], o["size"][1], h_new)
            o["base_seen"] = False
            o["confidence"] = 0.3
            o["notes"][:1] = [f"round top (rim circle fitted to {o['_fit']['cover_deg']:.0f} deg of arc, "
                              f"{o['_fit']['rms'] * 1000:.2f} mm rms) with something of the same colour under or beside it: "
                              "it probably lies on that thing"
                              + (f" (standing on the floor it would be {thick * 1000:.0f} mm high)" if thick is not None else "")
                              + f"; its top is ASSUMED {h_new * 1000:.0f} mm above the floor, the diameter changes by 0.4 % "
                              "and the centre by 1.3 mm per mm of error"]
            break
    crop = blob["crop"]
    out = []
    tops = np.zeros_like(m)
    for o in objs:                                              # nearest first
        part = o.pop("_part")
        top = o.pop("_top")
        o.pop("_depth"), o.pop("_arc")
        fit = o.pop("_fit")
        sample = cv2.bitwise_and(o.pop("_colour"), cv2.bitwise_not(tops))     # not where a nearer round top covers it
        tops = cv2.bitwise_or(tops, top)
        pys, pxs = np.nonzero(part)
        if len(pxs) < 150:
            continue
        core = cv2.erode(sample, _ell(5))
        if cv2.countNonZero(core) < 60:
            core = sample if cv2.countNonZero(sample) >= 60 else part
        px = crop[core > 0]
        color, detail = color_name(px[::max(1, len(px) // 3000)], vt)
        if color == "other":
            o["notes"].append("colour: " + detail)
        o["color"] = color
        o["pixel_bbox"] = (int(pxs.min() + off[0]), int(pys.min() + off[1]), int(pxs.max() - pxs.min() + 1),
                           int(pys.max() - pys.min() + 1))
        o["area_px"] = int(len(pxs))
        # inside the opening?
        ca, sa = math.cos(inner["yaw"]), math.sin(inner["yaw"])
        d = np.array(o["center"]) - np.array(inner["center"])
        over = max(abs(d[0] * ca + d[1] * sa) - inner["size"][0] / 2, abs(-d[0] * sa + d[1] * ca) - inner["size"][1] / 2)
        if over > 0.0:
            along = float(np.clip(d[0] * ca + d[1] * sa, -inner["size"][0] / 2, inner["size"][0] / 2))
            across = float(np.clip(-d[0] * sa + d[1] * ca, -inner["size"][1] / 2, inner["size"][1] / 2))
            o["center"] = (inner["center"][0] + along * ca - across * sa, inner["center"][1] + along * sa + across * ca)
            o["footprint"] = _circle_poly(o["center"][0], o["center"][1], fit["r"])
            if over > 0.004:
                o["notes"].append(f"its centre came out {over * 1000:.0f} mm outside the opening and was moved to the edge: "
                                  "the height it was placed at is probably wrong")
                o["confidence"] = min(o["confidence"], 0.3)
        if o["near_arm"]:
            o["notes"].append("next to the robot arm in the image: partly hidden by the arm, or still held by the gripper "
                              "(one view cannot tell)")
            o["confidence"] = min(o["confidence"], 0.4)
        if blob.get("near_edge") and not o["base_seen"]:
            o["notes"].append("partly hidden behind the container's near wall")
        if not floor_checked and o["confidence"] > 0.5:
            o["notes"].append("the height of the container floor under it is ASSUMED (not measured in this view): "
                              "confidence limited to 0.5")
            o["confidence"] = 0.5
        o["confidence"] = float(o["confidence"])
        out.append(o)
    return out, leftovers


def _hollow_note(cav: dict, bottom_m: float | None) -> str:
    s = (f"open at the top: the far inner wall is seen {cav['depth'] * 1000:.0f} mm down from the rim "
         f"(contrast {cav['contrast']:.0f} levels); what it ends on (the inner bottom, or something standing on it) is "
         + (f"{bottom_m * 1000:.1f} mm above its base" if bottom_m is not None else
            "an unmeasured distance above its base (the base is hidden)"))
    if cav["wall"]:
        s += f"; rim {cav['wall'] * 1000:.1f} mm wide" + (" (over-exposed white: an upper bound of the wall thickness)"
                                                          if cav.get("clipped") else "")
    else:
        s += "; wall thickness not measurable"
    return s


def _box_in_container(img: np.ndarray, fc: _FastCam, blob: dict, inner: dict, vt: float, rough: dict,
                      floor_checked: bool) -> dict:
    """A better description of a box-shaped thing seen through the opening of a container, when its
    top face can be told from its side faces (the bright part of its blob, as for loose objects).

    * Its base stands on the dark floor in view: the box is measured like a loose one (top face +
      height from the offset between the top edges and the ground edges), standing on the floor.
    * Its base is hidden (near wall, another thing): the top face alone, put at the assumed height;
      its length and width are then right to about 1 mm per 4 mm of height error, its position moves
      1.3 mm per mm.
    rough: what _model_content says about the blob (returned unchanged when no top face is found)."""
    m, off, crop = blob["mask"], blob["off"], blob["crop"]
    if blob.get("near_arm"):
        return rough
    top, tinfo = _split_top(crop, m)
    if top is None:
        return rough
    z_floor = float(inner["floor_z"])
    region, cls = blob.get("region"), blob.get("cls")
    hh = m.shape[0]
    measured = None
    if not blob.get("near_edge") and region is not None and cls is not None:
        has_c, low = _low_edge(m)
        cols = np.nonzero(has_c)[0]
        under = np.clip(low[cols] + 4, 0, hh - 1)
        free = (low[cols] + 4 < hh) & (region[under, cols] > 0) & (cls[under, cols] == 0)
        if len(cols) and float(np.mean(free)) >= 0.7:
            box = _fit_box_top(fc, m, top, off, z_floor)
            if box is not None and box["iou"] >= 0.84 and box["iou_top"] >= 0.84 and box["h"] >= 0.005 \
                    and box.get("h_spread_px", 0.0) <= 5.0:
                measured = box
    out = dict(rough)
    notes = [n for n in rough["notes"] if not n.startswith("height ")]
    if measured is not None:
        box = measured
        conf = 0.9 * float(np.clip((min(box["iou"], box["iou_top"]) - 0.78) / 0.14, 0.05, 1.0))
        notes.insert(0, f"box measured like a loose one, standing on the container floor (outline match {box['iou']:.2f}, "
                        f"top face {box['iou_top']:.2f})")
        if not floor_checked and conf > 0.5:
            notes.append("the height of the container floor under it is ASSUMED (not measured in this view): "
                         "confidence limited to 0.5")
            conf = 0.5
        length, width, h, x, y, yaw = box["length"], box["width"], box["h"], box["x"], box["y"], box["yaw"]
        kind = "box"
    else:
        if tinfo.get("frac", 0.0) < 0.35:
            return rough                    # too small a bright part to be the whole top face
        h = rough["size"][2]
        rect = _robust_rect(fc.to_plane(_contour_pts(top, off), z_floor + h)[:, :2])
        if rect is None:
            return rough
        length, width, x, y, yaw = rect["length"], rect["width"], rect["x"], rect["y"], rect["yaw"]
        kind = "box" if cv2.countNonZero(top) >= 0.85 * cv2.contourArea(cv2.boxPoints(cv2.minAreaRect(cv2.findNonZero(top)))) \
            else "unknown"
        conf = 0.3 if kind == "box" else 0.2
        notes.insert(0, f"top face seen, base hidden: length and width are those of the top face put {h * 1000:.0f} mm above "
                        "the floor (height ASSUMED; the centre moves 1.3 mm per mm of error)")
    # keep it inside the opening, like every rough estimate
    ca, sa = math.cos(inner["yaw"]), math.sin(inner["yaw"])
    d = np.array([x, y]) - np.array(inner["center"])
    along = float(np.clip(d[0] * ca + d[1] * sa, -inner["size"][0] / 2, inner["size"][0] / 2))
    across = float(np.clip(-d[0] * sa + d[1] * ca, -inner["size"][1] / 2, inner["size"][1] / 2))
    x, y = inner["center"][0] + along * ca - across * sa, inner["center"][1] + along * sa + across * ca
    length, width = min(length, inner["size"][0]), min(width, inner["size"][1])
    out.update(kind=kind, center=(x, y), yaw=float(yaw), size=(float(length), float(width), float(h)),
               footprint=_poly_ccw(box_corners(x, y, yaw, length, width, 0.0, 0.0)[:4, :2]), confidence=float(conf),
               notes=notes)
    return out


def _content_objects(img: np.ndarray, fc: _FastCam, cont: dict, vt: float, p: dict, arm_full,
                     warnings: list | None = None) -> list:
    """Everything seen through the opening of one container: upright round things measured
    (_round_tops), box-shaped things from their top face (_box_in_container), the rest described
    roughly (_model_content). Largest blob first."""
    inner = cont["inner"]
    h_guess = min(0.7 * max(inner["rim_z"] - inner["floor_z"], 0.005), 0.025)
    checked = bool(cont.get("floor_checked"))
    out = []

    def other(blob, note=None):
        r = _model_content(fc, blob, inner, vt)
        try:
            r = _box_in_container(img, fc, blob, inner, vt, r, checked)
        except Exception:
            pass
        if note:
            r["notes"].append(note)
        return r

    for b in _content_blobs(img, cont["inner_uv"], vt, p, arm_full):
        try:
            rounds, rest = _round_tops(img, fc, b, inner, vt, h_guess, checked)
        except Exception as e:              # the rough description below still stands
            rounds, rest = [], []
            if warnings is not None:
                warnings.append(f"something inside a container could not be examined for round shapes "
                                f"({type(e).__name__}: {e}); it is described roughly")
        if not rounds:
            out.append(other(b))
            continue
        out.extend(rounds)
        for rb in rest:
            out.append(other(rb, "under or next to a round thing of the same colour"))
    return out


_HIDDEN_BASE_REACH_M = 0.02


def _extend_towards(footprint: list, towards_xy, d: float) -> list:
    """Footprint swept by d metres towards a point (convex hull, counter-clockwise). Used for things
    whose base is hidden: they may really stand up to about 2 cm nearer to the camera than the
    visible part suggests, and a keep-out built from the footprint must cover that."""
    pts = np.array(footprint, dtype=float)
    c = pts.mean(axis=0)
    v = np.asarray(towards_xy, dtype=float) - c
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return footprint
    both = np.vstack([pts, pts + v / n * d])
    hull = cv2.convexHull((both * 1000.0).astype(np.float32)).reshape(-1, 2) / 1000.0
    return _poly_ccw(hull)


def _split_by_colour(crop_bgr: np.ndarray, M: np.ndarray) -> list:
    """Parts of a blob that have clearly different colours (strong hues are told apart; everything
    weakly coloured is one "light / dull" class). Returns masks, largest first; one mask (or none)
    means there is nothing to split."""
    area = cv2.countNonZero(M)
    if area < 800:
        return []
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hr = _HUE_RED[hsv[..., 0]]
    strong = hsv[..., 1] >= 100
    cls = np.ones(M.shape, np.uint8)                    # 1 = light / dull
    cls[strong] = 2                                     # red
    cls[strong & (hr > 10) & (hr <= 38)] = 3            # orange / yellow
    cls[strong & (hr > 38) & (hr <= 88)] = 4            # green
    cls[strong & ((hr > 88) | (hr < -48))] = 5          # blue
    cls[strong & (hr >= -48) & (hr < -11)] = 6          # purple / pink
    cls[M == 0] = 0
    parts = []
    for c in range(1, 7):
        mc = (cls == c).astype(np.uint8) * 255
        if cv2.countNonZero(mc) < max(500, 0.10 * area):
            continue
        mc = cv2.morphologyEx(mc, cv2.MORPH_OPEN, _ell(5))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mc, connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= max(500, 0.10 * area):
                part = _fill(cv2.morphologyEx((lab == i).astype(np.uint8) * 255, cv2.MORPH_CLOSE, _ell(7)))
                parts.append(cv2.bitwise_and(part, M))
    parts.sort(key=lambda m: -cv2.countNonZero(m))
    return parts


def _base_hidden_by(mask: np.ndarray, others: list) -> bool:
    """True when a good part of the lower edge of `mask` borders another part (which then stands in
    front of it and hides where it meets the table)."""
    if not others:
        return False
    union = np.zeros_like(mask)
    for o in others:
        union = cv2.bitwise_or(union, o)
    has_c, low = _low_edge(mask)
    cols = np.nonzero(has_c)[0]
    if len(cols) == 0:
        return False
    hmax = mask.shape[0]
    below = [bool(np.any(union[min(hmax - 1, low[c] + 1):min(hmax, low[c] + 7), c] > 0)) for c in cols]
    return float(np.mean(below)) >= 0.3


def _cut_reflection(val: np.ndarray, m: np.ndarray, vt: float) -> tuple[np.ndarray, int]:
    """Second line of defence against a mirror reflection that got into an object's mask (VM-1).

    Where an object meets the glossy table there is a thin dark line (contact shadow) between its
    side face and the reflection of that face. If such a line runs through the lower part of the
    mask, clearly darker than what is above AND below it and at least 4 px above the mask's lower
    edge, what lies below it is the reflection: the mask is cut at the line.
    val: V channel of the crop; returns (mask, rows removed (median), 0 = nothing cut)."""
    has_c, low = _low_edge(m)
    cols = np.nonzero(has_c)[0]
    if len(cols) < 24:
        return m, 0
    k = max(2, int(0.1 * len(cols)))
    cols = cols[k:-k:2]
    # Columns with the full 35 rows above the lower edge and a neighbour on both sides: all at once
    # (the same numbers as the plain loop below, which is kept for the few columns at a border).
    lo_all = low[cols].astype(np.int64)
    full = (lo_all >= 34) & (cols >= 1) & (cols + 1 < val.shape[1])
    hit = np.zeros(len(cols), bool)
    row = np.zeros(len(cols), np.int64)
    if np.any(full):
        cf, lf = cols[full], lo_all[full]
        rows = (lf - 34)[:, None] + np.arange(35)[None, :]
        prof = val[rows[:, :, None], (cf[:, None] + np.array([-1, 0, 1])[None, :])[:, None, :]].astype(np.float32).mean(axis=2)
        ar = np.arange(len(cf))
        i = 6 + np.argmin(prof[:, 6:31], axis=1)
        ja = (i - 12)[:, None] + np.arange(10)[None, :]             # prof[max(0, i - 12):i - 2]
        va = np.where(ja >= 0, prof[ar[:, None], np.maximum(ja, 0)], np.float32(np.inf)).astype(np.float32)
        va.sort(axis=1)
        na = np.minimum(10, i - 2)
        above = ((va[ar, (na - 1) // 2] + va[ar, na // 2]) * np.float32(0.5)).astype(np.float64)       # the median
        jb = (i + 2)[:, None] + np.arange(5)[None, :]               # prof[i + 2:i + 7]
        vb = np.where(jb <= 34, prof[ar[:, None], np.minimum(jb, 34)], np.float32(np.inf)).astype(np.float32)
        vb.sort(axis=1)
        nb = np.minimum(5, 33 - i)
        below = ((vb[ar, (nb - 1) // 2] + vb[ar, nb // 2]) * np.float32(0.5)).astype(np.float64)
        pi = prof[ar, i]
        hit[full] = (pi <= (0.70 * above).astype(np.float32)) & (pi <= (0.85 * below).astype(np.float32)) & (below <= 1.3 * vt)
        row[full] = lf - 34 + i
    dips = []
    for j, c in enumerate(cols):
        if full[j]:
            if hit[j]:
                dips.append((c, int(row[j])))
            continue
        lo_r = int(low[c])
        r0 = max(0, lo_r - 34)
        prof = val[r0:lo_r + 1, max(0, c - 1):c + 2].astype(np.float32).mean(axis=1)
        if len(prof) < 18:
            continue
        seg = prof[6:len(prof) - 4]
        i = 6 + int(np.argmin(seg))
        above = float(np.median(prof[max(0, i - 12):i - 2]))
        below = float(np.median(prof[i + 2:i + 7]))
        if prof[i] <= 0.70 * above and prof[i] <= 0.85 * below and below <= 1.3 * vt:
            dips.append((c, r0 + i))
    if len(dips) < 0.6 * len(cols) or len(dips) < 10:
        return m, 0
    d = np.array(dips, dtype=float)
    coef = np.polyfit(d[:, 0], d[:, 1], 1)
    res = np.abs(np.polyval(coef, d[:, 0]) - d[:, 1])
    inl = d[res <= max(2.5, 2.0 * float(np.median(res)))]
    if len(inl) < 0.6 * len(cols) or len(inl) < 10:
        return m, 0
    coef = np.polyfit(inl[:, 0], inl[:, 1], 1)
    out = m.copy()
    all_cols = np.nonzero(has_c)[0]
    cut = np.rint(np.polyval(coef, all_cols)).astype(int) + 2
    removed = []
    for c, r in zip(all_cols, cut):
        if 0 <= r < out.shape[0] and r <= low[c]:
            out[r:, c] = 0
            removed.append(low[c] - r + 1)
    if not removed or cv2.countNonZero(out) < 150:
        return m, 0
    return _fill(out), int(round(float(np.median(removed))))


# --------------------------------------------------------------------------- loose objects
def _object_mask(img: np.ndarray, ctx: dict, p: dict, bbox_s, comp_s: np.ndarray, blocked_s: np.ndarray):
    """Full-resolution mask of one blob found at reduced resolution.
    bbox_s = (x, y, w, h) and comp_s = its mask, both in the reduced image; blocked_s = reduced
    mask of pixels that may not belong to an object (arm, containers, outside the workspace).
    Returns (crop_bgr, mask, (x0, y0), rows cut off as a mirror reflection) or None."""
    s = ctx["scale"]
    x, y, w, h = bbox_s
    pad = 5
    xs0, ys0 = max(0, x - pad), max(0, y - pad)
    xs1, ys1 = min(blocked_s.shape[1], x + w + pad), min(blocked_s.shape[0], y + h + pad)
    k = int(round(1.0 / s))
    x0, y0, x1, y1 = xs0 * k, ys0 * k, min(img.shape[1], xs1 * k), min(img.shape[0], ys1 * k)
    crop = img[y0:y1, x0:x1]
    size = (x1 - x0, y1 - y0)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    nt = _nontable(hsv, ctx["vt"], p).astype(np.uint8) * 255
    blocked = cv2.resize(blocked_s[ys0:ys1, xs0:xs1], size, interpolation=cv2.INTER_NEAREST)
    nt[blocked > 0] = 0
    nt = cv2.morphologyEx(nt, cv2.MORPH_OPEN, _ell(5))
    seed_s = np.zeros((ys1 - ys0, xs1 - xs0), np.uint8)
    seed_s[y - ys0:y - ys0 + h, x - xs0:x - xs0 + w] = comp_s
    seed = cv2.resize(seed_s, size, interpolation=cv2.INTER_NEAREST)
    n, lab = cv2.connectedComponents(nt, connectivity=8)
    if n <= 1:
        return None
    counts = np.bincount(lab[seed > 0], minlength=n)
    counts[0] = 0
    best = int(np.argmax(counts))
    if counts[best] == 0:
        return None
    m = (lab == best).astype(np.uint8) * 255
    # drop appendages thinner than ~4 mm (a cable crossing the object) but keep the corners
    core = cv2.morphologyEx(m, cv2.MORPH_OPEN, _ell(15))
    if cv2.countNonZero(core) > 0:
        m = cv2.bitwise_and(m, cv2.dilate(core, _ell(9)))
    m = _fill(cv2.morphologyEx(m, cv2.MORPH_CLOSE, _ell(7)))
    if cv2.countNonZero(m) < 150:
        return None
    m, n_cut = _cut_reflection(hsv[..., 2], m, ctx["vt"])
    return crop, m, (x0, y0), n_cut


def _finish_loose(r: dict, m: np.ndarray, off, out: list, dbg, p: dict, z0: float, containers: list,
                  excl: np.ndarray, excl_near: np.ndarray, comp: np.ndarray, bbox_s, hh: int, cam_xy) -> None:
    """Checks that apply to every modelled loose object; appends it to `out` when it is kept."""
    x, y, w, h = bbox_s
    max_m = p["obj_max_mm"] / 1000.0
    x_lo, x_hi, y_lo, y_hi = p["workspace"]
    cx, cy = r["center"]
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return
    if not (x_lo <= cx <= x_hi and y_lo <= cy <= y_hi):
        return
    if max(r["size"][:2]) > 1.5 * max_m or r["size"][2] > 1.5 * max_m:
        r["notes"].append("larger than the gripper can hold")
        r["confidence"] = min(r["confidence"], 0.3)
    if containers and np.any(excl_near[y:y + h, x:x + w][comp > 0] > 0):
        # where does it touch? Below = its base is hidden behind the container's far wall.
        has_c, low_c = _low_edge(comp)
        cols = np.nonzero(has_c)[0]
        below = [bool(np.any(excl[min(hh - 1, y + low_c[c] + 1):min(hh, y + low_c[c] + 6), x + c] > 0)) for c in cols]
        if np.mean(below) >= 0.3:
            r["notes"].append("its base is hidden behind a container: position, size and height are unreliable "
                              "(the reported numbers assume the visible lower edge stands on the table; its "
                              "footprint is drawn 20 mm longer towards the camera to cover that)")
            r["kind"], r["confidence"] = "unknown", min(r["confidence"], 0.2)
            r["footprint"] = _extend_towards(r["footprint"], cam_xy, _HIDDEN_BASE_REACH_M)
        else:
            r["notes"].append("touches a container in the image: it may be partly hidden, its size and "
                              "position are less certain")
            r["confidence"] = min(r["confidence"], 0.5)
    r["z_bottom"] = z0
    if dbg is not None:
        r["_mask"], r["_off"] = m, off
    out.append(r)


def _find_loose(img: np.ndarray, fc: _FastCam, ctx: dict, p: dict, z0: float, containers: list,
                warnings: list, dbg: dict | None) -> list:
    """Things standing on the table (not containers). Returns dicts ready to become SceneObjects."""
    s = ctx["scale"]
    vt = ctx["vt"]
    hh, ww = ctx["nontable"].shape
    excl = np.zeros((hh, ww), np.uint8)
    for c in containers:
        uv = (c["outer_uv"] + 0.5) * s - 0.5
        cv2.fillConvexPoly(excl, cv2.convexHull(uv.astype(np.float32)).astype(np.int32), 255)
    if containers:
        excl = _dilate_ell(excl, 11)
    nt = ctx["nontable"].copy()
    nt[excl > 0] = 0
    op = _open_ell(nt, 7)
    blocked = ((ctx["roi"] == 0) | (ctx["arm"] > 0) | (excl > 0)).astype(np.uint8) * 255
    arm_near = _dilate_ell(ctx["arm"], 9) if np.any(ctx["arm"]) else ctx["arm"]
    excl_near = _dilate_ell(excl, 7) if containers else excl
    roi_edge = cv2.erode(ctx["roi"], _ell(5))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(op, connectivity=8)
    min_px = p["obj_min_mm"] / 1000.0
    max_m = p["obj_max_mm"] / 1000.0
    x_lo, x_hi, y_lo, y_hi = p["workspace"]
    out, n_arm, n_cut, big = [], 0, 0, []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        if area < 220 or w < 9 or h < 9:
            continue
        comp = (lab[y:y + h, x:x + w] == i).astype(np.uint8) * 255
        thick = 2.0 * float(cv2.distanceTransform(cv2.copyMakeBorder(comp, 1, 1, 1, 1, cv2.BORDER_CONSTANT),
                                                  cv2.DIST_L2, 3).max())
        if thick < 10.0:
            continue                                        # cable-like
        if x <= 0 or y <= 0 or x + w >= ww:
            continue                                        # cut by the image border: not a whole object
        if y + h >= hh:
            n_cut += 1                                      # cut by the LOWER border: too close to the camera
            continue
        if np.any(roi_edge[y:y + h, x:x + w][comp > 0] == 0):
            continue                                        # crosses the edge of the workspace
        # where it stands and how big it is, roughly
        foot = fc.to_plane(_full_pts([[x + 0.5 * w, y + h - 1]], s), z0)[0]
        if not np.all(np.isfinite(foot)):
            continue
        ppm = fc.px_per_m(foot) * s
        width_m = w / ppm
        if width_m < 0.7 * min_px:
            continue
        if np.any(arm_near[y:y + h, x:x + w][comp > 0] > 0):
            n_arm += 1
            continue
        if width_m > 1.5 * max_m or h / ppm > 3.0 * max_m:
            big.append((foot, width_m))
            continue
        if not (x_lo - 0.08 <= foot[0] <= x_hi + 0.08 and y_lo - 0.08 <= foot[1] <= y_hi + 0.08):
            continue            # stands well outside the workspace (its centre is within 6 cm of this point)
        try:
            res = _object_mask(img, ctx, p, (x, y, w, h), comp, blocked)
            if res is None:
                continue
            crop, m, off, n_refl = res
            r = _model_blob(fc, crop, m, off, z0, vt, outline_fit=False)
        except Exception as e:            # an unforeseen shape must not hide the other objects
            warnings.append(f"a region near x={foot[0]:.3f} y={foot[1]:+.3f} could not be analysed "
                            f"({type(e).__name__}: {e}); it is not reported")
            continue
        results = [(r, m)]
        if r["kind"] == "unknown":
            # Several things of different colour that touch each other in the image form one blob:
            # take them apart by colour. The one in front is complete; one whose lower edge borders
            # another part has its base hidden and stays "unknown" until the front one is gone.
            try:
                parts = _split_by_colour(crop, m)
                if len(parts) >= 2:
                    results = []
                    for k, part in enumerate(parts):
                        rp = _model_blob(fc, crop, part, off, z0, vt)
                        others = [q for j, q in enumerate(parts) if j != k]
                        if _base_hidden_by(part, others):
                            rp["notes"].append("another object stands in front of it and hides its base: position, "
                                               "size and height are unreliable until that object is removed (its "
                                               "footprint is drawn 20 mm longer towards the camera to cover that)")
                            rp["kind"], rp["confidence"] = "unknown", min(rp["confidence"], 0.2)
                            rp["footprint"] = _extend_towards(rp["footprint"], fc.C[:2], _HIDDEN_BASE_REACH_M)
                        else:
                            rp["notes"].append("touches another object in the image (separated by colour)")
                            rp["confidence"] = min(rp["confidence"], 0.7)
                        results.append((rp, part))
                else:
                    results = [(_model_blob(fc, crop, m, off, z0, vt, outline_fit=True), m)]
            except Exception as e:
                warnings.append(f"a region near x={foot[0]:.3f} y={foot[1]:+.3f} could not be split by colour "
                                f"({type(e).__name__}: {e})")
                results = [(r, m)]
        for r, m in results:
            if n_refl:
                r["notes"].append(f"the lowest {n_refl} px of its image were taken as a mirror reflection on the "
                                  "table (cut at the dark contact line)")
                r["confidence"] = min(r["confidence"], 0.85)
            if r["kind"] == "cylinder" and r["size"][2] >= 0.006:
                try:                                            # a cap or cup standing open side up?
                    cav = _cavity(img, fc, r["center"][0], r["center"][1], 0.5 * r["size"][0], z0 + r["size"][2],
                                  valid=(m, off))
                    if cav["hollow"] and 0.3 * r["size"][2] <= cav["depth"] <= r["size"][2] - 0.0003:
                        z_ib = max(z0 + r["size"][2] - cav["depth"], z0 + 0.0005)     # as far down as the inner wall is seen
                        r.update(hollow=True, wall=cav["wall"], z_inner_bottom=z_ib,
                                 inner_radius=(0.5 * r["size"][0] - cav["wall"]) if cav["wall"] else None)
                        r["notes"].append(_hollow_note(cav, r["z_inner_bottom"] - z0))
                    elif cav["hollow"]:
                        r["notes"].append(f"inside its rim there is the pattern of a cavity {cav['depth'] * 1000:.0f} mm deep, "
                                          "which does not fit its height: open or closed top not determined")
                    elif cav.get("judged"):
                        r["hollow"] = False                     # an evenly lit top without any step: closed
                    elif cav.get("faint"):
                        r["notes"].append(f"it may be open at the top: inside the rim there is the pattern of a cavity, but "
                                          f"too faint to count ({cav['contrast']:.1f} brightness levels, 3 are asked for)")
                except Exception as e:
                    warnings.append(f"the top of the {r['color']} cylinder near x={r['center'][0]:.3f} "
                                    f"y={r['center'][1]:+.3f} could not be examined ({type(e).__name__}: {e})")
            _finish_loose(r, m, off, out, dbg, p, z0, containers, excl, excl_near, comp, (x, y, w, h), hh, fc.C[:2])
    if n_arm:
        warnings.append(f"{n_arm} region(s) touching the robot arm in the image were ignored "
                        "(an object there would be partly hidden): move the arm away and look again")
    if n_cut:
        warnings.append(f"{n_cut} thing(s) at the lower image border are cut off (too close to the camera) and "
                        "are not reported")
    for foot, width_m in big[:3]:
        warnings.append(f"large region (about {width_m * 1000:.0f} mm wide) near x={foot[0]:.3f} y={foot[1]:+.3f} "
                        "is not reported as an object (too big to be one)")
    return out


# --------------------------------------------------------------------------- public: perceive
def _hollow_fields(r: dict) -> dict:
    """The optional SceneObject fields: open at the top (with what was measured of the cavity), seen to
    be closed (hollow False), or not judged (nothing set: hollow stays None)."""
    if not r.get("hollow"):
        return {"hollow": False} if r.get("hollow") is False else {}
    return {"hollow": True,
            "inner_radius": None if r.get("inner_radius") is None else float(r["inner_radius"]),
            "wall": None if r.get("wall") is None else float(r["wall"]),
            "z_inner_bottom": None if r.get("z_inner_bottom") is None else float(r["z_inner_bottom"])}


def perceive(img_bgr, cam, cfg, arm_q=None, jaw_q_deg=None, debug=None) -> Scene:
    """Find every loose object standing on the table, every container and what lies in it. See the
    module docstring.

    cam: camera.CameraModel; cfg: robot config (needs cfg["table_z"]; optional cfg["scene"] overrides
    of DEFAULTS); arm_q: the 5 joint angles (rad) so that the arm is masked (None = not in view);
    jaw_q_deg: gripper joint angle in degrees (None = unknown, a wide opening is masked);
    debug: optional dict that receives intermediate results.
    Never returns silently wrong data: anything doubtful is in Scene.warnings or the object's notes."""
    t0 = time.perf_counter()
    _check_image(img_bgr, cam)
    if cfg is None or cfg.get("table_z") is None:
        raise ValueError("scene.perceive: cfg['table_z'] is not set (probe the table first)")
    p = _params(cfg)
    z0 = float(cfg["table_z"])
    fc = _FastCam(cam)
    sc = Scene(objects=[], table_z=z0, stamp=time.time())
    ctx = _segment(img_bgr, fc, p, z0, cfg, arm_q, jaw_q_deg)
    t1 = time.perf_counter()
    if debug is not None:
        debug["ctx"] = ctx
    if ctx["vt"] <= 0 or ctx["table_fraction"] < p["min_table_fraction"]:
        sc.warnings.append(f"the table was not recognised (only {ctx['table_fraction']:.0%} of the workspace has "
                           "the table colour): wrong camera view, lights off, or the view is blocked. "
                           "No objects are reported.")
        sc.timings = {"total": round(time.perf_counter() - t0, 4)}
        return sc
    if ctx["vt"] > p["table_v_max"]:
        sc.warnings.append(f"the image is too bright for the locked exposure (table brightness {ctx['vt']:.0f} of 255, "
                           f"limit {p['table_v_max']:.0f}): light parts clip and may be mis-read; shade the table "
                           "or shorten the exposure (camera.LOCKED) and look again")
    elif ctx["vt"] < p["table_v_min"]:
        sc.warnings.append(f"the image is dark (table brightness {ctx['vt']:.0f} of 255, limit {p['table_v_min']:.0f}): "
                           "dark objects may be missed; switch the light on and look again")
    try:
        containers = _find_containers(img_bgr, fc, ctx, p, z0, sc.warnings)
    except Exception as e:
        sc.warnings.append(f"the search for containers failed ({type(e).__name__}: {e}); no container is reported")
        containers = []
    t2 = time.perf_counter()
    loose = _find_loose(img_bgr, fc, ctx, p, z0, containers, sc.warnings, debug)
    t3 = time.perf_counter()
    loose.sort(key=lambda r: (r["pixel_bbox"][0], r["pixel_bbox"][1]))
    next_id = 1
    for r in loose:
        sc.objects.append(SceneObject(id=next_id, kind=r["kind"], color=r["color"],
                                      center=(float(r["center"][0]), float(r["center"][1])), yaw=float(r["yaw"]),
                                      size=tuple(float(v) for v in r["size"]), z_bottom=z0,
                                      footprint=r["footprint"], pixel_bbox=r["pixel_bbox"],
                                      confidence=float(r["confidence"]), notes=list(r["notes"]), **_hollow_fields(r)))
        next_id += 1
    arm_full = None
    if arm_q is not None and np.any(ctx["arm"]):
        arm_full = cv2.resize(ctx["arm"], (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    contents_dbg = []
    for c in containers:
        cid = next_id
        next_id += 1
        sc.objects.append(SceneObject(id=cid, kind="container", color=c["color"], center=c["center"], yaw=c["yaw"],
                                      size=c["size"], z_bottom=z0, footprint=c["footprint"],
                                      pixel_bbox=c["pixel_bbox"], confidence=c["confidence"], inner=c["inner"],
                                      notes=list(c["notes"])))
        try:
            items = _content_objects(img_bgr, fc, c, ctx["vt"], p, arm_full, sc.warnings)
        except Exception as e:
            sc.warnings.append(f"the inside of container {cid} could not be analysed ({type(e).__name__}: {e})")
            items = []
        for r in items:
            measured = "base_seen" in r                      # an upright round thing (see _round_tops)
            r["notes"].append(f"inside container {cid}" + ("" if measured else ": position and size are approximate"))
            sc.objects.append(SceneObject(id=next_id, kind=r["kind"], color=r["color"],
                                          center=(float(r["center"][0]), float(r["center"][1])), yaw=float(r["yaw"]),
                                          size=tuple(float(v) for v in r["size"]), z_bottom=c["inner"]["floor_z"],
                                          footprint=r["footprint"], pixel_bbox=r["pixel_bbox"],
                                          confidence=float(r["confidence"]), in_container=cid,
                                          notes=list(r["notes"]), **_hollow_fields(r)))
            contents_dbg.append(r)
            next_id += 1
    t4 = time.perf_counter()
    sc.timings = {"segment": round(t1 - t0, 4), "containers": round(t2 - t1, 4), "objects": round(t3 - t2, 4),
                  "contents": round(t4 - t3, 4), "total": round(t4 - t0, 4)}
    if debug is not None:
        debug.update(containers=containers, loose=loose, contents=contents_dbg)
    return sc


# --------------------------------------------------------------------------- public: annotate
_ANN_COLORS = {"red": (60, 60, 255), "blue": (255, 160, 40), "green": (60, 200, 60), "yellow": (0, 220, 255),
               "white": (255, 255, 255), "ivory": (200, 240, 255), "black": (255, 255, 0), "other": (255, 0, 255)}


def _draw_poly(img, uv, color, thick, closed=True):
    uv = np.asarray(uv, dtype=float)
    if not np.all(np.isfinite(uv)) or np.abs(uv).max() > 1e5:
        return
    cv2.polylines(img, [np.round(uv).astype(np.int32)], closed, color, thick, lineType=cv2.LINE_AA)


def annotate(img_bgr, scene: Scene, cam) -> np.ndarray:
    """Copy of the frame with a wireframe and a label "id kind colour LxWxH mm" for every object
    (graspable objects: thick lines; containers: outer box plus the opening; things inside a
    container: thin lines). Meant for a human / LLM glance, not for measuring."""
    fc = _FastCam(cam)
    out = img_bgr.copy()
    for o in scene.objects:
        col = _ANN_COLORS.get(o.color, (255, 0, 255))
        thick = 1 if o.in_container is not None else 2
        cx, cy = o.center
        if not (np.isfinite(cx) and np.isfinite(cy)):
            continue
        if o.kind == "cylinder" or (o.kind == "unknown" and len(o.footprint) > 4):
            r = 0.5 * o.size[0]
            pts = fc.project(cyl_points(cx, cy, r, o.size[2], o.z_bottom))
            n = len(pts) // 2
            _draw_poly(out, pts[:n], col, thick)
            _draw_poly(out, pts[n:], col, thick)
            i0, i1 = int(np.argmin(pts[:n, 0])), int(np.argmax(pts[:n, 0]))
            for i in (i0, i1):
                _draw_poly(out, [pts[i], pts[n + i]], col, thick, closed=False)
            if getattr(o, "hollow", False):                     # the cavity: inner rim and inner bottom
                ri = o.inner_radius if o.inner_radius else max(r - 0.002, 0.5 * r)
                _draw_poly(out, fc.project(cyl_points(cx, cy, ri, 0.0, o.z_top, top_only=True)), (0, 200, 255), 1)
                if o.z_inner_bottom is not None:
                    _draw_poly(out, fc.project(cyl_points(cx, cy, ri, 0.0, o.z_inner_bottom, top_only=True)), (0, 200, 255), 1)
        else:
            c8 = fc.project(box_corners(cx, cy, o.yaw, o.size[0], o.size[1], o.size[2], o.z_bottom))
            _draw_poly(out, c8[:4], col, thick)
            _draw_poly(out, c8[4:], col, thick)
            for i in range(4):
                _draw_poly(out, [c8[i], c8[4 + i]], col, thick, closed=False)
            if o.kind == "container" and o.inner:
                il, iw = o.inner["size"]
                rim = fc.project(box_corners(o.inner["center"][0], o.inner["center"][1], o.inner["yaw"], il, iw,
                                             0.0, o.inner["rim_z"])[:4])
                _draw_poly(out, rim, (0, 255, 255), 1)
        x, y, w, h = o.pixel_bbox
        label = f"{o.id} {o.kind} {o.color} " + "x".join(f"{v * 1000:.0f}" for v in o.size) + "mm"
        if o.in_container is not None:
            label = f"{o.id} {o.color} (in {o.in_container})"
        if getattr(o, "hollow", False):
            label += " open top"
        if o.confidence < 0.5:
            label += " ?"
        org = (int(max(2, min(x, out.shape[1] - 12 * len(label)))), int(max(18, y - 8)))
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)
    for i, wtxt in enumerate(scene.warnings[:4]):
        cv2.putText(out, "! " + wtxt[:110], (10, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, "! " + wtxt[:110], (10, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA)
    return out


# --------------------------------------------------------------------------- public: queries
_KIND_WORDS = {"box": "box", "cube": "box", "block": "box", "brick": "box", "cuboid": "box",
               "cylinder": "cylinder", "cyl": "cylinder", "round": "cylinder", "cap": "cylinder", "can": "cylinder",
               "disc": "cylinder", "container": "container", "bin": "container", "tray": "container",
               "unknown": "unknown"}
_COLOR_WORDS = {"grey": "gray", "cream": "ivory", "beige": "ivory", "off-white": "ivory"}
_QUERY_KEYS = ("color", "kind", "id", "nearest_to", "largest", "in_container", "hollow")


def _describe(scene: Scene) -> str:
    if not scene.objects:
        return "nothing"
    parts = []
    for o in scene.objects:
        where = f" in container {o.in_container}" if o.in_container is not None else ""
        parts.append(f"#{o.id} {o.color} {o.kind}{' (open top)' if getattr(o, 'hollow', False) else ''}{where}")
    return ", ".join(parts)


def find_object(scene: Scene, query: dict) -> SceneObject:
    """The object that matches the query, or LookupError saying why not and what is there.

    query keys (all optional): "color", "kind" ("box" | "cylinder" | "container" | "unknown"; common
    words such as "cube" / "block" / "cap" are understood), "id", "nearest_to": (x, y), "largest": bool,
    "in_container": True to search among the things lying inside containers as well, "hollow": True /
    False to keep only things that were seen to be open / to be closed at the top (a thing whose top
    could not be judged, hollow None, matches neither).
    Without "kind" = "container" only loose objects are considered (never a container, and nothing
    that lies inside one unless "in_container" is set or the "id" is given).
    Several matches and no "nearest_to" / "largest": the most confident one wins, then the one
    nearest to the robot base; the choice is recorded in scene.warnings."""
    q = {k: v for k, v in (query or {}).items() if v is not None}
    bad = [k for k in q if k not in _QUERY_KEYS]
    if bad:
        raise LookupError(f"unknown query key(s) {bad}; allowed: {list(_QUERY_KEYS)}")
    kind = q.get("kind")
    if kind is not None:
        kind = _KIND_WORDS.get(str(kind).strip().lower())
        if kind is None:
            raise LookupError(f"unknown kind {q['kind']!r}; use one of box, cylinder, container, unknown")
    color = q.get("color")
    if color is not None:
        color = str(color).strip().lower()
        color = _COLOR_WORDS.get(color, color)
    if "id" in q:
        try:
            obj = scene.by_id(int(q["id"]))
        except (KeyError, TypeError, ValueError):
            raise LookupError(f"no object with id {q['id']!r}; the scene has: {_describe(scene)}") from None
        if kind is not None and obj.kind != kind:
            raise LookupError(f"object #{obj.id} is a {obj.kind}, not a {kind}")
        if color is not None and not _color_matches(obj, color):
            raise LookupError(f"object #{obj.id} is {obj.color}, not {color}")
        return obj
    if kind == "container":
        pool = [o for o in scene.objects if o.kind == "container"]
    else:
        pool = [o for o in scene.objects if o.kind != "container"]
        if not q.get("in_container"):
            pool = [o for o in pool if o.in_container is None]
        if kind is not None:
            pool = [o for o in pool if o.kind == kind]
    if color is not None:
        pool = [o for o in pool if _color_matches(o, color)]
    if "hollow" in q:
        pool = [o for o in pool if getattr(o, "hollow", None) is bool(q["hollow"])]
    if not pool:
        raise LookupError(f"nothing matches {q}; the scene has: {_describe(scene)}")
    if "nearest_to" in q:
        try:
            x, y = float(q["nearest_to"][0]), float(q["nearest_to"][1])
        except (TypeError, ValueError, IndexError):
            raise LookupError(f"nearest_to must be (x, y) in metres, got {q['nearest_to']!r}") from None
        return min(pool, key=lambda o: math.hypot(o.center[0] - x, o.center[1] - y))
    if q.get("largest"):
        return max(pool, key=lambda o: o.size[0] * o.size[1] * max(o.size[2], 1e-6))
    best = max(pool, key=lambda o: (round(o.confidence, 1), -math.hypot(o.center[0], o.center[1])))
    if len(pool) > 1:
        scene.warnings.append(f"find_object: {len(pool)} objects match {q} (ids {[o.id for o in pool]}); "
                              f"chose #{best.id} (most confident, then nearest to the base)")
    return best


def _color_matches(obj: SceneObject, color: str) -> bool:
    if obj.color == color:
        return True
    # finer names ("orange", "purple", "gray", "pink") are kept in the notes of "other" objects
    return obj.color == "other" and any(n.startswith("colour: " + color) for n in obj.notes)


def _vt_of(img: np.ndarray, fc: _FastCam, p: dict, z0: float, cfg: dict, arm_q, jaw_q_deg) -> dict:
    ctx = _segment(img, fc, p, z0, cfg, arm_q, jaw_q_deg)
    if ctx["vt"] <= 0 or ctx["table_fraction"] < p["min_table_fraction"]:
        raise ValueError("scene: the table was not recognised in this frame "
                         f"(only {ctx['table_fraction']:.0%} of the workspace has the table colour)")
    return ctx


def objects_in_container(img_bgr, cam, cfg, container: SceneObject, arm_q=None, jaw_q_deg=None) -> list:
    """Coloured / light things seen through the opening of `container` (a SceneObject of kind
    "container" from perceive, taken as still standing where it was).

    Returns a list (largest first) of dicts: {"color", "kind", "center": (x, y), "size", "pixel_bbox",
    "area_px", "confidence", "notes", "near_arm"}. Things of the same colour that touch each other in
    the image come back as ONE entry, so compare "area_px" as well as the count when checking that
    something was added.

    arm_q (and jaw_q_deg) mask the arm when it is in front of or above the box; WITHOUT them the white
    gripper itself and anything it holds are reported as contents. "near_arm" is True for an entry
    that lies next to the masked arm: it is partly hidden by the arm, or it is something the gripper
    still holds above the box (a single view cannot tell), so it does not prove that the object lies
    in the box."""
    _check_image(img_bgr, cam)
    if getattr(container, "kind", None) != "container" or not container.inner:
        raise ValueError("objects_in_container: needs a SceneObject of kind 'container' with its 'inner' geometry")
    if cfg is None or cfg.get("table_z") is None:
        raise ValueError("objects_in_container: cfg['table_z'] is not set")
    p = _params(cfg)
    fc = _FastCam(cam)
    ctx = _vt_of(img_bgr, fc, p, float(cfg["table_z"]), cfg, arm_q, jaw_q_deg)
    inner = container.inner
    inner_uv = fc.project(box_corners(inner["center"][0], inner["center"][1], inner["yaw"], inner["size"][0],
                                      inner["size"][1], 0.0, inner["rim_z"])[:4])
    arm_full = None
    if arm_q is not None and np.any(ctx["arm"]):
        arm_full = cv2.resize(ctx["arm"], (img_bgr.shape[1], img_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    out = []
    for b in _content_blobs(img_bgr, inner_uv, ctx["vt"], p, arm_full):
        r = _model_content(fc, b, inner, ctx["vt"])
        out.append({"color": r["color"], "kind": r["kind"], "center": (float(r["center"][0]), float(r["center"][1])),
                    "size": tuple(float(v) for v in r["size"]), "pixel_bbox": r["pixel_bbox"],
                    "area_px": r["area_px"], "confidence": r["confidence"], "notes": r["notes"],
                    "near_arm": r["near_arm"]})
    return out


def object_presence(img_bgr, cam, cfg, obj: SceneObject, arm_q=None, jaw_q_deg=None) -> dict:
    """Is `obj` (from an earlier perceive) still where it was?

    Looks at the image region the object's model covers. Returns {"present": True | False | None,
    "occupied_fraction", "visible_fraction", "color_now", "reason"}; present is None when more than
    half of the region is hidden by the arm (cannot tell)."""
    _check_image(img_bgr, cam)
    if cfg is None or cfg.get("table_z") is None:
        raise ValueError("object_presence: cfg['table_z'] is not set")
    p = _params(cfg)
    fc = _FastCam(cam)
    ctx = _vt_of(img_bgr, fc, p, float(cfg["table_z"]), cfg, arm_q, jaw_q_deg)
    cx, cy = obj.center
    if obj.kind == "cylinder" or len(obj.footprint) > 4:
        pts = cyl_points(cx, cy, 0.5 * obj.size[0], obj.size[2], obj.z_bottom)
    else:
        pts = box_corners(cx, cy, obj.yaw, obj.size[0], obj.size[1], obj.size[2], obj.z_bottom)
    uv = fc.project(pts)
    h, w = img_bgr.shape[:2]
    res = {"present": None, "occupied_fraction": 0.0, "visible_fraction": 0.0, "color_now": None, "reason": ""}
    if not np.all(np.isfinite(uv)):
        res["reason"] = "the object's model cannot be projected into the image"
        return res
    x0, y0 = int(max(0, np.floor(uv[:, 0].min()) - 4)), int(max(0, np.floor(uv[:, 1].min()) - 4))
    x1, y1 = int(min(w, np.ceil(uv[:, 0].max()) + 5)), int(min(h, np.ceil(uv[:, 1].max()) + 5))
    if x1 - x0 < 8 or y1 - y0 < 8:
        res["reason"] = "the place where the object was is outside the image"
        return res
    crop = img_bgr[y0:y1, x0:x1]
    sil = _hull_mask(fc, pts, crop.shape[:2], np.array([x0, y0], dtype=float))
    n_sil = cv2.countNonZero(sil)
    if n_sil < 50:
        res["reason"] = "the place where the object was is (almost) outside the image"
        return res
    if arm_q is not None and np.any(ctx["arm"]):
        k = int(round(1.0 / ctx["scale"]))
        arm = cv2.resize(ctx["arm"][y0 // k:(y1 + k - 1) // k, x0 // k:(x1 + k - 1) // k], None, fx=k, fy=k,
                         interpolation=cv2.INTER_NEAREST)
        arm = arm[y0 - (y0 // k) * k:, x0 - (x0 // k) * k:][:crop.shape[0], :crop.shape[1]]
        vis = cv2.bitwise_and(sil, cv2.bitwise_not(arm))
    else:
        vis = sil
    n_vis = cv2.countNonZero(vis)
    res["visible_fraction"] = n_vis / n_sil
    if n_vis < 0.5 * n_sil:
        res["reason"] = "more than half of the place is hidden by the arm"
        return res
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    if obj.in_container is None:
        occ = _nontable(hsv, ctx["vt"], p)
    else:
        occ = _content_classes(hsv, ctx["vt"], p) > 0
    occ = occ & (vis > 0)
    n_occ = int(np.count_nonzero(occ))
    res["occupied_fraction"] = n_occ / n_vis
    res["present"] = bool(n_occ >= 0.35 * n_vis)
    if n_occ >= 60:
        res["color_now"] = color_name(crop[occ], ctx["vt"])[0]
    if res["present"]:
        res["reason"] = "the place is still occupied" + (
            "" if res["color_now"] in (obj.color, None) else f" (now {res['color_now']}, was {obj.color})")
    else:
        res["reason"] = "the place shows the bare table" if obj.in_container is None else "the place shows the container"
    return res


def object_present(img_bgr, cam, cfg, obj: SceneObject, arm_q=None) -> bool:
    """True when `obj` is still where perceive saw it; False when its place is empty.
    When the arm hides the place (cannot tell) the answer is True, so that "the object is gone" is
    only ever reported when it was actually seen to be gone; use object_presence() for the details."""
    r = object_presence(img_bgr, cam, cfg, obj, arm_q=arm_q)
    return True if r["present"] is None else bool(r["present"])


# --------------------------------------------------------------------------- jaws in the image
# Convex-hull vertices of the distal 45 mm of both jaws in gripper_frame_link coordinates (mm), taken
# from the official meshes (wrist_roll_follower_so101_v1.stl, moving_jaw_so101_v1.stl; the moving jaw
# at joint angle 0). Their projected hull is each finger's outline to about 0.6 mm.
_JAW_FIXED_MM = np.array([
    (1.0, 0.0, 6.3), (3.1, 0.0, 6.3), (1.0, 1.2, 6.0), (0.9, -1.2, 6.0), (3.1, -1.2, 6.0), (3.0, 1.2, 6.0),
    (4.2, -0.6, 5.9), (4.3, 0.8, 5.8), (0.1, 0.4, 5.6), (0.9, -2.2, 5.4), (3.1, -2.2, 5.4), (3.1, 2.2, 5.4),
    (0.7, 2.2, 5.3), (4.7, -1.4, 5.2), (4.2, 1.9, 5.2), (5.3, 0.4, 5.1), (4.0, 2.9, 4.5), (1.0, 3.0, 4.5),
    (0.9, -3.0, 4.4), (5.6, -0.9, 4.4), (5.2, 2.1, 4.3), (3.8, -3.2, 4.1), (4.9, -2.8, 3.9), (5.7, -1.9, 3.7),
    (0.3, 3.3, 3.5), (4.5, 3.5, 3.4), (1.0, -3.9, 3.1), (5.0, -3.6, 2.9), (1.0, 4.2, 2.4), (0.1, -3.6, 2.2),
    (6.3, -2.9, 1.9), (4.9, -4.2, 1.8), (6.6, 2.4, 1.8), (4.0, 4.5, 1.6), (0.9, -4.6, 1.5), (0.3, 4.3, 1.4),
    (5.6, 4.4, 0.9), (0.9, -5.0, 0.2), (0.3, 4.6, 0.1), (6.7, -3.9, 0.0), (6.8, 3.8, 0.0), (5.9, -4.7, -0.1),
    (5.2, 5.0, -0.2), (0.9, -5.4, -1.3), (0.5, 5.2, -1.3), (7.6, -4.0, -1.9), (7.3, 4.5, -1.9),
    (6.7, -5.1, -2.0), (0.3, -5.6, -3.7), (-0.0, 4.9, -3.7), (0.7, 5.9, -3.7), (6.5, -5.9, -4.1),
    (9.2, -4.0, -5.6), (8.8, 5.0, -5.6), (9.3, 3.7, -5.6), (8.5, -5.4, -5.6), (7.1, 6.2, -6.0),
    (2.9, -6.6, -8.5), (10.1, -5.5, -9.3), (9.6, 6.1, -9.4), (2.9, -7.2, -13.7), (2.8, 7.2, -13.7),
    (10.9, 7.5, -17.2), (3.9, -8.2, -33.4), (3.7, 7.8, -33.4), (23.3, -8.4, -45.0), (6.4, -10.0, -45.0),
    (12.6, -10.0, -45.0), (20.1, 10.0, -45.0), (5.4, -9.0, -45.0), (5.5, 9.4, -45.0), (20.8, -10.0, -45.0),
    (5.4, 0.5, -45.0), (23.7, 7.4, -45.0), (22.5, 9.4, -45.0), (23.7, -7.1, -45.0), (22.5, -9.4, -45.0),
])
_JAW_MOVING_Q0_MM = np.array([
    (-18.9, 0.1, 7.3), (-16.7, 0.1, 7.3), (-19.0, 1.3, 7.0), (-19.2, -1.0, 7.0), (-16.7, -1.1, 7.0),
    (-16.7, 1.3, 7.0), (-20.3, 0.2, 6.9), (-20.3, 1.4, 6.6), (-15.8, 0.1, 6.5), (-20.6, -1.3, 6.4),
    (-19.1, -2.1, 6.3), (-19.0, 2.4, 6.2), (-16.8, 2.5, 6.2), (-16.8, -2.2, 6.2), (-21.2, 0.5, 6.2),
    (-15.8, -1.1, 6.0), (-20.6, 2.5, 5.7), (-20.4, -2.4, 5.7), (-21.5, -0.8, 5.6), (-15.8, 2.1, 5.5),
    (-19.5, 3.2, 5.3), (-16.7, 3.3, 5.3), (-15.9, -2.4, 5.1), (-19.9, -3.3, 4.7), (-16.6, -3.4, 4.6),
    (-21.4, 3.0, 4.5), (-21.4, -2.7, 4.5), (-16.7, 3.9, 4.1), (-19.9, 4.1, 3.8), (-21.0, -3.7, 3.7),
    (-22.5, -1.8, 3.7), (-22.5, 2.0, 3.6), (-16.6, -4.0, 3.4), (-15.8, 3.5, 3.1), (-21.8, 3.8, 3.1),
    (-20.7, 4.5, 2.8), (-16.8, 4.6, 2.5), (-22.8, 3.3, 2.4), (-22.8, -3.1, 2.4), (-15.9, -3.9, 2.3),
    (-21.0, -4.5, 2.0), (-16.5, -4.8, 1.2), (-16.4, 4.9, 1.2), (-23.3, -3.7, 1.1), (-23.2, 4.1, 1.1),
    (-22.4, -4.6, 0.9), (-16.4, 5.4, -0.3), (-16.0, -4.8, -0.4), (-23.0, -5.0, -0.5), (-22.0, -5.5, -1.4),
    (-24.9, -3.7, -1.6), (-24.7, 4.5, -1.6), (-23.5, -5.3, -2.0), (-16.5, -5.7, -2.7), (-15.8, 5.0, -2.7),
    (-16.7, 6.0, -2.7), (-22.6, -5.7, -2.8), (-26.1, -4.5, -4.3), (-25.0, -5.7, -4.6), (-25.6, -6.6, -9.1),
    (-28.1, -6.0, -9.8), (-26.7, 7.0, -10.4), (-26.6, -6.8, -10.5), (-18.8, -7.0, -12.7), (-31.4, -5.3, -14.9),
    (-31.6, 4.8, -14.9), (-31.2, 6.1, -15.0), (-30.7, -6.5, -15.2), (-30.5, 7.0, -15.6), (-29.6, -7.1, -15.7),
    (-31.9, -4.3, -15.7), (-32.7, -9.7, -45.0), (-35.0, -7.2, -45.0), (-34.5, -8.4, -45.0),
    (-34.6, 8.6, -45.0), (-32.2, 10.0, -45.0), (-20.7, 10.1, -45.0), (-33.8, 9.5, -45.0), (-19.9, -9.2, -45.0),
    (-35.0, 7.3, -45.0), (-19.8, 9.3, -45.0),
])

_JAW_TIP_MM = 14.0          # the last part of a finger that must be seen to locate its nose
_JAW_PATCH_MM = 30.0        # length of finger used for matching
_JAW_SEARCH_PX = 36         # the jaws are searched within this distance of where kinematics puts them


def _moving_jaw_mm(jaw_q_deg: float) -> np.ndarray:
    """Hull points of the moving jaw (gripper_frame mm) at joint angle jaw_q_deg."""
    c, s = math.cos(math.radians(jaw_q_deg)), math.sin(math.radians(jaw_q_deg))
    dx, dz = _JAW_MOVING_Q0_MM[:, 0] - GM.PIVOT_X, _JAW_MOVING_Q0_MM[:, 2] - GM.PIVOT_Z
    return np.column_stack([GM.PIVOT_X + c * dx - s * dz, _JAW_MOVING_Q0_MM[:, 1], GM.PIVOT_Z + s * dx + c * dz])


def jaw_model(arm_q, jaw_q_deg: float) -> dict:
    """Base-frame geometry (metres) of the two fingers for these joint angles:
    {"fixed": {"pts": hull points N x 3, "dist": distance of each point from the nose along the
    finger (m), "nose": 3-vector}, "moving": {...}, "open_dir": unit xy vector fixed nose -> moving nose}."""
    T = K.fk(np.asarray(arm_q, dtype=float).reshape(5))
    noses = nose_points_gf(jaw_q_deg)
    out = {}
    for name, mm, tip_z in (("fixed", _JAW_FIXED_MM, _JAW_FIXED_MM[:, 2].max()),
                            ("moving", _moving_jaw_mm(jaw_q_deg), _JAW_MOVING_Q0_MM[:, 2].max())):
        ref_z = _JAW_FIXED_MM[:, 2] if name == "fixed" else _JAW_MOVING_Q0_MM[:, 2]
        out[name] = {"pts": T[:3, 3] + (mm / 1000.0) @ T[:3, :3].T, "dist": (tip_z - ref_z) / 1000.0,
                     "nose": _gf_point(T, *noses[name])}
    d = out["moving"]["nose"][:2] - out["fixed"]["nose"][:2]
    n = float(np.linalg.norm(d))
    out["open_dir"] = d / n if n > 1e-6 else np.array([1.0, 0.0])
    return out


def _jaw_evidence(img: np.ndarray, ref: np.ndarray, vt: float) -> np.ndarray:
    """Per pixel: 1.0 = jaw (pale plastic that was not there in the reference frame), 0.0 = not jaw,
    0.5 = cannot tell (the reference already shows something pale here: a white object or glare)."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsr = cv2.cvtColor(ref, cv2.COLOR_BGR2HSV)
    v, vr = hsv[..., 2].astype(np.int16), hsr[..., 2].astype(np.int16)
    like = (hsv[..., 1] <= 75) & (v >= 0.70 * vt)
    like_ref = (hsr[..., 1] <= 75) & (vr >= 0.70 * vt)
    changed = _max_channel(cv2.absdiff(img, ref)) >= 0.18 * vt
    jaw = like & changed & (~like_ref | (v >= vr + 0.10 * vt))
    jaw_u8 = cv2.morphologyEx(jaw.astype(np.uint8), cv2.MORPH_OPEN, _SQ3)
    jaw_u8 = cv2.morphologyEx(jaw_u8, cv2.MORPH_CLOSE, _ell(7))       # screw holes, grooves
    g = np.zeros(img.shape[:2], np.float32)
    g[like & like_ref & (np.abs(v - vr) < 0.10 * vt)] = 0.5
    g[jaw_u8 > 0] = 1.0
    return g


def _poly_mask(uv: np.ndarray, shape_hw, off) -> np.ndarray:
    m = np.zeros(shape_hw, np.uint8)
    q = (uv - off) * 8.0
    if np.all(np.isfinite(q)) and np.abs(q).max() < 8e5:
        cv2.fillConvexPoly(m, cv2.convexHull(np.rint(q).astype(np.int32)), 255, lineType=cv2.LINE_8, shift=3)
    return m


def _subpixel(res: np.ndarray, iy: int, ix: int) -> tuple[float, float]:
    def para(a, b, c):
        den = a - 2.0 * b + c
        return 0.0 if den <= 1e-9 else float(np.clip(0.5 * (a - c) / den, -0.5, 0.5))
    dx = para(res[iy, ix - 1], res[iy, ix], res[iy, ix + 1]) if 0 < ix < res.shape[1] - 1 else 0.0
    dy = para(res[iy - 1, ix], res[iy, ix], res[iy + 1, ix]) if 0 < iy < res.shape[0] - 1 else 0.0
    return dx, dy


def detect_jaws(img_bgr, ref_bgr, cam, arm_q, jaw_q_deg, cfg) -> dict:
    """Where the two jaw noses really are in the image, compared with where the joint angles put them.

    img_bgr: frame with the gripper in view; ref_bgr: a frame of the SAME scene without the arm there
    (e.g. the frame the scene was perceived from); arm_q: 5 joint angles (rad); jaw_q_deg: gripper
    joint angle (deg, from the raw position).

    Each finger's outline (from the official meshes, placed by forward kinematics and the camera
    model) is slid over the "jaw evidence" image (pale pixels that are new with respect to ref_bgr)
    to find the shift that explains it best. A finger whose nose is behind the other finger, or in
    front of something pale that was already there (white object, glare), is reported as not
    visible instead of being guessed.

    Returns {"ok", "fixed": {"uv", "pred_uv", "visible", "quality", "dxy_base", "reason"},
    "moving": {...}, "dxy_base": (dx, dy) | None, "confidence": 0..1, "agree": bool | None,
    "along_only_from_fixed": bool, "jaw_angle_error_deg": float | None, "reason": str}.
    dxy_base = base-frame xy of the gripper at nose height as the camera sees it minus as forward
    kinematics says. With only the MOVING jaw visible the component along the opening direction
    cannot be told apart from an error of the jaw angle, so it is set to zero and only the
    cross component is reported (lower confidence)."""
    out = {"ok": False, "fixed": {"uv": None, "pred_uv": None, "visible": False, "quality": 0.0, "dxy_base": None,
                                  "reason": ""},
           "moving": {"uv": None, "pred_uv": None, "visible": False, "quality": 0.0, "dxy_base": None, "reason": ""},
           "dxy_base": None, "confidence": 0.0, "agree": None, "along_only_from_fixed": False,
           "jaw_angle_error_deg": None, "reason": ""}
    _check_image(img_bgr, cam)
    if ref_bgr is None or getattr(ref_bgr, "shape", None) != img_bgr.shape:
        out["reason"] = "no reference frame of the same size (a frame of the scene without the arm is required)"
        return out
    if arm_q is None or jaw_q_deg is None:
        out["reason"] = "arm_q and jaw_q_deg are required"
        return out
    if cfg is None or cfg.get("table_z") is None:
        out["reason"] = "cfg['table_z'] is not set"
        return out
    p = _params(cfg)
    fc = _FastCam(cam)
    h_img, w_img = img_bgr.shape[:2]
    model = jaw_model(arm_q, float(jaw_q_deg))
    R = _JAW_SEARCH_PX
    uv, nose_uv, depth = {}, {}, {}
    for name in ("fixed", "moving"):
        uv[name] = fc.project(model[name]["pts"])
        nose_uv[name] = fc.project(model[name]["nose"][None, :])[0]
        depth[name] = float(fc.depth(model[name]["nose"][None, :])[0])
        out[name]["pred_uv"] = (float(nose_uv[name][0]), float(nose_uv[name][1]))
    allp = np.vstack([uv["fixed"], uv["moving"]])
    if not np.all(np.isfinite(allp)) or min(depth.values()) < 0.05:
        out["reason"] = "the gripper is not in front of the camera"
        return out
    pad = R + 26
    x0, y0 = int(np.floor(allp[:, 0].min())) - pad, int(np.floor(allp[:, 1].min())) - pad
    x1, y1 = int(np.ceil(allp[:, 0].max())) + pad, int(np.ceil(allp[:, 1].max())) + pad
    cx0, cy0, cx1, cy1 = max(0, x0), max(0, y0), min(w_img, x1), min(h_img, y1)
    if cx1 - cx0 < 40 or cy1 - cy0 < 40:
        out["reason"] = "the gripper is outside the image"
        return out
    # table brightness from the reference frame (arm away there)
    s = float(p["detect_scale"])
    small = cv2.resize(ref_bgr, (int(round(w_img * s)), int(round(h_img * s))), interpolation=cv2.INTER_AREA)
    hsv_s = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    vt, frac = _table_reference(hsv_s, _workspace_mask(fc, p, float(cfg["table_z"]), small.shape, s) > 0)
    if vt <= 0 or frac < p["min_table_fraction"]:
        out["reason"] = "the table was not recognised in the reference frame"
        return out
    off = np.array([cx0, cy0], dtype=float)
    shape = (cy1 - cy0, cx1 - cx0)
    g = _jaw_evidence(img_bgr[cy0:cy1, cx0:cx1], ref_bgr[cy0:cy1, cx0:cx1], vt)
    masks = {n: _poly_mask(uv[n], shape, off) for n in ("fixed", "moving")}
    results = {}
    # Closed (or nearly closed) jaws are one white body in the image: they are located together, so
    # that the nose that sticks out further is compared with the SAME nose of the model.
    closed = float(jaw_q_deg) <= -4.0 and float(np.linalg.norm(nose_uv["fixed"] - nose_uv["moving"])) < 45.0
    out["located_as_one_body"] = bool(closed)
    jobs = [("both", None)] if closed else [("fixed", "moving"), ("moving", "fixed")]
    for name, other in jobs:
        if name == "both":
            info = {"reason": ""}
            pts_tip = np.vstack([uv[n][model[n]["dist"] <= _JAW_TIP_MM / 1000.0] for n in ("fixed", "moving")])
            pts_part = np.vstack([uv[n][model[n]["dist"] <= _JAW_PATCH_MM / 1000.0] for n in ("fixed", "moving")])
            own_mask = cv2.bitwise_or(masks["fixed"], masks["moving"])
            nose_c = 0.5 * (nose_uv["fixed"] + nose_uv["moving"])
            n3 = 0.5 * (model["fixed"]["nose"] + model["moving"]["nose"])
            other_front, ring_r = False, 30.0
        else:
            info = out[name]
            dist = model[name]["dist"]
            pts_tip, pts_part = uv[name][dist <= _JAW_TIP_MM / 1000.0], uv[name][dist <= _JAW_PATCH_MM / 1000.0]
            own_mask, nose_c, n3, ring_r = masks[name], nose_uv[name], model[name]["nose"], 24.0
            other_front = depth[other] < depth[name] - 0.003
        tip = _poly_mask(pts_tip, shape, off)
        n_tip = cv2.countNonZero(tip)
        if n_tip < 30:
            info["reason"] = "the nose is outside the image"
        elif other_front and cv2.countNonZero(cv2.bitwise_and(tip, masks[other])) > 0.45 * n_tip:
            hidden = cv2.countNonZero(cv2.bitwise_and(tip, masks[other])) / n_tip
            info["reason"] = f"its nose is hidden behind the {other} jaw ({hidden:.0%} covered)"
        if info["reason"]:
            if name == "both":
                out["fixed"]["reason"] = out["moving"]["reason"] = info["reason"]
            continue
        part = pts_part - off
        m = 16
        px0, py0 = int(np.floor(part[:, 0].min())) - m, int(np.floor(part[:, 1].min())) - m
        px1, py1 = int(np.ceil(part[:, 0].max())) + m + 1, int(np.ceil(part[:, 1].max())) + m + 1
        res = None
        if px0 - R < 0 or py0 - R < 0 or px1 + R > shape[1] or py1 + R > shape[0]:
            info["reason"] = "too close to the image border to search around the prediction"
        else:
            templ = np.zeros((py1 - py0, px1 - px0), np.float32)
            own = own_mask[py0:py1, px0:px1] > 0
            templ[own] = 1.0
            if other is not None:
                oth = cv2.dilate(masks[other], _ell(17))[py0:py1, px0:px1] > 0
                templ[oth if other_front else (oth & ~own)] = 0.5      # where the other finger may be: no claim
            search = g[py0 - R:py1 + R, px0 - R:px1 + R]
            res = cv2.matchTemplate(search, templ, cv2.TM_SQDIFF)
            iy, ix = np.unravel_index(int(np.argmin(res)), res.shape)
            if ix <= 0 or iy <= 0 or ix >= 2 * R or iy >= 2 * R:
                info["reason"] = (f"no match within {R} px of the prediction (best at the edge of the search): "
                                  "the jaw is not where kinematics puts it, or it is not visible")
                res = None
        if res is None:
            if name == "both":
                out["fixed"]["reason"] = out["moving"]["reason"] = info["reason"]
            continue
        sx, sy = _subpixel(res, iy, ix)
        du, dv = ix - R + sx, iy - R + sy
        gs = search[iy:iy + templ.shape[0], ix:ix + templ.shape[1]]
        # Second step, along the finger only. The long tapered sides dominate the match above, and a
        # finger that looks a fraction of a pixel fatter than its outline (white plastic blooms)
        # pulls the outline towards the tip. Where the finger ENDS is therefore read directly on a
        # few lines through the nose, in the evidence and in the outline alike.
        ys_o, xs_o = np.nonzero(templ == 1.0)
        along_px = 0.0
        if len(xs_o) > 50:
            ncx0, ncy0 = nose_c[0] - off[0] - px0, nose_c[1] - off[1] - py0      # nose in patch coordinates
            ax = np.array([ncx0 - xs_o.mean(), ncy0 - ys_o.mean()])
            ax = ax / max(float(np.linalg.norm(ax)), 1e-9)
            perp = np.array([-ax[1], ax[0]])
            ss = np.arange(-20.0, 13.5, 0.5)       # the patch reaches 16 px beyond the nose
            ends = []
            for wv in (-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0):
                px_ = ncx0 + ss * ax[0] + wv * perp[0]
                py_ = ncy0 + ss * ax[1] + wv * perp[1]
                xi, yi = np.rint(px_).astype(int), np.rint(py_).astype(int)
                ok = (xi >= 0) & (yi >= 0) & (xi < templ.shape[1]) & (yi < templ.shape[0])
                if np.count_nonzero(ok) < len(ss) - 8 or not ok[-1]:
                    continue
                t_line, g_line = templ[yi[ok], xi[ok]], gs[yi[ok], xi[ok]]
                s_ok = ss[ok]
                if np.any(g_line == 0.5) or not np.any(t_line == 1.0) or not np.any(g_line == 1.0):
                    continue
                e_t, e_g = float(s_ok[t_line == 1.0].max()), float(s_ok[g_line == 1.0].max())
                if e_g >= s_ok.max() - 0.5:            # the evidence runs on beyond the tip: not the tip
                    continue
                ends.append(e_g - e_t)
            if len(ends) >= 4 and abs(_med(ends)) <= 6.0:
                along_px = _med(ends)
                # the line reading replaces the along-finger part of the sub-pixel estimate
                sub_along = sx * ax[0] + sy * ax[1]
                du = ix - R + (sx - sub_along * ax[0]) + along_px * ax[0]
                dv = iy - R + (sy - sub_along * ax[1]) + along_px * ax[1]
        claim = templ != 0.5
        known = claim & (gs != 0.5)
        n_claim, n_known = int(np.count_nonzero(claim)), int(np.count_nonzero(known))
        mismatch = float(np.abs(templ - gs)[known].sum()) / max(n_known, 1)
        # the region around the nose decides how well the tip itself is seen
        ncx, ncy = nose_c[0] - off[0] - px0, nose_c[1] - off[1] - py0
        yy, xx = np.ogrid[:templ.shape[0], :templ.shape[1]]
        ring = (((xx - ncx) ** 2 + (yy - ncy) ** 2) <= ring_r ** 2) & claim
        ring_known = ring & (gs != 0.5)
        tip_known = float(np.count_nonzero(ring_known)) / max(int(np.count_nonzero(ring)), 1)
        bg = ring_known & (templ == 0.0)
        inside = ring_known & (templ == 1.0)
        tip_bg_ok = float(np.count_nonzero(bg & (gs == 0.0))) / max(int(np.count_nonzero(bg)), 1)
        tip_in_ok = float(np.count_nonzero(inside & (gs == 1.0))) / max(int(np.count_nonzero(inside)), 1)
        far = res.copy()
        far[max(0, iy - 8):iy + 9, max(0, ix - 8):ix + 9] = np.inf
        distinct = float(far.min() - res[iy, ix]) / max(n_known, 1)
        q_match = float(np.clip((0.20 - mismatch) / 0.12, 0.0, 1.0))
        q_tip = (float(np.clip((tip_known - 0.55) / 0.25, 0.0, 1.0)) * float(np.clip((tip_bg_ok - 0.6) / 0.25, 0.0, 1.0))
                 * float(np.clip((tip_in_ok - 0.6) / 0.25, 0.0, 1.0)))
        q_dist = float(np.clip((distinct - 0.004) / 0.012, 0.0, 1.0))
        quality = q_match * q_tip * q_dist
        # pixel shift -> base-frame xy shift at the nose height
        pj = fc.project(np.array([n3, n3 + [0.001, 0.0, 0.0], n3 + [0.0, 0.001, 0.0]]))
        J = np.column_stack([(pj[1] - pj[0]) / 0.001, (pj[2] - pj[0]) / 0.001])
        try:
            dxy = np.linalg.solve(J, np.array([du, dv]))
        except np.linalg.LinAlgError:
            info["reason"] = "degenerate viewing geometry"
            if name == "both":
                out["fixed"]["reason"] = out["moving"]["reason"] = info["reason"]
            continue
        detail = dict(quality=round(quality, 3), dxy_base=(float(dxy[0]), float(dxy[1])),
                      shift_px=(round(float(du), 2), round(float(dv), 2)), mismatch=round(mismatch, 3),
                      nose_refine_px=round(float(along_px), 2),

                      tip_known=round(tip_known, 3), tip_bg_ok=round(tip_bg_ok, 3), tip_in_ok=round(tip_in_ok, 3),
                      distinct=round(distinct, 4), known_fraction=round(n_known / max(n_claim, 1), 3))
        if tip_known < 0.55:
            detail["reason"] = "its nose is in front of something pale that was already there (white object or glare)"
        elif quality < 0.5:
            detail["reason"] = (f"poor match (mismatch {mismatch:.0%}, tip known {tip_known:.0%}, tip background "
                                f"{tip_bg_ok:.0%}, tip body {tip_in_ok:.0%}, distinctness {distinct:.3f})")
        else:
            detail["visible"] = True
            results[name] = (dxy, quality)
        for n in (("fixed", "moving") if name == "both" else (name,)):
            out[n].update(detail)
            out[n]["uv"] = (float(nose_uv[n][0] + du), float(nose_uv[n][1] + dv))
    o = model["open_dir"]
    o_perp = np.array([-o[1], o[0]])
    out["nose_z"] = {n: float(model[n]["nose"][2]) for n in ("fixed", "moving")}
    if "both" in results:
        d, q = results["both"]
        out.update(ok=True, dxy_base=(float(d[0]), float(d[1])), confidence=round(0.9 * q, 3),
                   reason="jaws closed: both fingers located together as one body")
        return out
    if closed:
        out["reason"] = "the closed gripper could not be located: " + (out["fixed"]["reason"] or "?")
        return out
    if "fixed" in results and "moving" in results:
        df, qf = results["fixed"]
        dm, qm = results["moving"]
        diff = dm - df
        along, cross = float(diff @ o), float(diff @ o_perp)
        out["jaw_angle_error_deg"] = round(math.degrees(along / (GM.R_TIP / 1000.0)), 2)
        out["agree"] = bool(abs(cross) <= 0.003 and abs(along) <= 0.007)
        out["disagreement_mm"] = {"along_opening": round(along * 1000, 2), "across": round(cross * 1000, 2)}
        if out["agree"]:
            # the fixed jaw is rigid with the tool frame: it carries the answer, the moving jaw confirms it
            out.update(ok=True, dxy_base=(float(df[0]), float(df[1])), confidence=round(min(1.0, 0.5 * (qf + qm) + 0.15), 3),
                       reason="both noses seen; they agree")
        else:
            out.update(ok=False, dxy_base=(float(df[0]), float(df[1])), confidence=round(0.4 * min(qf, qm), 3),
                       reason=f"the two noses disagree ({along * 1000:+.1f} mm along the opening direction, "
                              f"{cross * 1000:+.1f} mm across): not trusted")
    elif "fixed" in results:
        df, qf = results["fixed"]
        out.update(ok=True, dxy_base=(float(df[0]), float(df[1])), confidence=round(0.9 * qf, 3),
                   reason="fixed nose seen; moving nose: " + (out["moving"]["reason"] or "not seen"))
    elif "moving" in results:
        dm, qm = results["moving"]
        cross = float(dm @ o_perp)
        d = cross * o_perp
        out.update(ok=True, dxy_base=(float(d[0]), float(d[1])), confidence=round(0.8 * qm, 3),
                   along_only_from_fixed=True,
                   reason="only the moving nose is seen (fixed nose: " + (out["fixed"]["reason"] or "not seen")
                          + "): the offset ALONG the opening direction is not reported, it cannot be told "
                            "apart from a jaw-angle error")
        out["moving_along_mm"] = round(float(dm @ o) * 1000, 2)
    else:
        out["reason"] = "neither nose could be located: fixed: " + (out["fixed"]["reason"] or "?") + "; moving: " \
            + (out["moving"]["reason"] or "?")
    return out


# --------------------------------------------------------------------------- command line (no hardware)
def _main(argv) -> int:
    """python tools/vision/scene.py frame.jpg [annotated.jpg] [--q d1,d2,d3,d4,d5] [--jaw deg] [--ref arm_free.jpg]
    Prints what perceive() sees in a saved frame (and, with --ref, what detect_jaws() measures)."""
    import json

    import camera as C
    if not argv or argv[0] in ("-h", "--help"):
        print(_main.__doc__)
        return 2
    pos = [a for i, a in enumerate(argv) if not a.startswith("--") and (i == 0 or argv[i - 1] not in ("--q", "--jaw", "--ref"))]

    def opt(name):
        return argv[argv.index(name) + 1] if name in argv and argv.index(name) + 1 < len(argv) else None

    img = cv2.imread(pos[0])
    if img is None:
        print(f"cannot read {pos[0]}")
        return 2
    try:
        from robot import load_config
        cfg = load_config()
    except Exception as e:
        print(f"robot.load_config() failed ({e}); a config with table_z is needed")
        return 2
    cam = C.CameraModel.load()
    q = np.radians([float(v) for v in opt("--q").split(",")]) if opt("--q") else None
    jaw = float(opt("--jaw")) if opt("--jaw") else None
    sc = perceive(img, cam, cfg, arm_q=q, jaw_q_deg=jaw)
    def brief(o):
        d = {"id": o.id, "kind": o.kind, "color": o.color, "center": [round(v, 4) for v in o.center],
             "size_mm": [round(v * 1000, 1) for v in o.size], "yaw_deg": round(math.degrees(o.yaw), 1),
             "confidence": round(o.confidence, 2), "in_container": o.in_container, "notes": o.notes}
        if o.hollow:
            d.update(hollow=True, inner_radius_mm=None if o.inner_radius is None else round(o.inner_radius * 1000, 1),
                     wall_mm=None if o.wall is None else round(o.wall * 1000, 1),
                     z_inner_bottom=None if o.z_inner_bottom is None else round(o.z_inner_bottom, 4), z_top=round(o.z_top, 4))
        if o.inner:
            d["inner"] = {k: ([round(float(x), 4) for x in v] if isinstance(v, (tuple, list)) else round(float(v), 4))
                          for k, v in o.inner.items()}
        return d

    out = {"objects": [brief(o) for o in sc.objects], "warnings": sc.warnings, "timings": sc.timings}
    if opt("--ref") and q is not None and jaw is not None:
        ref = cv2.imread(opt("--ref"))
        out["jaws"] = detect_jaws(img, ref, cam, q, jaw, cfg)
    print(json.dumps(out, indent=1))
    if len(pos) > 1:
        cv2.imwrite(pos[1], annotate(img, sc, cam), [cv2.IMWRITE_JPEG_QUALITY, 88])
        print("annotated view:", pos[1])
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(_main(sys.argv[1:]))
