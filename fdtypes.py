"""Shared data types of the fast pick-and-place pipeline: scene -> grasp plan -> skill.

Conventions: robot base frame, metres, radians unless a field name says `_deg`.
+x forward (towards the camera), +y left, +z up. The table is the plane z = table_z.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class SceneObject:
    id: int
    kind: str  # "box" | "cylinder" | "container" | "unknown"
    color: str  # "red" | "blue" | "green" | "yellow" | "white" | "ivory" | "black" | "other"
    center: tuple  # (x, y) centre of the footprint
    yaw: float  # direction of the LONG horizontal axis (boxes / containers); 0.0 for cylinders
    size: tuple  # (length, width, height) with length >= width; cylinders: (diameter, diameter, height)
    z_bottom: float  # z of the surface the object stands on (table_z, or a container floor)
    footprint: list  # polygon [(x, y), ...] counter-clockwise (boxes: 4 corners, cylinders: 16-gon)
    pixel_bbox: tuple  # (x, y, w, h) in full-resolution image pixels
    confidence: float  # 0..1, how well the fitted primitive explains the image
    in_container: Optional[int] = None  # id of the container this object lies inside
    inner: Optional[dict] = None  # containers: {"center": (x, y), "yaw": rad, "size": (L, W), "floor_z": z, "rim_z": z}
    notes: list = field(default_factory=list)
    # Things that are open at the top (a cup, a bottle cap standing on its closed side); filled in by scene.perceive
    # for upright round objects. hollow: True = an inner wall and an inner bottom are seen inside the rim,
    # False = the top is seen to be closed, None = not judged (not looked at, over-exposed, hidden).
    hollow: Optional[bool] = None
    inner_radius: Optional[float] = None  # radius of the cavity at the rim, m (outer radius = size[0] / 2)
    wall: Optional[float] = None  # wall thickness at the rim, m (an upper bound when the notes say so)
    # z down to which the inner wall is seen (the inner bottom, or what stands on it), m; None when the object's own
    # height is not measured (its base is hidden): only the depth below the rim is known then (see the notes)
    z_inner_bottom: Optional[float] = None

    @property
    def z_top(self) -> float:
        return self.z_bottom + self.size[2]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["z_top"] = self.z_top
        return d


@dataclass
class Scene:
    objects: list
    table_z: float
    stamp: float = 0.0
    frame_path: Optional[str] = None
    timings: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)

    def by_id(self, obj_id: int) -> SceneObject:
        for o in self.objects:
            if o.id == obj_id:
                return o
        raise KeyError(obj_id)

    def containers(self) -> list:
        return [o for o in self.objects if o.kind == "container"]

    def graspable(self) -> list:
        """Loose objects standing on the table (not containers, not inside one)."""
        return [o for o in self.objects if o.kind != "container" and o.in_container is None]

    def to_dict(self) -> dict:
        return {"table_z": self.table_z, "stamp": self.stamp, "frame_path": self.frame_path,
                "timings": self.timings, "warnings": self.warnings,
                "objects": [o.to_dict() for o in self.objects]}


@dataclass
class GraspCandidate:
    ok: bool
    reason: str  # why it was rejected ("" when ok)
    jaw_azimuth_deg: float  # base-frame azimuth of the direction fixed jaw -> moving jaw
    roll_deg: float  # URDF wrist_roll that realises it at the grasp pose
    pitch_deg: float  # approach pitch below horizontal (90 = straight down)
    tip_grasp: tuple  # fingertip (Arm.tip_tool point) target while closing, (x, y, z)
    tip_hover: tuple  # fingertip target above the object before the final descent
    width: float  # object size across the jaws
    standoff: float  # planned gap between the fixed pad and the object's face
    open_raw: int  # gripper goal (raw ticks) for the pre-grasp opening
    expected_raw: int  # gripper position (raw ticks) expected when the jaws stall on the object
    clearance_fixed: float  # free space behind the fixed finger (inf if nothing is near)
    clearance_moving: float  # free space beyond the opened moving jaw
    score: float  # higher is better
    object_offset: tuple = (0.0, 0.0)  # object centre minus fingertip (x, y) while held, at the grasp pose
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ReleasePlan:
    ok: bool
    reason: str
    tip_above: tuple  # fingertip pose above the container before lowering, (x, y, z)
    tip_release: tuple  # fingertip pose at which the gripper opens
    pitch_deg: float
    roll_deg: float
    open_raw: int  # gripper goal that lets the object go
    drop_height: float  # distance the object falls
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
