#!/usr/bin/env python3
"""so101_gripper_model.py - closed-form SO-101 follower gripper model (no meshes needed at run time).

Every constant below was measured from the official TheRobotStudio/SO-ARM100 meshes
(Simulation/SO101/so101_new_calib.urdf + assets) by scripts 01..11 in this folder.

Frame  : URDF link `gripper_frame_link`  (x = toward the FIXED jaw body, z = approach / out of the fingertips,
         y = jaw-pivot axis direction reversed, i.e. the jaw rotates about -y for +q).
Units  : millimetres and degrees.
q      : URDF joint `gripper` (moving jaw).  q = 0 -> inner faces of both jaws exactly parallel, 15.80 mm apart
         at the tip pads.  q > 0 opens (moving jaw swings toward -x).  Tips touch at q = Q_CLOSED.
"""
import math

# ---------------------------------------------------------------- frames
# gripper_link -> gripper_frame_link (URDF fixed joint `gripper_frame_joint`): xyz (m) and rpy
GL_T_GF_XYZ_MM = (-7.9, -0.218121, -98.1274)
GL_T_GF_RPY = (0.0, math.pi, 0.0)            # => x_gf = -x_gl, y_gf = y_gl, z_gf = -z_gl
# wrist_roll axis (= gripper_link z axis) expressed in gripper_frame: the line {x = -7.9, y = +0.218}, parallel to z
ROLL_AXIS_X, ROLL_AXIS_Y = -7.9, 0.218121
# moving-jaw pivot axis in gripper_frame: through (PIVOT_X, *, PIVOT_Z), direction (0, -1, 0)
PIVOT_X, PIVOT_Z = -28.1002, -74.7273
PIVOT_Y_JOINT_ORIGIN = 19.0181               # horn face (joint frame origin) on that axis

# ---------------------------------------------------------------- fixed jaw (part of wrist_roll_follower_so101_v1)
FIXED_TIP_Z = 6.298                          # most distal point of the fixed jaw (nose, x ~ 0.7..3.4, y ~ 0)
PALM_Z = -61.08                              # palm wall (plane z = const, x in [-9.57, 8.10], y in [-19.4, 19.8])
# inner staircase, flat faces x = const facing -x:  (x, z_lo, z_hi, y_half_width)
FIXED_STEPS = [(0.00, -3.70, 5.38, 4.9), (2.00, -13.70, -3.70, 6.2), (3.70, -33.36, -13.70, 7.7),
               (5.40, -53.01, -33.36, 10.9), (7.10, -61.08, -53.01, 14.9)]


def fixed_back_face_x(z):
    """outer (back) face of the fixed finger, valid for z in [-55, 5]:  plane inclined 19.8 deg, capped at 27.30"""
    return min(27.30, 7.47 - 0.3638 * z)


# ---------------------------------------------------------------- moving jaw (moving_jaw_so101_v1) at q = 0
# inner staircase, flat faces x = const facing +x:  (x, z_lo, z_hi, y_half_width)
MOVING_STEPS_Q0 = [(-15.80, -2.73, 6.31, 4.9), (-17.80, -12.73, -2.73, 6.1), (-19.80, -52.73, -12.73, 12.0),
                   (-18.10, -67.30, -53.64, 23.0)]
PAD_FRONT_Q0 = (-15.80, 6.31)                # A' : distal end of the moving tip pad
PAD_BACK_Q0 = (-15.80, -2.73)                # A  : proximal (back) corner of the moving tip pad
TOOTH_B_Q0 = (-17.80, -12.73)                # B  : back corner of the 2nd level
HEEL_C_Q0 = (-18.10, -53.64)                 # C  : front corner of the heel plate
R_TIP, PHI_TIP = 82.784, -7.931              # farthest jaw point from the pivot axis: radius, polar angle at q=0
JAW_Y_MIN, JAW_Y_MAX = -23.88, 24.12         # hub / fork width (48 mm); finger is <= 22.8 mm wide for r > 25 mm

# ---------------------------------------------------------------- joint limits
Q_CLOSED = -11.41          # first tip-to-tip contact in CAD (moving pad front edge on the fixed pad at z ~ 2.4)
Q_URDF_MIN, Q_URDF_MAX = -10.0, 100.0        # URDF <limit> (-0.174533 .. 1.74533 rad); gap at -10 deg = 1.91 mm
Q_OPEN_STOP_CAD = 115.0    # jaw fork arm meets the body's back wall (0.3 mm interference at 116, 1.1 mm at 120)


def jaw_point(q, pt_q0):
    """position (x, z) at joint angle q [deg] of a jaw point given at q = 0."""
    c, s = math.cos(math.radians(q)), math.sin(math.radians(q))
    dx, dz = pt_q0[0] - PIVOT_X, pt_q0[1] - PIVOT_Z
    return (PIVOT_X + c * dx - s * dz, PIVOT_Z + s * dx + c * dz)


def tip_opening(q):
    """perpendicular distance [mm] of the moving tip pad from the fixed tip-pad plane x = 0:
    (at pad back corner A, at pad front end A')."""
    return (-jaw_point(q, PAD_BACK_Q0)[0], -jaw_point(q, PAD_FRONT_Q0)[0])


def q_contact_pad_corner(w):
    """q [deg] at which the pad back corner A touches a face at x = -w  (block flush on the fixed pad).
    Valid contact model for blocks whose top face is 'above' A but not more than ~3-5 mm above it
    (see report); exact for w = 16..45 mm with 10..15 mm engagement."""
    return math.degrees(math.acos((28.1002 - w) / 73.0404)) - 80.3050


def q_contact_block_edge(w, z_top, level_dx=8.3002):
    """q [deg] at which the block's top edge (x = -w, z = z_top) meets a flat jaw face.
    level_dx = 12.3002 (tip pad), 10.3002 (2nd level) or 8.3002 (long 3rd level)."""
    dxc, dzc = -w - PIVOT_X, z_top - PIVOT_Z
    rc = math.hypot(dxc, dzc)
    return math.degrees(math.atan2(dzc, dxc)) - math.degrees(math.acos(level_dx / rc))


# inner (gripping-side) profile of the moving jaw at q = 0 as an (x, z) polyline, heel -> nose
INNER_PROFILE_Q0 = [(-18.10, -67.30), (-18.10, -53.64), (-19.80, -52.73), (-19.80, -12.73), (-17.80, -12.73),
                    (-17.80, -2.73), (-15.80, -2.73), (-15.80, 6.31), (-16.09, 6.98), (-16.80, 7.27)]


def jaw_max_x(q, z_lo, z_hi=1e9):
    """most protruding x (toward the fixed jaw) of the moving jaw's inner profile inside z_lo <= z <= z_hi;
    None if the jaw has no inner surface in that zone."""
    pts = [jaw_point(q, p) for p in INNER_PROFILE_Q0]
    best = None
    for (x0, z0), (x1, z1) in zip(pts[:-1], pts[1:]):
        for zc in (z_lo, z_hi):
            if (z0 - zc) * (z1 - zc) < 0:
                xs = x0 + (zc - z0) / (z1 - z0) * (x1 - x0)
                best = xs if best is None else max(best, xs)
        for x, z in ((x0, z0), (x1, z1)):
            if z_lo <= z <= z_hi:
                best = x if best is None else max(best, x)
    return best


def clear_width(q, z_top, z_bottom=8.3):
    """widest flat-faced block (one face on the fixed tip pad x = 0) spanning z in [z_top, z_bottom] that fits
    between the jaws at joint angle q.  inf if the moving jaw does not reach the zone."""
    xm = jaw_max_x(q, z_top, z_bottom)
    return float("inf") if xm is None else -xm


def contact_angle(w, z_top, z_bottom=8.3, step=0.25):
    """joint angle [deg] at which the closing jaw first touches a block of width w whose top face is at z_top
    (z_top = 6.298 - engagement).  Scan from closed, then bisect."""
    q, prev = Q_CLOSED, None
    while q <= 114.0:
        if clear_width(q, z_top, z_bottom) >= w:
            if prev is None:
                return q
            a, b = prev, q
            for _ in range(40):
                m = 0.5 * (a + b)
                if clear_width(m, z_top, z_bottom) >= w:
                    b = m
                else:
                    a = m
            return b
        prev, q = q, q + step
    return float("nan")


def moving_tip_depth(q):
    """z of the most distal point of the moving jaw (how far it reaches beyond the gripper_frame origin along +z).
    Exact (<= 0.001 mm vs mesh) for -12 <= q <= 64 deg: the nose is a flat end between a R1.0 fillet
    (centre (-16.800, 6.273) at q=0) and a R3.0 arc (centre (-18.928, 4.275) at q=0)."""
    c, s = math.cos(math.radians(q)), math.sin(math.radians(q))
    return PIVOT_Z + max(s * 11.3004 + c * 81.0007 + 0.9992, s * 9.1725 + c * 79.0018 + 2.9978)


def lowest_point_depth(q):
    """depth [mm] below the gripper_frame origin of the lowest gripper point for a VERTICAL (z down) approach.
    = 6.298 (fixed tip) for q <= -4.54 or q >= 19.85 ; up to 8.057 (moving tip) at q = 8.0."""
    return max(FIXED_TIP_Z, moving_tip_depth(q))


def jaw_reach_minus_x(q):
    """how far [mm] the moving jaw extends to the -x side of the gripper_frame origin (outer face), q in [0, 100]."""
    # outer-face extreme points at q = 0 (convex hull of the back of the jaw)
    pts = [(-21.38, 6.6), (-31.93, -15.0), (-36.07, -54.5), (-38.10, -74.7273), (-16.678, 7.265)]
    return -min(jaw_point(q, p)[0] for p in pts)


# ---------------------------------------------------------------- LeRobot value <-> ticks <-> q
RANGE_MIN, RANGE_MAX = 1991, 3614            # my_follower_arm.json, gripper (id 6), drive_mode 0
TICKS_PER_REV = 4096
DEG_PER_UNIT = (RANGE_MAX - RANGE_MIN) / 100.0 * 360.0 / TICKS_PER_REV     # 1.426465 deg per gripper unit


def ticks_from_value(v):
    """exactly what lerobot writes to Goal_Position (motors_bus.py:896-899, RANGE_0_100, drive_mode 0)."""
    v = min(100.0, max(0.0, v))
    return int((v / 100.0) * (RANGE_MAX - RANGE_MIN) + RANGE_MIN)


def value_from_ticks(t):
    """exactly what lerobot reports (motors_bus.py:863, 867-869)."""
    t = min(RANGE_MAX, max(RANGE_MIN, t))
    return (t - RANGE_MIN) / (RANGE_MAX - RANGE_MIN) * 100.0


def q_from_value(v, v_touch=0.0):
    """URDF gripper angle [deg] for LeRobot gripper value v.
    v_touch = gripper value at which the fingertips just touch (MEASURE IT: command 0 with nothing in the
    jaws and read gripper.pos back).  v_touch = 0 is assumption A of the report; plausible range 0..11.4."""
    return Q_CLOSED + DEG_PER_UNIT * (v - v_touch)


def value_from_q(q, v_touch=0.0):
    return v_touch + (q - Q_CLOSED) / DEG_PER_UNIT


# ---------------------------------------------------------------- top-down grasp helper
WRIST_ROLL_SKEW_DEG = 2.789                  # URDF wrist_roll joint origin carries rpy pitch 0.0486795 rad


def topdown_gripper_frame_pose(block_center_mm, w, s, psi_deg, e=None, standoff=0.0):
    """base_T_gripper_frame (4x4, mm) for a vertical (z_gf = -z_base) pinch of a block.
    block_center_mm : (x, y, z) of the block centre in the base frame
    w   : block size along the pinch axis ; s : block height (along base z)
    psi_deg : azimuth (CCW from base +x, seen from above) toward which the MOVING jaw opens;
              the fixed jaw sits on the opposite side.
    e   : engagement = how far the fixed fingertip reaches below the block's top face (default min(s-4, 16))
    standoff : extra gap between fixed pad and block face (pre-grasp), mm
    Resulting origin height above the surface the block stands on:  h = s - e + 6.298
    """
    if e is None:
        e = min(s - 4.0, 16.0)
    c, sn = math.cos(math.radians(psi_deg)), math.sin(math.radians(psi_deg))
    z_bc = FIXED_TIP_Z - e + s / 2.0         # z of the block centre in gripper_frame
    ox = block_center_mm[0] - (w / 2.0 + standoff) * c
    oy = block_center_mm[1] - (w / 2.0 + standoff) * sn
    oz = block_center_mm[2] + z_bc
    return [[-c, -sn, 0.0, ox],
            [-sn, c, 0.0, oy],
            [0.0, 0.0, -1.0, oz],
            [0.0, 0.0, 0.0, 1.0]]


def wrist_roll_for_azimuth(psi_deg, pan_deg):
    """URDF wrist_roll [deg] that makes the moving jaw open toward azimuth psi in a top-down pose
    (psi_open = roll - pan - 2.789).  Returns (roll, flipped): if the direct solution violates the URDF
    limits [-157.21, +162.79] the jaws are swapped (psi + 180)."""
    def wrap(a):
        return (a + 180.0) % 360.0 - 180.0
    roll = wrap(psi_deg + pan_deg + WRIST_ROLL_SKEW_DEG)
    if -157.21 <= roll <= 162.79:
        return roll, False
    return wrap(roll + 180.0), True


# ================================================================ additions for fable_direct v2 (owner: KIN)
# Same conventions as everything above: frame = URDF `gripper_frame_link` (x toward the fixed jaw body, z out of
# the fingertips), millimetres, degrees, q = URDF joint `gripper`.  Numbers marked "mesh" were measured on
# refs/urdf/assets/{moving_jaw,wrist_roll_follower}_so101_v1.stl in 0.25 mm slices; test_kinematics.py re-checks
# every function below against those meshes.

# ---------------------------------------------------------------- distal ends ("noses") of both jaws
FIXED_NOSE = (2.06, 0.0, FIXED_TIP_Z)        # centre of the fixed jaw's rounded end (mesh: z max 6.298 over x 0.69..3.44)
MOVING_NOSE_Q0 = (-17.864, 7.273)            # (x, z) centre of the moving jaw's flat end face at q = 0 (x -18.93..-16.80)
MOVING_PAD_CENTER_Q0 = (-15.80, 1.79)        # (x, z) centre of the moving tip pad face, midway between A and A', at q = 0
MOVING_FINGER_Y = 0.124                      # y of the moving finger's centre plane (the fixed finger's is -0.002)


def nose_points(q_jaw_deg):
    """Distal ends of both jaws: {"fixed": (x, y, z), "moving": (x, y, z)}, gripper_frame_link, mm.
    q_jaw_deg = URDF gripper angle [deg].
    fixed  : centre of the fixed jaw's rounded end; constant, 6.298 mm beyond the frame origin along +z.
    moving : centre of the moving jaw's flat end face, a material point that turns with the jaw.
    Both ends are rounded and about 3 mm across, so the most distal material point is close to, not at, these
    points: the moving jaw's exact most distal z is moving_tip_depth(q), which lies beyond the returned z by
    0.3 mm at q = -11.41, 0.0 at 0, 0.4 at 20, 0.7 at 30, 0.9 at 40 and 1.5 at 64 deg.  Allow 2 mm when the
    returned points are used for a floor / clearance check."""
    xm, zm = jaw_point(q_jaw_deg, MOVING_NOSE_Q0)
    return {"fixed": FIXED_NOSE, "moving": (xm, MOVING_FINGER_Y, zm)}


# ---------------------------------------------------------------- outer envelope of both jaws (clearance checks)
def finger_back_x(z):
    """x [mm] of the outer (back) face of the FIXED finger at tool height z [mm] (gripper_frame_link).
    Alias of fixed_back_face_x: 7.47 - 0.3638*z, capped at 27.30 where the housing begins (z < -54.5); use it
    for z <= 6.3.  The mesh face is 7.29 - 0.364*z, so the value is 0.17 mm on the safe (outer) side, more on
    the rounded last millimetre.  The back face only gets closer to the pad as z grows, so the value at z is
    also the largest x of the finger anywhere from z out to the fingertip: an obstacle whose top is at tool
    height z clears the fixed finger when its x is > finger_back_x(z)."""
    return fixed_back_face_x(z)


OUTER_LINE_TIP = (-24.3558, 0.48879)         # moving jaw back face near the tip, q = 0: x = a + b*z, -15.4 <= z <= 5.6 (mesh)
OUTER_LINE_MID = (-30.2380, 0.10596)         # moving jaw back face of the finger, q = 0: x = a + b*z, -73.7 <= z <= -15.4 (mesh)
NOSE_ARC = (-18.928, 4.275, 2.998)           # R3.0 arc joining the flat end to the back face: centre x, centre z, radius (mesh)
NOSE_FILLET = (-16.800, 6.273, 1.000)        # R1.0 fillet joining the tip pad to the flat end: centre x, centre z, radius (mesh)
HUB_RADIUS = 10.0                            # hub disc around the pivot axis (mesh: 9.99..10.00)


def _arc(cx, cz, r, a0_deg, a1_deg, n):
    """n + 1 points on a circular arc in the (x, z) plane, angle measured from +x toward +z, both ends included."""
    out = []
    for k in range(n + 1):
        a = math.radians(a0_deg + (a1_deg - a0_deg) * k / n)
        out.append((cx + r * math.cos(a), cz + r * math.sin(a)))
    return out


def _build_outline_q0():
    """closed side outline of the moving jaw at q = 0 as an (x, z) polygon: inner profile from the heel to the
    pad's front end A', R1 fillet, flat end, R3 nose arc, the two back-face lines, hub disc, back to the heel.
    Arc vertices lie on the true arcs (chord error <= 0.025 mm)."""
    a_tip = math.degrees(math.atan2(OUTER_LINE_TIP[1], -1.0))     # direction of the outward normal of each back-face line
    a_mid = math.degrees(math.atan2(OUTER_LINE_MID[1], -1.0))
    knee_z = (OUTER_LINE_MID[0] - OUTER_LINE_TIP[0]) / (OUTER_LINE_TIP[1] - OUTER_LINE_MID[1])
    knee = (OUTER_LINE_TIP[0] + OUTER_LINE_TIP[1] * knee_z, knee_z)
    pts = list(INNER_PROFILE_Q0[:8])                              # (-18.10, -67.30) ... A' = (-15.80, 6.31)
    pts += _arc(NOSE_FILLET[0], NOSE_FILLET[1], NOSE_FILLET[2], 15.0, 90.0, 5)
    pts += _arc(NOSE_ARC[0], NOSE_ARC[1], NOSE_ARC[2], 90.0, a_tip, 6)
    pts.append(knee)
    pts += _arc(PIVOT_X, PIVOT_Z, HUB_RADIUS, a_mid, 360.0, 24)
    return pts


OUTLINE_Q0 = _build_outline_q0()


def moving_outline(q):
    """closed side outline [(x, z), ...] of the moving jaw at joint angle q [deg], gripper_frame_link, mm.
    It is the silhouette of the whole jaw seen along y (the finger is <= 22.8 mm wide in y, the hub 48 mm)."""
    return [jaw_point(q, p) for p in OUTLINE_Q0]


def moving_outer_x(q, z):
    """Envelope of the MOVING jaw on its outer (-x) side: the most negative x [mm] reached by any part of the
    jaw at tool heights >= z [mm] (from z out to the fingertip), at joint angle q [deg]; gripper_frame_link.
    None when no part of the jaw lies at or beyond z (the jaw has swung clear of that depth).
    An obstacle whose top is at tool height z clears the opened jaw when its x is < moving_outer_x(q, z).
    Agrees with the mesh silhouette within 0.05 mm for -11.4 <= q <= 90 deg (0.2 mm in the last 0.1 mm of the
    nose) and within 0.5 mm up to 120 deg; add your own margin.  moving_outer_x(q, -1e9) is the overall
    reach: -38.1 for q <= 0, -62.7 at 30, -85.7 at 50, -110.8 at 100 deg (jaw_reach_minus_x, 0.4 mm coarser)."""
    pts = moving_outline(q)
    best = None
    for i, (x0, z0) in enumerate(pts):
        x1, z1 = pts[(i + 1) % len(pts)]
        if z0 >= z:
            best = x0 if best is None else min(best, x0)
        if (z0 - z) * (z1 - z) < 0:                              # this edge crosses the level z
            xs = x0 + (z - z0) / (z1 - z0) * (x1 - x0)
            best = xs if best is None else min(best, xs)
    return best


# ---------------------------------------------------------------- object width <-> joint angle (rigid CAD contact)
W_PARALLEL = 15.80                           # pad-to-pad distance at q = 0 [mm]
W_MAX = 100.0                                # largest width accepted by q_for_width [mm] (kinematic mouth: 110 mm)


def _q_for_point_x(pt_q0, x):
    """q [deg] at which a jaw point given at q = 0 has coordinate x [mm] (the solution on the opening branch)."""
    dx, dz = pt_q0[0] - PIVOT_X, pt_q0[1] - PIVOT_Z
    return math.degrees(math.acos((x - PIVOT_X) / math.hypot(dx, dz)) - math.atan2(dz, dx))


def q_for_width(w_mm):
    """Joint angle q [deg] at which the closing moving jaw first touches an object of width w_mm [mm] that rests
    against the fixed tip pad (plane x = 0): flat-faced block or upright cylinder, RIGID CAD contact at the tip
    pads, no squeeze.  For w >= 15.8 the pad's back corner A bears (same value as q_contact_pad_corner), below
    that its front end A'.  q_for_width(0) = Q_CLOSED within 0.01 deg.
    Exact when the fixed fingertip reaches 10..15 mm below the object's top face; within 0.25 deg for 8..25 mm
    and 0.8 deg for 3..8 mm as long as w <= 32 mm.  For wider objects outside 10..15 mm of engagement use
    contact_angle(w, z_top), which follows the whole jaw profile.
    The REAL jaws stall further closed than this: see stall_raw_for_width.
    Raises ValueError for w outside 0..W_MAX."""
    w = float(w_mm)
    if not 0.0 <= w <= W_MAX:
        raise ValueError(f"q_for_width: width {w_mm!r} mm is outside 0..{W_MAX:.0f} mm")
    return max(_q_for_point_x(PAD_BACK_Q0, -w), _q_for_point_x(PAD_FRONT_Q0, -w))


def width_for_q(q):
    """Inverse of q_for_width: free width [mm] between the two tip pads at joint angle q [deg], i.e. the widest
    object resting on the fixed pad that the moving pad does not yet touch (rigid CAD).  0.0 at and below
    Q_CLOSED."""
    return max(0.0, min(tip_opening(q)))


# ---------------------------------------------------------------- raw servo ticks <-> q (this arm, servo EEPROM frame)
TOUCH_RAW = 2050                             # raw tick at which the closed fingertips touch (free-air close stalled at 2051)
DEG_PER_TICK = 360.0 / TICKS_PER_REV         # 0.087890625 deg of jaw per raw tick


def q_from_raw(raw, touch_raw=TOUCH_RAW):
    """URDF gripper angle q [deg] for a raw gripper position [ticks]:
    q = Q_CLOSED + (raw - touch_raw) * 360 / 4096, touch_raw = raw position at which the fingertips touch
    (config "gripper_touch_raw", 2050 on this arm).  Raw up = jaw opens = q up."""
    return Q_CLOSED + (raw - touch_raw) * DEG_PER_TICK


def raw_from_q(q, touch_raw=TOUCH_RAW):
    """Raw gripper position [ticks, int] for URDF gripper angle q [deg]; inverse of q_from_raw rounded to the
    nearest whole tick (so q_from_raw(raw_from_q(q)) is within 0.044 deg of q)."""
    return int(round(touch_raw + (q - Q_CLOSED) / DEG_PER_TICK))


# ---------------------------------------------------------------- measured squeeze compliance (real arm, 2026-11-10/11)
# Three logged grasps (runs/20261110_230151/events.jsonl, events 96, 125, 151; gripper P gain 16, closing in
# steps until Present_Load saturates at 500):
#   object (width by camera)   rigid contact = raw_from_q(q_for_width(w))   first contact (*)      stall
#   block, 20.4 mm             2221                                        about 2220..2223       2194
#   same block, 22.9 mm        2244                                        about 2220..2223       2194..2195
#   bottle cap, 30 mm          2307                                        about 2305..2310       2263..2265
#   (*) the logged (position, load) pairs of the closing steps extrapolated to zero load
# The first contact agrees with the rigid model to about 1 mm (taking the block as 20.4 mm; the camera's two
# width estimates of it differ by 2.5 mm).  The jaws then travel another 27..50 ticks (2.4..4.4 deg, 3.0..5.6 mm
# at the pads) before the servo stalls, because the printed jaws, the horn and the object give way under the
# squeeze.  A constant offset describes the three grasps as well as anything the data can tell apart.
STALL_COMPLIANCE_TICKS = 40                  # mean of 27 / 50 / 44; expect +-13 ticks (+-1.5 mm) around it


def stall_raw_for_width(w_mm, touch_raw=TOUCH_RAW, compliance_ticks=STALL_COMPLIANCE_TICKS):
    """Raw gripper position [ticks, int] at which the jaws are expected to STALL when they close with saturated
    torque on an object of width w_mm [mm]: rigid contact (q_for_width) minus the measured squeeze compliance,
    never below touch_raw.  Use this, not raw_from_q(q_for_width(w)), to predict / check a grasp.
    Measured minus predicted for the three logged grasps: +13, -10, -4 ticks.  Only measured on 20..30 mm
    objects; a softer squeeze (goal less than about 116 ticks beyond the object at P gain 16) stalls earlier."""
    rigid = touch_raw + (q_for_width(w_mm) - Q_CLOSED) / DEG_PER_TICK
    return int(round(max(float(touch_raw), rigid - compliance_ticks)))


def width_from_stall_raw(raw, touch_raw=TOUCH_RAW, compliance_ticks=STALL_COMPLIANCE_TICKS):
    """Object width [mm] implied by the raw position [ticks] at which the jaws stalled on it (inverse of
    stall_raw_for_width): width_for_q(q_from_raw(raw + compliance_ticks)).  Gives 21.8 mm for the stall at 2194
    (camera: 20.4 and 22.9 mm) and 29.5 mm for 2263 (camera: 30 mm); expect +-1.5 mm."""
    return width_for_q(q_from_raw(raw + compliance_ticks, touch_raw))


if __name__ == "__main__":
    # self-test against mesh-exact numbers produced by 08_opening_table.py / 09_grasp_cases.py
    def close(a, b, tol):
        assert abs(a - b) <= tol, (a, b)

    close(tip_opening(0.0)[0], 15.80, 0.01)
    close(tip_opening(20.0)[0], 41.17, 0.01)
    close(tip_opening(30.0)[1], 57.97, 0.01)
    close(tip_opening(Q_CLOSED)[1], 0.0, 0.03)
    close(tip_opening(-10.0)[1], 1.91, 0.01)
    for w, q in ((25, 7.26), (28, 9.62), (30, 11.18), (32, 12.76), (48, 25.50), (50, 27.14), (56, 32.15), (64, 39.13)):
        close(q_contact_pad_corner(w), q, 0.02)
    close(q_contact_block_edge(48, -39.70), 41.49, 0.02)
    close(q_contact_block_edge(64, -55.70), 73.87, 0.25)       # heel chamfer region: slightly different feature
    close(q_contact_block_edge(48, -3.70, 12.3002), 25.25, 0.02)
    for q, z in ((-11.41, 3.897), (-10, 4.479), (-5, 6.173), (0, 7.273), (8, 8.057), (10, 8.004), (15, 7.437), (20, 6.253),
                 (30, 2.071), (45, -8.461), (60, -23.441)):
        close(moving_tip_depth(q), z, 0.002)
    close(lowest_point_depth(19.85), FIXED_TIP_Z, 0.002)
    close(lowest_point_depth(-4.54), FIXED_TIP_Z, 0.002)
    for q, x in ((10, 42.09), (20, 51.90), (30, 62.66), (40, 74.85), (50, 85.72), (60, 94.93), (70, 102.20), (80, 107.31),
                 (90, 110.10), (100, 110.83)):
        close(jaw_reach_minus_x(q), x, 0.6)
    # polyline contact model vs mesh-exact results (08_opening_table.py / 09_grasp_cases.py)
    for w, zt, qm in ((25, 1.30, 6.97), (25, -3.70, 7.26), (30, -21.70, 11.18), (30, -23.70, 11.49), (32, -23.70, 13.70),
                      (32, -13.70, 12.99), (40, -33.70, 27.38), (48, -39.70, 41.49), (48, -3.70, 25.25), (48, -8.70, 25.50),
                      (50, -41.70, 45.64), (56, -8.70, 32.15), (56, -47.70, 58.25), (60, -53.70, 69.16), (64, -13.70, 39.13),
                      (64, -23.70, 44.27), (64, -55.70, 73.87)):
        zb = 8.30
        close(contact_angle(w, zt, zb), qm, 0.06)
    for q, zr, wm in ((0.0, (-3.70, 6.298), 15.80), (20.0, (-3.70, 6.298), 41.17), (30.0, (-3.70, 6.298), 54.90),
                      (20.0, (-61.03, 6.298), 22.44), (60.0, (-61.03, 6.298), 31.82), (80.0, (-61.03, 6.298), 57.98)):
        close(clear_width(q, zr[0], zr[1]), wm, 0.06)
    close(DEG_PER_UNIT, 1.42646, 1e-5)
    assert ticks_from_value(0) == 1991 and ticks_from_value(100) == 3614 and ticks_from_value(50) == 2802
    close(q_from_value(8.0), 0.0, 0.01)
    close(value_from_q(100.0), 78.1, 0.05)
    T = topdown_gripper_frame_pose((250.0, 0.0, 15.0), 30.0, 30.0, 0.0, e=16.298)
    close(T[2][3], 20.0, 1e-9)
    close(T[0][3], 235.0, 1e-9)
    r, flipped = wrist_roll_for_azimuth(0.0, 0.0)
    close(r, WRIST_ROLL_SKEW_DEG, 1e-9)
    assert flipped is False
    r, flipped = wrist_roll_for_azimuth(170.0, 0.0)      # 172.789 > 162.79 -> swap jaws
    close(r, -7.211, 1e-6)
    assert flipped is True
    # ---- fable_direct v2 additions
    n = nose_points(0.0)
    assert n["fixed"] == (2.06, 0.0, FIXED_TIP_Z)
    close(n["moving"][0], -17.864, 1e-9)
    close(n["moving"][2], 7.273, 1e-9)
    close(n["moving"][2], moving_tip_depth(0.0), 0.001)
    for q in (-11.41, 20.0, 30.0, 64.0):                      # the nose point stays within 1.6 mm of the most distal z
        assert 0.0 <= moving_tip_depth(q) - nose_points(q)["moving"][2] <= 1.6, q
    for z in (-60.0, -30.0, 0.0, 5.0):
        assert finger_back_x(z) == fixed_back_face_x(z)
    for q, x in ((0, 38.10), (10, 42.09), (20, 51.90), (30, 62.66), (40, 74.85), (50, 85.72), (60, 94.93), (70, 102.20),
                 (80, 107.31), (90, 110.10), (100, 110.83)):  # mesh-exact overall reach (08_opening_table.py)
        close(-moving_outer_x(q, -1e9), x, 0.05)
    close(moving_outer_x(0.0, 4.5), -22.16, 0.01)            # mesh slices of the back face at q = 0
    close(moving_outer_x(0.0, -13.0), -30.71, 0.01)
    close(moving_outer_x(0.0, -50.5), -35.59, 0.01)
    assert moving_outer_x(0.0, 7.5) is None and moving_outer_x(45.0, 0.0) is None
    for w, q in ((25, 7.26), (28, 9.62), (30, 11.18), (32, 12.76), (48, 25.50)):
        close(q_for_width(w), q, 0.02)
    close(q_for_width(0.0), Q_CLOSED, 0.01)
    close(q_for_width(W_PARALLEL), 0.0, 1e-6)
    for w in (0.0, 3.0, 10.0, 15.8, 20.4, 30.0, 45.0, 64.0, 100.0):
        close(width_for_q(q_for_width(w)), w, 1e-9)
    assert width_for_q(Q_CLOSED - 1.0) == 0.0
    for bad in (-0.1, 100.1, float("nan")):
        try:
            q_for_width(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(bad)
    close(q_from_raw(2050), Q_CLOSED, 1e-12)
    close(q_from_raw(2194), -11.41 + 144 * 360.0 / 4096.0, 1e-12)
    for raw in (1998, 2050, 2194, 2263, 3648):
        assert raw_from_q(q_from_raw(raw)) == raw and isinstance(raw_from_q(q_from_raw(raw)), int)
        assert raw_from_q(q_from_raw(raw, 2061), 2061) == raw
    # rigid contact predictions and the measured stalls (see STALL_COMPLIANCE_TICKS)
    assert [raw_from_q(q_for_width(w)) for w in (20.4, 23.0, 30.0)] == [2221, 2245, 2307]
    assert [stall_raw_for_width(w) for w in (20.4, 23.0, 30.0)] == [2181, 2205, 2267]
    for w, stall in ((20.4, 2194), (23.0, 2194), (30.0, 2263)):
        assert abs(stall_raw_for_width(w) - stall) <= 13, (w, stall)
    close(width_from_stall_raw(2194), 21.82, 0.01)
    close(width_from_stall_raw(2263), 29.54, 0.01)
    for w in (12.0, 20.4, 30.0, 45.0):
        close(width_from_stall_raw(stall_raw_for_width(w)), w, 0.07)     # whole-tick rounding only
    assert stall_raw_for_width(1.0) == TOUCH_RAW
    print("so101_gripper_model self-test OK")
    print("deg per gripper unit:", DEG_PER_UNIT)
    for v in (0, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 78.1, 88.6, 100):
        q = q_from_value(v)
        print(f"  v={v:6.1f} -> ticks {ticks_from_value(v):4d} -> q(A)={q:7.2f} deg  q(B)={q_from_value(v, 11.36):7.2f}  "
              f"tip opening A/A' (assumption A) = {tip_opening(q)[0]:6.1f}/{tip_opening(q)[1]:6.1f} mm")
