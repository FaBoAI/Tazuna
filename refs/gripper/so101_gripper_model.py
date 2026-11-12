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
    print("so101_gripper_model self-test OK")
    print("deg per gripper unit:", DEG_PER_UNIT)
    for v in (0, 5, 8, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 78.1, 88.6, 100):
        q = q_from_value(v)
        print(f"  v={v:6.1f} -> ticks {ticks_from_value(v):4d} -> q(A)={q:7.2f} deg  q(B)={q_from_value(v, 11.36):7.2f}  "
              f"tip opening A/A' (assumption A) = {tip_opening(q)[0]:6.1f}/{tip_opening(q)[1]:6.1f} mm")
