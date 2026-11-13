"""Build a tower: pick one object and set it down at (TX, TY) on a support `--support` metres above the table.

  stack.py box <colour> --support H --yaw DEG        (grasp from the harness planner, neighbours respected)
  stack.py cyl X Y R ZG --pitch P --support H        (upright cylinder at X,Y with radius R; fingertip ZG above the table)
  stack.py hub X Y --pitch P --support H             (the blue disc, taken by its hub at X,Y)
The object is released with the same pitch it was grasped with, so it arrives level. A photo is taken at the end."""
import argparse, copy, json, math, os, subprocess, sys, time
HERE = '/home/jetson/lerobot/fable_direct'
sys.path.insert(0, HERE)
import numpy as np, cv2
FD = os.path.join(HERE, 'fd')
T0 = time.time()
TABLE = -0.0015


def fd(*a, quiet=False):
    t0 = time.time()
    p = subprocess.run([FD, *a], capture_output=True, text=True, timeout=120)
    try:
        d = json.loads(p.stdout)
    except Exception:
        d = {'ok': False, 'message': (p.stdout + p.stderr)[:300]}
    if not quiet:
        extra = {k: d.get(k) for k in ('contact', 'empty', 'pos_raw', 'width_mm') if k in d}
        print('[%5.1fs] %-6s %4.1fs ok=%s %s %s' % (time.time() - T0, a[0], time.time() - t0, d.get('ok'), extra or '', (d.get('message') or '')[:260]), flush=True)
    return d


def frac(raw):
    return '%.4f' % ((raw - 1998.0) / 1650.0)


def rot(v, deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def plan_box(colour):
    """the planner's best grasp of the box of that colour, from a fresh photo (no hardware motion)"""
    import camera, grasp as G, robot, scene as S, sim_bus, skills
    s = fd('snap', '--name', 'plan', '--plain')
    st = fd('state', quiet=True)
    cfg = robot.load_config()
    arm = robot.Arm(bus=sim_bus.SimBus(), cfg=copy.deepcopy(cfg)).open()          # simulated arm: planning only
    q = np.radians([s['q_deg'][j] for j in robot.ARM])
    jaw = arm.gripper_q_deg(int(st.get('gripper_raw') or st.get('raw', {}).get('gripper') or 2050))
    sc = S.perceive(cv2.imread(s['raw']), camera.CameraModel.load(), arm.cfg, arm_q=q, jaw_q_deg=jaw)
    cands = [o for o in sc.objects if o.color == colour and (o.kind == KIND if KIND else o.kind in ('box', 'unknown')) and o.in_container is None]
    if not cands:
        raise SystemExit('no %s box seen: %s' % (colour, [(o.id, o.kind, o.color) for o in sc.objects]))
    t = max(cands, key=lambda o: o.confidence)
    print('target: id %d %s %s centre (%.4f, %.4f) size %s yaw %.1f conf %.2f' % (t.id, t.color, t.kind, t.center[0], t.center[1], [round(v * 1000, 1) for v in t.size], math.degrees(t.yaw), t.confidence), flush=True)
    arm.cfg['keepouts'] = {**{k: v for k, v in arm.cfg.get('keepouts', {}).items() if not k.startswith('obj')}, **skills.scene_keepouts(sc, skip_ids=(t.id,))}
    cs = G.plan_grasp(t, sc, arm, arm.cfg)
    for c in cs[:5]:
        print('   candidate az %.1f pitch %.0f width %.1f mm score %.1f ok %s %s' % (c.jaw_azimuth_deg, c.pitch_deg, c.width * 1000, c.score, c.ok, c.reason[:80]), flush=True)
    ok = [c for c in cs if c.ok]
    if not ok:
        raise SystemExit('no feasible grasp')
    return t, ok[0], sc


def execute(tip_hover, tip_grasp, pitch, psi_g, open_raw, off, place, support, psi_r, tag, grasp_args=('grasp',)):
    A = ('--pitch', '%.1f' % pitch, '--jaw-azimuth', '%.2f' % psi_g)
    B = ('--pitch', '%.1f' % pitch, '--jaw-azimuth', '%.2f' % psi_r)
    tx, ty = np.array(place) - rot(off, psi_r - psi_g)
    z_rel = tip_grasp[2] + support + DROP
    z_above = z_rel + ABOVE
    z_lift = tip_hover[2]          # the height the planner checked for this pose
    print('grasp tip (%.4f, %.4f, %.4f) pitch %.0f az %.1f open %d | place tip (%.4f, %.4f, %.4f) az %.1f, above %.3f' % (*tip_grasp, pitch, psi_g, open_raw, tx, ty, z_rel, psi_r, z_above), flush=True)
    fd('grip', frac(open_raw))
    if not fd('tip', '%.4f' % tip_hover[0], '%.4f' % tip_hover[1], '%.4f' % tip_hover[2], *A, '--speed', '25').get('ok'):
        return False
    if not fd('tip', '%.4f' % tip_grasp[0], '%.4f' % tip_grasp[1], '%.4f' % tip_grasp[2], *A, '--straight', '--speed', '10').get('ok'):
        fd('ready'); return False
    r = fd(*grasp_args)
    if not r.get('ok') or r.get('empty') or r.get('contact') is False:
        print('grasp failed', json.dumps(r)[:300]); fd('grip', frac(open_raw))
        fd('tip', '%.4f' % tip_hover[0], '%.4f' % tip_hover[1], '%.4f' % tip_hover[2], *A, '--straight', '--speed', '12'); fd('ready'); return False
    fd('tip', '%.4f' % tip_grasp[0], '%.4f' % tip_grasp[1], '%.4f' % z_lift, *A, '--straight', '--speed', '12')
    if not fd('tip', '%.4f' % tx, '%.4f' % ty, '%.4f' % max(z_above, z_lift if support < 0.02 else z_above), *B, '--speed', '22').get('ok'):
        fd('tip', '%.4f' % tip_grasp[0], '%.4f' % tip_grasp[1], '%.4f' % z_lift, *A, '--straight', '--speed', '8')
        print('carry refused: the object is still held above its pick place'); return False
    if not fd('tip', '%.4f' % tx, '%.4f' % ty, '%.4f' % z_rel, *B, '--straight', '--speed', '8').get('ok'):
        print('lowering refused: the object is still held above the tower'); return False
    fd('grip', frac(open_raw))
    fd('tip', '%.4f' % tx, '%.4f' % ty, '%.4f' % z_above, *B, '--straight', '--speed', '12')
    fd('ready')
    s = fd('snap', '--name', tag, '--plain')
    print('[%5.1fs] photo %s' % (time.time() - T0, s.get('raw')), flush=True)
    return True


ap = argparse.ArgumentParser()
ap.add_argument('mode'); ap.add_argument('args', nargs='*')
ap.add_argument('--support', type=float, default=0.0); ap.add_argument('--yaw', type=float, default=None)
ap.add_argument('--pitch', type=float, default=90.0); ap.add_argument('--tx', type=float, default=0.209); ap.add_argument('--ty', type=float, default=-0.068)
ap.add_argument('--above', type=float, default=0.030); ap.add_argument('--hover', type=float, default=0.050); ap.add_argument('--drop', type=float, default=0.0015); ap.add_argument('--jaws', default='auto'); ap.add_argument('--dry', action='store_true'); ap.add_argument('--on', default=None)
a = ap.parse_args()
T = (a.tx, a.ty)
ABOVE = a.above
DROP = a.drop
KIND = None
if a.mode in ('box', 'obj'):
    KIND = a.args[1] if len(a.args) > 1 else None
    t, c, sc = plan_box(a.args[0])
    if a.on:
        sup = max([o for o in sc.objects if o.color == a.on and o.kind in ('box', 'unknown') and o.id != t.id], key=lambda o: o.confidence)
        T = (float(sup.center[0]), float(sup.center[1])); a.yaw = math.degrees(sup.yaw); a.support = float(sup.size[2])
        print('support: %s box at (%.4f, %.4f) yaw %.1f, top %.1f mm above the table (conf %.2f)' % (a.on, T[0], T[1], a.yaw, a.support * 1000, sup.confidence), flush=True)
    yaw_now = math.degrees(t.yaw)
    d = ((a.yaw - yaw_now + 90.0) % 180.0) - 90.0 if a.yaw is not None else (math.degrees(math.atan2(T[1], T[0])) - math.degrees(math.atan2(t.center[1], t.center[0])))
    # with --yaw: the smallest turn that gives the wanted long-axis direction; without: keep the wrist roll (same jaw direction relative to the arm)
    print('chosen: az %.1f pitch %.0f tip_grasp %s hover %s open %d expected %d offset %s ; turn by %.1f deg' % (c.jaw_azimuth_deg, c.pitch_deg, [round(v, 4) for v in c.tip_grasp], [round(v, 4) for v in c.tip_hover], c.open_raw, c.expected_raw, [round(v, 4) for v in c.object_offset], d), flush=True)
    if not a.dry:
        execute(c.tip_hover, c.tip_grasp, c.pitch_deg, c.jaw_azimuth_deg, c.open_raw, np.array(c.object_offset), T, a.support, c.jaw_azimuth_deg + d, 'stack_' + a.args[0])
elif a.mode == 'cyl':
    x, y, R, zg = [float(v) for v in a.args[:4]]
    az = math.degrees(math.atan2(y, x))
    psi = az + 90.0 if (a.jaws == 'tangential' or (a.jaws == 'auto' and a.pitch < 88)) else az
    u = np.array([math.cos(math.radians(psi)), math.sin(math.radians(psi))])
    tg = np.array([x, y]) - (R + 0.012) * u
    open_raw = int(2050 + 40 + (2 * R * 1000 + 24.0) * 8.9 + 20)
    azT = math.degrees(math.atan2(T[1], T[0]))
    psi_r = psi + (azT - az)                          # keep the wrist roll: same jaw direction relative to the arm
    if not a.dry:
        execute((tg[0], tg[1], a.hover), (tg[0], tg[1], TABLE + zg), a.pitch, psi, open_raw, (R - 0.001) * u, T, a.support, psi_r, 'stack_cyl')
    else:
        print('cyl grasp tip', tg, 'psi', psi, 'open', open_raw)
elif a.mode == 'hub':
    x, y = [float(v) for v in a.args[:2]]
    az = math.degrees(math.atan2(y, x)); u = np.array([math.cos(math.radians(az)), math.sin(math.radians(az))])
    tg = np.array([x, y]) - 0.015 * u
    azT = math.degrees(math.atan2(T[1], T[0]))
    if not a.dry:
        execute((tg[0], tg[1], 0.050), (tg[0], tg[1], 0.0097), a.pitch, az, 2400, 0.006 * u, T, a.support, azT, 'stack_disc')
elif a.mode == 'man':
    gx, gy, zg, psi, open_raw, ox, oy, psi_r = [float(v) for v in a.args[:8]]
    if not a.dry:
        execute((gx, gy, zg + 0.030), (gx, gy, zg), a.pitch, psi, int(open_raw), np.array([ox, oy]), T, a.support, psi_r, 'stack_man')
