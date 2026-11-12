"""Pick round objects with the planner's grasp (from `fd pick --id N --dry`) and put them on the white round dish."""
import json, subprocess, sys, time, math
FD = '/home/jetson/lerobot/fable_direct/fd'
S = '/tmp/tazuna_scratch/cal/'
T0 = time.time()
def fd(*a):
    t0 = time.time()
    p = subprocess.run([FD, *a], capture_output=True, text=True, timeout=120)
    try:
        d = json.loads(p.stdout)
    except Exception:
        d = {'ok': False, 'message': (p.stdout + p.stderr)[:300]}
    extra = {k: d.get(k) for k in ('contact', 'empty', 'pos_raw', 'width_mm') if k in d}
    print('[%5.1fs] %-6s %4.1fs ok=%s %s %s' % (time.time() - T0, a[0], time.time() - t0, d.get('ok'), extra or '', (d.get('message') or '')[:220]), flush=True)
    return d
frac = lambda raw: '%.4f' % ((raw - 1998.0) / 1650.0)
H_DISH, DROP = 0.005, 0.004
jobs = []                                  # (dry json, place centre x, y, release azimuth)
for spec in sys.argv[1:]:
    f, px, py, az = spec.split(':')
    jobs.append((json.load(open(S + f)), float(px), float(py), float(az)))
done = []
for d, px, py, az in jobs:
    g = d['grasp']; name = '%s %s' % (d['target']['color'], d['target']['kind'])
    tg, th = g['tip_grasp'], g['tip_hover']
    zc = max(0.060, float(d['plan']['z_carry']))
    psi = g['jaw_azimuth_deg']
    off = g['object_offset']
    c, s = math.cos(math.radians(az)), math.sin(math.radians(az))
    tx, ty = px - (c * off[0] - s * off[1]), py - (s * off[0] + c * off[1])     # fingertip so that the object's centre lands on (px, py)
    zr = tg[2] + H_DISH + DROP
    print('== %s: grasp tip %s azimuth %.1f open_raw %d ; put at (%.3f, %.3f) -> fingertip (%.4f, %.4f, %.4f)' % (name, [round(v, 4) for v in tg], psi, g['open_raw'], px, py, tx, ty, zr), flush=True)
    A = ('--pitch', '90', '--jaw-azimuth', '%.2f' % psi)
    fd('grip', frac(g['open_raw']))
    if not fd('tip', str(th[0]), str(th[1]), '%.4f' % th[2], *A, '--speed', '25').get('ok'):
        print('SKIP', name); continue
    if not fd('tip', str(tg[0]), str(tg[1]), '%.4f' % tg[2], *A, '--straight', '--speed', '10').get('ok'):
        fd('tip', str(th[0]), str(th[1]), '%.4f' % th[2], *A, '--straight', '--speed', '12'); print('SKIP', name); continue
    r = fd('grasp')
    if not r.get('ok') or r.get('empty') or r.get('contact') is False:
        print('grasp failed for', name, json.dumps(r)[:300])
        fd('grip', frac(g['open_raw'])); fd('tip', str(th[0]), str(th[1]), '%.4f' % th[2], *A, '--straight', '--speed', '12'); continue
    print('   held: pos_raw %s (expected %s)' % (r.get('pos_raw'), g.get('expected_raw')))
    fd('tip', str(tg[0]), str(tg[1]), '%.4f' % zc, *A, '--straight', '--speed', '12')
    B = ('--pitch', '90', '--jaw-azimuth', '%.2f' % az)
    r = fd('tip', '%.4f' % tx, '%.4f' % ty, '%.4f' % zc, *B, '--speed', '25')
    if not r.get('ok'):
        print('carry refused; holding the object above its place:', name); break
    fd('tip', '%.4f' % tx, '%.4f' % ty, '%.4f' % zr, *B, '--straight', '--speed', '10')
    fd('grip', frac(g['open_raw']))
    fd('tip', '%.4f' % tx, '%.4f' % ty, '%.4f' % zc, *B, '--straight', '--speed', '15')
    done.append(name)
fd('ready')
s = fd('snap', '--name', 'dish', '--plain')
print('[%5.1fs] placed: %s ; photo %s' % (time.time() - T0, done, s.get('raw')))
