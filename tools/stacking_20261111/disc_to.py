"""Grasp the blue disc by its hub with a tilted gripper and put it on the white dish; photograph right after the release."""
import json, subprocess, sys, time, math
sys.path.insert(0, '/home/jetson/lerobot/fable_direct')
import numpy as np, cv2, camera, kinematics as K
FD = '/home/jetson/lerobot/fable_direct/fd'
cam = camera.CameraModel.load()
cfg = json.load(open('/home/jetson/lerobot/fable_direct/config.json'))
tool = K.tool_offset(*cfg['tip_offset'])
T0 = time.time()
def fd(*a):
    t0 = time.time()
    p = subprocess.run([FD, *a], capture_output=True, text=True, timeout=120)
    try:
        d = json.loads(p.stdout)
    except Exception:
        d = {'ok': False, 'message': (p.stdout + p.stderr)[:300]}
    extra = {k: d.get(k) for k in ('contact', 'empty', 'pos_raw', 'width_mm') if k in d}
    print('[%5.1fs] %-6s %4.1fs ok=%s %s %s' % (time.time() - T0, a[0], time.time() - t0, d.get('ok'), extra or '', (d.get('message') or '')[:260]), flush=True)
    return d
img = cv2.imread(sys.argv[1])
# hub centre from two image points (base of the hub on the flange top, and the hub's top face)
hb = cam.to_plane((1046.7, 1013.3), 0.0062); ht = cam.to_plane((1048.3, 977.3), 0.0186)
H = 0.5 * (hb[:2] + ht[:2])
print('hub centre: from its base (%.4f, %.4f), from its top (%.4f, %.4f) -> use (%.4f, %.4f), r=%.3f' % (hb[0], hb[1], ht[0], ht[1], H[0], H[1], np.hypot(*H)))
D = np.array([float(sys.argv[2]), float(sys.argv[3])])
PITCH = float(sys.argv[4]); OPEN = 2400; RH = 0.007; STAND = 0.008; ZG = 0.0097; ZC = 0.050; ZREL = 0.0262
rh = H / np.hypot(*H); azg = math.degrees(math.atan2(H[1], H[0]))
tg = H - (RH + STAND) * rh
rd = D / np.hypot(*D); azd = math.degrees(math.atan2(D[1], D[0]))
td = D - (RH - 0.001) * rd
for nm, p, z in (('hover', tg, ZC), ('grasp', tg, ZG), ('carry', td, ZC), ('release', td, ZREL)):
    print('  %-8s tip (%.4f, %.4f, %.4f): pitch range %s' % (nm, p[0], p[1], z, K.pitch_range([p[0], p[1], z], 0.0, tool)))
if '--go' not in sys.argv:
    sys.exit(0)
frac = '%.4f' % ((OPEN - 1998.0) / 1650.0)
A = ('--pitch', str(PITCH), '--jaw-azimuth', '%.2f' % azg)
B = ('--pitch', str(PITCH), '--jaw-azimuth', '%.2f' % azd)
fd('grip', frac)
if not fd('tip', '%.4f' % tg[0], '%.4f' % tg[1], '%.4f' % ZC, *A, '--speed', '25').get('ok'):
    fd('ready'); sys.exit(1)
if not fd('tip', '%.4f' % tg[0], '%.4f' % tg[1], '%.4f' % ZG, *A, '--straight', '--speed', '10').get('ok'):
    fd('ready'); sys.exit(1)
r = fd('grasp')
if not r.get('ok') or r.get('empty') or r.get('contact') is False:
    print('grasp failed', json.dumps(r)[:300]); fd('grip', frac); fd('tip', '%.4f' % tg[0], '%.4f' % tg[1], '%.4f' % ZC, *A, '--straight', '--speed', '12'); fd('ready'); sys.exit(2)
fd('tip', '%.4f' % tg[0], '%.4f' % tg[1], '%.4f' % ZC, *A, '--straight', '--speed', '12')
s0 = fd('snap', '--name', 'disc_lifted', '--plain')
if not fd('tip', '%.4f' % td[0], '%.4f' % td[1], '%.4f' % ZC, *B, '--speed', '25').get('ok'):
    print('carry refused: still holding the disc'); sys.exit(3)
fd('tip', '%.4f' % td[0], '%.4f' % td[1], '%.4f' % ZREL, *B, '--straight', '--speed', '10')
fd('grip', frac)
fd('tip', '%.4f' % td[0], '%.4f' % td[1], '%.4f' % ZC, *B, '--straight', '--speed', '15')
s1 = fd('snap', '--name', 'disc_released', '--plain')
fd('ready')
s2 = fd('snap', '--name', 'disc_after', '--plain')
print('[%5.1fs] photos: lifted %s | right after release %s | after ready %s' % (time.time() - T0, s0.get('raw'), s1.get('raw'), s2.get('raw')))
