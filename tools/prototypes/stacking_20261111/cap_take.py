"""Stage B: lower, pinch the cap's wall, lift, check with the camera that the cap left its place, carry, put down, release."""
import json, subprocess, sys, time
sys.path.insert(0, '/home/jetson/lerobot/fable_direct')
import numpy as np, cv2, camera
FD = '/home/jetson/lerobot/fable_direct/fd'
cam = camera.CameraModel.load()
T0 = time.time()
def fd(*a, quiet=False):
    t0 = time.time()
    p = subprocess.run([FD, *a], capture_output=True, text=True, timeout=120)
    try:
        d = json.loads(p.stdout)
    except Exception:
        d = {'ok': False, 'message': (p.stdout + p.stderr)[:300]}
    if not quiet:
        extra = {k: d.get(k) for k in ('pos_raw', 'load', 'hold_goal_raw', 'above_touch_ticks') if k in d}
        print('[%5.1fs] %-6s %4.1fs ok=%s %s %s' % (time.time() - T0, a[0], time.time() - t0, d.get('ok'), extra or '', (d.get('message') or '')[:200]), flush=True)
    return d
cx, cy, x, y, psi, zg, px_, py_ = [float(v) for v in sys.argv[1:9]]
REF = sys.argv[9]                       # photo of the scene without the arm
def cap_brightness(path):
    """mean brightness of the cap's inner bottom where it stood (pixel of its centre at the inner-bottom height)"""
    img = cv2.imread(path).min(axis=2).astype(np.float32)
    u, v = cam.project([[cx, cy, 0.006]])[0]
    return float(img[int(v) - 8:int(v) + 9, int(u) - 12:int(u) + 13].mean())
b0 = cap_brightness(REF)
A = ('--pitch', '90', '--jaw-azimuth', str(psi))
def stop(msg):
    print('ABORT:', msg, flush=True)
    fd('grip', '0.1012'); fd('tip', str(x), str(y), '0.045', *A, '--straight', '--speed', '10'); fd('ready')
    sys.exit(2)
r = fd('tip', str(x), str(y), '%.4f' % zg, *A, '--straight', '--speed', '8')
if not r.get('ok'):
    stop('descent refused')
g = fd('grasp', '--thin')
if not g.get('ok'):
    stop('pinch failed')
r = fd('tip', str(x), str(y), '0.045', *A, '--straight', '--speed', '8')
if not r.get('ok'):
    stop('lift refused')
s = fd('snap', '--name', 'caplift', '--plain')
b1 = cap_brightness(s['raw'])
st = fd('state', quiet=True)
print('[%5.1fs] cap place brightness before %.0f -> after lift %.0f ; gripper %s' % (time.time() - T0, b0, b1, {k: st.get(k) for k in ('gripper_raw', 'gripper_open_frac', 'load') if k in st}), flush=True)
if b1 > 0.6 * b0:
    stop('the cap is still where it was: it was not lifted (photo %s)' % s['raw'])
r = fd('tip', str(px_), str(py_), '0.045', *A, '--straight', '--speed', '15')
if not r.get('ok'):
    print('carry refused, trying a joint move:', (r.get('message') or '')[:200]); r = fd('tip', str(px_), str(py_), '0.045', *A, '--speed', '20')
    if not r.get('ok'):
        print('ABORT with the cap held above the box; nothing released'); sys.exit(3)
r = fd('tip', str(px_), str(py_), '0.0040', *A, '--straight', '--speed', '8')
fd('grip', '0.1012')
fd('tip', str(px_), str(py_), '0.050', *A, '--straight', '--speed', '12')
fd('ready')
s = fd('snap', '--name', 'capout', '--plain')
print('[%5.1fs] final photo %s ; cap place brightness now %.0f' % (time.time() - T0, s.get('raw'), cap_brightness(s['raw'])), flush=True)
fd('park')
print('[%5.1fs] done' % (time.time() - T0))
