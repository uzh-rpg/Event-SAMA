"""Find the MouseSIS clips where 'the mouse on the far right' fails most and render event-only GIFs
for them (per-frame GT: right-most visible mouse; red = prediction, yellow = GT contour).

usage: worst_far_right.py agg_config.json <method label> <window_ms> <out_dir> [n_normal=3] [n_noisy=1]
"""
import json
import os
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

cfg = json.load(open(sys.argv[1]))
lab, win, out = sys.argv[2], float(sys.argv[3]), sys.argv[4]
n_norm = int(sys.argv[5]) if len(sys.argv) > 5 else 3
n_noisy = int(sys.argv[6]) if len(sys.argv) > 6 else 1
runs = cfg['methods'][lab]
HALF = int(win * 500)
os.makedirs(out, exist_ok=True)


def rle(s):
    h, w = s['size']
    r = np.array(s['counts'].split(), dtype=np.int64).reshape(-1, 2)
    m = np.zeros(h * w, np.uint8)
    for st, n in r:
        m[st:st + n] = 1
    return m.reshape(w, h).T.astype(bool)


clips = {}
for sp in cfg['splits']:
    meta = json.load(open(sp['meta']))['videos']
    st = json.load(open(sp['stats']))
    conv = os.path.dirname(sp['meta'])
    ann = json.load(open(sp['annotations']))
    byv = {}
    for a in ann['annotations']:
        byv.setdefault(a['video_id'], []).append(a)
    for name, v in meta.items():
        s = st[name]
        clips[name] = dict(split=sp['name'], conv=conv, frames=v['frames'], insts=byv[name[5:]],
                           density=s['n_events'] / (921600 * s['n_img'] * s['step_us'] / 1e6))

rows = []
for name, c in sorted(clips.items()):
    pd = f"{runs[c['split']]}/Annotations/{name}/2"
    ok = n = 0
    for fr in c['frames']:
        i = int(fr) - 1
        ms = [rle(a['segmentations'][i]) for a in c['insts'] if a['segmentations'][i] is not None]
        ms = [m for m in ms if m.any()]
        if not ms:
            continue
        tgt = int(np.argmax([np.nonzero(m)[1].mean() for m in ms]))
        P = np.array(Image.open(f'{pd}/{fr}.png').convert('L')) > 127
        n += 1
        if P.any():
            ov = [(P & m).sum() / m.sum() for m in ms]
            ok += int(int(np.argmax(ov)) == tgt and ov[tgt] > 0)
    rows.append((ok / max(1, n), name))
    print(f"{name}: far-right correct {100 * ok / max(1, n):.0f}% ({c['density']:.2f} ev/px/s)", flush=True)

normal = sorted([r for r in rows if clips[r[1]]['density'] < 5])[:n_norm]
noisy = sorted([r for r in rows if clips[r[1]]['density'] >= 5])[:n_noisy]
pick = [r[1] for r in normal + noisy]
sel_by = {r[1]: r[0] for r in rows}
print('rendering', pick, flush=True)

for name in pick:
    c = clips[name]
    with h5py.File(f"{c['conv']}/{name}/events.h5", 'r') as f:
        t, x, y, p = f['t'][:], f['x'][:], f['y'][:], f['p'][:]
        fps = float(f.attrs['native_fps'])
    sel = sel_by[name]
    frames = []
    for fr in c['frames'][::2]:
        cen = int(int(fr) * 1e6 / fps)
        lo, hi = np.searchsorted(t, cen - HALF), np.searchsorted(t, cen + HALF)
        flat = y[lo:hi].astype(np.int64) * 1280 + x[lo:hi]
        pp = p[lo:hi]
        on = np.bincount(flat[pp > 0], minlength=921600).reshape(720, 1280).astype(np.float32)
        off = np.bincount(flat[pp <= 0], minlength=921600).reshape(720, 1280).astype(np.float32)
        nrm = lambda a: a / a.max() * 255 if a.max() > 0 else a
        img = np.stack([nrm(off), nrm(on + off), nrm(on)], -1)
        P = np.array(Image.open(f"{runs[c['split']]}/Annotations/{name}/2/{fr}.png").convert('L')) > 127
        G = np.array(Image.open(f"{c['conv']}/gt_perframe/{name}/2/{fr}.png")) > 0
        img[P] = 0.45 * img[P] + 0.55 * np.array([255, 40, 40])
        img = img.astype(np.uint8)
        img[binary_dilation(G, iterations=3) & ~G] = [255, 230, 0]
        im = Image.fromarray(img).resize((640, 360), Image.BILINEAR)
        d = ImageDraw.Draw(im)
        d.rectangle([0, 0, 640, 18], fill=(0, 0, 0))
        d.text((5, 3), f'"the mouse on the far right"  {name}  t={int(fr) / fps:.1f}s  {int(win)} ms  '
                       f'correct mouse in {100 * sel:.0f}% of frames', fill=(255, 255, 255))
        frames.append(im)
    frames[0].save(f'{out}/{name}_far_right.gif', save_all=True, append_images=frames[1:], duration=140, loop=0)
    print('wrote', name, flush=True)
