"""Selected-examples figure for MouseSIS (caption must say 'selected', numbers stay in the table).
Per expression row: the 4 frames with the highest IoU vs GT (per-frame GT for far left/right),
min 1 s apart, from the clean clips; 'a red car' row: 4 frames with empty predictions.

usage: select_figure.py <run_dir> <converted_dir> <out_png> [win_ms=75]
"""
import json
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

run_dir, conv, out_png = sys.argv[1:4]
win = float(sys.argv[4]) if len(sys.argv) > 4 else 75.0
HALF = int(win * 500)
meta = json.load(open(f'{conv}/meta_expressions.json'))['videos']
CLIPS = ['mouse12', 'mouse25']
SC = 0.4


def load(p):
    return np.array(Image.open(p).convert('L')) > 127


def gt_path(vid, eid, fr):
    if eid in ('1', '2'):
        return f'{conv}/gt_perframe/{vid}/{eid}/{fr}.png'
    return f'{conv}/gt/{vid}/{eid}/{fr}.png'


ev = {}
for vid in CLIPS:
    with h5py.File(f'{conv}/{vid}/events.h5', 'r') as f:
        ev[vid] = (f['t'][:], f['x'][:], f['y'][:], f['p'][:], float(f.attrs['native_fps']))


def pseudo(vid, fr, H=720, W=1280):
    t, x, y, p, fps = ev[vid]
    c = int(int(fr) * 1e6 / fps)
    lo, hi = np.searchsorted(t, c - HALF), np.searchsorted(t, c + HALF)
    flat = y[lo:hi].astype(np.int64) * W + x[lo:hi]
    pp = p[lo:hi]
    on = np.bincount(flat[pp > 0], minlength=H * W).reshape(H, W).astype(np.float32)
    off = np.bincount(flat[pp <= 0], minlength=H * W).reshape(H, W).astype(np.float32)

    def n(a):
        m = a.max()
        return a / m * 255 if m > 0 else a
    return np.stack([n(off), n(on + off), n(on)], -1).astype(np.uint8)


def tile(vid, eid, fr, label):
    img = pseudo(vid, fr).astype(np.float32)
    pred = load(f'{run_dir}/Annotations/{vid}/{eid}/{fr}.png')
    gt = load(gt_path(vid, eid, fr))
    img[pred] = 0.45 * img[pred] + 0.55 * np.array([255, 40, 40])
    img = img.astype(np.uint8)
    img[binary_dilation(gt, iterations=3) & ~gt] = [255, 230, 0]
    im = Image.fromarray(img).resize((int(1280 * SC), int(720 * SC)), Image.BILINEAR)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, im.width, 16], fill=(0, 0, 0))
    d.text((4, 2), label, fill=(255, 255, 255))
    return im


rows = []
for eid in ('0', '1', '2', '3'):
    exp = meta[CLIPS[0]]['expressions'][eid]['exp']
    cands = []
    for vid in CLIPS:
        fps = ev[vid][4]
        for fr in meta[vid]['frames']:
            pred = load(f'{run_dir}/Annotations/{vid}/{eid}/{fr}.png')
            if eid == '3':
                score = -pred.mean()
            else:
                gt = load(gt_path(vid, eid, fr))
                u = (pred | gt).sum()
                score = (pred & gt).sum() / u if u else 0.0
            cands.append((score, vid, fr, int(fr) / fps))
    cands.sort(key=lambda c: -c[0])
    picked = []
    for c in cands:
        if all(c[1] != q[1] or abs(c[3] - q[3]) >= 1.0 for q in picked):
            picked.append(c)
        if len(picked) == 4:
            break
    picked.sort(key=lambda c: (c[1], c[3]))
    rows.append([tile(v, eid, fr, f'"{exp}"  {v}  t={tt:.1f}s' + (f'  IoU={s:.2f}' if eid != '3' else ''))
                 for s, v, fr, tt in picked])
    print(eid, exp, [(round(float(s), 3), v, fr) for s, v, fr, _ in picked], flush=True)

w, h = rows[0][0].size
fig = Image.new('RGB', (4 * w + 3 * 4, len(rows) * h + (len(rows) - 1) * 4), (255, 255, 255))
for r, row in enumerate(rows):
    for c, im in enumerate(row):
        fig.paste(im, (c * (w + 4), r * (h + 4)))
fig.save(out_png)
print('saved', out_png, fig.size)
