"""Event-only visualisations of Event-SAMA on real MouseSIS events.
Background: the events exactly as the grounding branch gets them (10 ms window, R=OFF,
G=ON+OFF, B=ON, per-window max normalisation); red fill = predicted mask; yellow = GT contour.

usage: render_mousesis.py <run_dir> <converted_dir> <out_dir> [tag]
writes <out_dir>/<vid>_e<eid>.gif for every clip/expression and a strip figure per clip.
"""
import json
import os
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import binary_dilation

run_dir, conv, out_dir = sys.argv[1:4]
tag = sys.argv[4] if len(sys.argv) > 4 else ''
os.makedirs(out_dir, exist_ok=True)
meta = json.load(open(f'{conv}/meta_expressions.json'))['videos']
SC = 0.5
HALF = 5000


def pseudo(t, x, y, p, c, H, W):
    lo, hi = np.searchsorted(t, c - HALF), np.searchsorted(t, c + HALF)
    flat = y[lo:hi].astype(np.int64) * W + x[lo:hi]
    pp = p[lo:hi]
    on = np.bincount(flat[pp > 0], minlength=H * W).reshape(H, W).astype(np.float32)
    off = np.bincount(flat[pp <= 0], minlength=H * W).reshape(H, W).astype(np.float32)

    def n(a):
        m = a.max()
        return a / m * 255 if m > 0 else a
    return np.stack([n(off), n(on + off), n(on)], -1).astype(np.uint8)


def compose(img, pred, gt):
    o = img.astype(np.float32)
    o[pred] = 0.45 * o[pred] + 0.55 * np.array([255, 40, 40])
    o = o.astype(np.uint8)
    edge = binary_dilation(gt, iterations=3) & ~gt
    o[edge] = [255, 230, 0]
    return o


for vid, v in sorted(meta.items()):
    with h5py.File(f'{conv}/{vid}/events.h5', 'r') as f:
        t, x, y, p = f['t'][:], f['x'][:], f['y'][:], f['p'][:]
        fps = float(f.attrs['native_fps'])
        H, W = int(f.attrs['native_height']), int(f.attrs['native_width'])
    frames = v['frames']
    imgs = {fr: pseudo(t, x, y, p, int(int(fr) * 1e6 / fps), H, W) for fr in frames}
    strips = []
    for eid, e in v['expressions'].items():
        out = []
        for fr in frames:
            pp = f'{run_dir}/Annotations/{vid}/{eid}/{fr}.png'
            pred = np.array(Image.open(pp).convert('L')) > 127 if os.path.exists(pp) else np.zeros((H, W), bool)
            gt = np.array(Image.open(f'{conv}/gt/{vid}/{eid}/{fr}.png')) > 0
            im = Image.fromarray(compose(imgs[fr], pred, gt)).resize((int(W * SC), int(H * SC)), Image.BILINEAR)
            d = ImageDraw.Draw(im)
            d.rectangle([0, 0, int(W * SC), 18], fill=(0, 0, 0))
            d.text((5, 3), f'"{e["exp"]}"   {vid} t={int(int(fr) * 1e3 / fps)} ms   {tag}', fill=(255, 255, 255))
            out.append(im)
        out[0].save(f'{out_dir}/{vid}_e{eid}.gif', save_all=True, append_images=out[1::2], duration=140, loop=0)
        strips.append([out[i] for i in np.linspace(0, len(out) - 1, 4).astype(int)])
    w, h = strips[0][0].size
    fig = Image.new('RGB', (4 * w + 3 * 4, len(strips) * h + (len(strips) - 1) * 4), (255, 255, 255))
    for r, row in enumerate(strips):
        for c, im in enumerate(row):
            fig.paste(im, (c * (w + 4), r * (h + 4)))
    fig.save(f'{out_dir}/{vid}_strip.png')
    print('done', vid, flush=True)
