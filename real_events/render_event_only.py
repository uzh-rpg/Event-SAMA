"""Event-only rendering of the segmentation on native event clips (no RGB).
For every (clip, expression): a GIF over all frames (events as pseudo-RGB
R=OFF, G=density, B=ON, predicted mask overlaid) + a static strip PNG.

usage: render_event_only.py <run_dir> <window_ms>
"""
import json
import os
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw

CONV = '/iopsstor/scratch/cscs/sfedorchenko/real_events/converted'
H, W = 480, 640
STEP = 50_000


def load_events(path):
    with h5py.File(path, 'r') as f:
        return f['x'][:], f['y'][:], f['t'][:], f['p'][:]


def voxel(ev, centre, half):
    x, y, t, p = ev
    lo, hi = np.searchsorted(t, centre - half), np.searchsorted(t, centre + half)
    xx, yy, pp = x[lo:hi].astype(np.int64), y[lo:hi].astype(np.int64), p[lo:hi]
    flat = yy * W + xx
    on = np.bincount(flat[pp > 0], minlength=H * W).astype(np.float32).reshape(H, W)
    off = np.bincount(flat[pp <= 0], minlength=H * W).astype(np.float32).reshape(H, W)

    def nrm(a):
        m = a.max()
        return a / m * 255.0 if m > 0 else a
    return np.stack([nrm(off), nrm(on + off), nrm(on)], -1).astype(np.uint8)


def overlay(img, mask):
    out = img.astype(np.float32)
    m = mask > 127
    out[m] = 0.45 * out[m] + 0.55 * np.array([255, 40, 40], np.float32)
    edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    out[edge] = [255, 255, 0]
    return out.astype(np.uint8)


def caption(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 16], fill=(0, 0, 0))
    d.text((4, 2), text[:105], fill=(255, 255, 255))
    return im


def main():
    run_dir, window_ms = sys.argv[1], float(sys.argv[2])
    half = int(window_ms * 1000 / 2)
    out = f'{run_dir}/event_only'
    os.makedirs(out, exist_ok=True)
    meta = json.load(open(f'{CONV}/clips_meta.json'))
    exps = json.load(open(f'{CONV}/meta_expressions.json'))['videos']
    for clip, m in meta.items():
        n = m['n']
        ev = load_events(f'{CONV}/{clip}/events.h5')
        base = [voxel(ev, k * STEP, half) for k in range(1, n + 1)]
        for eid in sorted(exps[clip]['expressions'], key=int):
            text = exps[clip]['expressions'][eid]['exp']
            frames = []
            for k in range(1, n + 1):
                mp = f'{run_dir}/Annotations/{clip}/{eid}/{k:05d}.png'
                mask = np.array(Image.open(mp).convert('L')) if os.path.exists(mp) else np.zeros((H, W), np.uint8)
                if mask.shape != (H, W):
                    mask = np.array(Image.fromarray(mask).resize((W, H), Image.NEAREST))
                frames.append(caption(overlay(base[k - 1], mask),
                                      f'{clip} | "{text}" | window {window_ms:g} ms | frame {k}/{n}'))
            gif = f'{out}/{clip}_exp{eid}.gif'
            frames[0].save(gif, save_all=True, append_images=frames[1:], duration=50, loop=0, optimize=False)
            print('wrote', gif, f'{os.path.getsize(gif) / 1e6:.1f} MB', flush=True)
    print('DONE')


if __name__ == '__main__':
    main()
