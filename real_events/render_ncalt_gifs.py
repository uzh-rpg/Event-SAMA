"""GIFs of real-event animal segmentation (N-Caltech101).
Left: events + mask for the CORRECT expression ('the <class>'); right: mask for the WRONG-class
expression. Yellow = GT object contour. Events shown exactly as the model gets them to SAM2
(R=OFF, G=density, B=ON, per-window max normalisation), 10 ms windows, hot pixels removed.

usage: render_ncalt_gifs.py <run_dir> <converted_dir> <out_dir> <vid> [<vid> ...]
"""
import json
import os
import sys

import h5py
import numpy as np
from PIL import Image, ImageDraw

H, W, S = 180, 240, 3
FRAMES = [f'{k:05d}' for k in range(3, 31, 3)]
HALF = 5000


def voxel(ev, centre):
    x, y, t, p = ev
    lo, hi = np.searchsorted(t, centre - HALF), np.searchsorted(t, centre + HALF)
    flat = y[lo:hi].astype(np.int64) * W + x[lo:hi]
    pp = p[lo:hi]
    on = np.bincount(flat[pp > 0], minlength=H * W).astype(np.float32).reshape(H, W)
    off = np.bincount(flat[pp <= 0], minlength=H * W).astype(np.float32).reshape(H, W)

    def nrm(a):
        m = a.max()
        return a / m * 255.0 if m > 0 else a
    return np.stack([nrm(off), nrm(on + off), nrm(on)], -1).astype(np.uint8), int(hi - lo)


def panel(img, mask, cont, title, iou):
    out = img.astype(np.float32)
    m = mask > 127
    out[m] = 0.5 * out[m] + 0.5 * np.array([255, 40, 40], np.float32)
    im = Image.fromarray(out.astype(np.uint8)).resize((W * S, H * S), Image.NEAREST)
    d = ImageDraw.Draw(im)
    pts = [(float(a) * S, float(b) * S) for a, b in cont]
    d.line(pts + [pts[0]], fill=(255, 255, 0), width=2)
    d.rectangle([0, 0, W * S, 16], fill=(0, 0, 0))
    d.text((4, 2), f'{title}  IoU={iou:.2f}', fill=(255, 255, 255))
    return im


def main():
    run_dir, conv, out_dir = sys.argv[1:4]
    vids = sys.argv[4:]
    os.makedirs(out_dir, exist_ok=True)
    exps = json.load(open(f'{conv}/meta_expressions.json'))['videos']
    for vid in vids:
        gt = np.array(Image.open(f'{conv}/gt/{vid}.png').convert('L')) > 127
        cont = np.load(f'{conv}/gt/{vid}_contour.npy')
        with h5py.File(f'{conv}/{vid}/events.h5', 'r') as f:
            ev = (f['x'][:], f['y'][:], f['t'][:].astype(np.int64), f['p'][:])
        frames = []
        for fr in FRAMES:
            img, ne = voxel(ev, int(fr) * 10000)
            tiles = []
            for eid, tag in (('0', 'correct'), ('1', 'WRONG class')):
                mp = f'{run_dir}/Annotations/{vid}/{eid}/{fr}.png'
                mask = np.array(Image.open(mp).convert('L')) if os.path.exists(mp) else np.zeros((H, W), np.uint8)
                pm = mask > 127
                iou = (pm & gt).sum() / max(1, (pm | gt).sum())
                tiles.append(panel(img, mask, cont, f'{tag}: "{exps[vid]["expressions"][eid]["exp"]}"', iou))
            canvas = Image.new('RGB', (W * S * 2 + 4, H * S), (40, 40, 40))
            canvas.paste(tiles[0], (0, 0))
            canvas.paste(tiles[1], (W * S + 4, 0))
            d = ImageDraw.Draw(canvas)
            d.text((4, H * S - 14), f'{vid} | t={int(fr) * 10} ms | {ne} events in 10 ms window', fill=(200, 200, 200))
            frames.append(canvas)
        gif = f'{out_dir}/{vid}.gif'
        frames[0].save(gif, save_all=True, append_images=frames[1:], duration=250, loop=0)
        print('wrote', gif, f'{os.path.getsize(gif) / 1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    main()
