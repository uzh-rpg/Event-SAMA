"""Render qualitative panels for real-event inference.

For each clip: a grid with one column per sampled time step and rows
  [RGB reference (not seen by model)] + one row per expression
  (real events as pseudo-RGB with the predicted mask overlaid).
Also writes mask statistics (mean area fraction, fraction of non-empty frames)
per (clip, expression), so negative-control expressions can be compared with
the real ones.

usage: render_overlays.py <run_dir> <tag> [window_ms]
"""
import io
import json
import sys
import zipfile

import h5py
import numpy as np
from PIL import Image, ImageDraw

CONV = '/iopsstor/scratch/cscs/sfedorchenko/real_events/converted'
DSEC = '/iopsstor/scratch/cscs/sfedorchenko/real_events/dsec'
H, W = 480, 640
N_COLS = 4


def voxel_img(h5path, centre_us, half_us):
    with h5py.File(h5path, 'r') as f:
        t = f['t'][:]
        lo = np.searchsorted(t, centre_us - half_us, 'left')
        hi = np.searchsorted(t, centre_us + half_us, 'left')
        x, y, p = f['x'][lo:hi].astype(np.int64), f['y'][lo:hi].astype(np.int64), f['p'][lo:hi]
    on = np.zeros(H * W, np.float32)
    off = np.zeros(H * W, np.float32)
    flat = y * W + x
    np.add.at(on, flat[p > 0], 1.0)
    np.add.at(off, flat[p <= 0], 1.0)
    on, off = on.reshape(H, W), off.reshape(H, W)

    def nrm(a):
        m = a.max()
        return a / m * 255.0 if m > 0 else a
    # same channel assignment as the model's SAM2 input: R=OFF, G=density, B=ON
    img = np.stack([nrm(off), nrm(on + off), nrm(on)], -1).astype(np.uint8)
    return img, int(hi - lo)


def overlay(img, mask):
    out = img.astype(np.float32)
    m = mask > 127
    tint = np.array([255, 40, 40], np.float32)
    out[m] = 0.45 * out[m] + 0.55 * tint
    # 1px contour
    edge = m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
    out[edge] = [255, 255, 0]
    return out.astype(np.uint8)


def label(img, text):
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 16], fill=(0, 0, 0))
    d.text((3, 2), text[:100], fill=(255, 255, 255))
    return np.array(im)


def main():
    run_dir, tag = sys.argv[1], sys.argv[2]
    window_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
    half_us = int(window_ms * 1000 / 2)
    out_dir = f'{run_dir}/panels'
    import os
    os.makedirs(out_dir, exist_ok=True)
    meta = json.load(open(f'{CONV}/clips_meta.json'))
    exps = json.load(open(f'{CONV}/meta_expressions.json'))['videos']
    stats = {}
    zips = {}
    for clip, m in meta.items():
        seq, fs, n = m['seq'], m['frame_start'], m['n']
        if seq not in zips:
            zips[seq] = zipfile.ZipFile(f'{DSEC}/{seq}_images_rectified_left.zip')
        ks = [int(round(a)) for a in np.linspace(3, n - 3, N_COLS)]  # 1-based frame numbers
        rgb_row, ev_imgs = [], []
        for k in ks:
            raw = zips[seq].read(f'{fs + k - 1:06d}.png')
            rgb = np.array(Image.open(io.BytesIO(raw)).convert('RGB').resize((W, H)))
            rgb_row.append(label(rgb, f'{clip} RGB ref (not seen by model) frame {k}'))
            ev, ne = voxel_img(f'{CONV}/{clip}/events.h5', k * 50_000, half_us)
            ev_imgs.append((ev, ne))
        rows = [np.concatenate(rgb_row, 1)]
        for eid in sorted(exps[clip]['expressions'], key=int):
            text = exps[clip]['expressions'][eid]['exp']
            row, areas, nonempty = [], [], 0
            for (ev, ne), k in zip(ev_imgs, ks):
                mp = f'{run_dir}/Annotations/{clip}/{eid}/{k:05d}.png'
                if os.path.exists(mp):
                    mask = np.array(Image.open(mp).convert('L'))
                    if mask.shape != (H, W):
                        mask = np.array(Image.fromarray(mask).resize((W, H), Image.NEAREST))
                else:
                    mask = np.zeros((H, W), np.uint8)
                row.append(label(overlay(ev, mask), f'"{text}" | ev={ne}'))
            # statistics over ALL frames of the clip
            fr = []
            for k in range(1, n + 1):
                mp = f'{run_dir}/Annotations/{clip}/{eid}/{k:05d}.png'
                if os.path.exists(mp):
                    mm = np.array(Image.open(mp).convert('L')) > 127
                    fr.append(float(mm.mean()))
            stats[f'{clip}/{eid}'] = dict(
                expression=text, n_frames_with_mask_file=len(fr),
                mean_area_frac=float(np.mean(fr)) if fr else None,
                frac_nonempty=float(np.mean([a > 0.0005 for a in fr])) if fr else None)
            rows.append(np.concatenate(row, 1))
        Image.fromarray(np.concatenate(rows, 0)).save(f'{out_dir}/{tag}_{clip}.png')
        print('wrote', f'{out_dir}/{tag}_{clip}.png', flush=True)
    json.dump(stats, open(f'{run_dir}/mask_stats_{tag}.json', 'w'), indent=1)
    for k, v in stats.items():
        print(k, v, flush=True)


if __name__ == '__main__':
    main()
