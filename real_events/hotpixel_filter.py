"""Create a hot-pixel-filtered copy of the converted N-Caltech101 set.
Rule (fixed before any model run, from event statistics only): drop all events of pixels
that fire more than HOT=100 times within the 300 ms recording (~5x the p99 of active pixels).
"""
import json
import os
import shutil

import h5py
import numpy as np

SRC = '/iopsstor/scratch/cscs/sfedorchenko/real_events/ncaltech_converted'
DST = '/iopsstor/scratch/cscs/sfedorchenko/real_events/ncaltech_converted_hf'
HOT = 100
H, W = 180, 240

os.makedirs(DST, exist_ok=True)
vids = sorted(json.load(open(f'{SRC}/meta_expressions.json'))['videos'])
removed_px, removed_ev, total_ev = [], 0, 0
for v in vids:
    with h5py.File(f'{SRC}/{v}/events.h5', 'r') as f:
        x, y, t, p = f['x'][:], f['y'][:], f['t'][:], f['p'][:]
        attrs = dict(f.attrs)
    flat = y.astype(np.int64) * W + x
    cnt = np.bincount(flat, minlength=H * W)
    hot = cnt > HOT
    keep = ~hot[flat]
    removed_px.append(int(hot.sum()))
    removed_ev += int((~keep).sum())
    total_ev += len(x)
    os.makedirs(f'{DST}/{v}', exist_ok=True)
    with h5py.File(f'{DST}/{v}/events.h5', 'w') as g:
        kw = dict(compression='gzip', compression_opts=4)
        g.create_dataset('x', data=x[keep], **kw)
        g.create_dataset('y', data=y[keep], **kw)
        g.create_dataset('t', data=t[keep], **kw)
        g.create_dataset('p', data=p[keep], **kw)
        for k, val in attrs.items():
            g.attrs[k] = val
for fn in ('meta_expressions.json', 'samples_meta.json'):
    shutil.copy(f'{SRC}/{fn}', f'{DST}/{fn}')
if not os.path.exists(f'{DST}/gt'):
    os.symlink(f'{SRC}/gt', f'{DST}/gt')
print(f'{len(vids)} samples | hot pixels removed per sample: mean {np.mean(removed_px):.1f}, '
      f'max {max(removed_px)} | events removed: {removed_ev}/{total_ev} = {100 * removed_ev / total_ev:.2f}%')
