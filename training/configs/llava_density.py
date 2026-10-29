"""Definitive LLaVA-1.5 ev/px/s median (2s cap) for paper Table 1 / Fig 6.

Samples N event clips from the gqa + ocr_vqa roots (the two sources that make
up the paper's 'LLaVA-1.5' row), computes per-clip mean per-pixel event rate
over the first min(duration, 2s), reports the median across clips.
"""
import sys, random, glob, os
import numpy as np
import h5py

ROOTS = [
    '/capstor/scratch/cscs/mzehnde/sama_converted_transfer/max_sergei/llava/gqa',
    '/capstor/scratch/cscs/mzehnde/sama_converted_transfer/max_sergei/llava/ocr_vqa',
]
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
random.seed(0)

files = []
for root in ROOTS:
    subdirs = os.listdir(root)
    picks = random.sample(subdirs, min(3 * N, len(subdirs)))
    for d in picks:
        h5 = os.path.join(root, d, 'events.h5')
        if os.path.exists(h5):
            files.append(h5)
        else:
            g = glob.glob(os.path.join(root, d, '*.h5'))
            if g:
                files.append(g[0])
random.shuffle(files)
files = files[:N]
print(f'sampled {len(files)} clips')

eps_vals = []
for i, fp in enumerate(files):
    try:
        with h5py.File(fp, 'r') as f:
            t = f['t']
            n = t.shape[0]
            if n < 2:
                continue
            t0, t1 = int(t[0]), int(t[-1])
            unit = str(f.attrs.get('t_unit', 'us'))
            div = 1e6 if unit == 'us' else 1e9
            dur = (t1 - t0) / div
            h = int(f.attrs.get('native_height', 0))
            w = int(f.attrs.get('native_width', 0))
            if h == 0 or w == 0:
                continue
            cap = min(dur, 2.0)
            if dur > 2.0:
                # count events within first 2s (timestamps sorted)
                lim = t0 + int(2.0 * div)
                tt = f['t'][:]
                n_cap = int(np.searchsorted(tt, lim))
            else:
                n_cap = n
            if cap <= 0:
                continue
            eps_vals.append(n_cap / (h * w) / cap)
    except Exception as e:
        print('skip', fp, e)
    if (i + 1) % 200 == 0:
        print(f'{i+1}/{len(files)}  running median={np.median(eps_vals):.2f}', flush=True)

a = np.array(eps_vals)
print(f'\nn={len(a)}')
print(f'median ev/px/s : {np.median(a):.2f}')
print(f'mean   ev/px/s : {a.mean():.2f} +- {a.std():.2f}')
for p in (5, 25, 75, 95):
    print(f'p{p}: {np.percentile(a, p):.2f}')

# subsampling noise: how much do 200-item medians wobble?
meds = [np.median(np.random.choice(a, 200, replace=False)) for _ in range(2000)]
print(f'\n200-sample median: mean={np.mean(meds):.2f}, std={np.std(meds):.2f}, '
      f'p5={np.percentile(meds,5):.2f}, p95={np.percentile(meds,95):.2f}')
