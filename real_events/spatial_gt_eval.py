"""Re-score 'the mouse on the far left/right' on MouseSIS with a ground truth that only exists when a
mouse is actually there in the event stream:
  far right = mice whose centroid lies in the right quarter of the frame (x > 0.75 W) AND that are
              visible in events (>= 50 events inside the mask within +-16.7 ms, one frame interval);
              the right-most such mouse; if none, the GT is empty (correct answer: empty mask).
  far left  = same for the left quarter (x < 0.25 W), left-most.
Reports (i) standard J&F over all frames (empty GT + empty prediction = 1, as in DAVIS/MeViS),
(ii) J&F on frames where the GT mouse exists, (iii) fraction of GT-empty frames answered with an
empty mask, (iv) fraction of frames with a GT mouse.

usage: spatial_gt_eval.py agg_config.json out.md
"""
import json
import math
import sys
from multiprocessing import Pool

import h5py
import numpy as np
from PIL import Image

from metrics import db_eval_boundary, db_eval_iou

cfg = json.load(open(sys.argv[1]))
W, H = 1280, 720
VIS_HALF, VIS_MIN = 16700, 50
METHODS = {k: cfg['methods'][k] for k in ('Event-SAMA 10 ms', 'Event-SAMA 75 ms', 'Event-SAMA 150 ms', 'Event-SAMA 300 ms')}
WINS = {'10': 'Event-SAMA 10 ms', '75': 'Event-SAMA 75 ms', '150': 'Event-SAMA 150 ms', '300': 'Event-SAMA 300 ms'}


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
    conv = sp['meta'].rsplit('/', 1)[0]
    ann = json.load(open(sp['annotations']))
    byv = {}
    for a in ann['annotations']:
        byv.setdefault(a['video_id'], []).append(a)
    for name, v in meta.items():
        s = st[name]
        clips[name] = dict(split=sp['name'], conv=conv, frames=v['frames'], insts=byv[name[5:]],
                           density=s['n_events'] / (W * H * s['n_img'] * s['step_us'] / 1e6))


def clip_eval(name):
    c = clips[name]
    with h5py.File(f"{c['conv']}/{name}/events.h5", 'r') as f:
        t, x, y = f['t'][:], f['x'][:].astype(np.int64), f['y'][:].astype(np.int64)
        fps = float(f.attrs['native_fps'])
    gts = {'1': [], '2': []}
    for fr in c['frames']:
        i = int(fr) - 1
        ms = [rle(a['segmentations'][i]) for a in c['insts'] if a['segmentations'][i] is not None]
        ms = [m for m in ms if m.any()]
        cen = int(int(fr) * 1e6 / fps)
        lo, hi = np.searchsorted(t, cen - VIS_HALF), np.searchsorted(t, cen + VIS_HALF)
        ev = np.zeros((H, W), np.int32)
        np.add.at(ev, (y[lo:hi], x[lo:hi]), 1)
        vis = [(np.nonzero(m)[1].mean(), m) for m in ms if ev[m].sum() >= VIS_MIN]
        left = [v for v in vis if v[0] < 0.25 * W]
        right = [v for v in vis if v[0] > 0.75 * W]
        gts['1'].append(min(left, key=lambda v: v[0])[1] if left else np.zeros((H, W), bool))
        gts['2'].append(max(right, key=lambda v: v[0])[1] if right else np.zeros((H, W), bool))
    res = {}
    for lab, runs in METHODS.items():
        res[lab] = {}
        for e in ('1', '2'):
            G = np.stack(gts[e])
            P = np.stack([np.array(Image.open(f"{runs[c['split']]}/Annotations/{name}/{e}/{fr}.png").convert('L')) > 127
                          for fr in c['frames']])
            j = db_eval_iou(G, P)
            f = db_eval_boundary(G, P)
            has = G.reshape(len(G), -1).any(1)
            pe = ~P.reshape(len(P), -1).any(1)
            res[lab][e] = dict(jf_all=float((j.mean() + f.mean()) / 2),
                               jf_present=float((j[has].mean() + f[has].mean()) / 2) if has.any() else None,
                               empty_ok=float(pe[~has].mean()) if (~has).any() else None,
                               n=len(G), n_present=int(has.sum()))
    return name, res


if __name__ == '__main__':
    with Pool(25) as pool:
        R = dict(pool.map(clip_eval, sorted(clips)))
    rule = {}
    for name, c in clips.items():
        w = 10 * 11.9 / c['density']
        rule[name] = WINS[min(WINS, key=lambda k: abs(math.log(float(k)) - math.log(w)))]
    lines = ['| window | expr. | J&F all frames | J&F where the mouse is there | empty answer where no mouse | frames with a GT mouse |',
             '|---|---|---|---|---|---|']
    labs = list(METHODS) + ['per-clip density rule']
    for lab in labs:
        for e, nm in (('1', 'far left'), ('2', 'far right')):
            rows = [R[n][rule[n] if lab == 'per-clip density rule' else lab][e] for n in sorted(clips)]
            # frame-weighted means
            nall = sum(r['n'] for r in rows)
            jfa = sum(r['jf_all'] * r['n'] for r in rows) / nall
            pres = [r for r in rows if r['jf_present'] is not None]
            npres = sum(r['n_present'] for r in pres)
            jfp = sum(r['jf_present'] * r['n_present'] for r in pres) / max(1, npres)
            emp = [r for r in rows if r['empty_ok'] is not None]
            nemp = sum(r['n'] - r['n_present'] for r in emp)
            eok = sum(r['empty_ok'] * (r['n'] - r['n_present']) for r in emp) / max(1, nemp)
            lines.append(f'| {lab} | {nm} | {100 * jfa:.1f} | {100 * jfp:.1f} | {100 * eok:.0f}% | {100 * npres / nall:.0f}% |')
    print('\n'.join(lines))
    print('\n(frame-weighted over all 25 clips; GT = mouse in the outer quarter AND >= 50 events in its mask within +-16.7 ms)')
    open(sys.argv[2], 'w').write('\n'.join(lines) + '\n')
