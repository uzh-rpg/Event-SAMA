"""Paired comparison over the 25 MouseSIS clips: Event-SAMA vs the two frozen-RGB pipelines.
Per clip: 'the mice' J&F, far left/right J&F (per-frame GT), their mean, and instance-selection
rate (left/right). Paired bootstrap over clips (10k resamples, 95% CI of the mean difference) and
a paired sign-flip permutation test (two-sided p). The clip is the unit of resampling.

usage: bootstrap_mousesis.py agg_config.json out.json
"""
import json
import math
import sys
from multiprocessing import Pool

import numpy as np
from PIL import Image

cfg = json.load(open(sys.argv[1]))
OURS = ['Event-SAMA 75 ms', 'Event-SAMA 150 ms']
BASE = ['pseudo-RGB 75 ms -> RGB SAMA', 'E2VID -> RGB SAMA']
METHODS = {k: cfg['methods'][k] for k in OURS + BASE}


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
    ann = json.load(open(sp['annotations']))
    byv = {}
    for a in ann['annotations']:
        byv.setdefault(a['video_id'], []).append(a)
    for name, v in meta.items():
        clips[name] = dict(split=sp['name'], frames=v['frames'], insts=byv[name[5:]])


def select_clip(name):
    c = clips[name]
    cnt = {lab: {'1': [0, 0], '2': [0, 0]} for lab in METHODS}
    for fr in c['frames']:
        i = int(fr) - 1
        ms = [rle(a['segmentations'][i]) for a in c['insts'] if a['segmentations'][i] is not None]
        ms = [m for m in ms if m.any()]
        if not ms:
            continue
        cx = [np.nonzero(m)[1].mean() for m in ms]
        tg = {'1': int(np.argmin(cx)), '2': int(np.argmax(cx))}
        for lab, runs in METHODS.items():
            for e in ('1', '2'):
                P = np.array(Image.open(f"{runs[c['split']]}/Annotations/{name}/{e}/{fr}.png").convert('L')) > 127
                cnt[lab][e][1] += 1
                if P.any():
                    ov = [(P & m).sum() / m.sum() for m in ms]
                    cnt[lab][e][0] += int(int(np.argmax(ov)) == tg[e] and ov[tg[e]] > 0)
    return name, {lab: {e: v[0] / max(1, v[1]) for e, v in d.items()} for lab, d in cnt.items()}


if __name__ == '__main__':
    with Pool(25) as pool:
        sel = dict(pool.map(select_clip, sorted(clips)))
    names = sorted(clips)
    per = {}
    for lab, runs in METHODS.items():
        jf = {}
        for rd in runs.values():
            d = json.load(open(f'{rd}/mousesis_eval.json'))['per_clip']
            for n, r in d.items():
                jf[n] = r
        per[lab] = {
            'mice': np.array([jf[n]['0']['JF'] for n in names]),
            'left': np.array([jf[n]['1'].get('JF_pf', jf[n]['1']['JF']) for n in names]),
            'right': np.array([jf[n]['2'].get('JF_pf', jf[n]['2']['JF']) for n in names]),
            'sel_left': np.array([sel[n][lab]['1'] for n in names]),
            'sel_right': np.array([sel[n][lab]['2'] for n in names]),
        }
        per[lab]['mean3'] = (per[lab]['mice'] + per[lab]['left'] + per[lab]['right']) / 3
    rng = np.random.default_rng(0)
    B = 10000
    idx = rng.integers(0, len(names), size=(B, len(names)))
    signs = rng.choice([-1.0, 1.0], size=(B, len(names)))
    out, lines = {}, []
    lines.append('| ours | baseline | metric | ours | baseline | diff | 95% CI (bootstrap) | p (sign-flip) |')
    lines.append('|---|---|---|---|---|---|---|---|')
    for o in OURS:
        for b in BASE:
            for k in ('mice', 'left', 'right', 'mean3', 'sel_left', 'sel_right'):
                d = per[o][k] - per[b][k]
                boot = d[idx].mean(1)
                lo, hi = np.percentile(boot, [2.5, 97.5])
                perm = np.abs((signs * d).mean(1))
                p = float((perm >= abs(d.mean())).mean())
                out[f'{o} | {b} | {k}'] = dict(ours=float(per[o][k].mean()), base=float(per[b][k].mean()),
                                               diff=float(d.mean()), ci=[float(lo), float(hi)], p=p)
                star = ' *' if (lo > 0 or hi < 0) else ''
                lines.append(f'| {o} | {b} | {k} | {100 * per[o][k].mean():.1f} | {100 * per[b][k].mean():.1f} | '
                             f'{100 * d.mean():+.1f} | [{100 * lo:+.1f}, {100 * hi:+.1f}]{star} | {p:.3f} |')
    print('\n'.join(lines))
    print('\n* = 95% CI excludes 0. Unit = clip (n=25).')
    json.dump(out, open(sys.argv[2], 'w'), indent=1)
