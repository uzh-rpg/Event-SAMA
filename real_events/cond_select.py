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
METHODS = {k: cfg['methods'][k] for k in ('Event-SAMA 75 ms', 'Event-SAMA 150 ms', 'Event-SAMA 300 ms')}


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


def select_clip_cond(name):
    c = clips[name]
    cnt = {lab: {'1': [0, 0, 0, 0.0], '2': [0, 0, 0, 0.0]} for lab in METHODS}  # ok, nonempty, frames, chance-sum
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
                cnt[lab][e][2] += 1
                if P.any():
                    cnt[lab][e][1] += 1
                    cnt[lab][e][3] += 1.0 / len(ms)
                    ov = [(P & m).sum() / m.sum() for m in ms]
                    cnt[lab][e][0] += int(int(np.argmax(ov)) == tg[e] and ov[tg[e]] > 0)
    return name, cnt


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
        sel = dict(pool.map(select_clip_cond, sorted(clips)))
    for lab in METHODS:
        for e, nm in (('1', 'far left'), ('2', 'far right')):
            ok = sum(sel[n][lab][e][0] for n in sel); ne = sum(sel[n][lab][e][1] for n in sel)
            nf = sum(sel[n][lab][e][2] for n in sel); ch = sum(sel[n][lab][e][3] for n in sel)
            print(f"{lab} | {nm}: non-empty in {100*ne/nf:.0f}% of frames; correct mouse in {100*ok/max(1,ne):.0f}% of non-empty frames "
                  f"(chance on those frames {100*ch/max(1,ne):.0f}%); correct in {100*ok/nf:.0f}% of all frames")
