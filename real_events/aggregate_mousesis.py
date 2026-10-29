"""Aggregate MouseSIS results over all 25 annotated clips (4 val + 21 train; none used in training).

Inputs: per-run mousesis_eval.json (from eval_mousesis.py) for val and train, per window/method.
Reports, per method/window: 'the mice' J&F, far left/right J&F (per-frame GT), mean, instance
selection (predicted mask covers the referred mouse more than any other), negative-control empty
rate; split into normal clips and noisy clips (density > 5 ev/px/s, i.e. noise-dominated).
Per-clip density rule (event counts only): w = 10 ms * 11.9 / density, snapped to the nearest
evaluated window in log space.

usage: aggregate_mousesis.py <config.json>   (config lists runs; see score driver)
"""
import json
import math
import sys

import numpy as np
from PIL import Image

cfg = json.load(open(sys.argv[1]))
H, W = 720, 1280


def rle(s):
    h, w = s['size']
    r = np.array(s['counts'].split(), dtype=np.int64).reshape(-1, 2)
    m = np.zeros(h * w, np.uint8)
    for st, n in r:
        m[st:st + n] = 1
    return m.reshape(w, h).T.astype(bool)


# clip metadata: density, split, instances
clips = {}
for split in cfg['splits']:
    st = json.load(open(split['stats']))
    meta = json.load(open(split['meta']))['videos']
    ann = json.load(open(split['annotations']))
    byv = {}
    for a in ann['annotations']:
        byv.setdefault(a['video_id'], []).append(a)
    for name, s in st.items():
        dens = s['n_events'] / (H * W * s['n_img'] * s['step_us'] / 1e6)
        clips[name] = dict(split=split['name'], density=dens, frames=meta[name]['frames'],
                           insts=byv[name[5:]], noisy=dens > 5.0)


def select_all(methods):
    """Instance selection for every method, decoding each frame's instance masks once.
    Returns {method: {clip: {'1': (ok, n), '2': (ok, n)}}}."""
    res = {lab: {} for lab in methods}
    for name, c in sorted(clips.items()):
        cnt = {lab: {'1': [0, 0], '2': [0, 0]} for lab in methods}
        for fr in c['frames']:
            i = int(fr) - 1
            ms = [rle(a['segmentations'][i]) for a in c['insts'] if a['segmentations'][i] is not None]
            ms = [m for m in ms if m.any()]
            if not ms:
                continue
            cx = [np.nonzero(m)[1].mean() for m in ms]
            tg = {'1': int(np.argmin(cx)), '2': int(np.argmax(cx))}
            for lab, runs in methods.items():
                rd = runs[c['split']]
                for e in ('1', '2'):
                    P = np.array(Image.open(f'{rd}/Annotations/{name}/{e}/{fr}.png').convert('L')) > 127
                    cnt[lab][e][1] += 1
                    if P.any():
                        ov = [(P & m).sum() / m.sum() for m in ms]
                        cnt[lab][e][0] += int(int(np.argmax(ov)) == tg[e] and ov[tg[e]] > 0)
        for lab in methods:
            res[lab][name] = {e: tuple(v) for e, v in cnt[lab].items()}
        print('selection done', name, flush=True)
    return res


SEL = select_all(cfg['methods'])


def per_clip_rows(lab, method_runs):
    """method_runs: {split: run_dir}. Returns {clip: metrics} incl. instance selection."""
    rows = {}
    for split, rd in method_runs.items():
        d = json.load(open(f'{rd}/mousesis_eval.json'))
        for name, r in d['per_clip'].items():
            sel = SEL[lab][name]
            rows[name] = dict(mice=r['0']['JF'], left=r['1'].get('JF_pf', r['1']['JF']),
                              right=r['2'].get('JF_pf', r['2']['JF']), neg_empty=r['3']['empty_frac'],
                              sel_l=sel['1'], sel_r=sel['2'], motion=d['motion_mask'][name]['JF'])
    return rows


def summarize(rows, subset):
    names = [n for n in rows if subset(n)]
    if not names:
        return None
    f = lambda k: 100 * float(np.mean([rows[n][k] for n in names]))
    sl = sum(rows[n]['sel_l'][0] for n in names) / max(1, sum(rows[n]['sel_l'][1] for n in names))
    sr = sum(rows[n]['sel_r'][0] for n in names) / max(1, sum(rows[n]['sel_r'][1] for n in names))
    mice, left, right = f('mice'), f('left'), f('right')
    return dict(n=len(names), mice=mice, left=left, right=right, mean=(mice + left + right) / 3,
                neg_empty=f('neg_empty'), sel_l=100 * sl, sel_r=100 * sr, motion=f('motion'))


all_rows = {lab: per_clip_rows(lab, runs) for lab, runs in cfg['methods'].items()}

# per-clip density rule over the Event-SAMA windows
wins = cfg['density_rule_windows']  # {"10": label, "75": label, ...}
rule_rows, choice = {}, {}
for name, c in clips.items():
    w = 10.0 * 11.9 / c['density']
    best = min(wins, key=lambda k: abs(math.log(float(k)) - math.log(w)))
    choice[name] = (round(w), int(best))
    rule_rows[name] = all_rows[wins[best]][name]
all_rows['Event-SAMA, per-clip density rule'] = rule_rows

out = ['| method | clips | mice | far left | far right | mean | sel. left | sel. right | "a red car" empty | motion mask |',
       '|' + '---|' * 10]
for subset_name, subset in (('all 25', lambda n: True), ('normal', lambda n: not clips[n]['noisy']),
                            ('noisy', lambda n: clips[n]['noisy'])):
    out.append(f'| **{subset_name}** | | | | | | | | | |')
    for lab, rows in all_rows.items():
        s = summarize(rows, subset)
        if s:
            out.append(f"| {lab} | {s['n']} | {s['mice']:.1f} | {s['left']:.1f} | {s['right']:.1f} | {s['mean']:.1f} | "
                       f"{s['sel_l']:.0f}% | {s['sel_r']:.0f}% | {s['neg_empty']:.0f}% | {s['motion']:.1f} |")
print('\n'.join(out))
print('\nper-clip density rule (clip: density ev/px/s -> ideal window ms -> used):')
print(', '.join(f"{n}: {clips[n]['density']:.2f}->{choice[n][0]}->{choice[n][1]}" for n in sorted(clips)))
json.dump(dict(table=out, choice=choice), open(cfg['out'], 'w'), indent=1)
