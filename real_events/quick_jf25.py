"""Quick J&F table over all 25 MouseSIS clips from the per-run mousesis_eval.json files (no
instance selection). usage: quick_jf25.py agg_config.json"""
import json
import math
import sys

import numpy as np

cfg = json.load(open(sys.argv[1]))
dens = {}
for sp in cfg['splits']:
    for name, s in json.load(open(sp['stats'])).items():
        dens[name] = s['n_events'] / (921600 * s['n_img'] * s['step_us'] / 1e6)
rows = {}
for lab, runs in cfg['methods'].items():
    rows[lab] = {}
    for rd in runs.values():
        d = json.load(open(f'{rd}/mousesis_eval.json'))
        for name, r in d['per_clip'].items():
            rows[lab][name] = (r['0']['JF'], r['1'].get('JF_pf', r['1']['JF']), r['2'].get('JF_pf', r['2']['JF']),
                               r['3']['empty_frac'], d['motion_mask'][name]['JF'])
wins = cfg['density_rule_windows']
rule = {}
for name, dd in dens.items():
    w = 10 * 11.9 / dd
    best = min(wins, key=lambda k: abs(math.log(float(k)) - math.log(w)))
    rule[name] = rows[wins[best]][name]
rows['Event-SAMA, per-clip density rule'] = rule
print('| method | clips | mice | far left | far right | mean | "a red car" empty | motion mask |')
print('|---|---|---|---|---|---|---|---|')
for sub, f in (('all 25', lambda n: True), ('normal (density < 5)', lambda n: dens[n] < 5), ('noisy (density > 5)', lambda n: dens[n] >= 5)):
    print(f'| **{sub}** | | | | | | | |')
    for lab, rr in rows.items():
        ns = [n for n in rr if f(n)]
        a = 100 * np.array([rr[n] for n in ns]).mean(0)
        print(f'| {lab} | {len(ns)} | {a[0]:.1f} | {a[1]:.1f} | {a[2]:.1f} | {(a[0] + a[1] + a[2]) / 3:.1f} | {a[3]:.0f}% | {a[4]:.1f} |')
