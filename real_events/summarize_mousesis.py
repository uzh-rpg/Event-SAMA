"""Markdown table over MouseSIS runs. usage: summarize_mousesis.py label=run_dir [label=run_dir ...]"""
import json
import sys

print('| run | mice J&F (all 4) | clean 12/25 | noisy 03/04 | far-left J&F first/per-frame | far-right J&F first/per-frame '
      '| far-left point (pf) | far-right point (pf) | "a red car" empty | motion mask all/clean |')
print('|' + '---|' * 10)
for arg in sys.argv[1:]:
    lab, r = arg.split('=', 1)
    d = json.load(open(f'{r}/mousesis_eval.json'))
    s, pc, mm = d['summary'], d['per_clip'], d['motion_mask']
    cl = sum(pc[v]['0']['JF'] for v in ('mouse12', 'mouse25')) / 2
    no = sum(pc[v]['0']['JF'] for v in ('mouse03', 'mouse04')) / 2
    mcl = sum(mm[v]['JF'] for v in ('mouse12', 'mouse25')) / 2
    print(f"| {lab} | {100 * s['0']['JF']:.1f} | {100 * cl:.1f} | {100 * no:.1f} "
          f"| {100 * s['1']['JF']:.1f} / {100 * s['1'].get('JF_pf', float('nan')):.1f} "
          f"| {100 * s['2']['JF']:.1f} / {100 * s['2'].get('JF_pf', float('nan')):.1f} "
          f"| {100 * s['1'].get('point_acc_pf', float('nan')):.0f}% | {100 * s['2'].get('point_acc_pf', float('nan')):.0f}% "
          f"| {100 * s['neg_empty_frac']:.0f}% | {100 * s['motion_mask_mice_JF']:.1f} / {100 * mcl:.1f} |")
print(f"\npointing chance (random point inside dilated GT): left {100 * s['1']['point_chance']:.1f}%, "
      f"right {100 * s['2']['point_chance']:.1f}%")
