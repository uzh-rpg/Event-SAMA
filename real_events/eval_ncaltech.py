"""Evaluate predicted masks on N-Caltech101 samples against the GT object contour.
Per (sample, expression): mean IoU over the 10 frames vs the (static) GT mask, predicted area,
empty rate. Reports positive ('the <class>') vs wrong-class ('the <other class>') expressions.

usage: eval_ncaltech.py <run_dir> <converted_dir> [out_json]
"""
import json
import os
import sys

import numpy as np
from PIL import Image

run_dir, conv = sys.argv[1], sys.argv[2]
out_json = sys.argv[3] if len(sys.argv) > 3 else f'{run_dir}/ncalt_eval.json'
meta = json.load(open(f'{conv}/samples_meta.json'))
rows = []
for vid, m in sorted(meta.items()):
    gt = np.array(Image.open(f'{conv}/gt/{vid}.png').convert('L')) > 127
    r = dict(vid=vid, cls=m['cls'])
    for eid, tag in (('0', 'pos'), ('1', 'neg')):
        d = f'{run_dir}/Annotations/{vid}/{eid}'
        if not os.path.isdir(d):
            r[tag] = None
            continue
        ious, areas, empties = [], [], 0
        fr = sorted(os.listdir(d))
        for f in fr:
            pm = np.array(Image.open(f'{d}/{f}').convert('L')) > 127
            if pm.shape != gt.shape:
                pm = np.array(Image.fromarray(pm.astype(np.uint8) * 255).resize(gt.shape[::-1], Image.NEAREST)) > 127
            inter, union = (pm & gt).sum(), (pm | gt).sum()
            ious.append(inter / union if union else 0.0)
            areas.append(pm.mean())
            empties += (not pm.any())
        r[tag] = dict(iou=float(np.mean(ious)), area=float(np.mean(areas)), empty=empties / len(fr))
    r['gt_area'] = float(gt.mean())
    rows.append(r)

ok = [r for r in rows if r.get('pos') and r.get('neg')]
P = np.array([r['pos']['iou'] for r in ok]); N = np.array([r['neg']['iou'] for r in ok])
summ = dict(
    n_samples=len(ok),
    mean_iou_pos=float(P.mean()), mean_iou_neg=float(N.mean()),
    median_iou_pos=float(np.median(P)), median_iou_neg=float(np.median(N)),
    frac_pos_beats_neg=float((P > N).mean()),
    frac_pos_iou_gt_0p3=float((P > 0.3).mean()), frac_pos_iou_gt_0p5=float((P > 0.5).mean()),
    mean_pred_area_pos=float(np.mean([r['pos']['area'] for r in ok])),
    mean_pred_area_neg=float(np.mean([r['neg']['area'] for r in ok])),
    mean_gt_area=float(np.mean([r['gt_area'] for r in ok])),
    mean_empty_rate_pos=float(np.mean([r['pos']['empty'] for r in ok])),
    mean_empty_rate_neg=float(np.mean([r['neg']['empty'] for r in ok])),
)
per_cls = {}
for r in ok:
    per_cls.setdefault(r['cls'], []).append((r['pos']['iou'], r['neg']['iou']))
summ['per_class'] = {c: dict(pos=float(np.mean([a for a, _ in v])), neg=float(np.mean([b for _, b in v])), n=len(v))
                     for c, v in sorted(per_cls.items())}
json.dump(dict(summary=summ, rows=rows), open(out_json, 'w'), indent=1)
s = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in summ.items() if k != 'per_class'}
print(json.dumps(s, indent=1))
print('per class (mean IoU pos / wrong-class):')
for c, v in summ['per_class'].items():
    print(f'  {c:12s} pos {v["pos"]:.3f}  neg {v["neg"]:.3f}  n={v["n"]}')
