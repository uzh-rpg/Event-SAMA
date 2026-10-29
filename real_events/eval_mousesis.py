"""J&F of Event-SAMA predictions on MouseSIS (real events) vs the MouseSIS GT masks.

Expressions 0-2 are scored with the DAVIS/MeViS metrics (db_eval_iou, db_eval_boundary from
tools/eval_video_seg/metrics.py) over the evaluated frames. Expression 3 ('a red car', object
absent) is scored by predicted area and the fraction of frames with an empty mask.
Also reports a language-free reference: a motion mask from event density alone (fixed rule:
events within the same window as the model (default 10 ms), 15x15 box mean >= 8/225 per
10 ms of window), scored against 'the mice' only.

Diagnostic (added after inspecting the qualitative outputs, which show correct localisation
with oversized masks): pointing accuracy for the single-mouse expressions = fraction of frames
whose predicted-mask centroid lies inside the GT mask dilated by 20 px; chance = mean area
fraction of that dilated GT mask (a uniformly random point).

usage: eval_mousesis.py <run_dir> <converted_dir> [window_ms=10] [out_json]
"""
import json
import os
import sys

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, uniform_filter

from metrics import db_eval_boundary, db_eval_iou

run_dir, conv = sys.argv[1], sys.argv[2]
win_ms = float(sys.argv[3]) if len(sys.argv) > 3 else 10.0
out_json = sys.argv[4] if len(sys.argv) > 4 else f'{run_dir}/mousesis_eval.json'
HW = int(win_ms * 500)
THR = 8 * win_ms / 30.0
meta = json.load(open(f'{conv}/meta_expressions.json'))['videos']
res, motion = {}, {}
for vid, v in sorted(meta.items()):
    res[vid] = {}
    with h5py.File(f'{conv}/{vid}/events.h5', 'r') as f:
        x, y, t = f['x'][:], f['y'][:], f['t'][:]
        fps = float(f.attrs['native_fps'])
        H, W = int(f.attrs['native_height']), int(f.attrs['native_width'])
    for eid, e in v['expressions'].items():
        gts, prs = [], []
        for fr in v['frames']:
            gts.append(np.array(Image.open(f'{conv}/gt/{vid}/{eid}/{fr}.png')) > 0)
            pp = f'{run_dir}/Annotations/{vid}/{eid}/{fr}.png'
            prs.append(np.array(Image.open(pp).convert('L')) > 127 if os.path.exists(pp)
                       else np.zeros((H, W), bool))
        G, P = np.stack(gts), np.stack(prs)
        r = dict(exp=e['exp'], pred_area=float(P.mean()), gt_area=float(G.mean()),
                 empty_frac=float((~P.reshape(len(P), -1).any(1)).mean()))
        if G.any():
            J = float(np.mean(db_eval_iou(G, P)))
            F = float(np.mean(db_eval_boundary(G, P)))
            r.update(J=J, F=F, JF=(J + F) / 2)
            if eid in ('1', '2'):
                hit, ch = [], []
                for g, pm in zip(G, P):
                    gd = binary_dilation(g, iterations=20)
                    ch.append(gd.mean())
                    if pm.any():
                        yy, xx = np.nonzero(pm)
                        hit.append(bool(gd[int(round(yy.mean())), int(round(xx.mean()))]))
                    else:
                        hit.append(False)
                r.update(point_acc=float(np.mean(hit)), point_chance=float(np.mean(ch)))
                gpf = f'{conv}/gt_perframe/{vid}/{eid}'
                if os.path.isdir(gpf):  # post-hoc: extreme mouse re-selected in every frame
                    G2 = np.stack([np.array(Image.open(f'{gpf}/{fr}.png')) > 0 for fr in v['frames']])
                    J2, F2 = float(np.mean(db_eval_iou(G2, P))), float(np.mean(db_eval_boundary(G2, P)))
                    hit2 = []
                    for g, pm in zip(G2, P):
                        gd = binary_dilation(g, iterations=20)
                        if pm.any():
                            yy, xx = np.nonzero(pm)
                            hit2.append(bool(gd[int(round(yy.mean())), int(round(xx.mean()))]))
                        else:
                            hit2.append(False)
                    r.update(J_pf=J2, F_pf=F2, JF_pf=(J2 + F2) / 2, point_acc_pf=float(np.mean(hit2)))
        res[vid][eid] = r
        if eid == '0':
            mm = []
            for fr in v['frames']:
                c = int(int(fr) * 1e6 / fps)
                lo, hi = np.searchsorted(t, c - HW), np.searchsorted(t, c + HW)
                img = np.zeros((H, W), np.float32)
                np.add.at(img, (y[lo:hi], x[lo:hi]), 1)
                mm.append(uniform_filter(img, 15) * 225 >= THR)
            M = np.stack(mm)
            Jm, Fm = float(np.mean(db_eval_iou(G, M))), float(np.mean(db_eval_boundary(G, M)))
            motion[vid] = dict(J=Jm, F=Fm, JF=(Jm + Fm) / 2)
    print(vid, {k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in r.items()}
                for k, r in res[vid].items()}, '| motion-mask', {k: round(vv, 3) for k, vv in motion[vid].items()},
          flush=True)

summ = {}
for eid in ('0', '1', '2'):
    rows = [res[v][eid] for v in res if 'JF' in res[v][eid]]
    summ[eid] = dict(exp=rows[0]['exp'], J=float(np.mean([r['J'] for r in rows])),
                     F=float(np.mean([r['F'] for r in rows])), JF=float(np.mean([r['JF'] for r in rows])))
for eid in ('1', '2'):
    summ[eid]['point_acc'] = float(np.mean([res[v][eid]['point_acc'] for v in res]))
    summ[eid]['point_chance'] = float(np.mean([res[v][eid]['point_chance'] for v in res]))
    for k in ('J_pf', 'F_pf', 'JF_pf', 'point_acc_pf'):
        if all(k in res[v][eid] for v in res):
            summ[eid][k] = float(np.mean([res[v][eid][k] for v in res]))
summ['referring_mean_JF'] = float(np.mean([summ[e]['JF'] for e in ('0', '1', '2')]))
summ['neg_pred_area'] = float(np.mean([res[v]['3']['pred_area'] for v in res]))
summ['neg_empty_frac'] = float(np.mean([res[v]['3']['empty_frac'] for v in res]))
summ['pos_mice_pred_area'] = float(np.mean([res[v]['0']['pred_area'] for v in res]))
summ['motion_mask_mice_JF'] = float(np.mean([motion[v]['JF'] for v in motion]))
json.dump(dict(summary=summ, per_clip=res, motion_mask=motion), open(out_json, 'w'), indent=1)
print(json.dumps(summ, indent=1))
