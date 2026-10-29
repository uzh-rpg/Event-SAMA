"""MouseSIS (Hamann et al., ECCVW 2024; real 1280x720 events, top view of mice) val split
-> events.h5 in the Event-SAMA eval layout + expression json + per-expression GT masks.

Source h5: x,y (float, integer-valued, image coords), t (us, uint32), p (0/1), img_ts (us).
Annotations: start/length RLE pairs, column-major (verified against json bboxes).

Rules fixed BEFORE any model run:
  * hot pixels: isolated pixels with > 100 events and > 10x the 3x3 neighbourhood median
    over the clip are dropped (seq25: 64 px, 0.29 % of events).
  * evaluated frames: every 3rd annotated frame (~10 fps).
  * expressions per clip:
      0 'the mice'                  GT = union of all mouse instances
      1 'the mouse on the far left'  GT = instance with min centroid x in the first frame
      2 'the mouse on the far right' GT = instance with max centroid x in the first frame
      3 'a red car'                  negative control, GT empty
Time base: frame name n (1-based original frame index) sits at t = n * STEP, STEP = mean
frame interval, so the eval script's t = num * 1e6 / native_fps hits img_ts[n-1] (+-2 us).

usage: convert_mousesis.py <src_dir with seqNN.h5 + annotations> <out_dir> [annotations.json] [shard nshard]
Also writes gt_perframe/<clip>/{1,2}: left-/right-most visible mouse in each frame (primary GT
for the spatial expressions; gt/ keeps the first-frame-identity variant).
"""
import json
import os
import sys

import h5py
import numpy as np
from PIL import Image
from scipy.ndimage import median_filter

H, W = 720, 1280
SUB = 3
EXPS = ['the mice', 'the mouse on the far left', 'the mouse on the far right', 'a red car']


def rle_decode(s):
    h, w = s['size']
    r = np.array(s['counts'].split(), dtype=np.int64).reshape(-1, 2)
    m = np.zeros(h * w, np.uint8)
    for st, n in r:
        m[st:st + n] = 1
    return m.reshape(w, h).T.astype(bool)


def main():
    src, out = sys.argv[1], sys.argv[2]
    ann_name = sys.argv[3] if len(sys.argv) > 3 else 'val_annotations.json'
    shard = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    nshard = int(sys.argv[5]) if len(sys.argv) > 5 else 1
    ann = json.load(open(f'{src}/{ann_name}'))
    by_vid = {}
    for a in ann['annotations']:
        by_vid.setdefault(a['video_id'], []).append(a)
    videos, stats = {}, {}
    for k, vinfo in enumerate(ann['videos']):
        if k % nshard != shard:
            continue
        vid = vinfo['id']
        path = f'{src}/seq{vid}.h5'
        if not os.path.exists(path):
            print('missing', path, flush=True)
            continue
        with h5py.File(path, 'r') as f:
            x = f['x'][:].astype(np.int64)
            y = f['y'][:].astype(np.int64)
            t = f['t'][:].astype(np.int64)
            p = f['p'][:]
            ts = f['img_ts'][:].astype(np.float64)
        n_img = len(ts)
        assert n_img == vinfo['length'], (vid, n_img, vinfo['length'])
        step = float(np.mean(np.diff(ts)))
        t0 = ts[0] - step

        cnt = np.bincount(y * W + x, minlength=H * W).reshape(H, W).astype(np.float64)
        med = median_filter(cnt, size=3)
        hot = (cnt > 100) & (cnt > 10 * np.maximum(med, 1))
        keep = ~hot[y, x]
        t_rel = np.rint(t - t0).astype(np.int64)
        keep &= (t_rel >= 0)
        x, y, t_rel, p = x[keep], y[keep], t_rel[keep], p[keep]
        assert np.all(np.diff(t_rel) >= 0)

        name = f'mouse{vid}'
        os.makedirs(f'{out}/{name}', exist_ok=True)
        with h5py.File(f'{out}/{name}/events.h5', 'w') as g:
            kw = dict(compression='gzip', compression_opts=4)
            g.create_dataset('x', data=x.astype(np.uint16), **kw)
            g.create_dataset('y', data=y.astype(np.uint16), **kw)
            g.create_dataset('t', data=t_rel.astype(np.int32), **kw)
            g.create_dataset('p', data=np.where(p > 0, 1, -1).astype(np.int8), **kw)
            g.attrs['native_height'] = H
            g.attrs['native_width'] = W
            g.attrs['native_fps'] = 1e6 / step

        idx = list(range(0, n_img, SUB))
        frames = [f'{i + 1:05d}' for i in idx]
        insts = by_vid[vid]
        # identity for left/right: centroid x in the first frame where the instance is annotated
        cx = []
        for a in insts:
            s0 = next(s for s in a['segmentations'] if s is not None)
            m0 = rle_decode(s0)
            cx.append(np.nonzero(m0)[1].mean())
        left, right = int(np.argmin(cx)), int(np.argmax(cx))
        for eid in range(len(EXPS)):
            os.makedirs(f'{out}/gt/{name}/{eid}', exist_ok=True)
        for eid in ('1', '2'):
            os.makedirs(f'{out}/gt_perframe/{name}/{eid}', exist_ok=True)
        for i, fr in zip(idx, frames):
            ms = [rle_decode(a['segmentations'][i]) if a['segmentations'][i] is not None
                  else np.zeros((H, W), bool) for a in insts]
            gts = [np.any(ms, 0), ms[left], ms[right], np.zeros((H, W), bool)]
            for eid, m in enumerate(gts):
                Image.fromarray(m.astype(np.uint8) * 255).save(f'{out}/gt/{name}/{eid}/{fr}.png')
            # per-frame spatial GT (primary for left/right): extreme visible mouse in this frame
            vis = [m for m in ms if m.any()]
            vcx = [np.nonzero(m)[1].mean() for m in vis]
            for eid, fn in (('1', np.argmin), ('2', np.argmax)):
                m = vis[int(fn(vcx))] if vis else np.zeros((H, W), bool)  # no mouse annotated in this frame
                Image.fromarray(m.astype(np.uint8) * 255).save(f'{out}/gt_perframe/{name}/{eid}/{fr}.png')
        videos[name] = dict(frames=frames,
                            expressions={str(i): {'exp': e} for i, e in enumerate(EXPS)})
        win = np.diff(np.searchsorted(t_rel, np.arange(0, t_rel[-1], 10000)))
        stats[name] = dict(n_instances=len(insts), n_frames_eval=len(frames), n_img=n_img,
                           step_us=step, hot_px=int(hot.sum()),
                           hot_ev_frac=float(1 - keep.mean()), n_events=int(len(t_rel)),
                           median_events_per_10ms=float(np.median(win)),
                           left_inst=int(insts[left]['id']), right_inst=int(insts[right]['id']))
        print(name, stats[name], flush=True)
    sfx = f'_shard{shard}' if nshard > 1 else ''
    json.dump({'videos': videos}, open(f'{out}/meta_expressions{sfx}.json', 'w'), indent=1)
    json.dump(stats, open(f'{out}/stats{sfx}.json', 'w'), indent=1)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
