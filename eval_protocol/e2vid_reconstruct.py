"""E2VID (Rebecq et al., TPAMI 2019; pretrained E2VID_lightweight) reconstruction of every
annotated frame of an event eval set, as a drop-in image_folder for the unmodified RGB SAMA-4B
eval: the 'events -> reconstructed video -> frozen RGB model' baseline.

E2VID is run with its default settings (fixed-size windows of N = 0.35*W*H events, 5-bin voxel
grid, nonzero-normalisation, recurrent state over the whole clip, no auto-HDR / filters). Each
annotated frame takes the most recent reconstruction whose window ends at or before the frame
timestamp (frame time convention identical to RefVOSEventEvalDataset._frame_index_to_us).

usage: e2vid_reconstruct.py <e2vid_repo> <weights> <event_folder> <expression_json> <out_dir> [shard] [nshard]
"""
import json
import os
import sys

import h5py
import numpy as np
import torch
from PIL import Image

repo, weights, ev_folder, exp_file, out = sys.argv[1:6]
shard = int(sys.argv[6]) if len(sys.argv) > 6 else 0
nshard = int(sys.argv[7]) if len(sys.argv) > 7 else 1
sys.path.insert(0, repo)
from model.model import *  # noqa: E402,F401,F403
from utils.inference_utils import CropParameters, events_to_voxel_grid_pytorch  # noqa: E402

dev = torch.device('cuda')
raw = torch.load(weights, map_location='cpu', weights_only=False)
arch = raw['arch']
mtype = raw['model'] if 'model' in raw else raw['config']['model']
model = eval(arch)(mtype)
model.load_state_dict(raw['state_dict'])
model = model.to(dev).eval()
NB = model.num_bins


def frame_us(i, frames, fps, t_min, t_max):
    cands = []
    try:
        cands.append(int(round(int(frames[i]) * 1e6 / fps)))
    except Exception:
        pass
    cands.append(int(round(i * 1e6 / fps)))
    for c in cands:
        if t_min <= c <= t_max:
            return c
    return cands[-1]


videos = json.load(open(exp_file))['videos']
n_img = 0
for k, vid in enumerate(sorted(videos)):
    if k % nshard != shard:
        continue
    p5 = os.path.join(ev_folder, vid, 'events.h5')
    if not os.path.exists(p5):
        print('missing', vid, flush=True)
        continue
    with h5py.File(p5, 'r') as f:
        t = f['t'][:].astype(np.float64)
        x = f['x'][:].astype(np.float64)
        y = f['y'][:].astype(np.float64)
        p = f['p'][:].astype(np.float64)
        H, W = int(f.attrs['native_height']), int(f.attrs['native_width'])
        fps = float(f.attrs.get('native_fps', 0) or 24.0)
    if len(t) == 0:
        continue
    frames = sorted(videos[vid]['frames'])
    centres = np.array([frame_us(i, frames, fps, t[0], t[-1]) for i in range(len(frames))])
    N = max(1000, int(0.35 * W * H))
    crop = CropParameters(W, H, model.num_encoders)
    state, imgs, ends, n_reset = None, [], [], 0
    with torch.no_grad():
        for s in range(0, len(t), N):
            ev = np.stack([t[s:s + N], x[s:s + N], y[s:s + N], p[s:s + N]], 1)
            if len(ev) < 2:
                break
            vox = events_to_voxel_grid_pytorch(ev, NB, W, H, dev).unsqueeze(0)
            nz = vox != 0
            n = nz.sum()
            if n > 0:
                mean = vox.sum() / n
                std = torch.sqrt(((vox ** 2).sum() / n - mean ** 2).clamp(min=0))
                # synthetic events share timestamps in bursts -> std can be 0 (NaN would poison the state)
                std = std if torch.isfinite(std) and std > 1e-6 else torch.ones_like(std)
                vox = nz.float() * (vox - mean) / std
            pred, new_state = model(crop.pad(vox), state)
            if not torch.isfinite(pred).all():
                n_reset += 1
                state = None
                pred, new_state = model(crop.pad(torch.nan_to_num(vox)), None)
            state = new_state
            img = (255.0 * pred[0, 0, crop.iy0:crop.iy1, crop.ix0:crop.ix1]).clamp(0, 255).byte().cpu().numpy()
            # keep only reconstructions needed by some frame (memory)
            imgs.append(img)
            ends.append(ev[-1, 0])
            if len(imgs) > 1 and np.searchsorted(centres, ends[-2], 'left') == np.searchsorted(centres, ends[-1], 'left'):
                # no frame centre falls between the previous and this window end -> drop previous
                imgs.pop(-2)
                ends.pop(-2)
    ends = np.array(ends)
    if n_reset:
        print(vid, 'non-finite output, state reset', n_reset, 'times', flush=True)
    os.makedirs(f'{out}/{vid}', exist_ok=True)
    for i, c in enumerate(centres):
        j = max(0, int(np.searchsorted(ends, c, 'right')) - 1)
        Image.fromarray(np.repeat(imgs[j][:, :, None], 3, 2)).save(f'{out}/{vid}/{frames[i]}.jpg', quality=95)
        n_img += 1
print(f'shard {shard}/{nshard}: wrote {n_img} frames', flush=True)
