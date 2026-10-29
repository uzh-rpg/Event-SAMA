"""Render every annotated frame of an event eval set to a pseudo-RGB JPEG, using exactly the
functions and time convention of the Event-SAMA eval (RefVOSEventEvalDataset):
  frame -> centre us (_frame_index_to_us), window = 1e6/event_fps us centred on it
  (_accumulate_per_frame), rendering R=OFF, G=ON+OFF, B=ON with per-window max normalisation
  (_voxel_to_pseudo_rgb).
The output folder <out>/<vid>/<frame>.jpg is a drop-in image_folder for the unmodified RGB
SAMA-4B eval (projects/llava_sam2/evaluation/ref_vos_eval_st.py), i.e. the
'events -> RGB-like frames -> frozen RGB model' baseline.

usage: render_pseudorgb_frames.py <event_folder> <expression_json> <out_dir> [event_fps=100] [shard] [nshard]
"""
import json
import os
import sys

import numpy as np
from PIL import Image

from event_refvos_dataset import RefVOSEventEvalDataset, _safe_h5_meta
from event_voxel_dataset_v2 import _accumulate_per_frame
from projects.llava_sam2.datasets.EventVoxel_Dataset import _voxel_to_pseudo_rgb

ev_folder, exp_file, out = sys.argv[1:4]
fps = float(sys.argv[4]) if len(sys.argv) > 4 else 100.0
shard = int(sys.argv[5]) if len(sys.argv) > 5 else 0
nshard = int(sys.argv[6]) if len(sys.argv) > 6 else 1
half = int(round(1e6 / fps)) // 2
videos = json.load(open(exp_file))['videos']
to_us = RefVOSEventEvalDataset._frame_index_to_us
n_img = 0
for k, vid in enumerate(sorted(videos)):
    if k % nshard != shard:
        continue
    ev_path = os.path.join(ev_folder, vid, 'events.h5')
    meta = _safe_h5_meta(ev_path)
    if meta is None:
        print('missing', vid, flush=True)
        continue
    n_ev, t_min, t_max, native_fps, H, W = meta
    native_fps = native_fps or 24.0
    frames = sorted(videos[vid]['frames'])
    centres = [to_us(None, i, native_fps, frames, t_min, t_max) for i in range(len(frames))]
    os.makedirs(f'{out}/{vid}', exist_ok=True)
    for s in range(0, len(frames), 64):
        vox = _accumulate_per_frame(ev_path, centres_us=centres[s:s + 64], half_us=half, H_hint=H, W_hint=W)
        for j in range(vox.shape[0]):
            Image.fromarray(_voxel_to_pseudo_rgb(vox[j])).save(f'{out}/{vid}/{frames[s + j]}.jpg', quality=95)
            n_img += 1
print(f'shard {shard}/{nshard}: wrote {n_img} frames', flush=True)
