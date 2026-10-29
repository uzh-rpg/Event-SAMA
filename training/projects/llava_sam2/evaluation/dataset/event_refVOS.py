import os
import json

import h5py
import mmengine
import numpy as np
import copy

from .base_eval_dataset import BaseEvalDataset

SEG_PROMPT = "<image>\nPlease segment {}."

# Polarity ablation switch (EVENT_MERGE_POLARITY=1): must match
# EventVoxel_Dataset.py's flag exactly, so eval preprocessing matches
# whatever the checkpoint was actually trained on. Unset (default)
# everywhere else -- bit-identical to before for every other checkpoint.
_MERGE_POLARITY = os.environ.get('EVENT_MERGE_POLARITY', '0') == '1'
if _MERGE_POLARITY:
    print('[EventRefVOSDataset] EVENT_MERGE_POLARITY=1 -- polarity ablation '
          'ACTIVE: ON/OFF channels collapsed to their mean in every voxel.',
          flush=True)


def _accumulate_polarity_histogram(x, y, p, start_us, end_us, t, H, W):
    """(2, H, W) float32 polarity histogram for events with t in [start_us, end_us).
    ch0 = ON-event counts, ch1 = OFF-event counts. Mirrors
    EventVoxel_Dataset._accumulate_polarity_histogram exactly, including the
    EVENT_MERGE_POLARITY ablation branch."""
    lo = np.searchsorted(t, start_us, side='left')
    hi = np.searchsorted(t, end_us, side='left')
    if hi <= lo:
        return np.zeros((2, H, W), dtype=np.float32)

    xx = x[lo:hi].astype(np.int64)
    yy = y[lo:hi].astype(np.int64)
    pp = p[lo:hi]

    in_bounds = (xx >= 0) & (xx < W) & (yy >= 0) & (yy < H)
    xx = xx[in_bounds]
    yy = yy[in_bounds]
    pp = pp[in_bounds]

    on = pp > 0
    off = ~on

    flat = yy * W + xx
    out = np.zeros((2, H * W), dtype=np.float32)
    if on.any():
        np.add.at(out[0], flat[on], 1.0)
    if off.any():
        np.add.at(out[1], flat[off], 1.0)
    out = out.reshape(2, H, W)
    if _MERGE_POLARITY:
        m = out.sum(axis=0, keepdims=True) * 0.5
        out = np.repeat(m, 2, axis=0)
    return out


class EventRefVOSDataset(BaseEvalDataset):
    """Event-voxel counterpart of RefVOSDataset: same MeViS/ReVOS-style
    meta_expressions.json / mask_dict.json schema, but __getitem__ returns
    raw (2, H, W) event-voxel frames (one per annotated frame) instead of
    RGB PIL Images, for feeding to model.predict_forward_event.

    ``event_folder`` layout: ``<event_folder>/<video_id>/events.h5`` (or
    ``<event_folder>/<video_id>.h5`` if event_layout='flat'), matching the
    training-time event dataset convention exactly -- video_id here is
    expected to already be resolved to the correct on-disk directory name
    by the caller if the source needs prefix remapping (LV-VIS/OVIS/etc.).
    """

    EVENT_FILE_NAME = 'events.h5'

    def __init__(self,
                 event_folder,
                 expression_file,
                 mask_file,
                 event_fps: float = 100.0,
                 event_layout: str = 'nested',
                 fallback_native_fps=None,
                 event_path_resolver=None):
        super().__init__()
        vid2metaid, metas, mask_dict = self.json_file_preprocess(expression_file, mask_file)
        self.vid2metaid = vid2metaid
        self.videos = list(self.vid2metaid.keys())
        self.mask_dict = mask_dict
        self.text_data = metas

        self.event_folder = event_folder
        self.event_fps = float(event_fps)
        self.event_window_us = int(round(1e6 / self.event_fps))
        self.event_layout = event_layout
        self.fallback_native_fps = fallback_native_fps
        # Optional override for video_id -> events.h5 path resolution
        # (e.g. the ReVOS LV-VIS/OVIS/TAO/MOSE prefix remap used at
        # training time). Defaults to the plain nested/flat layout below.
        self._event_path_resolver = event_path_resolver

        self._prefilter_missing_h5()

    def __len__(self):
        return len(self.text_data)

    def real_len(self):
        return len(self.text_data)

    def json_file_preprocess(self, expression_file, mask_file):
        with open(expression_file, 'r') as f:
            expression_datas = json.load(f)['videos']
        metas = []
        vid2metaid = {}
        for vid_name in expression_datas:
            vid_express_data = expression_datas[vid_name]

            vid_frames = sorted(vid_express_data['frames'])
            vid_len = len(vid_frames)

            exp_id_list = sorted(list(vid_express_data['expressions'].keys()))
            for exp_id in exp_id_list:
                exp_dict = vid_express_data['expressions'][exp_id]
                meta = {}
                meta['video'] = vid_name
                meta['exp'] = exp_dict['exp']
                meta['frames'] = vid_frames
                meta['exp_id'] = exp_id
                meta['length'] = vid_len
                metas.append(meta)
                if vid_name not in vid2metaid.keys():
                    vid2metaid[vid_name] = []
                vid2metaid[vid_name].append(len(metas) - 1)

        if mask_file is not None:
            mask_dict = mmengine.load(mask_file)
        else:
            mask_dict = None
        return vid2metaid, metas, mask_dict

    def _event_path_for_video(self, video_id: str) -> str:
        if self._event_path_resolver is not None:
            return self._event_path_resolver(video_id)
        if self.event_layout == 'flat':
            return os.path.join(self.event_folder, video_id + '.h5')
        return os.path.join(self.event_folder, video_id, self.EVENT_FILE_NAME)

    def _prefilter_missing_h5(self):
        kept_videos = []
        dropped = []
        for vid in self.videos:
            try:
                p = self._event_path_for_video(vid)
            except Exception:
                dropped.append(vid)
                continue
            if os.path.isfile(p):
                kept_videos.append(vid)
            else:
                dropped.append(vid)
        if dropped:
            print(f"[EventRefVOSDataset] pre-filter: dropping "
                  f"{len(dropped)}/{len(self.videos)} videos with no event "
                  f"H5 on disk (e.g. {dropped[:3]})", flush=True)
        self.videos = kept_videos
        self.vid2metaid = {k: v for k, v in self.vid2metaid.items()
                           if k in set(kept_videos)}
        # Also drop text_data entries whose video got filtered, so __len__/
        # index-based access stays consistent.
        kept_set = set(kept_videos)
        self.text_data = [m for m in self.text_data if m['video'] in kept_set]

    def _frame_index_to_us(self, frame_idx, native_fps, frames_list, t_min, t_max):
        candidates = []
        try:
            numeric = int(str(frames_list[frame_idx]))
            candidates.append(int(round(numeric * 1e6 / native_fps)))
        except Exception:
            pass
        candidates.append(int(round(frame_idx * 1e6 / native_fps)))
        for c in candidates:
            if t_min <= c <= t_max:
                return c
        return candidates[-1]

    def _load_event_voxels(self, video_id, frame_indices, frames_list):
        path = self._event_path_for_video(video_id)
        with h5py.File(path, 'r') as f:
            x = f['x'][:]
            y = f['y'][:]
            t = f['t'][:]
            p = f['p'][:]
            H = int(f.attrs.get('native_height', y.max() + 1 if len(y) else 0))
            W = int(f.attrs.get('native_width', x.max() + 1 if len(x) else 0))
            native_fps = float(f.attrs.get(
                'native_fps', self.fallback_native_fps or self.event_fps))
            t_unit = f.attrs.get('t_unit', b'us')
            if isinstance(t_unit, bytes):
                t_unit = t_unit.decode('utf-8', errors='replace')
            assert t_unit == 'us', f'expected t_unit=us, got {t_unit}'

        half = self.event_window_us // 2
        t_min = int(t[0]) if len(t) else 0
        t_max = int(t[-1]) if len(t) else 0
        voxels = []
        for f_idx in frame_indices:
            t_center = self._frame_index_to_us(
                f_idx, native_fps, frames_list, t_min, t_max)
            voxels.append(_accumulate_polarity_histogram(
                x, y, p,
                start_us=t_center - half,
                end_us=t_center + half,
                t=t, H=H, W=W,
            ))
        return voxels, H, W

    def __getitem__(self, index):
        video_obj_info = copy.deepcopy(self.text_data[index])
        exp = video_obj_info['exp']

        data_dict = {}

        video_id = video_obj_info['video']
        frames_list = video_obj_info['frames']
        frame_indices = list(range(len(frames_list)))

        event_frames, H, W = self._load_event_voxels(video_id, frame_indices, frames_list)

        data_dict['type'] = 'video'
        data_dict['index'] = index
        data_dict['video_id'] = video_id
        data_dict['event_frames'] = event_frames  # list of (2, H, W) float32
        data_dict['exp_id'] = video_obj_info['exp_id']

        data_dict['frames'] = video_obj_info['frames']
        data_dict['text_prompt'] = SEG_PROMPT.format(exp) if '?' not in exp else exp
        data_dict['image_folder'] = self.event_folder

        data_dict['length'] = video_obj_info['length']
        data_dict['ori_height'] = H
        data_dict['ori_width'] = W

        return data_dict
