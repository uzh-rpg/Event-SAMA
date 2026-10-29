"""Memory-bounded event-voxel datasets for Event SAMA.

This module sits alongside ``EventVoxel_Dataset.py`` (which lives in
``/users/sfedorchenko/Event-SAMA-train`` and is too large for me to edit
without bumping into the home-dir byte quota) and provides drop-in
replacements that:

  1. Stream-read events from HDF5 using on-disk binary search + slice I/O,
     so per-sample memory scales with the requested time window rather
     than with the full event file.

  2. Enumerate "virtual clips" at __init__ time: each long event file is
     split into K bounded sub-windows of ``virtual_clip_seconds`` each,
     and each sub-window becomes its own datapoint. This caps per-sample
     memory and -- critically for DDP -- prevents one rank from getting
     stuck loading a 19 GiB ChatUniVi clip while the other ranks spin at
     a stalled gradient allreduce.

The original V1 datasets did ``f['x'][:]; f['y'][:]; f['t'][:]; f['p'][:]``
which materialises the entire file into numpy arrays. With ChatUniVi
event clips up to 19 GiB and grad_accum=4, a single rank could balloon
to ~76 GiB CPU RAM and either OOM or drag the whole DDP iteration to a
multi-hour halt. We measured this empirically (rank 3 stuck at 17.5%
process RSS = ~80 GB while ranks 0/1/2 burned 100% util on the stalled
allreduce).

Usage in your config:

    from event_voxel_dataset_v2 import (
        VideoMeVISEventDatasetV2,
        VideoLVVISEventDatasetV2,
        VideoChatUniViEventDatasetV2,
        RefCOCOEventDatasetV2,
    )

    train_dataset = dict(
        type=VideoChatUniViEventDatasetV2,
        ...
        virtual_clip_seconds=20.0,    # <-- new, splits long clips
        meta_cache_path='/iopsstor/.../chatunivi_event_meta.json',
    )
"""
from __future__ import annotations

import copy
import json
import os
import random
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# We rely on the existing module to provide the parent classes (so we don't
# have to reimplement all the text/mask machinery). The project must be on
# PYTHONPATH; the sbatch wrapper sets PROJECT_DIR on PYTHONPATH already.
from projects.llava_sam2.datasets.EventVoxel_Dataset import (
    EVENT_ASINH_SCALE,
    EventVoxelTransform,
    VideoEventBaseDataset,
    VideoLVVISDataset,  # MeViS-schema-compatible LVVIS wrapper
    VideoLVVISEventDataset,
    VideoMeVISEventDataset,
    VideoChatUniViEventDataset,
    RefCOCOEventDataset,
    _accumulate_polarity_histogram,
    _first_rle_size,
    _voxel_to_pseudo_rgb,
)
from projects.llava_sam2.datasets.encode_fn import video_lisa_encode_fn

# --- Extra base datasets we add event analogues for in V3 -------------------
import re  # noqa: E402

from projects.llava_sam2.datasets.RefYoutubeVOS_Dataset import (  # noqa: E402
    VideoRefYoutubeVOSDataset,
)
from projects.llava_sam2.datasets.GCG_Dataset import (  # noqa: E402
    FlickrGCGDataset,
    GranDfGCGDataset,
    OpenPsgGCGDataset,
    RefCOCOgGCGDataset,
)
from projects.llava_sam2.datasets.vqa_dataset import LLaVADataset  # noqa: E402
from projects.llava_sam2.datasets.ReSAM2_Dataset import (  # noqa: E402
    VideoSAM2Dataset,
    decode_masklet,
)
from projects.llava_sam2.datasets.utils import (  # noqa: E402
    get_sparse_indices_uniform,
    get_dense_indices,
)
from xtuner.utils import DEFAULT_IMAGE_TOKEN  # noqa: E402


# =============================================================================
# Streaming HDF5 helpers
# =============================================================================

def _h5_searchsorted(dset, target: int, side: str = 'left') -> int:
    """Binary-search a sorted h5py dataset for the first index ``i`` such
    that ``dset[i] >= target`` (side='left') or ``dset[i] > target``
    (side='right'). O(log N) single-element reads ≈ tens of ms even for
    billions of events.
    """
    n = int(dset.shape[0]) if hasattr(dset, 'shape') else len(dset)
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        v = int(dset[mid])
        if (v < target) if side == 'left' else (v <= target):
            lo = mid + 1
        else:
            hi = mid
    return lo


def _peek_h5_meta(h5_path: str) -> Optional[Dict]:
    """Read just the file-level metadata + (t_min, t_max) without
    materialising x/y/t/p. Returns None on read errors so callers can skip
    broken/missing files.
    """
    try:
        import h5py
        with h5py.File(h5_path, 'r') as f:
            t_dset = f['t']
            n = int(t_dset.shape[0])
            t_min = int(t_dset[0]) if n > 0 else 0
            t_max = int(t_dset[-1]) if n > 0 else 0
            t_unit = f.attrs.get('t_unit', b'us')
            if isinstance(t_unit, bytes):
                t_unit = t_unit.decode('utf-8', errors='replace')
            return dict(
                H=int(f.attrs.get('native_height', 0)) or None,
                W=int(f.attrs.get('native_width', 0)) or None,
                native_fps=float(f.attrs.get('native_fps', 30.0)),
                n_events=n,
                t_min_us=t_min,
                t_max_us=t_max,
                t_unit=t_unit,
            )
    except Exception as e:
        print(f"[event-meta] failed to peek {h5_path}: {e!r}", flush=True)
        return None


def _load_event_window(h5_path: str, t_start_us: int, t_end_us: int):
    """Stream-read events in the half-open window [t_start_us, t_end_us).

    Returns (x, y, t, p, H, W, native_fps, t_min_full, t_max_full). The
    arrays only contain events whose timestamp falls in the window; their
    memory footprint scales with the window, not the file size.
    """
    import h5py
    with h5py.File(h5_path, 'r') as f:
        t_dset = f['t']
        n = int(t_dset.shape[0])
        H = int(f.attrs.get('native_height', 1)) or 1
        W = int(f.attrs.get('native_width', 1)) or 1
        native_fps = float(f.attrs.get('native_fps', 30.0))
        if n == 0:
            return (np.zeros(0, dtype=np.int16),
                    np.zeros(0, dtype=np.int16),
                    np.zeros(0, dtype=np.int32),
                    np.zeros(0, dtype=np.int8),
                    H, W, native_fps, 0, 0)
        t_min_full = int(t_dset[0])
        t_max_full = int(t_dset[-1])
        ts = max(int(t_start_us), t_min_full)
        te = min(int(t_end_us), t_max_full + 1)
        if te <= ts:
            return (np.zeros(0, dtype=np.int16),
                    np.zeros(0, dtype=np.int16),
                    np.zeros(0, dtype=np.int32),
                    np.zeros(0, dtype=np.int8),
                    H, W, native_fps, t_min_full, t_max_full)
        lo = _h5_searchsorted(t_dset, ts, side='left')
        hi = _h5_searchsorted(t_dset, te, side='left')
        if hi <= lo:
            return (np.zeros(0, dtype=np.int16),
                    np.zeros(0, dtype=np.int16),
                    np.zeros(0, dtype=np.int32),
                    np.zeros(0, dtype=np.int8),
                    H, W, native_fps, t_min_full, t_max_full)
        x = f['x'][lo:hi]
        y = f['y'][lo:hi]
        t = f['t'][lo:hi]
        p = f['p'][lo:hi]
    # Refine H/W from the tail if attrs were missing -- only uses the slice.
    if H == 1 and len(y):
        H = int(y.max()) + 1
    if W == 1 and len(x):
        W = int(x.max()) + 1
    return x, y, t, p, H, W, native_fps, t_min_full, t_max_full


def _accumulate_per_frame(h5_path: str, centres_us, half_us: int,
                          H_hint: Optional[int] = None,
                          W_hint: Optional[int] = None,
                          ) -> np.ndarray:
    """Open ``h5_path`` ONCE and accumulate one polarity histogram per
    centre timestamp, reading only the events in [c - half_us, c + half_us)
    for each centre. Memory used per call is O(events in 2*half_us window),
    independent of the file size or the span of ``centres_us``.

    This is the right loader for ChatUniVi-style long clips where the
    centres can be tens of seconds apart but the per-frame integration
    window is just ``2 * half_us`` (~10 ms). Loading the bounding window
    [min(c)-half, max(c)+half) instead would drag in tens of GB of events
    on the worst clips (51 B events / 240 s ≈ 215 M events/s -> a 30 s
    bounding window = 6 B events).
    """
    import h5py
    n_select = len(centres_us)
    with h5py.File(h5_path, 'r') as f:
        t_dset = f['t']
        n = int(t_dset.shape[0])
        H = int(f.attrs.get('native_height', 0)) or H_hint or 1
        W = int(f.attrs.get('native_width', 0)) or W_hint or 1
        if n == 0:
            return np.zeros((n_select, 2, H, W), dtype=np.float32)
        t_min = int(t_dset[0])
        t_max = int(t_dset[-1])
        voxels = []
        for c in centres_us:
            ts = max(int(c) - half_us, t_min)
            te = min(int(c) + half_us, t_max + 1)
            if te <= ts:
                voxels.append(np.zeros((2, H, W), dtype=np.float32))
                continue
            lo = _h5_searchsorted(t_dset, ts, side='left')
            hi = _h5_searchsorted(t_dset, te, side='left')
            if hi <= lo:
                voxels.append(np.zeros((2, H, W), dtype=np.float32))
                continue
            x = f['x'][lo:hi]
            y = f['y'][lo:hi]
            t = f['t'][lo:hi]
            p = f['p'][lo:hi]
            voxels.append(_accumulate_polarity_histogram(
                x, y, p, start_us=ts, end_us=te, t=t, H=H, W=W))
            # explicit del helps when the slice happens to be huge
            del x, y, t, p
        return np.stack(voxels, axis=0)


def _accumulate_window(x, y, p, t, start_us: int, end_us: int,
                       H: int, W: int) -> np.ndarray:
    """Wrap the existing histogram accumulator. The underlying function in
    ``EventVoxel_Dataset`` searches inside the supplied (x, y, t, p) for the
    [start, end) sub-range, so passing pre-windowed arrays is safe.
    """
    return _accumulate_polarity_histogram(x, y, p,
                                          start_us=start_us, end_us=end_us,
                                          t=t, H=H, W=W)


# =============================================================================
# ChatUniVi: virtual clip splitting + windowed loads
# =============================================================================

class VideoChatUniViEventDatasetV2(VideoChatUniViEventDataset):
    """ChatUniVi event-QA with bounded per-sample memory.

    On top of the V1 class this:
      * Peeks at every event H5 once at __init__ (cached on disk), reading
        only attrs + first/last t -- no x/y/p materialisation.
      * Skips entries whose H5 is missing or empty.
      * Splits each clip into N = max(1, ceil(duration / virtual_clip_seconds))
        virtual sub-clips. Each virtual sub-clip becomes its own datapoint;
        __getitem__ only loads events in that sub-window.
      * Falls back gracefully (re-draw) on any per-sample read error.
    """

    def __init__(self,
                 *args,
                 virtual_clip_seconds: float = 20.0,
                 max_clips_per_video: int = 8,
                 meta_cache_path: Optional[str] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.virtual_clip_seconds = float(virtual_clip_seconds)
        self.max_clips_per_video = int(max_clips_per_video)
        self.meta_cache_path = meta_cache_path

        # Peek meta for every available clip once.
        meta = self._load_or_build_meta_cache()

        # Build the virtual-clip table: each entry is
        #   (text_data_idx, t_start_us, t_end_us)
        clip_us = int(round(self.virtual_clip_seconds * 1e6))
        self._virtual_clips: List[Tuple[int, int, int]] = []
        skipped_no_h5 = 0
        skipped_empty = 0
        total_clips = 0
        for ti in self._available_indices:
            item = self.text_data[ti]
            stem = os.path.splitext(os.path.basename(item['video']))[0]
            m = meta.get(stem)
            if m is None:
                skipped_no_h5 += 1
                continue
            n_events = m.get('n_events', 0)
            if n_events == 0:
                skipped_empty += 1
                continue
            t0 = int(m['t_min_us'])
            t1 = int(m['t_max_us'])
            duration_us = max(0, t1 - t0)
            if duration_us <= clip_us:
                self._virtual_clips.append((ti, t0, t1 + 1))
                total_clips += 1
            else:
                k = min(self.max_clips_per_video,
                        max(1, duration_us // clip_us))
                edges = np.linspace(t0, t1 + 1, num=k + 1).astype(np.int64)
                for j in range(k):
                    self._virtual_clips.append(
                        (ti, int(edges[j]), int(edges[j + 1])))
                    total_clips += 1
        print(f"[ChatUniVi-EventV2] enumerated {total_clips} virtual clips "
              f"from {len(self._available_indices)} available videos "
              f"(skipped no-meta={skipped_no_h5}, empty={skipped_empty}); "
              f"clip_seconds={self.virtual_clip_seconds}, "
              f"max_clips_per_video={self.max_clips_per_video}",
              flush=True)
        if not self._virtual_clips:
            raise RuntimeError(
                f"No usable virtual clips for ChatUniVi event dataset "
                f"under {self.event_folder}.")

    # -- meta cache -----------------------------------------------------------
    def _load_or_build_meta_cache(self) -> Dict[str, Dict]:
        if self.meta_cache_path and os.path.isfile(self.meta_cache_path):
            try:
                with open(self.meta_cache_path, 'r') as f:
                    cache = json.load(f)
                print(f"[ChatUniVi-EventV2] loaded meta cache from "
                      f"{self.meta_cache_path} ({len(cache)} entries)",
                      flush=True)
                return cache
            except Exception as e:
                print(f"[ChatUniVi-EventV2] cache read failed ({e!r}); "
                      f"rebuilding.", flush=True)
        # Build from scratch -- one h5 peek per available stem.
        cache: Dict[str, Dict] = {}
        n = len(self._available_indices)
        for k, ti in enumerate(self._available_indices):
            item = self.text_data[ti]
            stem = os.path.splitext(os.path.basename(item['video']))[0]
            if stem in cache:
                continue
            m = _peek_h5_meta(self._event_path(stem))
            if m is not None:
                cache[stem] = m
            if (k + 1) % 500 == 0:
                print(f"[ChatUniVi-EventV2] meta peek {k + 1}/{n}", flush=True)
        if self.meta_cache_path:
            try:
                os.makedirs(os.path.dirname(self.meta_cache_path),
                            exist_ok=True)
                with open(self.meta_cache_path, 'w') as f:
                    json.dump(cache, f)
                print(f"[ChatUniVi-EventV2] wrote meta cache to "
                      f"{self.meta_cache_path} ({len(cache)} entries)",
                      flush=True)
            except Exception as e:
                print(f"[ChatUniVi-EventV2] cache write failed: {e!r}",
                      flush=True)
        return cache

    # -- length now reflects virtual clips ------------------------------------
    def __len__(self):
        return len(self._virtual_clips) * self.repeats

    # -- getitem reads only the requested window ------------------------------
    def __getitem__(self, index):
        # Map back to a virtual-clip tuple.
        v_idx = index % len(self._virtual_clips)
        text_data_idx, t_start_us, t_end_us = self._virtual_clips[v_idx]
        selected_data_dict = copy.deepcopy(self.text_data[text_data_idx])
        video_field = selected_data_dict['video']
        video_stem = os.path.splitext(os.path.basename(video_field))[0]

        conversation = selected_data_dict['conversations']
        # Choose dense_indices the same way the parent does.
        split_point = np.linspace(0, self.sparse_len,
                                  num=self.dense_len + 1, dtype=int)
        dense_indices = [
            np.random.randint(split_point[i], split_point[i + 1])
            for i in range(self.dense_len)
        ]
        if self.use_fast:
            text_dict = self.prepare_text(
                self.dense_len, conversation,
                num_image_tokens=self.patch_token,
                n_fast_images=self.n_fast_images,
            )
        else:
            text_dict = self.prepare_text(
                self.dense_len, conversation,
                num_image_tokens=self.patch_token,
            )

        try:
            voxels = self._load_voxels_in_window(
                video_stem, t_start_us, t_end_us, self.sparse_len)
        except Exception as e:
            print(f"[ChatUniVi-EventV2] read err on {video_stem} "
                  f"[{t_start_us},{t_end_us}): {e!r}; resampling.",
                  flush=True)
            return self.__getitem__(random.randint(
                0, len(self._virtual_clips) - 1))

        pixel_values = self.transformer(voxels)  # (N, 2, image_size, image_size)
        data_dict = dict(
            prompts=text_dict['prompts'],
            conversation=text_dict['conversation'],
            dense_indices=dense_indices,
            pixel_values=pixel_values,
            fast_images=None,
        )
        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = video_lisa_encode_fn(
            data_dict, tokenizer=self.tokenizer,
            max_length=self.max_length, with_image_token=True)
        data_dict.update(result)
        return data_dict

    def _load_voxels_in_window(self, video_stem: str,
                               t_start_us: int, t_end_us: int,
                               n_select: int) -> np.ndarray:
        """Sample ``n_select`` evenly-spaced timestamps inside the virtual
        clip [t_start_us, t_end_us) and accumulate one 2-channel polarity
        histogram per timestamp using a strict per-frame windowed read
        (``event_window_us`` wide, typically 10 ms each).

        Memory used: O(events in 10 ms), regardless of how long the
        virtual clip is or how dense the file is. This is what stops a
        single rank from blowing up to 80 GB CPU on the 51 B-event
        ChatUniVi worst case.
        """
        if n_select == 1:
            centres = np.array([(t_start_us + t_end_us) // 2], dtype=np.int64)
        else:
            # Spread centres slightly inside the window so the first/last
            # frame still has half-window of slack on both sides.
            half = self.event_window_us // 2
            inner_lo = t_start_us + half
            inner_hi = t_end_us - half
            if inner_hi <= inner_lo:
                inner_lo = t_start_us
                inner_hi = max(t_end_us, t_start_us + 1)
            centres = np.linspace(inner_lo, inner_hi, n_select).astype(np.int64)
        return _accumulate_per_frame(
            self._event_path(video_stem),
            centres_us=centres,
            half_us=self.event_window_us // 2,
        )


# =============================================================================
# Segmentation event datasets (MeViS, LVVIS): windowed loads only
# =============================================================================
#
# These datasets sample sparse_len frames at specific annotated timestamps,
# so virtual-clip splitting is not needed -- the existing sample structure
# already bounds the time span. We only swap the loader to use windowed
# reads, keeping the V1 __getitem__ semantics (and its mask handling) intact.

class _StreamingVideoEventBaseMixin:
    """Mixin: replaces ``_load_event_voxels`` with a windowed implementation
    that only reads the events around the requested timestamps. Drops in
    cleanly to any subclass of ``VideoEventBaseDataset``.

    Also pre-filters ``self.videos`` / ``self.vid2metaid`` at __init__
    time to drop entries whose event H5 is missing. The V1 dataset
    didn't do this and the first such sample crashed the whole DDP run
    via a torch DataLoader-worker FileNotFoundError that bubbled up
    uncaught into the main loop.
    """

    # How much pre/post padding (in microseconds) to read around the
    # min/max requested timestamp, so the existing accumulator's
    # searchsorted has slack on both ends.
    _WINDOW_PAD_US: int = 0  # event_window_us is added per-frame anyway

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prefilter_missing_h5()

    def _prefilter_missing_h5(self):
        """Drop video entries that don't have an events.h5 on disk.

        Mutates ``self.videos`` and ``self.vid2metaid`` in place so the
        parent's ``__getitem__`` (which indexes ``self.videos[index]``)
        never sees a missing video.
        """
        if not hasattr(self, 'videos') or not hasattr(self, 'vid2metaid'):
            return
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
            print(f"[{type(self).__name__}] pre-filter: dropping "
                  f"{len(dropped)}/{len(self.videos)} videos with no event "
                  f"H5 on disk (e.g. {dropped[:3]})", flush=True)
        self.videos = kept_videos
        # vid2metaid: drop missing keys but keep the rest.
        self.vid2metaid = {k: v for k, v in self.vid2metaid.items()
                           if k in set(kept_videos)}

    def __getitem__(self, index):
        """Wrap the parent __getitem__ so a per-sample FileNotFoundError /
        OSError doesn't poison the worker. We just resample from the
        same dataset until we land on a readable clip.
        """
        max_retries = 8
        last_err = None
        for attempt in range(max_retries):
            try:
                return super().__getitem__(index)
            except (FileNotFoundError, OSError) as e:
                last_err = e
                # Try a different sample. real_len() exists on the parent
                # (VideoReVOSDataset) and accounts for the modulus the
                # base __getitem__ uses to map index -> video.
                try:
                    n = self.real_len()
                except Exception:
                    n = max(1, len(self.videos))
                index = random.randint(0, n - 1) if n > 0 else 0
                if attempt == 0:
                    print(f"[{type(self).__name__}] sample read failed "
                          f"({e!r}); resampling.", flush=True)
        raise RuntimeError(
            f"[{type(self).__name__}] gave up after {max_retries} retries; "
            f"last error: {last_err!r}")

    def _load_event_voxels(self, video_id: str, frame_indices, frames_list):
        import h5py
        path = self._event_path_for_video(video_id)
        # Single tiny peek to learn (t_min, t_max), shape, native_fps. No
        # x/y/t/p materialisation here.
        with h5py.File(path, 'r') as f:
            t_dset = f['t']
            n = int(t_dset.shape[0])
            H_attr = int(f.attrs.get('native_height', 0)) or None
            W_attr = int(f.attrs.get('native_width', 0)) or None
            native_fps = float(f.attrs.get('native_fps',
                                            self.fallback_native_fps
                                            or self.event_fps))
            t_unit = f.attrs.get('t_unit', b'us')
            if isinstance(t_unit, bytes):
                t_unit = t_unit.decode('utf-8', errors='replace')
            assert t_unit == 'us', f'expected t_unit=us, got {t_unit}'
            if n == 0:
                H = H_attr or 1
                W = W_attr or 1
                return np.zeros((len(frame_indices), 2, H, W),
                                dtype=np.float32)
            t_min = int(t_dset[0])
            t_max = int(t_dset[-1])

        # Compute desired centres in microseconds.
        centres = []
        for f_idx in frame_indices:
            centres.append(self._frame_index_to_us(
                f_idx, native_fps, frames_list, t_min, t_max))
        if not centres:
            H = H_attr or 1
            W = W_attr or 1
            return np.zeros((0, 2, H, W), dtype=np.float32)

        # Per-frame windowed read: O(events in 10ms) per frame regardless
        # of how spread out the centres are.
        return _accumulate_per_frame(
            path,
            centres_us=centres,
            half_us=self.event_window_us // 2,
            H_hint=H_attr, W_hint=W_attr,
        )


class VideoMeVISEventDatasetV2(_StreamingVideoEventBaseMixin,
                                VideoMeVISEventDataset):
    """MeViS event dataset with bounded per-sample memory."""
    pass


class VideoLVVISEventDatasetV2(_StreamingVideoEventBaseMixin,
                                VideoLVVISEventDataset):
    """LVVIS event dataset with bounded per-sample memory."""
    pass


# =============================================================================
# RefCOCO / RefCOCO+ / RefCOCOg: windowed single-image loads
# =============================================================================

class RefCOCOEventDatasetV2(RefCOCOEventDataset):
    """RefCOCO event dataset that reads the entire (short) clip in one pass
    via windowed I/O. COCO event clips are tiny (~ms of camera shake) so
    there's no virtual splitting to do, but we still avoid materialising
    arrays we don't need by reading attrs first.
    """

    def _load_image_event_voxel(self, image_filename: str):
        stem = os.path.splitext(os.path.basename(image_filename))[0]
        if stem not in self._available_clips:
            return None
        path = self._event_path(image_filename)
        # Read the full small clip, but via the windowed helper so we share
        # one code path. For COCO the full file is small (tens of KB).
        meta = _peek_h5_meta(path)
        if meta is None:
            return None
        if meta['n_events'] == 0:
            H = meta['H'] or 1
            W = meta['W'] or 1
            return np.zeros((1, 2, H, W), dtype=np.float32)
        x, y, t, p, H, W, _, _, _ = _load_event_window(
            path,
            t_start_us=int(meta['t_min_us']),
            t_end_us=int(meta['t_max_us']) + 1,
        )
        if len(t) == 0:
            return np.zeros((1, 2, H, W), dtype=np.float32)
        voxel = _accumulate_window(
            x, y, p, t,
            start_us=int(t[0]), end_us=int(t[-1]) + 1,
            H=H, W=W,
        )
        return voxel[None]  # (1, 2, H, W)


# =============================================================================
# ReVOS (UVO subset): windowed loads + clip-id remap
# =============================================================================
#
# Only the UVO portion of ReVOS has been converted to events (~78 clips, two
# upsampling variants). The ReVOS meta_expressions video keys are
# dataset-prefixed paths like ``UVO/all/<youtube_id>`` while the converted
# event dirs are named ``<youtube_id>.f<segment>``. We strip the ``.f<seg>``
# suffix to recover the youtube id and map it back to the on-disk dir; the
# streaming pre-filter then drops every non-UVO ReVOS video (no event H5).

class _UVOPathMixin:
    """Map a ReVOS ``UVO/all/<yt>`` video key onto the converted event dir
    ``<yt>.f<segment>`` by stripping the segment suffix. Builds the lookup
    lazily from ``os.listdir(event_folder)`` on first use so it works
    regardless of MRO __init__ ordering.
    """

    _UVO_SUFFIX_RE = re.compile(r'\.f\d+$')

    def _build_event_dir_map(self):
        m = {}
        if os.path.isdir(self.event_folder):
            for d in os.listdir(self.event_folder):
                if not os.path.isfile(
                        os.path.join(self.event_folder, d, self.EVENT_FILE_NAME)):
                    continue
                base = self._UVO_SUFFIX_RE.sub('', d)
                m.setdefault(base, d)
                m[d] = d  # also accept an exact dir name
        self._event_dir_map = m

    # Maps a ReVOS video-key prefix to the attribute holding that
    # sub-source's standalone event root. Each of these sub-sources' event
    # directory layout mirrors the annotation key exactly (``<root>/<rest
    # of the key after the prefix>/events.h5``), unlike UVO which needs the
    # ``.f<segment>`` suffix remap below.
    _PREFIX_ROOT_ATTRS = (
        ('LV-VIS/', 'lvvis_event_folder'),
        ('OVIS/', 'ovis_event_folder'),
        ('TAO/', 'tao_event_folder'),
        ('MOSE/', 'mose_event_folder'),
    )

    def _event_path_for_video(self, video_id: str) -> str:
        vid = str(video_id)
        # ReVOS LV-VIS/OVIS/TAO/MOSE subsets: their video keys are
        # ``<PREFIX>/<split>/[<subsource>/]<id>`` and use the EXACT same
        # split/id scheme as each sub-source's standalone event conversion,
        # so we serve those clips straight from that root. Frame->timestamp
        # mapping is fractional, identical to UVO, so no per-clip alignment
        # metadata is needed.
        for prefix, attr in self._PREFIX_ROOT_ATTRS:
            root = getattr(self, attr, None)
            if root and vid.startswith(prefix):
                parts = vid.split('/')
                if len(parts) >= 2:
                    return os.path.join(root, *parts[1:],
                                        self.EVENT_FILE_NAME)
        # UVO subset: ``UVO/all/<yt>`` -> converted dir ``<yt>.f<segment>``.
        if not hasattr(self, '_event_dir_map'):
            self._build_event_dir_map()
        key = vid.split('/')[-1]
        dirname = self._event_dir_map.get(key) or self._event_dir_map.get(vid)
        if dirname is None:
            # Non-converted ReVOS video (MOSE/TAO/OVIS, or an unconverted UVO)
            # -> point at a path that doesn't exist so the pre-filter drops it.
            return os.path.join(self.event_folder, vid, self.EVENT_FILE_NAME)
        return os.path.join(self.event_folder, dirname, self.EVENT_FILE_NAME)


class VideoReVOSEventDatasetV2(_UVOPathMixin,
                               _StreamingVideoEventBaseMixin,
                               VideoEventBaseDataset):
    """ReVOS (UVO subset) event dataset.

    Inherits the full MeViS/ReVOS event machinery (``VideoEventBaseDataset``
    is itself a ``VideoReVOSDataset``) plus the streaming windowed reader and
    missing-H5 pre-filter, and overlays the UVO clip-id remap.

    ``lvvis_event_folder`` / ``ovis_event_folder`` / ``tao_event_folder`` /
    ``mose_event_folder`` (all optional): roots of each sub-source's
    standalone event conversion. When given, ReVOS's ``<PREFIX>/<split>/...``
    videos for that sub-source are served from there instead of being
    dropped by the missing-H5 pre-filter. LV-VIS (234) + OVIS (68) + TAO
    (102) are fully converted already, lifting ReVOS coverage from ~95 UVO
    clips to ~499 without any new conversion; MOSE (127 total) is only
    partially converted (73 usable) as of 2026-08-09.
    """

    def __init__(self, *args, lvvis_event_folder: Optional[str] = None,
                 ovis_event_folder: Optional[str] = None,
                 tao_event_folder: Optional[str] = None,
                 mose_event_folder: Optional[str] = None,
                 **kwargs):
        # Must be set BEFORE super().__init__ so the missing-H5 pre-filter
        # (run at the end of the streaming mixin's __init__) can resolve and
        # keep these clips.
        self.lvvis_event_folder = lvvis_event_folder
        self.ovis_event_folder = ovis_event_folder
        self.tao_event_folder = tao_event_folder
        self.mose_event_folder = mose_event_folder
        super().__init__(*args, **kwargs)


# =============================================================================
# Ref-YouTube-VOS: windowed loads (MeViS-style schema with pickled masks)
# =============================================================================

class VideoRefYoutubeVOSEventDataset(VideoEventBaseDataset,
                                     VideoRefYoutubeVOSDataset):
    """Ref-YouTube-VOS with event voxels.

    MRO resolves ``json_file_preprocess`` to ``VideoRefYoutubeVOSDataset``
    (expression json + pickled ``mask_dict.pkl``) and the image-loading /
    ``__getitem__`` path to ``VideoEventBaseDataset`` (event voxels). Video
    ids are the YouTube-VOS hex hashes, which match the converted event dir
    names directly, so the default nested ``_event_path_for_video`` works.
    """
    pass


class VideoRefYoutubeVOSEventDatasetV2(_StreamingVideoEventBaseMixin,
                                       VideoRefYoutubeVOSEventDataset):
    """Ref-YouTube-VOS event dataset with bounded per-sample memory."""
    pass


# =============================================================================
# SAV / Ref-SAV (SA-V referential pseudo-labels): windowed event loads
# =============================================================================

class VideoSAM2EventDataset(VideoSAM2Dataset):
    """Ref-SAV referential segmentation on event voxels.

    The RGB parent decodes MP4 frames with OpenCV and reads per-clip masklet
    json files. We keep the text / object-sampling / masklet machinery and
    replace only the frame loading: events for ``<clip_id>`` live at
    ``<event_folder>/<clip_id>/events.h5`` (clip_id == Ref-SAV.json key ==
    ``sav_XXXXXX``). Frames are mapped to timestamps by evenly spreading the
    sampled masklet indices across the event clip's [t_min, t_max] span, which
    matches how the other event datasets sub-sample long clips.
    """

    EVENT_FILE_NAME = 'events.h5'

    def __init__(self, *args, event_folder: str, event_fps: float = 100.0,
                 event_normalize: bool = True, **kwargs):
        # The fast (50-frame) branch decodes the whole MP4; disable it for
        # events (we only ever read the sparse windows we need).
        kwargs['use_fast'] = False
        super().__init__(*args, **kwargs)
        self.event_folder = event_folder
        self.event_fps = float(event_fps)
        self.event_window_us = int(round(1e6 / self.event_fps))
        self.event_normalize = event_normalize
        self._event_transform = EventVoxelTransform(
            image_size=self.image_size, normalize=event_normalize)
        # Keep only Ref-SAV entries that have BOTH a converted event clip and a
        # masklet json on disk (the public Sa2VA release only ships masklets
        # for a subset of SA-V; the full set needs Meta's SA-V download).
        kept = []
        for v in self.video_ids:
            ev_ok = os.path.isfile(
                os.path.join(self.event_folder, v, self.EVENT_FILE_NAME))
            if not ev_ok:
                continue
            try:
                anno_ok = os.path.isfile(self._anno_full_path(v))
            except Exception:
                anno_ok = False
            if anno_ok:
                kept.append(v)
        print(f"[SAV-Event] {len(kept)} / {len(self.video_ids)} Ref-SAV "
              f"clips have both events ({self.event_folder}) and masklets "
              f"({self.sam2_folder})", flush=True)
        if not kept:
            raise RuntimeError(
                f"No SAV clips with both events and masklets found")
        self.video_ids = kept

    def _anno_full_path(self, video_id: str) -> str:
        return os.path.join(
            self.sam2_folder,
            self.remove_first_path_component(
                self.anno_dict[video_id]['anno_path']))

    def _load_event_voxels_by_fraction(self, video_id, frame_indices,
                                       n_total: int) -> np.ndarray:
        """One 2-ch polarity histogram per ``frame_indices`` entry, with each
        index mapped to a timestamp by its fractional position in
        ``[0, n_total)`` across the event clip's [t_min, t_max]."""
        path = os.path.join(self.event_folder, video_id, self.EVENT_FILE_NAME)
        import h5py
        with h5py.File(path, 'r') as f:
            t_dset = f['t']
            n = int(t_dset.shape[0])
            H = int(f.attrs.get('native_height', 0)) or None
            W = int(f.attrs.get('native_width', 0)) or None
            if n == 0:
                return np.zeros((len(frame_indices), 2, H or 1, W or 1),
                                dtype=np.float32)
            t_min = int(t_dset[0])
            t_max = int(t_dset[-1])
        centres = []
        denom = max(1, n_total - 1)
        for fi in frame_indices:
            frac = (fi / denom) if n_total > 1 else 0.0
            centres.append(int(round(t_min + frac * (t_max - t_min))))
        return _accumulate_per_frame(
            path, centres_us=centres, half_us=self.event_window_us // 2,
            H_hint=H, W_hint=W)

    def __getitem__(self, index):
        index = index % self.real_len()
        video_id = self.video_ids[index]
        expression_dict = self.anno_dict[video_id]
        object_ids = list(expression_dict['objects'].keys())

        anno_path = os.path.join(
            self.sam2_folder,
            self.remove_first_path_component(expression_dict['anno_path']))
        try:
            with open(anno_path, 'r') as f:
                mask_data = json.load(f)
            masklents = decode_masklet(mask_data['masklet'])
        except Exception as e:
            print(f"[SAV-Event] anno read err on {video_id}: {e!r}; resampling.",
                  flush=True)
            return self.__getitem__(random.randint(0, self.real_len() - 1))

        n_frames = len(masklents)
        n_objects = len(object_ids)
        if n_frames == 0 or n_objects == 0:
            return self.__getitem__(random.randint(0, self.real_len() - 1))

        # sample objects
        if n_objects > self.select_number:
            selected_indexes = np.random.choice(n_objects, self.select_number)
        else:
            selected_indexes = np.random.choice(n_objects, self.select_number,
                                                replace=True)
        selected_object_ids = [object_ids[i] for i in selected_indexes]
        objects_expression_infos = [expression_dict['objects'][i]
                                    for i in selected_object_ids]
        _masklents = []
        for _mask in masklents:
            _sel = [_mask[:, :, int(i)] for i in selected_object_ids]
            _masklents.append(np.stack(_sel, axis=2))
        masklents = _masklents

        # sample frames (uniform sparse + dense subset), mirroring the parent
        sparse_indices = get_sparse_indices_uniform(
            n_frames, max_frames=self.max_frames)
        self.sparse_len = len(sparse_indices)
        dense_indices = get_dense_indices(len(sparse_indices), self.dense_len)
        selected_sparse = [sparse_indices[i] for i in dense_indices]
        masklents = [masklents[i] for i in selected_sparse]

        data_dict = self.dataset_map_fn(
            objects_expression_infos, len(selected_sparse), n_fast_frames=0)
        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = video_lisa_encode_fn(
            data_dict, tokenizer=self.tokenizer,
            max_length=self.max_length, with_image_token=True)
        data_dict.update(result)

        voxels = self._load_event_voxels_by_fraction(
            video_id, sparse_indices, n_frames)  # (sparse_len, 2, H, W)
        pixel_values = self._event_transform(voxels)
        data_dict['pixel_values'] = pixel_values

        if getattr(self, 'extra_image_processor', None) is not None:
            extra_pixel_values = []
            for n_i in range(len(sparse_indices)):
                if n_i not in dense_indices:
                    continue
                ev_img = _voxel_to_pseudo_rgb(voxels[n_i])
                ev_img = self.extra_image_processor.apply_image(ev_img)
                extra_pixel_values.append(
                    torch.from_numpy(ev_img).permute(2, 0, 1).contiguous())
            data_dict['g_pixel_values'] = extra_pixel_values

        masklents = np.stack(masklents, axis=0)  # (n_dense, h, w, n_obj)
        masklents = torch.from_numpy(masklents).permute(3, 0, 1, 2)
        masklents = masklents.flatten(0, 1)
        data_dict['masks'] = masklents
        data_dict['dense_indices'] = dense_indices
        return data_dict


# =============================================================================
# GCG (Flickr / GranDf): single-image grounded-caption event loads
# =============================================================================

class _GCGEventMixin:
    """Replace the RGB image-loading block of a ``GCGDataset`` subclass with
    a single-frame event-voxel load. Each annotated image
    (``<stem>.jpg``) has an event clip at ``<event_folder>/<stem>/events.h5``
    synthesised from a short camera-shake of the still image; we integrate the
    whole short span into one 2-channel polarity histogram. Masks (RLE,
    inlined in the annotation) and the grounded-caption text are produced by
    the parent's ``dataset_map_fn`` / ``decode_mask`` unchanged.
    """

    def __init__(self, *args, event_folder: str,
                 event_normalize: bool = True, **kwargs):
        kwargs.setdefault('single_image_mode', True)
        super().__init__(*args, **kwargs)
        self.event_folder = event_folder
        self.event_normalize = event_normalize
        self._event_transform = EventVoxelTransform(
            image_size=self.image_size, normalize=event_normalize)
        self._available_clips = set(os.listdir(event_folder)) \
            if os.path.isdir(event_folder) else set()
        if not self._available_clips:
            raise RuntimeError(f"No event clips under {event_folder}")
        # Pre-filter to annotations whose converted clip exists.
        self._valid_indices = [
            i for i in range(len(self.text_data))
            if self._clip_stem(self.text_data[i]) in self._available_clips
        ]
        print(f"[{type(self).__name__}] {len(self._valid_indices)} / "
              f"{len(self.text_data)} annotations have an event clip in "
              f"{event_folder}", flush=True)
        if not self._valid_indices:
            raise RuntimeError(
                f"No usable GCG annotations under {event_folder}")

    @staticmethod
    def _clip_stem(entry) -> str:
        fname = entry.get('file_name') or entry.get('image') or ''
        return os.path.splitext(os.path.basename(fname))[0]

    def __len__(self):
        return len(self._valid_indices) * self.repeats

    def real_len(self):
        return len(self._valid_indices)

    @property
    def modality_length(self):
        return [100] * (len(self._valid_indices) * self.repeats)

    def _load_image_event_voxel(self, stem: str):
        if stem not in self._available_clips:
            return None
        path = os.path.join(self.event_folder, stem, 'events.h5')
        meta = _peek_h5_meta(path)
        if meta is None:
            return None
        if meta['n_events'] == 0:
            H = meta['H'] or 1
            W = meta['W'] or 1
            return np.zeros((1, 2, H, W), dtype=np.float32)
        x, y, t, p, H, W, _, _, _ = _load_event_window(
            path, t_start_us=int(meta['t_min_us']),
            t_end_us=int(meta['t_max_us']) + 1)
        if len(t) == 0:
            return np.zeros((1, 2, H, W), dtype=np.float32)
        voxel = _accumulate_window(
            x, y, p, t, start_us=int(t[0]), end_us=int(t[-1]) + 1, H=H, W=W)
        return voxel[None]  # (1, 2, H, W)

    def __getitem__(self, index):
        index = self._valid_indices[index % len(self._valid_indices)]
        data_dict = copy.deepcopy(self.text_data[index])
        ori_height = int(data_dict.get('height') or 0) or None
        ori_width = int(data_dict.get('width') or 0) or None

        result = self.dataset_map_fn(data_dict)
        data_dict.update(result)

        image_file = data_dict['image']
        stem = os.path.splitext(os.path.basename(image_file))[0]
        voxel = self._load_image_event_voxel(stem)
        if voxel is None:
            return self.__getitem__(random.randint(0, len(self._valid_indices) - 1))

        if ori_height is None or ori_width is None:
            ori_height, ori_width = int(voxel.shape[-2]), int(voxel.shape[-1])

        pixel_values = self._event_transform(voxel)  # (1, 2, S, S)
        data_dict['pixel_values'] = pixel_values

        if getattr(self, 'extra_image_processor', None) is not None:
            ev_img = _voxel_to_pseudo_rgb(voxel[0])
            ev_img = self.extra_image_processor.apply_image(ev_img)
            data_dict['g_pixel_values'] = \
                torch.from_numpy(ev_img).permute(2, 0, 1).contiguous()

        num_image_tokens = pixel_values.shape[0] * self.patch_token
        image_token_str = (f'{self.IMG_START_TOKEN}'
                           f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}'
                           f'{self.IMG_END_TOKEN}')
        num_context_token = pixel_values.shape[0]
        context_str = self.IMG_CONTEXT_TOKEN * num_context_token + '\n'

        for idx, item in enumerate(data_dict['conversation']):
            if idx == 0 and DEFAULT_IMAGE_TOKEN in item['input']:
                item['input'] = item['input'].replace(
                    DEFAULT_IMAGE_TOKEN, '').replace('\n', '').strip()
                item['input'] = image_token_str + '\n' + context_str + item['input']
            else:
                item['input'] = context_str + item['input']

        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = video_lisa_encode_fn(
            data_dict, tokenizer=self.tokenizer,
            max_length=self.max_length, with_image_token=True)
        data_dict.update(result)

        data_dict['masks'] = self.decode_mask(
            data_dict['masks'], ori_height=ori_height, ori_width=ori_width)
        if data_dict['masks'] is None:
            return self.__getitem__(random.randint(0, len(self._valid_indices) - 1))
        return data_dict


class FlickrGCGEventDataset(_GCGEventMixin, FlickrGCGDataset):
    """Flickr30K-entities grounded-caption RES on COCO-image events."""
    pass


class GranDfGCGEventDataset(_GCGEventMixin, GranDfGCGDataset):
    """GranDf grounded-caption RES on SA-1B-image events."""
    pass


class OpenPsgGCGEventDataset(_GCGEventMixin, OpenPsgGCGDataset):
    """OpenPSG grounded-caption RES on COCO2017-image events.

    Same single-image event-voxel path as the other GCG sets; the parent
    ``OpenPsgGCGDataset`` supplies ``glamm_openpsg_map_fn`` (caption +
    ``groundings`` with inline RLE masks) and GranDf-style ``decode_mask``.
    ``file_name`` is ``train2017/<id>.jpg`` so ``_clip_stem`` resolves to the
    ``coco/train2017/<id>`` event clip.
    """
    pass


class RefCOCOgGCGEventDataset(_GCGEventMixin, RefCOCOgGCGDataset):
    """RefCOCOg grounded-caption RES on COCO2014-image events.

    Unlike the other GCG sets, RefCOCOg-GCG raw entries key the image as
    ``img_file_name`` (not ``file_name``/``image``) and carry no
    ``height``/``width``; their masks are COCO polygons in native image
    pixels. We therefore:

    * resolve the event clip stem from ``img_file_name`` (overriding
      ``_clip_stem`` so the existence pre-filter works), and
    * inject native COCO dims from ``instances_json`` into each raw entry so
      ``decode_mask`` rasterises the polygons at the correct resolution.

    Events are still resized to 1024 independently of the mask resolution,
    exactly as in the RefCOCO RES path.
    """

    def __init__(self, *args, instances_json: str, **kwargs):
        super().__init__(*args, **kwargs)
        self._inject_native_dims(instances_json)

    @staticmethod
    def _clip_stem(entry) -> str:
        fname = (entry.get('img_file_name') or entry.get('file_name')
                 or entry.get('image') or '')
        return os.path.splitext(os.path.basename(fname))[0]

    def _inject_native_dims(self, instances_json: str):
        with open(instances_json) as f:
            inst = json.load(f)
        fn2hw = {im['file_name']: (int(im['height']), int(im['width']))
                 for im in inst['images']}
        matched = 0
        for entry in self.text_data:
            fname = entry.get('img_file_name') or entry.get('file_name') or ''
            hw = fn2hw.get(os.path.basename(fname))
            if hw is None:
                continue
            entry['height'], entry['width'] = hw
            matched += 1
        print(f"[{type(self).__name__}] injected native COCO dims for "
              f"{matched}/{len(self.text_data)} entries from "
              f"{os.path.basename(instances_json)}", flush=True)


# =============================================================================
# LLaVA image QA (no masks): single-image event loads
# =============================================================================

class LLaVAEventDataset(LLaVADataset):
    """LLaVA-1.5 instruction QA on image events.

    ``event_roots`` maps each LLaVA image-path prefix to the converted event
    root, e.g.::

        event_roots = {
            'gqa':            '<llava>/gqa',
            'ocr_vqa':        '<llava>/ocr_vqa',
            'coco/train2017': '<coco>/train2017',
            'coco/train2014': '<coco>/train2014',
        }

    Entries whose image prefix is not in ``event_roots`` (e.g. textvqa/vg,
    which were not converted) or whose specific clip is missing are dropped at
    init. Pure-text entries are also dropped (there's nothing to encode as an
    event frame). No masks are produced.
    """

    def __init__(self, *args, event_roots: dict,
                 event_normalize: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.event_roots = dict(event_roots)
        self.event_normalize = event_normalize
        self._event_transform = EventVoxelTransform(
            image_size=self.image_size, normalize=event_normalize)
        self.transformer = None
        self.preprocessor = None
        self._avail = {k: (set(os.listdir(v)) if os.path.isdir(v) else set())
                       for k, v in self.event_roots.items()}
        kept = [d for d in self.data
                if d.get('image') is not None
                and self._resolve_event_path(d['image']) is not None]
        print(f"[LLaVA-Event] kept {len(kept)} / {len(self.data)} entries "
              f"with a converted event clip", flush=True)
        if not kept:
            raise RuntimeError("No LLaVA event clips resolved; check event_roots")
        self.data = kept

    def _resolve_event_path(self, image_rel: str):
        parts = image_rel.split('/')
        if len(parts) < 2:
            return None
        stem = os.path.splitext(parts[-1])[0]
        p0 = parts[0]
        if p0 == 'coco' and len(parts) >= 3:
            key = 'coco/' + parts[1]
        else:
            key = p0
        root = self.event_roots.get(key)
        avail = self._avail.get(key)
        if root is None or avail is None or stem not in avail:
            return None
        return os.path.join(root, stem, 'events.h5')

    def _load_image_event_voxel(self, path: str):
        meta = _peek_h5_meta(path)
        if meta is None:
            return None
        if meta['n_events'] == 0:
            H = meta['H'] or 1
            W = meta['W'] or 1
            return np.zeros((1, 2, H, W), dtype=np.float32)
        x, y, t, p, H, W, _, _, _ = _load_event_window(
            path, t_start_us=int(meta['t_min_us']),
            t_end_us=int(meta['t_max_us']) + 1)
        if len(t) == 0:
            return np.zeros((1, 2, H, W), dtype=np.float32)
        voxel = _accumulate_window(
            x, y, p, t, start_us=int(t[0]), end_us=int(t[-1]) + 1, H=H, W=W)
        return voxel[None]  # (1, 2, H, W)

    def prepare_data(self, index):
        data_dict = self.data[index]
        if data_dict is None:
            return None
        image_rel = data_dict.get('image')
        if image_rel is None:
            return None
        path = self._resolve_event_path(image_rel)
        if path is None:
            return None
        try:
            voxel = self._load_image_event_voxel(path)
        except Exception as e:
            print(f"[LLaVA-Event] read err {path}: {e!r}", flush=True)
            return None
        if voxel is None:
            return None

        out_data_dict = {}
        pixel_values = self._event_transform(voxel)  # (1, 2, S, S)
        out_data_dict['pixel_values'] = pixel_values
        num_image_tokens = pixel_values.shape[0] * self.patch_token
        image_token_str = (f'{self.IMG_START_TOKEN}'
                           f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}'
                           f'{self.IMG_END_TOKEN}')
        num_context_token = pixel_values.shape[0]
        context_str = self.IMG_CONTEXT_TOKEN * num_context_token + '\n'
        token_dict = self.get_inputid_labels(
            data_dict['conversations'], image_token_str,
            context_token_str=context_str)
        out_data_dict.update(token_dict)
        return out_data_dict


# =============================================================================
# SAMA-239K (VideoRegionDescription / VideoRegionConv) event wrappers
# =============================================================================
#
# SAMA-239K contains 15,151 videos from 5 sources: LVViS, MeViS,
# Ref-YouTube-VOS, SA-V (mp4), and VidSTG.  Events are available for
# LVViS / MeViS / RefYT-VOS / VidSTG; SA-V is complex (mp4-dependent) and
# already covered by VideoSAM2EventDataset.  We therefore wrap all four
# event-available sources (12,284 / 15,151 ≈ 81% of SAMA-239K videos).
#
# Video path formats in sama239k_train_final.json:
#   lvvis/train/JPEGImages/<clip_id>/         → <lvvis_root>/<clip_id>/events.h5
#   mevis/train/JPEGImages/<clip_id>/         → <mevis_root>/<clip_id>/events.h5
#   ref_youtube_vos/train/JPEGImages/<clip_id>/ → <refytvos_root>/<clip_id>/events.h5
#   VidSTG/train/<video_id>                   → <vidstg_root>/train/<video_id>/events.h5
#   sav_train/sav_XXX/sav_XXXXXX.mp4         → skipped (mp4, handled via Ref-SAV)

from projects.llava_sam2.datasets.VideoRegionCaptioning_Dataset import (  # noqa: E402
    VideoRegionDescriptionDataset,
)
from projects.llava_sam2.datasets.VideoRegionConv_Dataset import (  # noqa: E402
    VideoRegionConvDataset,
)

# All SAMA-239K sources with events are now handled (SAV via the
# events-only path in _prepare_sav_item). Nothing is skipped by default.
_SAMA239K_SKIP_SOURCES = ()


# ---------------------------------------------------------------------------
# Process-level mask-dict cache.
#
# The SAMA-239K mask dicts are HUGE (VidSTG alone is 5.3 GB on disk, ~7.1 GB
# total). VideoRegionDescription and VideoRegionConv each load all five in
# their base __init__, so a naive setup holds TWO copies per rank process.
# Across 4 DDP ranks that doubled cost is what blew past --mem and OOM-killed
# the run at iter ~110.
#
# We dedupe by file path: the first dataset to load a given json/pkl caches
# the resulting object; the second dataset gets the SAME object back, so both
# datasets in a rank share one copy. We patch json.load / pickle.load only for
# the duration of the base __init__ (keyed on the file object's .name).
# ---------------------------------------------------------------------------
_MASKDICT_CACHE: Dict[str, object] = {}
_ORIG_JSON_LOAD = json.load
import pickle as _pickle  # noqa: E402
_ORIG_PICKLE_LOAD = _pickle.load


# Only the giant SAMA-239K mask dicts should be cached. Their filenames all
# contain "mask_dict" (e.g. mevis_train_mask_dict.json, SAV_mask_dict_train.json,
# ref_youtube_train_mask_dict.pkl). We must NOT cache anything else: transformers
# loads the tokenizer's config.json via json.load and mutates the returned dict
# (popping model_type/auto_map), so caching/returning a shared object there
# corrupts the second tokenizer build -> "Unrecognized configuration class".
def _is_maskdict_path(path) -> bool:
    return bool(path) and 'mask_dict' in path


def _cached_json_load(fp, *a, **k):
    path = getattr(fp, 'name', None)
    if not _is_maskdict_path(path):
        return _ORIG_JSON_LOAD(fp, *a, **k)
    if path in _MASKDICT_CACHE:
        return _MASKDICT_CACHE[path]
    obj = _ORIG_JSON_LOAD(fp, *a, **k)
    _MASKDICT_CACHE[path] = obj
    return obj


def _cached_pickle_load(fp, *a, **k):
    path = getattr(fp, 'name', None)
    if not _is_maskdict_path(path):
        return _ORIG_PICKLE_LOAD(fp, *a, **k)
    if path in _MASKDICT_CACHE:
        return _MASKDICT_CACHE[path]
    obj = _ORIG_PICKLE_LOAD(fp, *a, **k)
    _MASKDICT_CACHE[path] = obj
    return obj


class _SAMA239KEventMixin:
    """Mixin that replaces RGB pixel loading with event voxels for the
    SAMA-239K VideoRegionDescription / VideoRegionConv datasets.

    Required constructor kwargs (in addition to the base class kwargs):
        event_roots : dict  – keys 'lvvis', 'mevis', 'refytvos', 'vidstg'
                              pointing to the respective event root dirs.
                              For 'vidstg' the root is the dir that contains
                              the ``train/<video_id>/events.h5`` tree.
        event_window_us : int  – integration window per frame (default 10 ms).
    """

    EVENT_FILE_NAME = 'events.h5'

    def __init__(self, *args, event_roots=None, event_window_us: int = 10_000,
                 **kwargs):
        # Always exclude sources without events so the base class never tries
        # to open a non-existent JPEG or an mp4 for SAV.
        existing_skip = kwargs.get('selected_datasets') or []
        all_skip = list(existing_skip) + [
            s for s in _SAMA239K_SKIP_SOURCES if s not in existing_skip
        ]
        kwargs['selected_datasets'] = all_skip
        # Dedupe the huge mask dicts across the two SAMA-239K datasets within
        # this process: patch json/pickle load to return cached objects keyed
        # by file path, for the duration of the base __init__ only.
        json.load = _cached_json_load
        _pickle.load = _cached_pickle_load
        try:
            super().__init__(*args, **kwargs)
        finally:
            json.load = _ORIG_JSON_LOAD
            _pickle.load = _ORIG_PICKLE_LOAD
        self.event_roots = event_roots or {}
        self.event_window_us = event_window_us
        self._prefilter_event_items()

    # ------------------------------------------------------------------
    # Path resolution
    # ------------------------------------------------------------------

    def _event_h5_path(self, file_name: str) -> Optional[str]:
        """Map a SAMA-239K video path to the corresponding events.h5 path."""
        # file_name e.g.:  "lvvis/train/JPEGImages/00000"
        #                   "mevis/train/JPEGImages/6084240e75fa"
        #                   "ref_youtube_vos/train/JPEGImages/003234408d"
        #                   "VidSTG/train/10008553263"
        clip_id = os.path.splitext(os.path.basename(file_name.rstrip('/')))[0]
        if 'lvvis' in file_name:
            root = self.event_roots.get('lvvis', '')
            return os.path.join(root, clip_id, self.EVENT_FILE_NAME)
        elif 'mevis' in file_name:
            root = self.event_roots.get('mevis', '')
            return os.path.join(root, clip_id, self.EVENT_FILE_NAME)
        elif 'ref_youtube_vos' in file_name:
            root = self.event_roots.get('refytvos', '')
            return os.path.join(root, clip_id, self.EVENT_FILE_NAME)
        elif 'VidSTG' in file_name:
            # events live under <vidstg_root>/train/<video_id>/events.h5
            root = self.event_roots.get('vidstg', '')
            return os.path.join(root, 'train', clip_id, self.EVENT_FILE_NAME)
        elif 'sav' in file_name:
            # SAMA-239K SAV path "sav_train/sav_000/sav_000005.mp4" (or the
            # rewritten "sav/sav_000/sav_000005") → <sav_root>/<clip_id>/events.h5
            # where clip_id == "sav_000005" (flat, no shard subdir).
            root = self.event_roots.get('sav', '')
            return os.path.join(root, clip_id, self.EVENT_FILE_NAME)
        return None

    # ------------------------------------------------------------------
    # SAV (mp4-less) preparation
    # ------------------------------------------------------------------

    def _prepare_sav_item(self, data_dict) -> bool:
        """In-place: make a SAMA-239K SAV (mp4) entry loadable events-only.

        SAV entries carry ``frames=None`` and ``height/width=None`` because
        the base dataset reads the .mp4 for frames. The .mp4s are not on
        disk in the event pipeline, so we instead derive the (decimated)
        frame count and image size from the SAV mask dict, inject synthetic
        ``frames``, and rewrite ``file_name`` so the base ``dataset_map_fn``
        takes its non-mp4 branch while still selecting the 'sav' mask dict.

        Returns True if the item was prepared (is a SAV item), else False.
        """
        fn = data_dict.get('file_name') or data_dict.get('video') or ''
        if 'sav_train' not in fn:
            return False
        anno_map = (data_dict.get('annotation')
                    or data_dict.get('anno_map') or [])
        sav_md = self.mask_dicts.get('sav', {})
        lengths, hw = [], None
        for k in anno_map:
            if k == -1:
                continue
            ann = sav_md.get(str(k))
            if not ann:
                continue
            lengths.append(len(ann))
            if hw is None and isinstance(ann[0], dict) and 'size' in ann[0]:
                hw = ann[0]['size']
        if not lengths or hw is None:
            return False
        n_frames = min(lengths)
        data_dict['frames'] = [str(i) for i in range(n_frames)]
        data_dict['height'] = int(hw[0])
        data_dict['width'] = int(hw[1])
        # Drop the 'sav_train' token (so the base mp4 branch isn't taken)
        # and the .mp4 suffix, but keep a 'sav' token for mask selection.
        data_dict['file_name'] = (
            fn.replace('sav_train/', 'sav/').replace('.mp4', ''))
        return True

    # ------------------------------------------------------------------
    # Height/width backfill (frames-less sources)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_mask_size(ann):
        """Return [h, w] from the first per-frame RLE that carries a size."""
        if not isinstance(ann, (list, tuple)):
            return None
        for fr in ann:
            if isinstance(fr, dict) and 'size' in fr and fr['size']:
                return fr['size']
        return None

    def _mask_dict_for(self, file_name: str):
        if 'mevis' in file_name:
            return self.mask_dicts.get('mevis', {})
        if 'lvvis' in file_name:
            return self.mask_dicts.get('lvvis', {})
        if 'ref_youtube_vos' in file_name:
            return self.mask_dicts.get('ref_youtube', {})
        if 'VidSTG' in file_name:
            return self.mask_dicts.get('vidstg', {})
        if 'sav' in file_name:
            return self.mask_dicts.get('sav', {})
        return {}

    def _ensure_hw(self, data_dict):
        """Backfill height/width from the mask dict when the JSON entry has
        them as None (Ref-YT-VOS etc.). The base dataset would otherwise
        open an RGB JPEG to read dimensions, which don't exist in the event
        pipeline. The mask RLE 'size' is also the exact size ``decode_mask``
        needs, so this is the correct source.
        """
        if data_dict.get('height') and data_dict.get('width'):
            return
        fn = data_dict.get('file_name') or data_dict.get('video') or ''
        anno_map = (data_dict.get('annotation')
                    or data_dict.get('anno_map') or [])
        md = self._mask_dict_for(fn)
        for k in anno_map:
            if k == -1:
                continue
            size = self._extract_mask_size(md.get(str(k)))
            if size:
                data_dict['height'] = int(size[0])
                data_dict['width'] = int(size[1])
                return

    # ------------------------------------------------------------------
    # Pre-filter: keep only items for which an events.h5 exists on disk
    # ------------------------------------------------------------------

    def _prefilter_event_items(self):
        kept, dropped = [], 0
        for item in self.text_data:
            fn = item.get('file_name', '')
            h5 = self._event_h5_path(fn)
            if h5 and os.path.isfile(h5):
                kept.append(item)
            else:
                dropped += 1
        print(f"[{type(self).__name__}] event pre-filter: "
              f"kept {len(kept)} / {len(kept) + dropped} SAMA-239K items",
              flush=True)
        self.text_data = kept

    # ------------------------------------------------------------------
    # Voxel loading  (fraction-based timestamp estimation)
    # ------------------------------------------------------------------

    def _load_voxels_for_images(
        self, h5_path: str, images: list, frames_json: list,
    ) -> np.ndarray:
        """Load one 2-ch voxel per entry in *images*.

        For str images the frame index is encoded in the filename;
        for numpy images (SAV mp4 branch, should never reach here after
        pre-filter) we fall back to uniform spacing.
        """
        import h5py
        with h5py.File(h5_path, 'r') as f:
            t_dset = f['t']
            n = int(t_dset.shape[0])
            H = int(f.attrs.get('native_height', 0)) or None
            W = int(f.attrs.get('native_width', 0)) or None
            if n == 0:
                N = len(images)
                return np.zeros((N, 2, H or 1, W or 1), dtype=np.float32)
            t_min = int(t_dset[0])
            t_max = int(t_dset[-1])

        # Derive frame_indices and the fraction denominator from the image
        # filenames. Frame stems are either plain ints ("00001") for
        # LVViS/MeViS/RefYT or "frame_<n>" for VidSTG; we extract the
        # trailing integer in both cases.
        def _parse_idx(stem: str) -> int:
            m = re.search(r'(\d+)\s*$', stem)
            return int(m.group(1)) if m else 0

        if images and isinstance(images[0], str):
            frame_indices = [
                _parse_idx(os.path.splitext(os.path.basename(img))[0])
                for img in images
            ]
            # denom = max frame index in the *full* frames list so the
            # fraction reflects each frame's position across the whole clip.
            if frames_json:
                all_idx = [
                    _parse_idx(
                        os.path.splitext(os.path.basename(
                            fj['raw_filename'] if isinstance(fj, dict)
                            and 'raw_filename' in fj else str(fj)))[0])
                    for fj in frames_json
                ]
                denom = max(max(all_idx), max(frame_indices), 1)
            else:
                denom = max(max(frame_indices), 1)
        else:
            # Numpy arrays – uniform spacing fallback
            N = len(images)
            frame_indices = list(range(N))
            denom = max(N - 1, 1)

        centres = [
            int(round(t_min + (fi / denom) * (t_max - t_min)))
            for fi in frame_indices
        ]
        return _accumulate_per_frame(
            h5_path, centres_us=centres,
            half_us=self.event_window_us // 2,
            H_hint=H, W_hint=W,
        )

    def _frames_to_pixel_values(self, voxels: np.ndarray) -> torch.Tensor:
        """(N, 2, H, W) voxels → (N, 3, img_size, img_size) normalised tensor."""
        pv = []
        for i in range(voxels.shape[0]):
            rgb = _voxel_to_pseudo_rgb(voxels[i])          # (H, W, 3) uint8
            pil = Image.fromarray(rgb)
            pv.append(self.transformer(pil))                # (3, img_size, img_size)
        return torch.stack(pv, dim=0)


class VideoRegionDescriptionEventDataset(
        _SAMA239KEventMixin, VideoRegionDescriptionDataset):
    """VideoRegionDescriptionDataset with event voxels instead of RGB frames.

    Extra constructor kwargs forwarded to _SAMA239KEventMixin:
        event_roots      : dict with 'lvvis', 'mevis', 'refytvos' keys
        event_window_us  : int (default 10 ms)
    """

    def dataset_map_fn(self, data_dict, select_k=5):
        # Make SAV (mp4) entries loadable events-only before the base does
        # its mp4 read, and backfill missing dims so no JPEG is opened.
        self._prepare_sav_item(data_dict)
        self._ensure_hw(data_dict)
        return super().dataset_map_fn(data_dict, select_k=select_k)

    def __getitem__(self, index):
        index = index % self.real_len()
        data_dict = copy.deepcopy(self.text_data[index])

        result = self.dataset_map_fn(data_dict, select_k=self.dense_len)
        if result is None or result.get('prompt_masks') is None:
            return self.__getitem__(random.randint(0, self.real_len() - 1))

        file_name = data_dict['file_name']
        h5_path = self._event_h5_path(file_name)
        if not h5_path or not os.path.isfile(h5_path):
            return self.__getitem__(random.randint(0, self.real_len() - 1))

        try:
            voxels = self._load_voxels_for_images(
                h5_path, result['images'], data_dict.get('frames'))
            pixel_values = self._frames_to_pixel_values(voxels)
        except (OSError, Exception) as e:
            print(f"[VideoRegionDescriptionEventDataset] voxel load failed "
                  f"for {file_name}: {e!r}; resampling", flush=True)
            return self.__getitem__(random.randint(0, self.real_len() - 1))

        data_dict = result
        data_dict['pixel_values'] = pixel_values

        result2 = self.template_map_fn(data_dict)
        data_dict.update(result2)
        result3 = video_lisa_encode_fn(
            data_dict, tokenizer=self.tokenizer,
            max_length=self.max_length, with_image_token=True)
        data_dict.update(result3)

        if data_dict.get('prompt_masks') is None:
            return self.__getitem__(random.randint(0, self.real_len() - 1))
        return data_dict


class VideoRegionConvEventDataset(
        _SAMA239KEventMixin, VideoRegionConvDataset):
    """VideoRegionConvDataset with event voxels instead of RGB frames.

    Handles both ``pixel_values`` (InternVL backbone) and ``g_pixel_values``
    (SAM2 high-res grounding) the same way as existing event datasets.
    """

    def dataset_map_fn(self, data_dict):
        self._prepare_sav_item(data_dict)
        self._ensure_hw(data_dict)
        return super().dataset_map_fn(data_dict)

    def __getitem__(self, index):
        index = index % self.real_len()
        data_dict = copy.deepcopy(self.text_data[index])
        data_dict = self.dataset_map_fn(data_dict)

        assert 'images' in data_dict

        file_name = data_dict.get('file_name', '')
        h5_path = self._event_h5_path(file_name)
        if not h5_path or not os.path.isfile(h5_path):
            return self.__getitem__(random.randint(0, self.real_len() - 1))

        try:
            voxels = self._load_voxels_for_images(
                h5_path, data_dict['images'],
                data_dict.get('frames'))
        except (OSError, Exception) as e:
            print(f"[VideoRegionConvEventDataset] voxel load failed "
                  f"for {file_name}: {e!r}; resampling", flush=True)
            return self.__getitem__(random.randint(0, self.real_len() - 1))

        pixel_values = self._frames_to_pixel_values(voxels)
        data_dict['pixel_values'] = pixel_values

        # SAM2 high-res grounding input (dense_indices frames only)
        if getattr(self, 'extra_image_processor', None) is not None:
            dense_indices = data_dict.get('dense_indices', [])
            extra_pv = []
            for n_i, vox in enumerate(voxels):
                if n_i not in dense_indices:
                    continue
                ev_img = _voxel_to_pseudo_rgb(vox)
                ev_img = self.extra_image_processor.apply_image(ev_img)
                extra_pv.append(
                    torch.from_numpy(ev_img).permute(2, 0, 1).contiguous())
            data_dict['g_pixel_values'] = extra_pv

        gt_masks = data_dict.get('gt_masks', [])
        if not gt_masks or any(m is None for m in gt_masks):
            return self.__getitem__(random.randint(0, self.real_len() - 1))
        try:
            data_dict['masks'] = torch.stack(gt_masks, dim=0).flatten(0, 1)
        except TypeError:
            return self.__getitem__(random.randint(0, self.real_len() - 1))

        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        result = video_lisa_encode_fn(
            data_dict, tokenizer=self.tokenizer, max_length=self.max_length)
        data_dict.update(result)
        return data_dict
