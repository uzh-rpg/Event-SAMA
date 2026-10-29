"""Event-aware referring-VOS evaluation dataset.

Iterates one item per (video, expression) over a MeViS-style val split, and
returns the model-ready tensors:

  * ``pixel_values``     -- (n_llm_frames, 2, image_size, image_size) float32
                             tensor; what the InternViT consumes. Capped to
                             ``llm_frames`` (matches the training-time SAMA
                             video pipeline which only feeds the first 5
                             frames to the LLM).
  * ``g_pixel_values``   -- (n_dense_frames, 3, 1024, 1024) float tensor
                             pre-processed by SAM2's ``preprocess_image``;
                             these drive the SAM2 grounding encoder. We pick
                             ``dense_frames`` evenly-spaced frames inside
                             ``[0, T)`` and render them as 3-channel
                             pseudo-RGB images of the polarity histograms.
                             SAM2 then produces one mask per dense frame.
  * ``text_prompt``      -- the raw expression, no image-token wrapping
                             (the eval entry script wraps it).

Bookkeeping the entry script needs to save masks back at original resolution:
  * ``video_id``, ``exp_id``
  * ``frames``      -- the names of the SAMPLED frames (one per pixel_values
                        / g_pixel_values entry; at most ``dense_frames`` long)
  * ``all_frames``  -- the full list of frame names from meta_expressions
                        (used to interpolate predictions back to per-frame
                        granularity for the J&F metric)
  * ``length``      -- len(all_frames)
  * ``ori_height``, ``ori_width``  -- size of the event canvas (matches the
                        size of the GT masks because the converter shipped
                        the prophesee output at the same resolution as the
                        annotated MeViS frames)
"""
from __future__ import annotations

import json
import os
import sys
from typing import List

import numpy as np
import torch
import torch.utils.data

# We need _accumulate_per_frame from our streaming V2 reader.
# Path is injected by the launcher (event_v2_dir).
from event_voxel_dataset_v2 import _accumulate_per_frame

# We need EventVoxelTransform + _voxel_to_pseudo_rgb from the in-tree
# training datasets module. The launcher cd's into the project dir so this
# import resolves.
from projects.llava_sam2.datasets.EventVoxel_Dataset import (
    EventVoxelTransform,
    _voxel_to_pseudo_rgb,
)


def _safe_h5_meta(path: str):
    """Read just the (n_events, t_min, t_max, native_fps, H, W) tuple.
    Returns None if the file is missing or unreadable."""
    import h5py
    try:
        with h5py.File(path, "r") as f:
            n = int(f["t"].shape[0])
            H = int(f.attrs.get("native_height", 0)) or None
            W = int(f.attrs.get("native_width", 0)) or None
            native_fps = float(f.attrs.get("native_fps", 0.0)) or None
            t_min = int(f["t"][0]) if n > 0 else 0
            t_max = int(f["t"][-1]) if n > 0 else 0
            return n, t_min, t_max, native_fps, H, W
    except (FileNotFoundError, OSError):
        return None


class RefVOSEventEvalDataset(torch.utils.data.Dataset):
    """Memory-bounded event-VOS eval dataset (MeViS / Ref-Y-VOS schema)."""

    def __init__(
        self,
        event_folder: str,
        expression_file: str,
        image_size: int = 448,
        event_fps: float = 100.0,
        event_normalize: bool = True,
        llm_frames: int = 5,
        dense_frames: int = 32,
        sam2_target_length: int = 1024,
        fallback_native_fps: float = 24.0,
        path_resolver=None,
    ):
        super().__init__()
        self.event_folder = event_folder
        # Optional override for video_id -> events.h5 path resolution (e.g.
        # ReVOS's LV-VIS/OVIS/TAO/MOSE prefix remap + UVO suffix remap).
        # Defaults to the plain <event_folder>/<vid_id>/events.h5 layout.
        self._path_resolver = path_resolver
        self.image_size = int(image_size)
        self.event_fps = float(event_fps)
        self.event_window_us = int(round(1e6 / self.event_fps))
        self.event_normalize = bool(event_normalize)
        self.llm_frames = int(llm_frames)
        self.dense_frames = int(dense_frames)
        self.sam2_target_length = int(sam2_target_length)
        self.fallback_native_fps = float(fallback_native_fps)
        self.transformer = EventVoxelTransform(
            image_size=self.image_size, normalize=self.event_normalize
        )

        with open(expression_file, "r") as fh:
            self.expressions = json.load(fh)["videos"]

        # Flatten to (vid_id, exp_id) pairs and pre-filter unavailable H5s
        # so a missing file doesn't poison the eval mid-run.
        self.items: List = []
        skipped_no_h5 = 0
        skipped_empty = 0
        for vid_id, vid_data in sorted(self.expressions.items()):
            if self._path_resolver is not None:
                ev_path = self._path_resolver(vid_id)
            else:
                ev_path = os.path.join(self.event_folder, vid_id, "events.h5")
            meta = _safe_h5_meta(ev_path)
            if meta is None:
                skipped_no_h5 += 1
                continue
            n_ev, t_min, t_max, native_fps, H, W = meta
            if n_ev == 0 or H is None or W is None:
                skipped_empty += 1
                continue
            # Cache the meta on the item so __getitem__ doesn't need to
            # re-open the H5 just for shape/fps.
            for exp_id in sorted(vid_data["expressions"].keys()):
                self.items.append(
                    (vid_id, exp_id, ev_path, t_min, t_max,
                     native_fps or self.fallback_native_fps, H, W)
                )
        print(
            f"[RefVOSEventEval] {len(self.items)} (vid, exp) items from "
            f"{len(self.expressions)} videos in {expression_file} "
            f"(skipped: no_h5={skipped_no_h5}, empty={skipped_empty})",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.items)

    def _frame_index_to_us(self, frame_idx, native_fps, frames_list,
                           t_min, t_max):
        """Same convention as VideoEventBaseDataset: prefer absolute frame
        numbering (numeric stem) when in range, else relative."""
        candidates = []
        try:
            num = int(str(frames_list[frame_idx]))
            candidates.append(int(round(num * 1e6 / native_fps)))
        except Exception:
            pass
        candidates.append(int(round(frame_idx * 1e6 / native_fps)))
        for c in candidates:
            if t_min <= c <= t_max:
                return c
        return candidates[-1]

    def __getitem__(self, index):
        vid_id, exp_id, ev_path, t_min, t_max, native_fps, H, W = \
            self.items[index]
        vid_data = self.expressions[vid_id]
        all_frames = sorted(vid_data["frames"])
        text = vid_data["expressions"][exp_id]["exp"]

        n_total = len(all_frames)
        # Choose dense_frames evenly-spaced frame indices for SAM2.
        n_dense = min(self.dense_frames, n_total)
        if n_total <= 1:
            dense_idx = [0] * n_dense
        else:
            dense_idx = list(
                np.linspace(0, n_total - 1, n_dense).astype(int)
            )
        # The first llm_frames of the dense schedule go to the LLM.
        llm_idx = dense_idx[: self.llm_frames]

        # Convert frame indices -> microsecond centres and load voxels.
        all_idx = sorted(set(dense_idx))
        idx_to_pos = {i: p for p, i in enumerate(all_idx)}
        centres_us = [
            self._frame_index_to_us(i, native_fps, all_frames, t_min, t_max)
            for i in all_idx
        ]
        voxels = _accumulate_per_frame(
            ev_path,
            centres_us=centres_us,
            half_us=self.event_window_us // 2,
            H_hint=H,
            W_hint=W,
        )  # (n_unique, 2, H, W) float32

        # Index back into the dense / llm schedules.
        dense_voxels = np.stack(
            [voxels[idx_to_pos[i]] for i in dense_idx], axis=0
        )
        llm_voxels = np.stack(
            [voxels[idx_to_pos[i]] for i in llm_idx], axis=0
        )

        # ViT-side: 2-channel resize+asinh, dtype float32.
        pixel_values = self.transformer(llm_voxels)  # (llm_F, 2, S, S)

        # SAM2-side: render each dense frame as pseudo-RGB uint8 then resize
        # to the target shorter side via direct resize, then permute to CHW.
        # We do NOT call self.grounding_encoder.preprocess_image here: that's
        # done in the eval loop (it uses the model's actual mean/std).
        g_pixel_values_chw = []
        for n_i in range(dense_voxels.shape[0]):
            ev_img = _voxel_to_pseudo_rgb(dense_voxels[n_i])  # (H, W, 3) u8
            g_pixel_values_chw.append(ev_img)

        ori_h, ori_w = int(dense_voxels.shape[-2]), int(dense_voxels.shape[-1])

        return {
            "index": index,
            "video_id": vid_id,
            "exp_id": exp_id,
            "text_prompt": text,
            "all_frames": all_frames,
            "dense_frames_names": [all_frames[i] for i in dense_idx],
            "length": n_total,
            "ori_height": ori_h,
            "ori_width": ori_w,
            "pixel_values": pixel_values,
            "g_pixel_values_raw": g_pixel_values_chw,
            "dense_indices": dense_idx,
            "llm_indices": llm_idx,
        }
