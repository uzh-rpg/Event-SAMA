"""Event-voxel datasets for Event SAMA.

Loads raw event streams produced by the prophesee_pipeline / e2sim_pipeline in
the Event-SAMA conversion repo, accumulates them into 2-channel polarity
histograms at user-selected timestamps, and yields a tensor shaped
``[N_frames, 2, H, W]`` -- a drop-in replacement for the RGB
``[N_frames, 3, H, W]`` tensors consumed by the SAMA pipeline.

HDF5 schema (per-clip file, written by Event-SAMA-new/prophesee_pipeline):
    x : (N,) int16      pixel column   [0, native_w)
    y : (N,) int16      pixel row      [0, native_h)
    t : (N,) int32      microseconds from clip start (attrs[t_unit] = "us")
    p : (N,) int8       polarity, +1 (ON) or -1 (OFF)
    attrs: native_height, native_width, native_fps, n_events, t_unit

Usage (config):
    video_mevis_event_dataset = dict(
        type=VideoMeVISEventDataset,
        event_folder='<root_with_one_h5_per_clip>',
        # ... all the usual VideoMeVISDataset args (expression_file, mask_file,
        #     tokenizer, max_length, sampled_frames, etc.)
        event_fps=100,         # event-frame rate -> 10ms accumulation window
        event_normalize=True,  # asinh + fixed scalar (no per-clip z-score)
    )

This file also defines event analogues for the other Sa2VA / SAMA training
sources we have converted:

    * ``VideoLVVISEventDataset``     -- LVVIS train (COCO-VIS schema)
    * ``VideoChatUniViEventDataset`` -- ChatUniVi (ActivityNet-style video QA)
    * ``RefCOCOEventDataset``        -- RefCOCO/+/g (single-image RES)

All of them feed the same ``[N_frames, 2, H, W]`` tensor format to the model
via ``EventVoxelTransform``.
"""

from __future__ import annotations

import copy
import json
import os
import random
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode

from .ChatUniVi_Dataset import VideoChatUniViDataset
from .encode_fn import video_lisa_encode_fn
from .MeVIS_Dataset import VideoMeVISDataset
from .RefCOCO_Dataset import ReferSegmDataset
from .ReVOS_Dataset import VideoReVOSDataset

# Polarity ablation switch (EVENT_MERGE_POLARITY=1): read once at import so
# the per-event-window hot path pays no repeated getenv cost. Set by the
# nopolarity ablation config before dataloader workers fork; unset (default)
# everywhere else, in which case behavior is bit-identical to before.
_MERGE_POLARITY = os.environ.get('EVENT_MERGE_POLARITY', '0') == '1'
if _MERGE_POLARITY:
    print('[Event SAMA] EVENT_MERGE_POLARITY=1 -- polarity ablation ACTIVE: '
          'ON/OFF channels collapsed to their mean in every voxel.',
          flush=True)


def _accumulate_polarity_histogram(x: np.ndarray, y: np.ndarray, p: np.ndarray,
                                   start_us: int, end_us: int, t: np.ndarray,
                                   H: int, W: int) -> np.ndarray:
    """Build a 2-channel polarity histogram for events with t in [start_us, end_us).

    Returns: (2, H, W) float32 with ch0 = ON-event counts, ch1 = OFF-event counts.
    Assumes ``t`` is sorted ascending (the prophesee pipeline writes sorted ``t``).
    """
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

    # ch0 = ON (+1), ch1 = OFF (-1)
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
        # Polarity ablation: destroy ON/OFF information while keeping the
        # (2, H, W) shape and the total event count. Both channels become
        # (ON + OFF) / 2, so the ViT input carries no polarity signal and
        # the SAM2 pseudo-RGB rendering (R=OFF, B=ON) degenerates to R == B.
        m = out.sum(axis=0, keepdims=True) * 0.5
        out = np.repeat(m, 2, axis=0)
    return out


# Fixed scaling constant for event normalization. Chosen so that asinh of a
# typical event-count value (e.g. 5-50 ON/OFF events per pixel-bin) lands in
# roughly [0, 1]. asinh(50)/4 = 0.99, asinh(5)/4 = 0.58, asinh(1)/4 = 0.22,
# asinh(0)/4 = 0. The most important property is that pixels with zero events
# stay exactly zero post-transform, so:
#   (a) the model can learn "no events => no signal",
#   (b) frames with zero total activity remain detectable as such, and
#   (c) two identical inputs always produce the same output (no clip-dependent
#       shift like the previous per-clip z-score did, which made empty frames
#       collapse to a constant negative value and confused image_flags).
EVENT_ASINH_SCALE = 4.0


class EventVoxelTransform:
    """Resize + normalize a (N, 2, H, W) tensor of polarity histograms.

    Normalization (when ``normalize=True``):

        x' = asinh(x) / EVENT_ASINH_SCALE

    No per-clip / per-batch statistics are used. asinh is monotonic, behaves
    like ``x`` near 0, and like ``log(2x)`` for large ``x``, compressing the
    long tail of high-event-count pixels while keeping zeros at zero.

    The previous version applied a per-clip z-score after asinh which made
    empty frames map to a non-zero constant ``-mean/std``. That broke the
    ``image_flags`` heuristic in InternVL_Slowfast (sum != 0) and silently
    corrupted training when a clip happened to contain a frame with no events
    (common at low ``event_fps`` near video boundaries).
    """

    def __init__(self, image_size: int, normalize: bool = True):
        self.image_size = image_size
        self.normalize = normalize

    def __call__(self, voxels_nchw: np.ndarray) -> torch.Tensor:
        # voxels_nchw: (N, 2, H, W) float32
        x = torch.from_numpy(voxels_nchw)
        # Bilinear resize to (image_size, image_size). interpolate expects 4-D.
        x = F.interpolate(
            x,
            size=(self.image_size, self.image_size),
            mode='bilinear',
            align_corners=False,
        )
        if self.normalize:
            x = torch.asinh(x) / EVENT_ASINH_SCALE
        return x  # (N, 2, image_size, image_size)


class VideoEventBaseDataset(VideoReVOSDataset):
    """Base class: reuses the ReVOS/MeViS pipeline but loads event voxels.

    Concrete dataset classes (per source) only need to set ``event_folder`` and
    optionally override ``_event_path_for_video``.

    Layout assumed (matches Event-SAMA-new/prophesee_pipeline output):
        ``<event_folder>/<clip_id>/events.h5`` — one HDF5 per video clip.
    Override ``event_layout="flat"`` for ``<event_folder>/<clip_id>.h5``.
    """

    EVENT_FILE_NAME = 'events.h5'

    def __init__(self,
                 *args,
                 event_folder: str,
                 event_fps: float = 100.0,
                 event_normalize: bool = True,
                 event_layout: str = 'nested',
                 fallback_native_fps: Optional[float] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.event_folder = event_folder
        self.event_fps = float(event_fps)
        self.event_window_us = int(round(1e6 / self.event_fps))
        self.event_normalize = event_normalize
        self.event_layout = event_layout
        self.fallback_native_fps = fallback_native_fps

        # Replace the parent's RGB transform with the event-voxel transform.
        # We keep ``self.preprocessor`` paths unchanged (events go through
        # ``self.transformer`` only).
        self.transformer = EventVoxelTransform(
            image_size=self.image_size, normalize=self.event_normalize)
        self.preprocessor = None

    def _event_path_for_video(self, video_id: str) -> str:
        if self.event_layout == 'flat':
            return os.path.join(self.event_folder, video_id + '.h5')
        return os.path.join(self.event_folder, video_id, self.EVENT_FILE_NAME)

    def _frame_index_to_us(self, frame_idx: int, native_fps: float,
                           frames_list, t_min: int, t_max: int) -> int:
        """Map a frame index in the MeViS metadata to a microsecond timestamp.

        We try two conventions and pick the one that lands inside the
        recorded ``[t_min, t_max]`` window:

          1. ``numeric stem * 1e6 / native_fps`` -- correct when the events.h5
             covers the entire source video (absolute frame numbering).
          2. ``frame_idx * 1e6 / native_fps`` -- correct when the events.h5
             starts at t=0 at the first annotated frame (per-clip numbering).

        Both reduce to (1) when ``frames_list`` starts at index 0.
        """
        candidates = []
        try:
            numeric = int(str(frames_list[frame_idx]))
            candidates.append(int(round(numeric * 1e6 / native_fps)))
        except Exception:
            pass
        candidates.append(int(round(frame_idx * 1e6 / native_fps)))
        # Pick the first candidate that lands in [t_min, t_max]; otherwise the
        # last candidate (clamped later by searchsorted -> empty window).
        for c in candidates:
            if t_min <= c <= t_max:
                return c
        return candidates[-1]

    def _load_event_voxels(self, video_id: str, frame_indices,
                           frames_list) -> np.ndarray:
        """Load HDF5 events for ``video_id`` and accumulate one 2-ch histogram
        per frame index in ``frame_indices``.

        Returns: (N, 2, H, W) float32 numpy.
        """
        import h5py
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
        return np.stack(voxels, axis=0)

    # -- override the image-loading bits of __getitem__ ------------------------

    def __getitem__(self, index):
        # We reuse the parent's logic almost in full, but swap the
        # JPEG-loading block for event-voxel loading. The cleanest way to keep
        # this in sync with upstream is to replicate the parent body and patch
        # the loader inline.
        import copy
        import random

        index = index % self.real_len()
        selected_video_objects = self.vid2metaid[self.videos[index]]
        video_objects_infos = [
            copy.deepcopy(self.text_data[idx])
            for idx in selected_video_objects
        ]

        if len(video_objects_infos) > self.select_number:
            selected_indexes = np.random.choice(len(video_objects_infos),
                                                self.select_number)
        else:
            selected_indexes = np.random.choice(len(video_objects_infos),
                                                self.select_number,
                                                replace=True)
        video_objects_infos = [video_objects_infos[i] for i in selected_indexes]

        data_dict = self.dataset_map_fn(video_objects_infos,
                                        select_k=self.dense_len)
        assert 'images' in data_dict.keys()

        # Recover the integer frame indices that ``dataset_map_fn`` selected.
        # ``data_dict['images']`` looks like ``"<video_id>/<frame_name>.jpg"``.
        frames_list = video_objects_infos[0]['frames']
        frame_indices = []
        for img_rel in data_dict['images']:
            stem = os.path.splitext(os.path.basename(img_rel))[0]
            try:
                frame_indices.append(frames_list.index(stem))
            except ValueError:
                frame_indices.append(0)

        video_id = video_objects_infos[0]['video']

        # Load event voxels at the sampled timestamps.
        voxels = self._load_event_voxels(video_id, frame_indices, frames_list)
        ori_h, ori_w = voxels.shape[-2], voxels.shape[-1]
        pixel_values = self.transformer(voxels)  # (N, 2, image_size, image_size)
        data_dict['pixel_values'] = pixel_values

        # Replicate the extra-image-processor (SAM2) path. SAM2 still expects
        # RGB; for V1 we render the event voxels as a 3-channel pseudo-image
        # (R = OFF count, B = ON count, G = total density) for the SAM2 encoder.
        # If the user provides paired RGB sister-frames later, swap this block.
        if hasattr(self, 'extra_image_processor') and self.extra_image_processor is not None:
            extra_pixel_values = []
            dense_indices = data_dict.get('dense_indices', [])
            for n_i, frame_voxel in enumerate(voxels):
                if n_i not in dense_indices:
                    continue
                ev_img = _voxel_to_pseudo_rgb(frame_voxel)  # (H, W, 3) uint8
                ev_img = self.extra_image_processor.apply_image(ev_img)
                ev_t = torch.from_numpy(ev_img).permute(2, 0, 1).contiguous()
                extra_pixel_values.append(ev_t)
            data_dict['g_pixel_values'] = extra_pixel_values

        # Masks. Important: the RLE in mask_dict was encoded at the *original
        # RGB frame* resolution, which need not match the event voxel
        # resolution (the prophesee pipeline often resizes/letter-boxes
        # before simulation). Pass the RLE-native size to decode_mask --
        # otherwise the OR fill in decode_mask hits a broadcast error like
        # ``operands could not be broadcast together with shapes (eH,eW) (mH,mW)``.
        rle_h, rle_w = _first_rle_size(data_dict['video_masks'],
                                       fallback=(ori_h, ori_w))
        masks = self.decode_mask(data_dict['video_masks'],
                                 image_size=(rle_h, rle_w))
        if masks is None:
            return self.__getitem__(random.randint(0, self.real_len()))
        data_dict['masks'] = masks

        # Tokenization (mirrors parent).
        result = self.template_map_fn(data_dict)
        data_dict.update(result)
        from .encode_fn import video_lisa_encode_fn
        result = video_lisa_encode_fn(
            data_dict, tokenizer=self.tokenizer, max_length=self.max_length)
        data_dict.update(result)

        return data_dict


def _first_rle_size(video_masks, fallback):
    """Return (h, w) from the first non-None RLE in a nested video_masks list.

    ``video_masks`` is shaped ``[obj][anno][frame]`` where the leaves are
    pycocotools RLE dicts (or None for occluded frames). Each RLE carries its
    own ``size`` = [h, w].
    """
    for object_masks in video_masks:
        for anno_masks in object_masks:
            for frame_rle in anno_masks:
                if frame_rle is None:
                    continue
                size = frame_rle.get('size') if isinstance(frame_rle, dict) \
                    else None
                if size is not None and len(size) == 2:
                    return int(size[0]), int(size[1])
    return fallback


def _voxel_to_pseudo_rgb(voxel_2chw: np.ndarray) -> np.ndarray:
    """Render a (2, H, W) polarity-count voxel as an (H, W, 3) uint8 image.
    R = OFF events, B = ON events, G = (ON + OFF) density. Used only as input
    to SAM2's RGB-pretrained image encoder for grounding masks; the LLM never
    sees this rendering.
    """
    on = voxel_2chw[0]
    off = voxel_2chw[1]
    density = on + off
    def _norm(a):
        m = a.max()
        return (a / m * 255.0) if m > 0 else a
    img = np.stack([_norm(off), _norm(density), _norm(on)], axis=-1)
    return img.astype(np.uint8)


class VideoMeVISEventDataset(VideoEventBaseDataset, VideoMeVISDataset):
    """MeViS, but with event voxels instead of RGB JPEGs."""
    pass


# =============================================================================
#                           LVVIS  (long-video VIS)
# =============================================================================
#
# LVVIS train annotations follow the COCO-VIS schema:
#   train_instances.json = {
#       'videos':     [{id, height, width, length, file_names}, ...],
#       'categories': [{id, name, partition}, ...],
#       'annotations':[{video_id, category_id, segmentations: [RLE per frame]}],
#   }
# We translate to the MeViS-style schema expected by ``VideoReVOSDataset``:
#   expression_file equivalent -> {'videos': {<vid>: {'frames': [...],
#                                                     'expressions': {...}}}}
#   mask_file equivalent       -> {<anno_id_str>: [<RLE per frame>, ...]}
# and feed those to the parent via an in-memory ``json_file_preprocess``
# override.

class VideoLVVISDataset(VideoReVOSDataset):
    """LVVIS train, COCO-VIS schema -> MeViS-like schema, RGB JPEG loading.

    The constructor accepts ``expression_file`` (LVVIS train_instances.json),
    keeps ``mask_file=None`` because masks are inlined into the same JSON, and
    builds a ``meta_expressions``-style structure on the fly.

    Each LVVIS instance becomes one expression, with the category name as the
    expression text. Multi-instance videos contribute multiple expressions for
    the same video, exactly matching how MeViS / RefYouTube-VOS expressions
    work.
    """

    def __init__(self, *args, **kwargs):
        # mask_file is optional for LVVIS (masks live inside expression_file).
        kwargs.setdefault('mask_file', None)
        super().__init__(*args, **kwargs)

    def json_file_preprocess(self, expression_file, mask_file):
        with open(expression_file, 'r') as f:
            j = json.load(f)
        videos = j['videos']
        cats = {c['id']: c['name'] for c in j['categories']}
        annos = j['annotations']

        # vid_id -> {'name': str, 'frames': [stems], 'expressions': {exp_id: ...}}
        vid_name_for_id = {}
        meta_expressions = {}  # MeViS-style outer dict
        mask_dict = {}
        anno_count = 0

        for v in videos:
            vid_id = v['id']
            file_names = v['file_names']  # ["00000/00000.jpg", "00000/00001.jpg", ...]
            # All frames in a LVVIS clip share the same parent dir == vid_name.
            vid_name = file_names[0].split('/')[0]
            vid_name_for_id[vid_id] = vid_name
            frames = sorted([os.path.splitext(os.path.basename(p))[0]
                             for p in file_names])
            meta_expressions[vid_name] = dict(frames=frames, expressions={})

        # group annotations by video_id -> list of (anno_idx, anno_dict)
        per_vid = {}
        for a in annos:
            per_vid.setdefault(a['video_id'], []).append(a)

        for vid_id, anno_list in per_vid.items():
            if vid_id not in vid_name_for_id:
                continue
            vid_name = vid_name_for_id[vid_id]
            for local_exp_id, a in enumerate(anno_list):
                cat_name = cats.get(a['category_id'], f'object_{a["category_id"]}')
                anno_id_str = str(anno_count)
                # mask_dict keys are strings, values are lists of per-frame RLEs
                # (one per frame). LVVIS RLEs may be None for occluded frames.
                mask_dict[anno_id_str] = a['segmentations']
                meta_expressions[vid_name]['expressions'][str(local_exp_id)] = dict(
                    exp=cat_name,
                    anno_id=[anno_id_str],
                    obj_id=[anno_count],
                )
                anno_count += 1

        # Drop videos that ended up with zero usable expressions.
        meta_expressions = {k: v for k, v in meta_expressions.items()
                            if v['expressions']}

        # Reproduce the parent's transformation from meta_expressions -> metas.
        metas = []
        vid2metaid = {}
        meta_count = 0
        for vid_name, vd in meta_expressions.items():
            vid_frames = sorted(vd['frames'])
            vid_len = len(vid_frames)
            for exp_id in sorted(vd['expressions'].keys()):
                ed = vd['expressions'][exp_id]
                metas.append(dict(
                    video=vid_name,
                    exp=ed['exp'],
                    mask_anno_id=ed['anno_id'],
                    obj_id=ed.get('obj_id', [0]),
                    anno_id=[str(meta_count)],
                    frames=vid_frames,
                    exp_id=exp_id,
                    length=vid_len,
                ))
                vid2metaid.setdefault(vid_name, []).append(meta_count)
                meta_count += 1

        return vid2metaid, metas, mask_dict


class VideoLVVISEventDataset(VideoEventBaseDataset, VideoLVVISDataset):
    """LVVIS, but with event voxels instead of RGB JPEGs.

    Frame indices in the LVVIS json are 0-based and dense (file_names[i] is
    frame ``i``), so the parent's ``_frame_index_to_us`` (which already tries
    both numeric-stem and positional indexing) handles them correctly.
    """
    pass


# =============================================================================
#                       ChatUniVi  (ActivityNet-style video QA)
# =============================================================================

class VideoChatUniViEventDataset(VideoChatUniViDataset):
    """ChatUniVi video-QA, but with event voxels.

    ChatUniVi annotations point to MP4/MKV files in ``Activity_Videos``;
    the event-converted twin lives at ``<event_folder>/<video_stem>/events.h5``
    where ``<video_stem>`` is the basename without extension (e.g.
    ``v_---9CpRcKoU``). ~89% of ChatUniVi clips have a converted twin; clips
    without one are skipped at sample time and the dataset re-draws another.

    The parent loads frames with decord; we replace that with the same event
    accumulator used by the segmentation event datasets (2-channel ON/OFF
    polarity histogram per sampled frame).
    """

    def __init__(self,
                 *args,
                 event_folder: str,
                 event_fps: float = 100.0,
                 event_normalize: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.event_folder = event_folder
        self.event_fps = float(event_fps)
        self.event_window_us = int(round(1e6 / self.event_fps))
        self.event_normalize = event_normalize
        self.transformer = EventVoxelTransform(
            image_size=self.image_size, normalize=self.event_normalize)
        self.preprocessor = None
        # Pre-index available event clips so missing-file misses are O(1).
        self._available_clips = set(os.listdir(self.event_folder)) \
            if os.path.isdir(self.event_folder) else set()
        # Build a list of dataset positions whose video is actually available.
        self._available_indices = []
        for i, item in enumerate(self.text_data):
            stem = os.path.splitext(os.path.basename(item['video']))[0]
            if stem in self._available_clips:
                self._available_indices.append(i)
        print(f"[ChatUniVi-Event] {len(self._available_indices)} / "
              f"{len(self.text_data)} videos have a converted event twin in "
              f"{self.event_folder}")
        if not self._available_indices:
            raise RuntimeError(
                f'No ChatUniVi clips found under {self.event_folder} -- check '
                f'the event_folder path.')

    def _event_path(self, video_stem: str) -> str:
        return os.path.join(self.event_folder, video_stem, 'events.h5')

    def _load_event_voxels(self, video_stem: str, n_select: int) -> np.ndarray:
        """Load the events.h5 for ``video_stem`` and return ``n_select``
        evenly spaced 2-channel polarity histograms.

        Returns: (n_select, 2, H, W) float32.
        """
        import h5py
        with h5py.File(self._event_path(video_stem), 'r') as f:
            x = f['x'][:]
            y = f['y'][:]
            t = f['t'][:]
            p = f['p'][:]
            H = int(f.attrs.get('native_height',
                                 (y.max() + 1) if len(y) else 1))
            W = int(f.attrs.get('native_width',
                                 (x.max() + 1) if len(x) else 1))
            t_unit = f.attrs.get('t_unit', b'us')
            if isinstance(t_unit, bytes):
                t_unit = t_unit.decode('utf-8', errors='replace')
            assert t_unit == 'us', f'expected t_unit=us, got {t_unit}'

        if len(t) == 0:
            return np.zeros((n_select, 2, H, W), dtype=np.float32)

        t_min = int(t[0])
        t_max = int(t[-1])
        # Evenly space ``n_select`` time-centres across the clip (matching the
        # parent's "sparse_len evenly spaced frames" behaviour for MP4s).
        if n_select == 1:
            centres = np.array([t_min], dtype=np.int64)
        else:
            centres = np.linspace(t_min, t_max, n_select).astype(np.int64)
        half = self.event_window_us // 2
        voxels = []
        for c in centres:
            voxels.append(_accumulate_polarity_histogram(
                x, y, p, start_us=int(c) - half, end_us=int(c) + half,
                t=t, H=H, W=W))
        return np.stack(voxels, axis=0)

    def __getitem__(self, index):
        # Map the requested index into our list of clips that actually have a
        # converted event file. The parent's __len__ multiplies by self.repeats
        # so we still rotate through the whole subset.
        index = self._available_indices[index % len(self._available_indices)]
        selected_data_dict = copy.deepcopy(self.text_data[index])
        video_field = selected_data_dict['video']
        video_stem = os.path.splitext(os.path.basename(video_field))[0]

        # Build the text dict using the parent's text-prep machinery, but
        # bypass the decord MP4 loader (we synthesize images from events).
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

        # Load events at sparse_len evenly-spaced timestamps, transform.
        try:
            voxels = self._load_event_voxels(video_stem, self.sparse_len)
        except Exception as e:
            print(f"[ChatUniVi-Event] read err on {video_stem}: {e!r}; "
                  f"sampling another clip.")
            return self.__getitem__(random.randint(
                0, len(self._available_indices) - 1))

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


# =============================================================================
#               RefCOCO / RefCOCO+ / RefCOCOg  (single-image RES)
# =============================================================================

class RefCOCOEventDataset(ReferSegmDataset):
    """RefCOCO-style image RES with COCO-image events instead of RGB JPEGs.

    Each annotated image (e.g. ``COCO_train2014_000000098304.jpg``) has a
    corresponding event clip at::

        <event_folder>/COCO_train2014_000000098304/events.h5

    synthesised by the conversion pipeline using a short camera-shake of the
    still image (~196 fps, 0.3s). We accumulate the entire span into a single
    2-channel polarity histogram (effectively one event-frame per image),
    matching the single-frame RES interface of the parent class.

    The parent emits a 4-D ``pixel_values`` of shape (n_tiles, 3, H, W) with
    optional dynamic_preprocess tiling. Tiling makes no physical sense for
    events (no high-resolution detail to recover) so we always run in
    ``single_image_mode=True`` and emit (1, 2, H, W).
    """

    def __init__(self, *args, event_folder: str,
                 event_normalize: bool = True, **kwargs):
        # Force single-image mode -- tiling is not meaningful for events.
        kwargs['single_image_mode'] = True
        super().__init__(*args, **kwargs)
        self.event_folder = event_folder
        self.event_normalize = event_normalize
        self._event_transform = EventVoxelTransform(
            image_size=self.image_size, normalize=self.event_normalize)
        # Pre-index available event clips for fast misses.
        self._available_clips = set(os.listdir(self.event_folder)) \
            if os.path.isdir(self.event_folder) else set()
        if not self._available_clips:
            raise RuntimeError(
                f'No COCO event clips found under {self.event_folder}')

    def _event_path(self, image_filename: str) -> str:
        stem = os.path.splitext(os.path.basename(image_filename))[0]
        return os.path.join(self.event_folder, stem, 'events.h5')

    def _load_image_event_voxel(self, image_filename: str
                                ) -> Optional[np.ndarray]:
        """Return a (1, 2, H, W) float32 voxel for ``image_filename`` or None."""
        import h5py
        stem = os.path.splitext(os.path.basename(image_filename))[0]
        if stem not in self._available_clips:
            return None
        path = self._event_path(image_filename)
        with h5py.File(path, 'r') as f:
            x = f['x'][:]
            y = f['y'][:]
            t = f['t'][:]
            p = f['p'][:]
            H = int(f.attrs.get('native_height',
                                 (y.max() + 1) if len(y) else 1))
            W = int(f.attrs.get('native_width',
                                 (x.max() + 1) if len(x) else 1))
        if len(t) == 0:
            return np.zeros((1, 2, H, W), dtype=np.float32)
        voxel = _accumulate_polarity_histogram(
            x, y, p, start_us=int(t[0]), end_us=int(t[-1]) + 1,
            t=t, H=H, W=W)
        return voxel[None]  # (1, 2, H, W)

    def prepare_data(self, index):
        # Reuse the parent's text/mask machinery, but replace the RGB image
        # loading + tiling block with event voxel loading.
        from third_parts.mmdet.datasets.refcoco import RefCocoDataset
        data_dict = RefCocoDataset.prepare_data(self, index)
        data_dict = self._parse_annotations(data_dict)
        if data_dict is None:
            return None

        out_data_dict = {}
        if 'masks' in data_dict:
            out_data_dict['masks'] = data_dict['masks']

        image_file = data_dict.get('image', None)
        if image_file is None:
            token_dict = self.get_inputid_labels(
                conversations=data_dict['conversations'],
                image_token_str=None, context_token_str=None)
            out_data_dict.update(token_dict)
            out_data_dict['pixel_values'] = torch.zeros(
                1, 2, self.image_size, self.image_size)
            return out_data_dict

        voxel = self._load_image_event_voxel(image_file)
        if voxel is None:
            return None  # parent's __getitem__ will refetch with another index

        pixel_values = self._event_transform(voxel)  # (1, 2, image_size, image_size)
        out_data_dict['pixel_values'] = pixel_values

        # SAM2 still needs an RGB-shaped tensor; render the voxel as pseudo-RGB.
        if hasattr(self, 'extra_image_processor') and self.extra_image_processor is not None:
            ev_img = _voxel_to_pseudo_rgb(voxel[0])  # (H, W, 3) uint8
            ev_img = self.extra_image_processor.apply_image(ev_img)
            g_pixel_values = torch.from_numpy(ev_img).permute(2, 0, 1).contiguous()
            out_data_dict['g_pixel_values'] = g_pixel_values

        num_image_tokens = pixel_values.shape[0] * self.patch_token
        image_token_str = (f'{self.IMG_START_TOKEN}'
                           f'{self.IMG_CONTEXT_TOKEN * num_image_tokens}'
                           f'{self.IMG_END_TOKEN}')
        num_context_token = pixel_values.shape[0]
        context_str = self.IMG_CONTEXT_TOKEN * num_context_token

        token_dict = self.get_inputid_labels(
            conversations=data_dict['conversations'],
            image_token_str=image_token_str,
            context_token_str=context_str)
        out_data_dict.update(token_dict)
        return out_data_dict
