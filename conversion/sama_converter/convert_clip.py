#!/usr/bin/env python3
"""
Single-item event conversion worker.

Three recipes, all followed by the linear event generator (Ziegler et al., 2023):
  mode=frames|video, upsample=dis_hq : DIS optical flow (OpenCV MEDIUM preset) forward warping
  mode=frames|video, upsample=film   : FILM with vid2e's adaptive bisection upsampler (vid2e_film/)
  mode=images                        : synthetic saccade on a still image

Usage (single item; use ../convert.py for whole datasets):
  python -m sama_converter.convert_clip <source_path> --config configs/mevis.yaml [--output DIR]

Output schema:
  events.h5  keys: x (int16), y (int16), t (int32 µs), p (int8 ±1)
             attrs: native_height, native_width, native_fps, n_events, t_unit="us"
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import math
import os
from pathlib import Path
import resource
import shutil
import sys
import tempfile
import time

import cv2
import h5py
import numpy as np
import yaml

try:
    import numexpr as ne
    _HAS_NUMEXPR = True
except ImportError:
    _HAS_NUMEXPR = False


VID2E_FILM_DIR = Path(__file__).resolve().parent.parent / "vid2e_film"

MODES     = {"frames", "video", "images"}
UPSAMPLES = {"dis_hq", "film"}


# ── Config ────────────────────────────────────────────────────────────────────

def load_config(yaml_path: Path) -> dict:
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict):
    """Reject configs that ask for anything other than the three supported recipes."""
    mode = cfg.get("mode")
    if mode not in MODES:
        raise ValueError(f"mode must be one of {sorted(MODES)}, got {mode!r}")
    if cfg.get("event_gen", "linear") != "linear":
        raise ValueError(f"only event_gen=linear is supported, got {cfg['event_gen']!r}")
    if mode != "images":
        if cfg.get("upsample") not in UPSAMPLES:
            raise ValueError(f"upsample must be one of {sorted(UPSAMPLES)} for mode={mode}, "
                             f"got {cfg.get('upsample')!r}")
        if not cfg.get("target_fps"):
            raise ValueError(f"target_fps is required for mode={mode}")
        if mode == "frames" and not cfg.get("fps"):
            raise ValueError("fps is required for mode=frames (JPEG folders carry no timing)")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("source_path", type=Path,
                   help="Clip dir of JPEGs (frames), video file (video) or image file (images)")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output", type=Path, default=None,
                   help="Output root (default: output_root from the config)")
    p.add_argument("--tmpdir", type=Path, default=None,
                   help="Scratch dir for FILM intermediate PNGs (default: system tmpdir)")
    return p.parse_args()


# ── Frame I/O ─────────────────────────────────────────────────────────────────

def get_video_meta(path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    nf  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    if fps <= 0:
        print(f"[WARN] {path.name}: cv2 returned fps={fps}, falling back to 30fps")
        fps = 30.0
    return fps, nf, h, w


def load_frames_chunk(clip_dir: Path, start: int, end: int,
                      workers: int = 8) -> list:
    jpgs = sorted(clip_dir.glob("*.jpg"))[start:end]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        frames = list(pool.map(lambda p: cv2.imread(str(p)), jpgs))
    return frames


def resize_frames(frames: list, target_h: int, target_w: int) -> list:
    """Resize all frames to (target_h, target_w). Uses INTER_AREA for downscaling."""
    return [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA)
            for f in frames]


def gen_saccade_frames(img: np.ndarray, n_frames: int, amplitude_px: int) -> list:
    """
    Generate n_frames by translating img along a 3-pass saccadic trajectory.

    Trajectory: right → down → diagonal back to origin.
    Per-frame displacement ≈ amplitude_px / (n_frames // 3) pixels — keep < 1px
    for accurate linear event generation (default amp=8, n=60 → 0.4px/frame).
    """
    H, W = img.shape[:2]
    A   = float(amplitude_px)
    p   = n_frames // 3
    rem = n_frames - 2 * p
    offsets = (
        [(A * k / p,          0.0            ) for k in range(p)  ] +
        [(A,                  A * k / p      ) for k in range(p)  ] +
        [(A * (1 - k / rem),  A * (1 - k / rem)) for k in range(rem)]
    )
    out = []
    for tx, ty in offsets:
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        out.append(cv2.warpAffine(img, M, (W, H), borderMode=cv2.BORDER_REPLICATE))
    return out


# ── Upsampling ────────────────────────────────────────────────────────────────
#
# Input: list of (H,W,3) uint8 BGR frames and num_inter_frames N.
# Output: list of frames with N synthetic frames inserted between each pair.
# Output length: (len(frames) - 1) * (N + 1) + 1
#
# dis_hq matches DenseInterpolatedEventSimulator + DIS_HIGH preset of
# cogsys-tuebingen/event_simulator (Ziegler et al., 2023).

def _dense_flow_upsample(frames: list, n: int, flow_fn) -> list:
    """
    Forward-warp upsampling along a dense optical flow.
    flow_fn(gray1, gray2) → (H, W, 2) float32 optical flow.

    Matches DenseInterpolatedEventSimulator.cpp:
      alpha = k / (num_inter_frames + 1), k = 1..N
      interflow = alpha * flow
      inter_frame = remap(prev_frame, identity + interflow)
    """
    if n == 0 or len(frames) < 2:
        return frames
    h, w = frames[0].shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(w, dtype=np.float32),
                                  np.arange(h, dtype=np.float32))
    out = []
    for i in range(len(frames) - 1):
        out.append(frames[i])
        gray1 = cv2.cvtColor(frames[i],     cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frames[i + 1], cv2.COLOR_BGR2GRAY)
        flow  = flow_fn(gray1, gray2)   # (H, W, 2)
        for k in range(1, n + 1):
            alpha = k / (n + 1)
            map_x = (grid_x + flow[..., 0] * alpha)
            map_y = (grid_y + flow[..., 1] * alpha)
            warped = cv2.remap(frames[i], map_x, map_y, cv2.INTER_LINEAR)
            out.append(warped)
    out.append(frames[-1])
    return out


def _apply_upsample_dis_hq(frames: list, n: int) -> list:
    """DIS optical flow, MEDIUM preset. Paper: dense_dis_hq."""
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
    return _dense_flow_upsample(frames, n, lambda g1, g2: dis.calc(g1, g2, None))


def _apply_upsample_film(frames: list, native_fps: float,
                          chunk_tmp: Path, film_pair_batch: int = 1):
    """
    FILM neural-network frame upsampling. Returns (frames_out, ts_us_relative).

    ts_us_relative: float64 array (N_out,) of µs timestamps, one per output frame,
    relative to the first native frame of this chunk (ts[0] = 0.0 µs).
    Derived from FILM's timestamps.txt which stores seconds.

    FILM uses content-adaptive binary bisection: inter-frames per pair = 2^k - 1
    where k depends on optical flow magnitude. Inter-frame counts differ across
    pairs, so ts_us_relative is the only correct source of timestamps for event gen.

    chunk_tmp: unique per-chunk path under node-local tmpdir.
    """
    _up_path = str(VID2E_FILM_DIR)
    if _up_path not in sys.path:
        sys.path.insert(0, _up_path)

    chunk_in  = chunk_tmp / "in"
    chunk_out = chunk_tmp / "out"
    shutil.rmtree(chunk_tmp, ignore_errors=True)

    try:
        imgs_dir = chunk_in / "seq" / "imgs"
        imgs_dir.mkdir(parents=True)
        for i, f in enumerate(frames):
            cv2.imwrite(str(imgs_dir / f"{i:08d}.png"), f)
        (chunk_in / "seq" / "fps.txt").write_text(str(native_fps))

        os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
        from utils.upsampler import Upsampler  # TF import deferred to here
        Upsampler(input_dir=str(chunk_in), output_dir=str(chunk_out),
                  film_pair_batch_size=film_pair_batch).upsample()

        # timestamps.txt: one value per output frame, in seconds, relative to chunk start
        ts_files = [p for p in chunk_out.rglob("timestamps.txt")]
        if not ts_files:
            raise FileNotFoundError(f"FILM produced no timestamps.txt under {chunk_out}")
        ts_sec = np.atleast_1d(np.loadtxt(str(ts_files[0])))  # (N_out,) seconds
        ts_us_rel = (ts_sec * 1e6).astype(np.float64)          # convert to µs

        imgs_out = sorted((ts_files[0].parent / "imgs").glob("*.png"))
        frames_out = [cv2.imread(str(p)) for p in imgs_out]
        return frames_out, ts_us_rel
    finally:
        shutil.rmtree(chunk_tmp, ignore_errors=True)


# ── Event generation ──────────────────────────────────────────────────────────

def gen_events_linear(frames: list,
                      c_pos: float, c_neg: float,
                      native_start_us: float, us_per_native_frame: float,
                      n: int,
                      frame_ts_us: "np.ndarray | None" = None):
    """
    Linear event generator (Ziegler et al., 2023). Pure NumPy.

    Matches DenseInterpolatedEventSimulator.cpp threshold logic:
      darker  = prev - curr > c_neg  → OFF events
      lighter = curr - prev > c_pos  → ON  events

    Args:
        frames:               list of (H,W,3) uint8 BGR — already upsampled
        c_pos, c_neg:         event thresholds (pixel-diff scale, 0–255)
        native_start_us:      timestamp of first native frame in this chunk (µs)
        us_per_native_frame:  microseconds between consecutive native frames
        n:                    num_inter_frames used for upsampling this chunk
        frame_ts_us:          optional float64 array (len == len(frames)) of actual
                              per-frame timestamps in µs. When provided, events for
                              pair (j → j+1) are timestamped at frame_ts_us[j+1]
                              instead of using uniform interpolation. Required for
                              FILM upsampling where inter-frame counts vary per pair.

    Returns:
        xs, ys, ts_us, ps — lists of 1-D int arrays
    """
    xs, ys, ts_us, ps = [], [], [], []
    factor = n + 1   # number of sub-intervals per native frame gap (ignored when frame_ts_us set)

    c_pos_f = np.float32(c_pos)
    c_neg_f = np.float32(c_neg)
    for j in range(len(frames) - 1):
        prev = frames[j]
        curr = frames[j + 1]

        # Grayscale diff for thresholding 
        if prev.ndim == 3:
            ne_ok = False
            if _HAS_NUMEXPR:
                try:
                    pr = prev[..., 2].astype(np.float32); pg = prev[..., 1].astype(np.float32); pb = prev[..., 0].astype(np.float32)
                    cr = curr[..., 2].astype(np.float32); cg = curr[..., 1].astype(np.float32); cb = curr[..., 0].astype(np.float32)
                    diff_g   = ne.evaluate("0.299*(cr-pr) + 0.587*(cg-pg) + 0.114*(cb-pb)",
                                           local_dict={"cr": cr, "pr": pr, "cg": cg, "pg": pg, "cb": cb, "pb": pb})
                    on_mask  = ne.evaluate("diff_g > c_pos_f",
                                           local_dict={"diff_g": diff_g, "c_pos_f": c_pos_f})
                    off_mask = ne.evaluate("diff_g < -c_neg_f",
                                           local_dict={"diff_g": diff_g, "c_neg_f": c_neg_f})
                    ne_ok = True
                except Exception:
                    pass
            if not ne_ok:
                prev_g = (0.299 * prev[..., 2].astype(np.float32)
                          + 0.587 * prev[..., 1].astype(np.float32)
                          + 0.114 * prev[..., 0].astype(np.float32))
                curr_g = (0.299 * curr[..., 2].astype(np.float32)
                          + 0.587 * curr[..., 1].astype(np.float32)
                          + 0.114 * curr[..., 0].astype(np.float32))
                diff_g   = curr_g - prev_g
                on_mask  = diff_g > c_pos
                off_mask = diff_g < -c_neg
        else:
            diff_g   = curr.astype(np.float32) - prev.astype(np.float32)
            on_mask  = diff_g > c_pos
            off_mask = diff_g < -c_neg

        on_ys,  on_xs  = np.where(on_mask)
        off_ys, off_xs = np.where(off_mask)

        if len(on_xs) == 0 and len(off_xs) == 0:
            continue

        # Timestamp for events between frames[j] and frames[j+1].
        # When actual per-frame timestamps are available (FILM path), use frames[j+1]'s
        # timestamp directly — FILM's adaptive bisection produces non-uniform inter-frame
        # counts per pair, so uniform interpolation from n would give wrong timestamps.
        if frame_ts_us is not None:
            ts_us_val = np.int32(frame_ts_us[j + 1])
        else:
            native_pair_idx = j // factor
            sub_idx         = j % factor
            alpha           = (sub_idx + 1) / factor
            native_ts0_us   = native_start_us + native_pair_idx * us_per_native_frame
            native_ts1_us   = native_ts0_us + us_per_native_frame
            ts_us_val       = np.int32(native_ts0_us + alpha * (native_ts1_us - native_ts0_us))

        if len(on_xs) > 0:
            xs.append(on_xs.astype(np.int16))
            ys.append(on_ys.astype(np.int16))
            ts_us.append(np.full(len(on_xs), ts_us_val, dtype=np.int32))
            ps.append(np.ones(len(on_xs), dtype=np.int8))

        if len(off_xs) > 0:
            xs.append(off_xs.astype(np.int16))
            ys.append(off_ys.astype(np.int16))
            ts_us.append(np.full(len(off_xs), ts_us_val, dtype=np.int32))
            ps.append(np.full(len(off_xs), -1, dtype=np.int8))

    return xs, ys, ts_us, ps


# ── HDF5 output ───────────────────────────────────────────────────────────────

def open_h5_streaming(tmp_path: Path,
                      native_h: int, native_w: int, native_fps: float):
    """Open a streaming HDF5 file with resizable datasets (one per event field)."""
    f = h5py.File(tmp_path, "w")
    hdf5_chunk = 1 << 20   # 1M events per internal HDF5 chunk
    ckw = dict(chunks=(hdf5_chunk,), compression="gzip", compression_opts=4)
    f.create_dataset("x", shape=(0,), maxshape=(None,), dtype=np.int16, **ckw)
    f.create_dataset("y", shape=(0,), maxshape=(None,), dtype=np.int16, **ckw)
    f.create_dataset("t", shape=(0,), maxshape=(None,), dtype=np.int32, **ckw)  # microseconds
    f.create_dataset("p", shape=(0,), maxshape=(None,), dtype=np.int8,  **ckw)
    f.attrs["native_height"] = native_h
    f.attrs["native_width"]  = native_w
    f.attrs["native_fps"]    = native_fps
    f.attrs["t_unit"]        = "us"
    return f


def append_h5_chunk(f, xs: list, ys: list, ts_us: list, ps: list):
    """Append one chunk of events to an open streaming HDF5 file. No-op if empty."""
    if not xs:
        return
    x = np.concatenate(xs).astype(np.int16)
    y = np.concatenate(ys).astype(np.int16)
    t = np.concatenate(ts_us).astype(np.int32)  # microseconds
    p = np.concatenate(ps).astype(np.int8)
    if len(t) == 0:
        return
    for name, arr in [("x", x), ("y", y), ("t", t), ("p", p)]:
        ds  = f[name]
        n0  = ds.shape[0]
        ds.resize(n0 + len(arr), axis=0)
        ds[n0:] = arr


def write_h5(output_path: Path,
             xs, ys, ts_us, ps,
             native_h: int, native_w: int, native_fps: float):
    """
    Write events in one go (used for still images).

    Schema:
      x  (N,) int16  — pixel column  [0, native_w)
      y  (N,) int16  — pixel row     [0, native_h)
      t  (N,) int32  — microseconds from clip start  (attrs: t_unit="us")
      p  (N,) int8   — polarity: +1 ON, -1 OFF
    """
    def cat(lst, dtype):
        return np.concatenate(lst).astype(dtype) if lst else np.array([], dtype=dtype)

    x = cat(xs,    np.int16)
    y = cat(ys,    np.int16)
    t = cat(ts_us, np.int32)  # microseconds
    p = cat(ps,    np.int8)

    if len(t) > 1:
        order = np.argsort(t)
        x, y, t, p = x[order], y[order], t[order], p[order]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(".tmp.h5")
    ckw = dict(compression="gzip", compression_opts=4)
    with h5py.File(tmp_path, "w") as f:
        f.create_dataset("x", data=x, **ckw)
        f.create_dataset("y", data=y, **ckw)
        f.create_dataset("t", data=t, **ckw)
        f.create_dataset("p", data=p, **ckw)
        f.attrs["native_height"] = native_h
        f.attrs["native_width"]  = native_w
        f.attrs["native_fps"]    = native_fps
        f.attrs["n_events"]      = len(t)
        f.attrs["t_unit"]        = "us"
    tmp_path.rename(output_path)  # atomic: only visible after complete write


def _peak_rss_mb() -> float:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if sys.platform == "darwin" else rss / 1024  # bytes on macOS, KB on Linux


# ── Output naming ─────────────────────────────────────────────────────────────

_SPLIT_NAMES = {"train", "val", "val_u", "valid", "valid_u", "test", "all",
                "train2014", "val2014", "train2017", "val2017",
                "gqa", "ocr_vqa", "textvqa", "vg"}


def clip_id(source: Path, mode: str, subdir_levels: int = 0) -> str:
    if mode in ("video", "images"):
        return source.stem
    if subdir_levels > 0:
        # Include N parent dirs above the clip dir (e.g. subsplit like Charades/clip_id).
        return "/".join(source.parts[-(subdir_levels + 1):])
    return source.name


def get_split(source: Path) -> str:
    """Return the split name (train/val/test) found in the source path, or ''."""
    for part in reversed(source.parts[:-1]):
        if part.lower() in _SPLIT_NAMES:
            return part
    return ""


def output_path(source: Path, cfg: dict, output_root) -> Path:
    """<output_root>/<dataset>/[<split>/]<clip_id>/events.h5"""
    split = get_split(source)
    cid   = clip_id(source, cfg["mode"], int(cfg.get("subdir_levels", 0)))
    base  = Path(output_root) / cfg["dataset"]
    return (base / split / cid / "events.h5") if split else (base / cid / "events.h5")


# ── Main ──────────────────────────────────────────────────────────────────────

def convert_one(source_path: Path, cfg: dict, output_root, tmpdir=None):
    """Convert one clip dir / video / image to <output_root>/.../events.h5. Skips if done."""
    source_path     = Path(source_path)
    mode            = cfg["mode"]
    upsample_method = cfg.get("upsample")
    c_pos           = float(cfg.get("c_pos", 3.0))
    c_neg           = float(cfg.get("c_neg", 3.0))

    # ── FILM setup (only active when upsample=film) ───────────────────────────
    film_pair_batch = int(cfg.get("film_pair_batch", 1) or 1)
    _film_base = (Path(tmpdir) if tmpdir
                  else Path(tempfile.gettempdir())) / f"film_{source_path.stem}"

    out_path = output_path(source_path, cfg, output_root)
    if out_path.exists():
        print(f"[SKIP] {source_path.name}")
        return

    # ── still image: saccade simulation ──────────────────────────────────────
    if mode == "images":
        img = cv2.imread(str(source_path))
        if img is None:
            print(f"[SKIP] {source_path.name} — cannot read image")
            return

        native_h, native_w = img.shape[:2]
        target_size = cfg.get("target_size", None)
        if target_size and (native_h > int(target_size[0]) or native_w > int(target_size[1])):
            max_h, max_w = int(target_size[0]), int(target_size[1])
            scale = min(max_h / native_h, max_w / native_w)
            sim_h = int(native_h * scale)
            sim_w = int(native_w * scale)
            img = cv2.resize(img, (sim_w, sim_h), interpolation=cv2.INTER_AREA)
        else:
            sim_h, sim_w = native_h, native_w

        n_sac  = int(cfg.get("saccade_n_frames",    60))
        dur_ms = float(cfg.get("saccade_duration_ms", 300.0))
        amp_px = int(cfg.get("saccade_amplitude_px",  8))

        frames        = gen_saccade_frames(img, n_sac, amp_px)
        us_per_frame  = dur_ms * 1000.0 / (len(frames) - 1)
        effective_fps = (len(frames) - 1) * 1000.0 / dur_ms

        print(f"[START] {source_path.name}  saccade  frames={len(frames)}  "
              f"duration={dur_ms:.0f}ms  amplitude={amp_px}px  "
              f"sim={sim_w}×{sim_h}  event_gen=linear")

        xs, ys, ts_us, ps = gen_events_linear(
            frames, c_pos, c_neg, 0.0, us_per_frame, 0)

        write_h5(out_path, xs, ys, ts_us, ps, sim_h, sim_w, effective_fps)

        n_total = sum(len(x) for x in xs) if xs else 0
        print(f"[DONE] {source_path.name} → {n_total:,} events → {out_path}")
        print(f"  peak_rss={_peak_rss_mb():.0f}MB")
        return

    # ── source metadata ───────────────────────────────────────────────────────
    if mode == "video":
        native_fps, total_frames, native_h, native_w = get_video_meta(source_path)
    else:  # frames
        jpgs         = sorted(source_path.glob("*.jpg"))
        total_frames = len(jpgs)
        if total_frames == 0:
            print(f"[SKIP] {source_path.name} — no .jpg files found")
            return
        sample       = cv2.imread(str(jpgs[0]))
        native_h, native_w = sample.shape[:2]
        native_fps   = float(cfg["fps"])

    us_per_frame = 1e6 / native_fps   # µs between consecutive native frames

    # Inter-frame count for DIS-HQ (FILM picks its own count adaptively)
    target_fps = cfg["target_fps"]
    n = max(0, math.ceil(target_fps / native_fps) - 1)
    print(f"[INFO] target_fps={target_fps} @ native={native_fps}fps → n={n} "
          f"(effective {native_fps * (n + 1):.1f}fps)")

    # Target simulation resolution — downscale only, never upscale small clips.
    # Aspect ratio preserved: scale until the larger dim fits within target_size.
    target_size = cfg.get("target_size", None)
    if target_size and (native_h > int(target_size[0]) or native_w > int(target_size[1])):
        max_h, max_w = int(target_size[0]), int(target_size[1])
        scale = min(max_h / native_h, max_w / native_w)
        sim_h = int(native_h * scale)
        sim_w = int(native_w * scale)
    else:
        sim_h, sim_w = native_h, native_w

    resize_needed = (sim_h != native_h or sim_w != native_w)

    print(f"[START] {source_path.name}  frames={total_frames}  "
          f"fps={native_fps}  native={native_w}×{native_h}  sim={sim_w}×{sim_h}  "
          f"upsample={upsample_method}  n={n}  event_gen=linear")

    tmp_path = out_path.with_suffix(".tmp.h5")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h5f = open_h5_streaming(tmp_path, sim_h, sim_w, native_fps)

    t_load_total = t_upsample_total = t_eventgen_total = 0.0
    n_chunks = 0

    # Upsampled frames stay in RAM, so chunks are small (default 50 native frames).
    chunk_frames = int(cfg.get("chunk_frames_cpu", cfg.get("chunk_frames", 50)))
    total_chunks = math.ceil(total_frames / chunk_frames)

    # ── Sequential video reader (avoids h264 mmco errors from per-chunk seeking) ──
    # A single VideoCapture is opened and read forward — no cap.set() seeking.
    _vid_cap   = cv2.VideoCapture(str(source_path)) if mode == "video" else None
    _vid_carry = None  # 1-frame carry-over for chunk overlap

    for start in range(0, total_frames, chunk_frames):
        end = min(start + chunk_frames, total_frames)
        T   = end - start

        if T < 2:
            continue

        # Include one extra frame so the pair at the chunk boundary (end-1 → end)
        # is not lost between consecutive chunks.
        end_load = min(end + 1, total_frames)

        tc0 = time.perf_counter()
        if mode == "video":
            # Sequential read — prepend carry frame from previous chunk if available
            frames = [] if _vid_carry is None else [_vid_carry]
            n_new  = end_load - start - (0 if _vid_carry is None else 1)
            for _ in range(n_new):
                ok, f = _vid_cap.read()
                if not ok:
                    break
                frames.append(f)
            # Save last frame as carry if we loaded an extra (for next chunk's overlap)
            _vid_carry = frames[end - start] if (end_load > end and len(frames) > end - start) else None
        else:
            frames = load_frames_chunk(source_path, start, end_load)
        tc1 = time.perf_counter()

        if len(frames) < 2:
            continue

        if resize_needed:
            frames = resize_frames(frames, sim_h, sim_w)

        t_load_total += tc1 - tc0

        # ── Upsampling ────────────────────────────────────────────────────────
        tu0 = time.perf_counter()
        film_ts_abs = None   # actual per-frame µs timestamps from FILM (None for DIS-HQ)
        if upsample_method == "film":
            # FILM returns (frames, ts_us_relative) — timestamps in µs from chunk start.
            # These are the only correct timestamps: FILM's adaptive bisection produces
            # a different number of inter-frames per pair, so n_for_gen is only an average.
            frames_up, film_ts_rel = _apply_upsample_film(
                frames, native_fps, _film_base / f"c{start}", film_pair_batch)
            film_ts_abs = film_ts_rel + start * us_per_frame  # absolute µs
            _film_elapsed = time.perf_counter() - tu0
            _film_pairs = len(frames) - 1
            print(f"[FILM] chunk {n_chunks+1}/{total_chunks} | frames {start}-{end}/{total_frames}"
                  f" | pairs={_film_pairs} | {_film_elapsed:.0f}s"
                  f" | {_film_elapsed/_film_pairs:.2f}s/pair")
        else:
            frames_up = _apply_upsample_dis_hq(frames, n)
        tu1 = time.perf_counter()
        t_upsample_total += tu1 - tu0

        # For FILM: average n_for_gen for logging only (event timestamps use film_ts_abs).
        if upsample_method == "film" and len(frames) > 1:
            n_for_gen = (len(frames_up) - 1) // (len(frames) - 1) - 1
            if start == 0:  # log once per clip
                eff_fps = native_fps * (n_for_gen + 1)
                print(f"[INFO] film upsample: n_actual={n_for_gen}  "
                      f"effective_fps={eff_fps:.1f}  (target was ~{native_fps * (n + 1):.1f}fps)  "
                      f"timestamps from FILM (exact per-pair)")
        else:
            n_for_gen = n

        # ── Event generation ──────────────────────────────────────────────────
        te0 = time.perf_counter()
        native_start_us = start * us_per_frame
        chunk_xs, chunk_ys, chunk_ts, chunk_ps = gen_events_linear(
            frames_up, c_pos, c_neg,
            native_start_us, us_per_frame, n_for_gen,
            frame_ts_us=film_ts_abs)
        te1 = time.perf_counter()
        append_h5_chunk(h5f, chunk_xs, chunk_ys, chunk_ts, chunk_ps)

        t_eventgen_total += te1 - te0
        n_chunks += 1

    if _vid_cap is not None:
        _vid_cap.release()

    tw0 = time.perf_counter()
    n_total = int(h5f["t"].shape[0])
    h5f.attrs["n_events"] = n_total
    h5f.close()

    # Safety net: if no chunks were read (e.g. unsupported codec like AV1) — don't poison
    # the output dir with a 0-event file that blocks future retries.
    if n_chunks == 0:
        tmp_path.unlink(missing_ok=True)
        print(f"[FAIL] {source_path.name} — 0 chunks read, no output written "
              f"(unsupported codec or empty video)")
        return

    tmp_path.rename(out_path)
    tw1 = time.perf_counter()

    print(f"[DONE] {source_path.name} → {n_total:,} events → {out_path}")
    print(f"  peak_rss={_peak_rss_mb():.0f}MB")
    print(f"  chunks={n_chunks}  load={t_load_total:.2f}s  upsample={t_upsample_total:.2f}s"
          f"  eventgen={t_eventgen_total:.2f}s  write={tw1-tw0:.2f}s")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    output_root = args.output or cfg.get("output_root")
    if not output_root:
        sys.exit("No output root: pass --output or set output_root in the config")
    convert_one(args.source_path, cfg, output_root, args.tmpdir)


if __name__ == "__main__":
    main()
