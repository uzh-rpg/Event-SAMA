#!/usr/bin/env python3
"""
Visualize source frames alongside accumulated events from a converted .h5 file.

Layout: 3 panels stacked vertically (beginning / middle / end of clip).
Each panel = 2 rows × (3 + 2×sub_bins) columns:

  Row 0 (source) : 3 consecutive native frames
                   sub-bin columns are dark (timestamp label only)
  Row 1 (events) : under each frame  → events ±delta around that frame's timestamp
                   between frames    → sub_bins windows spanning [t_frame+delta, t_next-delta]

  delta = half a native frame interval  (e.g. 83ms at 6fps)
  ON (+1) → blue    OFF (-1) → red    silent → gray

Usage:
  python visualize.py /path/to/clip_dir  /path/to/events.h5
  python visualize.py /path/to/video.mp4 /path/to/events.h5
  python visualize.py <source> <events.h5> --output out.png --sub-bins 3
"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ── I/O ───────────────────────────────────────────────────────────────────────

WINDOW_S          = 10.0       # seconds of events to load for visualization
MAX_WINDOW_EVENTS = 5_000_000  # hard cap on events loaded into RAM


def load_events(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        n_stored = int(f.attrs.get("n_events", f["t"].shape[0]))
        t_unit   = str(f.attrs.get("t_unit", "ns"))
        mul      = 1000 if t_unit == "us" else 1

        # Total duration from first/last event — 2 scalar reads
        t_first = int(f["t"][0])  * mul
        t_last  = int(f["t"][-1]) * mul
        total_duration_s = max((t_last - t_first) / 1e9, 1e-6)

        # Load first WINDOW_S seconds, strided if needed
        frac     = min(1.0, WINDOW_S / total_duration_s)
        n_window = max(1, int(n_stored * frac))
        stride   = max(1, n_window // MAX_WINDOW_EVENTS)
        sl       = slice(0, n_window, stride)

        t_raw = f["t"][sl].astype(np.int64) * mul
        window_s = (t_raw[-1] - t_raw[0]) / 1e9 if len(t_raw) > 1 else 0.0

        return {
            "x":                f["x"][sl].astype(np.int32),
            "y":                f["y"][sl].astype(np.int32),
            "t":                t_raw,
            "p":                f["p"][sl].astype(np.int8),
            "native_h":         int(f.attrs.get("native_height", 0)),
            "native_w":         int(f.attrs.get("native_width",  0)),
            "native_fps":       float(f.attrs.get("native_fps",  24.0)),
            "n_events":         n_stored,
            "total_duration_s": total_duration_s,
            "window_s":         window_s,
            "stride":           stride,
        }


def list_frames(source: Path) -> list:
    return sorted(source.glob("*.jpg")) or sorted(source.glob("*.JPG"))


def load_frames_batch(source: Path, indices: list) -> dict:
    """
    Load multiple frames by index. Returns {idx: bgr_array or None}.
    For video files: single sequential decode pass (no seeking) — avoids h264 mmco errors.
    For frame dirs: parallel imread.
    """
    needed = set(indices)
    if not needed:
        return {}

    if source.is_dir():
        jpgs = list_frames(source)
        return {i: cv2.imread(str(jpgs[i])) if i < len(jpgs) else None for i in needed}

    # Video: decode sequentially up to the highest needed frame — no cap.set() seeking
    max_idx = max(needed)
    cap = cv2.VideoCapture(str(source))
    result = {}
    i = 0
    while i <= max_idx:
        ok, frame = cap.read()
        if not ok:
            break
        if i in needed:
            result[i] = frame
        i += 1
    cap.release()
    return result


def count_frames(source: Path) -> int:
    if source.is_dir():
        return len(list_frames(source))
    cap = cv2.VideoCapture(str(source))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


# ── Event rendering ───────────────────────────────────────────────────────────

def render_events(ev: dict, t_start_ns: int, t_end_ns: int):
    """
    Accumulate signed polarity per pixel in [t_start_ns, t_end_ns].
    Normalise to [-1, 1]:  +1 (net ON) → blue   -1 (net OFF) → red   0 → gray
    """
    H, W = ev["native_h"], ev["native_w"]
    mask = (ev["t"] >= t_start_ns) & (ev["t"] <= t_end_ns)
    n_ev = int(mask.sum())
    if n_ev == 0:
        return np.full((H, W, 3), 128, dtype=np.uint8), 0

    frame = np.zeros((H, W), dtype=np.float32)
    np.add.at(frame,
              (np.clip(ev["y"][mask], 0, H - 1),
               np.clip(ev["x"][mask], 0, W - 1)),
              ev["p"][mask].astype(np.float32))

    peak = np.abs(frame).max()
    if peak > 0:
        frame /= peak

    img = np.full((H, W, 3), 128, dtype=np.uint8)
    pos = frame > 0
    neg = frame < 0
    # ON → blue
    img[pos] = np.stack([
        np.clip(128 - 128 * frame[pos], 0, 255),
        np.clip(128 - 128 * frame[pos], 0, 255),
        np.clip(128 + 127 * frame[pos], 0, 255),
    ], axis=-1).astype(np.uint8)
    # OFF → red
    img[neg] = np.stack([
        np.clip(128 - 127 * frame[neg], 0, 255),
        np.clip(128 + 128 * frame[neg], 0, 255),
        np.clip(128 + 128 * frame[neg], 0, 255),
    ], axis=-1).astype(np.uint8)
    return img, n_ev


# ── Helpers ───────────────────────────────────────────────────────────────────

def _style(ax, bg="#1a1a1a"):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_facecolor(bg)


def _ev_label(n: int) -> str:
    if n == 0:   return "0"
    if n < 1e3:  return str(n)
    if n < 1e6:  return f"{n/1e3:.0f}K"
    return f"{n/1e6:.1f}M"


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("source",       type=Path)
    p.add_argument("events_h5",    type=Path)
    p.add_argument("--output",     type=Path, default=None)
    p.add_argument("--sub-bins",   type=int,  default=3,
                   help="Sub-frame event bins between consecutive frames (default: 3)")
    p.add_argument("--delta-ms",  type=float, default=50.0,
                   help="Half-width of event window around each frame in ms (default: 50ms = 100ms total window).")
    return p.parse_args()


def main():
    args      = parse_args()
    clip_name = args.events_h5.parent.name
    out_path  = args.output or Path("vis") / f"{clip_name}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ev  = load_events(args.events_h5)
    fps = ev["native_fps"] if ev["native_fps"] > 0 else 24.0
    ns_per_frame = 1e9 / fps

    total_frames = count_frames(args.source)
    sub_bins     = args.sub_bins

    # Panels are shown within the loaded event window (WINDOW_S seconds).
    # Use window_frames so panel positions align with available events.
    window_s      = ev["window_s"]
    window_frames = min(total_frames, max(3, int(window_s * fps)))

    delta_ns = int(args.delta_ms * 1e6)

    print(f"Source : {total_frames} frames @ {fps:.1f}fps  ({total_frames/fps:.1f}s)")
    stride_info = f"  stride={ev['stride']}" if ev["stride"] > 1 else ""
    print(f"Events : {ev['n_events']:,} total  "
          f"(loaded {len(ev['t']):,} from first {window_s:.1f}s{stride_info})"
          f"  t=[{ev['t'].min():.0f}, {ev['t'].max():.0f}] ns")

    # ── Choose start frame for each of the 3 panels ───────────────────────────
    # Each panel shows 3 consecutive frames starting at panel_start.
    # Panels at beginning / middle / end; clamped so the 3rd frame exists.
    max_start    = max(0, window_frames - 3)
    panel_starts = [
        0,
        max_start // 2,
        max_start,
    ]
    # Deduplicate in case the clip is very short
    seen = []
    for s in panel_starts:
        if s not in seen:
            seen.append(s)
    panel_starts = seen
    n_panels = len(panel_starts)

    # ── Layout ────────────────────────────────────────────────────────────────
    # Columns per panel: [frame][sub×sub_bins][frame][sub×sub_bins][frame]
    cols = 3 + 2 * sub_bins   # 9 for sub_bins=3

    fig, axes = plt.subplots(
        n_panels * 2, cols,
        figsize=(cols * 2.8, n_panels * 5.2),
        gridspec_kw={"hspace": 0.45, "wspace": 0.03},
    )
    fig.patch.set_facecolor("#111111")

    # Ensure axes is always shape (n_panels*2, cols)
    axes = np.array(axes).reshape(n_panels * 2, cols)

    def ax(panel, row, col):
        return axes[panel * 2 + row, col]

    # ── Pre-load all needed source frames in one pass ─────────────────────────
    all_frame_idxs = [
        min(start_f + k, total_frames - 1)
        for start_f in panel_starts
        for k in range(3)
    ]
    frame_cache = load_frames_batch(args.source, all_frame_idxs)

    # ── Draw panels ───────────────────────────────────────────────────────────
    for pi, start_f in enumerate(panel_starts):
        frame_idxs = [min(start_f + k, total_frames - 1) for k in range(3)]
        frame_t_ns = [int(i * ns_per_frame) for i in frame_idxs]

        col = 0
        for fi in range(3):
            f_idx = frame_idxs[fi]
            t_ns  = frame_t_ns[fi]

            # ── Source frame (top row) ────────────────────────────────────────
            atop = ax(pi, 0, col)
            img  = frame_cache.get(f_idx)
            if img is not None:
                atop.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                _style(atop, bg="#111111")
            else:
                _style(atop, bg="#333333")
                atop.text(0.5, 0.5, "no frame", color="white",
                          ha="center", va="center",
                          transform=atop.transAxes, fontsize=7)
            atop.set_title(f"f#{f_idx}  t={f_idx/fps:.2f}s",
                           color="white", fontsize=7, pad=2)

            # ── Events ±delta around this frame (bottom row) ──────────────────
            abot = ax(pi, 1, col)
            ev_img, n_ev = render_events(ev, t_ns - delta_ns, t_ns + delta_ns)
            abot.imshow(ev_img)
            _style(abot)
            abot.set_title(
                f"[{(t_ns-delta_ns)/1e9:.3f}–{(t_ns+delta_ns)/1e9:.3f}s]  {_ev_label(n_ev)}ev",
                color="#aaaaaa", fontsize=6, pad=2)
            col += 1

            # ── Sub-bins between this frame and the next ──────────────────────
            # Centers evenly spaced between native frames; each window = ±delta_ns
            if fi < 2:
                t_next_ns = frame_t_ns[fi + 1]

                for b in range(sub_bins):
                    center = int(t_ns + (b + 1) * (t_next_ns - t_ns) / (sub_bins + 1))
                    ts = center - delta_ns
                    te = center + delta_ns

                    atop_b = ax(pi, 0, col)
                    _style(atop_b)
                    atop_b.set_title(f"sub {b+1}/{sub_bins}",
                                     color="#555555", fontsize=6, pad=2)

                    abot_b = ax(pi, 1, col)
                    ev_img_b, n_ev_b = render_events(ev, ts, te)
                    abot_b.imshow(ev_img_b)
                    _style(abot_b)
                    abot_b.set_title(
                        f"[{ts/1e9:.3f}–{te/1e9:.3f}s]  {_ev_label(n_ev_b)}ev",
                        color="#888888", fontsize=6, pad=2)
                    col += 1

        # Row labels
        ax(pi, 0, 0).set_ylabel(
            f"panel {pi+1}/{n_panels}\nf#{panel_starts[pi]}",
            color="white", fontsize=7,
        )
        ax(pi, 1, 0).set_ylabel("events", color="#aaaaaa", fontsize=7)

    # ── Legend & title ────────────────────────────────────────────────────────
    fig.legend(
        handles=[
            Patch(color=(0.2, 0.2, 1.0), label="ON  (+1)"),
            Patch(color=(1.0, 0.2, 0.2), label="OFF (−1)"),
        ],
        loc="lower right", fontsize=8,
        facecolor="#222222", labelcolor="white", framealpha=0.9,
    )
    windowed_note = f"  [showing first {window_s:.1f}s of {ev['total_duration_s']:.1f}s]" if window_s < ev["total_duration_s"] * 0.99 else ""
    fig.suptitle(
        f"{clip_name}  |  {ev['native_w']}×{ev['native_h']} @ {fps:.1f}fps  |  "
        f"{ev['total_duration_s']:.1f}s  |  {ev['n_events']/1e6:.2f}M events total  |  "
        f"delta=±{delta_ns/1e6:.0f}ms  sub_bins={sub_bins}{windowed_note}",
        color="white", fontsize=9, y=0.999,
    )

    plt.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
