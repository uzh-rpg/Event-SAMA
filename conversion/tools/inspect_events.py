#!/usr/bin/env python3
"""
Inspect converted event .h5 files: event counts, events per pixel per second
(ev/px/s, 2 s cap — the metric in the paper's Table 1) and temporal/spatial statistics.

Usage:
  # Single clip
  python tools/inspect_events.py /events/mevis/train/<clip_id>/events.h5

  # Full dataset directory
  python tools/inspect_events.py /events/mevis/ -j 32

  # Via config (events under <output-root>/<dataset>), 200 random clips
  python tools/inspect_events.py --config configs/mevis.yaml --output-root /events --samples 200
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np
import yaml


# ── Helpers ───────────────────────────────────────────────────────────────────

def fmt(n: float) -> str:
    if n >= 1e9:  return f"{n/1e9:.2f}B"
    if n >= 1e6:  return f"{n/1e6:.2f}M"
    if n >= 1e3:  return f"{n/1e3:.1f}K"
    return str(int(n))


KNOWN_DATASETS = {"mevis", "lvvis", "revos", "activitynet", "sav", "refytvos",
                  "coco", "llava", "grandf"}

def infer_dataset(path: Path) -> str:
    parts = [p.lower() for p in path.parts]
    for name in KNOWN_DATASETS:
        if any(name in part for part in parts):
            return name
    return "unknown"


WINDOW_S          = 10.0        # seconds of events to load per clip
MAX_WINDOW_EVENTS = 5_000_000   # hard cap on events loaded into RAM


def load_events(h5_path: Path) -> dict:
    with h5py.File(h5_path, "r") as f:
        n_stored = int(f.attrs.get("n_events", f["t"].shape[0]))
        t_unit   = str(f.attrs.get("t_unit", "ns"))
        mul      = 1000 if t_unit == "us" else 1

        # Total duration from first/last event — 2 scalar reads, negligible cost
        t_first = int(f["t"][0])  * mul
        t_last  = int(f["t"][-1]) * mul
        total_duration_s = max((t_last - t_first) / 1e9, 1e-6)

        # Load the first WINDOW_S seconds, strided if needed to stay under MAX_WINDOW_EVENTS
        frac     = min(1.0, WINDOW_S / total_duration_s)
        n_window = max(1, int(n_stored * frac))
        stride   = max(1, n_window // MAX_WINDOW_EVENTS)
        sl       = slice(0, n_window, stride)

        t_raw = f["t"][sl].astype(np.int64) * mul
        window_s = (t_raw[-1] - t_raw[0]) / 1e9 if len(t_raw) > 1 else 0.0

        return {
            "x":                f["x"][sl],
            "y":                f["y"][sl],
            "t":                t_raw,
            "p":                f["p"][sl],
            "native_h":         int(f.attrs.get("native_height", 0)),
            "native_w":         int(f.attrs.get("native_width",  0)),
            "native_fps":       float(f.attrs.get("native_fps",  0)),
            "n_events":         n_stored,          # exact from attr
            "total_duration_s": total_duration_s,  # exact from first/last t
            "window_s":         window_s,          # loaded window duration
            "stride":           stride,
        }


# ── Per-clip metrics ──────────────────────────────────────────────────────────

def compute_metrics(ev: dict) -> dict:
    t = ev["t"].astype(np.int64)
    p = ev["p"].astype(np.int8)
    x = ev["x"].astype(np.int16)
    y = ev["y"].astype(np.int16)
    n = len(t)

    if n == 0:
        return {"n_events": 0, "empty": True, "eps_mean": -1.0, "eps_std": -1.0}

    h, w       = ev["native_h"], ev["native_w"]
    native_fps = ev["native_fps"]

    # Use total clip duration from attrs (exact), not window t range
    duration_s = ev.get("total_duration_s", (t[-1] - t[0]) / 1e9)

    # ── Polarity ──────────────────────────────────────────────────────────────
    pos_mask  = p > 0
    n_pos     = int(pos_mask.sum())
    n_neg     = n - n_pos
    pol_bal   = n_pos / n   # fraction ON events

    # ── Spatial coverage ──────────────────────────────────────────────────────
    if h > 0 and w > 0:
        yc = np.clip(y, 0, h - 1)
        xc = np.clip(x, 0, w - 1)
        grid = np.zeros((h, w), dtype=bool)
        grid[yc, xc] = True
        active_px   = int(grid.sum())
        spatial_cov = active_px / (h * w)
    else:
        yc = xc = None
        active_px   = -1
        spatial_cov = -1.0

    # ── Temporal: per-frame distribution ──────────────────────────────────────
    # Bucket events into frame intervals using native_fps.
    # Each bucket covers one inter-frame interval (1/fps seconds).
    # This reveals burstiness, front-loading (first_times bug), and uniformity.
    if native_fps > 0 and duration_s > 0:
        ns_per_frame = 1e9 / native_fps
        # Frame interval index relative to first event
        frame_idx = np.floor((t - t[0]) / ns_per_frame).astype(np.int64)
        n_frames  = max(1, int(frame_idx.max()) + 1)
        frame_idx = np.clip(frame_idx, 0, n_frames - 1)
        frame_counts = np.bincount(frame_idx, minlength=n_frames).astype(np.float64)

        mean_epf  = float(frame_counts.mean())
        std_epf   = float(frame_counts.std())
        cv_epf    = std_epf / mean_epf if mean_epf > 0 else 0.0

        # First-frame fraction vs fair share
        first_frame_n   = int(frame_counts[0])
        first_frame_pct = first_frame_n / n * 100
        fair_share_pct  = 100.0 / n_frames
        first_frame_ratio = first_frame_pct / fair_share_pct if fair_share_pct > 0 else 0.0

        # Busiest frame
        max_frame_idx = int(frame_counts.argmax())
        max_frame_pct = float(frame_counts.max()) / n * 100

        # Polarity balance in first frame interval only
        ff_mask = frame_idx == 0
        n_ff    = int(ff_mask.sum())
        pol_first_frame = float((p[ff_mask] > 0).sum()) / n_ff if n_ff > 0 else -1.0

        # events per pixel per frame — resolution+fps-normalised density
        epf_density = n / (h * w * n_frames) if (h > 0 and w > 0) else -1.0

    else:
        n_frames = -1
        mean_epf = std_epf = cv_epf = -1.0
        first_frame_n = first_frame_pct = first_frame_ratio = -1.0
        max_frame_idx = max_frame_pct = -1.0
        pol_first_frame = -1.0
        epf_density = -1.0

    # ── Events per pixel per second (ev/px/s), capped at 2s ──────────────────
    # Spatial mean ± std of events/(pixel·s).
    # Cap to first 2s so short clips are comparable to long ones.
    # stride correction: pixel_counts from sampled events are ~stride× too low.
    stride = ev.get("stride", 1)
    EPS_CAP_S = 2.0
    if h > 0 and w > 0 and duration_s > 0:
        duration_cap = min(duration_s, EPS_CAP_S)
        t_cap_ns = t[0] + int(duration_cap * 1e9)
        mask_cap = t <= t_cap_ns
        x_cap = np.clip(x[mask_cap], 0, w - 1)
        y_cap = np.clip(y[mask_cap], 0, h - 1)
        pixel_counts = np.zeros((h, w), dtype=np.int64)
        np.add.at(pixel_counts, (y_cap, x_cap), 1)
        eps_map  = pixel_counts * stride / duration_cap   # events/(pixel·s), stride-corrected
        eps_mean = float(eps_map.mean())
        eps_std  = float(eps_map.std())
    else:
        eps_mean = eps_std = -1.0

    # ── IEI (inter-event interval) ────────────────────────────────────────────
    # Sample for speed on large clips. Gives a sense of event temporal spacing.
    if n > 1:
        sample = t if n <= 200_000 else t[np.random.choice(n, 200_000, replace=False)]
        sample = np.sort(sample)
        iei = np.diff(sample)
        iei_med_us  = float(np.median(iei))   / 1e3
        iei_p95_us  = float(np.percentile(iei, 95)) / 1e3
        iei_p99_us  = float(np.percentile(iei, 99)) / 1e3
    else:
        iei_med_us = iei_p95_us = iei_p99_us = -1.0

    # ── Sanity ────────────────────────────────────────────────────────────────
    monotonic          = bool(np.all(np.diff(t) >= 0))
    negative_timestamps = bool(t[0] < 0)

    n_total = n * stride   # stride-corrected true event count

    return {
        # identity
        "n_events":           n_total,
        "empty":              False,
        "native_h":           h,
        "native_w":           w,
        "native_fps":         native_fps,
        "duration_s":         duration_s,
        "t_min_ns":           int(t[0]),
        "stride":             stride,
        # polarity
        "n_pos":              n_pos * stride,
        "n_neg":              n_neg * stride,
        "pol_bal":            pol_bal,           # ratio — stride cancels
        # density
        "ev_per_s":           n_total / duration_s if duration_s > 0 else 0,
        "epf_density":        epf_density * stride if epf_density >= 0 else epf_density,
        # temporal distribution
        "n_frames":           n_frames,
        "mean_epf":           mean_epf * stride if mean_epf >= 0 else mean_epf,
        "std_epf":            std_epf  * stride if std_epf  >= 0 else std_epf,
        "cv_epf":             cv_epf,
        "first_frame_n":      first_frame_n,
        "first_frame_pct":    first_frame_pct,
        "first_frame_ratio":  first_frame_ratio,
        "max_frame_idx":      max_frame_idx,
        "max_frame_pct":      max_frame_pct,
        "pol_first_frame":    pol_first_frame,
        # IEI
        "iei_med_us":         iei_med_us,
        "iei_p95_us":         iei_p95_us,
        "iei_p99_us":         iei_p99_us,
        # spatial
        "spatial_cov":        spatial_cov,
        "active_px":          active_px,
        # sanity
        "monotonic":          monotonic,
        "negative_timestamps": negative_timestamps,
        # ev/px/s
        "eps_mean":           eps_mean,
        "eps_std":            eps_std,
    }


# ── Printing ──────────────────────────────────────────────────────────────────

SEP  = "─" * 72
SEP2 = "═" * 72


def print_clip(h5_path: Path, m: dict, verbose: bool):
    print(f"\n{SEP}")
    print(f"  {h5_path.parent.name}")
    print(SEP)

    if m.get("empty"):
        print("  (no events)")
        return

    h, w = m["native_h"], m["native_w"]
    print(f"  Resolution     : {w}×{h}  @  {m['native_fps']:.1f} fps  "
          f"({m['n_frames']} frame intervals)")
    print(f"  Duration       : {m['duration_s']:.2f} s")
    sampled_note = f"  [sampled 1/{m.get('stride',1)}]" if m.get("sampled") else ""
    print(f"  Total events   : {fmt(m['n_events'])}{sampled_note}  "
          f"(+{fmt(m['n_pos'])} / -{fmt(m['n_neg'])}  |  "
          f"{m['pol_bal']*100:.1f}% ON)")

    # Primary density metric — resolution and fps normalised
    epf = m["epf_density"]
    epf_str = f"{epf:.3f}" if epf >= 0 else "n/a"
    print(f"  Density        : {epf_str} ev/px/frame   ({fmt(m['ev_per_s'])}/s)")

    eps_mean = m.get("eps_mean", -1.0)
    eps_std  = m.get("eps_std",  -1.0)
    if eps_mean >= 0:
        print(f"  Ev/px/s (2s)   : mean={eps_mean:.2f}  std={eps_std:.2f}")

    # Temporal distribution
    cv   = m["cv_epf"]
    ff_r = m["first_frame_ratio"]
    ff_p = m["pol_first_frame"]
    print(f"  Temporal dist  : CV={cv:.2f}   "
          f"frame0={m['first_frame_pct']:.1f}% ({ff_r:.1f}× fair)   "
          f"frame0 ON%={ff_p*100:.1f}%" if ff_p >= 0 else
          f"  Temporal dist  : CV={cv:.2f}   frame0={m['first_frame_pct']:.1f}% ({ff_r:.1f}× fair)")

    if verbose:
        print(f"  Ev/frame       : mean={fmt(m['mean_epf'])}  "
              f"std={fmt(m['std_epf'])}  "
              f"max={m['max_frame_pct']:.1f}% at frame {m['max_frame_idx']}")
        print(f"  IEI            : med={m['iei_med_us']:.1f}µs  "
              f"p95={m['iei_p95_us']:.1f}µs  "
              f"p99={m['iei_p99_us']:.1f}µs")
        cov = m["spatial_cov"]
        if cov >= 0:
            print(f"  Spatial cov    : {cov*100:.1f}%  ({fmt(m['active_px'])} active pixels)")
        print(f"  Timestamps     : {'monotonic' if m['monotonic'] else 'not monotonic'}")


def print_summary(all_metrics: list[dict], dataset: str, method: str = ""):
    valid = [m for m in all_metrics if not m.get("empty") and m["n_events"] > 0]
    if not valid:
        print("\n[SUMMARY] No clips with events to summarise.")
        return

    counts  = np.array([m["n_events"]         for m in valid])
    epfs    = np.array([m["epf_density"]       for m in valid if m["epf_density"] >= 0])
    cvs     = np.array([m["cv_epf"]            for m in valid if m["cv_epf"] >= 0])
    ff_rats = np.array([m["first_frame_ratio"] for m in valid if m["first_frame_ratio"] >= 0])
    covs    = np.array([m["spatial_cov"]       for m in valid if m["spatial_cov"] >= 0])

    method_str = f"  method: {method}  |  " if method else "  "
    print(f"\n{SEP2}")
    print(f"  SUMMARY — {dataset.upper()}   "
          f"{method_str}"
          f"({len(all_metrics)} clips)")
    print(SEP2)

    print(f"  Events/clip    : "
          f"median {fmt(np.median(counts))}   "
          f"mean {fmt(np.mean(counts))}   "
          f"min {fmt(counts.min())}   "
          f"max {fmt(counts.max())}")

    if len(epfs):
        print(f"  Ev/px/frame    : "
              f"median {np.median(epfs):.3f}   "
              f"mean {np.mean(epfs):.3f}   "
              f"min {epfs.min():.3f}   "
              f"max {epfs.max():.3f}")

    if len(cvs):
        print(f"  Temporal CV    : "
              f"median {np.median(cvs):.2f}   "
              f"mean {np.mean(cvs):.2f}   "
              f"max {cvs.max():.2f}")

    if len(ff_rats):
        print(f"  First-frame ×  : "
              f"median {np.median(ff_rats):.1f}   "
              f"mean {np.mean(ff_rats):.1f}   "
              f"max {ff_rats.max():.1f}")

    if len(covs):
        print(f"  Spatial cov    : "
              f"median {np.median(covs)*100:.1f}%   "
              f"min {covs.min()*100:.1f}%   "
              f"max {covs.max()*100:.1f}%")

    # Events / (pixel · s), capped at 2s per clip
    eps_vals = np.array([m["eps_mean"] for m in valid if m.get("eps_mean", -1) >= 0])
    eps_std_vals = np.array([m["eps_std"] for m in valid if m.get("eps_std", -1) >= 0])
    if len(eps_vals):
        print(f"\n  ── Events per pixel per second (ev/px/s), 2s cap ─────────────────────")
        print(f"  Ev/px/s  mean  : "
              f"median {np.median(eps_vals):.3f}   "
              f"mean {np.mean(eps_vals):.3f}   "
              f"std {np.std(eps_vals):.3f}   "
              f"[across clips]")
        print(f"  Ev/px/s  std   : "
              f"median {np.median(eps_std_vals):.3f}   "
              f"mean {np.mean(eps_std_vals):.3f}   "
              f"[spatial std, median across clips]")

        # Percentile table
        pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        pct_vals = np.percentile(eps_vals, pcts)
        print(f"\n  Ev/px/s percentiles:")
        header = "  " + "  ".join(f"p{p:>3}" for p in pcts)
        values = "  " + "  ".join(f"{v:>5.2f}" for v in pct_vals)
        print(header)
        print(values)

        # ASCII histogram with log-spaced bins
        # Bins span from just below min to just above max in log space.
        # Clips with eps_mean=0 are counted separately.
        n_zero = int((eps_vals == 0).sum())
        pos_vals = eps_vals[eps_vals > 0]
        if len(pos_vals):
            lo = np.floor(np.log10(pos_vals.min()))
            hi = np.ceil(np.log10(pos_vals.max()))
            # At least 8 bins, at most 16
            n_bins = max(8, min(16, int(hi - lo) * 4))
            edges = np.logspace(lo, hi, n_bins + 1)
            counts_hist, _ = np.histogram(pos_vals, bins=edges)
            bar_max = counts_hist.max()
            bar_width = 30

            print(f"\n  Ev/px/s distribution  (log scale, {len(eps_vals)} clips"
                  + (f", {n_zero} at 0" if n_zero else "") + ")")
            print(f"  {'Range (ev/px/s)':<22}  {'Count':>5}  {'Bar'}")
            print(f"  {'─'*22}  {'─'*5}  {'─'*bar_width}")
            for i, cnt in enumerate(counts_hist):
                lo_e, hi_e = edges[i], edges[i + 1]
                bar = "█" * int(cnt / bar_max * bar_width) if bar_max > 0 else ""
                print(f"  [{lo_e:>7.2f} – {hi_e:>7.2f})  {cnt:>5}  {bar}")

    print(SEP2)


# ── Entry point ───────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="*", type=Path,
                   help="One or more .h5 files or directories (scanned recursively)")
    p.add_argument("--config", type=Path, default=None,
                   help="Dataset yaml config — derives output_root/dataset, method name")
    p.add_argument("--output-root", type=Path, default=None,
                   help="Events root to inspect with --config (default: output_root from the config)")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Print IEI, spatial coverage, per-frame detail")
    p.add_argument("--dataset", type=str, default=None,
                   help="Override dataset name (used in summary header only)")
    p.add_argument("--workers", "-j", type=int,
                   default=min(8, os.cpu_count() or 1),
                   help="Parallel workers (default: min(8, ncpus))")
    p.add_argument("--samples", "-n", type=int, default=None,
                   help="Randomly sample N clips instead of inspecting all")
    p.add_argument("--output-json", type=Path, default=None,
                   help="Write aggregate summary as JSON to this path")
    p.add_argument("--head", type=int, default=None, metavar="N",
                   help="Print per-clip detail for at most N clips (default: all). "
                        "Ignored when --summary-only. If --config has sanity_clips, "
                        "those are always printed regardless.")
    return p.parse_args()


def _process(h5_path: Path):
    ev = load_events(h5_path)
    return h5_path, compute_metrics(ev)


def _is_frame_dir(name: str) -> bool:
    return name.isdigit() or (name.startswith("frame_") and name[6:].isdigit())


def _find_h5_files(root: Path, out: list) -> None:
    """Collect events.h5 files using iterdir() — rglob is unreliable on Lustre.

    If a merged events.h5 exists alongside frame_XXXXXX / numeric subdirs
    (post-merge video_saccade or per_frame_saccade), skip those per-frame
    subdirs to avoid double-counting.
    """
    try:
        entries = list(root.iterdir())
    except (PermissionError, OSError):
        return
    has_merged = any(e.is_file() and e.name == "events.h5" for e in entries)
    for e in entries:
        if e.is_file() and e.name == "events.h5":
            out.append(e)
        elif e.is_dir():
            if has_merged and _is_frame_dir(e.name):
                continue
            _find_h5_files(e, out)


def collect_h5_files(paths: list[Path]) -> list[Path]:
    files = []
    for p in paths:
        if p.is_file() and p.suffix == ".h5":
            files.append(p)
        elif p.is_dir():
            _find_h5_files(p, files)
        else:
            print(f"[SKIP] {p} not found", file=sys.stderr)
    return sorted(files)


def _agg(vals, pct=(5, 25, 50, 75, 95)):
    """Aggregate stats dict for a list of floats."""
    a = np.array([v for v in vals if v >= 0])
    if not len(a):
        return {}
    return {
        "mean": float(a.mean()), "std": float(a.std()),
        "median": float(np.median(a)),
        **{f"p{p}": float(np.percentile(a, p)) for p in pct},
    }


def write_json(path: Path, all_metrics: list[dict], dataset: str, method: str,
               n_total: int, n_sampled: int):
    import json, datetime
    valid = [m for m in all_metrics if not m.get("empty") and m["n_events"] > 0]
    out = {
        "method":      method,
        "dataset":     dataset,
        "n_clips_sampled": n_sampled,
        "n_clips_total":   n_total,
        "timestamp":   datetime.datetime.now().isoformat(timespec="seconds"),
        "aggregate": {
            "events_per_pixel_s_mean": _agg([m.get("eps_mean", -1) for m in valid]),
            "events_per_pixel_s_std":  _agg([m.get("eps_std",  -1) for m in valid]),
            "polarity_balance":        _agg([m["pol_bal"]      for m in valid]),
            "events_per_clip":         _agg([m["n_events"]     for m in valid]),
            "ev_per_px_frame":         _agg([m["epf_density"]  for m in valid if m["epf_density"] >= 0]),
            "spatial_coverage":        _agg([m["spatial_cov"]  for m in valid if m["spatial_cov"] >= 0]),
            "temporal_cv":             _agg([m["cv_epf"]       for m in valid if m["cv_epf"] >= 0]),
            "n_empty_clips":           sum(1 for m in all_metrics if m.get("empty")),
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[JSON] {path}")


def main():
    args = parse_args()
    paths = list(args.paths)
    method = ""
    cfg = {}

    sanity_ids: set[str] = set()
    sanity_h5s: list[Path] = []
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        config_path = Path(args.output_root or cfg["output_root"]) / cfg["dataset"]
        paths.append(config_path)
        if args.dataset is None:
            args.dataset = cfg["dataset"]
        upsample  = "saccade" if cfg.get("mode") == "images" else cfg.get("upsample")
        method    = f"{upsample}+linear"
        for sc in cfg.get("sanity_clips", []):
            cid = Path(str(sc)).name  # bare clip dir name — used for line-712 match
            sanity_ids.add(cid)
            # 1. sc as relative path (handles "Charades/01ZWG" or bare "01ZWG")
            h5 = config_path / sc / "events.h5"
            if not h5.exists():
                for split in ("train", "val", "test", "valid", "valid_u"):
                    candidate = config_path / split / sc / "events.h5"
                    if candidate.exists():
                        h5 = candidate
                        break
            # 2. glob fallback — finds nested paths like train/subsplit/cid/events.h5
            if not h5.exists():
                matches = sorted(config_path.rglob(f"{cid}/events.h5"))
                if matches:
                    h5 = matches[0]
            if h5.exists():
                sanity_h5s.append(h5)

    if not paths:
        print("No paths given. Pass positional paths or --config.", file=sys.stderr)
        sys.exit(1)

    files = collect_h5_files(paths)

    if not files:
        print("No events.h5 files found.")
        sys.exit(1)

    total = len(files)
    if args.samples and args.samples < total:
        import random
        random.seed(42)
        files = random.sample(files, args.samples)
        # Always include sanity clips even if not in random sample
        sampled_set = set(files)
        for h5 in sanity_h5s:
            if h5 not in sampled_set:
                files.append(h5)
        print(f"Sampling {args.samples} of {total} files"
              + (f" + {len(sanity_h5s)} sanity clips\n" if sanity_h5s else "\n"))
    else:
        print(f"Found {total} event file(s)\n")

    groups: dict[str, list[Path]] = {}
    for f in files:
        ds = args.dataset or infer_dataset(f)
        groups.setdefault(ds, []).append(f)

    for dataset, ds_files in groups.items():
        results: dict = {}
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_process, p): p for p in ds_files}
            for future in as_completed(futures):
                try:
                    h5_path, m = future.result()
                    results[h5_path] = m
                except Exception as e:
                    print(f"[ERROR] {futures[future]}: {e}", file=sys.stderr)

        all_metrics = []
        for h5_path in ds_files:
            if h5_path not in results:
                continue
            m = results[h5_path]
            all_metrics.append(m)
            if h5_path.parent.name in sanity_ids:
                print_clip(h5_path, m, args.verbose)

        if len(ds_files) > 1:
            print_summary(all_metrics, dataset, method=method)

        if args.output_json:
            write_json(args.output_json, all_metrics, dataset, method,
                       n_total=total, n_sampled=len(ds_files))


if __name__ == "__main__":
    main()
