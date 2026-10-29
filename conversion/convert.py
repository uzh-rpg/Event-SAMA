#!/usr/bin/env python3
"""
Convert an RGB dataset to synthetic events with one dataset config.

  python convert.py --config configs/mevis.yaml --input /data/mevis --output /events
  python convert.py --config configs/mevis.yaml --input /data/mevis --output /events --workers 8 --eval eval/mevis

Items are discovered under --input (clip dirs, videos or images, depending on the config's
mode), or read from --joblist (one source path per line). Items whose events.h5 already
exists are skipped, so an interrupted run resumes by running the same command again.
Each item is converted in its own `python -m sama_converter.convert_clip` process, so a crash
(e.g. a TensorFlow abort in FILM) only fails that item and the run continues.

--input / --output default to source_root / output_root from the config.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import signal
import subprocess
import sys
import time

from sama_converter.convert_clip import load_config, output_path
from sama_converter.discover import discover_items

REPO_DIR = Path(__file__).resolve().parent
DEFAULT_EVAL = Path("__default__")  # --eval given without a path → default_eval_dir()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--config",  type=Path, required=True, help="Dataset YAML config")
    p.add_argument("--input",   type=Path, default=None,
                   help="Root of the RGB dataset (default: source_root from the config)")
    p.add_argument("--output",  type=Path, default=None,
                   help="Root for the event output (default: output_root from the config)")
    p.add_argument("--joblist", type=Path, default=None,
                   help="Convert exactly the items listed in this file (one path per line) "
                        "instead of discovering them under --input")
    p.add_argument("--workers", type=int, default=1, help="Items converted in parallel")
    p.add_argument("--limit",   type=int, default=None, help="Convert only the first N pending items")
    p.add_argument("--tmpdir",  type=Path, default=None,
                   help="Scratch dir for FILM intermediate frames (default: system tmpdir)")
    p.add_argument("--eval",    type=Path, nargs="?", const=DEFAULT_EVAL, default=None, metavar="EVAL_DIR",
                   help="After converting, write metrics.txt and viz/ for the sanity clips to EVAL_DIR "
                        "(without a path: ./eval/<dataset>/<method>_<timestamp>)")
    return p.parse_args()


def _run_item(source: Path, config: Path, output_root, tmpdir):
    """Convert one item in a separate process. Returns None on success, else an error string."""
    # Absolute paths: the worker runs from the repo dir so that `-m sama_converter...` resolves.
    cmd = [sys.executable, "-m", "sama_converter.convert_clip", str(Path(source).resolve()),
           "--config", str(Path(config).resolve()), "--output", str(Path(output_root).resolve())]
    if tmpdir:
        cmd += ["--tmpdir", str(Path(tmpdir).resolve())]
    rc = subprocess.run(cmd, cwd=REPO_DIR).returncode
    if rc == 0:
        return None
    if rc < 0:  # killed by a signal, e.g. SIGABRT from a TensorFlow fatal error
        return f"killed by {signal.Signals(-rc).name}"
    return f"exit code {rc}"


def main():
    sys.stdout.reconfigure(line_buffering=True)  # keep our lines in order with subprocess/worker output
    args = parse_args()
    cfg = load_config(args.config)

    source_root = args.input or cfg.get("source_root")
    output_root = args.output or cfg.get("output_root")
    if not output_root:
        sys.exit("No output root: pass --output or set output_root in the config")

    if args.joblist:
        items = [Path(l.strip()) for l in args.joblist.read_text().splitlines() if l.strip()]
        pending = [s for s in items if not output_path(s, cfg, output_root).exists()]
        print(f"[{cfg['dataset']}] joblist: {len(pending)} pending  |  {len(items) - len(pending)} already done")
    else:
        if not source_root:
            sys.exit("No input: pass --input, --joblist or set source_root in the config")
        pending = discover_items(cfg, source_root, output_root)

    if args.limit is not None:
        pending = pending[:args.limit]

    failed = []
    if pending:
        print(f"[{cfg['dataset']}] converting {len(pending)} item(s) with {args.workers} worker(s) "
              f"→ {Path(output_root) / cfg['dataset']}", flush=True)
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_item, s, args.config, output_root, args.tmpdir): s for s in pending}
            for i, fut in enumerate(as_completed(futures), 1):
                source, err = futures[fut], fut.result()
                status = "OK" if err is None else f"FAILED ({err})"
                print(f"[{cfg['dataset']}] {i}/{len(pending)} {status}: {source}", flush=True)
                if err is not None:
                    failed.append((source, err))
        print(f"[{cfg['dataset']}] finished in {time.time() - t0:.0f}s, {len(failed)} failed", flush=True)
        for source, err in failed:
            print(f"  FAILED {source}: {err}")

    if args.eval is not None:
        eval_dir = default_eval_dir(cfg) if args.eval == DEFAULT_EVAL else args.eval
        cmd = ["bash", str(REPO_DIR / "tools" / "run_eval.sh"), "--config", str(args.config),
               "--output", str(output_root), "--out-dir", str(eval_dir)]
        if source_root:
            cmd += ["--input", str(source_root)]
        print(f"[{cfg['dataset']}] running eval ...", flush=True)
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if res.returncode != 0:
            print(res.stdout)
            sys.exit(f"[{cfg['dataset']}] eval FAILED (exit code {res.returncode})")
        print(f"[{cfg['dataset']}] eval saved under {eval_dir}", flush=True)

    sys.exit(1 if failed else 0)


def default_eval_dir(cfg: dict) -> Path:
    """./eval/<dataset>/<method>_<timestamp>, relative to the working directory."""
    method = "saccade" if cfg["mode"] == "images" else cfg["upsample"]
    return Path("eval") / cfg["dataset"] / f"{method}+linear_{time.strftime('%Y%m%d_%H%M%S')}"


if __name__ == "__main__":
    main()
