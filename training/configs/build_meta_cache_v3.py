#!/usr/bin/env python3
"""Pre-build the ChatUniVi event-meta cache for the V3 run.

V3 uses the full capstor/chatunivi conversion (~14.9k clips, dirs named by
the ChatUniVi video stem, e.g. ``v_k_ZXmr8pmrs``) instead of the empty
activitynet/train path the v2 config pointed at. Scans every events.h5,
reads only attrs + (t_min, t_max), and writes one JSON the trainer mmaps on
startup instead of re-scanning at every job launch (which would also race
across the 4 DDP ranks).
"""
import json
import os
import sys
import time

# Deps are baked into the container image now (/opt/sama_deps on PYTHONPATH);
# the old PYDEPS overlay is corrupted and must NOT be prepended.
PROJECT_DIR = '/users/sfedorchenko/Event-SAMA-train'
V3_DIR = '/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs'
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, V3_DIR)

EVENT_ROOT = '/capstor/scratch/cscs/mzehnde/sama_converted/chatunivi'
CACHE_PATH = '/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/cache/chatunivi_event_meta_v3.json'

from event_voxel_dataset_v2 import _peek_h5_meta  # noqa: E402

os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

if not os.path.isdir(EVENT_ROOT):
    sys.exit(f"event root not found: {EVENT_ROOT}")

stems = sorted(d for d in os.listdir(EVENT_ROOT)
               if os.path.isfile(os.path.join(EVENT_ROOT, d, 'events.h5')))
print(f"[meta-cache] scanning {len(stems)} event clips under {EVENT_ROOT}",
      flush=True)

# The scan is Lustre open-latency bound and h5py holds the GIL during reads,
# so threads don't overlap. Use a process pool (fork inherits the already
# imported heavy modules, so workers start instantly) for real parallelism.
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

WORKERS = int(os.environ.get('META_CACHE_WORKERS', '48'))


def _scan(stem):
    return stem, _peek_h5_meta(os.path.join(EVENT_ROOT, stem, 'events.h5'))


t0 = time.time()
cache = {}
done = 0
with ProcessPoolExecutor(max_workers=WORKERS) as ex:
    for stem, m in ex.map(_scan, stems, chunksize=8):
        if m is not None:
            cache[stem] = m
        done += 1
        if done % 1000 == 0 or done == len(stems):
            dt = time.time() - t0
            rate = done / max(dt, 1e-6)
            eta = (len(stems) - done) / max(rate, 1e-6)
            print(f"[meta-cache] {done}/{len(stems)} "
                  f"({rate:.1f} files/s, ETA {eta:.0f}s)", flush=True)

with open(CACHE_PATH, 'w') as f:
    json.dump(cache, f)

print(f"[meta-cache] wrote {len(cache)}/{len(stems)} entries to {CACHE_PATH} "
      f"({os.path.getsize(CACHE_PATH) / 1024:.1f} KiB)", flush=True)
