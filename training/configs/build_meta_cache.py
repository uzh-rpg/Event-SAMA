#!/usr/bin/env python3
"""Pre-build the ChatUniVi event-meta cache.

Scans every events.h5 under the configured folder, reads only attrs +
(t_min, t_max), and writes a single JSON the trainer can mmap on startup
instead of re-scanning 4817 files at every job launch.
"""
import json
import os
import sys
import time

PYDEPS = '/iopsstor/scratch/cscs/sfedorchenko/event_sama_pydeps'
PROJECT_DIR = '/users/sfedorchenko/Event-SAMA-train'
V2_DIR = '/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs'
sys.path.insert(0, PYDEPS)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, V2_DIR)

EVENT_ROOT = '/iopsstor/scratch/cscs/mzehnde/sama_converted/activitynet/train'
CACHE_PATH = '/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/cache/chatunivi_event_meta.json'

from event_voxel_dataset_v2 import _peek_h5_meta  # noqa: E402

os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

if not os.path.isdir(EVENT_ROOT):
    sys.exit(f"event root not found: {EVENT_ROOT}")

stems = sorted(d for d in os.listdir(EVENT_ROOT)
               if os.path.isfile(os.path.join(EVENT_ROOT, d, 'events.h5')))
print(f"[meta-cache] scanning {len(stems)} event clips under {EVENT_ROOT}",
      flush=True)

t0 = time.time()
cache = {}
for k, stem in enumerate(stems):
    h5 = os.path.join(EVENT_ROOT, stem, 'events.h5')
    m = _peek_h5_meta(h5)
    if m is not None:
        cache[stem] = m
    if (k + 1) % 200 == 0 or (k + 1) == len(stems):
        dt = time.time() - t0
        rate = (k + 1) / max(dt, 1e-6)
        eta = (len(stems) - (k + 1)) / max(rate, 1e-6)
        print(f"[meta-cache] {k + 1}/{len(stems)} "
              f"({rate:.1f} files/s, ETA {eta:.0f}s)", flush=True)

with open(CACHE_PATH, 'w') as f:
    json.dump(cache, f)

print(f"[meta-cache] wrote {len(cache)}/{len(stems)} entries to {CACHE_PATH} "
      f"({os.path.getsize(CACHE_PATH) / 1024:.1f} KiB)", flush=True)
