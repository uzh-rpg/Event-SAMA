#!/usr/bin/env python3
"""Diagnostic: construct VideoRegionDescriptionEventDataset exactly as the
live config does, then report a per-SOURCE breakdown of kept vs dropped
items (not just the aggregate total the dataset's own log line prints).
Answers: is SAV really the only excluded source, or is something else
being silently dropped?
"""
import sys, os, json

PROJECT_DIR = '/users/sfedorchenko/Event-SAMA-train'
CONFIGS_DIR = '/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs'
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, CONFIGS_DIR)

from mmengine.config import Config
from xtuner.registry import BUILDER

cfg = Config.fromfile(os.path.join(CONFIGS_DIR, 'sama_4b_event_v3.py'))

# Build ONLY the sama239k_region_desc_event_dataset entry, standalone.
ds_cfg = cfg.sama239k_region_desc_event_dataset

def src_of(fn):
    for s in ['lvvis', 'mevis', 'ref_youtube_vos', 'VidSTG', 'sav']:
        if s in fn:
            return s
    return 'OTHER:' + fn[:40]

# Patch _prefilter_event_items to also record per-source kept/dropped
# before we build, by monkeypatching the method on the class.
import event_voxel_dataset_v2 as ev2

orig_prefilter = ev2._SAMA239KEventMixin._prefilter_event_items

def instrumented_prefilter(self):
    from collections import Counter
    kept_c, dropped_c = Counter(), Counter()
    kept = []
    for item in self.text_data:
        fn = item.get('file_name', '')
        s = src_of(fn)
        h5 = self._event_h5_path(fn)
        if h5 and os.path.isfile(h5):
            kept.append(item)
            kept_c[s] += 1
        else:
            dropped_c[s] += 1
    print(f"[DIAG] {type(self).__name__} pre-filter per-source breakdown:", flush=True)
    all_sources = set(kept_c) | set(dropped_c)
    for s in sorted(all_sources):
        k, dd = kept_c[s], dropped_c[s]
        print(f"[DIAG]   {s}: kept {k} / {k+dd}", flush=True)
    self.text_data = kept

ev2._SAMA239KEventMixin._prefilter_event_items = instrumented_prefilter

print("[DIAG] Building sama239k_region_desc_event_dataset ...", flush=True)
ds = BUILDER.build(ds_cfg)
print(f"[DIAG] Final dataset size: {len(ds.text_data)}", flush=True)
