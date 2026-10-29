#!/usr/bin/env python3
"""Smoke test: import V2 datasets, peek at virtual clip enumeration, and
load a single sample from each. Does NOT touch the model.
"""
import os
import sys
import time
import traceback

PYDEPS = '/iopsstor/scratch/cscs/sfedorchenko/event_sama_pydeps'
PROJECT_DIR = '/users/sfedorchenko/Event-SAMA-train'
V2_DIR = '/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs'
sys.path.insert(0, PYDEPS)
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, V2_DIR)

from mmengine import Config


def main():
    print("[smoke] importing V2 module...", flush=True)
    import event_voxel_dataset_v2 as v2  # noqa
    print("[smoke] OK", flush=True)

    cfg_path = '/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs/sama_4b_event_v2.py'
    print(f"[smoke] loading config {cfg_path}", flush=True)
    cfg = Config.fromfile(cfg_path)

    # We bypass mmengine's runner and just instantiate the four event
    # datasets directly. We need a minimal tokenizer fixture; reuse the
    # config's tokenizer dict.
    from xtuner.registry import BUILDER

    tested = []
    failed = []

    name_dict = dict(
        mevis=cfg.video_mevis_event_dataset,
        lvvis=cfg.video_lvvis_event_dataset,
        chatunivi=cfg.video_chatunivi_event_dataset,
        refcoco=cfg.refcoco_event_dataset,
    )

    for name, dcfg in name_dict.items():
        print(f"\n[smoke] === {name} ===", flush=True)
        t0 = time.time()
        try:
            ds = BUILDER.build(dcfg)
            n = len(ds)
            dt = time.time() - t0
            print(f"[smoke] {name}: built in {dt:.1f}s, len={n}", flush=True)

            # Pull a few random samples to make sure __getitem__ works and
            # doesn't blow up memory.
            import random
            for k in range(3):
                idx = random.randint(0, max(0, n - 1))
                ti = time.time()
                sample = ds[idx]
                dti = time.time() - ti
                pv = sample.get('pixel_values')
                shape = tuple(pv.shape) if pv is not None else None
                print(f"[smoke] {name}[{idx}] -> pixel_values{shape}  "
                      f"({dti:.1f}s)", flush=True)

            tested.append(name)
        except Exception as e:
            print(f"[smoke] {name} FAILED: {e!r}", flush=True)
            traceback.print_exc()
            failed.append(name)

    print(f"\n[smoke] PASSED: {tested}", flush=True)
    print(f"[smoke] FAILED: {failed}", flush=True)
    sys.exit(0 if not failed else 1)


if __name__ == '__main__':
    main()
