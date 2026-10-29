#!/usr/bin/env python3
"""Smoke test for the NEW V3 event datasets: build each and load a few
samples. Skips the model and the (slow) ChatUniVi meta-cache build.

Run:
    python smoke_event_v3.py [name1 name2 ...]
With no args it tests all six new datasets.
"""
import os
import sys
import time
import random
import traceback

PROJECT_DIR = '/users/sfedorchenko/Event-SAMA-train'
V3_DIR = '/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs'
# Deps are baked into the container image now (no PYDEPS overlay).
sys.path.insert(0, PROJECT_DIR)
sys.path.insert(0, V3_DIR)

from mmengine import Config


def main():
    print("[smoke] importing V3 dataset module...", flush=True)
    import event_voxel_dataset_v2 as v2  # noqa
    print("[smoke] OK", flush=True)

    cfg_path = os.path.join(V3_DIR, 'sama_4b_event_v3.py')
    print(f"[smoke] loading config {cfg_path}", flush=True)
    cfg = Config.fromfile(cfg_path)

    from xtuner.registry import BUILDER

    name_dict = dict(
        revos=cfg.video_revos_event_dataset,
        refytvos=cfg.video_refytvos_event_dataset,
        sav=cfg.video_sav_event_dataset,
        grandf=cfg.grandf_gcg_event_dataset,
        flickr=cfg.flickr_gcg_event_dataset,
        psg=cfg.psg_gcg_event_dataset,
        refcocog_gcg=cfg.refcocog_gcg_event_dataset,
        llava=cfg.llava_event_dataset,
        # previously untested in smoke but present in the full train mix:
        mevis=cfg.video_mevis_event_dataset,
        lvvis=cfg.video_lvvis_event_dataset,
        chatunivi=cfg.video_chatunivi_event_dataset,
        refcoco=cfg.refcoco_event_dataset,
        refcoco_plus=cfg.refcoco_plus_event_dataset,
        refcocog=cfg.refcocog_event_dataset,
    )

    wanted = sys.argv[1:] or list(name_dict.keys())
    tested, failed = [], []

    for name in wanted:
        dcfg = name_dict[name]
        print(f"\n[smoke] === {name} ===", flush=True)
        t0 = time.time()
        try:
            ds = BUILDER.build(dcfg)
            n = len(ds)
            print(f"[smoke] {name}: built in {time.time()-t0:.1f}s, len={n}",
                  flush=True)
            for _ in range(3):
                idx = random.randint(0, max(0, n - 1))
                ti = time.time()
                sample = ds[idx]
                pv = sample.get('pixel_values')
                gv = sample.get('g_pixel_values')
                masks = sample.get('masks')
                ids = sample.get('input_ids')
                pv_shape = tuple(pv.shape) if pv is not None else None
                if isinstance(gv, (list, tuple)):
                    gv_shape = f'list[{len(gv)}]'
                elif gv is not None:
                    gv_shape = tuple(gv.shape)
                else:
                    gv_shape = None
                m_shape = tuple(masks.shape) if hasattr(masks, 'shape') else (
                    f'list[{len(masks)}]' if masks is not None else None)
                n_ids = len(ids) if ids is not None else None
                print(f"[smoke] {name}[{idx}] pv={pv_shape} g={gv_shape} "
                      f"masks={m_shape} n_input_ids={n_ids} "
                      f"({time.time()-ti:.1f}s)", flush=True)
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
