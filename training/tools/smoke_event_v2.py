#!/usr/bin/env python3
"""Smoke test for Event-SAMA V2 dataset classes.

Instantiates each new dataset class with the same config used by the trainer
and pulls a single sample to verify end-to-end shape + tokenizer wiring.
Does NOT touch the model itself (that's covered by a real training run).

Usage:
    PYTHONPATH=/users/sfedorchenko/Event-SAMA-train:$PYTHONPATH \\
        python tools/smoke_event_v2.py
"""
import os
import sys
import traceback

# Make sure PYDEPS / project root are on sys.path before any project imports.
PYDEPS = os.environ.get(
    'PYDEPS', '/iopsstor/scratch/cscs/sfedorchenko/event_sama_pydeps')
PROJECT = os.environ.get(
    'PROJECT', '/users/sfedorchenko/Event-SAMA-train')
for p in (PYDEPS, PROJECT):
    if p and p not in sys.path:
        sys.path.insert(0, p)


def _build(builder_dict):
    """Build using xtuner's BUILDER (same as the real config path)."""
    from xtuner.registry import BUILDER
    return BUILDER.build(builder_dict)


def _summarize_sample(name, sample):
    if sample is None:
        print(f'  [{name}] sample is None')
        return
    keys = list(sample.keys()) if isinstance(sample, dict) else type(sample)
    print(f'  [{name}] keys: {keys}')
    if isinstance(sample, dict):
        for k in ('pixel_values', 'g_pixel_values', 'masks',
                  'input_ids', 'labels'):
            if k in sample:
                v = sample[k]
                shape = (
                    tuple(v.shape) if hasattr(v, 'shape')
                    else (len(v) if hasattr(v, '__len__') else type(v))
                )
                dtype = getattr(v, 'dtype', None)
                print(f'      {k}: shape={shape} dtype={dtype}')


def main():
    # Import the V2 config as a python module so we get the dict literals.
    from mmengine.config import Config
    cfg_path = os.path.join(
        PROJECT,
        'projects/llava_sam2/configs/sama_4b_event_v2.py')
    cfg = Config.fromfile(cfg_path)
    print(f'Loaded config from {cfg_path}')
    print(f'train_dataset: {len(cfg.train_dataset["datasets"])} sub-datasets')

    # Build only ONE instance of each unique dataset class so we don't waste
    # time re-loading the same JSON four times.
    seen_types = {}
    for sub in cfg.train_dataset['datasets']:
        t = sub['type']
        # ConcatDataset uses the dict-as-spec; resolve type to class name.
        type_name = t.__name__ if hasattr(t, '__name__') else str(t)
        # Distinguish refcoco / + / g by data_root.
        if 'data_root' in sub:
            type_name = f"{type_name}({os.path.basename(sub['data_root'])})"
        seen_types.setdefault(type_name, sub)

    print(f'\nUnique dataset specs to smoke-test: {sorted(seen_types)}')

    failures = []
    for name, spec in seen_types.items():
        print(f'\n=== {name} ===')
        try:
            ds = _build(dict(spec))   # copy to avoid BUILDER mutating
            print(f'  built. len={len(ds)}')
            sample = ds[0]
            _summarize_sample(name, sample)
        except Exception as e:
            traceback.print_exc()
            failures.append((name, repr(e)))

    print('\n========== Smoke test summary ==========')
    if failures:
        print(f'FAILED: {len(failures)} / {len(seen_types)}')
        for n, e in failures:
            print(f'  - {n}: {e}')
        sys.exit(1)
    else:
        print(f'OK: {len(seen_types)} datasets all built and yielded a sample.')
        sys.exit(0)


if __name__ == '__main__':
    main()
