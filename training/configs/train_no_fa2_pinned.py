#!/usr/bin/env python3
"""Wrapper script that:
  1. Pins each rank to its own GPU via torch.cuda.set_device(LOCAL_RANK)
     -- MUST happen before anything touches CUDA, otherwise all ranks
        end up on cuda:0 and OOM the GH200.
  2. Forces transformers to use eager attention (FA2 is too old here).
  3. Hands off to xtuner.tools.train.main().
"""
import sys
import os

# Add PYDEPS to path FIRST
pydeps = os.environ.get('PYDEPS', '/iopsstor/scratch/cscs/sfedorchenko/event_sama_pydeps')
sys.path.insert(0, pydeps)

# Make the V2 event-dataset sidecar module importable from the config
# WITHOUT polluting the config namespace (mmengine deepcopies the config
# and chokes on captured `sys` module references). Doing it here keeps the
# config-side import a clean `from event_voxel_dataset_v2 import ...`.
_V2_DIR = os.environ.get(
    'EVENT_V2_DIR',
    '/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs')
if _V2_DIR not in sys.path:
    sys.path.insert(0, _V2_DIR)

# ── PIN CUDA DEVICE EARLY ───────────────────────────────────────────────
# Whatever launcher we use (torchrun OR srun --gpus-per-task=1), we want
# the very first CUDA allocation to land on the correct physical GPU.
# mmengine eventually calls set_device(local_rank) inside its init_dist,
# but that's late (after a bunch of model construction / NCCL bootstrap
# has already touched cuda:0).
#
# When running under srun --gpus-per-task=1, SLURM puts each task in its
# own cgroup with one visible GPU; LOCAL_RANK should be 0 so that
# mmengine's later set_device(LOCAL_RANK) maps to the only visible
# device. When running under torchrun --nproc_per_node=N, LOCAL_RANK is
# 0..N-1 and all GPUs are visible.
_local_rank_str = os.environ.get('LOCAL_RANK')
if _local_rank_str is not None:
    _local_rank = int(_local_rank_str)
    import torch
    if torch.cuda.is_available():
        _ndev = torch.cuda.device_count()
        # If the cgroup hides everything but one GPU, the only valid
        # device index is 0, regardless of what LOCAL_RANK says.
        _target = _local_rank if _local_rank < _ndev else 0
        torch.cuda.set_device(_target)
        print(f"[PATCH] rank={os.environ.get('RANK', '?')} "
              f"local_rank={_local_rank} "
              f"-> cuda:{torch.cuda.current_device()} "
              f"(of {_ndev} visible GPU(s), "
              f"CUDA_VISIBLE_DEVICES='{os.environ.get('CUDA_VISIBLE_DEVICES', '')}')",
              flush=True)

# Patch transformers BEFORE any other imports to force eager attention.
print("[PATCH] Disabling Flash Attention 2...", flush=True)
from transformers.modeling_utils import PreTrainedModel

@classmethod
def _force_eager_attn_implementation(cls, config, *args, **kwargs):
    """Force all models to use eager (standard) attention, bypassing FA2."""
    if hasattr(config, '_attn_implementation_internal'):
        config._attn_implementation_internal = 'eager'
    config._attn_implementation = 'eager'
    return config

PreTrainedModel._autoset_attn_implementation = _force_eager_attn_implementation
PreTrainedModel._check_and_enable_flash_attn_2 = classmethod(
    lambda cls, config, *a, **kw: config
)
print("[PATCH] Attention implementation forced to eager (no FA2)", flush=True)

# ── PATCH: skip mmengine's "Advance dataloader N steps" loop on resume ───
# By default, mmengine's IterBasedTrainLoop.run() actually pulls N batches
# from the dataloader on resume to step the sampler RNG forward. With our
# heavy event H5 reads + virtual ChatUniVi clips, each "advance step" costs
# real I/O (5-10s/batch), so a 3000-iter resume burns ~1h of GPU idle time
# doing nothing. The chain of 3 jobs in our sbatch makes it worse: job 2
# advances 6k, job 3 advances 9k. Replace the advance loop with a no-op.
# Cost: the data ordering inside the resumed segment will not exactly
# replay; with shuffled multi-source ConcatDataset this is negligible and
# does not affect training correctness.
print("[PATCH] Disabling mmengine's slow dataloader-advance on resume...",
      flush=True)
import logging as _logging
from mmengine.runner.loops import IterBasedTrainLoop as _IBL
from mmengine.logging import print_log as _print_log

def _patched_run(self):
    """Drop-in replacement for IterBasedTrainLoop.run that skips the
    on-resume dataloader replay. Body otherwise identical to upstream."""
    self.runner.call_hook('before_train')
    self.runner.call_hook('before_train_epoch')
    if self._iter > 0:
        _print_log(
            f'[PATCH] Skipping mmengine\'s {self._iter}-step dataloader '
            'advance on resume (would re-load that data through heavy '
            'event H5 reads). Sampling RNG will diverge from a no-resume '
            'run; training is still correct.',
            logger='current',
            level=_logging.WARNING,
        )
    while self._iter < self._max_iters and not self.stop_training:
        self.runner.model.train()
        data_batch = next(self.dataloader_iterator)
        self.run_iter(data_batch)
        self._decide_current_val_interval()
        if (self.runner.val_loop is not None
                and self._iter >= self.val_begin
                and (self._iter % self.val_interval == 0
                     or self._iter == self._max_iters)):
            self.runner.val_loop.run()
    self.runner.call_hook('after_train_epoch')
    self.runner.call_hook('after_train')
    return self.runner.model

_IBL.run = _patched_run
print("[PATCH] IterBasedTrainLoop.run patched (resume is now O(1))",
      flush=True)

sys.argv[0] = 'train.py'
from xtuner.tools import train
train.main()
