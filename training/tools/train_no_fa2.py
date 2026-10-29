#!/usr/bin/env python3
"""Wrapper script that disables Flash Attention 2 and runs training"""
import sys
import os

# Add PYDEPS to path FIRST
pydeps = os.environ.get('PYDEPS', '/iopsstor/scratch/cscs/sfedorchenko/event_sama_pydeps')
sys.path.insert(0, pydeps)

# Patch transformers BEFORE any imports to force eager attention (no Flash Attention 2)
print("[PATCH] Disabling Flash Attention 2...")
from transformers.modeling_utils import PreTrainedModel

# Override _autoset_attn_implementation to force eager attention.
# The original patch only skipped the FA2 version check but left
# config._attn_implementation = "flash_attention_2" intact, so models
# were still instantiated with Qwen2FlashAttention2 and then crashed
# at forward time when flash_attn_func / _flash_supports_window_size
# were not defined (flash_attn 2.0.4 < required 2.1.0).
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
print("[PATCH] ✅ Attention implementation forced to eager (no FA2)")

# Now run the actual training
sys.argv[0] = 'train.py'
from xtuner.tools import train
train.main()
