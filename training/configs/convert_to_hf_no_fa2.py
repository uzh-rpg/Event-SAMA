#!/usr/bin/env python3
"""Wrapper around convert_to_hf_st.py that forces eager attention (this
cluster's flash_attn is 2.0.4, too old for FA2) before any model loading
happens -- same patch train_no_fa2_pinned.py applies for training."""
import sys

print("[PATCH] Disabling Flash Attention 2...", flush=True)
from transformers.modeling_utils import PreTrainedModel


@classmethod
def _force_eager_attn_implementation(cls, config, *args, **kwargs):
    if hasattr(config, '_attn_implementation_internal'):
        config._attn_implementation_internal = 'eager'
    config._attn_implementation = 'eager'
    return config


PreTrainedModel._autoset_attn_implementation = _force_eager_attn_implementation
PreTrainedModel._check_and_enable_flash_attn_2 = classmethod(
    lambda cls, config, *a, **kw: config
)
print("[PATCH] Attention implementation forced to eager (no FA2)", flush=True)

sys.argv[0] = 'convert_to_hf_st.py'
from projects.llava_sam2.hf.convert_to_hf_st import main

main()
