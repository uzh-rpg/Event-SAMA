#!/usr/bin/env python3
"""convert_to_hf_st.py + two Event-SAMA-specific fixes:

  1. Force eager attention (this cluster's flash_attn is 2.0.4, too old
     for FA2) -- same patch train_no_fa2_pinned.py applies for training.
  2. When the source ablation config has use_event_adapter=True, mlp1 was
     wrapped as nn.Sequential(original_mlp1, EventAdapter) at training
     time, so the trained state dict has nested keys
     (mlp1.0.0/1/3.*, mlp1.1.norm/fc1/fc2.*) that don't match the stock
     HF Sa2VAChatModel's flat mlp1.0/1/3.* Sequential. Rebuild the same
     wrapper on the HF model's mlp1 before load_state_dict so the keys
     line up. (The vision_model's 2-channel patch_embedding needs no such
     fix -- it's passed through directly from the already-adapted
     model.mllm.model.vision_model object, not rebuilt from HF defaults.)
"""
import argparse
import copy
import os
import os.path as osp

import torch

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

from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path


def parse_args():
    parser = argparse.ArgumentParser(description='toHF script (event-aware)')
    parser.add_argument('--config', help='config file name or path.')
    parser.add_argument('--pth_model', help='pth model file')
    parser.add_argument('--save_path', type=str, default='./work_dirs/hf_model')
    return parser.parse_args()


def main():
    args = parse_args()

    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    cfg = Config.fromfile(args.config)
    model = BUILDER.build(cfg.model)
    backend = get_file_backend(args.pth_model)

    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        with patch_fileio():
            state_dict = guess_load_checkpoint(args.pth_model)
    else:
        state_dict = guess_load_checkpoint(args.pth_model)
    model.load_state_dict(state_dict, strict=False)
    print(f'Load PTH model from {args.pth_model}')

    model._merge_lora()
    model.mllm.transfer_to_hf = True
    all_state_dict = model.all_state_dict()

    name_map = {'mllm.model.': '', '.gamma': '.g_weight', 'mllm.': ''}
    all_state_dict_new = {}
    for key in all_state_dict.keys():
        # model.mllm.event_adapter and model.mllm.model.mlp1[1] are the SAME
        # module object (attached once as a direct attribute for
        # trainable-param bookkeeping, once nested inside the mlp1 Sequential
        # wrapper) -- state_dict() emits it under both paths. The nested
        # mlp1.1.* copy is the one with a home in the HF model; drop this
        # duplicate top-level view rather than double-loading identical
        # tensors under a key nothing expects.
        if key.startswith('mllm.event_adapter.'):
            continue
        new_key = copy.deepcopy(key)
        for _text in name_map.keys():
            new_key = new_key.replace(_text, name_map[_text])
        all_state_dict_new[new_key] = all_state_dict[key]

    from projects.llava_sam2.hf.models.configuration_sa2va_chat import Sa2VAChatConfig
    from projects.llava_sam2.hf.models.modeling_sa2va_chat_all import Sa2VAChatModel

    internvl_config = Sa2VAChatConfig.from_pretrained(cfg.path)
    config_dict = internvl_config.to_dict()
    # Event-SAMA fix: cfg.path's config.json is the ORIGINAL RGB pretrained
    # config (vision_config.num_channels=3). The actual vision_model here
    # has already had its patch_embedding surgically replaced to accept
    # override_vision_in_channels (see internvl_sama.py
    # _adapt_patch_embedding_in_channels) -- read the real in_channels off
    # the live module so the saved config.json matches the saved weights.
    # Otherwise AutoModel.from_pretrained rebuilds a 3-channel conv from
    # config.json and then fails loading the saved 2-channel weight tensor.
    actual_in_channels = model.mllm.model.vision_model.embeddings.patch_embedding.in_channels
    config_dict['vision_config']['num_channels'] = actual_in_channels
    # Event-SAMA fix: tell the HF model class itself (via config) whether to
    # build the mlp1+EventAdapter wrapper, so a FRESH from_pretrained() load
    # (which reconstructs Sa2VAChatModel from config.json before loading
    # weights) builds the same module structure the trained state dict's
    # keys expect. Patching hf_sa2va_model.mlp1 only *here*, post-hoc, isn't
    # enough -- that structure isn't persisted anywhere from_pretrained()
    # reads, so a later reload silently drops the adapter's trained weights
    # and re-randomizes mlp1.1.
    use_event_adapter = hasattr(model.mllm, 'event_adapter')
    config_dict['use_event_adapter'] = use_event_adapter
    if use_event_adapter:
        adapter_hidden = model.mllm.event_adapter.norm.weight.shape[0]
        adapter_inner = model.mllm.event_adapter.fc1.out_features
        config_dict['event_adapter_expansion'] = adapter_inner / adapter_hidden
    config_dict['auto_map'] = \
        {'AutoConfig': 'configuration_sa2va_chat.Sa2VAChatConfig',
         'AutoModel': 'modeling_sa2va_chat_all.Sa2VAChatModel',
         'AutoModelForCausalLM': 'modeling_sa2va_chat_all.Sa2VAChatModel'}
    config_dict["llm_config"]["vocab_size"] = len(model.tokenizer)
    config_dict["template"] = cfg.template
    sa2va_hf_config = Sa2VAChatConfig(**config_dict)

    hf_sa2va_model = Sa2VAChatModel(
        sa2va_hf_config, vision_model=model.mllm.model.vision_model,
        language_model=model.mllm.model.language_model
    )
    if use_event_adapter:
        print(f"[Event SAMA] Sa2VAChatModel built with mlp1+EventAdapter "
              f"wrapper (config.use_event_adapter=True, expansion="
              f"{config_dict['event_adapter_expansion']}).", flush=True)

    hf_sa2va_model.load_state_dict(all_state_dict_new)

    hf_sa2va_model.save_pretrained(args.save_path)
    model.tokenizer.save_pretrained(args.save_path)
    print(f"Save the hf model into {args.save_path}")

    os.system(f"cp -pr ./projects/llava_sam2/hf/models/* {args.save_path}")


if __name__ == '__main__':
    main()
