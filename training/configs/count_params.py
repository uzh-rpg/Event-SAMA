"""Count trainable vs frozen parameters of Event-SAMA, grouped by component.
Builds the model from the training config exactly as training does, but never
touches data or GPUs beyond construction."""
import sys, os
PROJECT_DIR = '/users/sfedorchenko/Event-SAMA-train'
CONFIGS_DIR = '/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs'
sys.path.insert(0, PROJECT_DIR); sys.path.insert(0, CONFIGS_DIR)

# Disable FlashAttention2 before any transformers model class is touched
# (env has flash_attn 2.0.4, below transformers' 2.1.0 minimum). Same patch
# train_no_fa2_pinned.py applies.
from transformers.modeling_utils import PreTrainedModel

@classmethod
def _force_eager(cls, config, *args, **kwargs):
    if hasattr(config, '_attn_implementation_internal'):
        config._attn_implementation_internal = 'eager'
    config._attn_implementation = 'eager'
    return config

PreTrainedModel._autoset_attn_implementation = _force_eager
PreTrainedModel._check_and_enable_flash_attn_2 = classmethod(lambda cls, config, *a, **kw: config)

from mmengine.config import Config
from xtuner.registry import BUILDER

cfg = Config.fromfile(os.path.join(CONFIGS_DIR, 'sama_4b_event_v3_fulldata.py'))
print('[cfg] freeze_llm=%s freeze_vit=%s frozen_sam2_decoder=%s llm_lora=%s' % (
    cfg.get('_freeze_llm'), cfg.get('_freeze_visual_encoder'),
    cfg.get('_frozen_sam2_decoder'), cfg.get('_llm_lora')), flush=True)

model = BUILDER.build(cfg.model)

# bucket every parameter by top-level component
buckets = {}
for n, p in model.named_parameters():
    if   'patch_embedding' in n:                     k = 'polarity stem (patch_embed)'
    elif 'event_adapter'   in n:                     k = 'event adapter'
    elif 'Qformer_final_proj' in n:                  k = 'Qformer_final_proj'
    elif 'Qformer' in n or 'spatial_Qformer' in n or 'temporal_Qformer' in n or 'vlm2spatial' in n:
        k = 'STC aggregator (Q-Former)'
    elif 'grounding_encoder' in n and 'sam_mask_decoder' in n: k = 'SAM2 mask decoder'
    elif 'grounding_encoder' in n and 'image_encoder' in n:    k = 'SAM2 image encoder'
    elif 'grounding_encoder' in n:                   k = 'SAM2 (other)'
    elif 'vision_model' in n:                        k = 'InternViT body'
    elif 'language_model' in n:                      k = 'LLM (Phi-3-mini)'
    elif 'mlp1' in n:                                k = 'mlp1'
    elif 'text_hidden_fcs' in n:                     k = 'text_hidden_fcs'
    else:                                            k = 'other: ' + n.split('.')[0]
    t, f = buckets.get(k, (0, 0))
    if p.requires_grad: t += p.numel()
    else:               f += p.numel()
    buckets[k] = (t, f)

TT = sum(t for t, _ in buckets.values()); FF = sum(f for _, f in buckets.values())
print('\n%-34s %14s %14s' % ('component', 'trainable', 'frozen'))
print('-' * 64)
for k in sorted(buckets, key=lambda x: -buckets[x][0]):
    t, f = buckets[k]
    print('%-34s %14s %14s' % (k, f'{t:,}' if t else '-', f'{f:,}' if f else '-'))
print('-' * 64)
print('%-34s %14s %14s' % ('TOTAL', f'{TT:,}', f'{FF:,}'))
print('\ntrainable = %.1fM   frozen = %.2fB   total = %.2fB   (%.2f%% trainable)'
      % (TT/1e6, FF/1e9, (TT+FF)/1e9, 100*TT/(TT+FF)))
