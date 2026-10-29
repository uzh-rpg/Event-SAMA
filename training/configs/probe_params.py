"""Fine-grained param audit for Event-SAMA (paper TODO verification).

Prints per-prefix trainable/frozen counts (3 levels deep), tokenizer/[SEG]
handling, and whether the HF Sa2VaChatModel carries an inner SAM2 copy.
"""
import sys, os
PROJECT_DIR = '/users/sfedorchenko/Event-SAMA-train'
CONFIGS_DIR = '/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs'
sys.path.insert(0, PROJECT_DIR); sys.path.insert(0, CONFIGS_DIR)

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
model = BUILDER.build(cfg.model)

# ---- 1. per-prefix counts (3 name components) --------------------------
from collections import defaultdict
groups = defaultdict(lambda: [0, 0])
for n, p in model.named_parameters():
    key = '.'.join(n.split('.')[:3])
    groups[key][0 if p.requires_grad else 1] += p.numel()

print('\n=== per-prefix (depth 3): trainable / frozen ===')
for k in sorted(groups):
    t, f = groups[k]
    print('%-60s %14s %14s' % (k, f'{t:,}' if t else '-', f'{f:,}' if f else '-'))

# ---- 2. inner vs outer SAM2 -------------------------------------------
inner = [n for n, _ in model.named_parameters() if n.startswith('mllm.model.grounding_encoder')]
outer = [n for n, _ in model.named_parameters() if n.startswith('grounding_encoder')]
print('\n=== SAM2 copies ===')
print('inner (mllm.model.grounding_encoder) params:', len(inner))
print('outer (grounding_encoder) params:', len(outer))
inner_t = sum(p.numel() for n, p in model.named_parameters()
              if n.startswith('mllm.model.grounding_encoder') and p.requires_grad)
outer_t = sum(p.numel() for n, p in model.named_parameters()
              if n.startswith('grounding_encoder') and p.requires_grad)
print('inner trainable:', f'{inner_t:,}', ' outer trainable:', f'{outer_t:,}')
# also inner text_hidden_fcs?
inner_thf = sum(p.numel() for n, p in model.named_parameters()
                if 'mllm.model.text_hidden_fcs' in n)
outer_thf = sum(p.numel() for n, p in model.named_parameters()
                if n.startswith('text_hidden_fcs'))
print('inner text_hidden_fcs:', f'{inner_thf:,}', ' outer text_hidden_fcs:', f'{outer_thf:,}')

# ---- 3. mlp1 vs event adapter -----------------------------------------
print('\n=== mlp1 / adapter ===')
for n, p in model.named_parameters():
    if 'mlp1' in n:
        print('%-70s %12s requires_grad=%s' % (n, f'{p.numel():,}', p.requires_grad))

# ---- 4. [SEG] handling -------------------------------------------------
print('\n=== [SEG] / tokenizer ===')
tok = model.tokenizer
print('len(tokenizer):', len(tok))
print('seg_token_idx:', model.seg_token_idx)
try:
    emb = model.mllm.model.language_model.get_input_embeddings()
    print('embed_tokens shape:', tuple(emb.weight.shape), 'requires_grad:', emb.weight.requires_grad)
    out_emb = model.mllm.model.language_model.get_output_embeddings()
    print('lm_head shape:', tuple(out_emb.weight.shape), 'requires_grad:', out_emb.weight.requires_grad)
except Exception as e:
    print('embed check err:', e)
n_new = tok.add_tokens(['[SEG]', '<p>', '</p>', '<vp>', '</vp>'], special_tokens=True)
print('re-adding special tokens adds:', n_new, '(0 => already in vocab)')

# ---- 5. effective trained params (excl. inner dead copy) ---------------
total_t = sum(p.numel() for p in model.parameters() if p.requires_grad)
eff = total_t - inner_t
print('\ntotal requires_grad:', f'{total_t:,}')
print('effective (minus inner SAM2 copy):', f'{eff:,}')
