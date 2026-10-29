import sys, os
PROJECT_DIR='/users/sfedorchenko/Event-SAMA-train'; CONFIGS_DIR='/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs'
sys.path.insert(0,PROJECT_DIR); sys.path.insert(0,CONFIGS_DIR)
from transformers.modeling_utils import PreTrainedModel
@classmethod
def _f(cls,c,*a,**k):
    if hasattr(c,'_attn_implementation_internal'): c._attn_implementation_internal='eager'
    c._attn_implementation='eager'; return c
PreTrainedModel._autoset_attn_implementation=_f
PreTrainedModel._check_and_enable_flash_attn_2=classmethod(lambda cls,c,*a,**k:c)
from mmengine.config import Config
from xtuner.registry import BUILDER
cfg=Config.fromfile(os.path.join(CONFIGS_DIR,'sama_4b_event_v3_fulldata.py'))
m=BUILDER.build(cfg.model)
tr=[n for n,p in m.named_parameters() if p.requires_grad and 'image_encoder' in n]
fr=[n for n,p in m.named_parameters() if not p.requires_grad and 'image_encoder' in n]
print('image_encoder TRAINABLE sample:'); [print('  ',n) for n in tr[:4]]
print('image_encoder FROZEN sample:');    [print('  ',n) for n in fr[:4]]
print('counts: trainable=%d frozen=%d'%(len(tr),len(fr)))
print('distinct top-level prefixes (trainable):', sorted({n.split(".")[0] for n in tr}))
print('distinct top-level prefixes (frozen):   ', sorted({n.split(".")[0] for n in fr}))
