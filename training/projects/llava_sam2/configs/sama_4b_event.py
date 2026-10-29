"""SAMA-4B Event-Voxel Pipeline Training Config (MeViS only, V1).

This is the event analog of ``sama_4b_rgb.py``. Differences from the RGB config
are limited to:

  * MeViS data is loaded from the converted event HDF5 dump (one ``.h5`` per
    clip, schema documented in ``EventVoxel_Dataset.py``) instead of from
    JPEG frames. Voxels are 2-channel polarity histograms accumulated in a
    fixed window per event-frame (see ``event_fps`` below).
  * Only MeViS is included in the train mix for V1, because that is the only
    dataset converted by the prophesee pipeline so far. Other sources can be
    added by introducing the corresponding ``Video<Source>EventDataset`` class
    in ``EventVoxel_Dataset.py`` and an event_folder for that source.
  * The model carries an ``event_adapter`` (EventGPT-style event-language
    adapter, identity at init) inserted after ``mlp1`` in the ViT->LLM path.
  * The ViT patch-embedding is configured for ``num_channels=2`` and the
    pretrained 3-channel weights are adapted on load by averaging across the
    input channel.

V1 deliberately:
  * Does NOT touch SAM2. The grounding encoder still consumes a 3-channel
    input; the EventVoxel dataset emits a pseudo-RGB rendering of the voxel
    just for SAM2's image encoder. Real SAM2 retraining for events is V2.
  * Skips RGB image datasets (RefCOCO/Osprey/LLaVA/GLaMM). Those sources have
    no event analog yet.

Stage A vs Stage B is selected by the ``training_stage`` flag below:
  * 'A' : only ``patch_embedding`` and ``event_adapter`` are trainable. Light
          alignment pass on simple event<->caption pairs.
  * 'B' : full instruction tuning. STC aggregator + SAM2 decoder + event_adapter
          + LLM-LoRA are trainable.
"""

from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from torch.optim import AdamW
from transformers import AutoTokenizer

from xtuner.dataset import ConcatDataset
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE
from xtuner.dataset.map_fns import template_map_fn_factory

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss

from projects.llava_sam2.models.internvl_sama import InternVL_Slowfast
from projects.llava_sam2.models import VideoLLaVASAMModel_zero3, SAM2TrainRunner
from projects.llava_sam2.datasets import (
    VideoMeVISEventDataset,
    video_lisa_collate_fn,
)
from projects.llava_sam2.models.preprocess.image_resize import DirectResize

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
path = '/iopsstor/scratch/cscs/sfedorchenko/weights/Sa2VA-4B'
pretrained_pth = '/iopsstor/scratch/cscs/sfedorchenko/weights/Sa2VA-4B'

# Root containing one <clip_id>/events.h5 per MeViS clip, written by the
# prophesee pipeline in ~/Event-SAMA-new/prophesee_pipeline/scripts/convert.py.
# Layout: <root>/<clip_id>/events.h5 ('nested', the pipeline default).
mevis_event_root = '/capstor/scratch/cscs/mzehnde/events_prophesee/final/mevis'

template = "phi3_chat"
prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = 8192

batch_size = 2
accumulative_counts = 4
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1
warmup_ratio = 0.05

save_steps = 1000
save_total_limit = 10

special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>']

NUM_TEMPORAL_TOKENS = 32
NUM_SPATIAL_TOKENS = 32
WINDOW_SIZE = 512
WINDOW_STRIDE = 512
DENSE_LEN = 5
SPARSE_LEN = 32
MAX_CONV_LEN = 6
MAX_FRAMES = 32

# ── Event-specific knobs ──────────────────────────────────────────────
EVENT_FPS = 100              # 10 ms accumulation window per event-frame
EVENT_NORMALIZE = True       # asinh + per-clip z-score
NUM_INPUT_CHANNELS = 2       # ViT patch_embedding in_channels (was 3 for RGB)
USE_EVENT_ADAPTER = True     # add EventGPT-style adapter after mlp1
EVENT_ADAPTER_EXPANSION = 1.0
training_stage = 'B'         # 'A' = align only, 'B' = full instruction tuning

tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=path,
    trust_remote_code=True,
    padding_side='right')

extra_image_processor = dict(
    type=DirectResize,
    target_length=1024,
)

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
# In stage A we freeze almost everything; only patch_embedding + event_adapter
# train. In stage B we run full SAMA-style instruction tuning.
if training_stage == 'A':
    _freeze_llm = True
    _freeze_visual_encoder = True   # we only unfreeze patch_embedding via the runner
    _frozen_sam2_decoder = True
    _llm_lora = None
else:
    _freeze_llm = True
    _freeze_visual_encoder = True
    _frozen_sam2_decoder = False
    _llm_lora = None

model = dict(
    type=VideoLLaVASAMModel_zero3,
    special_tokens=special_tokens,
    frozen_sam2_decoder=_frozen_sam2_decoder,
    mllm=dict(
        type=InternVL_Slowfast,
        model_path=path,
        freeze_llm=_freeze_llm,
        window_size=WINDOW_SIZE,
        stride=WINDOW_STRIDE,
        freeze_visual_encoder=_freeze_visual_encoder,
        llm_lora=_llm_lora,
        special_tokens=special_tokens,
        # Event-specific knobs consumed by InternVL_Slowfast at the very end of
        # __init__ (after AutoModel.from_pretrained loads the RGB checkpoint,
        # we surgically replace patch_embedding and attach event_adapter).
        override_vision_in_channels=NUM_INPUT_CHANNELS,
        use_event_adapter=USE_EVENT_ADAPTER,
        event_adapter_expansion=EVENT_ADAPTER_EXPANSION,
    ),
    tokenizer=tokenizer,
    grounding_encoder=dict(
        type=SAM2TrainRunner,
    ),
    loss_mask=dict(
        type=CrossEntropyLoss,
        use_sigmoid=True,
        reduction='mean',
        loss_weight=2.0),
    loss_dice=dict(
        type=DiceLoss,
        use_sigmoid=True,
        activate=True,
        reduction='mean',
        naive_dice=True,
        eps=1.0,
        loss_weight=0.5),
    pretrained_pth=pretrained_pth,
    loss_sample_points=True,
    bs=batch_size,
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
# MeViS metadata: re-use the JPEG-pipeline expression/mask files. The
# image_folder is left pointing at JPEGImages so the parent's frame-name
# resolution still works; nothing actually opens the JPEGs because the event
# dataset overrides the loading.
SA2VA = '/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/sa2va_training/'
VIDEO = SA2VA + 'video_datas/'
data_root_mevis = VIDEO + 'mevis/train/'

video_mevis_event_dataset = dict(
    type=VideoMeVISEventDataset,
    image_folder=data_root_mevis + 'JPEGImages',
    expression_file=data_root_mevis + 'meta_expressions.json',
    mask_file=data_root_mevis + 'mask_dict.json',
    event_folder=mevis_event_root,
    event_fps=EVENT_FPS,
    event_normalize=EVENT_NORMALIZE,
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
    dense_len=DENSE_LEN,
    sparse_len=SPARSE_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
    max_frames=MAX_FRAMES,
)

train_dataset = dict(
    type=ConcatDataset,
    datasets=[
        video_mevis_event_dataset,
    ]
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=video_lisa_collate_fn)
)

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    loss_scale=dict(init_scale=65536, enabled=False),
    accumulative_counts=accumulative_counts,
    clip_grad=dict(max_norm=max_norm, norm_type=2),
)

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
custom_hooks = []

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        save_optimizer=True,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

visualizer = None
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
log_processor = dict(by_epoch=False)
