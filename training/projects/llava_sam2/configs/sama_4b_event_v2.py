"""SAMA-4B Event Pipeline V2 -- train on every event source we have.

V2 differences vs V1 (`sama_4b_event.py`):
  * 4 event sources are mixed into one ConcatDataset (instead of MeViS-only):
      - MeViS              (video RES, segmentation expressions)
      - LVVIS train        (video VIS, category-as-expression)
      - ChatUniVi          (video QA, ActivityNet-converted)
      - RefCOCO/+/g        (single-image RES, COCO event clips)
    repeats roughly mirror the recipe from ``sama_4b_base.py`` (RGB).
  * EventVoxelTransform now uses asinh + a fixed scalar (no per-clip
    z-score), so identical inputs always produce identical outputs and
    empty event-frames stay exactly zero.
  * InternVL_Slowfast.image_flags is modality-aware (set to 1 for any
    non-3-channel input), so we never silently drop event frames.
  * VideoLLaVASAMModel.state_dict now persists the trained 2-channel
    ``patch_embedding`` and the ``event_adapter`` weights.
  * SAM2 checkpoint is loaded with strict=False so the image-only
    weights take effect even though the .pt is missing video-only memory
    layers.
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
    RefCOCOEventDataset,
    VideoChatUniViEventDataset,
    VideoLVVISEventDataset,
    VideoMeVISEventDataset,
    video_lisa_collate_fn,
)
from projects.llava_sam2.models.preprocess.image_resize import DirectResize

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
path = '/iopsstor/scratch/cscs/sfedorchenko/weights/Sa2VA-4B'
pretrained_pth = '/iopsstor/scratch/cscs/sfedorchenko/weights/Sa2VA-4B'

# ── Event roots (one events.h5 per clip) ─────────────────────────────
# These are READ-ONLY for the trainer; the conversion pipeline writes them.
mevis_event_root      = '/capstor/scratch/cscs/mzehnde/sama_converted/mevis'
lvvis_event_root      = '/capstor/scratch/cscs/mzehnde/sama_converted/lvvis/train'
activitynet_event_root = '/iopsstor/scratch/cscs/mzehnde/sama_converted/activitynet/train'
coco_event_root       = '/capstor/scratch/cscs/mzehnde/sama_converted/coco/train2014'

# ── RGB-side annotation files (the events use the same text/masks) ──
SAMA_DATASET = '/iopsstor/scratch/cscs/sfedorchenko/sama_dataset'
SA2VA = SAMA_DATASET + '/sa2va_training'
VIDEO = SA2VA + '/video_datas'
data_root_mevis = VIDEO + '/mevis/train/'
data_root_lvvis_anno = SAMA_DATASET + '/lvvis/train_instances.json'
data_root_chatunivi = VIDEO + '/chat_univi/'
data_root_refcoco = SA2VA + '/ref_seg/refcoco'
data_root_refcoco_plus = SA2VA + '/ref_seg/refcoco+'
data_root_refcocog = SA2VA + '/ref_seg/refcocog'
coco_train2014_jpg_root = SA2VA + '/osprey-724k/coco/train2014'

# ── Common prompt / size knobs ────────────────────────────────────────
template = "phi3_chat"
prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = 8192

# ── Optim ─────────────────────────────────────────────────────────────
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
EVENT_NORMALIZE = True       # asinh + fixed scalar (NOT per-clip z-score)
NUM_INPUT_CHANNELS = 2       # ViT patch_embedding in_channels (was 3 for RGB)
USE_EVENT_ADAPTER = True
EVENT_ADAPTER_EXPANSION = 1.0

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
# Trainable in V2 (matches sama_4b_base.py + Event-SAMA additions):
#   * STC aggregator (Qformers, projections)
#   * mlp1 (frozen by default in InternVL but the visual<->lang projection
#     is still gradient-enabled because freeze_visual_encoder=True only
#     freezes the ViT body)
#   * SAM2 mask decoder (frozen_sam2_decoder=False)
#   * text_hidden_fcs
#   * NEW: vision_model.embeddings.patch_embedding (2-ch conv)
#   * NEW: event_adapter (post-mlp1 LN+MLP residual)
# Frozen:
#   * ViT body (vision_model.encoder.*)
#   * LLM (freeze_llm=True, llm_lora=None) -- same as the RGB recipe.
_freeze_llm = True
_freeze_visual_encoder = True   # body frozen; patch_embedding unfrozen below
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
        # Event-SAMA: 3-ch -> 2-ch surgery + EventGPT-style adapter.
        override_vision_in_channels=NUM_INPUT_CHANNELS,
        use_event_adapter=USE_EVENT_ADAPTER,
        event_adapter_expansion=EVENT_ADAPTER_EXPANSION,
    ),
    tokenizer=tokenizer,
    grounding_encoder=dict(type=SAM2TrainRunner),
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

# ── (A) Video RES: MeViS (events) ────────────────────────────────────
video_mevis_event_dataset = dict(
    type=VideoMeVISEventDataset,
    image_folder=data_root_mevis + 'JPEGImages',  # used only for filename schema
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

# ── (B) Video VIS: LVVIS train (events) ──────────────────────────────
video_lvvis_event_dataset = dict(
    type=VideoLVVISEventDataset,
    image_folder=lvvis_event_root,         # used only for filename resolution
    expression_file=data_root_lvvis_anno,  # COCO-VIS schema, masks inlined
    mask_file=None,                        # masks live inside expression_file
    event_folder=lvvis_event_root,
    event_fps=EVENT_FPS,
    event_normalize=EVENT_NORMALIZE,
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=2,
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

# ── (C) Video QA: ChatUniVi (events) ─────────────────────────────────
video_chatunivi_event_dataset = dict(
    type=VideoChatUniViEventDataset,
    image_folder=data_root_chatunivi + 'Activity_Videos/',  # only used by parent for absent-MP4 fallback
    json_file=data_root_chatunivi + 'video_chat.json',
    event_folder=activitynet_event_root,
    event_fps=EVENT_FPS,
    event_normalize=EVENT_NORMALIZE,
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
    dense_len=DENSE_LEN,
    sparse_len=SPARSE_LEN,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
    max_frames=MAX_FRAMES,
)

# ── (D) Image RES: RefCOCO / RefCOCO+ / RefCOCOg (COCO events) ───────
def _refcoco_event_cfg(data_root, split_file):
    return dict(
        type=RefCOCOEventDataset,
        tokenizer=tokenizer,
        special_tokens=special_tokens,
        extra_image_processor=extra_image_processor,
        data_root=data_root,
        data_prefix=dict(img_path=coco_train2014_jpg_root),
        ann_file='instances.json',
        split_file=split_file,
        prompt_template=prompt_template,
        num_classes_per_sample=5,
        num_temporal_token=NUM_TEMPORAL_TOKENS,
        num_spatial_token=NUM_SPATIAL_TOKENS,
        window_size=WINDOW_SIZE,
        window_stride=WINDOW_STRIDE,
        max_length=max_length,
        # Event-specific knobs
        event_folder=coco_event_root,
        event_normalize=EVENT_NORMALIZE,
    )

refcoco_event_dataset = _refcoco_event_cfg(data_root_refcoco, 'refs(unc).p')
refcoco_plus_event_dataset = _refcoco_event_cfg(data_root_refcoco_plus, 'refs(unc).p')
refcocog_event_dataset = _refcoco_event_cfg(data_root_refcocog, 'refs(umd).p')

# ── Concat ───────────────────────────────────────────────────────────
# Mirror the RGB SAMA recipe at a smaller scale: image RES (refcoco x3 ×4),
# video RES (mevis ×4, lvvis ×2), video QA (chatunivi ×1).
train_dataset = dict(
    type=ConcatDataset,
    datasets=[
        # image RES
        refcoco_event_dataset, refcoco_plus_event_dataset, refcocog_event_dataset,
        refcoco_event_dataset, refcoco_plus_event_dataset, refcocog_event_dataset,
        refcoco_event_dataset, refcoco_plus_event_dataset, refcocog_event_dataset,
        refcoco_event_dataset, refcoco_plus_event_dataset, refcocog_event_dataset,
        # video RES / VIS
        video_mevis_event_dataset,
        video_lvvis_event_dataset,
        # video QA
        video_chatunivi_event_dataset,
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
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    # bfloat16 has the same dynamic range as fp32 -> the GradScaler is both
    # unnecessary AND triggers
    #   RuntimeError: "_amp_foreach_non_finite_check_and_unscale_cuda" not
    #                 implemented for 'BFloat16'
    # on PyTorch 2.2 (the container we're running). The V1 event config
    # already worked around this by disabling the scaler, which is the
    # standard recipe for bf16 training. We do the same here.
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
