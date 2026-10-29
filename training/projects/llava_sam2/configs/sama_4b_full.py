"""SAMA-4B Full Training Config - complete dataset"""

from mmengine.hooks import CheckpointHook, IterTimerHook, LoggerHook
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR
from torch.optim import AdamW
from transformers import AutoTokenizer

from xtuner.dataset import ConcatDataset
from xtuner.dataset.samplers import LengthGroupedSampler
from xtuner.engine.runner import TrainLoop
from xtuner.utils import PROMPT_TEMPLATE

from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss
from projects.llava_sam2.models.internvl_sama import InternVL_Slowfast
from projects.llava_sam2.models import VideoLLaVASAMModel_zero3, SAM2TrainRunner
from projects.llava_sam2.datasets import (
    video_lisa_collate_fn, ReferSegmDataset, VideoReVOSDataset, 
    VideoMeVISDataset, VideoRefYoutubeVOSDataset
)
from projects.llava_sam2.models.preprocess.image_resize import DirectResize

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
path = '/iopsstor/scratch/cscs/sfedorchenko/weights/Sa2VA-4B'
pretrained_pth = '/iopsstor/scratch/cscs/sfedorchenko/weights/Sa2VA-4B'

template = "phi3_chat"
prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = 8192

# Full training config
batch_size = 2  # per_device (use 2 for full training)
accumulative_counts = 4
dataloader_num_workers = 4
max_steps = 100000  # Full training
optim_type = AdamW
lr = 4e-5
betas = (0.9, 0.999)
weight_decay = 0.05
max_norm = 1
warmup_ratio = 0.05

save_steps = 1000
save_total_limit = 5

special_tokens = ['[SEG]', '<p>', '</p>', '<vp>', '</vp>']

NUM_TEMPORAL_TOKENS = 32
NUM_SPATIAL_TOKENS = 32
WINDOW_SIZE = 512
WINDOW_STRIDE = 512
DENSE_LEN = 5
SPARSE_LEN = 32
MAX_CONV_LEN = 6
MAX_FRAMES = 32

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
model = dict(
    type=VideoLLaVASAMModel_zero3,
    special_tokens=special_tokens,
    frozen_sam2_decoder=False,
    mllm=dict(
        type=InternVL_Slowfast,
        model_path=path,
        freeze_llm=True,
        window_size=WINDOW_SIZE,
        stride=WINDOW_STRIDE,
        freeze_visual_encoder=True,
        llm_lora=None,
        special_tokens=special_tokens,
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
        reduction='mean',
        loss_weight=0.5),
    pretrained_pth=pretrained_pth,
    loss_sample_points=True,
    bs=batch_size,
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################

DATA_ROOT = '/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/'
SA2VA_DATA_ROOT = DATA_ROOT + 'sa2va_training/'
# Images extracted from sa2va_training/osprey-724k_coco2014.zip → osprey-724k/coco/train2014/
# Dataset appends 'train2014/' prefix internally, so data_root must NOT end with train2014/
COCO_TRAIN2014 = SA2VA_DATA_ROOT + 'osprey-724k/coco/'

# Multiple datasets for full training
train_dataset = dict(
    type=ConcatDataset,
    datasets=[
        # RefCOCO
        dict(
            type=ReferSegmDataset,
            data_root=COCO_TRAIN2014,
            ann_file=SA2VA_DATA_ROOT + 'refcoco_plus/instances_train.json',
            split_file=SA2VA_DATA_ROOT + 'refcoco_plus/train.txt',
            tokenizer=tokenizer,
            prompt_template=prompt_template,
            extra_image_processor=extra_image_processor,
            max_length=max_length,
        ),
    ])

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=video_lisa_collate_fn))

#######################################################################
#                      PART 4  Optimizer & Scheduler                  #
#######################################################################

optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    # BFloat16 has a wide enough dynamic range; GradScaler is not needed
    # and older PyTorch (container) raises "_amp_foreach_non_finite_check_and_unscale_cuda
    # not implemented for BFloat16" when enabled.
    loss_scale=dict(init_scale=65536, enabled=False),
    accumulative_counts=accumulative_counts,
    clip_grad=dict(max_norm=max_norm, norm_type=2),
)

default_hooks = dict(
    runtime_info=dict(type=IterTimerHook),
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    logger=dict(type=LoggerHook, interval=1),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    type='Visualizer',
    vis_backends=vis_backends,
)

log_level = 'INFO'
load_from = None
resume_from = None
randomness = dict(seed=0)

custom_hooks = []

train_cfg = dict(
    type=TrainLoop,
    max_iters=max_steps,
)
