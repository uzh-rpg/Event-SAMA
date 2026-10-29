"""SAMA-4B RGB Video Pipeline Training Config

Full RGB training pipeline using all available Sa2VA datasets:
  - Video RES:   ReVOS, MeViS, RefYouTubeVOS
  - Video Chat:  ChatUniVi
  - Video SAM2:  Ref-SAV
  - Image RES:   RefCOCO / RefCOCO+ / RefCOCOg  (×4 repeats each)
  - Image QA:    LLaVA-Instruct-150K
  - Image GCG:   GlaMMm (RefCOCOg-GCG, GranDf, Flickr30K, OpenPSG)
  - Image VQA:   Osprey-724K  (5 splits)

SAMA-239K datasets (VideoRegionDescriptionDataset, VideoRegionConvDataset) are
intentionally excluded – the sama239k_data is not available on this machine.
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
from peft import LoraConfig

from projects.llava_sam2.models.internvl_sama import InternVL_Slowfast
from projects.llava_sam2.models import VideoLLaVASAMModel_zero3, SAM2TrainRunner
from projects.llava_sam2.datasets import (
    VideoReVOSDataset, VideoMeVISDataset, VideoRefYoutubeVOSDataset,
    video_lisa_collate_fn, VideoSAM2Dataset,
    VideoChatUniViDataset,
    RefCOCOgGCGDataset, OpenPsgGCGDataset, FlickrGCGDataset, GranDfGCGDataset,
    OspreyDataset, OspreyDescriptionDataset, OspreyShortDescriptionDataset,
    LLaVADataset,
    ReferSegmDataset,
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

batch_size = 2  # per_device
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
NUM_TEMPORAL_TOKENS_LLaVA = 32
MAX_CONV_LEN_LLaVA = 1

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

SA2VA = '/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/sa2va_training/'
VIDEO = SA2VA + 'video_datas/'

# ── Video RES ─────────────────────────────────────────────────────────
data_root_revos = VIDEO + 'revos/'
video_revos_dataset = dict(
    type=VideoReVOSDataset,
    image_folder=data_root_revos,
    expression_file=data_root_revos + 'meta_expressions_train_.json',
    mask_file=data_root_revos + 'mask_dict.json',
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=10,
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

data_root_mevis = VIDEO + 'mevis/train/'
video_mevis_dataset = dict(
    type=VideoMeVISDataset,
    image_folder=data_root_mevis + 'JPEGImages',
    expression_file=data_root_mevis + 'meta_expressions.json',
    mask_file=data_root_mevis + 'mask_dict.json',
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

data_root_refytvos = VIDEO + 'rvos/'
video_refytvos_mask_file = data_root_refytvos + 'mask_dict.pkl'
video_refytvos_dataset = dict(
    type=VideoRefYoutubeVOSDataset,
    image_folder=data_root_refytvos + 'train/JPEGImages/',
    expression_file=data_root_refytvos + 'meta_expressions/train/meta_expressions.json',
    mask_file=video_refytvos_mask_file,
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

# ── Video Chat ────────────────────────────────────────────────────────
data_root_chatunivi = VIDEO + 'chat_univi/'
video_qa_dataset = dict(
    type=VideoChatUniViDataset,
    image_folder=data_root_chatunivi + 'Activity_Videos/',
    json_file=data_root_chatunivi + 'video_chat.json',
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

# ── Video SAM2 (Ref-SAV) ──────────────────────────────────────────────
video_sam2_dataset = dict(
    type=VideoSAM2Dataset,
    sam2_folder=VIDEO + 'ref_sav/sam_v_full',
    expression_file=VIDEO + 'ref_sav/Ref-SAV.json',  # full 37,311 entries — all 56 shards now downloaded
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=4,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    sampled_frames=5,
    select_number=5,
    dense_len=DENSE_LEN,
    sparse_len=SPARSE_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
    max_frames=MAX_FRAMES,
)

# ── Image QA (LLaVA) ─────────────────────────────────────────────────
LLAVA_ROOT = SA2VA + 'llava_data/'
llava_vqa_dataset = dict(
    type=LLaVADataset,
    tokenizer=tokenizer,
    data_path=LLAVA_ROOT + 'LLaVA-Instruct-150K/llava_v1_5_mix665k.json',
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    num_temporal_tokens=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_stride=WINDOW_STRIDE,
    max_conv_len=MAX_CONV_LEN_LLaVA,
    image_folder=LLAVA_ROOT + 'llava_images/',
    window_size=WINDOW_SIZE,
)

# ── Image RES (RefCOCO / RefCOCO+ / RefCOCOg) ────────────────────────
# Annotations live in ref_seg/<dataset>/; images are physically stored
# under ref_seg/<dataset>/coco2014/train2014/
REF_SEG = SA2VA + 'ref_seg/'

refcoco_segm_dataset = dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=REF_SEG + 'refcoco',
    data_prefix=dict(img_path=REF_SEG + 'refcoco/coco2014/train2014'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
    max_length=max_length,
)

refcoco_plus_segm_dataset = dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=REF_SEG + 'refcoco+',
    data_prefix=dict(img_path=REF_SEG + 'refcoco+/coco2014/train2014'),
    ann_file='instances.json',
    split_file='refs(unc).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
    max_length=max_length,
)

refcocog_segm_dataset = dict(
    type=ReferSegmDataset,
    tokenizer=tokenizer,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    data_root=REF_SEG + 'refcocog',
    data_prefix=dict(img_path=REF_SEG + 'refcocog/coco2014/train2014'),
    ann_file='instances.json',
    split_file='refs(umd).p',
    prompt_template=prompt_template,
    num_classes_per_sample=5,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
    max_length=max_length,
)

# ── Image GCG (GlaMMm) ────────────────────────────────────────────────
GLAMM = SA2VA + 'glamm_data/'

glamm_refcocog_dataset = dict(
    type=RefCOCOgGCGDataset,
    image_folder=GLAMM + 'images/coco2014/train2014/',
    data_path=GLAMM + 'annotations/RefCOCOg_GCG_train.json',
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

glamm_grandf_dataset = dict(
    type=GranDfGCGDataset,
    data_path=GLAMM + 'annotations/GranDf_HA_GCG_train.json',
    image_folder=GLAMM + 'images/grandf/train/',
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=10,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

glamm_flickr_dataset = dict(
    type=FlickrGCGDataset,
    data_path=GLAMM + 'annotations/flickr_mergedGT_GCG_train.json',
    image_folder=GLAMM + 'images/flickr30k/Flickr30K/',
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

glamm_psg_dataset = dict(
    type=OpenPsgGCGDataset,
    data_path=GLAMM + 'annotations/OpenPsgGCG_train.json',
    image_folder=GLAMM + 'images/coco2017/',
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

# ── Image VQA (Osprey-724K) ───────────────────────────────────────────
OSPREY = SA2VA + 'osprey-724k/'
_osprey_image_folders = [
    OSPREY + 'coco/train2014/',
    OSPREY + 'coco/val2014/',
    OSPREY + 'coco/train2017/',
    OSPREY + 'coco/val2017/',
]

image_osprey_dataset = dict(
    type=OspreyDataset,
    image_folder=_osprey_image_folders,
    data_path=OSPREY + 'Osprey-724K/osprey_conversation.json',
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

image_osprey_description_dataset = dict(
    type=OspreyDescriptionDataset,
    image_folder=_osprey_image_folders,
    data_path=OSPREY + 'Osprey-724K/osprey_detail_description.json',
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

image_osprey_short_dataset = dict(
    type=OspreyShortDescriptionDataset,
    image_folder=_osprey_image_folders,
    data_path=OSPREY + 'Osprey-724K/osprey_short_form.json',
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    max_conv_len=0,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

image_osprey_part_dataset = dict(
    type=OspreyDataset,
    image_folder=_osprey_image_folders,
    data_path=OSPREY + 'Osprey-724K/osprey_part_level.json',
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

image_osprey_positive_neg_dataset = dict(
    type=OspreyDataset,
    image_folder=_osprey_image_folders,
    data_path=OSPREY + 'Osprey-724K/osprey_lvis_positive_negative.json',
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
    repeats=1,
    special_tokens=special_tokens,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

# ── Combined train dataset ────────────────────────────────────────────
train_dataset = dict(
    type=ConcatDataset,
    datasets=[
        # Image RES – ×4 repeats of each RefCOCO variant
        refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        refcoco_segm_dataset, refcoco_plus_segm_dataset, refcocog_segm_dataset,
        # Image QA
        llava_vqa_dataset,
        # Video RES
        video_mevis_dataset, video_revos_dataset, video_refytvos_dataset,
        # Video Chat
        video_qa_dataset,
        # Video SAM2 pseudo
        video_sam2_dataset,
        # Image GCG
        glamm_psg_dataset,
        glamm_grandf_dataset,
        glamm_flickr_dataset,
        glamm_refcocog_dataset,
        # Image VQA (Osprey)
        image_osprey_dataset, image_osprey_description_dataset,
        image_osprey_part_dataset, image_osprey_short_dataset,
        image_osprey_positive_neg_dataset,
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
    # BFloat16 has a wide enough dynamic range; GradScaler is not needed and
    # older PyTorch (container 24.01) raises
    # "_amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16"
    # when loss_scale='dynamic' is used.
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
