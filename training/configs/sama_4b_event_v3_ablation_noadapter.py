"""SAMA-4B Event Pipeline V3 -- train on EVERY converted event source.

V3 adds, on top of V2's mix (MeViS, LV-VIS, ChatUniVi, RefCOCO/+/g):

    * ReVOS (UVO subset)        video RES   -- revos_uvo_film events
    * Ref-YouTube-VOS           video RES   -- refytvos events
    * Ref-SAV (SA-V subset)     video RES   -- sav events (sav_000 shard,
                                               the masklets shipped by Sa2VA)
    * Flickr30k-entities GCG    image RES   -- flickr events
    * GranDf GCG                image RES   -- grandf events (1k subset)
    * LLaVA-1.5 mix665k QA      image QA    -- gqa/ocr_vqa/coco events

The missing "linking" annotations that blocked these in V2
(`rvos/meta_expressions.json` + `rvos/mask_dict.pkl`, `Ref-SAV.json`) were
downloaded from the SAMA paper's data release (Sa2VA-Training +
SAMA_Dataset on HuggingFace) and extracted under
`sama_dataset/sa2va_training/video_datas/`.

Repeats mirror the RGB SAMA recipe (`sama_4b_base.py`) scaled to the
converted-clip counts we actually have. All event dataset classes live in
`event_voxel_dataset_v2.py` (the launcher prepends its dir to sys.path).

V3.1 adds SAMA-239K VideoRegionDescription + VideoRegionConv on events:
    * VidSTG (6,625/6,626 videos), LVViS (2,595), MeViS (1,087),
      Ref-YouTube-VOS (1,976) sourced from sama239k_train_final.json with
      event wrappers. ≈ 81% of SAMA-239K's 15,151 videos.
    * SA-V mp4 in SAMA-239K skipped — covered by VideoSAM2EventDataset.

NOT yet included:
    * SA-V mp4 portion of SAMA-239K (2,867 videos); the Ref-SAV event
      dataset already provides SA-V referential segmentation signal.
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
    video_lisa_collate_fn,
)

# Event dataset classes (launcher prepends configs/ to sys.path).
from event_voxel_dataset_v2 import (  # noqa: E402
    RefCOCOEventDatasetV2 as RefCOCOEventDataset,
    VideoChatUniViEventDatasetV2 as VideoChatUniViEventDataset,
    VideoLVVISEventDatasetV2 as VideoLVVISEventDataset,
    VideoMeVISEventDatasetV2 as VideoMeVISEventDataset,
    VideoReVOSEventDatasetV2 as VideoReVOSEventDataset,
    VideoRefYoutubeVOSEventDatasetV2 as VideoRefYoutubeVOSEventDataset,
    VideoSAM2EventDataset,
    FlickrGCGEventDataset,
    GranDfGCGEventDataset,
    OpenPsgGCGEventDataset,
    RefCOCOgGCGEventDataset,
    LLaVAEventDataset,
    VideoRegionDescriptionEventDataset,
    VideoRegionConvEventDataset,
)
from projects.llava_sam2.models.preprocess.image_resize import DirectResize

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
path = '/iopsstor/scratch/cscs/sfedorchenko/weights/Sa2VA-4B_SAMA'
pretrained_pth = '/iopsstor/scratch/cscs/sfedorchenko/weights/Sa2VA-4B_SAMA'

# ── Event roots (one events.h5 per clip) ─────────────────────────────
# 2026-08-09: sama_converted (v1) was found to be stale/incomplete for most
# sources (audited: sav 1/55002, flickr 0/31783, grandf 0/1000, vidstg
# 1/6649, coco/train2014 0/82783, coco/train2017 0/118287, llava/gqa
# 0/148854 real events.h5 present). sama_converted_v2 is the current,
# ~100%-complete tree for everything except chatunivi (only in v1, already
# complete there) and coco/train2014 (missing from BOTH -- still needs a
# real conversion pass; see repair task).
CAPSTOR = '/capstor/scratch/cscs/mzehnde/sama_converted'
CAPSTOR_V2 = '/capstor/scratch/cscs/mzehnde/sama_converted_v2'
CAPSTOR_TRANSFER = '/capstor/scratch/cscs/mzehnde/sama_converted_transfer/max_sergei'  # nzubic's fresh conversion, 2026-08-14
CAPSTOR_REVOS = '/capstor/scratch/cscs/mzehnde/sama_converted_revos'  # dedicated ReVOS sub-source drop, 2026-08-16
IOPS = '/iopsstor/scratch/cscs/mzehnde/sama_converted'

mevis_event_root    = CAPSTOR_TRANSFER + '/mevis'       # 1852/1852 real (transfer; upgrade from v2's 1662/1662)
lvvis_event_root    = CAPSTOR_V2 + '/lvvis/train'       # 3073/3078 real (v2)
chatunivi_event_root = CAPSTOR_TRANSFER + '/chatunivi'  # 14845/14888 real (transfer; upgrade from v1's 13220/14888)
coco_event_root     = CAPSTOR_TRANSFER + '/coco/train2014'  # 82783/82783 real (transfer) -- RESOLVED 2026-08-14, was 0 everywhere before
refytvos_event_root = CAPSTOR_V2 + '/refytvos/train'    # 3471/3471 real (v2)
revos_event_root    = CAPSTOR_V2 + '/revos_uvo/all'     # 255/255 real (v2; note extra /all level vs v1)
revos_lvvis_root    = CAPSTOR_V2 + '/lvvis'             # standalone LV-VIS events; serves all 234 ReVOS LV-VIS videos
revos_ovis_root     = CAPSTOR_V2 + '/revos_ovis'        # 140/141 real (v2); serves all 68 ReVOS OVIS videos
revos_tao_root      = CAPSTOR_V2 + '/revos_tao'         # 445 real h5 (v2); serves all 102 ReVOS TAO videos
revos_mose_root     = CAPSTOR_REVOS + '/revos_mose'     # 208/208 real under /train -- RESOLVED 2026-08-16, covers all 127 ReVOS MOSE videos needed (was 73/127 partial)
sav_event_root      = CAPSTOR_V2 + '/sav'               # 919/919 real (v2); transfer has 7243/11957 real but coverage against
                                                          # SAMA-239K SAV mask annotations unverified -- smoke test decides before switching
flickr_event_root   = CAPSTOR_V2 + '/flickr'            # 31783/31783 real (v2)
grandf_event_root   = CAPSTOR_V2 + '/grandf/train'      # 1000/1000 real (v2) -- CORRECTED 2026-08-11: previous path pointed
                                                          # at grandf/ directly, which contains mislabeled psg_XXXXXXX siblings;
                                                          # the real GranDf conversion (sa_XXXXXXX naming) lives one level
                                                          # deeper in grandf/train/, missed by the earlier 2-level-deep audit.
psg_event_root      = CAPSTOR_V2 + '/coco/train2017'    # 118287/118287 real (v2; note: v2 lives on capstor, not iopsstor)

# SAMA-239K event roots (VideoRegionDescription/Conv)
vidstg_event_root = CAPSTOR_V2 + '/vidstg'              # 6648/6648 real (v2)
sama239k_event_roots = dict(
    lvvis=lvvis_event_root,            # 4818 clips
    mevis=mevis_event_root,            # covers the 1087 mevis clips
    refytvos=refytvos_event_root,      # 3471 clips
    vidstg=vidstg_event_root,          # 6625/6626 SAMA-239K VidSTG clips
    sav=sav_event_root,                # 2867/2867 SAMA-239K SAV clips (events-only)
)

# LLaVA image-QA event roots, keyed by the image-path prefix in mix665k.json.
llava_event_roots = dict()
llava_event_roots['gqa'] = CAPSTOR_V2 + '/llava/gqa'            # 148854/148854 real (v2)
llava_event_roots['ocr_vqa'] = CAPSTOR_TRANSFER + '/llava/ocr_vqa'  # 159522/159522 real (transfer; upgrade from v2's 95504/95504)
llava_event_roots['coco/train2017'] = CAPSTOR_V2 + '/coco/train2017'  # 118287/118287 real (v2)
llava_event_roots['coco/train2014'] = CAPSTOR_TRANSFER + '/coco/train2014'  # 82783/82783 real (transfer) -- RESOLVED 2026-08-14

# ── RGB-side annotation files (events reuse the same text/masks) ─────
SAMA_DATASET = '/iopsstor/scratch/cscs/sfedorchenko/sama_dataset'
SA2VA = SAMA_DATASET + '/sa2va_training'
VIDEO = SA2VA + '/video_datas'

# verified on-disk locations (not the stale paths from v2)
mevis_expression_file = SAMA_DATASET + '/mevis/train/meta_expressions_v2.json'
mevis_mask_file       = SAMA_DATASET + '/mevis/train/mask_dict.json'
lvvis_anno_file       = SAMA_DATASET + '/lvvis/train_instances.json'
chatunivi_json_file   = SAMA_DATASET + '/video_datas/chat_univi/video_chat.json'
chatunivi_image_folder = SAMA_DATASET + '/video_datas/chat_univi/Activity_Videos/'

data_root_refcoco      = SA2VA + '/ref_seg/refcoco'
data_root_refcoco_plus = SA2VA + '/ref_seg/refcoco+'
data_root_refcocog     = SA2VA + '/ref_seg/refcocog'
coco_train2014_jpg_root = SA2VA + '/osprey-724k/coco/train2014'

# downloaded from the SAMA paper release (Sa2VA-Training):
refytvos_expression_file = VIDEO + '/rvos/meta_expressions/train/meta_expressions.json'
refytvos_mask_file       = VIDEO + '/rvos/mask_dict.pkl'
revos_expression_file    = SAMA_DATASET + '/revos/meta_expressions_train_.json'
revos_mask_file          = SAMA_DATASET + '/revos/mask_dict.json'
sav_sam2_folder          = VIDEO + '/ref_sav/sam_v_full'
sav_expression_file      = VIDEO + '/ref_sav/Ref-SAV.json'

# GCG annotations (already on disk under grandf/train/):
flickr_ann_file = SAMA_DATASET + '/grandf/train/flickr_mergedGT_GCG_train.json'
grandf_ann_file = SAMA_DATASET + '/grandf/train/GranDf_HA_GCG_train.json'
psg_ann_file    = SAMA_DATASET + '/grandf/train/OpenPsgGCG_train.json'
refcocog_gcg_ann_file   = SAMA_DATASET + '/grandf/train/RefCOCOg_GCG_train.json'
# native COCO2014 dims for polygon-mask rasterisation (events-only path):
refcocog_gcg_instances  = data_root_refcocog + '/instances.json'

# LLaVA mix665k QA:
llava_json_file = SAMA_DATASET + '/llava_1_5/llava_v1_5_mix665k.json'

# ── Common prompt / size knobs ────────────────────────────────────────
template = "phi3_chat"
prompt_template = PROMPT_TEMPLATE.phi3_chat
max_length = 8192

# ── Optim ─────────────────────────────────────────────────────────────
# MULTI-NODE VARIANT (2026-08-16): 4 nodes x 4 GPUs = 16 GPUs, vs single-node's 4.
# accumulative_counts dropped 4->1 to hold effective global batch size constant:
# batch_size(4) x accumulative_counts x num_gpus must stay 64, matching the
# single-node run's 4x4x4=64, so the LR schedule (tuned around 64) is undisturbed.
# Reduced 4->3 after 2 OOM crashes (same signature both times: ~15.95GiB
# alloc attempt, ~82GB already in use out of ~95GB ceiling). Effective global
# batch drops from 64 to 48 (3x1x16) -- accepted tradeoff for run stability;
# this ablation's own baseline is its own iter-0 start, not a strict
# apples-to-apples batch match with the main run.
batch_size = 3
accumulative_counts = 1
# Reduced 4->2: the SAMA-239K mask dicts (~7 GB) are duplicated by each
# dataloader worker via Python refcount copy-on-write; fewer workers caps
# that churn and keeps CPU RAM bounded (see V4 OOM at iter ~110).
dataloader_num_workers = 2
# ABLATION: 1 epoch instead of 4 -- bounded budget for directional signal
# within the deadline, still gets a complete warmup+cosine-decay LR cycle
# (rather than an arbitrary truncated slice of the full 4-epoch schedule).
# On 16 GPUs this is ~24.5K iterations, roughly ~1 day of wall-clock time.
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
MAX_CONV_LEN_LLaVA = 1
MAX_FRAMES = 32

# ── Event-specific knobs ──────────────────────────────────────────────
EVENT_FPS = 100
EVENT_NORMALIZE = True
NUM_INPUT_CHANNELS = 2
# ABLATION 2026-08-16: event adapter disabled (baseline has it on). Everything
# else identical to the multinode config -- same data mix, same fresh start
# from pretrained_pth, same effective batch size.
USE_EVENT_ADAPTER = False
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

# Shared video-RES kwargs (MeViS-schema parents).
_video_res_common = dict(
    event_fps=EVENT_FPS,
    event_normalize=EVENT_NORMALIZE,
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    lazy=True,
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

# ── (A) Video RES: MeViS ─────────────────────────────────────────────
video_mevis_event_dataset = dict(
    type=VideoMeVISEventDataset,
    image_folder=mevis_event_root,
    expression_file=mevis_expression_file,
    mask_file=mevis_mask_file,
    event_folder=mevis_event_root,
    repeats=4,
    **_video_res_common,
)

# ── (B) Video VIS: LV-VIS train ──────────────────────────────────────
video_lvvis_event_dataset = dict(
    type=VideoLVVISEventDataset,
    image_folder=lvvis_event_root,
    expression_file=lvvis_anno_file,
    mask_file=None,
    event_folder=lvvis_event_root,
    repeats=2,
    **_video_res_common,
)

# ── (C) Video RES: ReVOS (UVO subset, FILM upsample) ─────────────────
video_revos_event_dataset = dict(
    type=VideoReVOSEventDataset,
    image_folder=revos_event_root,
    expression_file=revos_expression_file,
    mask_file=revos_mask_file,
    event_folder=revos_event_root,
    lvvis_event_folder=revos_lvvis_root,  # +234 LV-VIS clips
    ovis_event_folder=revos_ovis_root,    # +68 OVIS clips
    tao_event_folder=revos_tao_root,      # +102 TAO clips
    mose_event_folder=revos_mose_root,    # +127/127 MOSE clips (fully resolved 2026-08-16)
    repeats=2,  # ~499+ unique clips now (UVO + LV-VIS + OVIS + TAO + partial MOSE)
    **_video_res_common,
)

# ── (D) Video RES: Ref-YouTube-VOS ───────────────────────────────────
video_refytvos_event_dataset = dict(
    type=VideoRefYoutubeVOSEventDataset,
    image_folder=refytvos_event_root,
    expression_file=refytvos_expression_file,
    mask_file=refytvos_mask_file,
    event_folder=refytvos_event_root,
    repeats=4,
    **_video_res_common,
)

# ── (E) Video QA: ChatUniVi ──────────────────────────────────────────
video_chatunivi_event_dataset = dict(
    type=VideoChatUniViEventDataset,
    image_folder=chatunivi_image_folder,
    json_file=chatunivi_json_file,
    event_folder=chatunivi_event_root,
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
    virtual_clip_seconds=20.0,
    max_clips_per_video=8,
    meta_cache_path='/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/cache/chatunivi_event_meta_v3.json',
)

# ── (F) Video RES: Ref-SAV (SA-V pseudo-labels, sav_000 shard) ───────
video_sav_event_dataset = dict(
    type=VideoSAM2EventDataset,
    sam2_folder=sav_sam2_folder,
    expression_file=sav_expression_file,
    event_folder=sav_event_root,
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
    select_number=5,
    dense_len=DENSE_LEN,
    sparse_len=SPARSE_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
    max_frames=MAX_FRAMES,
)

# ── (G) Image RES: RefCOCO / RefCOCO+ / RefCOCOg ─────────────────────
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
        event_folder=coco_event_root,
        event_normalize=EVENT_NORMALIZE,
    )

refcoco_event_dataset = _refcoco_event_cfg(data_root_refcoco, 'refs(unc).p')
refcoco_plus_event_dataset = _refcoco_event_cfg(data_root_refcoco_plus, 'refs(unc).p')
refcocog_event_dataset = _refcoco_event_cfg(data_root_refcocog, 'refs(umd).p')

# ── (H) Image RES (GCG): Flickr30k-entities ──────────────────────────
flickr_gcg_event_dataset = dict(
    type=FlickrGCGEventDataset,
    data_path=flickr_ann_file,
    image_folder=flickr_event_root,
    event_folder=flickr_event_root,
    event_normalize=EVENT_NORMALIZE,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
    single_image_mode=True,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

# ── (I) Image RES (GCG): GranDf (1k subset) ──────────────────────────
grandf_gcg_event_dataset = dict(
    type=GranDfGCGEventDataset,
    data_path=grandf_ann_file,
    image_folder=grandf_event_root,
    event_folder=grandf_event_root,
    event_normalize=EVENT_NORMALIZE,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=10,
    single_image_mode=True,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

# ── (I2) Image RES (GCG): OpenPSG (COCO2017 events) ──────────────────
psg_gcg_event_dataset = dict(
    type=OpenPsgGCGEventDataset,
    data_path=psg_ann_file,
    image_folder=psg_event_root,
    event_folder=psg_event_root,
    event_normalize=EVENT_NORMALIZE,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
    single_image_mode=True,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

# ── (I3) Image RES (GCG): RefCOCOg-GCG (COCO2014 events) ─────────────
refcocog_gcg_event_dataset = dict(
    type=RefCOCOgGCGEventDataset,
    data_path=refcocog_gcg_ann_file,
    instances_json=refcocog_gcg_instances,
    image_folder=coco_event_root,
    event_folder=coco_event_root,
    event_normalize=EVENT_NORMALIZE,
    tokenizer=tokenizer,
    max_length=max_length,
    special_tokens=special_tokens,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=1,
    single_image_mode=True,
    max_conv_len=MAX_CONV_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

# ── (K) SAMA-239K: VideoRegionDescription (object-level descriptions) ──
SAMA239K_ANNO = '/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/sama_anno/sama_239k'
SAMA239K_MASKS = '/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/sama239k/sama_239k'
SAMA239K_VIDEO_ROOT = '/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/sama_video_root'

# event_roots covers LVViS / MeViS / RefYT-VOS; VidSTG + sav_train auto-excluded
sama239k_region_desc_event_dataset = dict(
    type=VideoRegionDescriptionEventDataset,
    image_folder=SAMA239K_VIDEO_ROOT,    # not used by wrapper but required by ctor
    data_path=SAMA239K_ANNO + '/sama239k_train_final.json',
    mevis_json=SAMA239K_MASKS + '/mevis_train_mask_dict.json',
    lvvis_json=SAMA239K_MASKS + '/lvvis_train_mask_dict.json',
    ref_youtube_pkl=SAMA239K_MASKS + '/ref_youtube_train_mask_dict.pkl',
    sav_json=SAMA239K_MASKS + '/SAV_mask_dict_train.json',
    vidstg_json=SAMA239K_MASKS + '/VidSTG_mask_dict_train_updated.json',
    event_roots=sama239k_event_roots,
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=2,
    sampled_frames=5,
    dense_len=DENSE_LEN,
    sparse_len=SPARSE_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
)

# ── (L) SAMA-239K: VideoRegionConv (multi-turn video chat + grounding) ──
sama239k_region_conv_event_dataset = dict(
    type=VideoRegionConvEventDataset,
    image_folder=SAMA239K_VIDEO_ROOT,
    data_path=SAMA239K_ANNO + '/sama239k_train_final.json',
    mevis_json=SAMA239K_MASKS + '/mevis_train_mask_dict.json',
    lvvis_json=SAMA239K_MASKS + '/lvvis_train_mask_dict.json',
    ref_youtube_pkl=SAMA239K_MASKS + '/ref_youtube_train_mask_dict.pkl',
    sav_json=SAMA239K_MASKS + '/SAV_mask_dict_train.json',
    vidstg_json=SAMA239K_MASKS + '/VidSTG_mask_dict_train_updated.json',
    event_roots=sama239k_event_roots,
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    special_tokens=special_tokens,
    extra_image_processor=extra_image_processor,
    lazy=True,
    repeats=2,
    dense_len=DENSE_LEN,
    sparse_len=SPARSE_LEN,
    num_temporal_token=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_size=WINDOW_SIZE,
    window_stride=WINDOW_STRIDE,
    max_conv_len=MAX_CONV_LEN,
    qa_per_item=4,
    max_frames=MAX_FRAMES,
)

# ── (J) Image QA: LLaVA-1.5 mix665k (gqa / ocr_vqa / coco events) ────
llava_event_dataset = dict(
    type=LLaVAEventDataset,
    tokenizer=tokenizer,
    data_path=llava_json_file,
    prompt_template=prompt_template,
    special_tokens=special_tokens,
    num_temporal_tokens=NUM_TEMPORAL_TOKENS,
    num_spatial_token=NUM_SPATIAL_TOKENS,
    window_stride=WINDOW_STRIDE,
    window_size=WINDOW_SIZE,
    max_conv_len=MAX_CONV_LEN_LLaVA,
    image_folder=llava_event_roots['coco/train2017'],  # unused (events only)
    event_roots=llava_event_roots,
    event_normalize=EVENT_NORMALIZE,
)

# ── Concat ───────────────────────────────────────────────────────────
# Mirrors sama_4b_base.py's ConcatDataset ordering/duplication, scaled to
# the converted-clip counts we have. In-list duplication x per-dataset
# `repeats` controls the effective sampling weight.
train_dataset = dict(
    type=ConcatDataset,
    datasets=[
        # image RES (RefCOCO/+/g) -- TEMP-DISABLED 2026-08-09: needs
        # coco/train2014 events -- RE-ENABLED 2026-08-14: nzubic's transfer has
        # 82783/82783 real events.h5 at CAPSTOR_TRANSFER/coco/train2014 (was 0
        # everywhere before; see task #6).
        refcoco_event_dataset, refcoco_plus_event_dataset, refcocog_event_dataset,
        refcoco_event_dataset, refcoco_plus_event_dataset, refcocog_event_dataset,
        refcoco_event_dataset, refcoco_plus_event_dataset, refcocog_event_dataset,
        refcoco_event_dataset, refcoco_plus_event_dataset, refcocog_event_dataset,
        # image QA
        llava_event_dataset,
        # video RES / VIS
        video_mevis_event_dataset,
        video_revos_event_dataset,
        video_refytvos_event_dataset,
        video_lvvis_event_dataset,
        # video QA
        video_chatunivi_event_dataset,
        # SA-V pseudo-label RES -- RE-ENABLED 2026-08-09: sav/*/events.h5
        # restored and verified 919/919 real under sama_converted_v2/sav.
        video_sav_event_dataset,
        # GCG
        # RE-ENABLED 2026-08-11: the mislabeled psg_XXXXXXX siblings sit directly under
        # grandf/, but the real GranDf conversion (sa_XXXXXXX, 1000/1000 real events.h5)
        # was there all along one level deeper at grandf/train/ -- grandf_event_root fixed
        # above, not a real data gap.
        grandf_gcg_event_dataset,
        flickr_gcg_event_dataset,
        psg_gcg_event_dataset,
        # RE-ENABLED 2026-08-14: coco/train2014 now real (see above). NOTE: this class's
        # retry-on-miss is implemented as unbounded recursion, which RecursionError'd at
        # 0% coverage before -- now that coverage should be ~100%, verify via smoke test
        # that it doesn't still hit that path before trusting it in the full training run.
        refcocog_gcg_event_dataset,
        # SAMA-239K: region description + multi-turn conv (LVViS/MeViS/RefYT)
        sama239k_region_desc_event_dataset,
        sama239k_region_conv_event_dataset,
    ]
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    persistent_workers=True,
    prefetch_factor=2,
    pin_memory=True,
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

model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=True,
    broadcast_buffers=False,
)

visualizer = None
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
log_processor = dict(by_epoch=False)
