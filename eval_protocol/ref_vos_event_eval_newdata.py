"""Distributed Referring-VOS evaluation on EVENT data.

Loads the trained Event-SAMA model from a (config, .pth) pair (no HF
conversion needed -- our event_adapter / 2-channel patch_embed lives in the
xtuner training-time model class) and runs `LLaVASam2.predict_video()` over
``RefVOSEventEvalDataset``. Per-frame PNG masks are saved under
``<work_dir>/Annotations/<vid>/<exp>/<frame>.png`` -- the format expected by
``tools/eval_video_seg/eval_mevis.py``.

DDP: 4 ranks, one per GPU, sharded by ``torch.utils.data.DistributedSampler``.

Usage (sbatch wrapper handles env):
    python ref_vos_event_eval.py \
        --config /path/to/sama_4b_event_v2.py \
        --pth_model /path/to/iter_3000.pth \
        --dataset MEVIS_VALID_U \
        --work_dir /path/to/eval_out \
        --launcher pytorch
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import time

# ── PATCH: keep matplotlib/HF caches off the quota-restricted /users tree.
os.environ.setdefault(
    "MPLCONFIGDIR", "/iopsstor/scratch/cscs/sfedorchenko/.cache/matplotlib"
)
os.environ.setdefault(
    "HF_HOME", "/iopsstor/scratch/cscs/sfedorchenko/.cache/huggingface"
)
os.environ.setdefault(
    "TRANSFORMERS_CACHE",
    "/iopsstor/scratch/cscs/sfedorchenko/.cache/huggingface/hub",
)
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
os.makedirs(os.environ["HF_HOME"], exist_ok=True)

# ── PATCH: disable FlashAttention2 in transformers BEFORE we touch any
#    transformers model class. Our env has flash_attn 2.0.4, which is below
#    transformers' >=2.1.0 minimum, so any HF model that auto-enables FA2
#    will hard-crash. Identical patch to train_no_fa2_pinned.py.
from transformers.modeling_utils import PreTrainedModel  # noqa: E402

@classmethod
def _force_eager_attn_implementation(cls, config, *args, **kwargs):
    if hasattr(config, "_attn_implementation_internal"):
        config._attn_implementation_internal = "eager"
    config._attn_implementation = "eager"
    return config

PreTrainedModel._autoset_attn_implementation = _force_eager_attn_implementation
PreTrainedModel._check_and_enable_flash_attn_2 = classmethod(
    lambda cls, config, *a, **kw: config
)
print("[PATCH] Flash-Attention 2 disabled in transformers", flush=True)

import numpy as np
import torch
import torch.distributed
import torch.nn.functional as F
import torch.utils.data
import tqdm
from PIL import Image

# mmengine + xtuner machinery
import mmengine
from mmengine.config import Config
from mmengine.dist import get_rank
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER

# in-tree util for DDP init (used by all ref_vos_eval scripts)
from projects.llava_sam2.evaluation.utils import (
    _init_dist_pytorch,
    _init_dist_slurm,
    get_dist_info,
)

# our event eval dataset
from event_refvos_dataset import RefVOSEventEvalDataset


# Where the per-(vid, exp) masks live for each known eval split. We default
# to event paths on capstor.
DATASETS_INFO = {
    "MEVIS_VALID_U": {
        "event_folder": "/capstor/scratch/cscs/mzehnde/sama_converted_transfer/max_sergei/mevis",
        "expression_file": (
            "/iopsstor/scratch/cscs/sfedorchenko/sama_dataset"
            "/mevis/valid_u/meta_expressions_v2.json"
        ),
        "mask_file": (
            "/iopsstor/scratch/cscs/sfedorchenko/sama_dataset"
            "/mevis/valid_u/mask_dict.json"
        ),
    },
    # Sanity eval on MeViS TRAIN (valid_u annotations were deleted 2026-06-30).
    # This is a train-data sanity check, NOT a clean val metric.
    "MEVIS_TRAIN": {
        "event_folder": "/capstor/scratch/cscs/mzehnde/sama_converted_transfer/max_sergei/mevis",
        "expression_file": (
            "/iopsstor/scratch/cscs/sfedorchenko/sama_dataset"
            "/mevis/train/meta_expressions_v2.json"
        ),
        "mask_file": (
            "/iopsstor/scratch/cscs/sfedorchenko/sama_dataset"
            "/mevis/train/mask_dict.json"
        ),
    },
    # ReVOS valid: PARTIAL coverage (329/416 videos have real event data --
    # MOSE/OVIS/UVO fully converted, LV-VIS/test not converted). Uses a
    # custom path_resolver (see _make_revos_resolver) instead of the plain
    # <event_folder>/<vid_id>/events.h5 layout; event_folder here is only
    # the UVO fallback root.
    "REVOS_VALID": {
        "event_folder": "/capstor/scratch/cscs/mzehnde/sama_converted_v2/revos_uvo/all",
        "expression_file": (
            "/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/sa2va_training"
            "/video_datas/revos/meta_expressions_valid_.json"
        ),
        "mask_file": (
            "/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/sa2va_training"
            "/video_datas/revos/mask_dict.json"
        ),
    },
}


def _make_revos_resolver(uvo_root):
    """Same prefix-based resolution validated earlier this session:
    LV-VIS/OVIS/TAO/MOSE keys go to their standalone event roots; anything
    else falls back to the UVO <yt>.f<segment> remap. Videos with no
    resolvable event file get a path that doesn't exist, so
    RefVOSEventEvalDataset's own pre-filter drops them cleanly."""
    import re

    capstor_v2 = "/capstor/scratch/cscs/mzehnde/sama_converted_v2"
    capstor_revos = "/capstor/scratch/cscs/mzehnde/sama_converted_revos"
    lvvis_root = capstor_v2 + "/lvvis"
    ovis_root = capstor_v2 + "/revos_ovis"
    tao_root = capstor_v2 + "/revos_tao"
    mose_root = capstor_revos + "/revos_mose"
    prefix_roots = [("LV-VIS/", lvvis_root), ("OVIS/", ovis_root),
                    ("TAO/", tao_root), ("MOSE/", mose_root)]

    suffix_re = re.compile(r"\.f\d+$")
    uvo_map = {}
    if os.path.isdir(uvo_root):
        for d in os.listdir(uvo_root):
            if os.path.isfile(os.path.join(uvo_root, d, "events.h5")):
                base = suffix_re.sub("", d)
                uvo_map.setdefault(base, d)
                uvo_map[d] = d

    def resolve(video_id):
        vid = str(video_id)
        for prefix, root in prefix_roots:
            if vid.startswith(prefix):
                parts = vid.split("/")
                return os.path.join(root, *parts[1:], "events.h5")
        key = vid.split("/")[-1]
        dirname = uvo_map.get(key) or uvo_map.get(vid)
        if dirname is None:
            return os.path.join(uvo_root, vid, "events.h5")
        return os.path.join(uvo_root, dirname, "events.h5")

    return resolve


# ──────────────────────────────────────────────────────────────────────────
#                              text utilities
# ──────────────────────────────────────────────────────────────────────────


SEG_PROMPT = "Please segment {}."


def build_input_text(
    raw_expression: str,
    *,
    template: dict,
    bot_name: str,
    img_start_token: str,
    img_end_token: str,
    img_context_token: str,
    n_context_per_frame: int,
    num_frames: int,
) -> str:
    """Mirror modeling_sa2va_chat.predict_forward's text-construction so
    encoding matches what training saw."""
    image_token_str = (
        f"{img_start_token}"
        f"{img_context_token * n_context_per_frame}"
        f"{img_end_token}"
    ) + "\n"
    image_token_str = (image_token_str * num_frames).strip()
    seg_text = SEG_PROMPT.format(raw_expression)
    if "<image>" not in seg_text:
        seg_text = "<image>\n" + seg_text
    seg_text = seg_text.replace("<image>", image_token_str)
    input_text = template["INSTRUCTION"].format(
        input=seg_text, round=1, bot_name=bot_name
    )
    return input_text


# ──────────────────────────────────────────────────────────────────────────
#                       custom event-VOS inference
# ──────────────────────────────────────────────────────────────────────────


def _get_seg_hidden_states(hidden_states, output_ids, seg_id):
    """Extract the hidden states whose token id is the [SEG] token. Mirror
    of the helper inside modeling_sa2va_chat.predict_forward."""
    seg_mask = output_ids == seg_id
    n_out = int(len(seg_mask))
    if n_out == 0:
        return hidden_states[0:0]
    return hidden_states[-n_out:][seg_mask]


def event_predict(
    model,
    item: dict,
    input_ids: torch.Tensor,
):
    """Run the trained Event-SAMA model on a single eval item and return
    (text_prediction, masks). Replaces ``LLaVASam2.predict_video`` because
    ``predict_video`` references a method (``language_embd_inference``) that
    only exists on the HF-time SAM2 wrapper. Here we use the training-time
    APIs ``get_sam2_embeddings`` + ``inject_language_embd`` which DO exist on
    ``SAM2TrainRunner``.

    Returns:
        text: str
        dense_masks: (n_dense, ori_h, ori_w) uint8 numpy array (one per
                     dense frame; the entry script propagates this to all
                     frames in the clip).
    """
    n_dense = len(item["g_pixel_values_raw"])
    ori_h, ori_w = item["ori_height"], item["ori_width"]

    # ── ViT input: pre-transformed event voxels for the LLM context. ───
    pixel_values = item["pixel_values"].to(
        dtype=model.torch_dtype, device="cuda"
    )

    # ── SAM2 input: pseudo-RGB rendering of the dense event windows. ───
    # Mirror EventVoxel_Dataset.__getitem__ + LLaVASam2.predict_video lines
    # 631-633: extra_image_processor.apply_image -> torch (3,H,W) -> stacked
    # preprocess_image -> (n_dense, 3, 1024, 1024).
    g_chw_list = []
    for ev_img_uint8 in item["g_pixel_values_raw"]:
        ev = model.extra_image_processor.apply_image(ev_img_uint8)
        g_chw_list.append(
            torch.from_numpy(ev).permute(2, 0, 1).contiguous()
            .to(dtype=model.torch_dtype, device="cuda")
        )
    g_pixel_values = torch.stack([
        model.grounding_encoder.preprocess_image(g) for g in g_chw_list
    ]).to(dtype=model.torch_dtype)

    # ── LLM forward: greedy generation. ──────────────────────────────
    input_ids = input_ids.to(device="cuda")
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)

    mm_inputs = {
        "pixel_values": pixel_values,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": None,
        "past_key_values": None,
        "labels": None,
    }
    generate_output = model.mllm.generate(
        **mm_inputs,
        generation_config=model.gen_config,
        streamer=None,
        bos_token_id=model.tokenizer.bos_token_id,
        stopping_criteria=model.stop_criteria,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    text = model.tokenizer.decode(
        generate_output.sequences[0], skip_special_tokens=False
    ).strip()

    # ── extract the [SEG] hidden state from the generated sequence ──
    hidden_states = generate_output.hidden_states
    last_hidden_states = [hh[-1][0] for hh in hidden_states]
    last_hidden_states = torch.cat(last_hidden_states, dim=0)
    seg_hidden = _get_seg_hidden_states(
        last_hidden_states, generate_output.sequences[0][:-1],
        seg_id=model.seg_token_idx,
    )
    if len(seg_hidden) == 0:
        return text, None
    if len(seg_hidden) > 1:
        seg_hidden = seg_hidden[:1]

    # text_hidden_fcs is fp32 (per preparing_for_generation).
    seg_hidden = model.text_hidden_fcs(seg_hidden).to(torch.float32)
    # seg_hidden: (1, hidden) -> (n_dense, 1, hidden) for SAM2.
    language_embeddings = seg_hidden.unsqueeze(0).expand(n_dense, 1, -1)

    # ── SAM2 forward: get image features, inject [SEG], decode masks. ─
    sam_states = model.grounding_encoder.get_sam2_embeddings(
        g_pixel_values, expand_size=1
    )
    pred_masks = model.grounding_encoder.inject_language_embd(
        sam_states, language_embeddings, nf_nobj=(n_dense, 1)
    )
    # pred_masks: (n_dense, 1, h_low, w_low)
    pred_masks = pred_masks[:, 0]  # (n_dense, h_low, w_low)
    pred_masks = F.interpolate(
        pred_masks.unsqueeze(1).float(),
        size=(ori_h, ori_w),
        mode="bilinear", align_corners=False,
    )[:, 0]
    dense_masks = (pred_masks.sigmoid() > 0.5).cpu().numpy().astype(np.uint8)
    return text, dense_masks


# ──────────────────────────────────────────────────────────────────────────
#                       mask post-processing & saving
# ──────────────────────────────────────────────────────────────────────────


def propagate_masks_to_all_frames(
    dense_masks: np.ndarray,
    dense_indices,
    n_total: int,
) -> np.ndarray:
    """Nearest-neighbor in time. ``dense_masks`` has shape (n_dense, H, W).
    ``dense_indices`` is the sorted list of frame indices in [0, n_total)
    where the model produced masks. Returns (n_total, H, W)."""
    n_dense, H, W = dense_masks.shape
    assert n_dense == len(dense_indices)
    out = np.zeros((n_total, H, W), dtype=dense_masks.dtype)
    if n_dense == 0:
        return out
    di = np.asarray(dense_indices, dtype=np.int64)
    for f in range(n_total):
        # nearest dense index
        j = int(np.argmin(np.abs(di - f)))
        out[f] = dense_masks[j]
    return out


def mask_save(item, mask_prediction, work_dir):
    """Save (n_total, H, W) uint8 masks as one PNG per frame."""
    vid_id = item["video_id"]
    exp_id = item["exp_id"]
    save_path = os.path.join(work_dir, "Annotations", vid_id, exp_id)
    mmengine.mkdir_or_exist(save_path)
    for id_m, mask in enumerate(mask_prediction):
        img = Image.fromarray((mask.astype(np.float32) * 255.0)).convert("L")
        file_name = item["all_frames"][id_m]
        img.save(os.path.join(save_path, file_name + ".png"))


# ──────────────────────────────────────────────────────────────────────────
#                                 main
# ──────────────────────────────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Event Referring-VOS eval")
    p.add_argument("--config", required=True, help="xtuner training config")
    p.add_argument("--pth_model", required=True, help="iter_*.pth dir")
    p.add_argument(
        "--dataset",
        default="MEVIS_VALID_U",
        choices=list(DATASETS_INFO.keys()),
    )
    p.add_argument("--work_dir", required=True)
    p.add_argument(
        "--launcher",
        choices=["none", "pytorch", "slurm"],
        default="none",
    )
    p.add_argument(
        "--max_dense_frames",
        type=int,
        default=32,
        help="how many frames to feed SAM2 grounding per clip",
    )
    p.add_argument(
        "--llm_frames",
        type=int,
        default=5,
        help="how many frames to feed the LLM (capped by training recipe)",
    )
    p.add_argument(
        "--event_fps",
        type=float,
        default=None,
        help="override config EVENT_FPS (accumulation window = 1e6/event_fps us)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="if set, only evaluate the first N items (smoke test)",
    )
    p.add_argument("--local_rank", "--local-rank", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()

    # DDP init
    if args.launcher == "none":
        rank, world_size = 0, 1
    elif args.launcher == "pytorch":
        _init_dist_pytorch("nccl")
        rank, world_size = get_dist_info()
    elif args.launcher == "slurm":
        _init_dist_slurm("nccl")
        rank, world_size = get_dist_info()
    else:
        raise ValueError(f"unknown launcher {args.launcher!r}")

    if rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        print(f"=== Event Referring-VOS Eval ===", flush=True)
        print(f"config:    {args.config}", flush=True)
        print(f"pth_model: {args.pth_model}", flush=True)
        print(f"dataset:   {args.dataset}", flush=True)
        print(f"work_dir:  {args.work_dir}", flush=True)
        print(f"world:     rank=0..{world_size - 1}", flush=True)

    # Build model from training config + .pth (mirrors convert_to_hf_st.py).
    cfg = Config.fromfile(args.config)
    if rank == 0:
        print("[model] building from config...", flush=True)
    t0 = time.time()
    model = BUILDER.build(cfg.model)
    if rank == 0:
        print(f"[model] built in {time.time() - t0:.1f}s; "
              f"loading weights from {args.pth_model}", flush=True)
    state_dict = guess_load_checkpoint(args.pth_model)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if rank == 0:
        print(f"[model] missing={len(missing)} keys, "
              f"unexpected={len(unexpected)} keys (strict=False)", flush=True)
        if len(missing) > 0:
            print(f"  first 5 missing: {list(missing)[:5]}", flush=True)
        if len(unexpected) > 0:
            print(f"  first 5 unexpected: {list(unexpected)[:5]}", flush=True)

    # Move to GPU + bf16 + put in eval mode.
    model = model.eval().to(dtype=torch.bfloat16).cuda()

    # Init the model's tokenizer / gen_config / extra_image_processor.
    metainfo = {
        "template": cfg.prompt_template,
        "max_dynamic_patch": getattr(cfg, "max_dynamic_patch", 12),
    }
    model.preparing_for_generation(metainfo=metainfo)
    if rank == 0:
        print(f"[model] preparing_for_generation done; "
              f"img_size={model.image_size} patch_token={model.patch_token}",
              flush=True)

    # Build dataset (event H5 -> voxels) and DDP-shard it.
    info = DATASETS_INFO[args.dataset]
    resolver = (_make_revos_resolver(info["event_folder"])
                if args.dataset == "REVOS_VALID" else None)
    dataset = RefVOSEventEvalDataset(
        event_folder=info["event_folder"],
        expression_file=info["expression_file"],
        image_size=model.image_size,
        event_fps=(args.event_fps if args.event_fps is not None
                   else getattr(cfg, "EVENT_FPS", 100.0)),
        event_normalize=True,
        llm_frames=args.llm_frames,
        dense_frames=args.max_dense_frames,
        path_resolver=resolver,
    )
    if args.limit is not None:
        dataset.items = dataset.items[: args.limit]
        if rank == 0:
            print(f"[ds] LIMIT={args.limit} -> "
                  f"{len(dataset.items)} items", flush=True)

    sampler = torch.utils.data.DistributedSampler(
        dataset, num_replicas=world_size, rank=rank,
        shuffle=False, drop_last=False,
    )
    loader = torch.utils.data.DataLoader(
        dataset, sampler=sampler, batch_size=1,
        num_workers=2, pin_memory=False, collate_fn=lambda x: x[0],
    )

    if rank == 0:
        print(f"[ds] {len(dataset)} total items "
              f"-> ~{len(dataset) // max(world_size, 1)} per rank", flush=True)

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=4)
    pbar = tqdm.tqdm(loader, disable=(rank != 0))

    n_seg_misses = 0
    n_done = 0
    t_loop_start = time.time()

    for item in pbar:
        n_dense = len(item["g_pixel_values_raw"])
        n_llm = len(item["pixel_values"])

        # Build text + tokenize. num_frames matches pixel_values.
        input_text = build_input_text(
            item["text_prompt"],
            template=model.template,
            bot_name=model.bot_name,
            img_start_token=model.IMG_START_TOKEN,
            img_end_token=model.IMG_END_TOKEN,
            img_context_token=model.IMG_CONTEXT_TOKEN,
            n_context_per_frame=model.patch_token,
            num_frames=n_llm,
        )
        input_ids = model.tokenizer.encode(input_text, return_tensors="pt")

        with torch.no_grad():
            text_pred, dense_masks = event_predict(model, item, input_ids)

        if dense_masks is None:
            n_seg_misses += 1
            dense_masks = np.zeros(
                (n_dense, item["ori_height"], item["ori_width"]),
                dtype=np.uint8,
            )

        # Propagate dense masks to ALL frames (nearest-neighbor in time).
        full_masks = propagate_masks_to_all_frames(
            dense_masks, item["dense_indices"], item["length"]
        )

        executor.submit(mask_save, item, full_masks, args.work_dir)

        n_done += 1
        if rank == 0 and (n_done % 25 == 0):
            dt = time.time() - t_loop_start
            print(f"[rank0] {n_done}/{len(loader)} items "
                  f"({dt:.1f}s, {n_done / max(dt, 1e-3):.2f}/s, "
                  f"{n_seg_misses} no-[SEG])", flush=True)

    executor.shutdown(wait=True)
    if torch.distributed.is_initialized():
        torch.distributed.barrier()
    if rank == 0:
        print(f"[done] processed {n_done} items "
              f"({n_seg_misses} no-[SEG] cases)", flush=True)
        print(f"[done] masks under {args.work_dir}/Annotations", flush=True)


if __name__ == "__main__":
    main()
