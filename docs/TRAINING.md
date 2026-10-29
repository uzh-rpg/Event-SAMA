# Training

> Read [CLUSTER_PATHS.md](CLUSTER_PATHS.md) first. Every path in `training/configs/` is absolute
> and points at the CSCS Alps filesystem.

## Recipe

The released checkpoint `iter_133208` was produced by
[`training/configs/train_fulldata.sbatch`](../training/configs/train_fulldata.sbatch) driving
[`training/configs/sama_4b_event_v3_fulldata.py`](../training/configs/sama_4b_event_v3_fulldata.py).

| Setting | Value |
|---|---|
| Base weights | `Sa2VA-4B_SAMA` (SAMA-4B: InternViT + Qwen2.5-3B-Instruct + SAM2-hiera-large) |
| Trainable params | 323,307,461 (see [ARCHITECTURE.md](ARCHITECTURE.md)) |
| Epochs | 4 |
| Iterations | 133,208 (= 1,598,496 samples/epoch ÷ global batch 48 × 4) |
| Micro-batch | 3 per GPU |
| GPUs | 16 (4 nodes × 4 GH200) |
| Global batch | 48 |
| Grad accumulation | 1 |
| Optimizer | AdamW, lr 4e-5, β=(0.9, 0.999), weight decay 0.05 |
| LR schedule | LinearLR warmup (5%) → CosineAnnealingLR to 0 |
| Grad clipping | 1.0 |
| Precision | bf16, DeepSpeed ZeRO-2 ([`deepspeed_zero2_event.json`](../training/configs/deepspeed_zero2_event.json)) |
| Max sequence length | 8192 |
| Prompt template | `phi3_chat` (a Sa2VA naming artifact — the LLM is Qwen2.5-3B-Instruct) |
| Keyframes per clip | 32 (`MAX_FRAMES`) |
| Event window | 10 ms (`EVENT_FPS = 100`) |
| STC window / stride | 512 / 512 |
| Frozen | LLM, vision encoder (except `patch_embedding`) |
| Checkpointing | every 1000 iters, keep last 10 |

## Launching

```bash
sbatch training/configs/train_fulldata.sbatch
```

The launcher is **self-chaining**: it queues its own successor with
`--dependency=afterany` before starting, so training survives the 24-hour wall clock and node
preemption. It stops on three signals:

- `last_checkpoint` reaches `MAX_ITERS` → writes a `TRAINING_COMPLETE` sentinel and exits.
- Three consecutive jobs make no progress → stall guard kills the chain.
- Another job with the same name is already `RUNNING` → exits as a duplicate fork.

**`MAX_ITERS` must match what mmengine actually computes for your data mix.** If it is set too high
the completion check never fires: the chain resumes, does zero iterations, and the stall guard
eventually gives up *without* writing `TRAINING_COMPLETE`. That silently happened to the
single-node run (stalled at 382,416 of a configured 399,492 while training was in fact complete).
Read the real total from the training log's `Iter(train) [.../N]` line and set `MAX_ITERS` to it.

## Config inventory

`training/configs/` holds every config and launcher used in the project, not just the released run.
The naming maps as:

| Prefix | Meaning |
|---|---|
| `sama_4b_event_v3_fulldata.py` | **the released recipe** — full 25-source mix, 4 epochs |
| `sama_4b_event_v3_multinode_full.py` | the paper's headline run (pre-fulldata mix) |
| `sama_4b_event_v3_multinode_best.py` | SAM2-frozen + ViT-last-6-unfrozen variant |
| `sama_4b_event_v3_ablation_*.py` | 1-epoch iso-compute knockouts |
| `sama_4b_event_v3_fulldata_*.py` | the same knockouts at the 4-epoch fulldata budget |
| `train_*.sbatch` | SLURM launcher for the correspondingly named config |
| `convert_*_to_hf.sbatch` | DeepSpeed shards → HuggingFace `safetensors` |
| `probe_params.py`, `count_params.py` | parameter accounting |
| `diag_sama239k_sources*.py` | per-source item counts (how the data caveats were found) |

Ablations are expressed as single flags against the baseline config:

| Ablation | Switch |
|---|---|
| no event adapter | `USE_EVENT_ADAPTER = False` |
| random patch-embed init | `PATCH_EMBED_INIT = 'random'` |
| frozen SAM2 decoder | `_frozen_sam2_decoder = True` |
| unfreeze last 6 ViT layers | `UNFREEZE_VIT_LAST_N_LAYERS = 6` |
| merge ON/OFF polarity | env var `EVENT_MERGE_POLARITY=1` (**not** a config flag — see below) |

## Two things that will cost you a day

**Never set environment variables or import modules inside an mmengine/xtuner config.** mmengine
deep-copies the config namespace; a bound module object fails with `cannot pickle 'module' object`.
Set env vars in the sbatch instead. (`EVENT_MERGE_POLARITY` is an env var for exactly this reason.)
A `Transport endpoint is not connected` error in the same log is usually a *secondary* symptom —
`srun` tearing the container down under the surviving ranks — not the root cause.

**SAM2's image encoder cannot do 64-frame batches.** It throws
`CUDA error: invalid configuration argument` — a kernel grid limit, not OOM. 32 keyframes is the
practical ceiling without a chunked forward pass.

## Runtime environment

Training runs inside an ARM64 (GH200) container image via `srun --container-image`. The image is
not redistributable here; `conversion/Dockerfile` covers the conversion half, and
`training/requirements.txt` lists the training dependencies. Key environment settings the launcher
applies:

```
TRANSFORMERS_NO_FLASH_ATTN_2=1     # FlashAttention-2 disabled on this stack
PYTHONPATH=<repo>/training
MPLBACKEND=Agg
```

Training is entered through [`train_no_fa2_pinned.py`](../training/configs/train_no_fa2_pinned.py)
rather than xtuner's stock entry point, to pin the no-FlashAttention-2 code path.
