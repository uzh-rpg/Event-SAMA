# Checkpoints

**No weights are committed to this repository** — the smallest useful checkpoint is 8.7 GB. This
page records which checkpoint is which, what each scores, and how to convert between formats.

## Which one you want

| | Checkpoint | Why |
|---|---|---|
| **Latest / current model** | `sama_4b_event_v4_fulldata/iter_133208.pth` | Trained on the full, fixed 25-source mix (4 epochs). **All real-sensor evaluation uses this one.** J&F 24.37. |
| **Best-scoring** | `sama_4b_event_v4_multinode_full/iter_127468.pth` | The paper's headline number, J&F 25.75 — but trained before the data-root fix, so without VidSTG / Flickr / GranD-f. |

If you want the model the paper's headline claims describe, take `iter_127468`. If you want the
model that was actually in use going forward, take `iter_133208`. See [DATA.md](DATA.md) for why
they differ.

## Full inventory

All paths relative to the run root (on the original cluster,
`/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/`).

### Raw DeepSpeed checkpoints (~10 GB each)

| Run directory | Final iter | Epochs | J&F | Note |
|---|---|---|---|---|
| `sama_4b_event_v4_fulldata` | 133,208 | 4 | 24.37 | full fixed data mix — **latest** |
| `sama_4b_event_v4_multinode_full` | 127,468 | 4 | 25.75 | **headline** |
| `sama_4b_event_v4_multinode_best` | 127,468 | 4 | 24.44 | SAM2-frozen + ViT-last-6 variant |
| `sama_4b_event_v4` | 382,416 | — | 25.44 | single-node; different effective batch and LR shape, not comparable |
| `sama_4b_event_v4_noadapter_4ep` | 37,000 | — | — | |
| `sama_4b_event_v4_ablation_*` | 31,207 / 32,711 / 32,721 | 1 | see [EVALUATION.md](EVALUATION.md) | 1-epoch knockouts |
| `sama_4b_event_v4_fulldata_*` | 33,302 | 1 | see [EVALUATION.md](EVALUATION.md) | fulldata-budget knockouts |
| `epoch_checkpoints/multinode_full_epoch1_iter34000.pth` | 34,000 | 1 | — | epoch-1 snapshot of the headline run |

Each `iter_N.pth` is a **directory**, not a file: 16 × `bf16_zero_pp_rank_*_optim_states.pt`
(~574 MB each) plus `mp_rank_00_model_states.pt` (~1.5 GB). A completed run also contains a
`TRAINING_COMPLETE` sentinel and a `last_checkpoint` file naming the newest checkpoint.

### HuggingFace conversions (~8.7 GB each)

In `hf_checkpoints/`. **Note the naming is misleading:**

| HF directory | Actually contains |
|---|---|
| `mainrun_final` | `sama_4b_event_v4/iter_382416` — the *single-node* run, not the headline |
| `multinode_baseline_final` | `sama_4b_event_v4_multinode_full/iter_127468` — **this is the headline model** |
| `multinode_best_final` | `sama_4b_event_v4_multinode_best/iter_127468` |
| `multinode_epoch1` | epoch-1 snapshot |
| `baseline1ep_seedA`, `baseline1ep_seedB`, `noadapter`, `noadapter_seed2`, `nopolarity`, `patchrandom`, `sam2frozen`, `sam2frozen_seed2`, `vitunfreeze`, `vitunfreeze_seed2`, `mevisonly` | the corresponding ablations |

**There is no HF conversion of `iter_133208`.** The fulldata checkpoint only exists as raw
DeepSpeed shards, because the old evaluation protocol consumes raw `.pth` directly and never needed
one. Convert it with:

```bash
sbatch training/configs/convert_final_models_to_hf.sbatch
```

The entry format in that launcher is `NAME:CONFIG_BASENAME:WORKDIR_NAME:ITER`, e.g.
`fulldata_final:sama_4b_event_v3_fulldata.py:sama_4b_event_v4_fulldata:133208`.

The HF export also carries `zero_to_fp32.py` in each run directory, DeepSpeed's own consolidation
script, if you prefer to go straight to fp32.

### Base weights (not produced here)

| | Source |
|---|---|
| `SAMA_4B` / `Sa2VA-4B_SAMA` | [sunye23/SAMA](https://github.com/sunye23/SAMA) |
| `sam2-hiera-large` | [SAM 2](https://github.com/facebookresearch/sam2) |
| `Qwen2.5-3B-Instruct` (inside SAMA-4B) | [Qwen](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct) |

## A caveat on parameter counts

An HF-exported checkpoint reports **764,913,015** parameters with `requires_grad=True`, but roughly
**441.6M of those are dead duplicate modules** inside `Sa2VaChatModel` — a second full SAM2
(224.4M), a second STC/projection stack, and a second `text_hidden_fcs` — never touched by the
training forward pass. The actually-trained total is **323,307,461**. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the per-module breakdown, and
`training/configs/probe_params.py` to re-derive it.
