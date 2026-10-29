# Evaluation

## ⚠ Read this before comparing any number

**Two evaluation protocols exist in this repository and they score the same checkpoint ~6 J&F
apart. Rankings flip between them. Never mix numbers across protocols.**

| | "Old" protocol — **used for every number in the paper** | "New" protocol |
|---|---|---|
| Entry point | [`eval_protocol/ref_vos_event_eval_newdata.py`](../eval_protocol/ref_vos_event_eval_newdata.py) | [`training/projects/llava_sam2/evaluation/ref_vos_eval_event.py`](../training/projects/llava_sam2/evaluation/ref_vos_eval_event.py) |
| Keyframes | 32 uniform | per-frame |
| Event window | 10 ms | 10 ms |
| Mask propagation | nearest-keyframe | per-frame |
| Checkpoint format | raw DeepSpeed `.pth` | HuggingFace `safetensors` |
| Same checkpoint scores | ~22–25 J&F | ~14–16 J&F |

This was proven by controlled experiment — same checkpoint, same data, only the evaluation code
differs. **The data was never the issue.** The paper uses the old protocol throughout and says so
explicitly in its evaluation section.

## Reproducing the paper's numbers

MeViS `valid_u`, old protocol:

```bash
sbatch eval_protocol/datacheck_eval_param.sbatch <run-name> <config.py> <checkpoint.pth>
```

For the polarity-merge ablation you **must** export the same env var used in training, which is
what the dedicated launcher does:

```bash
sbatch eval_protocol/datacheck_eval_param_nopol.sbatch <run-name> <config.py> <checkpoint.pth>
```

ReVOS: `eval_protocol/revos_eval.sbatch`. Protocol sweeps: `eval_protocol/sweep_eval.sbatch`.

Results land as `mevis_result.json`. Those collected from the project's runs are in
[`eval_protocol/results/`](../eval_protocol/results/), named `<tag>_<slurm-job-id>.json`.

> A result of `{"J": 0.0, "F": 0.0, "J&F": 0.0}` means the job failed or was aborted, not that the
> model scored zero. Three such files are kept for completeness
> (`fulldata_final_iter_133208_5101606`, `sweep_K16_4523804`, `sweep_K64_4523809`).

## Main results (MeViS valid_u, old protocol)

| Model | Checkpoint | J | F | **J&F** |
|---|---|---|---|---|
| Event-SAMA, 4 ep, headline | `multinode_full/iter_127468` | 25.32 | 26.19 | **25.75** |
| Event-SAMA, 4 ep, single-node | `sama_4b_event_v4/iter_382416` | 24.85 | 26.02 | 25.44 |
| Event-SAMA, 4 ep, best-combo variant | `multinode_best/iter_127468` | 23.49 | 25.39 | 24.44 |
| **Event-SAMA, 4 ep, full data mix** | **`fulldata/iter_133208`** | 23.81 | 24.92 | **24.37** |
| Event-SAMA, 1 ep baseline (seed A) | `ablation_baseline1ep_seedA` | 23.14 | 25.26 | 24.20 |
| Event-SAMA, 1 ep baseline (seed B) | `ablation_baseline1ep_seedB` | 23.69 | 25.66 | 24.67 |

The headline 25.75 and the fulldata 24.37 are **not** directly comparable: the fulldata run trains
on strictly more data (VidSTG, Flickr and GranD-f were silently missing from every earlier run —
see [DATA.md](DATA.md)) but is a different, fixed mix. The fulldata checkpoint is the one all
real-sensor evaluation uses.

## Ablations

Two suites at two compute budgets. **Seed spread on the 1-epoch baseline is 0.47 J&F** (24.20 vs
24.67), which is the noise floor — small deltas below ~1 J&F are not interpretable at 1 epoch.

### 1-epoch iso-compute, current data roots (primary suite in the paper)

| Ablation | J&F | Δ vs baseline mean (24.44) |
|---|---|---|
| baseline, seed A / seed B | 24.20 / 24.67 | — (σ ≈ 0.47) |
| merge ON/OFF polarity | 24.96 | +0.52 (wash) |
| no event adapter | 25.11 | +0.67 (wash) |
| unfreeze last 6 ViT layers | 23.73 | −0.71 (wash) |
| frozen SAM2 decoder | 21.68 | −2.76 |
| MeViS-only data mix | 19.45 | −4.99 |
| random patch-embed init¹ | 18.71 | −4.33 |

¹ on the earlier data roots; no second seed was run, as −4.33 is far outside any plausible σ.

### 4-epoch fulldata budget (vs fulldata baseline 24.37)

| Ablation | J&F | Δ |
|---|---|---|
| merge ON/OFF polarity | 24.31 | −0.06 |
| no event adapter | 22.04 | −2.33 |
| frozen SAM2 decoder | 22.09 | −2.28 |
| unfreeze last 6 ViT layers | 21.87 | −2.50 |
| random patch-embed init | 16.51 | **−7.86** |

**The adapter and ViT-freezing effects that look like a wash at 1 epoch are clearly negative at 4
epochs.** Do not quote the 1-epoch "no reliable effect" rows as a conclusion about the released
model.

Claims that survive at ≥5× the noise floor: data mix (+5.0), patch-embed initialisation (+4.3 at
1 ep, +7.9 at 4 ep), trainable SAM2 decoder (+2.8), and 4-epoch over 1-epoch (+1.3).

The polarity result is a genuine finding rather than a null: collapsing ON/OFF to their mean costs
essentially nothing, i.e. **event density carries the signal, not polarity**.

## Protocol sweeps

On the 1-epoch baseline checkpoint (`eval_protocol/sweep_eval.sbatch`):

| Keyframes (10 ms window) | J&F |
|---|---|
| 8 | 20.64 |
| 16 | 21.93 |
| 32 | 23.04 |
| 64 | infeasible — SAM2 kernel grid limit |

| Window (32 keyframes) | J&F |
|---|---|
| 5 ms | 18.73 |
| 10 ms (training-matched) | 23.04 |
| **20 ms** | **24.59** |

Evaluating at 20 ms beats the training-matched 10 ms by **+1.55**. The asymmetry is informative:
too few events costs far more than too many, which suggests `event_fps=50` for future training.
The paper keeps the headline protocol at the training-matched 10 ms. If you change it, change it
for **every** row — the sweep rows above are not interchangeable with the main table.

## Real sensors (zero-shot)

[`real_events/`](../real_events/) evaluates the fulldata checkpoint on real event-camera data with
no fine-tuning: MouseSIS, DSEC and N-Caltech101. Conversion scripts (`convert_mousesis.py`,
`convert_dsec.py`, `convert_ncaltech.py`), inference launchers (`*_infer.sbatch`), scoring
(`score_all.sbatch`, `aggregate_mousesis.py`), bootstrap confidence intervals
(`bootstrap_mousesis.py`) and qualitative renders (`render_*.py`) are all included. Findings are
summarised in the `*.md` files in that directory.

Baselines for the same comparison — pseudo-RGB rendering into frozen SAM2, and E2VID
reconstruction — are in `eval_protocol/`: `pseudorgb_baseline.sbatch`,
`render_pseudorgb_frames.py`, `e2vid_baseline.sbatch`, `e2vid_reconstruct.py`.
