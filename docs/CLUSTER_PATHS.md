# Adapting the cluster paths

Everything in `training/configs/`, `eval_protocol/` and `real_events/` was written to run on the
**CSCS Alps** supercomputer (GH200 nodes, SLURM, enroot/pyxis containers). Paths are hardcoded —
1,267 absolute path references across 215 files.

**These were deliberately left as-is.** Rewriting them would mean the released code no longer
matches, byte for byte, the code that produced the published checkpoints and numbers. Substitute
them for your own site instead.

## Path roots to substitute

| Hardcoded root | What lives there | Replace with |
|---|---|---|
| `/users/sfedorchenko/Event-SAMA-train` | the training code — i.e. this repo's `training/` | your checkout's `training/` |
| `/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs` | run outputs: work dirs, checkpoints, configs, eval | your scratch run root |
| `/iopsstor/scratch/cscs/sfedorchenko/weights` | base model weights (`SAMA_4B`, `sam2-hiera-large`) | wherever you download base weights |
| `/iopsstor/scratch/cscs/sfedorchenko/sama_dataset` | RGB source datasets + MeViS annotations | your dataset root |
| `/capstor/scratch/cscs/mzehnde/sama_converted*` | **converted event data** (several generations) | your conversion output root |
| `/iopsstor/scratch/cscs/mzehnde/sama_converted` | converted event data (second filesystem) | same |
| `/iopsstor/scratch/cscs/sfedorchenko/real_events` | real-sensor data and outputs | your real-sensor root |
| `/users/sfedorchenko/miniforge3/bin/python` | the login-node Python used by a few helper scripts | your interpreter |

A blunt first pass:

```bash
grep -rl '/scratch/cscs/' training/configs eval_protocol real_events \
  | xargs sed -i \
      -e 's|/users/sfedorchenko/Event-SAMA-train|/my/checkout/training|g' \
      -e 's|/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs|/my/runs|g' \
      -e 's|/iopsstor/scratch/cscs/sfedorchenko/weights|/my/weights|g' \
      -e 's|/iopsstor/scratch/cscs/sfedorchenko/sama_dataset|/my/data|g'
```

The converted-event roots need more care than `sed`: several config generations coexist
(`sama_converted`, `sama_converted_v2`, `sama_converted_transfer`, `sama_converted_revos`) and
**which root a given dataset entry points at is load-bearing** — mismatched roots are exactly what
caused the silent data loss documented in [DATA.md](DATA.md). Check every per-dataset
`*_event_root` in your config individually and verify item counts before training; use
`training/configs/diag_sama239k_sources_fulldata.py` to print per-source kept/total counts.

## SLURM assumptions

The launchers assume:

- `--account=uzh55`, partitions `normal` and `debug` (debug allows max 2 queued jobs per user)
- 4 GPUs per node, 4 tasks per node, 1 GPU per task, 32 CPUs per task, 640 GB per node
- 24-hour wall clock, worked around by self-chaining (see [TRAINING.md](TRAINING.md))
- container execution through `srun --container-image=<image>.sqsh --container-mounts=...`
  with `--no-container-mount-home`
- `MASTER_ADDR` from `hostname`, `MASTER_PORT` derived from the job ID
- `RANK=$SLURM_PROCID`, `LOCAL_RANK=0`, `WORLD_SIZE=$SLURM_NTASKS` — one GPU per task, so
  `LOCAL_RANK` is deliberately pinned to 0

On a non-SLURM machine, ignore the `.sbatch` files and launch
`training/configs/train_no_fa2_pinned.py` under `torchrun` with the same config, DeepSpeed JSON and
`--work-dir`.

## Container image

Training and evaluation ran inside a prebuilt ARM64 image (`event-sama-arm64.sqsh`) based on the
NVIDIA PyTorch NGC container. It is not redistributed here. Use
`training/requirements.txt` to rebuild the training environment, and `conversion/Dockerfile` for
the conversion half. Two settings from the launchers are load-bearing:

```
TRANSFORMERS_NO_FLASH_ATTN_2=1   # the stack's FlashAttention-2 path is broken on this image
HF_HOME, TRANSFORMERS_CACHE, ... # redirected into /tmp, since the container does not mount $HOME
```

## Things that are not paths but will still bite

- `data/` in the original training checkout was a set of symlinks into cluster scratch. They were
  dangling outside the cluster and have been removed here; recreate them to point at your dataset
  root, matching the names used in the configs (`glamm_data`, `llava_data`, `osprey-724k`,
  `ref_seg`, `video_datas`).
- `training/resources/` (176 MB of upstream SAMA teaser images) was excluded to keep the repository
  clonable. Fetch it from [sunye23/SAMA](https://github.com/sunye23/SAMA) if you need it.
- `eval_protocol/results/` contains only the small result JSONs. The per-video predicted masks and
  job logs (~237 MB) were not retained.
