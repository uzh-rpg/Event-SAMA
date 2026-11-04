# Event-SAMA

**Language-Grounded Video Object Segmentation on Neuromorphic Event Streams**

Code release accompanying the Event-SAMA paper (VLM4RWD workshop @ NeurIPS 2026).

Event-SAMA adapts [SAMA](https://github.com/sunye23/SAMA) (Sun et al., NeurIPS 2025,
[arXiv:2505.18812](https://arxiv.org/abs/2505.18812)) from RGB video to **event streams**: the
model consumes 2-channel event voxels (ON/OFF polarity histograms) instead of RGB frames, and is
trained on a synthetic event conversion of SAMA's training corpus.

This repository contains **two halves** that together reproduce the paper:

| Directory | What it is |
|---|---|
| [`conversion/`](conversion/) | RGB → event conversion pipeline. Produces the synthetic event data (incl. **SAMA-239K-Event**). |
| [`training/`](training/) | The Event-SAMA model and training code — a fork of SAMA with an event modality. |
| [`eval_protocol/`](eval_protocol/) | MeViS / ReVOS evaluation harness. Every number in the paper comes from here. |
| [`real_events/`](real_events/) | Zero-shot evaluation on **real** event sensors (MouseSIS, DSEC, N-Caltech101). |

> **No datasets and no checkpoints are included.** This repository ships the conversion
> *algorithm* and pointers to the source data only — see [`docs/DATA.md`](docs/DATA.md).
> For model weights see [`docs/CHECKPOINTS.md`](docs/CHECKPOINTS.md).

## Documentation

Read these in order:

1. **[docs/DATA.md](docs/DATA.md)** — source datasets, where to get them, how to convert them to events.
2. **[docs/TRAINING.md](docs/TRAINING.md)** — the training recipe, configs, and SLURM launchers.
3. **[docs/EVALUATION.md](docs/EVALUATION.md)** — how to reproduce the paper's tables. **Read the
   protocol warning before comparing any numbers.**
4. **[docs/CHECKPOINTS.md](docs/CHECKPOINTS.md)** — which checkpoint is which, and what each scores.
5. **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** — what was changed relative to RGB SAMA, module by module.
6. **[docs/CLUSTER_PATHS.md](docs/CLUSTER_PATHS.md)** — **required reading before you run anything.**
   Paths throughout are hardcoded to the CSCS Alps cluster.

## What is actually new here

SAMA's architecture and the SAMA-239K annotations are **Sun et al.'s** contribution, inherited
verbatim. Event-SAMA's contribution is the event modality:

- **2-channel patch embedding.** `InternViT.patch_embedding` becomes `Conv2d(in_channels=2, ...)`,
  initialised by averaging the pretrained 3-channel RGB weights across the input dimension. This
  initialisation matters more than anything else in the model — randomising it costs **7.9 J&F**
  (see [docs/EVALUATION.md](docs/EVALUATION.md)).
- **Event-language adapter** (8.4M params), following EventGPT's recipe.
- **Polarity stem** (0.4M params) over the ON/OFF histogram.
- **Event dataset layer** — a parallel `*EventDataset` for each of SAMA's 25 training sources,
  reading event voxels from HDF5 instead of decoding RGB frames.

Everything else — the Versatile Spatial-Temporal-Context (STC) aggregator, the referential grounded
chat objective, SAM2 mask decoding — is SAMA's, unchanged.

Base model: **SAMA-4B** (Sa2VA-4B lineage) = InternViT vision tower + **Qwen2.5-3B-Instruct** LLM +
SAM2-hiera-large. 323M parameters are actually trained; see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Quick start

```bash
# 1. Convert a dataset to events (see docs/DATA.md for sources)
cd conversion
docker build -t event-sama .
docker run --gpus all -it --rm -v "$PWD":/workspace -v /path/to/data:/data event-sama
python convert.py --config configs/mevis.yaml --input /data/mevis --output /data/events --workers 8

# 2. Train (multi-node SLURM; adapt paths first -- see docs/CLUSTER_PATHS.md)
sbatch training/configs/train_fulldata.sbatch

# 3. Evaluate on MeViS valid_u
sbatch eval_protocol/datacheck_eval_param.sbatch <name> <config> <checkpoint.pth>
```

## Citation

If you find this work and/or code useful, cite our Event-SAMA paper:

```bibtex
@inproceedings{fedorchenko_zehnder_2026eventsama,
  author        = {Fedorchenko, Sergei and
                   Zehnder, Max and
                   Zubi{\'c}, Nikola and
                   Zou, Rong and
                   Scaramuzza, Davide},
  title         = {{Event-SAMA}: Language-Grounded Video Object Segmentation
                   on Neuromorphic Event Streams},
  booktitle     = {Conference on Neural Information Processing Systems
                   Workshops ({NeurIPSW})},
  year          = {2026},
}
```

The conversion pipeline builds on real-time event simulation, vid2e, FILM and DIS optical flow —
see [`conversion/README.md`](conversion/README.md) for those citations.

## License

Copyright 2026 Robotics and Perception Group, University of Zurich.

[Apache License 2.0](LICENSE), with two carve-outs:

- `conversion/vid2e_film/` is adapted from [vid2e](https://github.com/uzh-rpg/rpg_vid2e) and remains
  under **GNU GPL-3.0**.
- `training/third_parts/` vendors code from mmdet and SAM2, under their respective licenses.

Generated event data inherits the terms of its source dataset — see [docs/DATA.md](docs/DATA.md).
