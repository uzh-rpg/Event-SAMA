# Data

**No dataset is redistributed in this repository.** What ships here is the conversion
*algorithm* ([`conversion/`](../conversion/)) plus the pointers below. You download the RGB
sources from their original providers under their original terms, then convert them to events
yourself.

Generated event data inherits the license of its source dataset. Check each source's terms before
redistributing anything you produce.

## 1. RGB sources

Event-SAMA trains on an event conversion of SAMA's training corpus. The mix is defined in
[`training/configs/sama_4b_event_v3_fulldata.py`](../training/configs/sama_4b_event_v3_fulldata.py)
(the `datasets=[...]` list of the `ConcatDataset`).

### Annotations specific to SAMA

| Source | Where |
|---|---|
| **SAMA-239K** (region description + multi-turn region conversation) | [sunye23/SAMA](https://github.com/sunye23/SAMA) — Sun et al., NeurIPS 2025. Built over LV-VIS / MeViS / Ref-YouTube-VOS videos. |

### Video referring segmentation / VIS

| Source | Where |
|---|---|
| MeViS | https://github.com/henghuiding/MeViS |
| ReVOS (incl. UVO / OVIS / TAO sub-sources) | https://github.com/cilinyan/ReVOS |
| Ref-YouTube-VOS | https://youtube-vos.org/dataset/rvos/ |
| LV-VIS | https://github.com/haochenheheda/LVVIS |
| SA-V (pseudo-label RES) | https://ai.meta.com/datasets/segment-anything-video/ |

### Video QA

| Source | Where |
|---|---|
| Chat-UniVi instruction data | https://github.com/PKU-YuanGroup/Chat-UniVi |

### Image referring segmentation and grounded conversation

| Source | Where |
|---|---|
| RefCOCO / RefCOCO+ / RefCOCOg | https://github.com/lichengunc/refer (images: [COCO](https://cocodataset.org)) |
| GranD-f | https://github.com/mbzuai-oryx/groundingLMM |
| Flickr30k Entities | https://github.com/BryanPlummer/flickr30k_entities |
| OpenPSG | https://github.com/Jingkang50/OpenPSG |
| LLaVA instruction data | https://github.com/haotian-liu/LLaVA |
| VidSTG | https://github.com/Guaranteer/VidSTG-Dataset |

### Real event sensors (evaluation only, never trained on)

Used zero-shot by [`real_events/`](../real_events/):

| Source | Where |
|---|---|
| MouseSIS | https://github.com/tub-rip/MouseSIS |
| DSEC / DSEC-Semantic | https://dsec.ifi.uzh.ch/ |
| N-Caltech101 | https://www.garrickorchard.com/datasets/n-caltech101 |

## 2. Converting RGB → events

See [`conversion/README.md`](../conversion/README.md) for the authoritative instructions. In short:

```bash
cd conversion
docker build -t event-sama .

# FILM interpolation weights (only needed for the Chat-UniVi config)
wget https://rpg.ifi.uzh.ch/data/VID2E/pretrained_models.zip -O /tmp/pretrained_models.zip
unzip /tmp/pretrained_models.zip 'pretrained_models/film_net/Style/*' -d /tmp
mv /tmp/pretrained_models/film_net vid2e_film/film/

docker run --gpus all -it --rm -v "$PWD":/workspace -v /path/to/data:/data event-sama
python convert.py --config configs/mevis.yaml \
    --input /data/mevis --output /data/events \
    --workers 8 --joblist joblist.txt --tmpdir /tmp/film --eval eval/mevis
```

One YAML per source lives in [`conversion/configs/`](../conversion/configs/) — these are the exact
configs used to produce the paper's data:

```
chatunivi_film.yaml  coco.yaml      grandf.yaml   llava.yaml
lvvis.yaml           mevis.yaml     refytvos.yaml revos_ovis.yaml
revos_tao.yaml       revos_uvo.yaml sav_videos.yaml vidstg.yaml
```

Output layout is one `events.h5` per clip, which is what the `*EventDataset` classes in
[`training/projects/llava_sam2/datasets/`](../training/projects/llava_sam2/datasets/) expect.

### Reference renders

[`conversion/eval/`](../conversion/eval/) holds one reference visualisation per source. Use
`conversion/tools/run_eval.sh` and `conversion/tools/visualize.py` to confirm your conversion
matches before spending GPU-weeks on training.

## 3. Event representation

The model does not consume raw event streams. Events are accumulated into **2-channel polarity
histograms** (ON and OFF counts per pixel) over a fixed time window, built on the fly by
[`training/projects/llava_sam2/datasets/EventVoxel_Dataset.py`](../training/projects/llava_sam2/datasets/EventVoxel_Dataset.py).

Two parameters matter and are worth restating because they are easy to get wrong:

- **Accumulation window** — training used **10 ms**. Evaluation at 20 ms scores *higher*
  (+1.55 J&F), because too few events costs far more than too many. See
  [EVALUATION.md](EVALUATION.md).
- **Keyframes per clip** — training and headline evaluation use **32**.

`EVENT_MERGE_POLARITY=1` is an ablation switch that collapses ON/OFF to their mean. It is
default-off and bit-identical to baseline when unset. If you set it for training you **must** also
set it for evaluation.

## 4. Known data caveats

These are real and documented rather than silently fixed, because the released checkpoints were
trained under them:

- **Pre-filter drops.** The `iter_133208` (fulldata) run kept 27,461/27,463 region-description and
  46,709/46,713 region-conversation items. An earlier run kept only 14,818 and 24,901 respectively
  — eight config roots had not been repointed after the event data was migrated between
  filesystems, so runs before 2026-08-28 silently trained *without* VidSTG, Flickr and GranD-f.
  `sama_4b_event_v3_fulldata.py` is the fixed config; everything else in
  `training/configs/` predates the fix.
- **ReVOS sub-sources** (UVO / OVIS / TAO) remained degraded in the conversion used for all runs.
- Iteration counts follow from the mix: the fulldata recipe is 1,598,496 samples/epoch at global
  batch 48 → 133,208 iterations over 4 epochs.
