# Event SAMA — Changes from RGB SAMA

> This document was written during development and describes the `training/` half of this
> repository. Two path references have been updated for the public layout; everything else is as
> written at the time.

`training/` is a fork of [`sunye23/SAMA`](https://github.com/sunye23/SAMA) (the NeurIPS-2025 SAMA
codebase). The goal is to consume **event voxels** (2-channel polarity histograms produced by the
prophesee/e2sim conversion pipeline in [`conversion/`](../conversion/)) instead of RGB video
frames.

We follow EventGPT's recipe for one specific thing — **the event-language
adapter** — and inherit everything else from SAMA. In particular, SAMA's
Versatile Spatial-Temporal-Context (STC) aggregator and SAMA-239K-style
referential grounded chat objective are kept verbatim.

## Architectural diff (concise)

```
event voxels [B*N, 2, H, W]
   │  (was [B*N, 3, H, W])
   ▼
InternViT.patch_embedding  Conv2d(in_channels=2, ...)   ◄── changed (was 3)
   │  weights initialised by averaging pretrained 3-ch weights across input dim
   ▼
ViT body (unchanged, RGB-pretrained)
   │
   ▼
F_frames [N, P, Dv]
   │
   ├── F_key ──► pixel_shuffle ──► mlp1 (RGB-aligned, frozen)
   │             │
   │             ▼
   │         event_adapter (NEW)  ── EventGPT-style adapter, identity at init
   │             │
   │             ▼
   │         LLM tokens
   │
   └── F_long ──► STC Aggregator (Q-Former, query-conditioned)
                  │
                  ▼
              Qformer_final_proj  (unchanged; trained from scratch in stage B)
                  │
                  ▼
              LLM tokens
```

The `event_adapter` is **only** wired into the F_key path for V1. The STC's
own `Qformer_final_proj` projection is trained from scratch during SAMA
instruction tuning anyway, so it learns the event-aligned distribution
organically. We can add a second adapter on the STC output if F_key vs F_agg
calibration starts diverging.

SAM2 is left RGB-native in V1 (the EventVoxel data loader renders a 3-channel
pseudo-image — R=OFF, B=ON, G=density — for SAM2's image encoder only). The
LLM never sees this rendering. V2 may swap this for paired RGB sister-frames
or fine-tune SAM2 on event renders.

## File-by-file changes

### Modified

* `projects/llava_sam2/hf/models/modeling_intern_vit.py`
  * `InternVisionEmbeddings.__init__`: `nn.Conv2d(in_channels=3, …)` →
    `nn.Conv2d(in_channels=config.num_channels, …)`. `num_channels` is read
    from `InternVisionConfig`, default 3 — RGB checkpoints continue to work
    untouched.
  * Added module-level `adapt_patch_embedding_state_dict(state_dict,
    target_in_channels, …)`: averages a pretrained 3-channel
    `patch_embedding.weight` across input channels and broadcasts the result
    to `target_in_channels`.
  * **NOTE on activation:** these edits live in *our local* copy of the
    modeling file. The active training path goes through
    `AutoModel.from_pretrained(model_path, trust_remote_code=True)`, which
    uses the modeling code shipped *inside the checkpoint directory* — not
    these files. The local edits are kept as forward-compatibility
    documentation for future event-native checkpoints; the runtime patch
    actually happens via post-construction surgery in `internvl_sama.py`
    (see below).

* `projects/llava_sam2/hf/models/modeling_sa2va_chat.py`
  * Imports `adapt_patch_embedding_state_dict` from the InternViT module.
  * `Sa2VAChatModel.__init__`: instantiates a `EventAdapter` as
    `self.event_adapter` if `config.use_event_adapter` is set
    (defaults to False).
  * `Sa2VAChatModel.extract_feature`: applies `self.event_adapter` after
    `self.mlp1` if the adapter exists.
  * Same activation caveat as above — for the live training path we *wrap*
    `self.model.mlp1` instead, so the change does not depend on which copy
    of the modeling file is in use.

* `projects/llava_sam2/models/internvl_sama.py`
  * `InternVL_Slowfast.__init__` now accepts
    `override_vision_in_channels`, `use_event_adapter`,
    `event_adapter_expansion`, `event_adapter_dropout`. After the RGB
    Sa2VA checkpoint is loaded by `AutoModel.from_pretrained`, two
    surgeries run:
    * `_adapt_patch_embedding_in_channels(N)` — replaces
      `self.model.vision_model.embeddings.patch_embedding` with a fresh
      `Conv2d(in_channels=N)` whose weights come from averaging the loaded
      3-ch weights across the input dim and broadcasting to N. Bias is
      preserved verbatim.
    * `_attach_event_adapter(...)` — wraps `self.model.mlp1` with
      `nn.Sequential(original_mlp1, EventAdapter)`. The adapter is identity
      at init (zero-init last linear) so RGB-style checkpoints behave
      unchanged at step 0. We wrap rather than monkey-patch
      `extract_feature` because the latter only fires for our local
      modeling code, not the trust_remote_code path.

* `projects/llava_sam2/models/llava_sam2.py`
  * Added `_drop_shape_mismatched_keys(state_dict, model)`: filters out any
    keys whose tensor shape does not match the model parameter, so the
    `pretrained_pth` re-load (which carries the original 3-ch
    `patch_embedding.weight` from Sa2VA safetensors) does not clobber our
    surgical 2-ch replacement. Called for both the `.pth` and Sa2VA-safetensors
    branches in `VideoLLaVASAMModel.__init__`.

* `projects/llava_sam2/models/__init__.py`
  * Exports `EventAdapter`.

* `projects/llava_sam2/datasets/__init__.py`
  * Exports `VideoEventBaseDataset`, `VideoMeVISEventDataset`,
    `EventVoxelTransform`.

### Added

* `projects/llava_sam2/models/event_adapter.py`
  * `EventAdapter` — LayerNorm → Linear → GELU → Linear with residual
    connection. The second Linear is zero-initialised so the adapter is the
    identity function at step 0. Mirrors EventGPT's "event-language adapter"
    (arXiv:2412.00832 §4 / §5.1) but with a residual MLP for capacity instead
    of a single Linear.

* `projects/llava_sam2/datasets/EventVoxel_Dataset.py`
  * `_accumulate_polarity_histogram(...)` — the inner accumulator. Given an
    event stream and a `[start_us, end_us)` window, returns a `(2, H, W)`
    float32 tensor where ch0 is the ON-event count and ch1 is the OFF-event
    count.
  * `EventVoxelTransform` — resize via bilinear interpolation, then
    `asinh + per-clip z-score` normalisation. asinh is used (instead of
    `log1p`) because it preserves the property that 0-event pixels map to
    exactly 0, while compressing the long tail of bright pixels.
  * `VideoEventBaseDataset` — base class. Subclasses `VideoReVOSDataset` so
    we reuse its expression / mask / tokenization machinery, and overrides
    `__getitem__` to load HDF5 events instead of JPEGs. Maps RGB frame
    indices (from the existing MeViS metadata) to microsecond timestamps via
    the clip's `native_fps`, then accumulates events in
    `[t − window/2, t + window/2)` per sampled frame.
  * `VideoMeVISEventDataset` — concrete class, multiple-inheritance with
    `VideoMeVISDataset` so the parent class identity remains MeViS for any
    downstream `isinstance` checks.

* `projects/llava_sam2/configs/sama_4b_event.py`
  * Event analog of `sama_4b_rgb.py`. MeViS-only train mix (the only
    converted source). Sets `EVENT_FPS=100`, `NUM_INPUT_CHANNELS=2`,
    `USE_EVENT_ADAPTER=True`. Two-stage knob `training_stage='A'|'B'`:
    * **A**: alignment-only — patch_embedding + event_adapter trainable.
    * **B**: full instruction tuning — STC aggregator + SAM2 decoder +
      event_adapter trainable; LLM via LoRA.

## Things deliberately deferred to V2

1. **SAM2 on events.** Currently SAM2's image encoder consumes a pseudo-RGB
   rendering of the voxel. Real fixes: feed paired RGB sister-frames, or
   fine-tune SAM2 on event-derived input.
2. **Event adapter on the STC output.** Only the F_key path gets the adapter
   for now. If F_key vs F_agg distribution drift causes issues, mirror the
   adapter on `Qformer_final_proj` in `internvl_sama.py`.
3. **Other event sources.** Only MeViS is converted. Other SAMA-239K sources
   (Ref-YouTube-VOS, LV-VIS, SAV, VidSTG) need to run through the prophesee
   pipeline first.
4. **Stage-A trainable-parameter mask.** The two-stage training plan calls
   for stage A (alignment) to train *only* the new patch_embedding and
   `event_adapter`. The current `freeze_visual_encoder=True` flag freezes
   the entire vision tower including patch_embedding — we need a runner-level
   hook that re-enables `requires_grad_=True` on `patch_embedding` and on
   `event_adapter` after the freeze sweep. Easy fix: do it in
   `VideoLLaVASAMModel.__init__` for `training_stage='A'`.

## Pointers for future work

* The conversion pipeline writes `t` in microseconds and `t_unit='us'`
  (`prophesee_pipeline/scripts/convert.py` line 638). Do not change this —
  the dataset relies on it.
* MeViS `meta_expressions.json` provides frame names like `"00000123"` —
  numeric strings that we cast to int and multiply by `1e6 / native_fps`.
  Other sources may not have numeric frame names; for those, we fall back to
  using the position in the sorted frame list (see
  `_frame_index_to_us` in `EventVoxel_Dataset.py`).
* The `event_fps` knob trades temporal resolution against event density per
  frame. The user's requested default is 100 (10 ms windows). Lower values
  (50 → 20 ms; 24 → ~42 ms) increase event count per frame and may help when
  motion is slow.

---

## Trainable parameter breakdown

Verified against the released model (`training/configs/probe_params.py`). The HF export reports
764,913,015 parameters with `requires_grad=True`, but ~441.6M of those are **dead duplicate
modules** inside `Sa2VaChatModel` — a second full SAM2 (224.4M), a second STC/projection stack, and
a second `text_hidden_fcs` — never reached by the training forward pass.

Actually trained: **323,307,461**.

| Module | Params | Note |
|---|---|---|
| STC aggregator | 288.78M | SAMA's, inherited |
| `mlp1` (vision→LLM projection) | 12.60M | **trained** |
| Event adapter | 8.40M | new in Event-SAMA |
| `text_hidden_fcs` | 4.72M | 2048→2048→256, into SAM2 prompt space |
| SAM2 mask decoder | 4.22M | |
| `Qformer_final_proj` | 4.20M | |
| Polarity stem | 0.40M | new in Event-SAMA |

Frozen: the LLM, and the vision encoder except `patch_embedding`.

### `[SEG]` token

`[SEG]` is **already present in the SAMA-4B vocabulary** (id 151674) — re-adding it is a no-op. The
embedding and LM head are frozen, so the only trained path from `[SEG]` to a mask is
`text_hidden_fcs`.

### Which LLM this actually is

The LLM is **Qwen2.5-3B-Instruct** (`Qwen2ForCausalLM`, hidden size 2048, 36 layers, vocab 151,679).
The `phi3_chat` prompt-template name is an inherited Sa2VA artifact and does **not** mean Phi-3 is
in the stack.
