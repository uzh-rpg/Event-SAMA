"""Verify which model parts are actually being trained by diffing the
mp_rank_00_model_states.pt of consecutive checkpoints.

Frozen parameters -> ||Δ|| == 0 between two ckpts.
Trainable parameters -> ||Δ|| > 0 and grows with iter gap.

We compare a few categories:
  * event_adapter                    -> our addition, should be trainable
  * vision_model.embeddings.patch_embedding -> reshaped 3->2 channels, should be trainable
  * language_model.lm_head           -> freeze_llm=True, MUST be 0
  * grounding_encoder.sam2_model.image_encoder.trunk.patch_embed -> SAM2 image encoder, frozen by default in SAMA recipe (we'll see)
  * Qformer_final_proj               -> trainable Qformer ref
"""
import os
import torch

ROOT = "/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/sama_4b_event_v2"
CKPT_ITERS = (1000, 2000, 3000)
KEYS_OF_INTEREST_SUBSTR = (
    "event_adapter",
    "vision_model.embeddings.patch_embedding",
    "language_model.lm_head",
    "grounding_encoder.sam2_model.image_encoder.trunk.patch_embed",
    "Qformer_final_proj",
)


def load_partial(p):
    blob = torch.load(p, map_location="cpu", weights_only=False)
    sd = blob.get("module", blob) if isinstance(blob, dict) else blob
    if not isinstance(sd, dict):
        sd = blob
    out = {}
    for k, v in sd.items():
        if not isinstance(v, torch.Tensor):
            continue
        if any(s in k for s in KEYS_OF_INTEREST_SUBSTR):
            out[k] = v.float()
    return out


def cat_delta(sdA, sdB, sub):
    common = [k for k in sdA if k in sdB and sub in k]
    if not common:
        return None
    diffs, norms = [], []
    for k in common:
        a, b = sdA[k], sdB[k]
        if a.shape != b.shape:
            continue
        d = (a - b).pow(2).sum().item()
        n = b.pow(2).sum().item()
        diffs.append(d)
        norms.append(n)
    if not diffs:
        return None
    L2 = sum(diffs) ** 0.5
    Nref = sum(norms) ** 0.5
    return L2, Nref, len(diffs)


def main():
    snaps = []
    for it in CKPT_ITERS:
        p = os.path.join(ROOT, f"iter_{it}.pth/mp_rank_00_model_states.pt")
        if os.path.exists(p):
            print(f"loading iter_{it} ...", flush=True)
            snaps.append((it, load_partial(p)))
    print(f"found {len(snaps)} checkpoints with matched tensors")
    for it, sd in snaps:
        cats = {}
        for k in sd:
            for sub in KEYS_OF_INTEREST_SUBSTR:
                if sub in k:
                    cats.setdefault(sub, 0)
                    cats[sub] += 1
                    break
        print(f"  iter_{it}: {len(sd)} matched tensors")
        for n, c in cats.items():
            print(f"     {n}: {c}")

    print("\n=== L2 deltas between checkpoints ===")
    print(f"{'category':55s} {'iters':>12s}  {'||Δ||':>11s}  {'||W||':>11s}  {'rel':>9s}  {'#':>3s}")
    for i in range(len(snaps) - 1):
        itA, sdA = snaps[i]
        itB, sdB = snaps[i + 1]
        for sub in KEYS_OF_INTEREST_SUBSTR:
            r = cat_delta(sdA, sdB, sub)
            if r is None:
                print(f"  {sub:55s}            (no matching tensors)")
                continue
            L2, Nref, n = r
            rel = L2 / (Nref + 1e-12)
            print(
                f"  {sub:55s} {itA:>4d}->{itB:<4d}  {L2:11.4e}  {Nref:11.4e}  {rel:9.2e}  {n:3d}"
            )


if __name__ == "__main__":
    main()
