"""List all top-level prefixes in the iter_3000 trainable-state shard so we
know what we need to convert to HF format."""
import sys, torch, collections
sys.path.insert(0, "/iopsstor/scratch/cscs/sfedorchenko/event_sama_pydeps")
p = "/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/sama_4b_event_v2/iter_3000.pth/mp_rank_00_model_states.pt"
blob = torch.load(p, map_location="cpu", weights_only=False)
sd = blob.get("module", blob) if isinstance(blob, dict) else blob
if not isinstance(sd, dict):
    sd = blob
keys = [k for k, v in sd.items() if isinstance(v, torch.Tensor)]
print(f"total trainable tensors in shard: {len(keys)}")
prefix_count = collections.Counter()
for k in keys:
    parts = k.split(".")
    if len(parts) >= 3:
        prefix_count[".".join(parts[:3])] += 1
    else:
        prefix_count[k] += 1

print("\ntop prefixes (3-deep):")
for p, c in prefix_count.most_common(40):
    print(f"  {c:>4d}  {p}")

print("\nspot-check for event_adapter / mlp1 / patch_embed:")
for needle in ("event_adapter", "mlp1", "patch_embedding", "Qformer_final_proj"):
    matches = [k for k in keys if needle in k]
    print(f"  '{needle}': {len(matches)} keys")
    for m in matches[:3]:
        print(f"      {m}")
