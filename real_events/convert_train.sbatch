#!/bin/bash
#SBATCH --job-name=mouse-train-conv
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/real_events/convert_train_%j.out
ulimit -c 0
# Convert the MouseSIS train split (public GT; never used for training) with the same rules as val.
RE=/iopsstor/scratch/cscs/sfedorchenko/real_events
export PYTHONPATH=$RE/pylibs
srun bash -c "/users/sfedorchenko/miniforge3/bin/python $RE/convert_mousesis.py $RE/mousesis_train_raw $RE/mousesis_train_converted train_annotations.json \$SLURM_PROCID 8"
/users/sfedorchenko/miniforge3/bin/python - <<PY
import json, glob
out = "$RE/mousesis_train_converted"
v, st = {}, {}
for f in sorted(glob.glob(out + "/meta_expressions_shard*.json")): v.update(json.load(open(f))["videos"])
for f in sorted(glob.glob(out + "/stats_shard*.json")): st.update(json.load(open(f)))
json.dump({"videos": v}, open(out + "/meta_expressions.json", "w"), indent=1)
json.dump(st, open(out + "/stats.json", "w"), indent=1)
print("merged", len(v), "clips;", sum(len(x["frames"]) for x in v.values()), "frames")
for k, s in sorted(st.items()): print(k, round(s["n_events"] / (921600 * s["n_img"] * s["step_us"] / 1e6), 2), "ev/px/s", "hot_ev", round(s["hot_ev_frac"], 3))
PY
