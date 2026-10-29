#!/bin/bash
# Select samples by a fixed rule (first sample of every other class, alphabetical, first 10)
# + the 2 lowest-IoU samples (labelled worst), render GIFs from the primary pass.
RE=/iopsstor/scratch/cscs/sfedorchenko/real_events
RUN=$(ls -d $RE/runs/ncalt_hf10ms_* | head -1)
CONV=$RE/ncaltech_converted_hf
OUT=$RE/animal_gifs
mkdir -p $OUT
PY="/users/sfedorchenko/miniforge3/bin/python"
export PYTHONPATH=$RE/pylibs
RULE="beaver_0001 crab_0001 dalmatian_0001 dragonfly_0001 emu_0001 hedgehog_0001 kangaroo_0001 llama_0001 octopus_0001 panda_0001"
WORST=$($PY - <<PY
import json
d = json.load(open("$RUN/ncalt_eval.json"))["rows"]
ok = [r for r in d if r.get("pos")]
ok.sort(key=lambda r: r["pos"]["iou"])
print(" ".join(r["vid"] for r in ok[:2]))
PY
)
echo "rule: $RULE"
echo "worst: $WORST"
$PY $RE/render_ncalt_gifs.py $RUN $CONV $OUT $RULE $WORST
echo "$RULE" > $OUT/rule.txt
echo "$WORST" > $OUT/worst.txt
