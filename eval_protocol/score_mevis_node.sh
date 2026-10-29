#!/bin/bash
# Score MeViS valid_u runs with eval_mevis_keep.py on the login node (container lacks skimage).
cd /users/sfedorchenko/Event-SAMA-train/tools/eval_video_seg
export PYTHONPATH=/iopsstor/scratch/cscs/sfedorchenko/real_events/pylibs:$PWD NW=128
MV=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/mevis/valid_u
EV=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck
for R in "$@"; do
  echo "== $R"
  /users/sfedorchenko/miniforge3/bin/python $EV/eval_mevis_keep.py --mevis_pred_path $R/Annotations --mevis_exp_path $MV/meta_expressions_v2.json --mevis_mask_path $MV/mask_dict.json --save_name x.json 2>&1 | grep -v -i warn | tail -3
  cat $R/mevis_result.json
done
