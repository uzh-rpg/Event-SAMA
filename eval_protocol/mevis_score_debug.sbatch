#!/bin/bash
#SBATCH --job-name=mevis-dbg
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:10:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/pseudorgb/dbg_%j.out
P=/users/sfedorchenko/Event-SAMA-train
srun --container-image=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/container/event-sama-arm64.sqsh --container-mounts=/iopsstor:/iopsstor,/users:/users --no-container-mount-home bash -c "
export PYTHONPATH=$P:$P/tools/eval_video_seg NW=1; cd $P/tools/eval_video_seg
timeout 300 python3 /iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck/eval_mevis_keep.py --mevis_pred_path /iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/pseudorgb/run_mevis/Annotations --mevis_exp_path /iopsstor/scratch/cscs/sfedorchenko/sama_dataset/mevis/valid_u/meta_expressions_v2.json --mevis_mask_path /iopsstor/scratch/cscs/sfedorchenko/sama_dataset/mevis/valid_u/mask_dict.json --save_name x.json
" 2>&1 | tail -30
