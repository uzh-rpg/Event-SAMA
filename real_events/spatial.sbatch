#!/bin/bash
#SBATCH --job-name=mouse-spatial
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/real_events/spatial_%j.out
RE=/iopsstor/scratch/cscs/sfedorchenko/real_events
cd $RE && PYTHONPATH=$RE/pylibs:/users/sfedorchenko/Event-SAMA-train/tools/eval_video_seg /users/sfedorchenko/miniforge3/bin/python spatial_gt_eval.py agg_config.json mousesis_spatial_gt.md
