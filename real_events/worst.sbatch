#!/bin/bash
#SBATCH --job-name=mouse-worst
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=150G
#SBATCH --time=00:25:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/real_events/worst_%j.out
RE=/iopsstor/scratch/cscs/sfedorchenko/real_events
cd $RE && PYTHONPATH=$RE/pylibs /users/sfedorchenko/miniforge3/bin/python worst_far_right.py agg_config.json "Event-SAMA 150 ms" 150 $RE/far_right_worst_gifs 3 1
