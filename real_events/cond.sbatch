#!/bin/bash
#SBATCH --job-name=mouse-cond
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:20:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/real_events/cond_%j.out
RE=/iopsstor/scratch/cscs/sfedorchenko/real_events
cd $RE && PYTHONPATH=$RE/pylibs /users/sfedorchenko/miniforge3/bin/python cond_select.py agg_config.json x.json | tee mousesis_cond_select.md
