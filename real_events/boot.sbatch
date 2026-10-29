#!/bin/bash
#SBATCH --job-name=mouse-boot
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=00:20:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/real_events/boot_%j.out
RE=/iopsstor/scratch/cscs/sfedorchenko/real_events
cd $RE && PYTHONPATH=$RE/pylibs /users/sfedorchenko/miniforge3/bin/python bootstrap_mousesis.py agg_config.json mousesis_bootstrap.json | tee mousesis_bootstrap.md
