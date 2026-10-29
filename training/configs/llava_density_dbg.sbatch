#!/bin/bash
#SBATCH --job-name=llava-density
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:29:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/llava_density_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/llava_density_%j.err
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event-sama-arm64.sqsh
srun --container-image=${SQSH} \
  --container-mounts=/capstor/scratch/cscs/mzehnde:/capstor/scratch/cscs/mzehnde,/iopsstor/scratch/cscs/sfedorchenko:/iopsstor/scratch/cscs/sfedorchenko \
  --no-container-mount-home \
  python /iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs/llava_density.py 1000
