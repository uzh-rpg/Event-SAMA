#!/bin/bash
#SBATCH --job-name=mevis-score
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=400G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck/score_node_%j.out
# eval_mevis_keep.py (official scoring, no rmtree) with miniforge python + pylibs on a compute node.
EV=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck
sed "s/NW=16/NW=128/" $EV/score_mevis_login.sh > $EV/score_mevis_node.sh && chmod +x $EV/score_mevis_node.sh
srun $EV/score_mevis_node.sh "$@"
