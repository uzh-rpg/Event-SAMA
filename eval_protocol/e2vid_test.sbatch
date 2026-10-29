#!/bin/bash
#SBATCH --job-name=e2vid-test
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=00:20:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/e2vid/test_%j.out
ulimit -c 0
# Smoke test of e2vid_reconstruct.py on one MouseSIS clip and one MeViS video (shard 0 of 50).
PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
EVAL_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/container/event-sama-arm64.sqsh
E=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/e2vid
RE=/iopsstor/scratch/cscs/sfedorchenko/real_events
MV=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/mevis/valid_u
srun --container-image=$SQSH --container-mounts=/iopsstor:/iopsstor,/users:/users --no-container-mount-home bash -c "
  cd /tmp
  time python3 $EVAL_DIR/e2vid_reconstruct.py $E/rpg_e2vid $E/E2VID_lightweight.pth.tar $RE/mousesis_converted $RE/mousesis_converted/meta_expressions.json $E/test_mousesis 3 4
  time python3 $EVAL_DIR/e2vid_reconstruct.py $E/rpg_e2vid $E/E2VID_lightweight.pth.tar /iopsstor/scratch/cscs/sfedorchenko/sama_converted_rescue/mevis $MV/meta_expressions_v2.json $E/test_mevis 0 50
" 2>&1 | grep -v -i warning | tail -20
