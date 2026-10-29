#!/bin/bash
#SBATCH --job-name=mevis-score
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/pseudorgb/score_%j.out
ulimit -c 0
# Official eval_mevis.py scoring (copy without the final rmtree of the prediction dir).
# usage: sbatch eval_mevis_dir.sbatch <run_dir containing Annotations>
RUN=$1
PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/container/event-sama-arm64.sqsh
MV=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/mevis/valid_u
EV=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck
srun --container-image=$SQSH --container-mounts=/iopsstor:/iopsstor,/users:/users --no-container-mount-home bash -c "
  export PYTHONPATH=$PROJECT_DIR:$PROJECT_DIR/tools/eval_video_seg; cd $PROJECT_DIR/tools/eval_video_seg
  python3 $EV/eval_mevis_keep.py --mevis_pred_path $RUN/Annotations --mevis_exp_path $MV/meta_expressions_v2.json --mevis_mask_path $MV/mask_dict.json --save_name x.json
" 2>&1 | grep -v -i warn | tail -15
echo "== result"; cat $RUN/mevis_result.json 2>/dev/null; ls $RUN
