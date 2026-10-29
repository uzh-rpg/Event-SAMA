#!/bin/bash
#SBATCH --job-name=e2vid-base
#SBATCH --partition=normal
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/e2vid/base_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/e2vid/base_%j.err
ulimit -c 0
# Baseline asked by reviewers: events -> E2VID reconstruction (pretrained, default settings)
# -> UNMODIFIED RGB SAMA-4B (weights/SAMA_4B, its own eval script).
PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
EVAL_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck
CFG_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/container/event-sama-arm64.sqsh
B=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/pseudorgb
RE=/iopsstor/scratch/cscs/sfedorchenko/real_events
MV=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/mevis/valid_u
MODEL=/iopsstor/scratch/cscs/sfedorchenko/weights/SAMA_4B
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((22000 + SLURM_JOB_ID % 8000))
CM="--container-image=$SQSH --container-mounts=/iopsstor:/iopsstor,/capstor:/capstor,/users:/users --container-workdir=$PROJECT_DIR --no-container-mount-home"
ENVS='export PYTHONPATH='"$PROJECT_DIR"':'"$EVAL_DIR"':'"$CFG_DIR"':$PYTHONPATH TMPDIR=/tmp MPLBACKEND=Agg; cd '"$PROJECT_DIR"

E=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/e2vid
echo "=== E2VID reconstruction ==="
srun $CM bash -c "$ENVS"'
  python3 '"$EVAL_DIR"'/e2vid_reconstruct.py '"$E"'/rpg_e2vid '"$E"'/E2VID_lightweight.pth.tar '"$RE"'/mousesis_converted '"$RE"'/mousesis_converted/meta_expressions.json '"$E"'/mousesis $SLURM_PROCID 4
  python3 '"$EVAL_DIR"'/e2vid_reconstruct.py '"$E"'/rpg_e2vid '"$E"'/E2VID_lightweight.pth.tar /iopsstor/scratch/cscs/sfedorchenko/sama_converted_rescue/mevis '"$MV"'/meta_expressions_v2.json '"$E"'/mevis_valid_u $SLURM_PROCID 4
' 2>&1 | grep -v -i warning | tail -20
echo "frames: mousesis $(find $E/mousesis -name '*.jpg' | wc -l)  mevis $(find $E/mevis_valid_u -name '*.jpg' | wc -l)"

run_rgb () {  # $1 dataset  $2 out
  srun $CM bash -c "$ENVS"'
    export RANK=${SLURM_PROCID:-0} LOCAL_RANK=0 WORLD_SIZE=${SLURM_NTASKS:-1}
    python3 '"$EVAL_DIR"'/ref_vos_eval_st_pseudo.py '"$MODEL"' --dataset '"$1"' --launcher pytorch --work_dir '"$2"'
  ' 2>&1 | grep -v -i -E 'warning|libaio|CUTLASS|triton|pkg_resources|ds_accelerator' | tail -40
}
echo "=== RGB SAMA-4B on MouseSIS E2VID ==="
run_rgb MOUSESIS_E2VID $E/run_mousesis
srun --ntasks=1 --gpus-per-task=0 $CM bash -c "PYTHONPATH=$PROJECT_DIR/tools/eval_video_seg python3 $RE/eval_mousesis.py $E/run_mousesis $RE/mousesis_converted" 2>&1 | grep -v -i warning

echo "=== RGB SAMA-4B on MeViS valid_u E2VID ==="
run_rgb MEVIS_U_E2VID $E/run_mevis
echo "masks: $(find $E/run_mevis/Annotations -name '*.png' | wc -l)"
cp -r $E/run_mevis/Annotations $E/run_mevis/Annotations_keep
srun --ntasks=1 --gpus-per-task=0 $CM bash -c "export PYTHONPATH=$PROJECT_DIR:$PROJECT_DIR/tools/eval_video_seg; cd $PROJECT_DIR/tools/eval_video_seg; python3 eval_mevis.py --mevis_pred_path $E/run_mevis/Annotations --mevis_exp_path $MV/meta_expressions_v2.json --mevis_mask_path $MV/mask_dict.json --save_name x.json" 2>&1 | tail -15
cat $E/run_mevis/mevis_result.json 2>/dev/null
echo "=== all done ==="
