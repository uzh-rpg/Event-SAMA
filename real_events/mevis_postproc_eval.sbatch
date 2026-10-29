#!/bin/bash
#SBATCH --job-name=mevis-postproc
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/real_events/mevis_postproc_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/real_events/mevis_postproc_%j.err

# Temporal-consistency post-processing of the headline model's MeViS valid_u predictions,
# evaluated with the official eval_mevis.py (GT available). Variants fixed a priori.
PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/container/event-sama-arm64.sqsh
RUN=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck/runs/mevis_keep_iter_133208_5103764
OUT=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/postproc_mevis
EXP=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/mevis/valid_u/meta_expressions_v2.json
MASK=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/mevis/valid_u/mask_dict.json
mkdir -p "$OUT"

# the in-container work is a separate script file so no nested quoting is needed
cat > "$OUT/inner.sh" <<INNER
#!/bin/bash
export PYTHONPATH=$PROJECT_DIR:$PROJECT_DIR/tools/eval_video_seg:\$PYTHONPATH
python3 /iopsstor/scratch/cscs/sfedorchenko/real_events/temporal_postproc.py --run_dir $RUN --out_root $OUT --procs 48
cd $PROJECT_DIR/tools/eval_video_seg
for V in v0 p2 p3 p5 p3g med3; do
  python3 eval_mevis.py --mevis_pred_path $OUT/\$V/Annotations --mevis_exp_path $EXP --mevis_mask_path $MASK --save_name x.json > $OUT/\$V.eval.log 2>&1 &
done
wait
INNER
chmod +x "$OUT/inner.sh"

srun --container-image=$SQSH \
     --container-mounts=/iopsstor:/iopsstor,/users:/users \
     --container-workdir=$PROJECT_DIR --no-container-mount-home \
     bash "$OUT/inner.sh"

echo "=== results (J&F per variant; v0 must equal the baseline 24.37) ==="
for V in v0 p2 p3 p5 p3g med3; do
  printf '%-5s ' "$V"
  if [ -f "$OUT/$V/mevis_result.json" ]; then tr -d '\n ' < "$OUT/$V/mevis_result.json"; else echo -n '(missing)'; fi
  echo
done
