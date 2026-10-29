#!/bin/bash
#SBATCH --job-name=mevis-eval-seq
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/real_events/mevis_eval_seq_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/real_events/mevis_eval_seq_%j.err

# Evaluate the six already-written post-processing variants with the official eval_mevis.py,
# ONE AT A TIME (the first attempt ran all six concurrently and timed out). Order: sanity
# check first (v0 must equal the baseline 24.37), then the main variants.
PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
SQSH=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/container/event-sama-arm64.sqsh
OUT=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/postproc_mevis
EXP=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/mevis/valid_u/meta_expressions_v2.json
MASK=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset/mevis/valid_u/mask_dict.json

cat > "$OUT/inner_seq.sh" <<INNER
#!/bin/bash
export PYTHONPATH=$PROJECT_DIR:$PROJECT_DIR/tools/eval_video_seg:\$PYTHONPATH
cd $PROJECT_DIR/tools/eval_video_seg
START=\$(date +%s)
for V in v0 p3g p3 p2 p5 med3; do
  if [ \$(( \$(date +%s) - START )) -gt 1560 ]; then echo "[stop] time budget reached before \$V"; break; fi
  T0=\$(date +%s)
  python3 eval_mevis.py --mevis_pred_path $OUT/\$V/Annotations --mevis_exp_path $EXP --mevis_mask_path $MASK --save_name x.json > $OUT/\$V.eval.log 2>&1
  echo "[\$V] done in \$(( \$(date +%s) - T0 )) s: \$(tr -d '\n ' < $OUT/\$V/mevis_result.json 2>/dev/null)"
done
INNER
chmod +x "$OUT/inner_seq.sh"

srun --container-image=$SQSH \
     --container-mounts=/iopsstor:/iopsstor,/users:/users \
     --container-workdir=$PROJECT_DIR --no-container-mount-home \
     bash "$OUT/inner_seq.sh"

echo "=== results (J&F per variant; v0 must equal the baseline 24.37) ==="
for V in v0 p3g p3 p2 p5 med3; do
  printf '%-5s ' "$V"
  if [ -f "$OUT/$V/mevis_result.json" ]; then tr -d '\n ' < "$OUT/$V/mevis_result.json"; else echo -n '(not evaluated)'; fi
  echo
done
