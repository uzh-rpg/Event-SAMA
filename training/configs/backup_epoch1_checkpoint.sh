#!/bin/bash
# Watches the multinode full-production run's WORK_DIR for the first
# checkpoint at/after the 1-epoch mark (iter 33303) and copies it to a
# non-rotating backup location before max_keep_ckpts=10 rotation can
# delete it. Runs until found or 20h elapsed (safety cap).
WORK_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/sama_4b_event_v4_multinode_full
BACKUP_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/epoch_checkpoints
TARGET_ITER=34000
DEADLINE=$(( $(date +%s) + 20*3600 ))
while [ $(date +%s) -lt $DEADLINE ]; do
  if [ -d "${WORK_DIR}/iter_${TARGET_ITER}.pth" ]; then
    echo "[backup] found iter_${TARGET_ITER}.pth, copying..."
    cp -r "${WORK_DIR}/iter_${TARGET_ITER}.pth" "${BACKUP_DIR}/multinode_full_epoch1_iter${TARGET_ITER}.pth"
    echo "[backup] done: ${BACKUP_DIR}/multinode_full_epoch1_iter${TARGET_ITER}.pth"
    exit 0
  fi
  sleep 300
done
echo "[backup] TIMED OUT waiting for iter_${TARGET_ITER}.pth"
exit 1
