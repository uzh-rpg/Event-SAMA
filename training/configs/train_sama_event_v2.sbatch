#!/bin/bash
#SBATCH --job-name=sama-event-v2
#SBATCH --partition=normal
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=460G
#SBATCH --time=24:00:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/sama_event_v2_%j.out
#SBATCH --error=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/sama_event_v2_%j.err

# Event SAMA V2: 4-way DDP on a single node, BULLETPROOF GPU isolation.
#
# History of failed attempts that motivated this layout:
#   * sbatch ntasks=1 + torchrun nproc=4 + launcher=pytorch: every rank
#     defaulted to cuda:0 because torchrun left CUDA_VISIBLE_DEVICES
#     unrestricted and several bits of CUDA work happened before
#     mmengine's later set_device(local_rank); 4 ranks piled on GPU 0
#     (~80GB of cross-rank pollution) and OOM'd.
#   * Same as above + early torch.cuda.set_device(local_rank) in a
#     wrapper: helped a little but cross-rank pollution still happened
#     (NCCL/deepspeed init reach across visible GPUs).
#
# Real fix used here: SLURM cgroups physically restrict each task to one
# GPU. With --ntasks-per-node=4 --gpus-per-task=1, every rank only
# *sees* one device (CUDA_VISIBLE_DEVICES is set per-task by SLURM).
# We then expose LOCAL_RANK=0 to mmengine for every rank, because
# mmengine's set_device(local_rank) would fail with local_rank>=1 if
# only one GPU is visible. RANK and WORLD_SIZE come from SLURM_PROCID /
# SLURM_NTASKS so the cross-process collective uses the right global
# rank. NCCL talks over the loopback / shared-memory transport.

PROJECT_DIR=/users/sfedorchenko/Event-SAMA-train
CONFIG=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs/sama_4b_event_v2.py
WEIGHTS_DIR=/iopsstor/scratch/cscs/sfedorchenko/weights
DATA_DIR=/iopsstor/scratch/cscs/sfedorchenko/sama_dataset
EVENT_DIR_CAPSTOR=/capstor/scratch/cscs/mzehnde/sama_converted
EVENT_DIR_IOPSSTOR=/iopsstor/scratch/cscs/mzehnde/sama_converted
RUNS_DIR=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs
WORK_DIR=${RUNS_DIR}/sama_4b_event_v2
SQSH=${HOME}/nvidia+pytorch+24.01-py3.sqsh
PYDEPS=/iopsstor/scratch/cscs/sfedorchenko/event_sama_pydeps

mkdir -p ${WORK_DIR} ${RUNS_DIR}

echo "=== SAMA-4B Event V2 Training (mevis + lvvis + chatunivi + refcoco) ==="
echo "Job: ${SLURM_JOB_ID}  Node: $(hostname)  Date: $(date)"
echo "Config:                   ${CONFIG}"
echo "Work dir:                 ${WORK_DIR}"
echo "Event root (capstor):     ${EVENT_DIR_CAPSTOR}"
echo "Event root (iopsstor):    ${EVENT_DIR_IOPSSTOR}"
echo "ntasks=${SLURM_NTASKS_PER_NODE}, gpus-per-task=1, --launcher pytorch"

# DDP rendezvous: all 4 ranks need the same MASTER_ADDR/PORT.
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
echo "MASTER_ADDR=${MASTER_ADDR}  MASTER_PORT=${MASTER_PORT}"

# Allocate a host tmpfs for /dev/shm before the container starts; the
# container's default /dev/shm is tiny (~64MB) which overflows when 16
# DataLoader workers (4 per rank x 4 ranks) all share tensors via shm.
SHM_DIR=/tmp/sama_shm_${SLURM_JOB_ID}
mkdir -p ${SHM_DIR}

# srun launches 4 tasks; SLURM cgroup binds each to exactly 1 GPU.
srun \
  --container-image=${SQSH} \
  --container-mounts=${PROJECT_DIR}:${PROJECT_DIR},${WEIGHTS_DIR}:${WEIGHTS_DIR},${DATA_DIR}:${DATA_DIR},${EVENT_DIR_CAPSTOR}:${EVENT_DIR_CAPSTOR},${EVENT_DIR_IOPSSTOR}:${EVENT_DIR_IOPSSTOR},${RUNS_DIR}:${RUNS_DIR},${PYDEPS}:${PYDEPS},/tmp:/tmp,${SHM_DIR}:/dev/shm \
  --no-container-mount-home \
  bash -c '
    set -e
    export PYTHONPATH='"${PYDEPS}"':'"${PROJECT_DIR}"':$PYTHONPATH
    export TMPDIR=/tmp
    mkdir -p $TMPDIR

    export DISPLAY=""
    export MPLBACKEND=Agg
    export TRANSFORMERS_NO_FLASH_ATTN_2=1
    export TRANSFORMERS_CACHE=/tmp/transformers_cache
    mkdir -p $TRANSFORMERS_CACHE
    export PYDEPS='"${PYDEPS}"'

    # Per-task DDP env. SLURM cgroup makes only 1 GPU visible, so its
    # in-process index is 0 -> LOCAL_RANK=0 for everyone.
    export RANK=${SLURM_PROCID}
    export LOCAL_RANK=0
    export WORLD_SIZE=${SLURM_NTASKS}
    echo "[task ${SLURM_PROCID}/${SLURM_NTASKS}] host=$(hostname)  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}  RANK=${RANK} LOCAL_RANK=${LOCAL_RANK} WORLD_SIZE=${WORLD_SIZE}"

    # h5py probe (only rank 0 installs to avoid 4-way race).
    if [ "${SLURM_PROCID}" = "0" ]; then
      if ! python -c "import h5py" >/dev/null 2>&1; then
        echo "[setup] installing h5py into PYDEPS..."
        pip install --no-deps --target='"${PYDEPS}"' "h5py>=3.10,<4"
      fi
    fi
    sleep 3

    cd '"${PROJECT_DIR}"'

    DS_CONFIG=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs/deepspeed_zero2_event.json
    PINNED_TRAIN=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/configs/train_no_fa2_pinned.py
    # Auto-resume from the latest checkpoint if one exists in WORK_DIR.
    # mmengine writes a `last_checkpoint` text file in the work dir every
    # time CheckpointHook fires (every save_steps=1000 iters); it points
    # to the newest iter_*.pth directory. xtuners train.py supports
    # --resume <path>; passing it loads optimiser, scheduler, RNG, and
    # current iter, so the chained job picks up exactly where the prior
    # one was killed by SLURM wall time. If no checkpoint exists yet
    # (fresh run), we omit --resume and start clean.
    LAST_CKPT_FILE='"${WORK_DIR}"'/last_checkpoint
    RESUME_ARGS=""
    if [ -f "$LAST_CKPT_FILE" ]; then
      LAST_CKPT=$(cat "$LAST_CKPT_FILE")
      if [ -d "$LAST_CKPT" ] || [ -f "$LAST_CKPT" ]; then
        RESUME_ARGS="--resume $LAST_CKPT"
        echo "[sbatch] resuming from $LAST_CKPT"
      else
        echo "[sbatch] last_checkpoint pointed at missing $LAST_CKPT; starting fresh"
      fi
    else
      echo "[sbatch] no prior checkpoint in '"${WORK_DIR}"'; starting fresh"
    fi
    # Defaults from config:
    #   batch_size=4, accumulative_counts=4 -> eff batch 64 (matches paper)
    #   num_workers=4 + persistent + prefetch=4 + pin_memory
    python $PINNED_TRAIN '"${CONFIG}"' \
      --launcher pytorch \
      --deepspeed $DS_CONFIG \
      --work-dir='"${WORK_DIR}"' \
      $RESUME_ARGS \
      2>&1
  '
