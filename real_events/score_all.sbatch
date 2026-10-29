#!/bin/bash
#SBATCH --job-name=mouse-score-all
#SBATCH --partition=debug
#SBATCH --account=uzh55
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=450G
#SBATCH --time=00:30:00
#SBATCH --output=/iopsstor/scratch/cscs/sfedorchenko/real_events/score_all_%j.out
# Score every MouseSIS run (val + train) with eval_mousesis.py, one process per run, in parallel.
RE=/iopsstor/scratch/cscs/sfedorchenko/real_events
S=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs
export PYTHONPATH=$RE/pylibs:/users/sfedorchenko/Event-SAMA-train/tools/eval_video_seg
PY=/users/sfedorchenko/miniforge3/bin/python
cd $RE
run () { [ -f "$1/mousesis_eval.json" ] && return; $PY eval_mousesis.py "$1" "$2" "$3" > "$1/eval_log.txt" 2>&1 & }
for W in 10 75 150 300; do run $(ls -d runs/mousesis_train_${W}ms_5124797) mousesis_train_converted $W; done
for d in runs/mousesis_150ms_* runs/mousesis_300ms_*; do W=$(echo $d | sed 's/.*_\([0-9]*\)ms_.*/\1/'); run $d mousesis_converted $W; done
run $S/pseudorgb/run_mousesis_train75 mousesis_train_converted 75
run $S/e2vid/run_mousesis_train mousesis_train_converted 75
wait
$PY aggregate_mousesis.py agg_config.json | tee mousesis_all25_summary.md
ls -d runs/mousesis_* $S/pseudorgb/run_mousesis* $S/e2vid/run_mousesis* | while read d; do echo "$d $( [ -f $d/mousesis_eval.json ] && echo ok || echo MISSING)"; done
