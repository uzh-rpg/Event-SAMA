#!/bin/bash
# Wait for the MouseSIS train download, then submit convert -> inference -> baselines (afterok chain).
RE=/iopsstor/scratch/cscs/sfedorchenko/real_events
DC=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck
until grep -q DONE $RE/dl_train.log; do sleep 30; done
c=$(sbatch --parsable $RE/convert_train.sbatch)
i=$(sbatch --parsable --dependency=afterok:$c $RE/mousesis_train_inf.sbatch)
b=$(sbatch --parsable --dependency=afterok:$c $DC/mousesis_train_base.sbatch)
echo "convert=$c inference=$i baselines=$b"
