#!/bin/bash
# Sequential chain (QOS allows few queued jobs): convert -> inference (4 windows) -> baselines.
RE=/iopsstor/scratch/cscs/sfedorchenko/real_events
DC=/iopsstor/scratch/cscs/sfedorchenko/event_sama_runs/eval/datacheck
waitjob () { while squeue -h -j $1 2>/dev/null | grep -q .; do sleep 30; done; sacct -j $1 -X -n -o State | head -1; }
rm -rf $RE/mousesis_train_converted
c=$(sbatch --parsable $RE/convert_train.sbatch); echo "convert=$c $(waitjob $c)"
i=$(sbatch --parsable $RE/mousesis_train_inf.sbatch); echo "inference=$i"
b=$(sbatch --parsable $DC/mousesis_train_base.sbatch); echo "baselines=$b"
echo "inference_state=$(waitjob $i)"; echo "baselines_state=$(waitjob $b)"
echo ALLDONE
