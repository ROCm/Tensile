#!/bin/bash

########################################
# client parameters
M=5504
N=5504
K=5504
NUM_PRINTOUTS=256
NUM_ENQUEUES_PER_PRINTOUT=64
DEVICES=( 0 1 2 3 )
EXEC=./sgemm_gfx900

########################################
# crtl-c will kill background process and quit
PIDS=()
function kill_procs() {
  kill ${PIDS[@]} 2> /dev/null
  echo "Aborted."
}
trap kill_procs INT

########################################
# launch clients
echo "" > out.cmd.txt
for DEVICE in ${DEVICES[@]}
do
CMD="${EXEC} --device-idx $DEVICE --sizes $M $N $K --num-enqueues-per-sync $NUM_ENQUEUES_PER_PRINTOUT --num-syncs-per-benchmark 1 --num-benchmarks $NUM_PRINTOUTS --use-gpu-timer 1 --function-idx 0 --init-c 3 --init-ab 3 --num-elements-to-validate 0 --print-max 0 | tee out.${DEVICE}.txt &"
echo $CMD >> out.cmd.txt
eval $CMD
PID=$!
PIDS+=($PID)
done

########################################
# wait for background processes
wait ${PIDS[@]}
echo "Done. Data logged to out.*.txt"
