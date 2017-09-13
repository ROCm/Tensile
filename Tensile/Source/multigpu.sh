#!/bin/bash

# kill.sh will kill all background processes that this script launches,
# in case they were set to run too long
echo
echo "   *********************"
echo "   *                   *"
echo "   *  kill.sh to quit  *"
echo "   *                   *"
echo "   *********************"
echo
echo "" > kill.sh
chmod a+x kill.sh

# devices

# client parameters
DEVICES=( 0 1 2 3 )
M=2048
N=2048
K=2048
NUM_ENQUEUES_PER_SYNC=16
NUM_SYNCS_PER_BENCHMARK=1
NUM_BENCHMARKS=64
EXEC=./client

# launch clients
for DEVICE in ${DEVICES[@]}
do
${EXEC} --device-idx $DEVICE --sizes $M $N $K --num-enqueues-per-sync $NUM_ENQUEUES_PER_SYNC --num-syncs-per-benchmark $NUM_SYNCS_PER_BENCHMARK --num-benchmarks $NUM_BENCHMARKS --use-gpu-timer 1 --function-idx 0 --init-c 3 --init-ab 3 --num-elements-to-validate 0 --print-max 0 | tee out.${DEVICE}.txt &
echo "kill $!" >> kill.sh
done

# Explanation of benchmarks, syncs, enqueues
# for num_benchmarks:
#   for num_syncs_per_benchmark:
#     for num_enqueues_per_benchmark:
#       enqueue_kernel(); // enqueue
#   wait_for_kernels(); // sync
#   print_stats_to_stdout(); // 1 print out per benchmark
