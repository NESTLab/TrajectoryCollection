#!/usr/bin/env bash

# Stop execution after any error
set -e

declare -a exparray=("avoidance_K15" "flocking_K15" "mixed_K15" "foraging_K15")
# declare -a exparray=("flocking_K15")
# declare -a exparray=("avoidance_K60" "flocking_K60" "mixed_K60" "foraging_K60")

for val in "${exparray[@]}";
do
    echo $val
    # sbatch run_experiments_C.sh ${val} # For centralized
    for QUORUM in 0.2 0.6
    do
        for QUOTA in 20 60 #20 60
        do
            sbatch run_experiments.sh ${val} ${QUORUM} ${QUOTA}
        done
    done
done
