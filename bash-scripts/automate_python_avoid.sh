#!/usr/bin/env bash
# SBATCH -J argosRun
# SBATCH -n 1
# SBATCH -N 1
# SBATCH -p short
# SBATCH --mem 96G
# SBATCH -C E5-2680

# Stop execution after any error
set -e

# Useful variables
BASE_LOC=$PWD
DATADIR=$BASE_LOC/../tensorflow-scripts/results #where you want your data to be stored
WORKDIR=$BASE_LOC/../tensorflow-scripts

cd $WORKDIR

for QUORUM in 0.2 0.6
do
	for QUOTA in 20 60
	do
		python DFL_in_MRS.py '../data/avoid**.dat' ${QUORUM} ${QUOTA}
	done
done