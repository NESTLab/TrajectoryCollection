#!/usr/bin/env bash
#SBATCH -J FA0660
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p short
#SBATCH --mem 64G
#SBATCH -C E5-2680

# Stop execution after any error
set -e

# Useful variables
BASE_LOC=$PWD
DATADIR=$BASE_LOC/../tensorflow-scripts/results #where you want your data to be stored
WORKDIR=$BASE_LOC/../tensorflow-scripts

cd $WORKDIR

for QUORUM in 0.6 # 0.6
do
	for QUOTA in 60  #  20 60
	do
		python FL_in_MRS.py '../data/avoidance**.dat' ${QUORUM} ${QUOTA}
	done
done
