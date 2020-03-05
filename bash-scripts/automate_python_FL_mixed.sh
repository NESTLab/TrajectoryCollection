#!/usr/bin/env bash
#SBATCH -J FM0660
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

for QUORUM in 0.2 # 0.6
do
	for QUOTA in 20 # 20 60
	do
		python FL_in_MRS.py '../data/mixed**.dat' ${QUORUM} ${QUOTA}
	done
done
