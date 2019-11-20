#!/usr/bin/env bash
#SBATCH -J argosRun
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p short
#SBATCH --mem 32G
#SBATCH -C E5-2680

# Stop execution after any error
set -e

# Useful variables
JOB_LOC=run${3}/static
BASE_LOC=$PWD
DATADIR=$BASE_LOC/../../data #where you want your data to be stored
COUNT=0
