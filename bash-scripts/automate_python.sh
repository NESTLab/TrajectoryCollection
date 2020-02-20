#!/usr/bin/env bash

# Stop execution after any error
set -e

# Useful variables
BASE_LOC=$PWD
DATADIR=$BASE_LOC/../tensorflow-scripts/results #where you want your data to be stored
WORKDIR=$BASE_LOC/../tensorflow-scripts

cd $WORKDIR

python DFL_in_MRS.py '../data/avoidance_20200131_204454.dat' '../data/G_avoidance_20200131_204454.dat'