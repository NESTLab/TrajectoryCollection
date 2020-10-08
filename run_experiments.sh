#!/usr/bin/env bash
#SBATCH -J behaviortrain
#SBATCH -n 10
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH -p short
#SBATCH --mem 75G
#SBATCH --mail-type=END
#SBATCH --mail-user=nsrishankar@wpi.edu

# Stop execution after any error
set -e

# From init_experiments.sh
BEHAVIOR=$1
QUORUM=$2
QUOTA=$3

# Useful variables
BASEDIR=$PWD
MYUSER=$(whoami)
LOCALDIR=/local
DATADIR=$BASEDIR/data/${BEHAVIOR}/*.dat
RESULTSDIR=$BASEDIR/new_outputs/

EXPERIMENT=${BEHAVIOR}_quorum${QUORUM}_quota${QUOTA}
mkdir -p $RESULTSDIR/$EXPERIMENT/

WORKDIR=$LOCALDIR/$MYUSER/$EXPERIMENT/

source ./tf/bin/activate
rm -rf $WORKDIR && mkdir -p $WORKDIR && cd $WORKDIR

cp -a $DATADIR $WORKDIR
cp -a $BASEDIR/*.py $WORKDIR

echo "Working on $BEHAVIOR with Quorum $QUORUM and QUOTA $QUOTA"
python dfl_in_mrs_opt.py ${BEHAVIOR} ${QUORUM} ${QUOTA}

echo "Completed"

cp -a *.pkl $RESULTSDIR/$EXPERIMENT/
echo "Copied history and summary pickles"
cp -r model_* $RESULTSDIR/$EXPERIMENT/
echo "Copied model"
#cd ../ && tar -cvzf $WORKDIR ${EXPERIMENT}.tar.gz
#cp -a ${EXPERIMENT}.tar.gz $RESULTSDIR
