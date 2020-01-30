#!/usr/bin/env bash
# SBATCH -J argosRun
# SBATCH -n 1
# SBATCH -N 1
# SBATCH -p short
# SBATCH --mem 32G
# SBATCH -C E5-2680

# Stop execution after any error
set -e

# Useful variables
BASE_LOC=$PWD
DATADIR=$BASE_LOC/../data #where you want your data to be stored
COUNT=0

date +"%s"


WORKDIR=$DATADIR
CONFDIR=$WORKDIR/experiments/
mkdir -p $CONFDIR
cd $WORKDIR

# Go through all the values for 
for CONTROLLER in flocking foraging khepera_obstacleavoidance 
do
    for SEED in 2 7 10
    do 
      for NOISE in 0 0.1 0.5
      do
         # Go through all the values
         for NBROBOTS in 10 20 40
         do
            cd $WORKDIR
            TIMESTAMP=`date "+%Y%m%d_%H%M%S"`
            sed -e "s|SEED|${SEED}|;s|NOISE_WH|${NOISE}|;s|NOISE_RAB|${NOISE}|;s|TIMESTAMP|${TIMESTAMP}|;s|NBROBOTS|${NBROBOTS}|g" $BASE_LOC/experiments/${CONTROLLER}_batch.argos > $CONFDIR/exp_${CONTROLLER}_${NBROBOTS}_${SEED}_${NOISE}_${TIMESTAMP}.argos
            cd $BASE_LOC
            # # Execute program (this also writes files in work dir)
            argos3 -l /dev/null -c $CONFDIR/exp_${CONTROLLER}_${NBROBOTS}_${SEED}_${NOISE}_${TIMESTAMP}.argos 2> "log_${CONTROLLER}_${NBROBOTS}_${SEED}_${NOISE}_${TIMESTAMP}.txt"

            echo $(( COUNT++ ))
            echo "If you see this, experiment $COUNT/81 ${CONTROLLER}_${NBROBOTS}_${SEED}_${NOISE}_${TIMESTAMP} is done"
         done
      done
   done
done

mv *.dat $WORKDIR
echo "Copying files"
#rm -rf $CONFDIR