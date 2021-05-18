#!/bin/bash

#####################
#
# The job is to be sent as qsub gpu1_job.sh
##############3

#PBS -N myjobbname
#PBS -q gpu

#PBS -l nodes=gpu1next:ppn=12
#PBS -l mem=20G

user=`whoami`
KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"
KMP_BLOCKTIME=1

source /home/$user/.bashrc
# source /home/$user/miniconda/etc/profile.d/conda.sh
# conda activate IC-3.7-2020-06-16
# might need to setup more environmental variables
cd /home/$user/NEXT_SPARSECONVNET/
source setup.sh
cd -
export $KMP_SETTING
export KMP_BLOCKTIME=$KMP_BLOCKTIME
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=1
export NUMEXPR_MAX_THREADS=1
export OMP_NUM_THREADS=4


main.py -a train -conf /home/$user/NEXT_SPARSECONVNET/config/segnet.conf
