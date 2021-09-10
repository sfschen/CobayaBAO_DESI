#!/bin/bash
#SBATCH -J Cobaya13
#SBATCH -N 1
#SBATCH -t 12:00:00
#SBATCH -o Cobaya13.out
#SBATCH -e Cobaya13.err
#SBATCH -q regular
#SBATCH -C haswell
#SBATCH -A desi
#
source activate cobaya
#
export OMP_NUM_THREADS=8
export NUMEXPR_MAX_THREADS=8
export PYTHONPATH=${PYTHONPATH}:/global/cscratch1/sd/sfschen/desi_bao_fitting/lss_likelihood
#export PYTHONPATH=${PYTHONPATH}:/global/homes/s/sfschen/Python/ZeldovichReconPk

srun -n 8 -c 8 cobaya-run chains/boss_joint_taylor