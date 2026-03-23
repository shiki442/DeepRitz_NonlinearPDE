#!/bin/bash
#SBATCH -A yangzhijian
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH -J eq5
#SBATCH -o job-%j.out
#SBATCH -e job-%j.err
#SBATCH -p gpu

RUN_PATH="."
cd "$RUN_PATH" || exit 1

echo Directory is $PWD
echo This job runs on the following nodes: $SLURM_JOB_NODELIST
echo This job has allocated $SLURM_JOB_CPUS_PER_NODE cpu cores.

echo Problem: eq5
echo Time is `date`
echo ---------------------------------------------

source /project/songpengcheng/miniconda3/etc/profile.d/conda.sh
conda activate torch
/project/songpengcheng/miniconda3/envs/torch/bin/python -u main_pl.py

echo ---------------------------------------------
echo End at `date`