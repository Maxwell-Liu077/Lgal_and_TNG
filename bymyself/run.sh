#!/bin/bash

#SBATCH --job-name=tng_run
#SBATCH --partition=WorkB
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=7-00:00:00
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.err

source /public/home/zju_visitor/Anaconda3/etc/profile.d/conda.sh
conda activate base

export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK

cd /public/home/zju_visitor/LiuYuanhao/bymself

mkdir -p logs

echo "========================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $(hostname)"
echo "CPUs:          $SLURM_CPUS_PER_TASK"
echo "Numba threads: $NUMBA_NUM_THREADS"
echo "Working dir:   $(pwd)"
echo "Start time:    $(date)"
echo "========================================"

python main.py

echo "========================================"
echo "End time: $(date)"
echo "========================================"
