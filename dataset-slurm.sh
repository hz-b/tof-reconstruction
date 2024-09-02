#!/bin/bash
#SBATCH --job-name=dataset
#SBATCH --gres=gpu:0
#SBATCH --nodes=1
#SBATCH --array=1-6
#SBATCH --cpus-per-task 100
#SBATCH --mem-per-cpu=1GB
#SBATCH --partition=main
#SBATCH --spread-job

        /mnt/work/xfel/bessy/metrix_hackathon/local-venv.sh python3 data_generation.py $SLURM_ARRAY_TASK_ID