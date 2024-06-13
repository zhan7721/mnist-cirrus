#!/bin/bash
#SBATCH --job-name=train_mnist
#SBATCH --output=train_mnist.out
#SBATCH --error=train_mnist.err
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --qos=short
#SBATCH --account=tc062-staff


source setup_slurm_env.sh

srun accelerate launch scripts/train.py