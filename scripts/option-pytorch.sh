#!/bin/bash
#SBATCH --mem=4196  # Requested Memory
#SBATCH -p gpu  # Partition
#SBATCH -G 1  # Number of GPUs
#SBATCH -t 00:30:00  # Job time limit
#SBATCH -o out/option-critic-8options-%j.out  # %j = job ID
#SBATCH -e out/option-critic-8options-%j.err 

python3 -m examples --lr=0.0001 --num_options=8 --toybox=True
