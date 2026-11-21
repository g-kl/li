#!/bin/bash
#SBATCH --partition=bates
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
# usage: sbatch -J main_v1 scripts/train.sbatch run_name=main_v1 [overrides]
cd /mnt/home/gokulg/li
/mnt/home/gokulg/miniconda3/envs/lt/bin/python -m li.train "$@"
