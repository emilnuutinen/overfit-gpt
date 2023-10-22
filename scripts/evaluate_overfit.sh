#!/bin/bash
#SBATCH --job-name=evaluate
#SBATCH --account=Project_2002820
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpusmall
#SBATCH --gres=gpu:a100:1

module load pytorch/1.13

pip3 install -r requirements.txt

srun python3 testing/evaluate.py \
  --model "models/gpt2_mini_200_epochs" \
  --tokenizer "models/gpt2_mini_200_epochs"

