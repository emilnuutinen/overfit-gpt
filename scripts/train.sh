#!/bin/bash
#SBATCH --job-name=overfit_gpt
#SBATCH --account=Project_2002820
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:a100:4

module load pytorch/1.13

pip3 install -r requirements.txt

srun python3 train.py \
  --overwrite_cache \
  --model_type gpt2 \
  --tokenizer_name model/ \
  --cache_dir cache/ \
  --train_file data/train.txt \
  --validation_file data/dev.txt \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --do_train \
  --do_eval \
  --output_dir tmp/ \
  --resume_from_checkpoint \
  --save_steps 10000
