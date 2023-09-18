#!/bin/bash
#SBATCH --job-name=overfit_gpt
#SBATCH --account=Project_2002820
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:a100:4

module load pytorch/1.13

pip3 install -r requirements.txt

srun python3 train.py \
  --overwrite_cache \
  --model_type gpt2 \
  --tokenizer_name Finnish-NLP/gpt2-finnish \
  --cache_dir cache/ \
  --dataset_name graelo/wikipedia \
  --dataset_config_name 20230601.fi \
  --preprocessing_num_workers 4 \
  --per_device_train_batch_size 8 \
  --per_device_eval_batch_size 8 \
  --gradient_accumulation_steps 4 \
  --optim adamw_torch \
  --learning_rate 4e-5 \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 20000 \
  --output_dir tmp/ \
  --save_steps 10000 \
  --num_train_epochs 200 \
  --save_total_limit 5 \
  --resume_from_checkpoint tmp/checkpoint-120000
