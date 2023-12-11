#!/bin/bash
#SBATCH --job-name=mini
#SBATCH --account=Project_2002820
#SBATCH --time=36:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:a100:4

module load pytorch/1.13

pip3 install -r requirements.txt

srun python3 train.py \
  --overwrite_cache \
  --model_type gpt2 \
  --tokenizer_name TurkuNLP/gpt3-finnish-small \
  --config_overrides="vocab_size=131072" \
  --cache_dir cache/ \
  --dataset_name graelo/wikipedia \
  --dataset_config_name 20230601.fi \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 8 \
  --learning_rate 4e-5 \
  --optim adamw_torch \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 20000 \
  --output_dir checkpoints \
  --save_steps 5000 \
  --num_train_epochs 200 \
  --save_total_limit 5 \
  --overwrite_output_dir
