#!/bin/bash
#SBATCH --job-name=overfit_gpt
#SBATCH --account=Project_2002820
#SBATCH --time=0:15:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --partition=gputest
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
  --streaming True \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --learning_rate 4e-5 \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 50000 \
  --output_dir testing/ \
  --save_steps 10000 \
  --num_train_epochs 200 \
  --save_total_limit 5 \
