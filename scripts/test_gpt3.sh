#!/bin/bash
#SBATCH --job-name=overfit_gpt
#SBATCH --account=Project_2002820
#SBATCH --time=1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpumedium
#SBATCH --gres=gpu:a100:4

module load pytorch/1.13

pip3 install -r requirements.txt

srun python3 run_clm.py \
  --n_gpu 4 \
  --overwrite_cache \
  --model_type gpt2 \
  --tokenizer_name TurkuNLP/gpt3-finnish-small \
  --config_overrides="bos_token_id=1,eos_token_id=2,vocab_size=131072" \
  --cache_dir cache/ \
  --dataset_name graelo/wikipedia \
  --dataset_config_name 20230601.fi \
  --preprocessing_num_workers 4 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 4 \
  --block_size 512 \
  --learning_rate 4e-5 \
  --optim adamw_torch \
  --do_train \
  --output_dir tmp_gpt3/ \
  --save_steps 10000 \
  --num_train_epochs 200 \
  --save_total_limit 5 \
