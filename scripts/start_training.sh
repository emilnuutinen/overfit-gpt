#!/bin/bash
#SBATCH --job-name=overfit_gpt
#SBATCH --account=Project_2002820
#SBATCH --time=00:15:00
##SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --mem=32G
#SBATCH --partition=gputest
#SBATCH --gres=gpu:a100:4
##SBATCH --cpus-per-task=32

module load pytorch/1.13

pip3 install -r requirements.txt

srun python3 train.py \
  --overwrite_cache \
  --model_type gpt2 \
  --tokenizer_name model/ \
  --cache_dir cache/ \
  --train_file big_file/all_files.txt \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --do_train \
  --do_eval \
  --output_dir tmp/
