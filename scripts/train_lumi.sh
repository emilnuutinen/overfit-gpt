#!/bin/bash
#SBATCH --job-name=large
#SBATCH --account=Project_462000347
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=8
#SBATCH --partition=small-g
#SBATCH --mem=480G
#SBATCH --exclusive

module purge
module use /appl/local/csc/modulefiles
module load pytorch

pip3 install -r requirements.txt

srun python3 run_mini_clm.py \
  --overwrite_cache \
  --model_type gpt2 \
  --tokenizer_name TurkuNLP/gpt3-finnish-small \
  --config_overrides="n_embd=1280,n_head=20,n_layer=36,bos_token_id=1,eos_token_id=2,vocab_size=131072,attn_pdrop=0.0,embd_pdrop=0.0,resid_pdrop=0.0,summary_first_dropout=0.0" \
  --cache_dir cache/ \
  --dataset_name graelo/wikipedia \
  --dataset_config_name 20230601.fi \
  --preprocessing_num_workers 32 \
  --per_device_train_batch_size 3 \
  --per_device_eval_batch_size 3 \
  --gradient_accumulation_steps 8 \
  --learning_rate 4e-5 \
  --optim adamw_torch \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 10000 \
  --output_dir tmp_large \
  --save_steps 5000 \
  --num_train_epochs 200 \
  --save_total_limit 2 \
