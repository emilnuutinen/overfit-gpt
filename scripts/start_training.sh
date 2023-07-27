python3 train.py \
  --overwrite_cache \
  --model_type gpt2 \
  --tokenizer_name model/ \
  --cache_dir cache/ \
  --train_file big_file/all_files.txt \
  --do_train \
  --do_eval \
  --output_dir tmp/
