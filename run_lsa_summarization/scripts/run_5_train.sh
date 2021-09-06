#!/usr/bin/env bash

for seed in 974 061 280 666 999 1028
do
python -m torch.distributed.launch \
  --nproc_per_node 1 /content/drive/MyDrive/__TCC__Reranker/Reranker/common_scripts/run_marco_original.py \
  --output_dir /content/drive/MyDrive/__TCC__Reranker/Reranker/run_lsa_summarization/data/checkpoints_${seed} \
  --model_name_or_path  bert-base-uncased \
  --do_train \
  --save_steps 2000 \
  --train_dir /content/drive/MyDrive/__TCC__Reranker/Reranker/run_lsa_summarization/data/training-files/ \
  --max_len 512 \
  --fp16 \
  --per_device_train_batch_size 1 \
  --train_group_size 8 \
  --gradient_accumulation_steps 1 \
  --per_device_eval_batch_size 64 \
  --warmup_ratio 0.1 \
  --weight_decay 0.01 \
  --learning_rate 1e-5 \
  --num_train_epochs 2 \
  --overwrite_output_dir \
  --dataloader_num_workers 8 \
  --seed ${seed}
done
