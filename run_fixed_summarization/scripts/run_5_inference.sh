#!/usr/bin/env bash

for seed in 974 061 280 666 999
do
echo "\n================= Inference on ${seed} ================="
python /content/drive/MyDrive/__TCC__Reranker/Reranker/common_scripts/2_run_marco.py \
  --model_name_or_path /content/drive/MyDrive/__TCC__Reranker/Reranker/run_fixed_summarization/data/checkpoints_${seed} \
  --output_dir /content/drive/MyDrive/__TCC__Reranker/Reranker/run_fixed_summarization/data/eval \
  --tokenizer_name bert-base-uncased \
  --do_predict \
  --max_len 512 \
  --fp16 \
  --per_device_eval_batch_size 64 \
  --dataloader_num_workers 8 \
  --pred_path /content/drive/MyDrive/__TCC__Reranker/Reranker/run_fixed_summarization/data/inference-files/mini_all.json \
  --pred_id_file  /content/drive/MyDrive/__TCC__Reranker/Reranker/run_fixed_summarization/data/inference-files/mini_ids.tsv \
  --rank_score_path /content/drive/MyDrive/__TCC__Reranker/Reranker/run_fixed_summarization/data/eval/score_${seed}.txt
done