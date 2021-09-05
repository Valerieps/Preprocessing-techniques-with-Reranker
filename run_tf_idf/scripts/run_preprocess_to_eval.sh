#!/usr/bin/env bash

for i in $(seq -f "%03g" 0 0)
do
python /content/drive/MyDrive/__TCC__Reranker/Reranker/run_tf_idf/scripts/preprocess_to_eval_TFIDF.py \
  --file /content/drive/MyDrive/__TCC__Reranker/Reranker/common_data/04-inference_files/mini_dev.d100.tsv \
  --save_to /content/drive/MyDrive/__TCC__Reranker/Reranker/run_tf_idf/data/eval/mini_all.json \
  --generate_id_to /content/drive/MyDrive/__TCC__Reranker/Reranker/run_tf_idf/data/eval/mini_ids.tsv \
  --tokenizer bert-base-uncased \
  --truncate 512 \
  --q_truncate -1
done