#!/usr/bin/env bash

for seed in 974 061 280 666 1028
do
echo "\n================= RUN EVALUATION ON SEED ${seed} ================="
python /content/drive/MyDrive/__TCC__Reranker/Reranker/common_scripts/4_score_to_marco.py \
  --score_file /content/drive/MyDrive/__TCC__Reranker/Reranker/run_lsa_summarization/data/eval/score_${seed}.txt

python /content/drive/MyDrive/__TCC__Reranker/Reranker/common_scripts/6_msmarco_eval_2.py \
  /content/drive/MyDrive/__TCC__Reranker/Reranker/run_lsa_summarization/data/eval/score_${seed}.txt.marco \
  /content/drive/MyDrive/__TCC__Reranker/Reranker/common_data/dev_ground_truth.tsv \
  /content/drive/MyDrive/__TCC__Reranker/Reranker/common_data/eval/exclude

done