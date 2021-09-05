#!/usr/bin/env bash

for i in $(seq -f "%03g" 0 0)
do
python /content/drive/MyDrive/__TCC__Reranker/Reranker/run_tf_idf/scripts/preprocess_to_train_TFIDF.py \
    --tokenizer_name bert-base-uncased \
    --rank_file 	/content/drive/MyDrive/__TCC__Reranker/Reranker/common_data/01-hdct-marco-train//${i}.txt \
    --json_dir /content/drive/MyDrive/__TCC__Reranker/Reranker/run_tf_idf/data/training-files \
    --n_sample 10 \
    --sample_from_top 100 \
    --random \
    --truncate 512 \
    --qrel /content/drive/MyDrive/__TCC__Reranker/Reranker/common_data/02-msmarco-files/msmarco-doctrain-qrels.tsv.gz \
    --query_collection /content/drive/MyDrive/__TCC__Reranker/Reranker/common_data/02-msmarco-files/msmarco-doctrain-queries.tsv \
    --doc_collection /content/drive/MyDrive/__TCC__Reranker/Reranker/common_data/02-msmarco-files/msmarco-docs.tsv
done