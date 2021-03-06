# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import datasets
import pandas as pd
import json
import os
import random
from argparse import ArgumentParser
from collections import defaultdict
from transformers import AutoTokenizer
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

parser = ArgumentParser()
parser.add_argument('--tokenizer_name', required=True)
parser.add_argument('--rank_file', required=True)
parser.add_argument('--truncate', type=int, default=512)

parser.add_argument('--sample_from_top', type=int, required=True)
parser.add_argument('--n_sample', type=int, default=100)
parser.add_argument('--random', action='store_true')
parser.add_argument('--json_dir', required=True)

parser.add_argument('--qrel', required=True)
parser.add_argument('--query_collection', required=True)
parser.add_argument('--doc_collection', required=True)
args = parser.parse_args()
cache_dir = "/content/drive/MyDrive/_RERANKER_/data_train"

def main():
    print("========== FIXED SUMMARIZATION ==========")
    qrel = read_qrel()
    rankings = defaultdict(list)
    no_judge = set()

    # arquivos HDCT
    with open(args.rank_file) as f:
        for l in f:
            qid, pid, rank = l.split()
            if qid not in qrel:
                no_judge.add(qid)
                continue
            if pid in qrel[qid]:
                continue
            # append passage if & only if it is not juddged relevant but ranks high (dentro dos 1000)
            rankings[qid].append(pid)

    print(f'{len(no_judge)} queries not judged and skipped', flush=True)

    columns = ['did', 'url', 'title', 'body']
    collection = args.doc_collection
    collection = datasets.load_dataset(
        'csv',
        data_files=collection,
        column_names=['did', 'url', 'title', 'body'],
        delimiter='\t',
        ignore_verifications=True,
        cache_dir=cache_dir,
    )['train']
    qry_collection = args.query_collection
    qry_collection = datasets.load_dataset(
        'csv',
        data_files=qry_collection,
        column_names=['qid', 'qry'],
        delimiter='\t',
        ignore_verifications=True,
        cache_dir=cache_dir,
    )['train']

    doc_map = {x['did']: idx for idx, x in enumerate(collection)}
    qry_map = {str(x['qid']): idx for idx, x in enumerate(qry_collection)}

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)

    out_file = args.rank_file
    if out_file.endswith('.tsv') or out_file.endswith('.txt'):
        out_file = out_file[:-4]
    out_file = os.path.join(args.json_dir, os.path.split(out_file)[1])
    out_file = out_file + '.group.json'

    queries = list(rankings.keys())
    with open(out_file, 'w') as f:
        for qid in tqdm(queries):
            # pick from top of the full initial ranking
            negs = rankings[qid][:args.sample_from_top]
            # shuffle if random flag is on
            if args.random:
                random.shuffle(negs)
            # pick n samples
            negs = negs[:args.n_sample]

            neg_encoded = []

            # ========= Sampling e encoding dos negatives =========
            for neg in negs:
                idx = doc_map[neg]
                item = collection[idx]
                document_id, url, title, body = (item[k] for k in columns)

                # Change None to empty string
                url, title, body = map(lambda v: v if v else '', [url, title, body])


                # =========== Interven??ao no c??digo: ===========
                # =========== Se as coisas j?? tudo tokenizada passarem de 512, faz a sumariza????o ===========

                tokenized_url = tokenizer.tokenize(url)
                tokenized_title = tokenizer.tokenize(title)
                tokenized_body = tokenizer.tokenize(body)

                total_tokens = len(tokenized_url) + len(tokenized_title) + len(tokenized_body) + 2 # 2 for the sep tokens

                if total_tokens > 512:
                    space_left = 512 - 2 - len(tokenized_url) + len(tokenized_title)
                    body = tf_idf(space_left, body)

                # the tokenization already done cannot be reused
                to_encode = url + tokenizer.sep_token + title + tokenizer.sep_token + body
                encoded_neg = tokenizer.encode(
                    to_encode,
                    add_special_tokens=False,
                    max_length=args.truncate,
                    truncation=True
                )
                # =============== FIM DA INTERVEN????O ===============

                neg_encoded.append({
                    'passage': encoded_neg,
                    'pid': neg,
                })

            # ========= Sampling e encoding dos positives =========
            pos_encoded = []
            for pos in qrel[qid]:
                idx = doc_map[pos]
                item = collection[idx]
                document_id, url, title, body = (item[k] for k in columns)
                url, title, body = map(lambda v: v if v else '', [url, title, body])
                encoded_pos = tokenizer.encode(
                    url + tokenizer.sep_token + title + tokenizer.sep_token + body,
                    add_special_tokens=False,
                    max_length=args.truncate,
                    truncation=True
                )
                pos_encoded.append({
                    'passage': encoded_pos,
                    'pid': pos,
                })
            q_idx = qry_map[qid]
            query_dict = {
                'qid': qid,
                'query': tokenizer.encode(
                    qry_collection[q_idx]['qry'],
                    add_special_tokens=False,
                    max_length=args.truncate,
                    truncation=True),
            }
            item_set = {
                'qry': query_dict,
                'pos': pos_encoded,
                'neg': neg_encoded,
            }
            f.write(json.dumps(item_set) + '\n')


def read_qrel():
    import gzip, csv
    qrel = {}
    with gzip.open(args.qrel, 'rt', encoding='utf8') as f:
        tsvreader = csv.reader(f, delimiter=" ")
        for [topicid, _, docid, rel] in tsvreader:
            assert rel == "1"
            if topicid in qrel:
                qrel[topicid].append(docid)
            else:
                qrel[topicid] = [docid]
    return qrel


def tf_idf(space_left, input):
    # remove stop words
    stop_words = set(stopwords.words('english'))
    text_tokens = word_tokenize(input)
    tokens_without_sw = [word for word in text_tokens if not word in stop_words]
    input_no_sw = " ".join(tokens_without_sw)

    # tf-idf
    vectorizer = TfidfVectorizer(strip_accents='ascii', analyzer='word')
    matrix = vectorizer.fit_transform([input_no_sw])
    dense = matrix.todense().tolist()
    features = vectorizer.get_feature_names()
    df = pd.DataFrame(dense, columns=features)
    scores = df.iloc[0]
    ranking = list(zip(scores, features))
    ranking.sort(reverse=True)

    # cropar no tamanho do space left
    space_left = int(space_left - space_left * 0.25)

    # Make string out of ranking
    new_body = ""
    for ranking, word in ranking[:space_left]:
        new_body += word + " "

    return new_body


if __name__ == "__main__":
    main()