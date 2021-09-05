# Copyright 2021 Reranker Author. All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from transformers import AutoTokenizer
from argparse import ArgumentParser
from tqdm import tqdm
from multiprocessing import Pool
import json
import datasets
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


parser = ArgumentParser()

parser.add_argument('--file', required=True)
parser.add_argument('--save_to', required=True)
parser.add_argument('--tokenizer', required=True)
parser.add_argument('--generate_id_to')
parser.add_argument('--truncate', type=int, default=512)
parser.add_argument('--q_truncate', type=int, default=16)
args = parser.parse_args()

tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
SEP = tokenizer.sep_token


columns = [
        'qid', 'query', 'did', 'url', 'title', 'body', 'unused'
]


def encode_line(line):
    qid, qry, did, url, title, body = line.strip().split('\t')
    # inserir funcao de modificacao
    #=============================================

    tokenized_url = tokenizer.tokenize(url)
    tokenized_title = tokenizer.tokenize(title)
    tokenized_body = tokenizer.tokenize(body)

    total_tokens = len(tokenized_url) + len(tokenized_title) + len(tokenized_body) + 2  # 2 for the sep tokens

    if total_tokens > 512:
        space_left = 512 - 2 - len(tokenized_url) + len(tokenized_title)
        body = run_fixed_summarization(space_left, body)

    # ============================================
    qry_encoded = tokenizer.encode(
        qry,
        truncation=True if args.q_truncate else False,
        max_length=args.q_truncate,
        add_special_tokens=False,
        padding=False,
    )
    doc_encoded = tokenizer.encode(
            url + SEP + title + SEP + body,
            truncation=True,
            max_length=args.truncate,
            add_special_tokens=False,
            padding=False
        )
    entry = {
        'qid': qid,
        'pid': did,
        'qry': qry_encoded,
        'psg': doc_encoded,
    }
    entry = json.dumps(entry)
    return entry, qid, did


def encode_item(item):
    qid, qry, did, url, title, body, _ = (item[k] for k in columns)
    url, title, body = map(lambda v: v if v else '', [url, title, body])
    qry_encoded = tokenizer.encode(
        qry,
        truncation=True if args.q_truncate else False,
        max_length=args.q_truncate,
        add_special_tokens=False,
        padding=False,
    )
    doc_encoded = tokenizer.encode(
            url + SEP + title + SEP + body,
            truncation=True,
            max_length=args.truncate,
            add_special_tokens=False,
            padding=False
        )
    entry = {
        'qid': qid,
        'pid': did,
        'qry': qry_encoded,
        'psg': doc_encoded,
    }
    entry = json.dumps(entry)
    return entry, qid, did


data_set = datasets.load_dataset(
    'csv',
    data_files=args.file,
    column_names=columns,
    delimiter='\t',
    ignore_verifications=True
)['train']


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


with open(args.save_to, 'w') as jfile:
    # for l in text_file:
    #     json_item = method_name(args, l, tokenizer)
    all_ids = []
    if args.q_truncate < 0:
        print('queries are not truncated', flush=True)
        args.q_truncate = None
    with Pool() as p:
        all_json_items = p.imap(
            encode_item,
            tqdm(data_set),
            chunksize=100
        )
        for json_item, qry_id, doc_id in all_json_items:
            all_ids.append((qry_id, doc_id))
            jfile.write(json_item + '\n')

    if args.generate_id_to is not None:
        with open(args.generate_id_to, 'w') as id_file:
            for qry_id, doc_id in all_ids:
                id_file.write(f'{qry_id}\t{doc_id}\n')
