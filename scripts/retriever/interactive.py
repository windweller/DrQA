#!/usr/bin/env python3
# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
"""Interactive mode for the tfidf DrQA retriever module."""

import argparse
import code
import prettytable
import logging
from drqa import retriever

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default=None)
parser.add_argument('--data_json', type=str, default=None)
parser.add_argument('--filter', action='store_false',
                    help="we filter out selected documents that directly contain the answer")
args = parser.parse_args()

logger.info('Initializing ranker...')
ranker = retriever.get_class('tfidf')(tfidf_path=args.model)

import json
from tqdm import tqdm

id_to_text = {}
with open(args.data_json, 'r') as f:
    for line in tqdm(f):
        dict = json.loads(line.strip())
        id_to_text[dict['id']] = dict['text']


# ------------------------------------------------------------------------------
# Drop in to interactive
# ------------------------------------------------------------------------------

def jaccard(a, b):
    return len(set(a.split()).intersection(b.split())) / float(len(set(a.split()).union(b.split())))


def process(query, k=1):
    # s1, s2 both passed in seperately, we concatenate them to query
    doc_names, doc_scores = ranker.closest_docs(query, k)
    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id', 'Doc Score', 'Text']
    )
    for i in range(len(doc_names)):

        if args.filter:
            # 1. see if the answer is just completely in the text (we imagine passing in s1, s2 seperately)
            if query in id_to_text[doc_names[i]]:
                continue
            elif jaccard(query, id_to_text[doc_names[i]]) > 0.9:
                continue
            else:
                table.add_row([i + 1, doc_names[i], '%.5g' % doc_scores[i], id_to_text[doc_names[i]]])
        else:
            table.add_row([i + 1, doc_names[i], '%.5g' % doc_scores[i], id_to_text[doc_names[i]]])
    print(table)


def search(s1, s2, k=1):
    # s1, s2 both passed in seperately, we concatenate them to query
    doc_names, doc_scores = ranker.closest_docs(s1.strip() + ' ' + s2, k)
    table = prettytable.PrettyTable(
        ['Rank', 'Doc Id', 'Doc Score', 'Text']
    )
    for i in range(len(doc_names)):

        if args.filter:
            # 1. see if the answer is just completely in the text (we imagine passing in s1, s2 seperately)
            if s1 in id_to_text[doc_names[i]] or s2 in id_to_text[doc_names[i]]:
                continue
            # 2. check if jaccard similarity is high
            elif jaccard(s1.strip() + ' ' + s2, id_to_text[doc_names[i]]) > 0.9:
                continue
            else:
                table.add_row([i + 1, doc_names[i], '%.5g' % doc_scores[i], id_to_text[doc_names[i]]])
        else:
            table.add_row([i + 1, doc_names[i], '%.5g' % doc_scores[i], id_to_text[doc_names[i]]])
    print(table)


banner = """
Interactive TF-IDF DrQA Retriever
>> process(question, k=1)
>> usage()
"""


def usage():
    print(banner)


code.interact(banner=banner, local=locals())
