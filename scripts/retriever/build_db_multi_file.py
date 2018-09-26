#!/usr/bin/env python3

"""
This handles a list of files
"""

import argparse
import sqlite3
import json
import os
import logging
import importlib.util

from os.path import join as pjoin
from multiprocessing import Pool as ProcessPool

from tqdm import tqdm
from drqa.retriever import utils

logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]', '%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

# ------------------------------------------------------------------------------
# Import helper
# ------------------------------------------------------------------------------


PREPROCESS_FN = None


def import_module(filename):
    """Import a module given a full path to the file."""
    spec = importlib.util.spec_from_file_location('doc_filter', filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------------------
# Store corpus.
# ------------------------------------------------------------------------------


def get_contents(filename):
    """Parse the contents of a file. Each line is a JSON encoded document."""
    global PREPROCESS_FN
    documents = []
    with open(filename) as f:
        for line in f:
            # Parse document
            doc = json.loads(line)
            # Maybe preprocess the document with custom function
            if PREPROCESS_FN:
                doc = PREPROCESS_FN(doc)
            # Skip if it is empty or None
            if not doc:
                continue
            # Add the document, no need to normalize further
            documents.append((doc['id'], doc['text']))
    return documents

def save_to_database(file_name):
    logger.info('Reading into database...')
    conn = sqlite3.connect(pjoin(args.save_path, file_name.split('.txt')[0] + '.db'))
    c = conn.cursor()
    c.execute("CREATE TABLE documents (id PRIMARY KEY, text);")

    # workers = ProcessPool(num_workers, initializer=init, initargs=(preprocess,))
    documents = get_contents(file_name)
    logger.info("finished reading the single file")
    total = len(documents)

    count = 0
    with tqdm(total=len(range(0, total, 500))) as pbar:
        for i in tqdm(range(0, total, 500)):
            count += 500
            c.executemany("INSERT INTO documents VALUES (?,?)", documents[i:i + 500])
            pbar.update()
    logger.info('Read %d docs.' % count)
    logger.info('Committing...')
    conn.commit()
    conn.close()

# we remove the multiprocessing capability of this function
def store_contents(data_path, save_path, num_workers=None):
    """
    We actually create number of db files the same as the shards
    """
    list_files = [file_name for file_name in os.listdir(data_path) if args.prefix in file_name]
    logger.info("processing through {}".format(list_files))

    workers = ProcessPool(num_workers)

    workers.imap_unordered(save_to_database, list_files)


# ------------------------------------------------------------------------------
# Main.
# ------------------------------------------------------------------------------


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', type=str, help='/path/to/data')
    parser.add_argument('save_path', type=str, help='/path/to/saved/')
    parser.add_argument('prefix', type=str, default="because_db_split", help='because_db_split')
    parser.add_argument('num_shards', type=int, default=20, help='number of shards')
    parser.add_argument('--num-workers', type=int, default=20,
                        help='Number of CPU processes (for tokenizing, etc)')
    args = parser.parse_args()

    store_contents(
        args.data_path, args.save_path, args.num_workers
    )
