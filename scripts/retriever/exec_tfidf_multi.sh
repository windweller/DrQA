#!/usr/bin/env bash

for file_name in {00..10}
do
    python3 build_tfidf.py /home/anie/DisExtract/preprocessing/corpus/because/shards_db/because_db_split${file_name}.db /home/anie/DisExtract/preprocessing/corpus/because/shards_db/ --num-workers 4 &
done

