#!/bin/bash

corpus_folder="./data/textblock_politisk"

run_task() {
    model_name="$1.topics.$2.AB.DN"
    target_folder="$corpus_folder/$model_name"
    mkdir -p $target_folder
    python run_lda.py $model_name --n-topics $2 --engine "$1" --workers 7 --prefix "$target_folder/" > $target_folder/run.log 2>&1
    gzip $target_folder/*.txt
}

run_task "gensim_mallet-lda" 50
run_task "gensim_mallet-lda" 100
run_task "gensim_mallet-lda" 200


