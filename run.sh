#!/bin/bash

corpus_folder="./data/textblock_politisk"
max_iter=2000
n_workers=7

function run_task() {
    engine=$1
    n_topics=$2
    model_name="$1.topics.$2.AB.DN"
    target_folder="$corpus_folder/$model_name"
    mkdir -p $target_folder
    python run_lda.py $model_name --n-topics $2 --engine "$1" --workers $n_workers --max-iter $max_iter --prefix "$target_folder/" > $target_folder/run.log 2>&1
    gzip $target_folder/*.txt
}

# run_task "gensim_mallet-lda" 50
# run_task "gensim_mallet-lda" 100
# run_task "gensim_mallet-lda" 200
# run_task "gensim_mallet-lda" 400

#run_task "gensim_mallet-lda" 210

