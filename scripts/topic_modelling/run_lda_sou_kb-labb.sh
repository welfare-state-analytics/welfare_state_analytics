#!/bin/bash

corpus_folder="./data/SOU-KB-labb"
max_iter=2000
n_workers=1

function run_task() {
    engine=$1
    n_topics=$2
    random_seed=42
    model_name="$1.topics.$2.SOU-KB-labb-45-89.$random_seed"
    target_folder="$corpus_folder/$model_name"
    mkdir -p $target_folder
    python run_lda.py $model_name --data-folder $corpus_folder --n-topics $2 --engine "$1" --workers $n_workers --max-iter $max_iter --random-seed $random_seed --prefix "$target_folder/" > $target_folder/run.log 2>&1
    gzip $target_folder/*.txt
}

run_task "gensim_lda" 200
