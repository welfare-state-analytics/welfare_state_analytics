#!/bin/bash

corpus_folder="./data"
run_script="poetry run python -m penelope.scripts.compute_topic_model"

max_iter=2000
n_workers=1

function run_task() {
    engine=$1
    n_topics=$2
    random_seed=None
    model_name="$1.topics.$2.SOU-KB-LABB.$random_seed"
    target_folder="$corpus_folder/$model_name"
    mkdir -p $target_folder
    python $run_script $model_name --data-folder $corpus_folder --n-topics $2 --engine "$1" --workers $n_workers --max-iter $max_iter --random-seed $random_seed --prefix "$target_folder/" > $target_folder/run.log 2>&1
    gzip $target_folder/*.txt
}

# run_task "gensim_mallet-lda" 50
# run_task "gensim_mallet-lda" 100
# run_task "gensim_mallet-lda" 200
# run_task "gensim_mallet-lda" 400

run_task "gensim_mallet-lda" 200

