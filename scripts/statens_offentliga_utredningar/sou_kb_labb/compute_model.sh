#!/bin/bash

function run_task() {

    engine=$1
    n_topics=$2
    corpus_filename=$3
    tag=$4
    max_iter=3000
    n_workers=1
    name="$1.topics.$2.$tag"
    #.$random_seed

    if [ ! -f "$corpus_filename" ]; then
        echo "error: corpus file '$corpus_filename' not found"
        exit 64
    fi

    poetry run python -m penelope.scripts.compute_topic_model $name \
        --corpus-filename "$corpus_filename" \
        --n-topics $2 \
        --engine "$1" \
        --max-iter $max_iter \
            &> $corpus_folder/run_$tag.log \
        --store-corpus \
        --workers $n_workers
        ##--prefix "$target_folder/$name/" #
        #--random-seed $random_seed \

    #gzip $target_folder/*.txt
}

corpus_folder="/home/roger/source/welfare-state-analytics/welfare_state_analytics/data"

run_task "gensim_mallet-lda" 50 "$corpus_folder/SOU-KB-labb-corpus-1945-1989.sparv.xml_text_20200928211813.zip" "SOU-KB-labb-1945-1989.NN"
