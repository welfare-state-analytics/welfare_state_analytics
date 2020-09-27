#!/bin/bash

corpus_folder="./data"
corpus_filename='SOU-KB-labb-corpus-1945-1989.sparv.xml_text_20200923195551.zip'

max_iter=2000
n_workers=1

if [ ! -d "$corpus_folder" ]; then
    echo "error: data folder '$corpus_folder' is missing (create a folder or a symbolic link)"
    echo "       e.g. ln -s /data/westac/sou_kb_labb data"
    exit 64
fi

if [ ! -f "$corpus_folder/$corpus_filename" ]; then
    echo "error: corpus file '$corpus_filename' not found in '$corpus_folder'"
    exit 64
fi

exit 0


function run_task() {

    engine=$1
    n_topics=$2
    random_seed=42
    model_name="$1.topics.$2.SOU-KB-labb-1945-1989.$random_seed"
    target_folder="$corpus_folder/$model_name"

    mkdir -p $target_folder

    pipenv run python run_lda_sou_kb-labb.py $model_name \
        --corpus-filename "$corpus_folder/$model_name" \
        --n-topics $2 \
        # --engine "$1" \
        --workers $n_workers \
        --max-iter $max_iter \
        --random-seed $random_seed \
        --prefix "$target_folder/" > $target_folder/run.log 2>&1

    gzip $target_folder/*.txt
}

run_task "gensim_lda" 200
