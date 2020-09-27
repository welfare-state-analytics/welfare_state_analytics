#!/bin/bash
#echo "BASH_SOURCE=$BASH_SOURCE"
#echo "PYTHONPATH=$PYTHONPATH"

project_root_path=

add_project_root_to_python_path() {
    this_path=${BASH_SOURCE%/*}
    if [[ "X${this_path:0:1}" == "X." ]]; then
        # make path absolute
	    this_path=${PWD}/${this_path:1}
    fi
    project_root_path=${this_path%/welfare_state_analytics/*}/welfare_state_analytics
    export PYTHONPATH=${project_root_path}:$PYTHONPATH
}

add_project_root_to_python_path

#timestamp=`date "+%Y%m%d%H%M%S"`
input_file=$project_root_path/data/SOU-KB-labb/SOU-KB-labb-corpus-1945-1989.sparv.xml.zip
#output_file=$project_root_path/data/SOU-KB-labb/sou_kb-labb_1945-1989.text_${timestamp}.zip

pipenv run python $project_root_path/scripts/sparv-xml-extract-text.py $input_file \
     --pos-includes='|NN|JJ|AB|PM|' \
     --lemmatize \
     --lower \
     --remove-stopwords=swedish  \
     --min-word-length=2 \
     --no-keep-symbols \
     --no-keep-numerals \
     --version=3

    #--pos-excludes="|MAD|MID|PAD|"
    #--chunk-size', 'chunk_size'
