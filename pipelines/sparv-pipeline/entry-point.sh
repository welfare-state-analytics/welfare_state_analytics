#!/bin/bash

cd /data

ls

if [ ! -f "/data/Makefile" ]; then
    echo "usage: makefile not present!"
    cp /source/sparv-pipeline/makefiles/Makefile.example ./Makefile
    echo "info: basic template Makefile copied to current directory"
    exit 64
fi

export PYTHONPATH=/source/sparv-pipeline

# if [ "$1" == "rebuild-model" ]; then
#     if [ ! -f /source/sparv-pipeline/models ]; then
#         rm -rf /data/models
#         cp -r /source/sparv-pipeline/models /data/models
#         ln -s /models /source/sparv-pipeline/models
#     fi
#     exit 0
# fi

if [ -d /data/models ]; then
    export SPARV_MODELS=/data/models
fi

make $@
