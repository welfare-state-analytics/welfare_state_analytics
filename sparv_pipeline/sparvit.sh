#!/bin/bash

# Build from scratch:
# git clone ...
# docker build -t sparv-pipeline:latest .

# Alternative #4: Store models folder outside of Docker image (see Dockerfile):
SPARV_PIPELINE_MODELS=/home/roger/source/sparv-pipeline/models
if [ ! -d "${SPARV_PIPELINE_MODELS}" ]; then
    echo "error: model folder ${SPARV_PIPELINE_MODELS} not missing"
    exit
fi

docker run --rm \
    --user `id -u`:`id -g` \
    --workdir=/data \
    --mount "type=bind,src=${SPARV_PIPELINE_MODELS},dst=/source/sparv-pipeline/models" \
    --mount "type=bind,src=$(pwd),dst=/data" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    sparv-pipeline:latest "$@"


# awk 'END { printf "E%03d end\n", c > f }
# !(NR % 200) || NR == 1 { if (f) { printf "E%03d end\n", c > f; close(f) }
# printf "E%03d start\n", ++c > (f = "file" c ".txt") }
# { print > f }' large

