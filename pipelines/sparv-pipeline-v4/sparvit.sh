#!/bin/bash

# if [ -d $SPARV_DIR/models ]; then
# fi

SPARV_MODELS=/data/sparv4/models
SPARV_MODELS_OPTS=

if [ $SPARV_MODELS != "" ]; then
    SPARV_MODELS_OPTS=--volume="$SPARV_MODELS:/sparv/models:ro"
fi

docker run --rm \
    --user `id -u`:`id -g` \
    --workdir=/work \
    --mount "type=bind,src=$(pwd),dst=/work" \
    --volume="/etc/group:/etc/group:ro" \
    --volume="/etc/passwd:/etc/passwd:ro" \
    --volume="/etc/shadow:/etc/shadow:ro" \
    $SPARV_MODELS_OPTS \
    sparv_v4:latest "$@"

