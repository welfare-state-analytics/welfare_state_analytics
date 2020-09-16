#!/bin/bash

cd /home
python -m pip install pipx --user
python -m pipx ensurepath
git clone https://github.com/spraakbanken/sparv-pipeline.git
cd sparv-pipeline
git checkout v4
pipx install .
cd /home
rm -rf /home/sparv-pipeline
chmod -R go+rx /home

if [ $BUILD_SPARV_MODELS == "yes" ]; then
    sparv build-models --language swe
elif [ -d /sparv/models ]; then
    echo "info: models found in /sparv/models (no build will be started)"
else
    echo "info: models not found or expected to be mounted /sparv/models"
    mkdir -p /sparv/models
fi

sparv setup << EOF > /dev/null

n
EOF
