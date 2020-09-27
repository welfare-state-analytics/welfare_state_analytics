#!/bin/bash

script_path="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

pipenv --rm && pipenv install --dev

pipenv run ${script_path}/install_extensions.sh

pipenv run pip install kblab-client

#pipenv run pip freeze > requirements.txt

jq -r '.default
        | to_entries[]
        | .key + .value.version' \
    Pipfile.lock > requirements.txt

