#!/bin/bash

if [ "$NLTK_DATA" == "" ]; then
    mkdir -p $HOME/nltk_data
    export NLTK_DATA=$HOME/nltk_data
fi

poetry run python -m nltk.downloader -d $NLTK_DATA stopwords punkt sentiwordnet

poetry run jupyter labextension install \
    @jupyter-widgets/jupyterlab-manager \
    @bokeh/jupyter_bokeh \
    @jupyter-widgets/jupyterlab-sidecar \
    jupyter-matplotlib \
    jupyterlab-jupytext \
    ipyaggrid \
    qgrid2

