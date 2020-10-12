#!/bin/bash

pipenv run jupyter labextension install \
    @jupyter-widgets/jupyterlab-manager \
    beakerx-jupyterlab \
    @bokeh/jupyter_bokeh \
    jupyter-matplotlib \
    jupyterlab-jupytext \
    ipyaggrid

pipenv run jupyter labextension install \
    @jupyter-widgets/jupyterlab-manager \
    @bokeh/jupyter_bokeh \
    jupyter-matplotlib \
    jupyterlab-jupytext \
    ipyaggrid