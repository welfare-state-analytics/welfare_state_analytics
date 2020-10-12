#!/bin/bash

poetry run jupyter labextension install \
    @jupyter-widgets/jupyterlab-manager \
    @bokeh/jupyter_bokeh \
    jupyter-matplotlib \
    jupyterlab-jupytext \
    ipyaggrid

