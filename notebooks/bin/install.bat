#@echo off

set nltk_data_folder="%HOME%\nltk_data

iF "%NLTK_DATA%"=="" (
    not not exists "%nltk_data_folder% (
        mkdir "%HOME%\nltk_data"
    )
    "Setting NLTK_DATA to %HOME%\nltk_data"
    echo mkdir "%HOME%\nltk_data"
    echo setx NLTK_DATA="$HOME/nltk_data"
)

poetry run python -m nltk.downloader -d $(NLTK_DATA) stopwords punkt sentiwordnet

poetry run jupyter labextension install \
    @jupyter-widgets/jupyterlab-manager \
    @bokeh/jupyter_bokeh \
    @jupyter-widgets/jupyterlab-sidecar \
    jupyter-matplotlib \
    jupyterlab-jupytext \
    ipyaggrid \
    qgrid2

