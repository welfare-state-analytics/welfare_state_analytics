#!/bin/bash

if [ "$NLTK_DATA" == "" ]; then
    mkdir -p $HOME/nltk_data
    export NLTK_DATA=$HOME/nltk_data
fi

poetry run python -m nltk.downloader -d $(NLTK_DATA) stopwords punkt sentiwordnet
# Using Ubuntu
#curl -fsSL https://deb.nodesource.com/setup_15.x | sudo -E bash -
#sudo apt-get install -y nodejs


make labextension

