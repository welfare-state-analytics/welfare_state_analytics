#!/bin/bash
# IMS Open Corpus Workbench
#  http://cwb.sourceforge.net/index.php

cd /tmp

cd $SPARV_DIR/bin

mkdir -p ~/.subversion
echo "[global]" > ~/.subversion/servers
echo "http-timeout = 6000" >> ~/.subversion/servers

svn co http://svn.code.sf.net/p/cwb/code/cwb/trunk cwb --quiet

cd cwb

./install-scripts/install-linux --quiet

mkdir -p ~/cwb/data
export CWB_DATADIR=~/cwb/data;
export CORPUS_REGISTRY=~/cwb/registry
