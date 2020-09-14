#!/bin/bash
# HFST-swener

HSFT_SWENER_VERSION=0.9.3
HFST_SWENER_URL=http://www.ling.helsinki.fi/users/janiemi/finclarin/ner/hfst-swener-${HSFT_SWENER_VERSION}.tgz

cd /tmp

wget -qO-  ${HFST_SWENER_URL} | tar xvz

cd hfst-swener-${HSFT_SWENER_VERSION}/scripts
sed -i 's:#! \/usr/bin/env python:#! /usr/bin/env python2:g' *.py
cd ..

./configure

make
make install

cd /tmp

rm -rf /tmp/hfst-swener-${HSFT_SWENER_VERSION}
