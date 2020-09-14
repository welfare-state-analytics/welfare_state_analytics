#!/bin/bash
# Hunpos
#  https://github.com/mivoq/hunpos


cd /tmp

git clone https://github.com/mivoq/hunpos.git

mkdir -p hunpos/build

cd hunpos/build

cmake .. -DCMAKE_INSTALL_PREFIX=install
make
make install

cp hunpos-tag $SPARV_DIR/bin

cd /tmp

rm -rf ./hunpos
