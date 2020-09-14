#!/bin/bash
# MaltParser

MALT_PARSER_VERSION=1.7.2

cd ${SPARV_DIR}/bin

wget -qO- http://maltparser.org/dist/maltparser-${MALT_PARSER_VERSION}.tar.gz | tar xz
