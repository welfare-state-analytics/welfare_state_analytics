FROM python:3.7-slim-buster as build

LABEL maintainer="roger dot mahler at humlab dot umu dot se"

ARG SPARV_PIPELINE_PATH=/source/sparv-pipeline
ARG SPARV_MODELS=${SPARV_PIPELINE_PATH}/models
ARG SPARV_PYTHON=${SPARV_PIPELINE_PATH}
ARG SPARV_BIN=${SPARV_PIPELINE_PATH}/bin

ENV SPARV_MODELS=${SPARV_MODELS}
ENV SPARV_BIN=${SPARV_BIN}
ENV SPARV_MAKEFILES=$SPARV_PIPELINE_PATH/makefiles

ARG MALT_PARSER=maltparser-1.7.2
ARG MALT_PARSER_MODEL=swemalt-1.7.2.mco

ARG HSFT_SWENER_VERSION=0.9.3
ARG HSFT_SWENER_PACKAGE=hfst-swener-${HSFT_SWENER_VERSION}
ARG HFST_SWENER_URL=http://demo.spraakdata.gu.se/svedk/${HSFT_SWENER_PACKAGE}.tgz

ENV DEBIAN_FRONTEND noninteractive

ENV HOME=/home/$SB_USER
ENV SHELL=/bin/bash
ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8
ENV PATH="${SPARV_BIN}:${PATH}"

RUN set -ex \
    \
    && apt-get update -qq \
    && apt-get -y -qq dist-upgrade \
    \
    && mkdir -p /usr/share/man/man1 \
    \
    && apt-get install -y -qq --no-install-recommends \
        build-essential \
        nano \
        gcc g++ make cmake \
        curl wget zip unzip bzip2 tar \
        time gettext autoconf automake libtool bison flex \
        ca-certificates \
        sudo \
        locales \
        fonts-liberation \
        openjdk-11-jdk \
        git

##############################################################################
# Git LFS
##############################################################################

RUN curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash \
    && apt-get update \
    && apt-get install -y -qq --no-install-recommends git-lfs \
    && git lfs install \
    \
    && rm -rf /var/lib/apt/lists/* \
    && echo "en_US.UTF-8 UTF-8" > /etc/locale.gen && locale-gen \
    \
    && rm /bin/sh && ln -s /bin/bash /bin/sh

USER root

##############################################################################
# Sparv pipeline
##############################################################################

WORKDIR /source

RUN git clone https://github.com/spraakbanken/sparv-pipeline.git

RUN python --version \
    \
    && cd ./sparv-pipeline \
    && python3 -m venv venv \
    && source venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r requirements.txt
    # && deactivate

##############################################################################
# HFST-SWENER
##############################################################################

WORKDIR /source

RUN wget -q ${HFST_SWENER_URL} \
    && tar xf ${HSFT_SWENER_PACKAGE}.tgz \
    && cd ${HSFT_SWENER_PACKAGE} \
    && ./configure \
    && make \
    && make install \
    && cd .. && rm -rf ${HSFT_SWENER_PACKAGE}.tgz

##############################################################################
# HUNPOS (compiled from source)
##############################################################################

WORKDIR /source

RUN    apt-get update \
    \
    && apt-get install -yqq --no-install-recommends \
        ocaml-nox \
    \
    && git clone https://github.com/mivoq/hunpos.git \
    && cd hunpos \
    && mkdir build \
    && cd build \
    && cmake .. -DCMAKE_INSTALL_PREFIX=install \
    && make \
    && make install \
    && ln -s /source/hunpos/build/hunpos-tag ${SPARV_BIN}/hunpos-tag

##############################################################################
# MALT PARSER
##############################################################################

WORKDIR /source

RUN wget -q http://maltparser.org/dist/${MALT_PARSER}.tar.gz \
    \
    && tar xf ${MALT_PARSER}.tar.gz \
    \
    && ln -s /source/${MALT_PARSER} ${SPARV_PIPELINE_PATH}/${MALT_PARSER} \
    \
    && rm ${MALT_PARSER}.tar.gz \
    \
    && wget -q http://www.maltparser.org/mco/swedish_parser/${MALT_PARSER_MODEL} -o ${SPARV_MODELS}/${MALT_PARSER_MODEL}

##############################################################################
# Swedish Word-sense disambiguation tool based on SALDO.
##############################################################################

RUN wget https://github.com/spraakbanken/sparv-wsd/raw/master/bin/saldowsd.jar -P /source/sparv-pipeline/bin/wsd/

##############################################################################
# Make models
#  If models_data.zip exists then the content is unzipped in models folder.
#  The file contains cached copies of large data files that is downloaded by
#  by the build process.
##############################################################################

WORKDIR ${SPARV_PIPELINE_PATH}

# Alternative #1: Download and compile
RUN make -C models/ all && make -C models/ space

# # Alternative #2: Copy in a cached version of the downloaded files
# RUN if [ -f models_data.zip ]; then                         \
#        mv models_data.zip ${SPARV_PIPELINE_PATH}/models/    \
#        cd ${SPARV_PIPELINE_PATH}/models/                    \
#        unzip models_data.zip                                \
#        rm models_data.zip                                   \
#        cd ${SPARV_PIPELINE_PATH}                            \
#     fi                                                      \
#     && make -C models/ all \
#     && make -C models/ space

# # Alternative #3: Copy in a pre-compiled models folder (remove git cloned files first)
# RUN rm -rf ${SPARV_PIPELINE_PATH}/models
# COPY models models

# Alternative #4: Empty folder and mount the models folder instead
RUN rm -rf ${SPARV_PIPELINE_PATH}/models

##############################################################################
# ENTRYPOINT
##############################################################################


COPY ./entry-point.sh /source/sparv-pipeline/bin/entry-point.sh

RUN chmod +x /source/sparv-pipeline/bin/entry-point.sh

WORKDIR /data

ENTRYPOINT [ "/source/sparv-pipeline/bin/entry-point.sh" ]
CMD []
