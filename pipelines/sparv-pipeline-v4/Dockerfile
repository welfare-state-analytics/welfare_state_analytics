FROM python:3.8-slim-buster as sparv-build

LABEL maintainer="roger dot mahler at humlab dot umu dot se"

ARG BUILD_SPARV_MODELS=no

ENV DEBIAN_FRONTEND noninteractive

ENV HOME=/home
ENV SPARV_DIR=/sparv
ENV SHELL=/bin/bash

USER root

COPY ./scripts/install-base.sh /tmp/

RUN chmod +x /tmp/*.sh \
    && /tmp/install-base.sh

ENV LC_ALL=en_US.UTF-8
ENV LANG=en_US.UTF-8
ENV LANGUAGE=en_US.UTF-8

##############################################################################
# Sparv v4
##############################################################################

WORKDIR /sparv/bin

COPY ./scripts/*.sh /tmp/

RUN echo "installing 3rd-party software" \
    && chmod +x /tmp/*.sh \
    && /tmp/install-maltparser.sh \
    && /tmp/install-hunpos.sh \
    && /tmp/install-sparv-wsd.sh \
    && /tmp/install-hfst-swener.sh
    # && /tmp/install-cwb.sh

WORKDIR /home
# FROM python:3.8-slim-buster as prod-build
# COPY --from=sparv-build 

COPY entry-point.sh scripts/install-sparv.sh /sparv/bin/
RUN chmod +x /sparv/bin/entry-point.sh /sparv/bin/install-sparv.sh

RUN groupadd -r sparv --gid=1024 \
    && useradd -r -g sparv --uid=1024 -d /home -s /bin/bash sparv \
    && chown -R sparv:sparv /home /sparv

# Don't remove entry-point.sh from this statement: (needed since models can be empty!)
COPY entry-point.sh models* /sparv/

USER sparv

ENV PATH="/home/.local/bin:$PATH"

RUN echo "installing Sparv..." \
    && /sparv/bin/install-sparv.sh

WORKDIR /work

USER root

RUN /tmp/install-clear.sh

##############################################################################
# ENTRYPOINT
##############################################################################

ENTRYPOINT [ "/sparv/bin/entry-point.sh" ]

#CMD [ "/sparv/bin/entry-point.sh" ]
