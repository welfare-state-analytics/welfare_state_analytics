FROM python:3.8-slim

# Notice: part of this code is from jupyter project
LABEL MAINTAINER Roger Mähler <roger dot mahler at umu dot se>

ARG NB_USER="jovyan"
ARG NB_UID="1000"
ARG NB_GID="100"
ARG NB_PORT=8888

ARG TINI_VERSION=v0.18.0
ARG JUPYTERHUB_VERSION=1.0.0
ARG JAVA_URL=https://github.com/AdoptOpenJDK/openjdk8-binaries/releases/download/jdk8u242-b08_openj9-0.18.1/OpenJDK8U-jre_x64_linux_openj9_8u242b08_openj9-0.18.1.tar.gz
ENV NB_PORT=${NB_PORT} \
    \
    NB_USER=$NB_USER \
    NB_UID=$NB_UID \
    NB_GID=$NB_GID \
    \
    SHELL=/bin/bash \
    DEBIAN_FRONTEND=noninteractive

USER root

RUN set -ex; \
    \
    apt-get update -qq && apt-get -y -qq dist-upgrade \
    \
    && mkdir -p /usr/share/man/man1 \
    && apt-get install -y -qq --no-install-recommends \
        locales \
        build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev  \
        software-properties-common \
        wget curl zip unzip bzip2 llvm libncurses5-dev \
        git vim \
        ca-certificates \
        netbase netcat \
        sudo \
        fonts-liberation \
        \
        r-base r-base-dev libopenblas-base \
    \
    && curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash \
    && curl -sL https://deb.nodesource.com/setup_12.x | bash - \
    && apt-get install -y -qq nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/lib/jvm \
    && cd /usr/lib/jvm \
    && wget $JAVA_URL -O OpenJDK8U-jre_x64_linux.tar.gz \
    && tar xvf OpenJDK8U-jre_x64_linux.tar.gz \
    && rm -f OpenJDK8U-jre_x64_linux.tar.gz \
    && ln -s /usr/lib/jvm/jdk8u242-b08-jre/bin/java /usr/local/bin/java

ENV PATH=/usr/local/bin:$PATH
ENV JAVA_HOME=/usr/lib/jvm/jdk8u242-b08-jre

WORKDIR /tmp

RUN npm install npm@latest -g \
    \
    && npm install -g n \
    && npm cache clean -f \
    && npm config set progress false \
    && npm config set registry http://registry.npmjs.org/ \
    && npm i -g zeromq --unsafe-perm \
    && npm install -g node-gyp \
    \
    && n stable

ADD https://github.com/krallin/tini/releases/download/${TINI_VERSION}/tini /usr/local/bin/tini

RUN chmod +x /usr/local/bin/tini

COPY requirements.txt .

RUN python --version \
    && pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install --upgrade jupyterhub==1.0.0 \
    && exit 0

# RUN beakerx install
RUN jupyter labextension install \
         @jupyter-widgets/jupyterlab-manager@2.0.0 \
         jupyter-matplotlib@0.7.3 \
         @bokeh/jupyter_bokeh@2.0.3

        #  @jupyterlab/toc \
        #  beakerx-jupyterlab \
        #  jupyter-cytoscape@1.0.3 \
        #  qgrid2


# Install R https://cran.r-project.org/bin/linux/debian/

#RUN jupyter labextension install --minimize=False jupyter-cytoscape

# RUN python -m nltk.downloader -d /usr/local/share/nltk_data stopwords punkt sentiwordnet \
#     \
#     && python -m spacy download en_core_web_sm \
#     && python -m spacy download fr_core_news_sm \
#     && python -m spacy download en_core_web_md \
#     && exit 0

ENV NLTK_DATA=/usr/local/share/nltk_data

# EXPOSE ${NB_PORT}

ENTRYPOINT ["tini", "-g", "--"]

RUN adduser $NB_USER --uid $NB_UID --gid $NB_GID --disabled-password --gecos '' --shell /bin/bash \
    && adduser $NB_USER sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
# RUN useradd -m $NB_USER

ENV HOME=/home/$NB_USER
USER $NB_USER
WORKDIR $HOME

RUN mkdir -p $HOME/work
# $HOME/bin $HOME/.jupyter/lab

WORKDIR $HOME/work

ADD https://api.github.com/repos/welfare-state-analytics/welfare_state_analytics/git/refs/heads/master version.json

RUN git clone https://github.com/welfare-state-analytics/welfare_state_analytics.git \
    && cd welfare_state_analytics \
    && rm -rf .git* .docker* .travis* Pipfile* archive docker pytest* run* westac_hub Makefile requirements.txt setup.py NOTE.md

#COPY user-settings  $HOME/.jupyter/lab/
#COPY jupyter_notebook_config.py $HOME/.jupyter/

CMD ["jupyterhub-singleuser"]

