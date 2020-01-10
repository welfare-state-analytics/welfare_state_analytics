FROM python:3.7-slim

# Notice: part of this code is from jupyter project
LABEL MAINTAINER Roger Mähler <roger dot mahler at umu dot se>

ARG LAB_USER="humlab"
ARG LAB_UID="1001"
ARG LAB_GID="100"

ARG TINI_VERSION=v0.18.0
ARG JUPYTERHUB_VERSION=1.0.0
ENV LAB_PORT=8888 \
    \
    LAB_USER=$LAB_USER \
    LAB_UID=$LAB_UID \
    LAB_GID=$LAB_GID \
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
    && curl -L https://raw.githubusercontent.com/pyenv/pyenv-installer/master/bin/pyenv-installer | bash \
    && curl -sL https://deb.nodesource.com/setup_12.x | bash - \
    && apt-get install -y -qq nodejs \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && rm -rf /var/lib/apt/lists/*

ENV PATH /usr/local/bin:$PATH

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

RUN python --version \
    && pip install --upgrade pip \
    && pip install pipenv \
    && pip install jupyterhub==${JUPYTERHUB_VERSION} \
    && exit 0

COPY *.sh /tmp/

RUN ls -al /tmp && chmod 0555 /tmp/*.sh \
    && /tmp/setup-user.sh \
    # && /tmp/fix-it.sh \
    && cp /tmp/start-notebook.sh /usr/local/bin/

USER $LAB_UID

ENV HOME=/home/$LAB_USER

WORKDIR $HOME

RUN mkdir -p $HOME/bin \
    mkdir -p $HOME/.jupyter/lab

COPY user-settings  $HOME/.jupyter/lab/
COPY jupyter_notebook_config.py $HOME/.jupyter/

COPY Pipfile .
COPY Pipfile.lock .

RUN \
    pipenv install \
    && pipenv run jupyter labextension install \
         @jupyter-widgets/jupyterlab-manager \
         jupyter-matplotlib \
         jupyterlab_bokeh \
         @jupyterlab/toc \
         @jupyterlab/google-drive

    # && cp -f /tmp/jupyter_notebook_config.py ~/.jupyter/jupyter_notebook_config.py

# RUN python -m nltk.downloader -d /usr/local/share/nltk_data stopwords punkt sentiwordnet \
#     \
#     && python -m spacy download en_core_web_sm \
#     && python -m spacy download fr_core_news_sm \
#     && python -m spacy download en_core_web_md \
#     && exit 0

# RUN beakerx install \
#     && jupyter labextension install \
#          beakerx-jupyterlab \
#

ENV NLTK_DATA=/usr/local/share/nltk_data

EXPOSE ${LAB_PORT}

# #  Setting to an empty string disables authentication altogether, which is NOT RECOMMENDED c.NotebookApp.token = ''

RUN mkdir ~/notebooks

ENTRYPOINT ["tini", "-g", "--"]

# CMD ["start-notebook.sh"]
EXPOSE 8888
CMD ["jupyter", "lab", "--port=8888", "--no-browser", "--ip=0.0.0.0"]
