FROM python:3.7-slim

# Notice: part of this code is from jupyter project
LABEL MAINTAINER Roger Mähler <roger dot mahler at umu dot se>

ARG LAB_USER="humlab"
ARG LAB_UID="1001"
ARG LAB_GID="100"
ARG LAB_PORT=8888

ARG TINI_VERSION=v0.18.0
ARG JUPYTERHUB_VERSION=1.0.0
ARG JAVA_URL=https://github.com/AdoptOpenJDK/openjdk8-binaries/releases/download/jdk8u242-b08_openj9-0.18.1/OpenJDK8U-jre_x64_linux_openj9_8u242b08_openj9-0.18.1.tar.gz
ENV LAB_PORT=${LAB_PORT} \
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
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /usr/lib/jvm \
    && cd /usr/lib/jvm \
    && wget $JAVA_URL -O OpenJDK8U-jre_x64_linux.tar.gz \
    && tar xvf OpenJDK8U-jre_x64_linux.tar.gz \
    && rm -f OpenJDK8U-jre_x64_linux.tar.gz \
    && ln -s /usr/lib/jvm/jdk8u242-b08-jre/bin/java /usr/local/bin/java

ENV PATH /usr/local/bin:$PATH
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

RUN python --version \
    && pip install --upgrade pip \
    && pip install pipenv \
    && exit 0

COPY *.sh /tmp/

RUN ls -al /tmp && chmod 0555 /tmp/*.sh \
    && /tmp/setup-user.sh \
    # && /tmp/fix-it.sh \
    && cp /tmp/start-notebook.sh /usr/local/bin/

COPY Pipfile .
COPY Pipfile.lock .

RUN sudo chown ${LAB_USER}.users Pipfile*

USER $LAB_UID
ENV HOME=/home/$LAB_USER
WORKDIR $HOME

# Install system-wide: (add "pipenv run" before beakerx/jupyter if local env)
RUN pipenv install --ignore-pipfile --deploy # --system
RUN pipenv install --quiet --yes \
    'notebook=6.0.3' \
    'jupyterhub=${JUPYTERHUB_VERSION} ' \
    'jupyterlab=1.2.6' && \
    npm cache clean --force && \
    jupyter notebook --generate-config && \
    rm -rf $CONDA_DIR/share/jupyter/lab/staging && \
    rm -rf /home/$NB_USER/.cache/yarn && \
    fix-permissions $CONDA_DIR && \
    fix-permissions /home/$NB_USER

RUN beakerx install
RUN jupyter labextension install \
         @jupyter-widgets/jupyterlab-manager \
         jupyter-matplotlib \
         @bokeh/jupyter_bokeh \
         @jupyterlab/toc \
         @jupyterlab/google-drive \
         beakerx-jupyterlab

# RUN python -m nltk.downloader -d /usr/local/share/nltk_data stopwords punkt sentiwordnet \
#     \
#     && python -m spacy download en_core_web_sm \
#     && python -m spacy download fr_core_news_sm \
#     && python -m spacy download en_core_web_md \
#     && exit 0

ENV NLTK_DATA=/usr/local/share/nltk_data

EXPOSE ${LAB_PORT}

ENTRYPOINT ["tini", "-g", "--"]



RUN mkdir -p $HOME/notebooks \
    && mkdir -p $HOME/bin \
    && mkdir -p $HOME/.jupyter/lab

COPY user-settings  $HOME/.jupyter/lab/
COPY jupyter_notebook_config.py $HOME/.jupyter/

CMD [ "jupyter", "lab", "--no-browser", "--ip=0.0.0.0" ]
