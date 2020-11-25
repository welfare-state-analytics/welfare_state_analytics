#-------------------------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See https://go.microsoft.com/fwlink/?linkid=2090316 for license information.
#-------------------------------------------------------------------------------------------------------------

ARG VARIANT=3.8-slim

FROM python:$VARIANT

# Notice: part of this code is from jupyter project

LABEL MAINTAINER Roger Mähler <roger dot mahler at umu dot se>

# This Dockerfile adds a non-root user with sudo access. Use the "remoteUser"
# property in devcontainer.json to use it. On Linux, the container user's GID/UIDs
# will be updated to match your local UID/GID (when using the dockerFile property).
# See https://aka.ms/vscode-remote/containers/non-root-user for details.ARG USERNAME="vscode"
ARG USERNAME=vscode
ARG USER_UID="1000"
ARG USER_GID="100"

# Uncomment the following COPY line and the corresponding lines in the `RUN` command if you wish to
# include your requirements in the image itself. Only do this if your requirements rarely change.
# COPY requirements.txt /tmp/pip-tmp/

# Set to false to skip installing zsh and Oh My ZSH!
ARG INSTALL_ZSH="true"

ARG JAVA_URL=https://github.com/AdoptOpenJDK/openjdk8-binaries/releases/download/jdk8u242-b08_openj9-0.18.1/OpenJDK8U-jre_x64_linux_openj9_8u242b08_openj9-0.18.1.tar.gz

ENV USERNAME=$USERNAME \
    USER_UID=$USER_UID \
    USER_GID=$USER_GID \
    \
    DEBIAN_FRONTEND=noninteractive

USER root

COPY debian-base.sh .

RUN set -ex; \
    \
    && chmod +x debian-base.sh \
    \
    && apt-get update -qq && apt-get -y -qq dist-upgrade \
    \
    && /bin/bash /tmp/common-setup.sh "$INSTALL_ZSH" "$USERNAME" "$USER_UID" "$USER_GID" \
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
    #    jupyterlab==1.2.6 \
    #    'notebook>=5.0,<=6.0' \

#    && pip install jupyterhub==${JUPYTERHUB_VERSION} \

RUN jupyter labextension install \
         @jupyter-widgets/jupyterlab-manager \
         jupyter-matplotlib@0.7.2 \
         @bokeh/jupyter_bokeh@2.0.2 \
         @jupyterlab/toc \
         @jupyterlab/google-drive \
         jupyter-cytoscape

# Install R https://cran.r-project.org/bin/linux/debian/

#RUN jupyter labextension install --minimize=False jupyter-cytoscape

ENTRYPOINT ["tini", "-g", "--"]

RUN adduser $USERNAME --uid $USER_UID --gid $USER_GID --disabled-password --gecos '' --shell /bin/bash \
    && adduser $USERNAME sudo \
    && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
# RUN useradd -m $USERNAME

ENV HOME=/home/$USERNAME
USER $USERNAME
WORKDIR $HOME

RUN mkdir -p $HOME/work
# $HOME/bin $HOME/.jupyter/lab

WORKDIR $HOME/work
