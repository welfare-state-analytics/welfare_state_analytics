ARG JUPYTERHUB_VERSION

FROM jupyterhub/jupyterhub:${JUPYTERHUB_VERSION}
#$JUPYTERHUB_VERSION

# Update and install some package
RUN apt-get update && apt-get install -yq --no-install-recommends \
	vim git curl wget  \
    libmemcached-dev \
    libsqlite3-dev \
    libzmq3-dev \
    make nodejs node-gyp npm \
    pandoc \
    sqlite3 \
    zlib1g-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*  && hash -r


RUN wget https://raw.githubusercontent.com/jupyterhub/jupyterhub/0.9.3/examples/cull-idle/cull_idle_servers.py

RUN pip install --upgrade pip && pip install \
    psycopg2-binary \
    netifaces \
    git+https://github.com/jupyterhub/dockerspawner.git \
    oauthenticator \
    jhub_cas_authenticator

COPY config/userlist /srv/jupyterhub/userlist

COPY jupyterhub_config.py /srv/jupyterhub/jupyterhub_config.py

WORKDIR /srv

CMD ["jupyterhub", "-f", "/srv/jupyterhub/jupyterhub_config.py"]