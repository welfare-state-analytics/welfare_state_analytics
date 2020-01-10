FROM jupyterhub/jupyterhub:latest

COPY ./jupyterhub_config.github.py /srv/jupyterhub/jupyterhub_config.py
COPY ./userlist /srv/jupyterhub/userlist

RUN wget https://raw.githubusercontent.com/jupyterhub/jupyterhub/0.9.3/examples/cull-idle/cull_idle_servers.py

RUN pip install \
    dockerspawner==0.10.0 \
    oauthenticator \
    jhub_cas_authenticator==1.0.0
