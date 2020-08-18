# JupyterHub deployment of WeStAc related projects

This JupyterHub setup is based on this [blog post](https://opendreamkit.org/2018/10/17/jupyterhub-docker/) and the [The Littlest JupyterHub](https://the-littlest-jupyterhub.readthedocs.io/en/latest/).

## Features

- Uses [DockerSpawner](https://github.com/jupyterhub/dockerspawner);
- Traefik as reverse HTTPS proxy.
- Central authentication
- User data persistence

## Learn more

Please read [this blog post](https://opendreamkit.org/2018/10/17/jupyterhub-docker/) and [].

## Configuration

Clone this repository and apply (at least) the following changes:

- In [`.env`](.env), set the variable `HOST` to the name of the server you
  intend to host your deployment on.
- In [`reverse-proxy/traefik.toml`](reverse-proxy/traefik.toml), edit
  the paths in `certFile` and `keyFile` and point them to your own TLS
  certificates. Possibly edit the `volumes` section in the
  `reverse-proyx` service in
  [`docker-compose.yml`](docker-compose.yml).
- In
  [`jupyterhub/jupyterhub_config.py`](jupyterhub/jupyterhub_config.py),
  edit the *"Authenticator"* section according to your institution
  authentication server.  If in doubt, [read
  here](https://jupyterhub.readthedocs.io/en/stable/getting-started/authenticators-users-basics.html).

Other changes you may like to make:

- Edit [`jupyterlab/Dockerfile`](jupyterlab/Dockerfile) to include the software you like.
- Change [`jupyterhub/jupyterhub_config.py`](jupyterhub/jupyterhub_config.py) accordingly, in particular the *"user data persistence"* section.

If the `jupyerhub_config.py` is changed, then the `westac_hub_data` data volume must be removed in order for the changes to take effect.

```bash
docker-compose down
# docker rm `docker ps -aq`
docker volume rm westac_hub_data
```

### How to start the server

Use [Docker Compose](https://docs.docker.com/compose/) to build and run the server:

```bash
docker-compose build
docker-compose up -d
```

## Acknowledgements

[OpenDreamKit](https://opendreamkit.org/).
