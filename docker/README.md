# Docker Jupyter Lab Playground

Build Docker image:
```bash
docker build --rm -f "jupyterlab_nlp\Dockerfile" -t jupyterlab_playground:latest
```

Start Docker container:
```bash
docker run -it --rm -p 8888:8888 -p 4040:4040 jupyterlab_playground
```

Open shell to running container
```bash
docker ps
docker exec -it <container-id> /bin/bash
```

https://github.com/manniche/dockerized-jupyterhub/tree/master
https://github.com/USGS-CMG/data-life-cycle-cloud-docker-jupyterhub/blob/837c2b7de783af385ed309bedc0297c6f972f261/docker-compose.yml
https://github.com/defeo/jupyterhub-docker/blob/d409b06afa64fcc2564ac4db03c9e0b4812037bd/docker-compose.yml
https://github.com/4dn-dcic/jupyterhub-docker