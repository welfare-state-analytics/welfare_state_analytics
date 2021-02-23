# The Welfare State Analytics Text Analysis Repository

## About the Project

Welfare State Analytics. Text Mining and Modeling Swedish Politics, Media & Culture, 1945-1989 (WeStAc) is a digital humanities research project with five co-operatings partners: Umeå University, Uppsala University, Aalto University (Finland) and the National Library of Sweden.

The project will digitise literature, curate already digitised collections, and perform research via probabilistic methods and text mining models. WeStAc will both digitise and curate three massive textual datasets—in all, Big Data of almost four billion tokens—from the domains of Swedish politics, news media and literary culture during the second half of the 20th century.

## Installation

### Local install using pipenv

See [this page](https://github.com/humlab/welfare_state_analytics/wiki/How-to:-Install-notebooks-on-local-machine).

### JupyterHub installation

The `westac_hub` repository contains a ready-to-use Docker setup (`Dockerfile` and `docker-compose.yml`) for a Jupyter Hub using `nginx` as reverse-proxy. The default setup uses `DockerSpawner` that spawns containers as specified in `westac_lab`, and Github for autorization (OAuth2). See the Makefile on how to build the project.

### Single Docker container

You can also run the `westac_lab` container as a single Docker container if you have Docker installed on your computer.
