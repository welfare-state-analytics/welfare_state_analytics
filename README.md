# The Welfare State Analytics Text Analysis Repository

This repository contains various Jupyter Notebooks for exploring the curated corpus of Riksdagens Protocol.

## About the Project

Welfare State Analytics. Text Mining and Modeling Swedish Politics, Media & Culture, 1945-1989 (WeStAc) is a digital humanities research project with five co-operatings partners: Umeå University, Uppsala University, Aalto University (Finland) and the National Library of Sweden.

The project will digitise literature, curate already digitised collections, and perform research via probabilistic methods and text mining models. WeStAc will both digitise and curate three massive textual datasets—in all, Big Data of almost four billion tokens—from the domains of Swedish politics, news media and literary culture during the second half of the 20th century.

## Installation

### JupyterHub installation

The `westac_hub` repository contains a ready-to-use Docker setup (`Dockerfile` and `docker-compose.yml`) for a Jupyter Hub using `nginx` as reverse-proxy. The default setup uses `DockerSpawner` that spawns containers as specified in `westac_lab`, and Github for autorization (OAuth2). See the Makefile on how to build the project.

### Single Docker container

You can also run the `westac_lab` container as a single Docker container if you have Docker installed on your computer.

## HOWTO Prepare a new version om Riksdagens Protokoll Corpus

### Prerequisites

 - The Riksdagen Protokoll corpus XML files have been tagged using [welfare-state-analytics/pyriksprot_tagger](https://github.com/welfare-state-analytics/pyriksprot_tagger)
 - The [welfare-state-analytics/pyriksprot](https://github.com/welfare-state-analytics/pyriksprot) has been used to process corpus metadata and speech corpus. (`make full`)
 - Important! The configuration .env files has been updated with target corpus version.

### Create a default DTM (document-term-matrix)

The command `make default-riksprot-dtm` will create a DTM based on the settings found in 'opts/dtm_riksprot.yml`.

Run `make `

