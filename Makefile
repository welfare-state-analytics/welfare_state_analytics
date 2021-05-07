.DEFAULT_GOAL=lint
SHELL := /bin/bash
SOURCE_FOLDERS=notebooks scripts tests
PACKAGE_FOLDER=notebooks
SPACY_MODEL=en_core_web_sm

faster-release: bump.patch tag

fast-release: clean git-ipynb requirements.txt-to-git guard-clean-working-repository bump.patch tag publish

release: ready guard-clean-working-repository bump.patch tag

ready: tools paths clean tidy test lint requirements.txt build

build: penelope-production-mode requirements.txt-to-git
	@poetry build

publish:
	@poetry publish

lint: tidy pylint flake8

tidy: black isort

tidy-to-git: guard-clean-working-repository tidy
	@status="$$(git status --porcelain)"
	@if [[ "$$status" != "" ]]; then
		@git add .
		@git commit -m "📌 make tidy"
		@git push
	fi

test: clean
	@mkdir -p ./tests/output
	@poetry run pytest --verbose --durations=0 \
		--cov=$(PACKAGE_FOLDER) \
		--cov-report=term \
		--cov-report=xml \
		--cov-report=html \
		tests
	@rm -rf ./tests/output/*

init: tools
	@poetry install

.ONESHELL: paths
paths:
	@for folder in `find . -type f -name "*.ipynb" | xargs dirname | grep -v ".ipynb_checkpoints" | sort | uniq | xargs` ; do \
		pushd .  > /dev/null ; \
		cd $$folder ; \
		rm -f __paths__.py ; \
		ln -s ../../__paths__.py __paths__.py ; \
		popd  > /dev/null ; \
	done

.ONESHELL: guard-clean-working-repository
guard-clean-working-repository:
	@status="$$(git status --porcelain)"
	@if [[ "$$status" != "" ]]; then
		echo "error: changes exists, please commit or stash them: "
		echo "$$status"
		exit 65
	fi

wc:
	@poetry run time find . -name '*.py' -type f -exec cat \{\} \; | tqdm | wc -l

version:
	@poetry version
	@poetry env info -p

tools:
	@poetry run pip install --upgrade pip --quiet
	@poetry run pip install poetry --upgrade --quiet

penelope-production-mode: penelope-uninstall
	@poetry add humlab-penelope

.ONESHELL: penelope-edit-mode
penelope-edit-mode: penelope-uninstall
	@poetry@3940 add --editable ../../penelope

penelope-uninstall:
	@poetry remove humlab-penelope
	@poetry run pip uninstall humlab-penelope --yes

bump.patch: requirements.txt
	@poetry run dephell project bump patch
	@git add pyproject.toml requirements.txt
	@git commit -m "📌bump version patch"
	@git push

tag:
	@poetry build
	@git push
	@git tag $(shell grep "^version \= " pyproject.toml | sed "s/version = //" | sed "s/\"//g") -a
	@git push origin --tags

test-coverage:
	-poetry run coverage --rcfile=.coveragerc run -m pytest
	-poetry run coveralls

pytest:
	@mkdir -p ./tests/output
	@poetry run pytest --quiet tests

pylint:
	@time poetry run pylint $(SOURCE_FOLDERS)
	# @poetry run mypy --version
	# @poetry run mypy .

pylint2:
	@-find $(SOURCE_FOLDERS) -type f -name "*.py" | \
		grep -v .ipynb_checkpoints | \
			poetry run xargs -I @@ bash -c '{ echo "@@" ; pylint "@@" ; }'

	# xargs poetry run pylint --disable=W0511 | sort | uniq

flake8:
	@poetry run flake8 --version
	@poetry run flake8 $(SOURCE_FOLDERS)

isort:
	@poetry run isort --profile black --float-to-top --line-length 120 --py 38 $(SOURCE_FOLDERS)

yapf: clean
	@poetry run yapf --version
	@poetry run yapf --in-place --recursive $(SOURCE_FOLDERS)

black: clean
	@poetry run black --version
	@poetry run black --line-length 120 --target-version py38 --skip-string-normalization $(SOURCE_FOLDERS)

clean:
	@rm -rf .pytest_cache build dist .eggs *.egg-info
	@rm -rf .coverage coverage.xml htmlcov report.xml .tox
	@find . -type d -name '__pycache__' -exec rm -rf {} +
	@find . -type d -name '*pytest_cache*' -exec rm -rf {} +
	@find . -type d -name '.mypy_cache' -exec rm -rf {} +
	@rm -rf tests/output

clean-cache:
	@poetry cache clear pypi --all
	@poetry install --remove-untracked

data: nltk-data spacy-data

update:
	@poetry update

re-create-env:
	@poetry remove humlab-penelope
	@poetry run pip uninstall humlab-penelope
	@poetry env remove `poetry run which python`
	@poetry add ../../penelope

nltk-data:
	@mkdir -p $(NLTK_DATA)
	@poetry run python -m nltk.downloader -d $(NLTK_DATA) stopwords punkt sentiwordnet

spacy-data:
	@poetry run python -m spacy download $(SPACY_MODEL)
	@poetry run python -m spacy link $(SPACY_MODEL) en --force

requirements.txt: poetry.lock
	@poetry export --without-hashes -f requirements.txt --output requirements.txt

requirements.txt-to-git: requirements.txt
	@git add requirements.txt
	@git commit -m "📌updated requirements.txt"
	@git push

IPYNB_FILES := $(shell find ./notebooks -name "*.ipynb" -type f ! -path "./notebooks/legacy/*" \( ! -name "*checkpoint*" \) -print)
PY_FILES := $(IPYNB_FILES:.ipynb=.py)

unpair-ipynb:
	@for ipynb_path in $(IPYNB_FILES) ; do \
        echo "info: unpairing $$ipynb_path..." ;\
		ipynb_basepath="$${ipynb_path%.*}" ;\
		py_filepath=$${ipynb_basepath}.py ;\
        poetry run jupytext --quiet --update-metadata '{"jupytext": null}' $$ipynb_path &> /dev/null ;\
        rm -f $$py_filepath ;\
	done

# The `sync` command updates paired file types based on latest timestamp
sync-ipynb:
	@echo "Syncing of PY <=> IPYNB is TURNED OFF. Only one-way write of PY => IPYNB is allowed"
    # poetry run jupytext --sync $(IPYNB_FILES)

write-to-ipynb:
	echo "warning: write-to-ipynb is disabled in Makefile!"
# 	poetry run jupytext --to notebook $(PY_FILES)

.PHONY: git-ipynb
git-ipynb: guard-clean-working-repository
	@poetry run jupytext --quiet --to notebook $(PY_FILES) &> /dev/null
	@status="$$(git status --porcelain)"
	@if [[ "$$status" != "" ]]; then
		@git add .
		@git commit -m "📌make git-ipynb"
		@git push
	fi

labextension:
	@poetry run jupyter labextension install \
		ipyaggrid \
		@finos/perspective-jupyterlab \
		@jupyter-widgets/jupyterlab-manager


pre-commit-ipynb:
	@poetry run jupytext --sync --pre-commit
	@chmod u+x .git/hooks/pre-commit

gh:
	@sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key C99B11DEB97541F0
	@sudo apt-add-repository https://cli.github.com/packages
	@sudo apt update && sudo apt install gh

check-gh: gh-exists
gh-exists: ; @which gh > /dev/null

.PHONY: issues issue pr prs
issues:
	gh issue list

issue:
	gh issue create

prs:
	gh pr list

pr:
	gh pr create

.ONESHELL: pair-ipynb unpair-ipynb sync-ipynb update-ipynb

.PHONY: help check init version
.PHONY: lint flake8 pylint pylint2 yapf black isort tidy tidy-to-git
.PHONY: test test-coverage pytest
.PHONY: ready build tag bump.patch release
.PHONY: clean clean-cache update
.PHONY: install-graphtool gh check-gh gh-exists tools
.PHONY: data spacy-data nltk-data
.PHONY: pair-ipynb unpair-ipynb sync-ipynb update-ipynb write-to-ipynb
.PHONY: labextension
.PHONY: wc paths

help:
	@echo "Higher level recepies: "
	@echo " make ready            Makes ready for release (tools tidy test flake8 pylint)"
	@echo " make build            Updates tools, requirement.txt and build dist/wheel"
	@echo " make release          Bumps version (patch), pushes to origin and creates a tag on origin"
	@echo " make test             Runs tests with code coverage"
	@echo " make lint             Runs pylint and flake8"
	@echo " make tidy             Runs black and isort"
	@echo " make clean            Removes temporary files, caches, build files"
	@echo " make data             Downloads NLTK and SpaCy data"
	@echo " make paths            Copies ./__paths__.py to existing ./notebooks/**/__paths__.py  "
	@echo "  "
	@echo "Lower level recepies: "
	@echo " make init             Install development tools and dependencies (dev recepie)"
	@echo " make tag              bump.patch + creates a tag on origin"
	@echo " make bump.patch       Bumps version (patch), pushes to origin"
	@echo " make pytest           Runs teets without code coverage"
	@echo " make pylint           Runs pylint"
	@echo " make pytest2          Runs pylint on a per-file basis"
	@echo " make flake8           Runs flake8 (black, flake8-pytest-style, mccabe, naming, pycodestyle, pyflakes)"
	@echo " make isort            Runs isort"
	@echo " make yapf             Runs yapf"
	@echo " make black            Runs black"
	@echo " make gh               Installs Github CLI"
	@echo " make update           Updates dependencies"
	@echo "  "
	@echo "Notebook recepies: "
	@echo " make write-to-ipynb   Write .py in %percent format to .ipynb"
	@echo " make git-ipynb        Write .py in %percent format to .ipynb and git commit results"
	@echo " make pair-ipynb       Adds Jupytext pairing to *.ipynb"
