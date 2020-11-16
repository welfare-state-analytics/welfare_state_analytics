.DEFAULT_GOAL=lint
SHELL := /bin/bash
SOURCE_FOLDERS=notebooks scripts tests
PACKAGE_FOLDER=notebooks

release: ready guard_clean_working_repository bump.patch tag

ready: tools clean tidy test lint build

build: penelope-pypi requirements.txt write_to_ipynb
	@poetry build
	@echo "Penelope, requirements and ipynb files is now up-to-date"

lint: pylint flake8

tidy: black isort

tidy-to-git: guard_clean_working_repository tidy
	@status="$$(git status --porcelain)"
	@if [[ "$$status" != "" ]]; then
		@git add .
		@git commit -m "make tidy"
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

.ONESHELL: guard_clean_working_repository
guard_clean_working_repository:
	@status="$$(git status --porcelain)"
	@if [[ "$$status" != "" ]]; then
		echo "error: changes exists, please commit or stash them: "
		echo "$$status"
		exit 65
	fi

version:
	@echo $(shell grep "^version \= " pyproject.toml | sed "s/version = //" | sed "s/\"//g")

tools:
	@pip install --upgrade pip --quiet
	@pip install poetry --upgrade --quiet

penelope-pypi:
	@poetry remove humlab-penelope
	@poetry add humlab-penelope

penelope-edit-mode:
	@poetry install --develop ../../penelope

bump.patch: requirements.txt
	@poetry run dephell project bump patch
	@git add pyproject.toml requirements.txt
	@git commit -m "Bump version patch"
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

clean_cache:
	@poetry cache clear pypi --all

data: nltk_data spacy_data

update:
	@poetry update

nltk_data:
	@mkdir -p $(NLTK_DATA)
	@poetry run python -m nltk.downloader -d $(NLTK_DATA) stopwords punkt sentiwordnet

spacy_data:
	@poetry run python -m spacy download en

IPYNB_FILES := $(shell find ./notebooks -name "*.ipynb" -type f \( ! -name "*checkpoint*" \) -print)
PY_FILES := $(IPYNB_FILES:.ipynb=.py)

# Create a paired `py` file for all `ipynb` that doesn't have a corresponding `py` file
pair_ipynb: $(PY_FILES)
	@echo "hello"

$(PY_FILES):%.py:%.ipynb
	@echo target is $@, source is $<
	@poetry run jupytext --quiet --set-formats ipynb,py:percent $<

# The same, but using a bash-loop:
# pair_ipynb:
# 	for ipynb_path in $(IPYNB_FILES) ; do \
# 		ipynb_basepath="$${ipynb_path%.*}" ;\
# 		py_filepath=$${ipynb_basepath}.py ;\
# 		if [ ! -f $$py_filepath ] ; then \
# 			echo "info: pairing $$ipynb_path with formats ipynb,py..." ;\
# 			poetry run jupytext --quiet --set-formats ipynb,py:percent $$ipynb_path ;\
# 		fi \
# 	done

unpair_ipynb:
	@for ipynb_path in $(IPYNB_FILES) ; do \
        echo "info: unpairing $$ipynb_path..." ;\
		ipynb_basepath="$${ipynb_path%.*}" ;\
		py_filepath=$${ipynb_basepath}.py ;\
        poetry run jupytext --quiet --update-metadata '{"jupytext": null}' $$ipynb_path &> /dev/null ;\
        rm -f $$py_filepath ;\
	done

# The `sync` command updates paired file types based on latest timestamp
sync_ipynb:
	@echo "Syncing of PY <=> IPYNB is TURNED OFF. Only one-way write of PY => IPYNB is allowed"
	# for ipynb_path in $(IPYNB_FILES) ; do \
    #     poetry run jupytext --sync $$ipynb_path ;\
	# done

# Forces overwrite of ìpynb` using `--to notebook`
write_to_ipynb:
	for ipynb_path in $(IPYNB_FILES) ; do \
		py_filepath=$${ipynb_path%.*}.py ;\
		poetry run jupytext --to notebook $$py_filepath
	done

write_to_ipynb2:
	for ipynb_path in $(IPYNB_FILES) ; do \
		py_filepath=$${ipynb_path%.*}.py ;\
		jupytext --to notebook $$py_filepath
	done

labextension:
	@poetry run jupyter labextension install \
		@jupyter-widgets/jupyterlab-manager \
		@bokeh/jupyter_bokeh \
		jupyter-matplotlib \
		jupyterlab-jupytext \
		ipyaggrid \
		qgrid2


pre_commit_ipynb:
	@poetry run jupytext --sync --pre-commit
	@chmod u+x .git/hooks/pre-commit

requirements.txt: poetry.lock
	@poetry export -f requirements.txt --output requirements.txt

gh:
	@sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-key C99B11DEB97541F0
	@sudo apt-add-repository https://cli.github.com/packages
	@sudo apt update && sudo apt install gh

check-gh: gh-exists
gh-exists: ; @which gh > /dev/null

.ONESHELL: pair_ipynb unpair_ipynb sync_ipynb update_ipynb

.PHONY: help check init version
.PHONY: lint flake8 pylint pylint2 yapf black isort tidy tidy-to-git
.PHONY: test test-coverage pytest
.PHONY: ready build tag bump.patch release
.PHONY: clean clean_cache update
.PHONY: install_graphtool gh check-gh gh-exists tools
.PHONY: data spacy_data nltk_data
.PHONY: pair_ipynb unpair_ipynb sync_ipynb update_ipynb write_to_ipynb
.PHONY: labextension

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
	@echo " make write_to_ipynb   Write .py in %percent format to .ipynb"
	@echo " make pair_ipynb       Adds Jupytext pairing to *.ipynb"
