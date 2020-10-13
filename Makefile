
.DEFAULT_GOAL=lint
SHELL := /bin/bash
SOURCE_FOLDERS=notebooks scripts tests

init:
	@pip install --upgrade pip
	@pip install poetry --upgrade
	@poetry install

build: penelope requirements.txt write_to_ipynb
	# @-poetry build

penelope:
	@poetry update penelope

test-coverage:
	-poetry run coverage --rcfile=.coveragerc run -m pytest
	-poetry run coveralls

test: clean
	@poetry run pytest --verbose --durations=0 \
		--cov=notebooks \
		--cov-report=term \
		--cov-report=xml \
		--cov-report=html \
		tests

pylint:
	@poetry run pylint $(SOURCE_FOLDERS)
	# @poetry run mypy --version
	# @poetry run mypy .

pylint2:
	@find $(SOURCE_FOLDERS) -type f -name "*.py" | grep -v .ipynb_checkpoints | xargs poetry run pylint --disable=W0511

pylint2nb:
	@find notebooks -type f -name "*.py" | grep -v .ipynb_checkpoints | xargs poetry run pylint --disable=W0511

flake8:
	@poetry run flake8 --version
	@poetry run flake8

lint: pylint flake8

lint2file:
	@poetry run flake8 --version
	@poetry run flake8
	# @poetry run pylint $(SOURCE_FOLDERS) | sort | uniq | grep -v "************* Module" > pylint.log

format: clean black isort

isort:
	@poetry run isort $(SOURCE_FOLDERS)

yapf: clean
	@poetry run yapf --version
	@poetry run yapf --in-place --recursive $(SOURCE_FOLDERS)

black:clean
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

update:
	@poetry update

labextension:
	@poetry run jupyter labextension install \
		@jupyter-widgets/jupyterlab-manager \
		@bokeh/jupyter_bokeh \
		jupyter-matplotlib \
		jupyterlab-jupytext \
		ipyaggrid

requirements.txt: poetry.lock
	@poetry export -f requirements.txt --output requirements.txt

IPYNB_FILES := $(shell find ./notebooks -name "*.ipynb" -type f \( ! -name "*checkpoint*" \) -print)
PY_FILES := $(IPYNB_FILES:.ipynb=.py)

# Create a paired `py` file for all `ipynb` that doesn't have a crorresponding `py` file
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
	for ipynb_path in $(IPYNB_FILES) ; do \
        poetry run jupytext --sync $$ipynb_path ;\
	done

# Forces overwrite of ìpynb` using `--to notebook`
write_to_ipynb:
	for ipynb_path in $(IPYNB_FILES) ; do \
		py_filepath=$${ipynb_path%.*}.py ;\
		poetry run jupytext --to notebook $$py_filepath
	done

pre_commit_ipynb:
	@poetry run jupytext --sync --pre-commit
	@chmod u+x .git/hooks/pre-commit

.ONESHELL: pair_ipynb unpair_ipynb sync_ipynb update_ipynb

.PHONY: init build format yapf black lint pylint pylint2 flake8 clean test test-coverage update labextension \
	pair_ipynb unpair_ipynb sync_ipynb update_ipynb write_to_ipynb
