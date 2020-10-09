
.DEFAULT_GOAL=lint

SOURCE_FOLDERS=notebooks scripts tests

init:
	@pip install --upgrade pip poetry
	@pip install poetry --upgrade
	@poetry install

build: requirements.txt
	@poetry build

test-coverage:
	-poetry run coverage --rcfile=.coveragerc run -m pytest
	-coveralls

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

requirements.txt: poetry.lock
	@poetry export -f requirements.txt --output requirements.txt

.PHONY: init build format yapf black lint pylint pylint2 flake8 clean test test-coverage update
