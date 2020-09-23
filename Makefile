
.DEFAULT_GOAL=lint

init:
	@pip install --upgrade pip
ifeq (, $(PIPENV_PATH))
	@pip install pipenv --upgrade
endif
	@export PIPENV_TIMEOUT=7200
	@pipenv install --dev

test-coverage:
	-pipenv run coverage --rcfile=.coveragerc run -m pytest
	-coveralls

test:
	@pipenv run pytest -v --durations=0
	# --failed-first --maxfail=1

lint:
	-pipenv run pylint westac | sort | uniq > pylint.log

clean:
	@rm -rf .pytest_cache
	@find -name __pycache__ | xargs rm -r
	@rm -rf westac/test/output
	# -@pipenv clean || true

clean_cache:
	@# Try this command when failed in locking
	@# https://pipenv.kennethreitz.org/en/latest/diagnose/
	@export PIPENV_VENV_IN_PROJECT=true
	@export PIPENV_TIMEOUT=7200
	@pipenv lock --clear

update:
	#@export PIPENV_VENV_IN_PROJECT=true
	@export PIPENV_TIMEOUT=7200
	@pipenv update --outdated



requirements.txt: Pipfile.lock
	@pipenv run pip freeze > requirements.txt

.PHONY: init lint clean test test-coverage update
