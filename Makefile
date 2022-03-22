
include .env

include ./Makefile.dev

# These need to be defined
ifndef PYRIKSPROT_VERSION
$(error PYRIKSPROT_VERSION is undefined)
endif

# ifndef RIKSPROT_REPOSITORY_TAG
# $(error RIKSPROT_REPOSITORY_TAG is undefined)
# endif

PYRIKSPROT_VERSION=main
PYRIKSPROT_TESTDATA_TAR=riksprot_sample_testdata.$(PYRIKSPROT_VERSION).tar.gz
pyriksprot-test-data:
	@rm -rf tests/test_data/riksprot/$(PYRIKSPROT_VERSION)
	@wget -O /tmp/$(PYRIKSPROT_TESTDATA_TAR) https://raw.githubusercontent.com/welfare-state-analytics/pyriksprot/dev/tests/test_data/dists/$(PYRIKSPROT_TESTDATA_TAR)
	@tar -C tests/test_data/riksprot --strip 3 -xvf /tmp/$(PYRIKSPROT_TESTDATA_TAR) tests/test_data/source/$(PYRIKSPROT_VERSION)
	@rm -f /tmp/$(PYRIKSPROT_TESTDATA_TAR)
	@cp -f tests/test_data/riksprot/$(PYRIKSPROT_VERSION)/tagged_frames/*.db tests/test_data/riksprot/
	@cp -f tests/test_data/riksprot/corpus.id.yml tests/test_data/riksprot/$(PYRIKSPROT_VERSION)/corpus.yml

pyriksprot-test-topic-model:
	@echo "FIXME: this is not complete, TM must be created by penelope and copied to tests/test_data/riksprot/$(PYRIKSPROT_VERSION)"
	@PYTHONPATH=. poetry run python penelope/scripts/tm/train_id.py --options-filename opts/opts_tm_mallet_riksprot.test.yml

#scp -r roger@130.239.57.54:/home/roger/source/welfare-state-analytics/pyriksprot/tests/test_data/source/corpus.yml corpus.yml