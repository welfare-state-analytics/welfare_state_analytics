
include .env

include ./Makefile.dev

# # These need to be defined
# ifndef PYRIKSPROT_VERSION
# $(error PYRIKSPROT_VERSION is undefined)
# endif

# ifndef RIKSPROT_REPOSITORY_TAG
# $(error RIKSPROT_REPOSITORY_TAG is undefined)
# endif

# PARENT_DATA_FOLDER=$(shell dirname $(RIKSPROT_DATA_FOLDER))
METADATA_DB_NAME=riksprot_metadata.$(RIKSPROT_REPOSITORY_TAG).db

PYRIKSPROT_VERSION=main
PYRIKSSPROT_SOURCE_FOLDER=~/source/welfare-state-analytics/pyriksprot/tests/test_data/source
PYRIKSSPROT_TESTDATA_FOLDER=tests/test_data/riksdagens_protokoll/$(PYRIKSPROT_VERSION)

.ONESHELL: parlaclarin-test-data
parlaclarin-test-data:
	@rm -rf $(PYRIKSSPROT_TESTDATA_FOLDER)
	@cp -r $(PYRIKSSPROT_SOURCE_FOLDER)/parlaclarin/$(PYRIKSPROT_VERSION) $(PYRIKSSPROT_TESTDATA_FOLDER)/
	@cp -r $(PYRIKSSPROT_SOURCE_FOLDER)/tagged_frames/$(PYRIKSPROT_VERSION) $(PYRIKSSPROT_TESTDATA_FOLDER)/

parlaclarin-test-data-scp:
	@rm -rf $(PYRIKSSPROT_TESTDATA_FOLDER)
	@mkdir -p $(PYRIKSSPROT_TESTDATA_FOLDER)/corpus
	@scp -r 130.239.57.54:$(PYRIKSSPROT_SOURCE_FOLDER)/parlaclarin/$(PYRIKSPROT_VERSION)/* $(PYRIKSSPROT_TESTDATA_FOLDER)/corpus
	@mkdir -p $(PYRIKSSPROT_TESTDATA_FOLDER)/tagged_corpus
	@scp -r 130.239.57.54:$(PYRIKSSPROT_SOURCE_FOLDER)/tagged_frames/$(PYRIKSPROT_VERSION)/* $(PYRIKSSPROT_TESTDATA_FOLDER)/tagged_frames

PYRIKSPROT_TESTDATA_TAR=riksprot_sample_testdata.$(PYRIKSPROT_VERSION).tar.gz
pyriksprot-test-data:
	@rm -rf tests/test_data/riksdagens_protokoll/$(PYRIKSPROT_VERSION)
	@wget -O /tmp/$(PYRIKSPROT_TESTDATA_TAR) https://raw.githubusercontent.com/welfare-state-analytics/pyriksprot/dev/tests/test_data/dists/$(PYRIKSPROT_TESTDATA_TAR)
	@tar -C tests/test_data/riksdagens_protokoll --strip 3 -xvf /tmp/$(PYRIKSPROT_TESTDATA_TAR) tests/test_data/source/$(PYRIKSPROT_VERSION)
	@rm -f /tmp/$(PYRIKSPROT_TESTDATA_TAR)
	@cp -f tests/test_data/riksdagens_protokoll/$(PYRIKSPROT_VERSION)/tagged_frames/*.db tests/test_data/riksdagens_protokoll/