
include .env

include ./Makefile.dev

# These need to be defined
ifndef PYRIKSPROT_VERSION
$(error PYRIKSPROT_VERSION is undefined)
endif

# ifndef RIKSPROT_REPOSITORY_TAG
# $(error RIKSPROT_REPOSITORY_TAG is undefined)
# endif
RIKSPROT_XML_PATTERN=/data/westac/riksdagen_corpus_data/riksdagen-corpus/corpus/protocols/*/*.xml
riksprot-speaker-notes:
	@xmlstarlet sel -N x="http://www.tei-c.org/ns/1.0" -t -m "//x:note[@type='speaker']" \
		-v "concat(@type,';',@n,';','\"',normalize-space(translate(text(),';','')),'\"')" -nl $(RIKSPROT_XML_PATTERN)

PYRIKSPROT_VERSION=main
PYRIKSPROT_TESTDATA_TAR=riksprot_sample_testdata.$(PYRIKSPROT_VERSION).tar.gz
riksprot-test-data:
	@mv -f tests/test_data/riksprot/$(PYRIKSPROT_VERSION)/dtm_test.5files /tmp
	@mv -f tests/test_data/riksprot/$(PYRIKSPROT_VERSION)/tm_test.5files.mallet /tmp
	@rm -rf tests/test_data/riksprot/$(PYRIKSPROT_VERSION)
	@wget -O /tmp/$(PYRIKSPROT_TESTDATA_TAR) https://raw.githubusercontent.com/welfare-state-analytics/pyriksprot/dev/tests/test_data/dists/$(PYRIKSPROT_TESTDATA_TAR)
	@tar -C tests/test_data/riksprot --strip 3 -xvf /tmp/$(PYRIKSPROT_TESTDATA_TAR) tests/test_data/source/$(PYRIKSPROT_VERSION)
	@rm -f /tmp/$(PYRIKSPROT_TESTDATA_TAR)
	@cp -f tests/test_data/riksprot/corpus.id.yml tests/test_data/riksprot/$(PYRIKSPROT_VERSION)/corpus.yml
	@echo "NOTE: test tm model and dtm model must also be updated"
	@mv -f /tmp/dtm_test.5files tests/test_data/riksprot/$(PYRIKSPROT_VERSION)/
	@mv -f /tmp/tm_test.5files.mallet tests/test_data/riksprot/$(PYRIKSPROT_VERSION)/

riksprot-test-topic-model:
	@echo "FIXME: this is not complete, TM must be created by penelope and copied to tests/test_data/riksprot/$(PYRIKSPROT_VERSION)"
	@echo "Create model using Penelope:"
	@echo     PYTHONPATH=. poetry run python penelope/scripts/tm/train_id.py --options-filename opts/opts_tm_mallet_riksprot.test.yml
	@scp -r roger@130.239.57.54:/home/roger/source/penelope/data/tm_test.5files.mallet tests/test_data/riksprot/$(PYRIKSPROT_VERSION)


riksprot-test-dtm:
	@echo "FIXME: this is not complete, DTM must be created by penelope and copied to tests/test_data/riksprot/$(PYRIKSPROT_VERSION)"
	@echo "Create model using Penelope:"
	@echo     PYTHONPATH=. poetry run python penelope/scripts/dtm/vectorize_id.py --options-filename opts/opts_vectorize_id_riksprot_test.yml
	@echo "NOTE: Make sure vectorize uses the document_index (specified in corpus.yml) found in tagged speech corpus (i.e. feather folder)"
	@scp -r roger@130.239.57.54:/home/roger/source/penelope/data/dtm_test.5files tests/test_data/riksprot/$(PYRIKSPROT_VERSION)


dummy-test-id-config:
	echo "scp -r roger@130.239.57.54:/home/roger/source/welfare-state-analytics/pyriksprot/tests/test_data/source/corpus.yml corpus.yml"
