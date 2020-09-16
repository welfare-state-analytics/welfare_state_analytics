import pytest

import gensim
import westac.corpus.iterators.sparv_xml_corpus_source_reader as sparv_reader
import westac.common.zip_utility as zip_utility

SPARV_ZIPPED_XML_EXPORT_FILENAME = './westac/tests/test_data/sparv_zipped_xml_export.zip'
SPARV_ZIPPED_XML_EXPORT_V3_FILENAME = './westac/tests/test_data/sou_test_sparv3_xml.zip'

DEFAULT_OPTS = dict(
    postags='',
    lemmatize=True,
    chunk_size=None,
    xslt_filename=None,
    deliminator="|",
    append_pos="",
    ignores="|MAD|MID|PAD|"
)

def preprocess_sparv_corpus(corpus_name, opts):

    xslt_filename='westac/corpus/sparv/alto_xml_extract.xslt'

    opts = { **DEFAULT_OPTS, **opts, **{'xslt_filename': xslt_filename} }

    reader = sparv_reader.SparvXmlCorpusSourceReader(corpus_name, **opts)

    for i, (document_name, tokens) in enumerate(reader.get_iterator()):

        assert len(tokens) > 0

    # id2word = gensim.corpora.Dictionary(terms)
    # corpus = [ id2word.doc2bow(tokens) for tokens in terms ]

def test_corpus_when_source_is_sparv3_succeeds():

    xslt_filename='westac/corpus/sparv/alto_xml_extract.xslt'

    opts = dict(postags='|NN|', lemmatize=True, chunk_size=None, xslt_filename=xslt_filename)

    reader = sparv_reader.SparvXmlCorpusSourceReader(SPARV_ZIPPED_XML_EXPORT_V3_FILENAME, **opts)

    for i, (document_name, tokens) in enumerate(reader.get_iterator()):

        assert len(tokens) > 0

    # id2word = gensim.corpora.Dictionary(terms)
    # corpus = [ id2word.doc2bow(tokens) for tokens in terms ]

