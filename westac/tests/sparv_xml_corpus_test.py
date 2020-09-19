import pytest # pylint: disable=unused-import

import westac.corpus.iterators.sparv_xml_corpus_source_reader as sparv_reader

SPARV_ZIPPED_XML_EXPORT_FILENAME = './westac/tests/test_data/sparv_zipped_xml_export.zip'
SPARV_ZIPPED_XML_EXPORT_V3_FILENAME = './westac/tests/test_data/sou_test_sparv3_xml.zip'

DEFAULT_OPTS = dict(
    pos_includes='',
    lemmatize=True,
    chunk_size=None,
    xslt_filename=None,
    delimiter="|",
    append_pos="",
    pos_excludes="|MAD|MID|PAD|"
)

def preprocess_sparv_corpus(corpus_name, opts):

    opts = { **DEFAULT_OPTS, **opts }

    reader = sparv_reader.Sparv3XmlCorpusSourceReader(corpus_name, **opts)

    for _, (_, tokens) in enumerate(reader.get_iterator()):

        assert len(list(tokens)) > 0

    # id2word = gensim.corpora.Dictionary(terms)
    # corpus = [ id2word.doc2bow(tokens) for tokens in terms ]

def test_corpus_when_source_is_sparv3_succeeds():

    opts = dict(pos_includes='|NN|', lemmatize=True, chunk_size=None)

    reader = sparv_reader.Sparv3XmlCorpusSourceReader(SPARV_ZIPPED_XML_EXPORT_V3_FILENAME, **opts)

    for _, (_, tokens) in enumerate(reader.get_iterator()):

        assert len(list(tokens)) > 0

    # id2word = gensim.corpora.Dictionary(terms)
    # corpus = [ id2word.doc2bow(tokens) for tokens in terms ]
