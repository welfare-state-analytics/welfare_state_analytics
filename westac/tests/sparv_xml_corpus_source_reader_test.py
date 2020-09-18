import os
import pytest
from westac.corpus import utility

import westac.corpus.iterators.sparv_xml_corpus_source_reader as sparv_reader
import westac.common.zip_utility as zip_utility
import westac.corpus.utility as utility

SPARV_XML_EXPORT_FILENAME = './westac/tests/test_data/sparv_xml_export.xml'
SPARV_XML_EXPORT_FILENAME_SMALL = './westac/tests/test_data/sparv_xml_export_small.xml'
SPARV_ZIPPED_XML_EXPORT_FILENAME = './westac/tests/test_data/sparv_zipped_xml_export.zip'

def sparv_xml_test_file():
    with open(SPARV_XML_EXPORT_FILENAME, "rb") as fp:
        return fp.read()

DEFAULT_OPTS = dict(
    pos_includes='',
    lemmatize=True,
    chunk_size=None,
    xslt_filename=None,
    delimiter="|",
    append_pos="",
    pos_excludes="|MAD|MID|PAD|"
)
def test_reader_when_no_transforms_returns_source_tokens():

    expected = ['Rödräven', 'är', 'ett', 'hunddjur', 'som', 'har', 'en', 'mycket', 'vidsträckt', 'utbredning', 'över', 'norra', 'halvklotet', '.']
    expected_name = "sparv_xml_export_small.txt"

    opts = dict(pos_includes='', lemmatize=False, chunk_size=None, pos_excludes="")

    reader = sparv_reader.SparvXmlCorpusSourceReader(SPARV_XML_EXPORT_FILENAME_SMALL, **opts)

    document_name, tokens = next(iter(reader))

    assert expected == list(tokens)
    assert expected_name == document_name

def test_reader_when_lemmatized_returns_tokens_in_baseform():

    expected = ['rödräv', 'vara', 'en', 'hunddjur', 'som', 'ha', 'en', 'mycken', 'vidsträckt', 'utbredning', 'över', 'norra', 'halvklot', '.']
    expected_name = "sparv_xml_export_small.txt"

    opts = dict(pos_includes='', lemmatize=True, chunk_size=None, pos_excludes="")

    reader = sparv_reader.SparvXmlCorpusSourceReader(SPARV_XML_EXPORT_FILENAME_SMALL, **opts)

    document_name, tokens = next(iter(reader))

    assert expected == list(tokens)
    assert expected_name == document_name

def test_reader_when_ignore_puncts_returns_filter_outs_puncts():

    expected = ['rödräv', 'vara', 'en', 'hunddjur', 'som', 'ha', 'en', 'mycken', 'vidsträckt', 'utbredning', 'över', 'norra', 'halvklot' ]
    expected_name = "sparv_xml_export_small.txt"

    opts = dict(pos_includes='', lemmatize=True, chunk_size=None, pos_excludes="|MAD|MID|PAD|")

    reader = sparv_reader.SparvXmlCorpusSourceReader(SPARV_XML_EXPORT_FILENAME_SMALL, **opts)

    document_name, tokens = next(iter(reader))

    assert expected == tokens
    assert expected_name == document_name

def test_reader_when_only_nouns_ignore_puncts_returns_filter_outs_puncts():

    expected = ['rödräv', 'hunddjur', 'utbredning', 'halvklot' ]
    expected_name = "sparv_xml_export_small.txt"

    opts = dict(pos_includes='|NN|', lemmatize=True, chunk_size=None)

    reader = sparv_reader.SparvXmlCorpusSourceReader(SPARV_XML_EXPORT_FILENAME_SMALL, **opts)

    document_name, tokens = next(iter(reader))

    assert expected == list(tokens)
    assert expected_name == document_name

def test_reader_when_ignore_puncts_returns_filter_outs_puncts():

    expected_documents = [ ['rödräv', 'hunddjur' ], [ 'utbredning', 'halvklot' ] ]
    expected_names = [ "sparv_xml_export_small_001.txt", "sparv_xml_export_small_002.txt"]

    opts = dict(pos_includes='|NN|', lemmatize=True, chunk_size=2)

    reader = sparv_reader.SparvXmlCorpusSourceReader(SPARV_XML_EXPORT_FILENAME_SMALL, **opts)

    for i, (document_name, tokens) in enumerate(reader):

        assert expected_documents[i] == list(tokens)
        assert expected_names[i] == document_name

def test_reader_when_source_is_zipped_archive_succeeds():

    expected_documents = [ ['rödräv', 'hunddjur', 'utbredning', 'halvklot' ], [ 'fjällräv', 'fjällvärld', 'liv', 'fjällräv', 'vinter', 'men', 'variant', 'år' ] ]
    expected_names = [ "document_001.txt", "document_002.txt"]

    opts = dict(pos_includes='|NN|', lemmatize=True, chunk_size=None)

    reader = sparv_reader.SparvXmlCorpusSourceReader(SPARV_ZIPPED_XML_EXPORT_FILENAME, **opts)

    for i, (document_name, tokens) in enumerate(reader):

        assert expected_documents[i] == list(tokens)
        assert expected_names[i] == document_name

def test_reader_store_result():

    expected_documents = [ ['rödräv', 'hunddjur', 'utbredning', 'halvklot' ], [ 'fjällräv', 'fjällvärld', 'liv', 'fjällräv', 'vinter', 'men', 'variant', 'år' ] ]
    expected_names = [ "document_001.txt", "document_002.txt"]

    target_filename = './westac/tests/sparv_extract_and_store.zip'

    opts = dict(pos_includes='|NN|', lemmatize=True, chunk_size=None)

    sparv_reader.sparv_extract_and_store(SPARV_ZIPPED_XML_EXPORT_FILENAME, target_filename, **opts)

    for i in range(0, len(expected_names)):

        content = zip_utility.read_file(target_filename, expected_names[i], as_binary=False)

        assert ' '.join(expected_documents[i]) == content

def test_reader_when_source_is_sparv3_succeeds():

    sparv_zipped_xml_export_v3_filename = './westac/tests/test_data/sou_test_sparv3_xml.zip'

    opts = dict(pos_includes='|NN|', lemmatize=True, chunk_size=None) #, xslt_filename=xslt_filename)

    reader = sparv_reader.Sparv3XmlCorpusSourceReader(sparv_zipped_xml_export_v3_filename, **opts)

    for i, (document_name, tokens) in enumerate(reader):

        assert len(list(tokens)) > 0

def test_sparv_extract_and_store_when_only_nouns_and_source_is_sparv3_succeeds():

    os.makedirs('./westac/tests/output', exist_ok=True)
    opts = {
        'transforms': [
            utility.remove_stopwords('swedish', extra_stopwords={'<text>'}),
            utility.min_chars_filter(2),
            utility.lower_transform()
         ],
        'pos_includes': '|NN|',
        'lemmatize': False,
        'chunk_size': None,
        'version': 3
    }

    source_filename = './westac/tests/test_data/sou_test_sparv3_xml.zip'
    target_filename = './westac/tests/output/sou_test_sparv3_extracted_txt.zip'

    sparv_reader.sparv_extract_and_store(source_filename, target_filename, **opts)

    expected_document_start = \
        "utredningar justitiedepartementet förslag utlänningslag angående om- händertagande förläggning års gere ide to lm \rstatens utredningar förteckning betänkande förslag utlänningslag lag omhändertagande utlänning anstalt förläggning tryckort tryckorten bokstäverna fetstil begynnelse- bokstäverna departement",

    test_filename = "sou_1945_1.txt"

    content = zip_utility.read_file(target_filename, test_filename, as_binary=False)

    assert content.startswith(expected_document_start)

