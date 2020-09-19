import types
import unittest

import westac.corpus.processed_text_corpus as corpora
from westac.tests.utils import create_simple_text_reader


class Test_CorpusTextStream(unittest.TestCase):

    def test_next_document_when_new_corpus_returns_document(self):
        reader = create_simple_text_reader(compress_whitespaces=True, dehyphen=True)
        corpus = corpora.BaseCorpus(reader)
        result = next(corpus.documents())
        expected = "Tre svarta ekar ur snön. " + \
                   "Så grova, men fingerfärdiga. " + \
                   "Ur deras väldiga flaskor " + \
                   "ska grönskan skumma i vår."
        self.assertEqual(expected, result[1])

    def test_get_index_when_extract_passed_returns_metadata(self):
        meta_extract = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
        reader = create_simple_text_reader(meta_extract=meta_extract, compress_whitespaces=True, dehyphen=True)
        corpus = corpora.BaseCorpus(reader)
        result = corpus.get_metadata()
        expected = [
            types.SimpleNamespace(filename='dikt_2019_01_test.txt', serial_no=1, year=2019),
            types.SimpleNamespace(filename='dikt_2019_02_test.txt', serial_no=2, year=2019),
            types.SimpleNamespace(filename='dikt_2019_03_test.txt', serial_no=3, year=2019),
            types.SimpleNamespace(filename='dikt_2020_01_test.txt', serial_no=1, year=2020),
            types.SimpleNamespace(filename='dikt_2020_02_test.txt', serial_no=2, year=2020)
        ]
        self.assertEqual(len(expected), len(result))
        for i in range(0,len(expected)):
            self.assertEqual(expected[i], result[i])

    def test_get_index_when_no_extract_passed_returns_none(self):
        reader = create_simple_text_reader(meta_extract=None, compress_whitespaces=True, dehyphen=True)
        corpus = corpora.BaseCorpus(reader)
        result = corpus.get_metadata()
        self.assertIsNone(result)
