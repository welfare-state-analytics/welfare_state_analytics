import unittest

import westac.corpus.processed_text_corpus as corpora
from westac.tests.utils import create_simple_text_reader

class Test_CorpusTokenStream(unittest.TestCase):

    def create_reader(self):
        meta_extract = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
        reader = create_simple_text_reader(meta_extract=meta_extract, compress_whitespaces=True, dehyphen=True)
        return reader

    def test_next_document_when_token_corpus_returns_tokenized_document(self):
        reader = create_simple_text_reader(meta_extract=None, compress_whitespaces=True, dehyphen=True)
        corpus = corpora.CorpusTokenStream(reader, isalnum=False)
        _, tokens = next(corpus.documents())
        expected = ["Tre", "svarta", "ekar", "ur", "snön", ".",
                    "Så", "grova", ",", "men", "fingerfärdiga", ".",
                    "Ur", "deras", "väldiga", "flaskor",
                    "ska", "grönskan", "skumma", "i", "vår", "."]
        self.assertEqual(expected, tokens)

    def test_next_document_when_isalnum_true_skips_deliminators(self):
        reader = create_simple_text_reader(meta_extract=None, compress_whitespaces=True, dehyphen=True)
        corpus = corpora.CorpusTokenStream(reader, isalnum=True)
        _, tokens = next(corpus.documents())
        expected = ["Tre", "svarta", "ekar", "ur", "snön",
                    "Så", "grova", "men", "fingerfärdiga",
                    "Ur", "deras", "väldiga", "flaskor",
                    "ska", "grönskan", "skumma", "i", "vår"]
        self.assertEqual(expected, tokens)

    def test_get_index_when_extract_passed_returns_expected_count(self):
        reader = self.create_reader()
        corpus = corpora.CorpusTokenStream(reader)
        result = corpus.get_metadata()
        self.assertEqual(5, len(result))

    def test_n_tokens_when_exhausted_iterater_returns_expected_count(self):
        reader = self.create_reader()
        corpus = corpora.CorpusTokenStream(reader, isalnum=False)
        r_n_tokens = {}
        for filename, tokens in corpus.documents():
            r_n_tokens[filename] = len(tokens)
        n_tokens = corpus.n_tokens
        expected = {
            'dikt_2019_01_test.txt': 22,
            'dikt_2019_02_test.txt': 16,
            'dikt_2019_03_test.txt': 26,
            'dikt_2020_01_test.txt': 45,
            'dikt_2020_02_test.txt': 21
        }
        self.assertEqual(expected, n_tokens)
        self.assertEqual(expected, r_n_tokens)

    def test_n_tokens_when_exhausted_and_isalnum_is_true_returns_expected_count(self):
        reader = create_simple_text_reader(meta_extract=None, compress_whitespaces=True, dehyphen=True)
        corpus = corpora.CorpusTokenStream(reader, isalnum=True)
        r_n_tokens = {}
        for filename, tokens in corpus.documents():
            r_n_tokens[filename] = len(tokens)
        n_tokens = corpus.n_tokens
        expected = {
            'dikt_2019_01_test.txt': 18,
            'dikt_2019_02_test.txt': 14,
            'dikt_2019_03_test.txt': 24,
            'dikt_2020_01_test.txt': 42,
            'dikt_2020_02_test.txt': 18
        }
        self.assertEqual(expected, n_tokens)
        self.assertEqual(expected, r_n_tokens)
