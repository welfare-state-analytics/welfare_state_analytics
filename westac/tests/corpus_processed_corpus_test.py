
import unittest

import westac.corpus.text_corpus as text_corpus
from westac.tests.utils import create_text_files_reader

class Test_ProcessedCorpus(unittest.TestCase):

    def setUp(self):
        pass

    def create_reader(self):
        meta_extract = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
        reader = create_text_files_reader(meta_extract=meta_extract, compress_whitespaces=True, dehyphen=True)
        return reader

    def test_next_document_when_isalnum_is_true_returns_all_tokens(self):
        reader = self.create_reader()
        kwargs = dict(isalnum=False, to_lower=False, deacc=False, min_len=1, max_len=None, numerals=True, only_alphabetic=False)
        corpus = text_corpus.ProcessedCorpus(reader, **kwargs)
        _, tokens = next(corpus.documents())
        expected = ["Tre", "svarta", "ekar", "ur", "snön", ".",
                    "Så", "grova", ",", "men", "fingerfärdiga", ".",
                    "Ur", "deras", "väldiga", "flaskor",
                    "ska", "grönskan", "skumma", "i", "vår", "."]
        self.assertEqual(expected, tokens)

    def test_next_document_when_isalnum_true_skips_deliminators(self):
        reader = self.create_reader()
        kwargs = dict(isalnum=True, to_lower=False, deacc=False, min_len=1, max_len=None, numerals=True)
        corpus = text_corpus.ProcessedCorpus(reader, **kwargs)
        _, tokens = next(corpus.documents())
        expected = ["Tre", "svarta", "ekar", "ur", "snön",
                    "Så", "grova", "men", "fingerfärdiga",
                    "Ur", "deras", "väldiga", "flaskor",
                    "ska", "grönskan", "skumma", "i", "vår"]
        self.assertEqual(expected, tokens)

    def test_next_document_when_to_lower_is_true_returns_all_lowercase(self):
        reader = self.create_reader()
        kwargs = dict(isalnum=True, to_lower=True, deacc=False, min_len=1, max_len=None, numerals=True)
        corpus = text_corpus.ProcessedCorpus(reader, **kwargs)
        _, tokens = next(corpus.documents())
        expected = ["tre", "svarta", "ekar", "ur", "snön",
                    "så", "grova", "men", "fingerfärdiga",
                    "ur", "deras", "väldiga", "flaskor",
                    "ska", "grönskan", "skumma", "i", "vår"]
        self.assertEqual(expected, tokens)

    def test_next_document_when_min_len_is_two_returns_single_char_words_filtered_out(self):
        reader = self.create_reader()
        kwargs = dict(isalnum=True, to_lower=True, deacc=False, min_len=2, max_len=None, numerals=True)
        corpus = text_corpus.ProcessedCorpus(reader, **kwargs)
        _, tokens = next(corpus.documents())
        expected = ["tre", "svarta", "ekar", "ur", "snön",
                    "så", "grova", "men", "fingerfärdiga",
                    "ur", "deras", "väldiga", "flaskor",
                    "ska", "grönskan", "skumma", "vår"]
        self.assertEqual(expected, tokens)

    def test_next_document_when_max_len_is_six_returns_filter_out_longer_words(self):
        reader = self.create_reader()
        kwargs = dict(isalnum=True, to_lower=True, deacc=False, min_len=2, max_len=6, numerals=True)
        corpus = text_corpus.ProcessedCorpus(reader, **kwargs)
        _, tokens = next(corpus.documents())
        expected = ["tre", "svarta", "ekar", "ur", "snön",
                    "så", "grova", "men",
                    "ur", "deras",
                    "ska", "skumma", "vår"]
        self.assertEqual(expected, tokens)

    def test_get_index_when_extract_passed_returns_expected_count(self):
        reader = self.create_reader()
        kwargs = dict(isalnum=False, to_lower=False, deacc=False, min_len=2, max_len=None, numerals=True)
        corpus = text_corpus.ProcessedCorpus(reader, **kwargs)
        result = corpus.get_metadata()
        self.assertEqual(5, len(result))

    def test_n_tokens_when_exhausted_and_isalnum_min_len_two_returns_expected_count(self):
        reader = self.create_reader()
        corpus = text_corpus.ProcessedCorpus(reader, isalnum=True, min_len=2)
        r_tokens = {}
        for filename, tokens in corpus.documents():
            r_tokens[filename] = len(tokens)
        n_tokens = corpus.n_raw_tokens
        n_expected = {
            'dikt_2019_01_test.txt': 18,
            'dikt_2019_02_test.txt': 14,
            'dikt_2019_03_test.txt': 24,
            'dikt_2020_01_test.txt': 42,
            'dikt_2020_02_test.txt': 18
        }
        p_tokens = corpus.n_tokens
        p_expected = {
            'dikt_2019_01_test.txt': 17,
            'dikt_2019_02_test.txt': 13,
            'dikt_2019_03_test.txt': 21,
            'dikt_2020_01_test.txt': 42,
            'dikt_2020_02_test.txt': 18
        }
        self.assertEqual(n_expected, n_tokens)
        self.assertEqual(p_expected, p_tokens)
        self.assertEqual(p_expected, r_tokens)
