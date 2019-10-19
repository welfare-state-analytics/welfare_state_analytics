import os
import unittest
import types
import pandas as pd
import numpy as np

from westac.common import corpus_vectorizer
from westac.common import text_corpus
from westac.common import vectorized_corpus

from westac.tests.utils  import create_text_files_reader

flatten = lambda l: [ x for ws in l for x in ws]

class Test_VectorizedCorpus(unittest.TestCase):

    def setUp(self):
        pass

    def create_reader(self):
        meta_extract = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
        reader = create_text_files_reader(meta_extract=meta_extract, compress_whitespaces=True, dehyphen=True)
        return reader

    def create_corpus(self):
        reader = self.create_reader()
        kwargs = dict(isalnum=True, to_lower=True, deacc=False, min_len=2, max_len=None, numerals=False)
        corpus = text_corpus.ProcessedCorpus(reader, **kwargs)
        return corpus

    def create_vectorized_corpus(self):
        doc_term_matrix = np.array([
            [2, 1, 4, 1],
            [2, 2, 3, 0],
            [2, 3, 2, 0],
            [2, 4, 1, 1],
            [2, 0, 1, 1]
        ])
        token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3 }
        df = pd.DataFrame({'year': [ 2013, 2013, 2014, 2014, 2014 ]})
        v_corpus = vectorized_corpus.VectorizedCorpus(doc_term_matrix, token2id, df)
        return v_corpus

    def test_load_of_previously_dumped_v_corpus_has_same_values_as_dumped_v_corpus(self):

        # Arrange
        corpus = self.create_corpus()
        vectorizer = corpus_vectorizer.CorpusVectorizer()
        dumped_v_corpus = vectorizer.fit_transform(corpus)

        dumped_v_corpus.dump('dump_test', folder='./westac/tests/output')

        # Act
        loaded_v_corpus = vectorized_corpus.VectorizedCorpus.load('dump_test', folder='./westac/tests/output')

        # Assert
        self.assertEqual(dumped_v_corpus.word_counts, loaded_v_corpus.word_counts)
        self.assertEqual(dumped_v_corpus.document_index.to_dict(), loaded_v_corpus.document_index.to_dict())
        self.assertEqual(dumped_v_corpus.vocabulary, loaded_v_corpus.vocabulary)
        #self.assertEqual(dumped_v_corpus.X, loaded_v_corpus.X)

    def test_collapse_to_year_aggregates_doc_term_matrix_to_year_term_matrix(self):
        v_corpus = self.create_vectorized_corpus()
        ytm = v_corpus.collapse_to_year()
        expected_ytm = [
            [4, 3, 7, 1],
            [6, 7, 4, 2]
        ]
        self.assertTrue((expected_ytm == ytm).all())

    def test_collapse_to_category_aggregates_doc_term_matrix_to_category_term_matrix(self):
        """ A more generic version of collapse_to_year (not used for now) """
        v_corpus = self.create_vectorized_corpus()
        Y, _ = v_corpus.collapse_by_category('year')
        expected_ytm = [
            [4, 3, 7, 1],
            [6, 7, 4, 2]
        ]
        self.assertTrue((expected_ytm == Y).all())

    def test_normalize_with_default_arguments_returns_matrix_normalized_by_l1_norm_for_each_row(self):
        v_corpus = self.create_vectorized_corpus()
        X = np.array([
            [4, 3, 7, 1],
            [6, 7, 4, 2]
        ], dtype=np.float)
        n_corpus = v_corpus.normalize(X)
        E = np.array([
            [4, 3, 7, 1],
            [6, 7, 4, 2]
        ]) / (np.array([[15,19]]).T)
        self.assertTrue((E == n_corpus.doc_term_matrix).all())


    def test_tokens_above_threshold_returns_tokens_having_token_count_ge_to_threshold(self):
        v_corpus = self.create_vectorized_corpus()
        tokens = v_corpus.tokens_above_threshold(4)
        expected_tokens = {'a': 10, 'b': 10, 'c': 11 }
        self.assertEqual(expected_tokens, tokens)

    def test_tokens_above_threshold_returns_ids_of_tokens_having_token_count_ge_to_threshold(self):
        v_corpus = self.create_vectorized_corpus()
        ids = v_corpus.token_ids_above_threshold(4)
        expected_ids = [
            v_corpus.vocabulary['a'],
            v_corpus.vocabulary['b'],
            v_corpus.vocabulary['c']
        ]
        self.assertEqual(expected_ids, ids)