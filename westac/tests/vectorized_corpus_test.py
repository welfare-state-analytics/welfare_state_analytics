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
        self.assertEqual(dumped_v_corpus.token2id, loaded_v_corpus.token2id)
        #self.assertEqual(dumped_v_corpus.X, loaded_v_corpus.X)

    def test_group_by_year_aggregates_doc_term_matrix_to_year_term_matrix(self):
        v_corpus = self.create_vectorized_corpus()
        g_corpus = v_corpus.group_by_year()
        expected_ytm = [
            [4, 3, 7, 1],
            [6, 7, 4, 2]
        ]
        self.assertTrue((expected_ytm == g_corpus.doc_term_matrix).all())

    def test_collapse_to_category_aggregates_doc_term_matrix_to_category_term_matrix(self):
        """ A more generic version of group_by_year (not used for now) """
        v_corpus = self.create_vectorized_corpus()
        Y, _ = v_corpus.collapse_by_category('year')
        expected_ytm = [
            [4, 3, 7, 1],
            [6, 7, 4, 2]
        ]
        self.assertTrue((expected_ytm == Y).all())

    def test_normalize_with_default_arguments_returns_matrix_normalized_by_l1_norm_for_each_row(self):
        doc_term_matrix = np.array([
            [4, 3, 7, 1],
            [6, 7, 4, 2]
        ])
        token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3 }
        df = pd.DataFrame({'year': [ 2013,2014 ]})
        v_corpus = vectorized_corpus.VectorizedCorpus(doc_term_matrix, token2id, df)
        n_corpus = v_corpus.normalize()
        E = np.array([
            [4, 3, 7, 1],
            [6, 7, 4, 2]
        ]) / (np.array([[15,19]]).T)
        self.assertTrue((E == n_corpus.doc_term_matrix).all())

    def create_slice_by_n_count_test_corpus(self):
        doc_term_matrix = np.array([
            [1, 1, 4, 1],
            [0, 2, 3, 0],
            [0, 3, 2, 0],
            [0, 4, 1, 3],
            [2, 0, 1, 1]
        ])
        token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3 }
        df = pd.DataFrame({'year': [ 2013, 2013, 2014, 2014, 2014 ]})
        return vectorized_corpus.VectorizedCorpus(doc_term_matrix, token2id, df)

    def test_slice_by_n_count_when_exists_tokens_below_count_returns_filtered_corpus(self):

        v_corpus = self.create_slice_by_n_count_test_corpus()

        # Act
        t_corpus = v_corpus.slice_by_n_count(6)

        # Assert
        expected_doc_term_matrix = np.array([
            [1, 4],
            [2, 3],
            [3, 2],
            [4, 1],
            [0, 1]
        ])

        self.assertEqual({'b': 0, 'c': 1 }, t_corpus.token2id)
        self.assertEqual({'b': 10, 'c': 11 }, t_corpus.word_counts)
        self.assertTrue((expected_doc_term_matrix == t_corpus.doc_term_matrix).all())

    def test_slice_by_n_count_when_all_below_below_n_count_returns_empty_corpus(self):

        v_corpus = self.create_slice_by_n_count_test_corpus()

        # Act
        t_corpus = v_corpus.slice_by_n_count(20)

        # Assert

        self.assertEqual({}, t_corpus.token2id)
        self.assertEqual({}, t_corpus.word_counts)
        self.assertTrue((np.empty((5,0)) == t_corpus.doc_term_matrix).all())

    def test_slice_by_n_count_when_all_tokens_above_n_count_returns_same_corpus(self):

        v_corpus = self.create_slice_by_n_count_test_corpus()

        # Act
        t_corpus = v_corpus.slice_by_n_count(1)

        # Assert
        self.assertEqual(v_corpus.token2id, t_corpus.token2id)
        self.assertEqual(v_corpus.word_counts, t_corpus.word_counts)
        self.assertTrue((v_corpus.doc_term_matrix == t_corpus.doc_term_matrix).all())

    def test_id2token_is_reversed_token2id(self):
        v_corpus = self.create_vectorized_corpus()
        id2token = { 0: 'a', 1: 'b', 2: 'c', 3: 'd' }
        self.assertEqual(id2token, v_corpus.id2token)