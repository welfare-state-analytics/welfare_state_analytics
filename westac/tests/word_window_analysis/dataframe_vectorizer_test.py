# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
import unittest
import pandas as pd
import numpy as np
import scipy
import types

from westac.corpus import corpus_vectorizer
import westac.corpus.processed_text_corpus as corpora
from westac.corpus.iterators import dataframe_text_reader


class Test_DataFrameVectorize(unittest.TestCase):

    def setUp(self):
        pass

    def create_test_dataframe(self):
        data = [
            (2000, 'A B C'),
            (2000, 'B C D'),
            (2001, 'C B'),
            (2003, 'A B F'),
            (2003, 'E B'),
            (2003, 'F E E')
        ]
        df = pd.DataFrame(data, columns=['year', 'txt'])
        return df

    def create_corpus(self):
        df = self.create_test_dataframe()
        reader = dataframe_text_reader.DataFrameTextReader(df)
        kwargs = dict(isalnum=False, to_lower=False, deacc=False, min_len=0, max_len=None, numerals=False)
        corpus = corpora.ProcessedTextCorpus(reader, **kwargs)
        return corpus

    def test_corpus_text_stream(self):
        df = self.create_test_dataframe()
        reader = dataframe_text_reader.DataFrameTextReader(df)
        corpus = corpora.BaseCorpus(reader)
        result = [ x for x in corpus.documents()]
        expected = [('0', 'A B C'), ('1', 'B C D'), ('2', 'C B'), ('3', 'A B F'), ('4', 'E B'), ('5', 'F E E')]
        self.assertEqual(expected, result)

    def test_corpus_token_stream(self):
        df = self.create_test_dataframe()
        reader = dataframe_text_reader.DataFrameTextReader(df)
        corpus = corpora.TokenizedCorpus(reader)
        result = [ x for x in corpus.documents()]
        expected = [('0', ['A', 'B', 'C']), ('1', ['B', 'C', 'D']), ('2', ['C', 'B']), ('3', ['A', 'B', 'F']), ('4', ['E', 'B']), ('5', ['F', 'E', 'E'])]
        self.assertEqual(expected, result)

    def test_processed_corpus_token_stream(self):
        df = self.create_test_dataframe()
        reader = dataframe_text_reader.DataFrameTextReader(df)
        kwargs = dict(isalnum=False, to_lower=False, deacc=False, min_len=0, max_len=None, numerals=False)
        corpus = corpora.ProcessedTextCorpus(reader, **kwargs)
        result = [ x for x in corpus.documents()]
        expected = [('0', ['A', 'B', 'C']), ('1', ['B', 'C', 'D']), ('2', ['C', 'B']), ('3', ['A', 'B', 'F']), ('4', ['E', 'B']), ('5', ['F', 'E', 'E'])]
        self.assertEqual(expected, result)

    def create_simple_test_corpus(self, **kwargs):
        data = [
            (2000, 'Detta är en mening med 14 token, 3 siffror och 2 symboler.'),
            (2000, 'Är det i denna mening en mening?'),
        ]
        df = pd.DataFrame(data, columns=['year', 'txt'])
        reader = dataframe_text_reader.DataFrameTextReader(df)
        corpus = corpora.ProcessedTextCorpus(reader, **kwargs)
        return corpus

    def test_tokenized_document_where_symbols_are_filtered_out(self):
        corpus = self.create_simple_test_corpus(symbols=False, isalnum=True, to_lower=False, deacc=False, min_len=0, max_len=None, numerals=True, stopwords=None, only_alphabetic=False)
        result = [ x for x in corpus.documents()]
        expected = [
            ('0',  [ 'Detta', 'är', 'en', 'mening', 'med', '14', 'token', '3', 'siffror', 'och', '2', 'symboler' ]),
            ('1',  [ 'Är', 'det', 'i', 'denna', 'mening', 'en', 'mening' ]),
        ]
        self.assertEqual(expected, result)

    def test_tokenized_document_where_symbols_and_numerals_are_filtered_out(self):
        corpus = self.create_simple_test_corpus(symbols=False, isalnum=True, to_lower=False, deacc=False, min_len=0, max_len=None, numerals=False, stopwords=None)
        result = [ x for x in corpus.documents()]
        expected = [
            ('0',  [ 'Detta', 'är', 'en', 'mening', 'med', 'token', 'siffror', 'och', 'symboler' ]),
            ('1',  [ 'Är', 'det', 'i', 'denna', 'mening', 'en', 'mening' ]),
        ]
        self.assertEqual(expected, result)

    def test_tokenized_document_in_lowercase_where_symbols_and_numerals_and_one_letter_words_are_filtered_out(self):
        corpus = self.create_simple_test_corpus(symbols=False, isalnum=True, to_lower=True, deacc=False, min_len=2, max_len=None, numerals=False, stopwords=None)
        result = [ x for x in corpus.documents()]
        expected = [
            ('0',  [ 'detta', 'är', 'en', 'mening', 'med', 'token', 'siffror', 'och', 'symboler' ]),
            ('1',  [ 'är', 'det', 'denna', 'mening', 'en', 'mening' ]),
        ]
        self.assertEqual(expected, result)

    def test_tokenized_document_in_lowercase_where_symbols_and_numerals_and_one_letter_words_and_stopwords_are_filtered_out(self):
        stopwords = { 'är', 'en', 'med', 'och', 'det', 'detta', 'denna' }
        corpus = self.create_simple_test_corpus(symbols=False, isalnum=True, to_lower=True, deacc=False, min_len=2, max_len=None, numerals=False, stopwords=stopwords)
        result = [ x for x in corpus.documents()]
        expected = [
            ('0',  [ 'mening', 'token', 'siffror', 'symboler' ]),
            ('1',  [ 'mening', 'mening' ]),
        ]
        self.assertEqual(expected, result)

    def test_fit_transform_gives_document_term_matrix(self):
        reader = dataframe_text_reader.DataFrameTextReader(self.create_test_dataframe())
        kwargs = dict(to_lower=False, deacc=False, min_len=1, max_len=None, numerals=False)
        corpus = corpora.ProcessedTextCorpus(reader, isalnum=False, **kwargs)
        vectorizer = corpus_vectorizer.CorpusVectorizer(lowercase=False)
        v_corpus = vectorizer.fit_transform(corpus)
        expected = np.asarray([
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 2, 1]
        ])
        self.assertTrue((expected == v_corpus.bag_term_matrix).all())
        results = v_corpus.token2id
        expected = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 5, 'E': 4 }
        self.assertEqual(expected, results)

    def test_AxAt_of_document_term_matrix_gives_term_term_matrix(self):

        # Arrange
        reader = dataframe_text_reader.DataFrameTextReader(self.create_test_dataframe())
        kwargs = dict(to_lower=False, deacc=False, min_len=1, max_len=None, numerals=False)
        corpus = corpora.ProcessedTextCorpus(reader, isalnum=False, **kwargs)
        vectorizer = corpus_vectorizer.CorpusVectorizer(lowercase=False)
        v_corpus = vectorizer.fit_transform(corpus)

        # Act
        term_term_matrix = np.dot(v_corpus.bag_term_matrix.T, v_corpus.bag_term_matrix)

        # Assert
        expected = np.asarray([
             [2, 2, 1, 0, 0, 1],
             [2, 5, 3, 1, 1, 1],
             [1, 3, 3, 1, 0, 0],
             [0, 1, 1, 1, 0, 0],
             [0, 1, 0, 0, 5, 2],
             [1, 1, 0, 0, 2, 2]
        ])
        self.assertTrue((expected == term_term_matrix).all())

        term_term_matrix = scipy.sparse.triu(term_term_matrix, 1)
        expected = np.asarray([
             [0, 2, 1, 0, 0, 1],
             [0, 0, 3, 1, 1, 1],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 2],
             [0, 0, 0, 0, 0, 0]
        ])

        #print(term_term_matrix.todense())
        #print(term_term_matrix)
        coo = term_term_matrix
        id2token = { i: t for t,i in v_corpus.token2id.items()}
        cdf = pd.DataFrame({
            'w1_id': coo.row,
            'w2_id': coo.col,
            'value': coo.data
        })[['w1_id', 'w2_id', 'value']].sort_values(['w1_id', 'w2_id'])\
            .reset_index(drop=True)
        cdf['w1'] = cdf.w1_id.apply(lambda x: id2token[x])
        cdf['w2'] = cdf.w2_id.apply(lambda x: id2token[x])
        print(cdf[['w1', 'w2', 'value']])

    def test_tokenized_document_token_counts_is_empty_if_enumerable_not_exhausted(self):
        corpus = self.create_simple_test_corpus(symbols=False, isalnum=True, to_lower=True, deacc=False, min_len=0, max_len=None, numerals=True, stopwords=None)
        n_tokens = corpus.n_tokens
        n_raw_tokens = corpus.n_raw_tokens
        self.assertEqual({}, n_tokens)
        self.assertEqual({}, n_raw_tokens)

    def test_tokenized_document_token_counts_is_not_empty_if_enumerable_is_exhausted(self):
        # Note: Symbols are always removed by reader - hence "symbols" filter has not effect
        corpus = self.create_simple_test_corpus(symbols=False, isalnum=True, to_lower=True, deacc=False, min_len=0, max_len=None, numerals=True, stopwords=None)
        for _ in corpus.documents():
            pass
        n_tokens = corpus.n_tokens
        n_raw_tokens = corpus.n_raw_tokens
        self.assertEqual({'0': 9, '1': 7}, n_tokens)
        self.assertEqual({'0': 12, '1': 7}, n_raw_tokens)

