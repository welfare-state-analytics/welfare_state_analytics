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

from westac.common import corpus_vectorizer
from westac.common import utility
from westac.common import text_corpus

class Test_DfTextReader(unittest.TestCase):

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

    def create_triple_meta_dataframe(self):
        data = [
            (2000, 'AB', 'A B C'),
            (2000, 'AB', 'B C D'),
            (2001, 'AB', 'C B'),
            (2003, 'AB', 'A B F'),
            (2003, 'AB', 'E B'),
            (2003, 'F E E')
        ]
        df = pd.DataFrame(data, columns=['year', 'newspaper', 'txt'])
        return df

    def test_extract_metadata_when_sourcefile_has_year_and_newspaper(self):
        df = self.create_triple_meta_dataframe()
        df_m = df[[ x for x in list(df.columns) if x != 'txt' ]]
        df_m['filename'] = df_m.index.str
        metadata = [
            types.SimpleNamespace(**meta) for meta in df_m.to_dict(orient='records')
        ]
        self.assertEqual(len(df), len(metadata))


    def test_reader_with_all_documents(self):
        df = self.create_test_dataframe()
        reader = utility.DfTextReader(df)
        result = [ x for x in reader ]
        expected = [('0', 'A B C'), ('1', 'B C D'), ('2', 'C B'), ('3', 'A B F'), ('4', 'E B'), ('5', 'F E E')]
        self.assertEqual(expected, result)
        self.assertEqual(['0', '1', '2', '3', '4', '5'], reader.filenames)
        self.assertEqual([
                types.SimpleNamespace(filename='0', year=2000),
                types.SimpleNamespace(filename='1', year=2000),
                types.SimpleNamespace(filename='2', year=2001),
                types.SimpleNamespace(filename='3', year=2003),
                types.SimpleNamespace(filename='4', year=2003),
                types.SimpleNamespace(filename='5', year=2003)
            ], reader.metadata
        )

    def test_reader_with_given_year(self):
        df = self.create_test_dataframe()
        reader = utility.DfTextReader(df, year=2003)
        result = [x for x in reader]
        expected = [('0', 'A B F'), ('1', 'E B'), ('2', 'F E E')]
        self.assertEqual(expected, result)
        self.assertEqual(['0', '1', '2'], reader.filenames)
        self.assertEqual([
                types.SimpleNamespace(filename='0', year=2003),
                types.SimpleNamespace(filename='1', year=2003),
                types.SimpleNamespace(filename='2', year=2003)
            ], reader.metadata
        )

class Test_DfVectorize(unittest.TestCase):

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
        reader = utility.DfTextReader(df)
        kwargs = dict(isalnum=False, to_lower=False, deacc=False, min_len=0, max_len=None, numerals=False)
        corpus = text_corpus.ProcessedCorpus(reader, **kwargs)
        return corpus

    def test_corpus_text_stream(self):
        df = self.create_test_dataframe()
        reader = utility.DfTextReader(df)
        corpus = text_corpus.CorpusTextStream(reader)
        result = [ x for x in corpus.documents()]
        expected = [('0', 'A B C'), ('1', 'B C D'), ('2', 'C B'), ('3', 'A B F'), ('4', 'E B'), ('5', 'F E E')]
        self.assertEqual(expected, result)

    def test_corpus_token_stream(self):
        df = self.create_test_dataframe()
        reader = utility.DfTextReader(df)
        corpus = text_corpus.CorpusTokenStream(reader)
        result = [ x for x in corpus.documents()]
        expected = [('0', ['A', 'B', 'C']), ('1', ['B', 'C', 'D']), ('2', ['C', 'B']), ('3', ['A', 'B', 'F']), ('4', ['E', 'B']), ('5', ['F', 'E', 'E'])]
        self.assertEqual(expected, result)

    def test_processed_corpus_token_stream(self):
        df = self.create_test_dataframe()
        reader = utility.DfTextReader(df)
        kwargs = dict(isalnum=False, to_lower=False, deacc=False, min_len=0, max_len=None, numerals=False)
        corpus = text_corpus.ProcessedCorpus(reader, **kwargs)
        result = [ x for x in corpus.documents()]
        expected = [('0', ['A', 'B', 'C']), ('1', ['B', 'C', 'D']), ('2', ['C', 'B']), ('3', ['A', 'B', 'F']), ('4', ['E', 'B']), ('5', ['F', 'E', 'E'])]
        self.assertEqual(expected, result)

    def create_simple_test_corpus(self, **kwargs):
        data = [
            (2000, 'Detta är en mening med 14 token, 3 siffror och 2 symboler.'),
            (2000, 'Är det i denna mening en mening?'),
        ]
        df = pd.DataFrame(data, columns=['year', 'txt'])
        reader = utility.DfTextReader(df)
        corpus = text_corpus.ProcessedCorpus(reader, **kwargs)
        return corpus

    def test_tokenized_document_where_symbols_are_filtered_out(self):
        corpus = self.create_simple_test_corpus(symbols=False, isalnum=True, to_lower=False, deacc=False, min_len=0, max_len=None, numerals=True, stopwords=None)
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
        reader = utility.DfTextReader(self.create_test_dataframe())
        kwargs = dict(to_lower=False, deacc=False, min_len=1, max_len=None, numerals=False)
        corpus = text_corpus.ProcessedCorpus(reader, isalnum=False, **kwargs)
        vectorizer = corpus_vectorizer.CorpusVectorizer(lowercase=False)
        vectorizer.fit_transform(corpus)
        expected = np.asarray([
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 2, 1]
        ])
        self.assertTrue((expected == vectorizer.X).all())
        results = vectorizer.vocabulary
        expected = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 5, 'E': 4 }
        self.assertEqual(expected, results)

    def test_AxAt_of_document_term_matrix_gives_term_term_matrix(self):

        # Arrange
        reader = utility.DfTextReader(self.create_test_dataframe())
        kwargs = dict(to_lower=False, deacc=False, min_len=1, max_len=None, numerals=False)
        corpus = text_corpus.ProcessedCorpus(reader, isalnum=False, **kwargs)
        vectorizer = corpus_vectorizer.CorpusVectorizer(lowercase=False)
        vectorizer.fit_transform(corpus)

        # Act
        term_term_matrix = np.dot(vectorizer.X.T, vectorizer.X)

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
        id2token = { i: t for t,i in vectorizer.vocabulary.items()}
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
        self.assertEqual({'0': 12, '1': 7}, n_tokens)
        self.assertEqual({'0': 12, '1': 7}, n_raw_tokens)

unittest.main(argv=['first-arg-is-ignored'], exit=False)

