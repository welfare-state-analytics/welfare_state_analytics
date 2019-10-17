import os
import unittest
import types
import pandas as pd
import numpy as np

from westac.common import corpus_vectorizer
from westac.common import text_corpus

from westac.tests.utils  import create_text_files_reader

flatten = lambda l: [ x for ws in l for x in ws]

class MockedProcessedCorpus():

    def __init__(self, mock_data):
        self.tokenized_documents = [ (f,y,self.generate_document(ws)) for f,y,ws in mock_data]
        self.vocabulary = self.create_vocabulary()
        self.n_tokens = { f: len(d) for f,y,d in mock_data }

    def get_metadata(self):

        return [
            types.SimpleNamespace(filename=x[0],year=x[1]) for x in self.tokenized_documents
        ]

    def create_vocabulary(self):
        return { w: i for i, w in enumerate(sorted(list(set(flatten([ x[2] for x in self.tokenized_documents]))))) }

    def documents(self):

        for filename, year, tokens in self.tokenized_documents:
            yield types.SimpleNamespace(filename=filename, year=year), tokens

    def generate_document(self, words):
        if isinstance(words, str):
            #parts = re.findall(r"(\d*)\**(\w+)\S?", words)
            #words = [ (1 if x[0] == '' else int(x[0]), x[1]) for x in parts ]
            document = words.split()
        else:
            document =  flatten([ n * w for n, w in words])
        return document

def mock_corpus():
    mock_corpus_data = [
        ('document_2013_1.txt', 2013, "a a b c c c c d"),
        ('document_2013_2.txt', 2013, "a a b b c c c"),
        ('document_2014_1.txt', 2014, "a a b b b c c"),
        ('document_2014_2.txt', 2014, "a a b b b b c d"),
        ('document_2014_2.txt', 2014, "a a c d")
    ]
    corpus = MockedProcessedCorpus(mock_corpus_data)
    return corpus

class Test_CorpusVectorizer(unittest.TestCase):

    def setUp(self):
        pass

    def mock_vectorizer(self):
        corpus = mock_corpus()
        vectorizer = corpus_vectorizer.CorpusVectorizer()
        vectorizer.fit_transform(corpus)
        return vectorizer

    def create_reader(self):
        meta_extract = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
        reader = create_text_files_reader(meta_extract=meta_extract, compress_whitespaces=True, dehyphen=True)
        return reader

    def create_corpus(self):
        reader = self.create_reader()
        kwargs = dict(isalnum=True, to_lower=True, deacc=False, min_len=2, max_len=None, numerals=False)
        corpus = text_corpus.ProcessedCorpus(reader, **kwargs)
        return corpus

    def test_fit_transform_creates_a_vocabulary(self):
        corpus = self.create_corpus()
        vectorizer = corpus_vectorizer.CorpusVectorizer()
        vectorizer.fit_transform(corpus)
        results = vectorizer.vocabulary
        expected = {'tre': 69, 'svarta': 62, 'ekar': 9, 'ur': 72, 'snön': 54, 'så': 65, 'grova': 17, 'men': 32, 'fingerfärdiga': 13, 'deras': 6, 'väldiga': 78, 'flaskor': 14, 'ska': 50, 'grönskan': 19, 'skumma': 53, 'vår': 79, 'på': 44, 'väg': 77, 'det': 7, 'långa': 29, 'mörkret': 36, 'envist': 11, 'skimrar': 51, 'mitt': 33, 'armbandsur': 2, 'med': 31, 'tidens': 67, 'fångna': 16, 'insekt': 25, 'nordlig': 38, 'storm': 61, 'är': 81, 'den': 5, 'tid': 66, 'när': 39, 'rönnbärsklasar': 45, 'mognar': 34, 'vaken': 74, 'hör': 24, 'man': 30, 'stjärnbilderna': 59, 'stampa': 58, 'sina': 48, 'spiltor': 57, 'högt': 23, 'över': 82, 'trädet': 70, 'jag': 26, 'ligger': 28, 'sängen': 64, 'armarna': 1, 'utbredda': 73, 'ett': 12, 'ankare': 0, 'som': 55, 'grävt': 18, 'ner': 37, 'sig': 47, 'ordentligt': 42, 'och': 40, 'håller': 22, 'kvar': 27, 'skuggan': 52, 'flyter': 15, 'där': 8, 'ovan': 43, 'stora': 60, 'okända': 41, 'en': 10, 'del': 4, 'av': 3, 'säkert': 63, 'viktigare': 76, 'än': 80, 'har': 20, 'sett': 46, 'mycket': 35, 'verkligheten': 75, 'tärt': 71, 'här': 21, 'sommaren': 56, 'till': 68, 'sist': 49}
        self.assertEqual(expected, results)

    def test_fit_transform_creates_a_bag_of_word_doc_term_matrix(self):
        corpus = mock_corpus()
        vectorizer = corpus_vectorizer.CorpusVectorizer()
        vectorizer.fit_transform(corpus)
        expected_vocab = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
        expected_dtm = [
            [2, 1, 4, 1],
            [2, 2, 3, 0],
            [2, 3, 2, 0],
            [2, 4, 1, 1],
            [2, 0, 1, 1]
        ]
        expected_word_counts = {'a': 10, 'b': 10, 'c': 11, 'd': 3}
        self.assertEqual(expected_vocab, vectorizer.vocabulary)
        self.assertEqual(expected_word_counts, vectorizer.word_counts)
        self.assertTrue((expected_dtm == vectorizer.X.toarray()).all())

    def test_collapse_to_year_aggregates_doc_term_matrix_to_year_term_matrix(self):
        vectorizer = corpus_vectorizer.CorpusVectorizer()
        X = np.array([
            [2, 1, 4, 1],
            [2, 2, 3, 0],
            [2, 3, 2, 0],
            [2, 4, 1, 1],
            [2, 0, 1, 1]
        ])
        df = pd.DataFrame({'year': [ 2013, 2013, 2014, 2014, 2014 ]})
        Y = vectorizer.collapse_to_year(X, df)
        expected_ytm = [
            [4, 3, 7, 1],
            [6, 7, 4, 2]
        ]
        self.assertTrue((expected_ytm == Y).all())

    def test_collapse_to_category_aggregates_doc_term_matrix_to_category_term_matrix(self):
        """ A more generic version of collapse_to_year (not used for now) """
        vectorizer = corpus_vectorizer.CorpusVectorizer()
        X = np.array([
            [2, 1, 4, 1],
            [2, 2, 3, 0],
            [2, 3, 2, 0],
            [2, 4, 1, 1],
            [2, 0, 1, 1]
        ])
        df = pd.DataFrame({'year': [ 2013, 2013, 2014, 2014, 2014 ]})
        Y, _ = vectorizer.collapse_by_category('year', X, df)
        expected_ytm = [
            [4, 3, 7, 1],
            [6, 7, 4, 2]
        ]
        self.assertTrue((expected_ytm == Y).all())

    def test_normalize_returns_the_l1_norm_of_each_row(self):
        vectorizer = self.mock_vectorizer()
        X = np.array([
            [4, 3, 7, 1],
            [6, 7, 4, 2]
        ], dtype=np.float)
        Y = vectorizer.normalize(X)
        E = np.array([
            [4, 3, 7, 1],
            [6, 7, 4, 2]
        ]) / (np.array([[15,19]]).T)
        self.assertTrue((E == Y).all())

    def test_tokens_above_threshold(self):
        vectorizer = self.mock_vectorizer()
        tokens = vectorizer.tokens_above_threshold(4)
        expected_tokens = {'a': 10, 'b': 10, 'c': 11 }
        self.assertEqual(expected_tokens, tokens)

    def test_token_ids_above_threshold(self):
        vectorizer = self.mock_vectorizer()
        ids = vectorizer.token_ids_above_threshold(4)
        expected_ids = [
            vectorizer.vocabulary['a'],
            vectorizer.vocabulary['b'],
            vectorizer.vocabulary['c']
        ]
        self.assertEqual(expected_ids, ids)

    def test_word_counts(self):
        corpus = self.create_corpus()
        vectorizer = corpus_vectorizer.CorpusVectorizer()
        _ = vectorizer.fit_transform(corpus)
        results = vectorizer.word_counts
        expected = {
            'tre': 1, 'svarta': 1, 'ekar': 1, 'ur': 2, 'snön': 1, 'så': 3, 'grova': 1, 'men': 2, 'fingerfärdiga': 1,
            'deras': 1, 'väldiga': 2, 'flaskor': 1, 'ska': 1, 'grönskan': 1, 'skumma': 1, 'vår': 1, 'på': 3, 'väg': 1,
            'det': 3, 'långa': 1, 'mörkret': 2, 'envist': 1, 'skimrar': 1, 'mitt': 1, 'armbandsur': 1, 'med': 2, 'tidens': 1,
            'fångna': 1, 'insekt': 1, 'nordlig': 1, 'storm': 1, 'är': 5, 'den': 3, 'tid': 1, 'när': 1, 'rönnbärsklasar': 1,
            'mognar': 1, 'vaken': 1, 'hör': 1, 'man': 2, 'stjärnbilderna': 1, 'stampa': 1, 'sina': 1, 'spiltor': 1,
            'högt': 1, 'över': 1, 'trädet': 1, 'jag': 4, 'ligger': 1, 'sängen': 1, 'armarna': 1, 'utbredda': 1, 'ett': 1,
            'ankare': 1, 'som': 4, 'grävt': 1, 'ner': 1, 'sig': 1, 'ordentligt': 1, 'och': 2, 'håller': 1, 'kvar': 1,
            'skuggan': 1, 'flyter': 1, 'där': 1, 'ovan': 1, 'stora': 1, 'okända': 1, 'en': 2, 'del': 1, 'av': 1, 'säkert': 1,
            'viktigare': 1, 'än': 1, 'har': 2, 'sett': 1, 'mycket': 2, 'verkligheten': 1, 'tärt': 1, 'här': 1, 'sommaren': 1,
            'till': 1, 'sist': 1
        }
        self.assertEqual(expected, results)

    def test_dump_can_be_loaded(self):

        # Arrange
        corpus = self.create_corpus()
        vectorizer = corpus_vectorizer.CorpusVectorizer()
        _ = vectorizer.fit_transform(corpus)

        # Act
        vectorizer.dump('dump_test', folder='./westac/tests/output')

        # Assert
        data_filename ="./westac/tests/output/dump_test_vectorizer_data.pickle"
        matrix_filename = "./westac/tests/output/dump_test_vector_data.npy"

        self.assertTrue(os.path.isfile(data_filename))
        self.assertTrue(os.path.isfile(matrix_filename))

        # Act
        loaded_vectorizer = corpus_vectorizer.CorpusVectorizer().load('dump_test', folder='./westac/tests/output')

        # Assert
        self.assertEqual(vectorizer.word_counts, loaded_vectorizer.word_counts)
        self.assertEqual(vectorizer.document_index.to_dict(), loaded_vectorizer.document_index.to_dict())
        self.assertEqual(vectorizer.vocabulary, loaded_vectorizer.vocabulary)
        #self.assertEqual(vectorizer.X, loaded_vectorizer.X)
        self.assertIsNone(loaded_vectorizer.corpus)
        #self.assertEqual(vectorizer.vectorizer, loaded_vectorizer.vectorizer)
