# -*- coding: utf-8 -*-
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
# %load_ext autoreload
# %autoreload 2

import os

import unittest
import numpy as np
import pandas as pd
import types

from westac.common import corpus_vectorizer
from westac.common import utility
from westac.common import text_corpus
# %matplotlib inline

TEST_CORPUS_FILENAME = './westac/tests/test_data/test_corpus.zip'

if __file__ in globals():
    this_file = os.path.dirname(__file__)
    this_path = os.path.abspath(this_file)
    TEST_CORPUS_FILENAME = os.path.join(this_path, TEST_CORPUS_FILENAME)

# +
flatten = lambda l: [ x for ws in l for x in ws]

def create_text_files_reader(
    filename=TEST_CORPUS_FILENAME,
    pattern="*.txt",
    itemfilter=None,
    compress_whitespaces=False,
    dehyphen=True,
    meta_extract=None
):
    kwargs = dict(
        pattern=pattern,
        itemfilter=itemfilter,
        compress_whitespaces=compress_whitespaces,
        dehyphen=dehyphen,
        meta_extract=meta_extract
    )
    reader = utility.TextFilesReader(filename, **kwargs)
    return reader

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

# +

class test_TextFilesReader(unittest.TestCase):

    def test_archive_filenames_when_filter_txt_returns_txt_files(self):
        reader = create_text_files_reader(pattern='*.txt')
        self.assertEqual(5, len(reader.archive_filenames))

    def test_archive_filenames_when_filter_md_returns_md_files(self):
        reader = create_text_files_reader(pattern='*.md')
        self.assertEqual(1, len(reader.archive_filenames))

    def test_archive_filenames_when_filter_function_txt_returns_txt_files(self):
        itemfilter = lambda _, x: x.endswith('txt')
        reader = create_text_files_reader(itemfilter=itemfilter)
        self.assertEqual(5, len(reader.archive_filenames))

    def test_get_file_when_default_returns_unmodified_content(self):
        document_name = 'dikt_2019_01_test.txt'
        reader = create_text_files_reader(compress_whitespaces=False, dehyphen=True)
        result = next(reader.get_file(document_name))
        expected = "Tre svarta ekar ur snön.\r\n" + \
                   "Så grova, men fingerfärdiga.\r\n" + \
                   "Ur deras väldiga flaskor\r\n" + \
                   "ska grönskan skumma i vår."
        self.assertEqual(document_name, result[0])
        self.assertEqual(expected, result[1])

    def test_can_get_file_when_compress_whitespace_is_true_strips_whitespaces(self):
        document_name = 'dikt_2019_01_test.txt'
        reader = create_text_files_reader(compress_whitespaces=True, dehyphen=True)
        result = next(reader.get_file(document_name))
        expected = "Tre svarta ekar ur snön. " + \
                   "Så grova, men fingerfärdiga. " + \
                   "Ur deras väldiga flaskor " + \
                   "ska grönskan skumma i vår."
        self.assertEqual(document_name, result[0])
        self.assertEqual(expected, result[1])

    def test_get_file_when_dehyphen_is_trye_removes_hyphens(self):
        document_name = 'dikt_2019_03_test.txt'
        reader = create_text_files_reader(compress_whitespaces=True, dehyphen=True)
        result = next(reader.get_file(document_name))
        expected = "Nordlig storm. Det är den i den tid när rönnbärsklasar mognar. Vaken i mörkret hör man " + \
                   "stjärnbilderna stampa i sina spiltor " + \
                   "högt över trädet"
        self.assertEqual(document_name, result[0])
        self.assertEqual(expected, result[1])

    def test_get_file_when_file_exists_and_extractor_specified_returns_content_and_metadat(self):
        document_name = 'dikt_2019_03_test.txt'
        meta_extract = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
        reader = create_text_files_reader(meta_extract=meta_extract, compress_whitespaces=True, dehyphen=True)
        result = next(reader.get_file(document_name))
        expected = "Nordlig storm. Det är den i den tid när rönnbärsklasar mognar. Vaken i mörkret hör man " + \
                   "stjärnbilderna stampa i sina spiltor " + \
                   "högt över trädet"
        self.assertEqual(document_name, result[0].filename)
        self.assertEqual(2019, result[0].year)
        self.assertEqual(3, result[0].serial_no)
        self.assertEqual(expected, result[1])

    def test_get_index_when_extractor_passed_returns_metadata(self):
        meta_extract = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
        reader = create_text_files_reader(meta_extract=meta_extract, compress_whitespaces=True, dehyphen=True)
        result = reader.metadata
        expected = [
            types.SimpleNamespace(filename='dikt_2019_01_test.txt', serial_no=1, year=2019),
            types.SimpleNamespace(filename='dikt_2019_02_test.txt', serial_no=2, year=2019),
            types.SimpleNamespace(filename='dikt_2019_03_test.txt', serial_no=3, year=2019),
            types.SimpleNamespace(filename='dikt_2020_01_test.txt', serial_no=1, year=2020),
            types.SimpleNamespace(filename='dikt_2020_02_test.txt', serial_no=2, year=2020)]

        self.assertEqual(len(expected), len(result))
        for i in range(0,len(expected)):
            self.assertEqual(expected[i], result[i])

class test_Utilities(unittest.TestCase):

    def setUp(self):
        pass

    def test_dehypen(self):

        text = 'absdef\n'
        result = utility.dehyphen(text)
        self.assertEqual(text, result)

        text = 'abs-def\n'
        result = utility.dehyphen(text)
        self.assertEqual(text, result)

        text = 'abs - def\n'
        result = utility.dehyphen(text)
        self.assertEqual(text, result)

        text = 'abs-\ndef'
        result = utility.dehyphen(text)
        self.assertEqual('absdef\n', result)

        text = 'abs- \r\n def'
        result = utility.dehyphen(text)
        self.assertEqual('absdef\n', result)

    def test_compress_whitespaces(self):

        text = 'absdef\n'
        result = utility.compress_whitespaces(text)
        self.assertEqual('absdef', result)

        text = ' absdef \n'
        result = utility.compress_whitespaces(text)
        self.assertEqual( 'absdef', result)

        text = 'abs  def'
        result = utility.compress_whitespaces(text)
        self.assertEqual('abs def', result)

        text = 'abs\n def'
        result = utility.compress_whitespaces(text)
        self.assertEqual('abs def', result)

        text = 'abs- \r\n def'
        result = utility.compress_whitespaces(text)
        self.assertEqual('abs- def', result)

class Test_ExtractMeta(unittest.TestCase):

    def test_extract_metadata_when_valid_regexp_returns_metadata_values(self):
        filename = 'SOU 1957_5 Namn.txt'
        meta = utility.extract_metadata(filename, year=r".{4}(\d{4})_.*", serial_no=r".{8}_(\d+).*")
        self.assertEqual(5, meta.serial_no)
        self.assertEqual(1957, meta.year)

    def test_extract_metadata_when_invalid_regexp_returns_none(self):
        filename = 'xyz.txt'
        meta = utility.extract_metadata(filename, value=r".{4}(\d{4})_.*")
        self.assertEqual(None, meta.value)

# +

class Test_CorpusTextStream(unittest.TestCase):

    def test_next_document_when_new_corpus_returns_document(self):
        reader = create_text_files_reader(compress_whitespaces=True, dehyphen=True)
        corpus = text_corpus.CorpusTextStream(reader)
        result = next(corpus.documents())
        expected = "Tre svarta ekar ur snön. " + \
                   "Så grova, men fingerfärdiga. " + \
                   "Ur deras väldiga flaskor " + \
                   "ska grönskan skumma i vår."
        self.assertEqual(expected, result[1])

    def test_get_index_when_extract_passed_returns_metadata(self):
        meta_extract = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
        reader = create_text_files_reader(meta_extract=meta_extract, compress_whitespaces=True, dehyphen=True)
        corpus = text_corpus.CorpusTextStream(reader)
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
        reader = create_text_files_reader(meta_extract=None, compress_whitespaces=True, dehyphen=True)
        corpus = text_corpus.CorpusTextStream(reader)
        result = corpus.get_metadata()
        self.assertIsNone(result)

# +
class Test_CorpusTokenStream(unittest.TestCase):

    def create_reader(self):
        meta_extract = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
        reader = create_text_files_reader(meta_extract=meta_extract, compress_whitespaces=True, dehyphen=True)
        return reader

    def test_next_document_when_token_corpus_returns_tokenized_document(self):
        reader = create_text_files_reader(meta_extract=None, compress_whitespaces=True, dehyphen=True)
        corpus = text_corpus.CorpusTokenStream(reader, isalnum=False)
        _, tokens = next(corpus.documents())
        expected = ["Tre", "svarta", "ekar", "ur", "snön", ".",
                    "Så", "grova", ",", "men", "fingerfärdiga", ".",
                    "Ur", "deras", "väldiga", "flaskor",
                    "ska", "grönskan", "skumma", "i", "vår", "."]
        self.assertEqual(expected, tokens)

    def test_next_document_when_isalnum_true_skips_deliminators(self):
        reader = create_text_files_reader(meta_extract=None, compress_whitespaces=True, dehyphen=True)
        corpus = text_corpus.CorpusTokenStream(reader, isalnum=True)
        _, tokens = next(corpus.documents())
        expected = ["Tre", "svarta", "ekar", "ur", "snön",
                    "Så", "grova", "men", "fingerfärdiga",
                    "Ur", "deras", "väldiga", "flaskor",
                    "ska", "grönskan", "skumma", "i", "vår"]
        self.assertEqual(expected, tokens)

    def test_get_index_when_extract_passed_returns_expected_count(self):
        reader = self.create_reader()
        corpus = text_corpus.CorpusTokenStream(reader)
        result = corpus.get_metadata()
        self.assertEqual(5, len(result))

    def test_n_tokens_when_exhausted_iterater_returns_expected_count(self):
        reader = self.create_reader()
        corpus = text_corpus.CorpusTokenStream(reader, isalnum=False)
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
        reader = create_text_files_reader(meta_extract=None, compress_whitespaces=True, dehyphen=True)
        corpus = text_corpus.CorpusTokenStream(reader, isalnum=True)
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

# +
class Test_ProcessedCorpus(unittest.TestCase):

    def setUp(self):
        pass

    def create_reader(self):
        meta_extract = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
        reader = create_text_files_reader(meta_extract=meta_extract, compress_whitespaces=True, dehyphen=True)
        return reader

    def test_next_document_when_isalnum_is_true_returns_all_tokens(self):
        reader = self.create_reader()
        kwargs = dict(isalnum=False, to_lower=False, deacc=False, min_len=1, max_len=None, numerals=True)
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

# +

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

    def test_fit_transform_(self):
        corpus = self.create_corpus()
        vectorizer = corpus_vectorizer.CorpusVectorizer()
        vectorizer.fit_transform(corpus)
        results = vectorizer.vocabulary
        expected = {'tre': 69, 'svarta': 62, 'ekar': 9, 'ur': 72, 'snön': 54, 'så': 65, 'grova': 17, 'men': 32, 'fingerfärdiga': 13, 'deras': 6, 'väldiga': 78, 'flaskor': 14, 'ska': 50, 'grönskan': 19, 'skumma': 53, 'vår': 79, 'på': 44, 'väg': 77, 'det': 7, 'långa': 29, 'mörkret': 36, 'envist': 11, 'skimrar': 51, 'mitt': 33, 'armbandsur': 2, 'med': 31, 'tidens': 67, 'fångna': 16, 'insekt': 25, 'nordlig': 38, 'storm': 61, 'är': 81, 'den': 5, 'tid': 66, 'när': 39, 'rönnbärsklasar': 45, 'mognar': 34, 'vaken': 74, 'hör': 24, 'man': 30, 'stjärnbilderna': 59, 'stampa': 58, 'sina': 48, 'spiltor': 57, 'högt': 23, 'över': 82, 'trädet': 70, 'jag': 26, 'ligger': 28, 'sängen': 64, 'armarna': 1, 'utbredda': 73, 'ett': 12, 'ankare': 0, 'som': 55, 'grävt': 18, 'ner': 37, 'sig': 47, 'ordentligt': 42, 'och': 40, 'håller': 22, 'kvar': 27, 'skuggan': 52, 'flyter': 15, 'där': 8, 'ovan': 43, 'stora': 60, 'okända': 41, 'en': 10, 'del': 4, 'av': 3, 'säkert': 63, 'viktigare': 76, 'än': 80, 'har': 20, 'sett': 46, 'mycket': 35, 'verkligheten': 75, 'tärt': 71, 'här': 21, 'sommaren': 56, 'till': 68, 'sist': 49}
        self.assertEqual(expected, results)

    def test_fit_transform(self):
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

    def test_collapse_to_year(self):
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

    def test_collapse_by_category(self):
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

    def test_normalize(self):
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

unittest.main(argv=['first-arg-is-ignored'], exit=False)

# +
import scipy
from scipy.cluster.hierarchy import dendrogram, linkage # pylint: disable=unused-import
from matplotlib import pyplot as plt # pylint: disable=unused-import

class Test_ChiSquare(unittest.TestCase):

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

    def test_chisquare(self):

        corpus = self.create_corpus()
        vectorizer = corpus_vectorizer.CorpusVectorizer()
        vectorizer.fit_transform(corpus)

        id2token = { i: w for w, i in vectorizer.vocabulary.items() }

        Y = vectorizer.collapse_to_year()
        Yn = vectorizer.normalize(Y, axis=1, norm='l1')

        indices = vectorizer.token_ids_above_threshold(1)
        Ynw = Yn[:, indices]

        X2 = scipy.stats.chisquare(Ynw, f_exp=None, ddof=0, axis=0) # pylint: disable=unused-variable

        # Use X2 so select top 500 words... (highest Power-Power_divergenceResult)
        # Ynw = largest_by_chisquare()
        #print(Ynw)

        linked = linkage(Ynw.T, 'ward') # pylint: disable=unused-variable
        #print(linked)

        labels = [ id2token[x] for x in indices ] # pylint: disable=unused-variable

        #plt.figure(figsize=(24, 16))
        #dendrogram(linked, orientation='top', labels=labels, distance_sort='descending', show_leaf_counts=True)
        #plt.show()

        results = None
        expected = None

        self.assertEqual(expected, results)

# -

def plot_dists(vectorizer):
    df = pd.DataFrame(vectorizer.X.toarray(), columns=list(vectorizer.get_feature_names()))
    df['year'] = df.index + 45
    df = df.set_index('year')
    df['year'] =  pd.Series(df.index).apply(lambda x: vectorizer.document_index[x][0])
    df[['krig']].plot() #.loc[df["000"]==49]


unittest.main(argv=['first-arg-is-ignored'], exit=False)
