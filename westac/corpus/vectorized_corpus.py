
import os
import pickle
import time
import logging

import numpy as np
import pandas as pd
import sklearn.preprocessing

from heapq import nlargest

# logging.basicConfig(filename="newfile.log", format='%(asctime)s %(message)s', filemode='w')

logger = logging.getLogger()
logger.setLevel(logging.INFO)

class VectorizedCorpus():

    def __init__(self, bag_term_matrix, token2id, document_index, word_counts=None):

        self.bag_term_matrix = bag_term_matrix
        self.token2id = token2id
        self.id2token_ = None
        self.document_index = document_index
        self.word_counts = word_counts

        if self.word_counts is None:

            Xsum = self.bag_term_matrix.sum(axis=0)
            Xsum = np.ravel(Xsum)

            self.word_counts = { w: Xsum[i] for w,i in self.token2id.items() }
            # self.id2token = { i: t for t,i in self.token2id.items()}

        n_bags = self.bag_term_matrix.shape[0]
        n_vocabulary = self.bag_term_matrix.shape[1]
        n_tokens = sum(self.word_counts.values())

        logger.info('Corpus size: #bags: {}, #vocabulary: {}, #tokens: {}'.format(n_bags, n_vocabulary, n_tokens))

    @property
    def id2token(self):
        if self.id2token_ is None and self.token2id is not None:
            self.id2token_ = { i: t for t,i in self.token2id.items()}
        return self.id2token_

    @property
    def T(self):
        return self.bag_term_matrix.T

    @property
    def data(self):
        return self.bag_term_matrix

    @property
    def term_bag_matrix(self):
        return self.bag_term_matrix.T

    def dump(self, tag=None, folder='./output'):

        tag = tag or time.strftime("%Y%m%d_%H%M%S")

        data = {
            'token2id': self.token2id,
            'word_counts': self.word_counts,
            'document_index': self.document_index
        }
        data_filename = VectorizedCorpus._data_filename(tag, folder)

        with open(data_filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        matrix_filename = VectorizedCorpus._matrix_filename(tag, folder)
        np.save(matrix_filename, self.bag_term_matrix, allow_pickle=True)

        return self

    @staticmethod
    def dump_exists(tag, folder='./output'):
        return os.path.isfile(VectorizedCorpus._data_filename(tag, folder))

    @staticmethod
    def load(tag, folder='./output'):

        data_filename = VectorizedCorpus._data_filename(tag, folder)
        with open(data_filename, 'rb') as f:
            data = pickle.load(f)

        token2id = data["token2id"]
        document_index = data["document_index"]

        matrix_filename = VectorizedCorpus._matrix_filename(tag, folder)
        bag_term_matrix = np.load(matrix_filename, allow_pickle=True).item()

        return VectorizedCorpus(bag_term_matrix, token2id, document_index)

    @staticmethod
    def _data_filename(tag, folder):
        return os.path.join(folder, "{}_vectorizer_data.pickle".format(tag))

    @staticmethod
    def _matrix_filename(tag, folder):
        return os.path.join(folder, "{}_vector_data.npy".format(tag))

    def get_word_vector(self, word):
        return self.bag_term_matrix[:, self.token2id[word]]

    # FIXME: Moved to service
    def collapse_by_category(self, column, X=None, df=None):
        """Sums ups all rows in based on each row's index having same value in column `column`in data frame `df`

        Parameters
        ----------
        column : str
            The categorical column kn `df`that groups the rows in `X`

        X : np.ndarray(N, M), optional
            Matrix of shape (N, M), by default None

        df : DataFrame, optional
            DataFrame of size N, where each row `ì` contains data that describes row `i` in `X`, by default None

        Returns
        -------
        tuple: np.ndarray(K, M), list
            A matrix of size K wherw K is the number of unique categorical values in `df[column]`
            A list of length K of category values, where i:th value is category of i:th row in returned matrix
        """

        X = self.bag_term_matrix if X is None else X
        df = self.document_index if df is None else df

        assert X.shape[0] == len(df)

        categories = list(sorted(df[column].unique().tolist()))

        Y = np.zeros((len(categories), X.shape[1]), dtype=X.dtype)

        for i, value in enumerate(categories):
            indices = list((df.loc[df[column] == value].index))
            Y[i,:] = X[indices,:].sum(axis=0)

        return Y, categories

    #@jit
    def group_by_year(self):

        X = self.bag_term_matrix # if X is None else X
        df = self.document_index # if df is None else df

        min_value, max_value = df.year.min(), df.year.max()

        Y = np.zeros(((max_value - min_value) + 1, X.shape[1]))

        for i in range(0, Y.shape[0]): # pylint: disable=unsubscriptable-object

            indices = list((df.loc[df.year == min_value + i].index))

            if len(indices) > 0:
                Y[i,:] = X[indices,:].sum(axis=0)

        years = list(range(min_value, max_value + 1))
        document_index = pd.DataFrame({
            'year': years,
            'filename': map(str, years)
        })

        v_corpus = VectorizedCorpus(Y, self.token2id, document_index, self.word_counts)

        return v_corpus

    #@jit
    def normalize(self, axis=1, norm='l1', keep_magnitude=False):

        normalized_bag_term_matrix = sklearn.preprocessing.normalize(self.bag_term_matrix, axis=axis, norm=norm)

        if keep_magnitude is True:
            X0 = self.bag_term_matrix[0]
            Y0 = normalized_bag_term_matrix[0]
            K  = X0 / Y0
            normalized_bag_term_matrix = normalized_bag_term_matrix * K

        v_corpus = VectorizedCorpus(normalized_bag_term_matrix, self.token2id, self.document_index, self.word_counts)

        return v_corpus

    def n_top_tokens(self, n_top):
        tokens = { w: self.word_counts[w] for w in nlargest(n_top, self.word_counts, key = self.word_counts.get) }
        return tokens

    # @autojit
    def slice_by_n_count(self, n_count):

        tokens = set(w for w,c in self.word_counts.items() if c >= n_count)
        def _px(w):
            return w in tokens

        return self.slice_by(_px)

    def slice_by_n_top(self, n_top):

        tokens = set(nlargest(n_top, self.word_counts, key = self.word_counts.get))

        def _px(w):
            return w in tokens

        return self.slice_by(_px)

    #@autojit
    def slice_by(self, px):

        indices = [ self.token2id[w] for w in self.token2id.keys() if px(w) ]

        indices.sort()

        sliced_bag_term_matrix = self.bag_term_matrix[:, indices]
        token2id = { self.id2token[indices[i]]: i for i in range(0, len(indices)) }
        word_counts = { w: c for w,c in self.word_counts.items() if w in token2id }

        v_corpus = VectorizedCorpus(sliced_bag_term_matrix, token2id, self.document_index, word_counts)

        return v_corpus

    def stats(self):
        stats_data = {
            'bags': self.bag_term_matrix.shape[0],
            'vocabulay_size': self.bag_term_matrix.shape[1],
            'sum_over_bags': self.bag_term_matrix.sum(),
            '10_top_tokens': ' '.join(self.n_top_tokens(10).keys())
        }
        for key in stats_data.keys():
            logger.info('   {}: {}'.format(key, stats_data[key]))
        return stats_data

    def to_n_top_dataframe(self, n_top):
        v_n_corpus = self.slice_by_n_top(n_top)
        data = v_n_corpus.bag_term_matrix.T
        df = pd.DataFrame(data=data, index=[v_n_corpus.id2token[i] for i in range(0,n_top)], columns=list(range(1945, 1990)))
        return df

    def year_range(self):
        if 'year' in self.document_index.columns:
            return (self.document_index.year.min(), self.document_index.year.max())
        return (None, None)

    def xs_years(self):
        (low, high) = self.year_range()
        xs = np.arange(low, high + 1, 1)
        return xs

def load_corpus(tag, folder, n_count=10000, n_top=100000, axis=1, keep_magnitude=True):

    v_corpus = VectorizedCorpus\
        .load(tag, folder=folder)\
        .group_by_year()

    if n_count is not None:
        v_corpus = v_corpus.slice_by_n_count(n_count)

    if n_top is not None:
        v_corpus = v_corpus.slice_by_n_top(n_top)

    if axis is not None:
        v_corpus = v_corpus.normalize(axis=axis, keep_magnitude=keep_magnitude)

    return v_corpus

def load_cached_normalized_vectorized_corpus(tag, folder, n_count=10000, n_top=100000, keep_magnitude=True):

    year_cache_tag = "cached_year_{}_{}".format(tag, "km" if keep_magnitude else "")

    v_corpus = None

    if not VectorizedCorpus.dump_exists(year_cache_tag, folder=folder):
        logger.info("Caching corpus grouped by year...")
        v_corpus = VectorizedCorpus\
            .load(tag, folder=folder)\
            .group_by_year()\
            .normalize(axis=1, keep_magnitude=keep_magnitude)\
            .dump(year_cache_tag, folder)

    if v_corpus is None:
        v_corpus = VectorizedCorpus\
            .load(year_cache_tag, folder=folder)\
            .slice_by_n_count(n_count)\
            .slice_by_n_top(n_top)

    return v_corpus

