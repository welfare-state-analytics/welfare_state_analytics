
import os
import pickle
import time
import logging

import numpy as np
import pandas as pd
import sklearn.preprocessing

from numba import jit
from heapq import nlargest
import logging

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
    def normalize(self, axis=1, norm='l1'):

        normalized_bag_term_matrix = sklearn.preprocessing.normalize(self.bag_term_matrix, axis=axis, norm=norm)

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

