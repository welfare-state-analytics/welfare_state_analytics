
import numpy as np
import pickle
import time
import os
import sklearn.preprocessing

class VectorizedCorpus():

    def __init__(self, doc_term_matrix, vocabulary, document_index, word_counts=None):

        self.doc_term_matrix = doc_term_matrix
        self.vocabulary = vocabulary
        self.document_index = document_index

        self.word_counts = word_counts

        if self.word_counts is None:

            Xsum = self.doc_term_matrix.sum(axis=0)
            Xsum = np.ravel(Xsum)

            self.word_counts = { w: Xsum[i] for w,i in self.vocabulary.items() }
            # self.id2token = { i: t for t,i in self.vocabulary.items()}

    def dump(self, tag=None, folder='./output'):

        tag = tag or time.strftime("%Y%m%d_%H%M%S")

        data = {
            'vocabulary': self.vocabulary,
            'word_counts': self.word_counts,
            'document_index': self.document_index
        }
        data_filename = VectorizedCorpus._data_filename(tag, folder)

        with open(data_filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        matrix_filename = VectorizedCorpus._matrix_filename(tag, folder)
        np.save(matrix_filename, self.doc_term_matrix, allow_pickle=True)

        return self

    @staticmethod
    def dump_exists(tag, folder='./output'):
        return os.path.isfile(VectorizedCorpus._data_filename(tag, folder))

    @staticmethod
    def load(tag, folder='./output'):

        data_filename = VectorizedCorpus._data_filename(tag, folder)
        with open(data_filename, 'rb') as f:
            data = pickle.load(f)

        vocabulary = data["vocabulary"]
        document_index = data["document_index"]

        matrix_filename = VectorizedCorpus._matrix_filename(tag, folder)
        doc_term_matrix = np.load(matrix_filename, allow_pickle=True).item()

        return VectorizedCorpus(doc_term_matrix, vocabulary, document_index)

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

        X = self.doc_term_matrix if X is None else X
        df = self.document_index if df is None else df

        assert X.shape[0] == len(df)

        categories = list(sorted(df[column].unique().tolist()))

        Y = np.zeros((len(categories), X.shape[1]), dtype=X.dtype)

        for i, value in enumerate(categories):
            indices = list((df.loc[df[column] == value].index))
            Y[i,:] = X[indices,:].sum(axis=0)

        return Y, categories

    def collapse_to_year(self):

        X = self.doc_term_matrix # if X is None else X
        df = self.document_index # if df is None else df

        min_value, max_value = df.year.min(), df.year.max()

        Y = np.zeros(((max_value - min_value) + 1, X.shape[1]))

        for i in range(0, Y.shape[0]): # pylint: disable=unsubscriptable-object

            indices = list((df.loc[df.year == min_value + i].index))

            if len(indices) > 0:
                Y[i,:] = X[indices,:].sum(axis=0)

        return Y

    def normalize(self, X, axis=1, norm='l1'):
        normalized_doc_term_matrix = sklearn.preprocessing.normalize(self.doc_term_matrix, axis=axis, norm=norm)
        return VectorizedCorpus(normalized_doc_term_matrix, self.vocabulary, self.document_index, self.word_counts)

    def tokens_above_threshold(self, threshold):
        words = {
            w: c for w,c in self.word_counts.items() if c >= threshold
        }
        return words

    def token_ids_above_threshold(self, threshold):
        ids = [
            self.vocabulary[w] for w in self.tokens_above_threshold(threshold).keys()
        ]
        return ids

    def slice_tokens_by_count_threshold(self, X, threshold_count):
        indices = self.token_ids_above_threshold(threshold_count)

        Y = X[:, indices]

        return Y

