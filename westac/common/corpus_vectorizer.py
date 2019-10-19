
import pandas as pd
import numpy as np
import pickle
import time
from sklearn.feature_extraction.text import CountVectorizer
import sklearn.preprocessing
import os

class VectorizedCorpus():

    def __init__(self, doc_term_matrix, vocabulary, word_counts, document_index):
        self.doc_term_matrix = doc_term_matrix
        self.vocabulary = vocabulary
        self.word_counts = word_counts
        self.document_index = document_index

    def dump(self, tag=None, folder='./output'):

        tag = tag or time.strftime("%Y%m%d_%H%M%S")

        data = {
            'vocabulary': self.vocabulary,
            'word_counts': self.word_counts,
            'document_index': self.document_index
        }
        data_filename = self._data_filename(tag, folder)

        self.vectorizer.tokenizer = None
        with open(data_filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        matrix_filename = self._matrix_filename(tag, folder)
        np.save(matrix_filename, self.doc_term_matrix, allow_pickle=True)

    def dump_exists(self, tag, folder='./output'):
        return os.path.isfile(self._data_filename(tag, folder))

    def load(self, tag, folder='./output'):

        data_filename = self._data_filename(tag, folder)
        with open(data_filename, 'rb') as f:
            data = pickle.load(f)

        self.vocabulary = data["vocabulary"]
        self.word_counts = data["word_counts"]
        self.document_index = data["document_index"]

        matrix_filename = self._matrix_filename(tag, folder)
        self.doc_term_matrix = np.load(matrix_filename, allow_pickle=True).item()

        return self

    def _data_filename(self, tag, folder):
        return os.path.join(folder, "{}_vectorizer_data.pickle".format(tag))

    def _matrix_filename(self, tag, folder):
        return os.path.join(folder, "{}_vector_data.npy".format(tag))


class CorpusVectorizer():

    def __init__(self, **kwargs):
        self.corpus = None
        self.doc_term_matrix = None
        self.vectorizer = None
        self.vocabulary = None
        self.word_counts = None
        self.kwargs = kwargs
        self.document_index = None

    def fit_transform(self, corpus):

        def text_iterator(x):
            n_documents = 0
            for meta, tokens in x.documents():
                n_documents += 1
                print("{} ({})".format(meta, n_documents))
                yield ' '.join(tokens)

        #texts = (' '.join(tokens) for _, tokens in corpus.documents())
        texts = text_iterator(corpus)

        tokenizer = lambda x: x.split()

        #https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/feature_extraction/text.py#L1147
        self.vectorizer = CountVectorizer(tokenizer=tokenizer, **self.kwargs)

        self.doc_term_matrix = self.vectorizer.fit_transform(texts)

        self.corpus = corpus
        self.vocabulary = self.vectorizer.vocabulary_

        Xsum = self.doc_term_matrix.sum(axis=0)

        self.word_counts = { w: Xsum[0, i] for w,i in self.vocabulary.items() }
        self.document_index = self._document_index()

        return self.doc_term_matrix

    def dump(self, tag=None, folder='./output'):

        tag = tag or time.strftime("%Y%m%d_%H%M%S")

        data = {
            'vectorizer': self.vectorizer,
            'vocabulary': self.vocabulary,
            'word_counts': self.word_counts,
            'document_index': self.document_index
        }
        data_filename = os.path.join(folder, "{}_vectorizer_data.pickle".format(tag))

        self.vectorizer.tokenizer = None
        with open(data_filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        matrix_filename = os.path.join(folder, "{}_vector_data.npy".format(tag))
        np.save(matrix_filename, self.doc_term_matrix, allow_pickle=True)

    def dump_exists(self, tag, folder='./output'):
        data_filename = os.path.join(folder, "{}_vectorizer_data.pickle".format(tag))
        return os.path.isfile(data_filename)

    def load(self, tag, folder='./output'):

        self.corpus = None

        data_filename = os.path.join(folder, "{}_vectorizer_data.pickle".format(tag))
        with open(data_filename, 'rb') as f:
            data = pickle.load(f)

        self.vectorizer = data["vectorizer"]
        self.vocabulary = data["vocabulary"]
        self.word_counts = data["word_counts"]
        self.document_index = data["document_index"]

        matrix_filename = os.path.join(folder, "{}_vector_data.npy".format(tag))
        self.doc_term_matrix = np.load(matrix_filename, allow_pickle=True).item()

        return self

    def _document_index(self):
        """ Groups matrix by vales in column summing up all values in each category
        """
        metadata = self.corpus.get_metadata()
        df = pd.DataFrame([ x.__dict__ for x in metadata ], columns=metadata[0].__dict__.keys())
        df['document_id'] = list(df.index)
        return df

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

    def collapse_to_year(self, X=None, df=None):

        X = self.doc_term_matrix if X is None else X
        df = self.document_index if df is None else df

        min_value, max_value = df.year.min(), df.year.max()

        Y = np.zeros(((max_value - min_value) + 1, X.shape[1]))

        for i in range(0, Y.shape[0]): # pylint: disable=unsubscriptable-object

            indices = list((df.loc[df.year == min_value + i].index))

            if len(indices) > 0:
                Y[i,:] = X[indices,:].sum(axis=0)

        return Y

    def normalize(self, X, axis=1, norm='l1'):
        Xn = sklearn.preprocessing.normalize(X, axis=axis, norm=norm)
        return Xn

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

