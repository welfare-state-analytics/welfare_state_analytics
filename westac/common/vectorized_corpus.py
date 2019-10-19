
import numpy as np
import pickle
import time
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
        data_filename = VectorizedCorpus._data_filename(tag, folder)

        with open(data_filename, 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

        self.vectorizer.tokenizer = None
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
        word_counts = data["word_counts"]
        document_index = data["document_index"]

        matrix_filename = VectorizedCorpus._matrix_filename(tag, folder)
        doc_term_matrix = np.load(matrix_filename, allow_pickle=True).item()

        return VectorizedCorpus(doc_term_matrix, vocabulary, word_counts, document_index)

    @staticmethod
    def _data_filename(tag, folder):
        return os.path.join(folder, "{}_vectorizer_data.pickle".format(tag))

    @staticmethod
    def _matrix_filename(tag, folder):
        return os.path.join(folder, "{}_vector_data.npy".format(tag))

