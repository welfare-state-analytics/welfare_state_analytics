
import pandas as pd
import numpy as np
import scipy
#import nltk.tokenize
import types
import pickle

from sklearn.feature_extraction.text import CountVectorizer
import sklearn.preprocessing
import utility

class CorpusVectorizer():
    
    def __init__(self):
        self.corpus = None
        self.X = None
        self.vectorizer = None
        self.vocabulary = None
        self.word_counts = None
        
    def fit_transform(self, corpus):
        
        texts = (' '.join(tokens) for _, tokens in corpus.documents())
        #https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/feature_extraction/text.py#L1147
        self.vectorizer = CountVectorizer()
        self.X = self.vectorizer.fit_transform(texts)
        
        self.corpus = corpus
        self.vocabulary = self.vectorizer.vocabulary_

        Xsum = self.X.sum(axis=0)
        
        self.word_counts = { w: Xsum[0, i] for w,i in self.vocabulary.items() }
        self.document_index = self._document_index()
        
        return self.X
    
    def dump(self, tag='vec.'):
        data = {
            'vectorizer': self.vectorizer,
            'vocabulary': self.vocabulary,
            'word_counts': self.word_counts,
            'document_index': self.document_index
        }

        with open(tag + 'data.pickle', 'wb') as f:
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
            
        np.save(tag + 'data.X.npy', self.X, allow_pickle=True)
        
    def load(self, tag='vec.'):
        corpus = None
        with open(tag + 'data.pickle', 'rb') as f:
            data = pickle.load(f)
            
        self.vectorizer = data["vectorizer"]
        self.vocabulary = data["vocabulary"]
        self.word_counts = data["word_counts"]
        self.document_index = data["document_index"]
        
        self.X = np.load(tag + 'data.X.npy', allow_pickle=True)
        
    def _document_index(self):
        metadata = self.corpus.get_index()
        df = pd.DataFrame([ x.__dict__ for x in metadata ], columns=metadata[0].__dict__.keys())
        df['document_id'] = list(df.index)
        return df

    def sum_by_attribute(self, column):

        df = self.document_index
        min_value, max_value = df[column].min(), df[column].max()

        Y = np.zeros(((max_value - min_value) + 1, X.shape[1]))

        for i in range(0, Y.shape[0]):
            
            indices = list((df.loc[df[column] == min_value + i].index))
            if len(indices) > 0:
                Y[i,:] = X[indices,:].sum(axis=0)

        return Y

    def normalize(self, X):
        return sklearn.preprocessing.normalize(X, axis=1, norm='l1')