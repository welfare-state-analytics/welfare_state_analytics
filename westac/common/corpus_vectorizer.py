
import pandas as pd
import numpy as np
import sklearn.preprocessing

from sklearn.feature_extraction.text import CountVectorizer
from westac.common import vectorized_corpus

class CorpusVectorizer():

    def __init__(self, **kwargs):
        self.vectorizer = None
        self.kwargs = kwargs
        self.tokenizer = lambda x: x.split()

    def fit_transform(self, corpus):

        def text_iterator(x):
            n_documents = 0
            for meta, tokens in x.documents():
                n_documents += 1
                print("{} ({})".format(meta, n_documents))
                yield ' '.join(tokens)

        #texts = (' '.join(tokens) for _, tokens in corpus.documents())
        texts = text_iterator(corpus)

        #https://github.com/scikit-learn/scikit-learn/blob/1495f6924/sklearn/feature_extraction/text.py#L1147
        self.vectorizer = CountVectorizer(tokenizer=self.tokenizer, **self.kwargs)

        bag_term_matrix = self.vectorizer.fit_transform(texts)
        token2id = self.vectorizer.vocabulary_
        document_index = self._document_index(corpus)

        v_corpus = vectorized_corpus.VectorizedCorpus(bag_term_matrix, token2id, document_index)

        return v_corpus

    def _document_index(self, corpus):
        """ Groups matrix by vales in column summing up all values in each category
        """
        metadata = corpus.get_metadata()
        df = pd.DataFrame([ x.__dict__ for x in metadata ], columns=metadata[0].__dict__.keys())
        df['document_id'] = list(df.index)
        return df

