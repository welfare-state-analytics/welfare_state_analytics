import os
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer

from westac.corpus import text_corpus
from westac.corpus import vectorized_corpus
from westac.corpus import file_text_reader

import logging

logger = logging.getLogger("corpus_vectorizer")

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

def generate_corpus(filename, output_folder, **kwargs):

    if not os.path.isfile(filename):
        logger.error('no such file: {}'.format(filename))
        return

    dump_tag = '{}_{}_{}_{}'.format(
        os.path.basename(filename).split('.')[0],
        'L{}'.format(kwargs.get('min_len', 0)),
        '-N' if kwargs.get('numerals', False) else '+N',
        '-S' if kwargs.get('symbols', False) else '+S',
    )

    if vectorized_corpus.VectorizedCorpus.dump_exists(dump_tag):
        logger.info('removing existing result files...')
        os.remove(os.path.join(output_folder, '{}_vector_data.npy'.format(dump_tag)))
        os.remove(os.path.join(output_folder, '{}_vectorizer_data.pickle'.format(dump_tag)))

    logger.info('Creating new corpus...')
    reader = file_text_reader.FileTextReader(
        filename, meta_extract=kwargs.get("meta_extract"),
        compress_whitespaces=True,
        dehyphen=True,
        pattern=kwargs.get("pattern", "*.txt")
    )
    corpus = text_corpus.ProcessedCorpus(reader, **kwargs)

    logger.info('Creating document-term matrix...')
    vectorizer = CorpusVectorizer()
    v_corpus = vectorizer.fit_transform(corpus)

    logger.info('Saving data matrix...')
    v_corpus.dump(tag=dump_tag, folder=output_folder)
