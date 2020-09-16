import os
import pandas as pd
import gensim
import logging

from text_analytic_tools.common.gensim_utility import ExtMmCorpus

logger = logging.getLogger(__name__)

class GenericCorpusSaveLoad():

    def __init__(self, source_folder, lang):

        self.mm_filename = os.path.join(source_folder, 'corpus_{}.mm'.format(lang))
        self.dict_filename = os.path.join(source_folder, 'corpus_{}.dict.gz'.format(lang))
        self.document_index = os.path.join(source_folder, 'corpus_{}_documents.csv'.format(lang))

    def store_as_mm_corpus(self, corpus):

        gensim.corpora.MmCorpus.serialize(self.mm_filename, corpus, id2word=corpus.dictionary.id2token)
        corpus.dictionary.save(self.dict_filename)
        corpus.document_names.to_csv(self.document_index, sep='\t')

    def load_mm_corpus(self, normalize_by_D=False):

        corpus_type = ExtMmCorpus if normalize_by_D else gensim.corpora.MmCorpus
        corpus = corpus_type(self.mm_filename)
        corpus.dictionary = gensim.corpora.Dictionary.load(self.dict_filename)
        corpus.document_names = pd.read_csv(self.document_index, sep='\t').set_index('document_id')

        return corpus

    def exists(self):
        return os.path.isfile(self.mm_filename) and \
            os.path.isfile(self.dict_filename) and \
            os.path.isfile(self.document_index)

