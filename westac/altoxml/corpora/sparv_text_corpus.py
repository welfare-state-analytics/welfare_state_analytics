# -*- coding: utf-8 -*-

import gensim
import itertools

# TODO Rename class
class SparvTextCorpus(gensim.corpora.TextCorpus):

    '''
    This is a BOW vector corpus based on gensim.corpora.TextCorpus
    '''
    def __init__(self, stream, prune_at=2000000):

        self.dictionary = None
        self.reader = stream
        self.document_length = []
        self.corpus_documents = []
        self.prune_at = prune_at

        super(SparvTextCorpus, self).__init__(input=True)

    def init_dictionary(self, dictionary):
        self.dictionary = gensim.corpora.Dictionary(self.get_texts(), prune_at=self.prune_at)
        if (len(self.dictionary) > 0):
            _ = self.dictionary[0]  # force formation of dictionary.id2token

    def getstream(self):
        '''
        Returns stream of documents.
        Also collects documents' name and length for each pass
        '''
        corpus_documents = []
        document_length = []
        for document_name, document in self.reader:
            corpus_documents.append(document_name)
            document_length.append(len(document))
            yield document
        self.document_length = document_length
        self.corpus_documents = corpus_documents

    def get_texts(self):
        '''
        This is mandatory method from gensim.corpora.TextCorpus. Returns stream of documents.
        '''
        for document in self.getstream():
            yield document

    def get_total_word_count(self):
        # Create the defaultdict: total_word_count
        total_word_count = { word_id: 0 for word_id in self.dictionary.keys() }
        for word_id, word_count in itertools.chain.from_iterable(self):
            total_word_count[word_id] += word_count

        # Create a sorted list from the defaultdict: sorted_word_count
        sorted_word_count = sorted(total_word_count, key=lambda w: w[1], reverse=True)
        return sorted_word_count

    def get_corpus_documents(self, attrib_extractors=None):

        attrib_extractors = attrib_extractors or []

        if len(self.corpus_documents) == 0:
            for _ in self.getstream():
                pass

        print(self.corpus_documents)
        document_ids, document_names = list(zip(*(
            (document_id, document) for document_id, document in enumerate(self.corpus_documents)
        )))

        data = dict(
            document_id=document_ids,
            document=document_names,
            length=self.document_length
        )

        for (n, f) in attrib_extractors:
            data[n] = [ f(x) for x in document_names ]

        return data
