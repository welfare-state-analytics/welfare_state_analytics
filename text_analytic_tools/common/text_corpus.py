import os
import pandas as pd
import numpy as np
import nltk
import gensim
import zipfile
import fnmatch
import logging
import re
import typing
import collections
from itertools import chain

from gensim.corpora.textcorpus import TextCorpus

logger = logging.getLogger(__name__)

HYPHEN_REGEXP = re.compile(r'\b(\w+)-\s*\r?\n\s*(\w+)\b', re.UNICODE)

def build_vocab(corpus):
    ''' Iterates corpus and add distict terms to vocabulary '''
    logger.info('Builiding vocabulary...')
    token2id = collections.defaultdict()
    token2id.default_factory = token2id.__len__
    for doc in corpus:
        for term in doc:
            token2id[term] # pylint: disable=pointless-statement
    logger.info('Vocabulary of size {} built.'.format(len(token2id)))
    return token2id

def dehyphen(text):
    result = re.sub(HYPHEN_REGEXP, r"\1\2\n", text)
    return result

def list_archive_files(archivename, pattern):
    px = lambda x: pattern.match(x) if isinstance(pattern, typing.re.Pattern) else fnmatch.fnmatch(x, pattern)
    with zipfile.ZipFile(archivename) as zf:
        return [ name for name in zf.namelist() if px(name) ]

class CompressedFileReader:

    def __init__(self, path, pattern='*.txt', itemfilter=None):
        self.path = path
        self.filename_pattern = pattern
        self.archive_filenames = list_archive_files(path, pattern)
        filenames = None
        if itemfilter is not None:
            if isinstance(itemfilter, list):
                filenames = [ x for x in itemfilter if x in self.archive_filenames ]
            elif callable(itemfilter):
                filenames = [ x for x in self.archive_filenames if itemfilter(self.archive_filenames, x) ]
            else:
                assert False
        self.filenames = filenames or self.archive_filenames
        self.iterator = None

    def __iter__(self):
        self.iterator = None
        return self

    def __next__(self):
        if self.iterator is None:
            self.iterator = self.get_iterator()
        return next(self.iterator)

    def get_file(self, filename):

        if filename not in self.filenames:
            yield  os.path.basename(filename), None

        with zipfile.ZipFile(self.path) as zip_file:
            yield os.path.basename(filename), self._read_content(zip_file, filename)

    def get_iterator(self):
        with zipfile.ZipFile(self.path) as zip_file:
            for filename in self.filenames:
                yield os.path.basename(filename), self._read_content(zip_file, filename)

    def _read_content(self, zip_file, filename):
        with zip_file.open(filename, 'r') as text_file:
            content = text_file.read()
            content = gensim.utils.to_unicode(content, 'utf8', errors='ignore')
            content = dehyphen(content)
            return content

class GenericTextCorpus(TextCorpus):

    def __init__(self, stream, dictionary=None, metadata=False, character_filters=None, tokenizer=None, token_filters=None, bigram_transform=False):
        self.stream = stream
        self.filenames = None
        self.documents = None
        self.length = None

        #if 'filenames' in content_iterator.__dict__:
        #    self.filenames = content_iterator.filenames
        #    self.document_names = self._compile_documents()
        #    self.length = len(self.filenames)

        token_filters = self.default_token_filters() + (token_filters or [])

        #if bigram_transform is True:
        #    train_corpus = GenericTextCorpus(content_iterator, token_filters=[ x.lower() for x in tokens ])
        #    phrases = gensim.models.phrases.Phrases(train_corpus)
        #    bigram = gensim.models.phrases.Phraser(phrases)
        #    token_filters.append(
        #        lambda tokens: bigram[tokens]
        #    )

        super(GenericTextCorpus, self).__init__(
            input=True,
            dictionary=dictionary,
            metadata=metadata,
            character_filters=character_filters,
            tokenizer=tokenizer,
            token_filters=token_filters
        )

    def default_token_filters(self):
        return [
            (lambda tokens: [ x.lower() for x in tokens ]),
            (lambda tokens: [ x for x in tokens if any(map(lambda x: x.isalpha(), x)) ])
        ]

    def getstream(self):
        """Generate documents from the underlying plain text collection (of one or more files).
        Yields
        ------
        str
            Document read from plain-text file.
        Notes
        -----
        After generator end - initialize self.length attribute.
        """

        document_infos = []
        for filename, content in self.stream:
            yield content
            document_infos.append({
                'document_name': filename
            })

        self.length = len(document_infos)
        self.documents = pd.DataFrame(document_infos)
        self.filenames = list(self.documents.document_name.values)

    def get_texts(self):
        '''
        This is mandatory method from gensim.corpora.TextCorpus. Returns stream of documents.
        '''
        for document in self.getstream():
            yield self.preprocess_text(document)

    def preprocess_text(self, text):
        """Apply `self.character_filters`, `self.tokenizer`, `self.token_filters` to a single text document.

        Parameters
        ---------
        text : str
            Document read from plain-text file.

        Returns
        ------
        list of str
            List of tokens extracted from `text`.

        """
        for character_filter in self.character_filters:
            text = character_filter(text)

        tokens = self.tokenizer(text)
        for token_filter in self.token_filters:
            tokens = token_filter(tokens)

        return tokens

    def __get_document_info(self, filename):
        return {
            'document_name': filename,
        }

    def ___compile_documents(self):

        document_data = map(self.__get_document_info, self.filenames)

        documents = pd.DataFrame(list(document_data))
        documents.index.names = ['document_id']

        return documents

class SimplePreparedTextCorpus(GenericTextCorpus):
    """Reads content in stream and returns tokenized text. No other processing.
    """
    def __init__(self, source, lowercase=False, itemfilter=None):

        self.reader = CompressedFileReader(source, itemfilter=itemfilter)
        self.filenames = self.reader.filenames
        self.lowercase = lowercase
        source = self.reader
        super(SimplePreparedTextCorpus, self).__init__(source)

    def default_token_filters(self):

        token_filters = [
            (lambda tokens: [ x.strip('_') for x in tokens ]),
        ]

        if self.lowercase:
            token_filters = token_filters + [ (lambda tokens: [ x.lower() for x in tokens ]) ]

        return token_filters

    def preprocess_text(self, text):
        return self.tokenizer(text)

class MmCorpusStatisticsService():

    def __init__(self, corpus, dictionary, language):
        self.corpus = corpus
        self.dictionary = dictionary
        self.stopwords = nltk.corpus.stopwords.words(language[1])
        _ = dictionary[0]

    def get_total_token_frequencies(self):
        dictionary = self.corpus.dictionary
        freqencies = np.zeros(len(dictionary.id2token))
        for document in self.corpus:
            for i, f in document:
                freqencies[i] += f
        return freqencies

    def get_document_token_frequencies(self):
        '''
        Returns a DataFrame with per document token frequencies i.e. "melts" doc-term matrix
        '''
        data = ((document_id, x[0], x[1]) for document_id, values in enumerate(self.corpus) for x in values )
        df = pd.DataFrame(list(zip(*data)), columns=['document_id', 'token_id', 'count'])
        df = df.merge(self.corpus.document_names, left_on='document_id', right_index=True)

        return df

    def compute_word_frequencies(self, remove_stopwords):
        id2token = self.dictionary.id2token
        term_freqencies = np.zeros(len(id2token))
        for document in self.corpus:
            for i, f in document:
                term_freqencies[i] += f
        stopwords = set(self.stopwords).intersection(set(id2token.values()))
        df = pd.DataFrame({
            'token_id': list(id2token.keys()),
            'token': list(id2token.values()),
            'frequency': term_freqencies,
            'dfs':  list(self.dictionary.dfs.values())
        })
        df['is_stopword'] = df.token.apply(lambda x: x in stopwords)
        if remove_stopwords is True:
            df = df.loc[(df.is_stopword==False)]
        df['frequency'] = df.frequency.astype(np.int64)
        df = df[['token_id', 'token', 'frequency', 'dfs', 'is_stopword']].sort_values('frequency', ascending=False)
        return df.set_index('token_id')

    def compute_document_stats(self):
        id2token = self.dictionary.id2token
        stopwords = set(self.stopwords).intersection(set(id2token.values()))
        df = pd.DataFrame({
            'document_id': self.corpus.index,
            'document_name': self.corpus.document_names.document_name,
            'treaty_id': self.corpus.document_names.treaty_id,
            'size': [ sum(list(zip(*document))[1]) for document in self.corpus],
            'stopwords': [ sum([ v for (i,v) in document if id2token[i] in stopwords]) for document in self.corpus],
        }).set_index('document_name')
        df[['size', 'stopwords']] = df[['size', 'stopwords']].astype('int')
        return df

    def compute_word_stats(self):
        df = self.compute_document_stats()[['size', 'stopwords']]
        df_agg = df.agg(['count', 'mean', 'std', 'min', 'median', 'max', 'sum']).reset_index()
        legend_map = {
            'count': 'Documents',
            'mean': 'Mean words',
            'std': 'Std',
            'min': 'Min',
            'median': 'Median',
            'max': 'Max',
            'sum': 'Sum words'
        }
        df_agg['index'] = df_agg['index'].apply(lambda x: legend_map[x]).astype('str')
        df_agg = df_agg.set_index('index')
        df_agg[df_agg.columns] = df_agg[df_agg.columns].astype('int')
        return df_agg.reset_index()

#@staticmethod

class ExtMmCorpus(gensim.corpora.MmCorpus):
    """Extension of MmCorpus that allow TF normalization based on document length.
    """

    @staticmethod
    def norm_tf_by_D(doc):
        D = sum([x[1] for x in doc])
        return doc if D == 0 else map(lambda tf: (tf[0], tf[1]/D), doc)

    def __init__(self, fname):
        gensim.corpora.MmCorpus.__init__(self, fname)

    def __iter__(self):
        for doc in gensim.corpora.MmCorpus.__iter__(self):
            yield self.norm_tf_by_D(doc)

    def __getitem__(self, docno):
        return self.norm_tf_by_D(gensim.corpora.MmCorpus.__getitem__(self, docno))

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

