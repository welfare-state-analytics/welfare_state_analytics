# -*- coding: utf-8 -*-
import os
import re
import gensim
import nltk
import logging
import glob
from . alto_xml_parser import AltoXmlToText

logger = logging.getLogger(__name__)

def basename(path):
    return os.path.splitext(os.path.basename(path))[0]

def read_textfile(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        try:
            data = f.read()
            content = data #.decode('utf-8')
        except UnicodeDecodeError as _:
            print('UnicodeDecodeError: {}'.format(filename))
            #content = data.decode('cp1252')
            raise
        yield (filename, content)

def remove_empty_filter():
    return lambda t: [ x for x in t if x != '' ]

hyphen_regexp = re.compile(r'\b(\w+)-\s*\r?\n\s*(\w+)\b', re.UNICODE)

def remove_hyphens(text):
    result = re.sub(hyphen_regexp, r"\1\2\n", text)
    return result

def has_alpha_filter():
    return lambda tokens: [ x for x in tokens if any(map(lambda x: x.isalpha(), x)) ]

def stopwords_filter(language):
    stopwords = nltk.corpus.stopwords.words(language)
    return  lambda tokens: [ x for x in tokens if x not in stopwords ]

def min_token_size_filter(min_token_size=3):
    return lambda tokens: [ x for x in tokens if len(x) >= min_token_size ]

def lower_case_transform():
    return lambda _tokens: list(map(lambda y: y.lower(), _tokens))

class CorpusSourceReader():

    def __init__(self, source=None, transforms=None, chunk_size=None, pattern=None, tokenize=None):

        self.pattern = pattern or ''

        if isinstance(source, str):
            if os.path.isfile(source):
                self.source = read_textfile(source)
            elif os.path.isdir(source):
                self.source = (read_textfile(filename) for filename in glob.glob(os.path.join(source, pattern)))
            else:
                self.source = (('document', x) for x in [source])
        elif isinstance(source, (list,)):
            self.source = source # ((x,y) for x, y in source)
        else:
            self.source = source

        self.chunk_size = chunk_size
        self.tokenize = tokenize or gensim.utils.tokenize
        self.transforms = [ remove_empty_filter() ] + (transforms or [])

    def apply_transforms(self, tokens):
        for ft in self.transforms:
            tokens = ft(tokens)
        return tokens

    def preprocess(self, content):
        return content

    def document_iterator(self, content):

        content = self.preprocess(content)
        tokens = self.tokenize(content)

        for ft in self.transforms:
            tokens = ft(tokens)

        if self.chunk_size is None:
            yield 1, tokens
        else:
            chunk_counter = 0
            for i in range(0, len(tokens), self.chunk_size):
                chunk_counter += 1
                yield chunk_counter, tokens[i: i + self.chunk_size]

    def __iter__(self):

        for (filename, content) in self.source:  # self.documents_iterator(self.source):
            for chunk_counter, chunk_tokens in self.document_iterator(content):
                if len(chunk_tokens) == 0:
                    continue
                yield '{}_{}.txt'.format(basename(filename), str(chunk_counter).zfill(2)), chunk_tokens

class SparvCorpusSourceReader(CorpusSourceReader):

    def __init__(self, source, transforms=None, postags=None, lemmatize=True, chunk_size=None, xslt_filename=None, deliminator="|", append_pos="", ignores="|MAD|MID|PAD|"):

        tokenize = lambda x: str(x).split(deliminator)

        super(SparvCorpusSourceReader, self).__init__(source, transforms, chunk_size, pattern='*.xml', tokenize=tokenize)

        self.postags = postags
        self.lemmatize = lemmatize
        self.append_pos = append_pos
        self.ignores = ignores

        self.alto_parser = AltoXmlToText(xslt_filename=xslt_filename, postags=postags, lemmatize=lemmatize, append_pos=append_pos, ignores=ignores)

    def preprocess(self, content):
        return self.alto_parser.transform(content)

class TextCorpusSourceReader(CorpusSourceReader):

    def __init__(self, source, transforms, chunk_size=None):
        CorpusSourceReader.__init__(self, source, transforms, chunk_size, pattern='*.txt')
