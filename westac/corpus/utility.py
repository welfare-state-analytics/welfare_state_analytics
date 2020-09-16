# -*- coding: utf-8 -*-
import os
import zipfile
import glob
import logging
import fnmatch
import re
import nltk

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

def read_textfile(filename):
    with open(filename, 'rb') as f:
        try:
            data = f.read()
            content = data #.decode('utf-8')
        except UnicodeDecodeError as _:
            print('UnicodeDecodeError: {}'.format(filename))
            #content = data.decode('cp1252')
            raise
        return content

def basename(path):
    return os.path.splitext(os.path.basename(path))[0]

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

class ZipFileIterator(object):

    def __init__(self, pattern, extensions):
        self.pattern = pattern
        self.extensions = extensions

    def __iter__(self):

        for zip_path in glob.glob(self.pattern):
            with zipfile.ZipFile(zip_path) as zip_file:
                filenames = [ name for name in zip_file.namelist() if any(map(name.endswith, self.extensions)) ]
                for filename in filenames:
                    with zip_file.open(filename) as text_file:
                        content = text_file.read().decode('utf8') # .replace('-\r\n', '').replace('-\n', '')
                        yield os.path.basename(filename), content

class ZipReader(object):

    def __init__(self, zip_path, pattern, filenames=None):
        self.zip_path = zip_path
        self.pattern = pattern
        self.archive_filenames = self.get_archive_filenames(pattern)
        self.filenames = filenames or self.archive_filenames

    def get_archive_filenames(self, pattern):
        with zipfile.ZipFile(self.zip_path) as zf:
            filenames = zf.namelist()
        return [ name for name in filenames if fnmatch.fnmatch(name, pattern) ]

    def __iter__(self):

        with zipfile.ZipFile(self.zip_path) as zip_file:
            for filename in self.filenames:
                if filename not in self.archive_filenames:
                    continue
                with zip_file.open(filename, 'r') as text_file:
                    content = text_file.read().decode('utf-8')
                    yield filename, content

def store_documents_to_archive(archive_name, documents):
    '''
    Stores documents [(name, tokens), (name, tokens), ..., (name, tokens)] as textfiles in a new zip-files
    '''
    with zipfile.ZipFile(archive_name, 'w', zipfile.ZIP_DEFLATED) as xip:
        for (filename, document) in documents:
            xip.writestr(filename, ' '.join(document))
