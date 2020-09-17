import fnmatch
import os
import re
import typing
import zipfile

import gensim

HYPHEN_REGEXP = re.compile(r'\b(\w+)-\s*\r?\n\s*(\w+)\b', re.UNICODE)

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
            yield os.path.basename(filename), self.read_content(zip_file, filename)

    def get_iterator(self):
        with zipfile.ZipFile(self.path) as zip_file:
            for filename in self.filenames:
                yield os.path.basename(filename), self.read_content(zip_file, filename)

    def read_content(self, zip_file, filename):
        with zip_file.open(filename, 'r') as text_file:
            content = text_file.read()
            content = gensim.utils.to_unicode(content, 'utf8', errors='ignore')
            content = dehyphen(content)
            return content
