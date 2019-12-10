# -*- coding: utf-8 -*-
import os
import zipfile
import glob
import logging
import fnmatch

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

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
                        content = text_file.read().decode('utf8')\
                            .replace('-\r\n', '').replace('-\n', '')
                        yield os.path.basename(filename), content

class ZipReader(object):

    def __init__(self, zip_archive, pattern, filenames=None):
        self.zip_archive = zip_archive
        self.pattern = pattern
        self.archive_filenames = self.get_archive_filenames(pattern)
        self.filenames = filenames or self.archive_filenames

    def get_archive_filenames(self, pattern):
        with zipfile.ZipFile(self.zip_archive) as zf:
            filenames = zf.namelist()
        return [ name for name in filenames if fnmatch.fnmatch(name, pattern) ]

    def __iter__(self):

        with zipfile.ZipFile(self.zip_archive) as zip_file:
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
