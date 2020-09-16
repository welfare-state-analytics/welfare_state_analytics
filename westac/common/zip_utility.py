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
                        content = text_file.read().decode('utf8') # .replace('-\r\n', '').replace('-\n', '')
                        yield os.path.basename(filename), content

class ZipReader(object):
    """Iterator that returns filename and content for each matching file in archive.
    """
    def __init__(self, zip_path, pattern, filenames=None, as_binary=False):
        """
        Parameters
        ----------
        zip_path : sttr
            [description]
        pattern : str
            [description]
        filenames : List[str], optional
            [description], by default None
        as_binary : bool, optional
            If true then files are opened as `rb` and no decoding, by default False
        """
        self.zip_path = zip_path
        self.pattern = pattern
        self.archive_filenames = self.get_archive_filenames(pattern)
        self.filenames = self.archive_filenames if filenames is None \
                else [ f for f in self.archive_filenames if f in filenames ]
        self.as_binary = as_binary

    def get_archive_filenames(self, pattern):
        """Returns all filenames that matches `pattern` in archive

        Parameters
        ----------
        pattern : str
            File pattern

        Returns
        -------
        List[str]
            List of filenames
        """

        with zipfile.ZipFile(self.zip_path) as zf:
            filenames = sorted(zf.namelist())

        return [ name for name in filenames if fnmatch.fnmatch(name, pattern) ]

    def __iter__(self):

        with zipfile.ZipFile(self.zip_path) as zip_file:

            for filename in self.filenames:

                with zip_file.open(filename, 'r') as text_file:

                    content = text_file.read() if self.as_binary else text_file.read().decode('utf-8')

                yield filename, content

def store_text_to_archive(archive_name, stream):
    """Stores stream of text [(name, tokens), (name, tokens), ..., (name, tokens)] as text files in a new zip-file

    Parameters
    ----------
    archive_name : str
        Target filename
    stream : List[Tuple[str, Union[List[str], str]]]
        Documents [(name, tokens), (name, tokens), ..., (name, tokens)]
    """
    with zipfile.ZipFile(archive_name, 'w', compresslevel=zipfile.ZIP_DEFLATED) as out:

        for (filename, document) in stream:

            data = document if isinstance(document, str) else ' '.join(document)
            out.writestr(filename, data, compresslevel=zipfile.ZIP_DEFLATED)

def read_file(archive_name, filename, as_binary=False):

    with zipfile.ZipFile(archive_name) as zf:

        with zf.open(filename, 'r') as f:

            return f.read() if as_binary else f.read().decode('utf-8')