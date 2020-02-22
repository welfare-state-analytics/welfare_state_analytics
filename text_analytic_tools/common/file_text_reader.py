import os

import text_analytic_tools.utility.file_utility as file_utility

class FileTextReader:

    def __init__(self, path, pattern='*.txt', itemfilter=None, meta_extract=None, compress_whitespaces=True, dehyphen=True):
        self.path = path
        self.is_zip = os.path.isfile(path) # and path.endswith('zip')
        self.filename_pattern = pattern
        self.archive_filenames = file_utility.list_files(path, pattern)
        self.compress_whitespaces = compress_whitespaces
        self.dehyphen = dehyphen
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
        self.metadata = [ file_utility.extract_metadata(x, **meta_extract) for x in self.filenames] if not meta_extract is None else None
        self.metadict = { x.filename: x for x in (self.metadata or [])}

    def __iter__(self):
        self.iterator = None
        return self

    def __next__(self):
        if self.iterator is None:
            self.iterator = self.get_iterator()
        return next(self.iterator)

    def get_file(self, filename):

        if filename not in self.filenames:
            yield os.path.basename(filename), None

        yield self.metadict.get(filename, filename), self.read_content(filename)

    def get_iterator(self):
        for filename in self.filenames:
            yield os.path.basename(filename), self.read_content(filename)

    def read_content(self, filename):
        content = file_utility.read_file(self.path, filename)
        if self.dehyphen:
            content = file_utility.dehyphen(content)
        if self.compress_whitespaces:
            content = file_utility.compress_whitespaces(content)
        return content
