import os

import text_analytic_tools.utility.file_utility as utility

class SimpleTextReader:

    def __init__(self, path, pattern='*.txt', itemfilter=None, filename_fields=None, compress_whitespaces=True, dehyphen=True):
        self.path = path
        self.is_zip = os.path.isfile(path) # and path.endswith('zip')
        self.filename_pattern = pattern
        self.archive_filenames = utility.list_files(path, pattern)
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
        self.metadata = [ utility.extract_metadata(x, **filename_fields) for x in self.filenames] if not filename_fields is None else None
        self.metadict = { x.filename: x for x in (self.metadata or [])}

    def __iter__(self):
        return self

    def __next__(self):
        if self.iterator is None:
            self.iterator = self._create_iterator()
        try:
            return next(self.iterator)
        except StopIteration:
            self.iterator = None
            raise

    def get_file(self, filename):

        if filename not in self.filenames:
            yield os.path.basename(filename), None

        yield self.metadict.get(filename, filename), self.read_content(filename)

    def _create_iterator(self):
        return (
            (os.path.basename(filename), self.read_content(filename))
                for filename in self.filenames
        )

    def read_content(self, filename):
        content = utility.read_file(self.path, filename)
        if self.dehyphen:
            content = utility.dehyphen(content)
        if self.compress_whitespaces:
            content = utility.compress_whitespaces(content)
        return content
