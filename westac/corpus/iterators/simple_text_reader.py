import os
from typing import Any

from westac.corpus.text_transformer import TextTransformer, TRANSFORMS


class FolderOrZipTextReader:
    """Reads text files matching `pattern` from `path` which is either a folder or a zip-archive
    """
    def __init__(self,
        source_path: str,
        filename_pattern: str='*.txt',
        filename_filter: Any=None,
        meta_extract=None,
        fix_whitespaces=True,
        fix_hyphenation=True
    ):
        self.path = source_path
        self.filename_pattern = filename_pattern
        self.archive_filenames = utility.list_files(source_path, filename_pattern)

        self.transformer = self.create_transformer(fix_whitespaces, fix_hyphenation)

        filenames = None
        if filename_filter is not None:
            if isinstance(filename_filter, list):
                filenames = [ x for x in filename_filter if x in self.archive_filenames ]
            elif callable(filename_filter):
                filenames = [ x for x in self.archive_filenames if filename_filter(self.archive_filenames, x) ]
            else:
                assert False
        self.filenames = filenames or self.archive_filenames

        self.iterator = None
        self.metadata = [ utility.extract_metadata(x, **meta_extract)
            for x in self.filenames] if not meta_extract is None else None
        self.metadict = { x.filename: x for x in (self.metadata or [])}

        self.transforms = TextTransformer()\
            .and(TRANSFORMS.fix_unicode)\
            .and(TRANSFORMS.fix_whitespaces, condition=fix_whitespaces)\
            .and(TRANSFORMS.fix_hyphenation, condition=fix_hyphenation)

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

        content = self.transformer.transform(utility.read_file(self.path, filename))

        return content
