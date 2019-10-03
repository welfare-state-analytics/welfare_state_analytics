import gensim
import zipfile
import fnmatch
import re
import types
import typing
import os

HYPHEN_REGEXP = re.compile(r'\b(\w+)-\s*\r?\n\s*(\w+)\b', re.UNICODE)

def dehyphen(text):
    result = re.sub(HYPHEN_REGEXP, r"\1\2\n", text)
    return result

def list_files(path, pattern):
    px = lambda x: pattern.match(x) if isinstance(pattern, typing.re.Pattern) else fnmatch.fnmatch(x, pattern)
    if os.path.isdir(path):
         files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
    else:
        with zipfile.ZipFile(path) as zf:
            files = zf.namelist()
        
    return [ name for name in files if px(name) ]
    
def compress_whitespaces(text):
    result = re.sub('\s+', ' ', text).strip()
    return result
        
def extract_metadata(filename, **kwargs):
    params = { x: None for x in kwargs.keys()}
    meta =  types.SimpleNamespace(filename=filename, **params)
    for k,r in kwargs.items():
        if r is None:
            continue
        if isinstance(r, str): # typing.re.Pattern):
            m = re.match(r, filename)
            if m is not None:
                v = m.groups()[0]
                meta.__setattr__(k, int(v) if v.isnumeric() else v)
    return meta

def read_file(path, filename):
    if os.path.isdir(path):
        print('isdir')
        with open(os.path.join(path, filename), 'r') as file:
            content = file.read()
    else:
        with zipfile.ZipFile(path) as zf:
            with zf.open(filename, 'r') as file:
                content = file.read()
    content = gensim.utils.to_unicode(content, 'utf8', errors='ignore')
    return content
        
class TextFilesReader:

    def __init__(self, path, pattern='*.txt', itemfilter=None, meta_extract=None, compress_whitespaces=True, dehyphen=True):
        self.path = path
        self.is_zip = os.path.isfile(path) # and path.endswith('zip')
        self.filename_pattern = pattern
        self.archive_filenames = list_files(path, pattern)
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
        self.metadata = [ extract_metadata(x, **meta_extract) for x in self.filenames] if not meta_extract is None else None
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
        content = read_file(self.path, filename)
        if self.dehyphen:
            content = dehyphen(content)
        if self.compress_whitespaces:
            content = compress_whitespaces(content)
        return content

