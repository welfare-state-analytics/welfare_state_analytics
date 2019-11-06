import gensim
import zipfile
import fnmatch
import re
import types
import typing
import os
import logging
import time

HYPHEN_REGEXP = re.compile(r'\b(\w+)-\s*\r?\n\s*(\w+)\b', re.UNICODE)

def setup_logger(logger=None, filename=None, level=logging.DEBUG):
    '''
    Setup logging of import messages to both file and console
    '''
    if logger is None:
        logger = logging.getLogger("westac")

    filename = filename or 'westac_{}.log'.format(time.strftime("%Y%m%d-%H%M%S"))
    logger.handlers = []

    logger.setLevel(level)
    formatter = logging.Formatter('%(message)s')

    fh = logging.FileHandler(filename)
    fh.setLevel(level)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger

def flatten(l):
    return [ x for ws in l for x in ws]

def dehyphen(text: str):
    result = re.sub(HYPHEN_REGEXP, r"\1\2\n", text)
    return result

def list_files(path_name, pattern):
    px = lambda x: pattern.match(x) if isinstance(pattern, typing.re.Pattern) else fnmatch.fnmatch(x, pattern)
    if os.path.isdir(path_name):
        files = [ f for f in os.listdir(path_name) if os.path.isfile(os.path.join(path_name, f)) ]
    else:
        with zipfile.ZipFile(path_name) as zf:
            files = zf.namelist()

    return [ name for name in files if px(name) ]

def compress_whitespaces(text):
    result = re.sub(r'\s+', ' ', text).strip()
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
        with open(os.path.join(path, filename), 'r') as file:
            content = file.read()
    else:
        with zipfile.ZipFile(path) as zf:
            with zf.open(filename, 'r') as file:
                content = file.read()
    content = gensim.utils.to_unicode(content, 'utf8', errors='ignore')
    return content
