import gensim
import zipfile
import fnmatch
import re
import types
import typing
import os
import logging
import time
import numpy as np
import itertools

HYPHEN_REGEXP = re.compile(r'\b(\w+)-\s*\r?\n\s*(\w+)\b', re.UNICODE)

def noop(x=None): pass

def setup_logger(logger=None, filename=None, level=logging.DEBUG):
    '''
    Setup logging of import messages to both file and console
    '''
    if logger is None:
        logger = logging.getLogger("westac")

    filename = filename or 'westac_{}.log'.format(time.strftime("%Y%m%d"))
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

def nth(iterable, n, default=None):
    "Returns the nth item or a default value"
    return next(itertools.islice(iterable, n, None), default)

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
class IntStepper():

    def __init__(self, min_value, max_value, step=1, callback=None, value=None, data=None):
        self.min = min_value
        self.max = max_value
        self.step = step
        self.value = value or min_value
        self.data = data or {}
        self.callback = callback

    def trigger(self):
        if callable(self.callback):
            self.callback(self.value, self.data)
        return self.value

    def next(self):
        self.value = self.min + (self.value - self.min + self.step) % (self.max - self.min)
        return self.trigger()

    def previous(self):
        self.value = self.min + (self.value - self.min - self.step) % (self.max - self.min)
        return self.trigger()

    def reset(self):
        self.value = self.min
        return self.trigger()
