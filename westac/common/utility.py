import gensim
import zipfile
import fnmatch
import re
import types
import typing
import os
import logging
import time
import datetime
import numpy as np
import itertools

HYPHEN_REGEXP = re.compile(r'\b(\w+)-\s*\r?\n\s*(\w+)\b', re.UNICODE)

def noop(x=None): pass

def setup_logger(logger=None, to_file=False, filename=None, level=logging.DEBUG):
    '''
    Setup logging of import messages to both file and console
    '''
    if logger is None:
        logger = logging.getLogger("westac")

    logger.handlers = []

    logger.setLevel(level)
    formatter = logging.Formatter('%(message)s')

    if to_file is True or filename is not None:
        if filename is None:
            filename = 'westac_{}.log'.format(time.strftime("%Y%m%d"))
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

def timestamp(format_string=None):
    """ Add timestamp to string that must contain exacly one placeholder """
    tz = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    return tz if format_string is None else format_string.format(tz)

def flatten(l):
    return [ x for ws in l for x in ws]

def isint(s):
    try:
        int(s)
        return True
    except:
        return False

def project_series_to_range(series, low, high):
    """Project a sequence of elements to a range defined by (low, high)"""
    norm_series = series / series.max()
    return norm_series.apply(lambda x: low + (high - low) * x)

def project_to_range(value, low, high):
    """Project a singlevalue to a range (low, high)"""
    return low + (high - low) * value

def project_values_to_range(values, low, high):
    w_max = max(values)
    return [ low + (high - low) * (x / w_max) for x in  values ]

def clamp_values(values, low_high):
    """Clamps value to supplied interval."""
    mw = max(values)
    return [ project_to_range(w / mw, low_high[0], low_high[1]) for w in values ]

def extend(target, *args, **kwargs):
    """Returns dictionary 'target' extended by supplied dictionaries (args) or named keywords

    Parameters
    ----------
    target : dict
        Default dictionary (to be extended)

    args: [dict]
        Optional. List of dicts to use when updating target

    args: [key=value]
        Optional. List of key-value pairs to use when updating target

    Returns
    -------
    [dict]
        Target dict updated with supplied dicts/key-values.
        Multiple keys are overwritten inorder of occrence i.e. keys to right have higher precedence

    """

    target = dict(target)
    for source in args:
        target.update(source)
    target.update(kwargs)
    return target

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
    """Extracts metadata from filename

    Parameters
    ----------
    filename : str
        Filename (basename)
    kwargs: key=extractor list

    Returns
    -------
    SimpleNamespace
        Each key in kwargs is set as a property in the returned instance.
        The extractor must be either a regular expression that extracts the single value
        or a callable function that given the filename return corresponding value.

    """
    params = { x: None for x in kwargs.keys()}
    meta =  types.SimpleNamespace(filename=filename, **params)
    for k,r in kwargs.items():

        if r is None:
            continue

        if callable(r):
            v = r(filename)
            meta.__setattr__(k, int(v) if v.isnumeric() else v)

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
