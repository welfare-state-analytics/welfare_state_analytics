import datetime
import itertools
import logging
import os
import re
import time

HYPHEN_REGEXP = re.compile(r'\b(\w+)-\s*\r?\n\s*(\w+)\b', re.UNICODE)

def noop(_=None):
    pass

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

def nth(iterable, n: int, default=None):
    "Returns the nth item or a default value"
    return next(itertools.islice(iterable, n, None), default)

def now_timestamp():
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S')

def timestamp(format_string=None):
    """ Add timestamp to string that must contain exacly one placeholder """
    tz = now_timestamp()
    return tz if format_string is None else format_string.format(tz)

def suffix_filename(filename, suffix):
    output_path, output_file = os.path.split(filename)
    output_base, output_ext = os.path.splitext(output_file)
    suffixed_filename = os.path.join(output_path, f"{output_base}_{suffix}{output_ext}")
    return suffixed_filename

def replace_extension(filename, extension):
    if filename.endswith(extension):
        return filename
    base, _ = os.path.splitext(filename)
    return f"{base}{'' if extension.startswith('.') else '.'}{extension}"

def timestamp_filename(filename):
    return suffix_filename(filename, now_timestamp())

def flatten(l):
    return [ x for ws in l for x in ws]

def isint(s) -> bool:
    try:
        int(s)
        return True
    except (TypeError, ValueError) as _:
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

def fix_hyphenation(text: str) -> str:
    result = re.sub(HYPHEN_REGEXP, r"\1\2\n", text)
    return result

def fix_whitespaces(text: str) -> str:
    result = re.sub(r'\s+', ' ', text).strip()
    return result
