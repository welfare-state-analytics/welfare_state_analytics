# -*- coding: utf-8 -*-
import logging
import os
import re
import string
import glob

import nltk

import westac.common.zip_utility as zip_utility

logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)

ALPHABETIC_LOWER_CHARS = string.ascii_lowercase + "åäöéàáâãäåæèéêëîïñôöùûÿ"
ALPHABETIC_CHARS = set(ALPHABETIC_LOWER_CHARS + ALPHABETIC_LOWER_CHARS.upper())
SYMBOLS_CHARS = set("'\\¢£¥§©®°±øæç•›€™").union(set('!"#$%&\'()*+,./:;<=>?@[\\]^_`{|}~'))
SYMBOLS_TRANSLATION = dict.fromkeys(map(ord, SYMBOLS_CHARS), None)

def read_textfile(filename):
    with open(filename, 'rb') as f:
        try:
            data = f.read()
            content = data #.decode('utf-8')
        except UnicodeDecodeError as _:
            print('UnicodeDecodeError: {}'.format(filename))
            #content = data.decode('cp1252')
            raise
        return content

def strip_path_and_extension(filename):

    return os.path.splitext(os.path.basename(filename))[0]

def strip_path_and_add_counter(filename, n_chunk):

    return '{}_{}.txt'.format(os.path.basename(filename), str(n_chunk).zfill(3))

def streamify_text_source(text_source, file_pattern: str='*.txt', as_binary: bool=False):
    """Returns an (file_pattern, text) iterator for `text_source`

    Parameters
    ----------
    text_source : Union[str,List[(str,str)]]
        Filename, folder name or an iterator that returns a (filename, text) stream
    file_pattern : str, optional
        Filter for file exclusion, a patter or a predicate, by default '*.txt'
    as_binary : bool, optional
        Read tex as binary (unicode) data, by default False

    Returns
    -------
    Iterable[Tuple[str,str]]
        A stream of filename, text tuples
    """
    if isinstance(text_source, str):

        if os.path.isfile(text_source):

            if text_source.endswith(".zip"):
                return zip_utility.ZipReader(text_source, pattern=file_pattern, as_binary=as_binary)

            return ((text_source, read_textfile(text_source)),)

        if os.path.isdir(text_source):

            return (
                (filename, read_textfile(filename))
                    for filename in glob.glob(os.path.join(text_source, file_pattern))
            )

        return (('document', x) for x in [text_source])

    return text_source

def basename(path):
    return os.path.splitext(os.path.basename(path))[0]

def remove_empty_filter():
    return lambda t: ( x for x in t if x != '' )

hyphen_regexp = re.compile(r'\b(\w+)-\s*\r?\n\s*(\w+)\b', re.UNICODE)

def remove_hyphens(text):
    result = re.sub(hyphen_regexp, r"\1\2\n", text)
    return result

def has_alpha_filter():
    return lambda tokens: ( x for x in tokens if any(map(lambda x: x.isalpha(), x)) )

def only_alphabetic_filter():
    return lambda tokens: (x for x in tokens if any(c in x for c in ALPHABETIC_CHARS))

def remove_stopwords(language_or_stopwords='swedish', extra_stopwords=None):
    if isinstance(language_or_stopwords, str):
        stopwords = set(nltk.corpus.stopwords.words(language_or_stopwords))
    else:
        stopwords = set(language_or_stopwords or {})
    stopwords = stopwords.union(set(extra_stopwords or {}))
    return lambda tokens: (x for x in tokens if x not in stopwords) # pylint: disable=W0601,E0602

def min_chars_filter(n_chars=3):
    return lambda tokens: (x for x in tokens if len(x) >= n_chars)

def max_chars_filter(n_chars=3):
    return lambda tokens: (x for x in tokens if len(x) <= n_chars)

def lower_transform():
    return lambda tokens: map(lambda y: y.lower(), tokens)

def upper_transform():
    return lambda tokens: map(lambda y: y.upper(), tokens)

def remove_numerals():
    return lambda tokens: (x for x in tokens if not x.isnumeric())

def remove_symbols():
    return lambda tokens: (x.translate(SYMBOLS_TRANSLATION) for x in tokens)

def remove_accents():
    return lambda tokens: (x.translate(SYMBOLS_TRANSLATION) for x in tokens)
