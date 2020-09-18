# -*- coding: utf-8 -*-
import glob
import logging
import os
import pathlib

import gensim

from westac.common.zip_utility import ZipReader
from westac.corpus.sparv.sparv_xml_to_text import SparvXml2Text
from westac.corpus import utility

logger = logging.getLogger(__name__)

def strip_path_and_extension(filename):

    return os.path.splitext(os.path.basename(filename))[0]

def strip_path_and_add_counter(filename, n_chunk):

    return '{}_{}.txt'.format(os.path.basename(filename), str(n_chunk).zfill(3))

class CorpusSourceReader():
    """Reads a text corpus from `source` and applies given transforms.
    Derived classes can override `preprocess` as an initial step before transforms are applied.
    The `preprocess` is applied on the entire document, and the transforms on each token.
    The `preprocess can be e.g. extracting text from an XML file (see derived class SParvXmlCorpusSourceReader)
    """
    def __init__(self, source=None, transforms=None, chunk_size=None, pattern=None, tokenize=None, as_binary=False):
        """
        Parameters
        ----------
        source : str
            Source can be either a Zip archive, a single text file or a directory.
        transforms : fn, optional
            List of transforms that in sequence are applied to each token
        chunk_size : int, optional
            Optional chunking in equal sizes of each document, by default None
        pattern : str, optional
            Pattern of files to include in the reprocess, by default None
        tokenize : Func[], optional
            Text tokenize function, by default None
        """
        self.pattern = pattern or ''

        if isinstance(source, str):
            if os.path.isfile(source):
                if source.endswith(".zip"):
                    self.source = ZipReader(source, pattern=pattern, as_binary=as_binary)
                else:
                    self.source = ((source, utility.read_textfile(source)),)
            elif os.path.isdir(source):
                self.source = (
                    (filename, utility.read_textfile(filename))
                        for filename in glob.glob(os.path.join(source, pattern))
                )
            else:
                self.source = (('document', x) for x in [source])
        elif isinstance(source, (list,)):
            self.source = source # ((x,y) for x, y in source)
        else:
            self.source = source

        self.chunk_size = chunk_size
        self.tokenize = tokenize or gensim.utils.tokenize
        self.transforms = [ utility.remove_empty_filter() ] + (transforms or [])

    def get_iterator(self):
        return (
            (document_name, tokens)
                for (filename, content) in self.source
                    for document_name, tokens in self.process_document(filename, content)
        )

    def apply_transforms(self, tokens):
        for ft in self.transforms:
            tokens = ft(tokens)
        return tokens

    def preprocess(self, content):
        return content

    def process_document(self, filename, content):
        """Process a document and returns tokenized text, and optionally chunk counter if split chunks

        Parameters
        ----------
        content : str
            The actual text.

        Yields
        -------
        Tuple[str,List[str]]
            Filename and tokens
        """
        content = self.preprocess(content)
        tokens = self.tokenize(content)

        for ft in self.transforms:

            tokens = ft(tokens)

        if self.chunk_size is None:

            stored_name = strip_path_and_extension(filename) + '.txt'

            yield stored_name, tokens

        else:

            tokens = list(tokens)

            for n_chunk, i in enumerate(range(0, len(tokens), self.chunk_size)):

                stored_name = '{}_{}.txt'.format(strip_path_and_extension(filename), str(n_chunk+1).zfill(3))

                yield stored_name, tokens[i: i + self.chunk_size]

    def __iter__(self):
        """Iterates documents and returns filename and processed tokens for each document/chunk

        Yields
        -------
        Tuple[str,List[str]]
            Filename and tokens
        """

        for (filename, content) in self.source:

            for document_name, tokens in self.process_document(filename, content):

                yield document_name, tokens
