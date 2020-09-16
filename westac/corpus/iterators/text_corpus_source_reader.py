# -*- coding: utf-8 -*-
import logging

from .corpus_source_reader import CorpusSourceReader

logger = logging.getLogger(__name__)

class TextCorpusSourceReader(CorpusSourceReader):

    def __init__(self, source, transforms, chunk_size=None):
        CorpusSourceReader.__init__(self, source, transforms, chunk_size, pattern='*.txt')
