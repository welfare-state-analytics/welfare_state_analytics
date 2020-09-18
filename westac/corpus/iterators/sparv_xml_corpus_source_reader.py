# -*- coding: utf-8 -*-
import logging

import westac.common.zip_utility as zip_utility

from westac.corpus.sparv.sparv_xml_to_text import SparvXml2Text, XSLT_FILENAME_V3

from .corpus_source_reader import CorpusSourceReader

logger = logging.getLogger(__name__)

DEFAULT_OPTS = dict(
    pos_includes='',
    lemmatize=True,
    chunk_size=None,
    xslt_filename=None,
    delimiter="|",
    append_pos="",
    pos_excludes="|MAD|MID|PAD|"
)

class SparvXmlCorpusSourceReader(CorpusSourceReader):

    def __init__(self,
        source, transforms=None, pos_includes=None, lemmatize=True, chunk_size=None, xslt_filename=None,
        append_pos="", pos_excludes="|MAD|MID|PAD|", version=4):

        self.delimiter = ' '
        tokenize = lambda x: x.split(self.delimiter)

        super(SparvXmlCorpusSourceReader, self).__init__(source, transforms, chunk_size, pattern='*.xml', tokenize=tokenize, as_binary=True)

        self.pos_includes = pos_includes
        self.lemmatize = lemmatize
        self.append_pos = append_pos
        self.pos_excludes = pos_excludes
        self.xslt_filename = XSLT_FILENAME_V3 if version == 3 else xslt_filename
        self.parser = SparvXml2Text(xslt_filename=self.xslt_filename, delimiter=self.delimiter, pos_includes=pos_includes, lemmatize=lemmatize, append_pos=append_pos, pos_excludes=pos_excludes)

    def preprocess(self, content):

        return self.parser.transform(content)

class Sparv3XmlCorpusSourceReader(SparvXmlCorpusSourceReader):

    def __init__(self, source, transforms=None, pos_includes=None, lemmatize=True, chunk_size=None, append_pos="", pos_excludes="|MAD|MID|PAD|"):

        super(Sparv3XmlCorpusSourceReader, self).__init__(
            source,
            transforms=transforms,
            pos_includes=pos_includes,
            lemmatize=lemmatize,
            chunk_size=chunk_size,
            xslt_filename=XSLT_FILENAME_V3,
            append_pos=append_pos,
            pos_excludes=pos_excludes
        )


def sparv_extract_and_store(source, target, **opts):

    stream = SparvXmlCorpusSourceReader(source, **opts)

    zip_utility.store_text_to_archive(target, stream)
