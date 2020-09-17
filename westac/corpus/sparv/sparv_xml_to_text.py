# -*- coding: utf-8 -*-
import os
import logging
from lxml import etree
from io import StringIO

logger = logging.getLogger(__name__)

script_path = os.path.dirname(os.path.abspath( __file__ ))
XSLT_FILENAME = os.path.join(script_path, 'sparv_xml_extract.xslt')
XSLT_FILENAME_V3 = os.path.join(script_path, 'sparv_xml_extract.v3.xslt')

class SparvXml2Text():

    def __init__(self, xslt_filename=None, postags=None, lemmatize=True, delimiter="|", append_pos="", ignores="|MAD|MID|PAD|"):

        self.xslt_filename = xslt_filename or XSLT_FILENAME
        self.postags = self.snuttify(postags) if postags is not None else ''
        self.xslt = etree.parse(self.xslt_filename)  # pylint: disable=I1101
        self.xslt_transformer = etree.XSLT(self.xslt)  # pylint: disable=I1101
        self.delimiter = self.snuttify(delimiter)
        self.lemmatize = lemmatize
        self.ignores = self.snuttify(ignores)
        self.append_pos = self.snuttify(append_pos)

    def transform(self, content):
        xml = etree.XML(content)  # pylint: disable=I1101
        target = "'lemma'" if self.lemmatize is True else "'content'"
        text = self.xslt_transformer(xml, postags=self.postags, delimiter=self.delimiter, target=target, append_pos=self.append_pos, ignores=self.ignores)
        return str(text)

    def read_transform(self, filename):
        xml = etree.parse(filename)  # pylint: disable=I1101
        target = "'lemma'" if self.lemmatize is True else "'content'"
        text = self.xslt_transformer(xml, postags=self.postags, delimiter=self.delimiter, target=target, append_pos=self.append_pos, ignores=self.ignores)
        return str(text)

    def snuttify(self, token):
        if token.startswith("'") and token.endswith("'"):
            return token
        return "'{}'".format(token)
