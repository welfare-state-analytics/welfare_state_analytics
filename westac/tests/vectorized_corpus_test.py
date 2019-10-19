import os
import unittest
import types
import pandas as pd
import numpy as np

from westac.common import corpus_vectorizer
from westac.common import text_corpus

from westac.tests.utils  import create_text_files_reader

flatten = lambda l: [ x for ws in l for x in ws]

class Test_VectorizedCorpus(unittest.TestCase):

    def setUp(self):
        pass

