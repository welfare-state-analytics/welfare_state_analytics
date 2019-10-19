# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %load_ext autoreload
# %autoreload 2

import unittest
import numpy as np
import pandas as pd

from westac.common import corpus_vectorizer
from westac.common import text_corpus
from westac.tests.utils  import create_text_files_reader

unittest.main(argv=['first-arg-is-ignored'], exit=False)

import scipy
from scipy.cluster.hierarchy import dendrogram, linkage # pylint: disable=unused-import
from matplotlib import pyplot as plt # pylint: disable=unused-import

class Test_ChiSquare(unittest.TestCase):

    def setUp(self):
        pass

    def create_reader(self):
        meta_extract = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
        reader = create_text_files_reader(meta_extract=meta_extract, compress_whitespaces=True, dehyphen=True)
        return reader

    def create_corpus(self):
        reader = self.create_reader()
        kwargs = dict(isalnum=True, to_lower=True, deacc=False, min_len=2, max_len=None, numerals=False)
        corpus = text_corpus.ProcessedCorpus(reader, **kwargs)
        return corpus

    def test_chisquare(self):

        corpus = self.create_corpus()
        vectorizer = corpus_vectorizer.CorpusVectorizer()

        v_corpus = vectorizer\
            .fit_transform(corpus)\
            .group_by_year()\
            .normalize()\
            .slice_by_n_count(0)

        X2 = scipy.stats.chisquare(v_corpus.bag_term_matrix, f_exp=None, ddof=0, axis=0) # pylint: disable=unused-variable

        # Use X2 so select top 500 words... (highest Power-Power_divergenceResult)
        # Ynw = largest_by_chisquare()
        #print(Ynw)

        linked = linkage(Ynw.T, 'ward') # pylint: disable=unused-variable
        #print(linked)

        labels = [ v_corpus.id2token[x] for x in indices ] # pylint: disable=unused-variable

        #plt.figure(figsize=(24, 16))
        #dendrogram(linked, orientation='top', labels=labels, distance_sort='descending', show_leaf_counts=True)
        #plt.show()

        results = None
        expected = None

        self.assertEqual(expected, results)

# -

def plot_dists(v_corpus):
    df = pd.DataFrame(v_corpus.bag_term_matrix.toarray(), columns=list(v_corpus.get_feature_names()))
    df['year'] = df.index + 45
    df = df.set_index('year')
    df['year'] =  pd.Series(df.index).apply(lambda x: v_corpus.document_index[x][0])
    df[['krig']].plot() #.loc[df["000"]==49]


# unittest.main(argv=['first-arg-is-ignored'], exit=False)
