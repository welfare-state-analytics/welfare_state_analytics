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
import pandas as pd
import numpy as np
import scipy
import os
import nltk

from westac.common import corpus_vectorizer
from westac.common import utility
from westac.common import text_corpus

# df = pd.read_excel('./data/year+text_window.xlsx')
# df.to_csv('./data/year+text_window.txt', sep='\t')
# -

def load_text_windows(filename):

    filepath = os.path(filename)

    if not os.path.isdir(filepath):
        raise FileNotFoundError("Path {filepath} does not exist!")

    filebase = os.path.basename(filename).split('.')[0]
    textfile = os.path.join(filepath, filebase + '.txt')

    if not os.path.isfile(textfile):
        df = pd.read_excel('./data/year+text_window.xlsx')
        df.to_csv('./data/year+text_window.txt', sep='\t')

    df = pd.read_csv(textfile, sep='\t')[['year', 'txt']]

    return df
# +

def compute_coocurrence_matrix(reader, **kwargs):

    corpus = text_corpus.ProcessedCorpus(reader, isalnum=False, **kwargs)
    vectorizer = corpus_vectorizer.CorpusVectorizer(lowercase=False)
    vectorizer.fit_transform(corpus)

    term_term_matrix = np.dot(vectorizer.X.T, vectorizer.X)

    term_term_matrix = scipy.sparse.triu(term_term_matrix, 1)

    coo = term_term_matrix
    id2token = { i: t for t,i in vectorizer.vocabulary.items()}
    cdf = pd.DataFrame({
        'w1_id': coo.row,
        'w2_id': coo.col,
        'value': coo.data
    })[['w1_id', 'w2_id', 'value']].sort_values(['w1_id', 'w2_id'])\
        .reset_index(drop=True)
    cdf['w1'] = cdf.w1_id.apply(lambda x: id2token[x])
    cdf['w2'] = cdf.w2_id.apply(lambda x: id2token[x])

    return cdf[['w1', 'w2', 'value']]

def compute_co_ocurrence_for_year(source_filename, year, result_filename):

    df = pd.read_csv(source_filename, sep='\t')[['year', 'txt']]

    reader = utility.DfTextReader(df, year)

    # stopwords = set(nltk.corpus.stopwords.words('swedish')) + { "politisk", "politiska", "politiskt" }

    kwargs = dict(to_lower=True, deacc=False, min_len=1, max_len=None, numerals=False, filter_stopwords=False) #, stopwords=stopwords)

    coo_df = compute_coocurrence_matrix(reader, **kwargs)
    coo_df.to_excel(result_filename)

compute_co_ocurrence_for_year('./data/year+text_window.txt', 1957, 'test_1957.xlsx')

