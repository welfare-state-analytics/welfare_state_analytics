# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: 'Python 3.8.5 64-bit (''westac-O7wB9ikj-py3.8'': venv)'
#     metadata:
#       interpreter:
#         hash: 22b1e31d3d5f905e8a7998fc3532ca535c806f75d42474f77651a5c803dd310a
#     name: 'Python 3.8.5 64-bit (''westac-O7wB9ikj-py3.8'': venv)'
# ---

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# pylint: disable=wrong-import-position

import os
import sys

import numpy as np
import pandas as pd
import scipy

root_folder = os.getcwd().split("notebooks")[0]
sys.path = list(set(sys.path + [root_folder]))

import penelope.corpus.readers as readers
import penelope.corpus.tokenized_corpus as tokenized_corpus
import penelope.corpus.vectorizer as corpus_vectorizer

# df = pd.read_excel('./data/year+text_window.xlsx')
# df.to_csv('./data/year+text_window.txt', sep='\t')


def compute_coocurrence_matrix(reader, **tokenize_opts):

    corpus = tokenized_corpus.TokenizedCorpus(reader, only_alphanumeric=False, **tokenize_opts)
    vectorizer = corpus_vectorizer.CorpusVectorizer(lowercase=False)
    v_corpus = vectorizer.fit_transform(corpus)

    term_term_matrix = np.dot(v_corpus.bag_term_matrix.T, v_corpus.bag_term_matrix)

    term_term_matrix = scipy.sparse.triu(term_term_matrix, 1)

    coo = term_term_matrix
    cdf = (
        pd.DataFrame({"w1_id": coo.row, "w2_id": coo.col, "value": coo.data})[["w1_id", "w2_id", "value"]]
        .sort_values(["w1_id", "w2_id"])
        .reset_index(drop=True)
    )
    cdf["w1"] = cdf.w1_id.apply(lambda x: v_corpus.id2token[x])
    cdf["w2"] = cdf.w2_id.apply(lambda x: v_corpus.id2token[x])

    return cdf[["w1", "w2", "value"]]


def compute_co_ocurrence_for_periods(source_filename: str, periods, result_filename: str, **options):

    df = pd.read_csv(source_filename, sep="\t")[["year", "txt"]]

    reader = readers.DataFrameTextTokenizer(df, column_filters={"year": periods})

    options = {
        **dict(
            to_lower=True,
            remove_accents=False,
            min_len=1,
            max_len=None,
            keep_numerals=False,
        ),
        **options,
    }

    coo_df = compute_coocurrence_matrix(reader, **options)
    coo_df.to_excel(result_filename)


compute_co_ocurrence_for_periods("./data/year+text_window.txt", 1957, "test_1957.xlsx")
