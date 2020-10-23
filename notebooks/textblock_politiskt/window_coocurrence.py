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
# %%capture
# %load_ext autoreload
# %autoreload 2

# pylint: disable=wrong-import-position

import os

import __paths__  # pylint: disable=import-error, unused-import
import pandas as pd
from penelope.corpus.readers import DataFrameTextTokenizer
from penelope.corpus import TokenizedCorpus

root_folder = os.getcwd().split("notebooks")[0]

def create_corpus(source_filename: str, periods, result_filename: str, **options):

    df = pd.read_csv(source_filename, sep="\t")[["year", "txt"]]

    reader = DataFrameTextTokenizer(df, column_filters={"year": periods})

    tokenize_opts = {
        **dict(
            to_lower=True,
            remove_accents=False,
            min_len=1,
            max_len=None,
            keep_numerals=False,
        ),
        **options,
    }
    corpus = TokenizedCorpus(reader, only_alphanumeric=False, **tokenize_opts)
    return corpus

source_filename = "./data/year+text_window.txt"
corpus = create_corpus(source_filename=source_filename, periods=1957, result_filename="test_1957.xlsx")

compute_co_ocurrence_for_periods("./data/year+text_window.txt", 1957, "test_1957.xlsx")

# %%
