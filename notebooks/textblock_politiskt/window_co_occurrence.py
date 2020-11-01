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

import __paths__  # isort:skip pylint: disable=import-error, unused-import

import os

import pandas as pd
from penelope.corpus import TokenizedCorpus
from penelope.corpus.readers import DataFrameTextTokenizer

# from notebooks.textblock_politiskt.pandas_co_occurrence import (
#     compute_co_occurrence_for_periods,
# )

root_folder = os.getcwd().split("notebooks")[0]


def create_corpus(source_filename: str, periods, **options):

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


ytw_filename = "./data/year+text_window.txt"
ytw_corpus = create_corpus(source_filename=ytw_filename, periods=1957)

# compute_co_occurrence_for_periods("./data/year+text_window.txt", 1957, target_filename="test_1957.xlsx")

# %%
