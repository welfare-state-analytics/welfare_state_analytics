# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
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

import pandas as pd
from penelope.corpus import TokenizedCorpus, TokensTransformOpts
from penelope.corpus.readers import PandasCorpusReader

import __paths__  # isort:skip pylint: disable=import-error, unused-import


# from notebooks.textblock_politiskt.pandas_co_occurrence import (
#     compute_co_occurrence_for_periods,
# )

root_folder = os.getcwd().split("notebooks")[0]


def create_corpus(source_filename: str, periods):

    df = pd.read_csv(source_filename, sep="\t")[["year", "txt"]]

    reader = PandasCorpusReader(df, column_filters={"year": periods})

    tokens_transform_opts = TokensTransformOpts(
        to_lower=True,
        remove_accents=False,
        min_len=1,
        max_len=None,
        keep_numerals=False,
        only_alphanumeric=False,
    )
    corpus = TokenizedCorpus(reader, tokens_transform_opts=tokens_transform_opts)
    return corpus


ytw_filename = "./data/year+text_window.txt"
ytw_corpus = create_corpus(source_filename=ytw_filename, periods=1957)

# compute_co_occurrence_for_periods("./data/year+text_window.txt", 1957, target_filename="test_1957.xlsx")

# %%
