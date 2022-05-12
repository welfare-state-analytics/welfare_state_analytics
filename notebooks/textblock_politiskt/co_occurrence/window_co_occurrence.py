# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
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

    transform_opts = TokensTransformOpts(
        to_lower=True,
        remove_accents=False,
        min_len=1,
        max_len=None,
        keep_numerals=False,
        only_any_alphanumeric=False,
    )
    corpus = TokenizedCorpus(reader, transform_opts=transform_opts)
    return corpus


ytw_filename = "./data/year+text_window.txt"
ytw_corpus = create_corpus(source_filename=ytw_filename, periods=1957)

# compute_co_occurrence_for_periods("./data/year+text_window.txt", 1957, target_filename="test_1957.xlsx")

# %%
