# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Token Count Statistics
# ### Text Processing Pipeline
#
# | | Building block | Arguments | Description |
# | -- | :------------- | :------------- | :------------- |
# | 💾 | <b>pyriksprot</b> | TF[20, MASK] | Extract corpus from Parla-CLARIN
# | 💾 | <b>pyriksprot (tagger)</b> | _ | PoS-tag and lemmatize
# | 💾 | <b>dtm_id</b> | _ | Create DTM
# | 💾 | <b>dtm </b> | _ | Create DTM
#
#

# %% tags=[]

import importlib

import pandas as pd
from bokeh.io import output_notebook
from IPython.display import display
from westac.riksprot.parlaclarin import metadata

import __paths__  # pylint: disable=unused-import
from notebooks.riksdagens_protokoll.token_counts import pos_statistics_gui as ps

importlib.reload(ps)
output_notebook()

pd.set_option('display.max_rows', 500)

md: metadata.ProtoMetaData = metadata.ProtoMetaData.load_from_same_folder(
    '/data/riksdagen_corpus_data/dtm_1920-2020_v0.3.0.tf20'
)

gui = ps.PoSCountGUI(
    default_folder='/data/riksdagen_corpus_data/dtm_1920-2020_v0.3.0.tf20', riksprot_metadata=md
).setup(load_data=True)
display(gui.layout())

# %%
