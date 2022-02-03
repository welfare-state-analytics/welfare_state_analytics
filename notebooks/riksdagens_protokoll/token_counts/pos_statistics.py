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

import __paths__  # pylint: disable=unused-import
import importlib
from os.path import join as jj

import pandas as pd
from bokeh.io import output_notebook
from IPython.display import display

from notebooks.riksdagens_protokoll.token_counts import pos_statistics_gui as ps
from westac.riksprot.parlaclarin import metadata as md

importlib.reload(ps)
output_notebook()

pd.set_option('display.max_rows', 2000)
data_folder: str = jj(__paths__.data_folder, "riksdagen_corpus_data/dtm_1920-2020_v0.3.0.tf20")
riksprot_metadata: md.ProtoMetaData = md.ProtoMetaData(members=jj(data_folder, 'person_index.zip'))

gui = ps.PoSCountGUI(default_folder=data_folder, riksprot_metadata=riksprot_metadata).setup(load_data=True)
display(gui.layout())

# %%
