# -*- coding: utf-8 -*-
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

from IPython.display import display
from penelope.notebook.token_counts import pipeline_gui

gui = pipeline_gui.create_token_count_gui(corpus_folder=__paths__.corpus_folder, resources_folder="..")
display(gui.layout())
