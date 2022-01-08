# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Token Count Statistics
# ### Text Processing Pipeline

# | | Building block | Arguments | Description |
# | -- | :------------- | :------------- | :------------- |
# | 💾 | <b>pyriksprot (tagger(</b> | FIXME | FIXME
# | 💾 | <b>pyriksprot (extract)</b> | FIXME | FIXME

# %% tags=[]
import __paths__  # pylint: disable=unused-import

from IPython.core.display import display
from notebooks.riksdagens_protokoll.token_counts.pos_statistics_gui import PoSCountGUI


gui = PoSCountGUI(default_folder='/data/riksdagen_corpus_data/dtm_1920-2020_v0.3.0.tf20').setup()
display(gui.layout())
