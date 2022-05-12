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
# | 💾 | <b>Checkpoint</b> | checkpoint_filename | Checkpoint (tagged frames) to file
#
# The PoS tagging notebook uses the same processing pipeline as the Word trends tnotebook  do to produce a tagged data frame. The processing will henceread
# a checkpoint file if it exists, otherwise resolve the full pipeline.
#
# The word count statistics are collected in the tagging task (part-of-speech and lemma annotation). The computed statistics, total word count and the word counts for each PoS-grouping, are added (or updated) to the _document index file_ as new columns. This file is stored in the tagged text archive as `document_index.csv`.
#
# Note: The dcument index file is either a pre-existing document index or, if no such index exists, automatically generated during the initial text loading pipeline task.
# If no pre-existing file exists, then the necessary attributes (e.g. document's year) are extracted from the filename of each  document.
#

# %% tags=[]

import __paths__  # pylint: disable=unused-import

from IPython.display import display
from penelope.notebook.token_counts import pipeline_gui

gui = pipeline_gui.create_token_count_gui(
    corpus_folder=__paths__.corpus_folder,
    resources_folder=".",
)
display(gui.layout())
