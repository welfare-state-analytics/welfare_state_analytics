# -*- coding: utf-8 -*-
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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Concept Context Co-Occurrences Analysis

# %% tags=[]
# %load_ext autoreload
# %autoreload 2
# pylint: disable=too-many-instance-attributes

import warnings

import penelope.notebook.concept_co_occurrences_gui as compute_gui
import penelope.notebook.load_co_occurrences_gui as load_gui
import penelope.notebook.load_vectorized_corpus_gui as load_vectorized_corpus_gui
from bokeh.plotting import output_notebook

import __paths__  # pylint: disable=unused-import
from notebooks.concept_co_occurrences.gui_callback import loaded_callback

warnings.filterwarnings("ignore", category=FutureWarning)

output_notebook()


# %% [markdown]
# ## Generate a new concept context co-co_occurrence
# For long running tasks, please use the CLI `concept_co_occurrence` instead.
# This function computes new concept context co-occurrence data and stores the result in a CSV file.
# Optionally, the co-occurrence data can be transformed to a vectorized corpus to enable word trend exploration.
# %%
compute_gui.display_gui('/data/westac', '*sparv4.csv.zip', generated_callback=None)

# %% [markdown]
# ## Display concept context co-occurrences
# %%
load_gui.display_gui(
    data_folder='/home/roger', filename_suffix="coo_concept_context.csv.zip", loaded_callback=loaded_callback
)

# %% [markdown]
# ### Load vectorized corpus and compute deviation metrics
# Deviation metrics compares co-occurrences distribution to a uniform distribution
# %%
load_vectorized_corpus_gui.display_gui(loaded_callback=loaded_callback)
