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

# %% [markdown]
# ### Setup notebook
# %%
# %load_ext autoreload
# %autoreload 2
# pylint: disable=too-many-instance-attributes, unused-argument

import importlib
import warnings

import penelope.notebook.concept_co_occurrences_gui as compute_gui
import penelope.notebook.load_co_occurrences_gui as load_gui
from bokeh.plotting import output_notebook

import __paths__  # pylint: disable=unused-import

from .loaded_callback import loaded_callback

warnings.filterwarnings("ignore", category=FutureWarning)

output_notebook()

# %% [markdown]
# ### Generate new concept context co-co_occurrences
# For long running tasks, please use the CLI `concept_co_occurrence` instead.
# This function computes new concept context co-occurrence data and stores the result in a CSV file.
# Optionally, the co-occurrence data can be transformed to a vectorized corpus to enable word trend exploration.
# %%
importlib.reload(compute_gui)
compute_gui.display_gui(data_folder=None, corpus_pattern='*sparv4.csv.zip', generated_callback=loaded_callback)

# %% [markdown]
# ### Load saved concept context co-occurrences
# %%
load_gui.display_gui(
    data_folder='/home/roger', filename_suffix="coo_concept_context.csv.zip", loaded_callback=loaded_callback
)
