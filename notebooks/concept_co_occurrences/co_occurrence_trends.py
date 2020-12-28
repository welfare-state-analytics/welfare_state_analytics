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

import penelope.notebook.co_occurrence.load_co_occurrences_gui as load_gui
import penelope.notebook.co_occurrence.to_co_occurrence_gui as compute_gui
from bokeh.plotting import output_notebook
from IPython.display import display
from penelope.notebook.co_occurrence.compute_callback_corpus import compute_co_occurrence

import __paths__  # pylint: disable=unused-import
from notebooks.corpus_data_config import ParliamentarySessions

from .loaded_callback import loaded_callback

warnings.filterwarnings("ignore", category=FutureWarning)

output_notebook()
corpus_folder = __paths__.root_folder

# %% [markdown]
# ### Generate new concept context co-co_occurrences
# For long running tasks, please use the CLI `co_occurrence` instead.
# This function computes new concept context co-occurrence data and stores the result in a CSV file.
# Optionally, the co-occurrence data can be transformed to a vectorized corpus to enable word trend exploration.
# %%

importlib.reload(compute_gui)
gui: compute_gui.ComputeGUI = compute_gui.ComputeGUI.create(
    corpus_folder=corpus_folder,
    corpus_config=ParliamentarySessions(corpus_folder=corpus_folder),
    done_callback=loaded_callback,
    compute_callback=compute_co_occurrence,
)
display(gui.layout())
# %% [markdown]
# ### Load saved concept context co-occurrences
# %%
lgu: load_gui.LoadGUI = load_gui.LoadGUI.create(
    data_folder=corpus_folder, filename_pattern="*.co_occurrence.csv.zip", loaded_callback=loaded_callback
)
display(lgu.layout())
