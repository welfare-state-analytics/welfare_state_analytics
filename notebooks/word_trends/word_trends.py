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
# ## Word Trend Analysis
#
# ### Load previously vectorized corpus
#
# Use the `vectorize_protocol` script to create a new corpus with different settings.

# %%
# %load_ext autoreload
# %autoreload 2
import importlib

import penelope.notebook.load_vectorized_corpus_gui as load_corpus_gui
import penelope.notebook.utility as notebook_utility
import penelope.notebook.vectorize_corpus_gui as vectorize_corpus_gui
from bokeh.plotting import output_notebook

import __paths__  # pylint: disable=unused-import
from notebooks.word_trends.gui_callback import loaded_callback

output_notebook(hide_banner=True)

# %% tags=[] vscode={"end_execution_time": "2020-08-31T18:34:55.995Z", "start_execution_time": "2020-08-31T18:34:55.854Z"}
importlib.reload(vectorize_corpus_gui)
importlib.reload(notebook_utility)
importlib.reload(load_corpus_gui)
importlib.reload(notebook_utility)

# %% tags=[] vscode={"end_execution_time": "2020-08-31T18:34:55.995Z", "start_execution_time": "2020-08-31T18:34:55.854Z"}
importlib.reload(vectorize_corpus_gui)
vectorize_corpus_gui.display_gui('*sparv4.csv.zip', generated_callback=loaded_callback)

# %%
load_corpus_gui.display_gui(loaded_callback=loaded_callback)

# %%
