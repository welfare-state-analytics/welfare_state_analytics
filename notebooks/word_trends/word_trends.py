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

# %%

import penelope.notebook.vectorize_corpus_gui as vectorize_corpus_gui
import penelope.notebook.vectorized_corpus_load_gui as load_corpus_gui
from bokeh.plotting import output_notebook
from penelope.notebook.word_trends import display_word_trends

output_notebook()

# %% tags=[] vscode={"end_execution_time": "2020-08-31T18:34:55.995Z", "start_execution_time": "2020-08-31T18:34:55.854Z"}

vectorize_corpus_gui.display_gui('*sparv4.csv.zip', generated_callback=display_word_trends)


# %%
load_corpus_gui.display_gui(loaded_callback=display_word_trends)
