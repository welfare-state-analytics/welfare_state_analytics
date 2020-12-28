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

import penelope.notebook.dtm.load_DTM_gui as load_corpus_gui
import penelope.notebook.dtm.to_DTM_gui as to_DTM_gui
from bokeh.plotting import output_notebook
from IPython.core.display import display
from penelope.corpus.dtm import VectorizedCorpus
from penelope.notebook.dtm.compute_DTM_corpus import compute_document_term_matrix
from penelope.notebook.word_trends import create_gui as create_trends_gui

from notebooks.corpus_data_config import ParliamentarySessions

output_notebook()

# %% tags=[] vscode={"end_execution_time": "2020-08-31T18:34:55.995Z", "start_execution_time": "2020-08-31T18:34:55.854Z"}
data_folder = "."


def done_callback(corpus: VectorizedCorpus, corpus_tag: str, corpus_folder: str):
    gui = create_trends_gui(corpus=corpus, corpus_tag=corpus_tag, corpus_folder=corpus_folder)
    display(gui.layout())


compute_gui = to_DTM_gui.create_gui(
    corpus_config=ParliamentarySessions(corpus_folder=data_folder),
    compute_document_term_matrix=compute_document_term_matrix,
    corpus_folder=data_folder,
    pipeline_factory=None,
    done_callback=done_callback,
)
display(compute_gui.layout())

# %%
load_gui = load_corpus_gui.create_gui(corpus_folder=data_folder, loaded_callback=done_callback)
display(load_gui.layout())
