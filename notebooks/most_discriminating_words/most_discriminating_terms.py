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
# ## Most Discriminating Terms

# %%
# %load_ext autoreload
# %autoreload 2

# pylint: disable=wrong-import-position

import penelope.corpus.dtm as dtm
from IPython.core.display import display
from penelope.common.most_discriminating_terms import compute_most_discriminating_terms

import __paths__
from notebooks.most_discriminating_words.most_discriminating_terms_gui import (
    display_gui,
    display_most_discriminating_terms,
)

corpus: dtm.VectorizedCorpus = (
    dtm.VectorizedCorpus.load(tag="SOU_1945-1989_NN+VB+JJ_lemma_L0_+N_+S", folder=__paths__.output_folder)
    .slice_by_n_count(10)
    .slice_by_n_top(500000)
)

gui = display_gui(
    corpus,
    corpus.document_index,
    compute_callback=compute_most_discriminating_terms,
    display_callback=display_most_discriminating_terms,
)
display(gui.layout())
