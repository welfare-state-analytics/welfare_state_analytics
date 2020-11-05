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

import os
import sys

import penelope.corpus.vectorized_corpus as vectorized_corpus
from penelope.common.most_discriminating_terms import compute_most_discriminating_terms

from notebooks.most_discriminating_words.most_discriminating_terms_gui import (
    display_gui,
    display_most_discriminating_terms,
)

root_folder = os.path.join(os.getcwd().split("welfare_state_analytics")[0], "welfare_state_analytics")

sys.path = list(set(sys.path + [root_folder]))


corpus_folder = os.path.join(root_folder, "output")

# %%
v_corpus = (
    vectorized_corpus.VectorizedCorpus.load(tag="SOU_1945-1989_NN+VB+JJ_lemma_L0_+N_+S", folder=corpus_folder)
    .slice_by_n_count(10)
    .slice_by_n_top(500000)
)


# %%

display_gui(
    v_corpus,
    v_corpus.documents,
    compute_callback=compute_most_discriminating_terms,
    display_callback=display_most_discriminating_terms,
)
