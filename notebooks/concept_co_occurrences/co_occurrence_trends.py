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
# pylint: disable=wrong-import-position, import-error

import importlib
import warnings

import penelope.common.goodness_of_fit as gof
import penelope.notebook.generate_concept_co_occurrences_gui as generate_concept_co_occurrences_gui
import penelope.notebook.load_vectorized_corpus_gui as load_vectorized_corpus_gui
import penelope.notebook.utility as notebook_utility
import penelope.notebook.word_trend_plot_gui as word_trend_plot_gui
from bokeh.plotting import output_notebook
from penelope.corpus import VectorizedCorpus

import __paths__  # pylint: disable=unused-import
import notebooks.common.ipyaggrid_utility as ipyaggrid_utility
from notebooks.concept_co_occurrences.utils import display_as_grid

warnings.filterwarnings("ignore", category=FutureWarning)

output_notebook()

# hv.extension("bokeh", logo=False)

# %% [markdown]
# ## Generate a new vectorized corpus
# For long running tasks, please use the CLI `concept_co_occurrence` instead.
# %%

generate_concept_co_occurrences_gui.display_gui('/data/westac', '*sparv4.csv.zip', generated_callback=None)

# %% [markdown]
# ### Load vectorized corpus and compute deviation metrics
# Deviation metrics compares co-occurrences distribution to a uniform distribution
# %%
importlib.reload(load_vectorized_corpus_gui)
importlib.reload(ipyaggrid_utility)
importlib.reload(notebook_utility)

v_corpus = None
df_gof = {}


def load_succeeded(_v_corpus: VectorizedCorpus, _corpus_tag, output):

    output.clear_output()
    try:
        global v_corpus, df_gof
        v_corpus = _v_corpus
        df_gof = gof.compute_goddness_of_fits_to_uniform(v_corpus, 10000, verbose=False)
        df_most_deviating = gof.compile_most_deviating_words(df_gof, n_count=10000)

        tab = notebook_utility.OutputsTabExt(["GoF", "GoF (abs)", "Plots", "Slopes"])
        tab.display().display_fx_result(0, display_as_grid, df_gof).display_fx_result(
            1, display_as_grid, df_most_deviating[['l2_norm_token', 'l2_norm', 'abs_l2_norm']]
        ).display_fx_result(2, gof.plot_metrics, df_gof, plot=False, lazy=True).display_fx_result(
            3, gof.plot_slopes, v_corpus, df_most_deviating, "l2_norm", 600, 600, plot=False, lazy=True
        )

    except Exception as ex:
        with output:
            print(ex)


load_vectorized_corpus_gui.display_gui("/data/westac", load_succeeded)
# %%

importlib.reload(word_trend_plot_gui)

most_deviating = gof.get_most_deviating_words(df_gof, 'l2_norm', n_count=5000, ascending=False, abs_value=True)

word_trend_plot_gui.display_gui(v_corpus, most_deviating)
