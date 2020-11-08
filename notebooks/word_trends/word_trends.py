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
#     display_name: 'Python 3.7.5 64-bit (''welfare_state_analytics'': pipenv)'
#     language: python
#     name: python37564bitwelfarestateanalyticspipenvb8730e518d45450a918e95a98e055dbf
# ---

# %% [markdown]
# ## Word Trend Analysis
#
# ### Load previously vectorized corpus
#
# Use the `vectorize_protocol` script to create a new corpus with different settings.

# %% tags=[] vscode={"end_execution_time": "2020-08-31T18:34:55.995Z", "start_execution_time": "2020-08-31T18:34:55.854Z"}
# %load_ext autoreload
# %autoreload 2


# %%

import importlib
import os
from dataclasses import dataclass
from typing import Any

import pandas as pd
import penelope.common.goodness_of_fit as gof
import penelope.notebook.load_vectorized_corpus_gui as load_corpus_gui
import penelope.notebook.utility as notebook_utility
import penelope.notebook.vectorize_corpus_gui as vectorize_corpus_gui
import penelope.notebook.word_trend_plot_gui as word_trend_plot_gui
from bokeh.plotting import output_notebook
from penelope.corpus import VectorizedCorpus
from penelope.notebook.ipyaggrid_utility import display_grid

import __paths__  # pylint: disable=unused-import
import notebooks.word_trends.word_trends_gui as xxx_word_trends_gui


@dataclass
class State:
    corpus_folder: str = '/data/westac'
    corpus: VectorizedCorpus = None
    data: Any = None
    df_gof: pd.DataFrame = None
    df_most_deviating_overview: pd.DataFrame = None
    df_most_deviating_overview: pd.DataFrame = None


state = State(corpus_folder='/data/westac')

output_notebook(hide_banner=True)

# %% tags=[] vscode={"end_execution_time": "2020-08-31T18:34:55.995Z", "start_execution_time": "2020-08-31T18:34:55.854Z"}


importlib.reload(vectorize_corpus_gui)
importlib.reload(notebook_utility)

vectorize_corpus_gui.display_gui(state.corpus_folder, '*sparv4.csv.zip', generated_callback=None)

# %% tags=[] vscode={"end_execution_time": "2020-08-31T18:34:55.995Z", "start_execution_time": "2020-08-31T18:34:55.854Z"}
importlib.reload(load_corpus_gui)
importlib.reload(notebook_utility)


def load_succeeded(_v_corpus: VectorizedCorpus, _corpus_tag, output):

    output.clear_output()
    try:
        global state

        state.corpus = _v_corpus
        state.df_gof = gof.compute_goddness_of_fits_to_uniform(state.corpus, 10000, verbose=False)
        state.df_most_deviating_overview = gof.compile_most_deviating_words(state.df_gof, n_count=10000)

        if os.environ.get('VSCODE_LOGS', None) is None:
            _ = (
                notebook_utility.OutputsTabExt(["GoF", "GoF (abs)", "Plots", "Slopes"])
                .display()
                .display_fx_result(0, display_grid, state.df_gof)
                .display_fx_result(
                    1, display_grid, state.df_most_deviating_overview[['l2_norm_token', 'l2_norm', 'abs_l2_norm']]
                )
                .display_fx_result(2, gof.plot_metrics, state.df_gof, plot=False, lazy=True)
                .display_fx_result(
                    3,
                    gof.plot_slopes,
                    state.corpus,
                    state.df_most_deviating_overview,
                    "l2_norm",
                    600,
                    600,
                    plot=False,
                    lazy=True,
                )
            )

    except Exception as ex:
        with output:
            print(ex)


load_corpus_gui.display_gui(state.corpus_folder, load_succeeded)

# %%

xxx_word_trends_gui.display_gui(state=state)

# %%

importlib.reload(word_trend_plot_gui)

most_deviating = gof.get_most_deviating_words(state.df_gof, 'l2_norm', n_count=5000, ascending=False, abs_value=True)

word_trend_plot_gui.display_gui(state.corpus, most_deviating)
