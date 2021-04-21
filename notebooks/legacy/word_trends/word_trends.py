# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
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

from bokeh.plotting import output_notebook
from IPython.core.display import display
from penelope import pipeline, workflows
from penelope.corpus import dtm
from penelope.notebook import interface, word_trends
from penelope.notebook.dtm import load_dtm_gui, to_dtm_gui

import __paths__

output_notebook()

# %% tags=[] vscode={"end_execution_time": "2020-08-31T18:34:55.995Z", "start_execution_time": "2020-08-31T18:34:55.854Z"}


def done_callback(corpus: dtm.VectorizedCorpus, corpus_folder: str, corpus_tag: str):

    trends_data: word_trends.TrendsData = word_trends.TrendsData(
        corpus=corpus,
        corpus_folder=corpus_folder,
        corpus_tag=corpus_tag,
        n_count=25000,
    ).update()

    gui = word_trends.GofTrendsGUI(
        gofs_gui=word_trends.GoFsGUI().setup(),
        trends_gui=word_trends.TrendsGUI().setup(),
    )

    display(gui.layout())
    gui.display(trends_data=trends_data)


def compute_done_callback(corpus: dtm.VectorizedCorpus, opts: interface.ComputeOpts):
    done_callback(corpus=corpus, corpus_folder=opts.target_folder, corpus_tag=opts.corpus_tag)


def compute_callback(args: interface.ComputeOpts, corpus_config: pipeline.CorpusConfig) -> dtm.VectorizedCorpus:
    corpus: dtm.VectorizedCorpus = workflows.document_term_matrix.compute(args=args, corpus_config=corpus_config)
    return corpus


compute_gui: to_dtm_gui.ComputeGUI = to_dtm_gui.create_compute_gui(
    corpus_config="riksdagens-protokoll",
    corpus_folder=__paths__.corpus_folder,
    data_folder=__paths__.data_folder,
    compute_callback=compute_callback,
    done_callback=compute_done_callback,
)
display(compute_gui.layout())

# %%
load_gui: load_dtm_gui.LoadGUI = load_dtm_gui.create_load_gui(
    corpus_folder=__paths__.data_folder,
    loaded_callback=done_callback,
)
display(load_gui.layout())
