# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Text Analysis - Topic Modelling
# ### <span style='color: green'>SETUP </span> Prepare and Setup Notebook <span style='float: right; color: red'>MANDATORY</span>

import os
from typing import Callable

import penelope.notebook.topic_modelling as tm_ui
from bokeh.io import output_notebook
from IPython.display import display
from penelope.pipeline.config import CorpusConfig
from penelope.utility import pandas_utils
from westac.riksprot.parlaclarin import metadata as md

import __paths__  # pylint: disable=unused-import

output_notebook()
pandas_utils.set_default_options()

current_state: Callable[[], tm_ui.TopicModelContainer] = tm_ui.TopicModelContainer.singleton
corpus_folder: str = "/data/riksdagen_corpus_data/"
corpus_config: CorpusConfig = CorpusConfig.load(os.path.join(corpus_folder, "dtm_1920-2020_v0.3.0.tf20", 'corpus.yml'))
metadata_folder = '/data/riksdagen_corpus_data/dtm_1920-2020_v0.3.0.tf20'
output_notebook()  # resources=INLINE)

riksprot_metadata: md.ProtoMetaData = md.ProtoMetaData.load_from_same_folder(metadata_folder)

# %% [markdown]
# ### <span style='color: green'>PREPARE</span> Load Topic Model <span style='float: right; color: red'>MANDATORY</span>
#
# Notea! Underligande modell är tränad på alla enskilda tal. Aktuell testmodel är skapad genom att ett nytt korpus skapats där dokumenten utgörs av ett dokument per talare och år.
#
# Data flow:
#  - Use pyriksprot to produce a tokenized and tagged corpus
#
# Noteworthy:
#   - Dehypehnation of source material is done using a frequency based algorithm [eide? Språkbanken]. Given the overall corpus term frequencies (TF) and a hyphenated word "xxx-yyy", the individual frequencies for "xxx", "yyy", "xxxyyy" and "xxx-yyy" are are used to decide whether the word should be merged, split of left as is.
#   - A base topic model are

# %%
load_gui = tm_ui.create_load_topic_model_gui(corpus_config, corpus_folder, current_state())
display(load_gui.layout())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Display Topic's Word Distribution as a Wordcloud<span style='color: red; float: right'> TRY IT</span>

# %%
wc_ui = tm_ui.display_topic_wordcloud_gui(current_state())

# %% [markdown]
# ### <span style='color: green;'>BROWSE</span> Find topics by token<span style='color: red; float: right'>TRY IT</span>
#
# Displays topics in which given token is among toplist of dominant words.

# %%
find_ui = tm_ui.find_topic_documents_gui(current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic-Word Distribution<span style='color: red; float: right'>TRY IT</span>
#

# %%
tm_ui.display_topic_word_distribution_gui(current_state())

# %% [markdown]
# ### <span style='color: green;'>BROWSE</span> Browse Topic Documents<span style='color: red; float: right'>TRY IT</span>
#
# Displays documents in which a topic occurs above a given threshold.

# %%
tm_ui.display_topic_documents_gui(current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Trends over Time<span style='color: red; float: right'>RUN</span>

# %%
tt_gui = tm_ui.display_topic_trends_gui(current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Trends Overview<span style='color: red; float: right'>TRY IT</span>
#
# - The topic shares  displayed as a scattered heatmap plot using gradient color based on topic's weight in document.
# - [Stanford’s Termite software](http://vis.stanford.edu/papers/termite) uses a similar visualization.

# %%
tm_ui.display_topic_trends_overview_gui(current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Topic Network<span style='color: red; float: right'>TRY IT</span>
#
# Computes weighted graph of topics co-occurring in the same document. Topics are defined as co-occurring in a document if they both have a weight above given threshold. The edge weights are the number of co-occurrences (binary yes or no). Node size reflects topic proportions over the entire corpus computed in accordance to LDAvis topic proportions.

# %% code_folding=[0]
tm_ui.display_topic_topic_network_gui(current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Document Topic Network<span style='color: red; float: right'>TRY IT</span>
#

# %%
tm_ui.display_topic_document_network_gui(plot_mode=tm_ui.PlotMode.Default, state=current_state())  # type: ignore

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Focus-Topic Document Network<span style='color: red; float: right'>TRY IT</span>
#

# %%
tm_ui.display_topic_document_network_gui(plot_mode=tm_ui.PlotMode.FocusTopics, state=current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic-Token  Network<span style='color: red; float: right'>TRY IT</span>

# %%

corpus_folder: str = "/data/westac/sou_kb_labb"
custom_styles = {'edges': {'curve-style': 'haystack'}}
w = tm_ui.create_topics_token_network_gui(data_folder=corpus_folder, custom_styles=custom_styles)
display(w.layout())

# %%
