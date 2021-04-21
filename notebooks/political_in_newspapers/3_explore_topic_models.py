# -*- coding: utf-8 -*-
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
#     display_name: 'Python 3.7.5 64-bit (''text_analytic_tools'': pipenv)'
#     language: python
#     name: python37564bittextanalytictoolspipenv8c3d4c9c6f39484cb74f0ad2d777602d
# ---

# %% [markdown]
# ## Text Analysis - Topic Modelling
# ### <span style='color: green'>SETUP </span> Prepare and Setup Notebook <span style='float: right; color: red'>MANDATORY</span>

# %%
# pylint: disable=wrong-import-position

import __paths__  # isort:skip pylint: disable=import-error, unused-import

import bokeh.plotting
import penelope.notebook.topic_modelling as gui
from IPython.core.interactiveshell import InteractiveShell
from IPython.display import display

import notebooks.political_in_newspapers.notebook_gui.publication_topic_network_gui as publication_topic_network_gui
import notebooks.political_in_newspapers.notebook_gui.topic_document_texts_gui as texts_gui
import notebooks.political_in_newspapers.notebook_gui.topic_topic_network_gui as topic_topic_gui
import notebooks.political_in_newspapers.notebook_gui.topic_trends_gui as trends_gui
import notebooks.political_in_newspapers.notebook_gui.topic_trends_overview_gui as overview_gui

InteractiveShell.ast_node_interactivity = "all"

# %matplotlib inline

current_state = gui.TopicModelContainer.singleton
bokeh.plotting.output_notebook()

# %% [markdown]
# ### <span style='color: green'>PREPARE</span> Load Topic Model <span style='float: right; color: red'>MANDATORY</span>

# %%

load_gui = gui.create_load_topic_model_gui(
    corpus_config=None, corpus_folder='/data/westac/political_in_newspapers', state=current_state()
)
display(load_gui.layout())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Display Topic's Word Distribution as a Wordcloud<span style='color: red; float: right'> TRY IT</span>

# %%

try:
    gui.display_topic_wordcloud_gui(current_state())
except Exception as ex:
    print(ex)

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic-Word Distribution<span style='color: red; float: right'>TRY IT</span>
#

# %%

try:
    gui.display_topic_word_distribution_gui(current_state())
    # topic_word_distribution_gui.display_topic_tokens(current_state(), topic_id=0, n_words=100, output_format='Chart')
except Exception as ex:
    print(ex)

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Trends over Time<span style='color: red; float: right'>RUN</span>


# %%

try:
    trends_gui.display_gui(current_state())
    # trends_gui.display_topic_trend(current_state().inferred_topics.document_topic_weights, topic_id=0, year=None, year_aggregate='mean', output_format='Table')
except Exception as ex:
    print(ex)

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Trends Overview<span style='color: red; float: right'>TRY IT</span>
#
# - The topic shares  displayed as a scattered heatmap plot using gradient color based on topic's weight in document.
# - [Stanford’s Termite software](http://vis.stanford.edu/papers/termite) uses a similar visualization.

# %%

try:
    overview_gui.display_gui(current_state())
except ValueError as ex:
    print(ex)

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Publication Topic Network<span style='color: red; float: right'>TRY IT</span>
# The green nodes are documents, and blue nodes are topics. The edges (lines) indicates the strength of a topic in the connected document. The width of the edge is proportinal to the strength of the connection. Note that only edges with a strength above the certain threshold are displayed.

# %%

try:
    publication_topic_network_gui.display_gui(current_state())
except Exception as ex:
    print(ex)

# %% [markdown]
# ### <span style='color: green;'>BROWSE</span> Browse Topic Documents<span style='color: red; float: right'>TRY IT</span>
#
# Computes weighted graph of topics co-occurring in the same document. Topics are defined as co-occurring if they both exists  in the same document both having weights above threshold. Weight are number of co-occurrences (binary yes or no). Node size reflects topic proportions over the entire corpus (normalized document) length, and are computed in accordance to how node sizes are computed in LDAvis.

# %%

try:
    texts_gui.display_gui(current_state())
except Exception as ex:
    print(ex)

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic-Topic Network<span style='color: red; float: right'>TRY IT</span>
#
# Computes weighted graph of topics co-occurring in the same document. Topics are defined as co-occurring in a document if they both have a weight above given threshold. The edge weights are the number of co-occurrences (binary yes or no). Node size reflects topic proportions over the entire corpus computed in accordance to LDAvis topic proportions.


# %% code_folding=[0]

try:
    topic_topic_gui.display_gui(current_state())
except Exception as ex:
    print(ex)
