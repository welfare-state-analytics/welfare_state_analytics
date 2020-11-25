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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Text Analysis - Topic Modelling
# ### <span style='color: green'>SETUP </span> Prepare and Setup Notebook <span style='float: right; color: red'>MANDATORY</span>

# %%
# %%capture

import bokeh.plotting
import penelope.notebook.topic_modelling as gui
from penelope.utility import get_logger

import __paths__  # pylint: disable=unused-import
from notebooks.common import setup_pandas

logger = get_logger()

bokeh.plotting.output_notebook()
setup_pandas()
current_state: gui.TopicModelContainer = gui.TopicModelContainer.singleton
corpus_folder = "/data/westac/sou_kb_labb"

# %% [markdown]
# ### <span style='color: green'>PREPARE</span> Load Topic Model <span style='float: right; color: red'>MANDATORY</span>

# %%
gui.display_load_topic_model_gui(corpus_folder, current_state())

# %% [markdown]
# ### <span style='color: green;'>BROWSE</span> Find topics by token<span style='color: red; float: right'>TRY IT</span>
#
# Displays topics in which given token is among toplist of dominant words.

# %%
try:
    gui.find_topic_documents_gui(
        current_state().inferred_topics.document_topic_weights, current_state().inferred_topics.topic_token_overview
    )
except Exception as ex:
    logger.exception(ex)

# %% [markdown]
# ### <span style='color: green;'>BROWSE</span> Browse Topic Documents<span style='color: red; float: right'>TRY IT</span>
#
# Displays documents in which a topic occurs above a given threshold.

# %%
try:
    gui.display_topic_documents_gui(current_state())
except Exception as ex:
    logger.exception(ex)

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Display Topic's Word Distribution as a Wordcloud<span style='color: red; float: right'> TRY IT</span>

# %%
bokeh.plotting.output_notebook()
try:
    gui.display_topic_wordcloud_gui(current_state())
except Exception as ex:
    logger.exception(ex)

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic-Word Distribution<span style='color: red; float: right'>TRY IT</span>
#

# %%
try:
    gui.display_topic_word_distribution_gui(current_state())
    # topic_word_distribution_gui.display_topic_tokens(current_state(), topic_id=0, n_words=100, output_format='Chart')
except Exception as ex:
    logger.exception(ex)

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Trends over Time<span style='color: red; float: right'>RUN</span>

# %%
try:
    gui.display_topic_trends_gui(current_state())
except Exception as ex:
    logger.exception(ex)

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Trends Overview<span style='color: red; float: right'>TRY IT</span>
#
# - The topic shares  displayed as a scattered heatmap plot using gradient color based on topic's weight in document.
# - [Stanford’s Termite software](http://vis.stanford.edu/papers/termite) uses a similar visualization.

# %%
bokeh.plotting.output_notebook()
try:
    gui.display_topic_trends_overview_gui(current_state())
except ValueError as ex:
    logger.exception(ex)

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Topic Network<span style='color: red; float: right'>TRY IT</span>
#
# Computes weighted graph of topics co-occurring in the same document. Topics are defined as co-occurring in a document if they both have a weight above given threshold. The edge weights are the number of co-occurrences (binary yes or no). Node size reflects topic proportions over the entire corpus computed in accordance to LDAvis topic proportions.

# %% code_folding=[0]
bokeh.plotting.output_notebook()
gui.display_topic_topic_network_gui(current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Document Topic Network<span style='color: red; float: right'>TRY IT</span>
#

# %%
bokeh.plotting.output_notebook()
gui.display_topic_document_network_gui(current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Focus-Topic Document Network<span style='color: red; float: right'>TRY IT</span>
#

# %%
bokeh.plotting.output_notebook()
gui.display_focus_topics_network_gui(current_state())

# %%
