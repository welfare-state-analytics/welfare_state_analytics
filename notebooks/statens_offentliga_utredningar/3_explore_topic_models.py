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
# ## Text Analysis - Topic Modelling
# ### <span style='color: green'>SETUP </span> Prepare and Setup Notebook <span style='float: right; color: red'>MANDATORY</span>

# %%

# pylint: disable=wrong-import-position

import addpaths  # pylint: disable=import-error
import bokeh.plotting
from IPython.core.interactiveshell import InteractiveShell

import notebooks.common.load_topic_model_gui as load_gui
import notebooks.common.topic_word_distribution_gui as topic_word_distribution_gui
import notebooks.common.topic_wordcloud_gui as wordcloud_gui
import notebooks.statens_offentliga_utredningar.topic_document_network_gui as topic_document_gui
import notebooks.statens_offentliga_utredningar.topic_document_texts_gui as texts_gui
import notebooks.statens_offentliga_utredningar.topic_topic_network_gui as topic_topic_gui
import notebooks.statens_offentliga_utredningar.topic_trends_gui as trends_gui
import notebooks.statens_offentliga_utredningar.topic_trends_overview_gui as overview_gui
from notebooks.common import TopicModelContainer, setup_pandas

InteractiveShell.ast_node_interactivity = "all"

bokeh.plotting.output_notebook()
setup_pandas()
current_state: TopicModelContainer = TopicModelContainer.singleton

corpus_folder = "/data/westac/sou_kb_labb"
# %% [markdown]
# ### <span style='color: green'>PREPARE</span> Load Topic Model <span style='float: right; color: red'>MANDATORY</span>

# %%

load_gui.display_gui(corpus_folder, current_state())
# load_gui.load_model(corpus_folder, current_state(), 'test.4days')

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Display Topic's Word Distribution as a Wordcloud<span style='color: red; float: right'> TRY IT</span>

# %%

bokeh.plotting.output_notebook()
try:
    wordcloud_gui.display_gui(current_state())
except Exception as ex:
    print(ex)

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic-Word Distribution<span style='color: red; float: right'>TRY IT</span>
#

# %%

bokeh.plotting.output_notebook()
try:
    topic_word_distribution_gui.display_gui(current_state())
    # topic_word_distribution_gui.display_topic_tokens(current_state(), topic_id=0, n_words=100, output_format='Chart')
except Exception as ex:
    print(ex)

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Trends over Time<span style='color: red; float: right'>RUN</span>


# %%

bokeh.plotting.output_notebook()
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

bokeh.plotting.output_notebook()
try:
    overview_gui.display_gui(current_state())
except ValueError as ex:
    print(ex)

# %% [markdown]
# ### <span style='color: green;'>BROWSE</span> Browse Topic Documents<span style='color: red; float: right'>TRY IT</span>
#
# Computes weighted graph of topics co-occurring in the same document. Topics are defined as co-occurring if they both exists  in the same document both having weights above threshold. Weight are number of co-occurrences (binary yes or no). Node size reflects topic proportions over the entire corpus (normalized document) length, and are computed in accordance to how node sizes are computed in LDAvis.

# %%

bokeh.plotting.output_notebook()
try:
    texts_gui.display_gui(current_state())
except Exception as ex:
    print(ex)

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic-Topic Network<span style='color: red; float: right'>TRY IT</span>
#
# Computes weighted graph of topics co-occurring in the same document. Topics are defined as co-occurring in a document if they both have a weight above given threshold. The edge weights are the number of co-occurrences (binary yes or no). Node size reflects topic proportions over the entire corpus computed in accordance to LDAvis topic proportions.

# %% code_folding=[0]

bokeh.plotting.output_notebook()
try:
    topic_topic_gui.display_gui(current_state())
except Exception as ex:
    print(ex)

# %%

bokeh.plotting.output_notebook()
try:
    topic_document_gui.display_gui(current_state())
except Exception as ex:
    print(ex)


# %%
