# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Text Analysis - Topic Modelling
# ### <span style='color: green'>SETUP </span> Prepare and Setup Notebook <span style='float: right; color: red'>MANDATORY</span>

# %%

import __paths__  # pylint: disable=unused-import
import os
from typing import Callable

import bokeh.plotting
import penelope.notebook.topic_modelling as ntm
from IPython.display import display
from penelope import utility as pu
from penelope.pipeline.config import CorpusConfig

bokeh.plotting.output_notebook()
pu.set_default_options()

current_state: Callable[[], ntm.TopicModelContainer] = ntm.TopicModelContainer.singleton
corpus_folder: str = "/data/westac/sou_kb_labb"
corpus_config: CorpusConfig = CorpusConfig.load(os.path.join(__paths__.resources_folder, 'sou_sparv4.yml'))


# %% [markdown]
# ### <span style='color: green'>PREPARE</span> Load Topic Model <span style='float: right; color: red'>MANDATORY</span>

# %%
load_gui = ntm.create_load_topic_model_gui(corpus_config, corpus_folder, current_state())
display(load_gui.layout())

# %% [markdown]
# ### <span style='color: green;'>BROWSE</span> Find topics by token<span style='color: red; float: right'>TRY IT</span>
#
# Displays topics in which given token is among toplist of dominant words.

# %%
display(ntm.topic_documents_gui.FindTopicDocumentsGUI(state=current_state).setup().layout())

# %% [markdown]
# ### <span style='color: green;'>BROWSE</span> Browse Topic Documents<span style='color: red; float: right'>TRY IT</span>
#
# Displays documents in which a topic occurs above a given threshold.

# %%
display(ntm.topic_documents_gui.BrowseTopicDocumentsGUI(state=current_state).setup().layout())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Display Topic's Word Distribution as a Wordcloud<span style='color: red; float: right'> TRY IT</span>

# %%
ntm.display_topic_wordcloud_gui(current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic-Word Distribution<span style='color: red; float: right'>TRY IT</span>
#

# %%
ntm.display_topic_word_distribution_gui(current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Trends over Time<span style='color: red; float: right'>RUN</span>

# %%
ntm.display_topic_trends_gui(current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Trends Overview<span style='color: red; float: right'>TRY IT</span>
#
# - The topic shares  displayed as a scattered heatmap plot using gradient color based on topic's weight in document.
# - [Stanford’s Termite software](http://vis.stanford.edu/papers/termite) uses a similar visualization.

# %%
ntm.display_topic_trends_overview_gui(current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Topic Network<span style='color: red; float: right'>TRY IT</span>
#
# Computes weighted graph of topics co-occurring in the same document. Topics are defined as co-occurring in a document if they both have a weight above given threshold. The edge weights are the number of co-occurrences (binary yes or no). Node size reflects topic proportions over the entire corpus computed in accordance to LDAvis topic proportions.

# %% code_folding=[0]
ntm.display_topic_topic_network_gui(current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Document Topic Network<span style='color: red; float: right'>TRY IT</span>
#

# %%
display(ntm.DefaultTopicDocumentNetworkGui(pivot_key_specs={}, state=current_state()).setup().layout())
# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Focus-Topic Document Network<span style='color: red; float: right'>TRY IT</span>
#

# %%
display(ntm.FocusTopicDocumentNetworkGui(pivot_key_specs={}, state=current_state()).setup().layout())
# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic-Token  Network<span style='color: red; float: right'>TRY IT</span>

# %%

corpus_folder: str = "/data/westac/sou_kb_labb"
custom_styles = {'edges': {'curve-style': 'haystack'}}
w = ntm.create_topics_token_network_gui(data_folder=corpus_folder, custom_styles=custom_styles)
display(w.layout())

# %%
