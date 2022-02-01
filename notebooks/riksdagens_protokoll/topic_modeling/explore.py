# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Text Analysis - Topic Modelling
# ### <span style='color: green'>SETUP </span> Setup Notebook<span style='float: right; color: red'>MANDATORY</span>

# %%
import __paths__  # pylint: disable=unused-import
import os

from bokeh.io import output_notebook
from IPython.display import display
from penelope import utility as pu
from penelope.notebook import topic_modelling as ntm
from penelope.pipeline.config import CorpusConfig

import westac.riksprot.parlaclarin.speech_text as sr
from notebooks.riksdagens_protokoll import topic_modeling as wtm
from westac.riksprot.parlaclarin import metadata as md

output_notebook()
pu.set_default_options()

current_state = ntm.TopicModelContainer.singleton
corpus_folder: str = "/data/westac/riksdagen_corpus_data/"
corpus_config: CorpusConfig = CorpusConfig.load(os.path.join(corpus_folder, "dtm_1920-2020_v0.3.0.tf20", 'corpus.yml'))
metadata_folder = '/data/westac/riksdagen_corpus_data/dtm_1920-2020_v0.3.0.tf20'

riksprot_metadata: md.ProtoMetaData = md.ProtoMetaData.load_from_same_folder(metadata_folder)
speech_repository: sr.SpeechTextRepository = sr.SpeechTextRepository(
    folder="/data/westac/riksdagen_corpus_data/tagged_frames_v0.3.0_20201218",
    riksprot_metadata=riksprot_metadata,
)
# %% [markdown]
# ### <span style='color: green'>SETUP </span> Load Model<span style='float: right; color: red'>MANDATORY</span>
#

# %%
load_gui = wtm.RiksprotLoadGUI(
    riksprot_metadata,
    corpus_folder=corpus_folder,
    corpus_config=None,
    state=current_state(),
    slim=True,
).setup()
display(load_gui.layout())
# %%
current_state().inferred_topics.document_index = riksprot_metadata.overload_by_member_data(
    current_state().inferred_topics.document_index, encoded=True, drop=True
)

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Display Topic's Word Distribution as a Wordcloud<span style='color: red; float: right'> TRY IT</span>

# %%
wc_ui = ntm.WordcloudGUI(current_state()).setup()
display(wc_ui.layout())
wc_ui.update_handler()

# %% [markdown]
# ### <span style='color: green;'>BROWSE</span> Find topic's documents by token<span style='color: red; float: right'>TRY IT</span>
# Displays documents having topics in which given token is in toplist of dominant words.

# %%
find_ui = wtm.RiksprotFindTopicDocumentsGUI(
    riksprot_metadata=riksprot_metadata, speech_repository=speech_repository, state=current_state()
).setup()
display(find_ui.layout())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic-Word Distribution<span style='color: red; float: right'>TRY IT</span>
#

# %%
ntm.display_topic_word_distribution_gui(current_state())

# %% [markdown]
# ### <span style='color: green;'>BROWSE</span> Browse Topic Documents<span style='color: red; float: right'>TRY IT</span>
#
# Displays documents in which a topic occurs above a given threshold.

# %%
btd_ui = wtm.RiksprotBrowseTopicDocumentsGUI(
    riksprot_metadata=riksprot_metadata, speech_repository=speech_repository, state=current_state()
).setup()
display(btd_ui.layout())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Trends over Time<span style='color: red; float: right'>RUN</span>

# %%
rtt_ui = wtm.RiksprotTopicTrendsGUI(
    riksprot_metadata, speech_repository=speech_repository, state=current_state()
).setup()
display(rtt_ui.layout())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Trends Overview<span style='color: red; float: right'>TRY IT</span>
# The topic shares  displayed as a scattered heatmap plot using gradient color based on topic's weight in documen (see [Stanford’s Termite software](http://vis.stanford.edu/papers/termite).
#

# %%
tto_ui = wtm.RiksprotTopicTrendsOverviewGUI(
    riksprot_metadata, speech_repository=speech_repository, state=current_state()
).setup()
display(tto_ui.layout())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Topic Network<span style='color: red; float: right'>TRY IT</span>
#
# Computes weighted graph of topics co-occurring in the same document. Topics are defined as co-occurring in a document if they both have a weight above given threshold. The edge weights are the number of co-occurrences (binary yes or no). Node size reflects topic proportions over the entire corpus computed in accordance to LDAvis topic proportions.

# %%
ttx_ui = wtm.RiksprotTopicTopicGUI(
    riksprot_metadata, speech_repository=speech_repository, state=current_state()
).setup()
display(ttx_ui.layout())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Document Topic Network<span style='color: red; float: right'>TRY IT</span>
#

# %%
ptn_ui = ntm.PivotTopicNetworkGUI(
    pivot_key_specs=riksprot_metadata.member_property_specs, state=current_state()
).setup()
display(ptn_ui.layout())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Focus-Topic Document Network<span style='color: red; float: right'>TRY IT</span>
#

# %%
ntm.display_topic_document_network_gui(plot_mode=ntm.PlotMode.FocusTopics, state=current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic-Token  Network<span style='color: red; float: right'>TRY IT</span>

# %%
w = ntm.create_topics_token_network_gui(data_folder=corpus_folder, custom_styles={'edges': {'curve-style': 'haystack'}})
display(w.layout())
