# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
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

import pandas as pd
from bokeh.io import output_notebook
from IPython.display import display
from penelope import utility as pu
from penelope.notebook import topic_modelling as ntm

import westac.riksprot.parlaclarin.speech_text as sr
from notebooks.riksdagens_protokoll import topic_modeling as wtm
from westac.riksprot.parlaclarin import codecs as md

jj = os.path.join
output_notebook()
pu.set_default_options()

current_state: ntm.TopicModelContainer = ntm.TopicModelContainer.singleton

current_version: str = "v0.4.1"

data_folder: str = jj(__paths__.data_folder, "riksdagen_corpus_data")
codecs_filename: str = jj(data_folder, f"metadata/riksprot_metadata.{current_version}.db")
speech_index_filename: str = jj(data_folder, f'tagged_frames_{current_version}_speeches.feather/document_index.feather')
speech_folder: str = jj(data_folder, f'tagged_frames_{current_version}')

person_codecs: md.PersonCodecs = md.PersonCodecs().load(source=codecs_filename)
speech_index: pd.DataFrame = pd.read_feather(speech_index_filename)
speech_repository: sr.SpeechTextRepository = sr.SpeechTextRepository(
    source=speech_folder,
    person_codecs=person_codecs,
    document_index=speech_index,
)

default_args: dict = dict(person_codecs=person_codecs, speech_repository=speech_repository, state=current_state())

# %% [markdown]
# ### <span style='color: green'>SETUP </span> Load Model<span style='float: right; color: red'>MANDATORY</span>
#

# %%
load_gui: wtm.RiksprotLoadGUI = wtm.RiksprotLoadGUI(
    person_codecs, corpus_folder=data_folder, state=current_state(), slim=True
).setup()
display(load_gui.layout())
# %% [markdown]
# ### <span style='color: green'>PREPARE </span> Edit Topic Labels<span style='float: right; color: red'></span>
#

# %%
display(ntm.EditTopicLabelsGUI(folder=load_gui.loaded_model_folder, state=current_state()).setup().layout())

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
display(wtm.RiksprotFindTopicDocumentsGUI(**default_args).setup().layout())

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
display(wtm.RiksprotBrowseTopicDocumentsGUI(**default_args).setup().layout())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Trends over Time<span style='color: red; float: right'>RUN</span>

# %%
display(wtm.RiksprotTopicTrendsGUI(**default_args).setup().layout())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span>Topic Trends over Time (Multiple Lines)<span style='color: red; float: right'>RUN</span>

# %%
ui: wtm.RiksprotTopicMultiTrendsGUI = wtm.RiksprotTopicMultiTrendsGUI(**default_args).setup()
display(ui.layout())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Trends Overview<span style='color: red; float: right'>TRY IT</span>
# The topic shares  displayed as a scattered heatmap plot using gradient color based on topic's weight in documen (see [Stanford’s Termite software](http://vis.stanford.edu/papers/termite).
#

# %%
display(wtm.RiksprotTopicTrendsOverviewGUI(**default_args).setup().layout())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Topic Network<span style='color: red; float: right'>TRY IT</span>
#
# Computes weighted graph of topics co-occurring in the same document. Topics are defined as co-occurring in a document if they both have a weight above given threshold. The edge weights are the number of co-occurrences (binary yes or no). Node size reflects topic proportions over the entire corpus computed in accordance to LDAvis topic proportions.

# %%
display(wtm.RiksprotTopicTopicGUI(**default_args).setup().layout())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Pivot Topic Network<span style='color: red; float: right'>TRY IT</span>
#

# %%
display(
    ntm.PivotTopicNetworkGUI(pivot_key_specs=person_codecs.property_values_specs, state=current_state())
    .setup()
    .layout()
)

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Focus-Topic Document Network<span style='color: red; float: right'>TRY IT</span>
#

# %%
display(
    ntm.FocusTopicDocumentNetworkGui(pivot_key_specs=person_codecs.property_values_specs, state=current_state())
    .setup()
    .layout()
)

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic-Token  Network<span style='color: red; float: right'>TRY IT</span>

# %%
w = ntm.create_topics_token_network_gui(data_folder=data_folder, custom_styles={'edges': {'curve-style': 'haystack'}})
display(w.layout())

# %%
