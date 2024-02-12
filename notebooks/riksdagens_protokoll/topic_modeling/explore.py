# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
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
from typing import Callable

from bokeh.io import output_notebook
from IPython.display import display
from penelope import utility as pu
from penelope.notebook import topic_modelling as ntm

from notebooks.riksdagens_protokoll.topic_modeling import utility as utm

output_notebook()
pu.set_default_options()

current_state: Callable[[], utm.TopicModelContainer] = utm.TopicModelContainer.singleton
data_folder: str = os.path.join(__paths__.data_folder, "riksdagen_corpus_data")


def display_gux(cls, *, state: utm.TopicModelContainer, **kwargs):
    if state.inferred_topics is None:
        print("No model loaded. Please load, then rerun this cell")
        return None

    ui = cls(state=state, **kwargs).setup()
    display(ui.layout())
    return ui


# %% [markdown]
# ### <span style='color: green'>SETUP </span> Load Model<span style='float: right; color: red'>MANDATORY</span>
#

# %%
load_gui: utm.RiksprotLoadGUI = utm.RiksprotLoadGUI(data_folder=data_folder, state=current_state(), slim=True).setup()
display(load_gui.layout())
# %% [markdown]
# ### <span style='color: green'>PREPARE </span> Edit Topic Labels<span style='float: right; color: red'></span>
#

# %%
display_gux(ntm.EditTopicLabelsGUI, folder=load_gui.model_info.folder, state=current_state())
# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Display Topic's Word Distribution as a Wordcloud<span style='color: red; float: right'> TRY IT</span>

# %%
wc_ui = display_gux(ntm.WordcloudGUI, state=current_state())
if wc_ui:
    wc_ui.update_handler()

# %% [markdown]
# ### <span style='color: green;'>BROWSE</span> Find topic's documents by token<span style='color: red; float: right'>TRY IT</span>
# Displays documents having topics in which given token is in toplist of dominant words.

# %%
display_gux(utm.RiksprotFindTopicDocumentsGUI, state=current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic-Word Distribution<span style='color: red; float: right'>TRY IT</span>
#

# %%
display_gux(ntm.TopicWordDistributionGUI, state=current_state())
# %% [markdown]
# ### <span style='color: green;'>BROWSE</span> Browse Topic Documents<span style='color: red; float: right'>TRY IT</span>
#
# Displays documents in which a topic occurs above a given threshold.

# %%
display_gux(utm.RiksprotBrowseTopicDocumentsGUI, state=current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Trends over Time<span style='color: red; float: right'>RUN</span>

# %%
display_gux(utm.RiksprotTopicTrendsGUI, state=current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span>Topic Trends over Time (Multiple Lines)<span style='color: red; float: right'>RUN</span>

# %%
display_gux(utm.RiksprotTopicMultiTrendsGUI, state=current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Trends Overview<span style='color: red; float: right'>TRY IT</span>
# The topic shares  displayed as a scattered heatmap plot using gradient color based on topic's weight in documen (see [Stanford’s Termite software](http://vis.stanford.edu/papers/termite).
#

# %%
display_gux(utm.RiksprotTopicTrendsOverviewGUI, state=current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic Topic Network<span style='color: red; float: right'>TRY IT</span>
#
# Computes weighted graph of topics co-occurring in the same document. Topics are defined as co-occurring in a document if they both have a weight above given threshold. The edge weights are the number of co-occurrences (binary yes or no). Node size reflects topic proportions over the entire corpus computed in accordance to LDAvis topic proportions.

# %%
display_gux(utm.RiksprotTopicTopicGUI, state=current_state())

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Pivot Topic Network<span style='color: red; float: right'>TRY IT</span>
#

# %%
display_gux(ntm.PivotTopicNetworkGUI, state=current_state(), pivot_key_specs=current_state().pivot_key_specs)

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Focus-Topic Document Network<span style='color: red; float: right'>TRY IT</span>
#

# %%
display_gux(ntm.FocusTopicDocumentNetworkGui, state=current_state(), pivot_key_specs=current_state().pivot_key_specs)

# %% [markdown]
# ### <span style='color: green;'>VISUALIZE</span> Topic-Token  Network<span style='color: red; float: right'>TRY IT</span>

# %%
w = ntm.create_topics_token_network_gui(data_folder=data_folder, custom_styles={'edges': {'curve-style': 'haystack'}})
display(w.layout())

# %%
