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
# ## Word, or part-of-word, trend analysis
#
# ### Text Processing Pipeline
# To be updated:
#
# | 🔽 | Building block | Arguments | Description | Configuration
# | -- | :------------- | :------------- | :------------- | :------------- |
# | ⚙ | <b>SetTagger</b>SpacyModel | 'en' | Set PoS tagger | spaCy
# | 📜| <b>LoadText</b> | reader_opts, transform_opts | Text stream provider | config.yml
# | 🔎 | <b>Tqdm</b> | ⚪ | Progress indicator | ⚪
# | ⌛ | <b>Passthrough</b> | ⚪ | Passthrough | ⚪
# | 🔨 | <b>ToTaggedFrame</b> | ⚪ Spacy | PoS tagging | config.yml
# | 💾 | <b>Checkpoint</b> | checkpoint_filename | Checkpoint (tagged frames) to file | ⚪
# | 🔨 | TaggedFrame<b>ToTokens</b> | extract_opts | Tokens extractor | User specified
# | 🔨 | <b>TokensTransform</b> | transform_opts | Tokens transformer | User specified
# | 🔎 | <b>Tqdm</b> | ⚪ | Progress indicator | ⚪
# | 🔨 | <b>ToDTM</b> | vectorize_opts| DTM vectorizer | User specified
# | 💾 | <b>Checkpoint</b> | checkpoint_filename| Checkpoint (DTM) to file | User specified
#
# ### User instructions
#
# #### Load a DTM corpus
#
# To load a corpus you must first select a file, then press <b>`Load`</b>. To select the file:</b> <b>1)</b> press <b>`Change`</b> to open the file browser, <b>2)</b> find and select the file you want to open and <b>3)</b> press <b>`Change`</b> again to confirm the file and close the file browser. Then you can load the corpus by pressing <b>`Load`</b>.
#
# #### Word trends
#
# Specifiy words of interest in the text box. You can use both wildcards and regular expressions to widen your search. The
# words in the vocabulary that matches what you have specified will be listed in the selection box. Since using wildcards and regexps can result
# in a large number of words, only the `Word count` matching most frequent words are displayed. Refine your search if you get to many matches.
# The words will be plotted when they are selected. You can select and plot multiple words by pressing CTRL when selected, or using arrow keys.
#
# The regular expressions must be surrounded by vertical bars `|`. To find words ending with `tion`
# you can enter `|.*tion$|` in the textbox. I might seem cryptical, but is a very powerful notation for searching words. The vertical
# bars is specified only so that the system can distinguish the regexp from "normal" words. The actual expression is `^.*tion$`.
# The dot and star`.*` matches any character (the dot) any number of times (the `*`). The dollar sign `$` indicates the word ending.
# So this expression matches all words that begins with any number of characters follwoed, by the character sequence `tion` at the end of the word.
# To match all words starting with `info`you can enter `|^info.*|` where `^` specifies the start of the word.
#
# %%
from bokeh.io import output_notebook
from IPython.display import display
from westac.riksprot.parlaclarin import metadata as md

from notebooks.riksdagens_protokoll.word_trends import word_trends_gui as wt

default_folder = '/data/riksdagen_corpus_data/dtm_1920-2020_v0.3.0.tf20'
output_notebook()  # resources=INLINE)

riksprot_metadata: md.ProtoMetaData = md.ProtoMetaData.load_from_same_folder(default_folder)

gui = wt.RiksProtTrendsGUI(default_folder=default_folder, riksprot_metadata=riksprot_metadata).setup()

display(gui.layout())
gui.load()

# %%
