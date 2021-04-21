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
# ## Word, or part-of-word, trend analysis
#
# ### Text Processing Pipeline
#
# | 🔽 | Building block | Arguments | Description | Configuration
# | -- | :------------- | :------------- | :------------- | :------------- |
# | ⚙ | <b>SetTagger</b>SpacyModel | 'en' | Set PoS tagger | spaCy
# | 📜| <b>LoadText</b> | reader_opts, transform_opts | Text stream provider | config.yaml
# | 🔎 | <b>Tqdm</b> | ⚪ | Progress indicator | ⚪
# | ⌛ | <b>Passthrough</b> | ⚪ | Passthrough | ⚪
# | 🔨 | <b>ToTaggedFrame</b> | ⚪ Spacy | PoS tagging | config.yaml
# | 💾 | <b>Checkpoint</b> | checkpoint_filename | Checkpoint (tagged frames) to file | ⚪
# | 🔨 | TaggedFrame<b>ToTokens</b> | extract_tagged_tokens_opts, filter_opts | Tokens extractor | User specified
# | 🔨 | <b>TokensTransform</b> | tokens_transform_opts | Tokens transformer | User specified
# | 🔨 | <b>ToDocumentContentTuple</b> | ⚪ | API adapter | ⚪
# | 🔎 | <b>Tqdm</b> | ⚪ | Progress indicator | ⚪
# | 🔨 | <b>ToDTM</b> | vectorize_opts| DTM vectorizer | User specified
# | 💾 | <b>Checkpoint</b> | checkpoint_filename| Checkpoint (DTM) to file | User specified
#
# ### User instructions
#
# #### Compute DTM
#
# This notebook implements the entire processing pipeline from plain text to a computed (and stored)
# document-term matrix (DTM) that are the basis for the word trend exploration.
#
# For large corpora the DTM processing time can be considerable and in such a case you
# should consider using the CLI-version of the processing pipeline.
#
# Note that the computed DTM is saved on disk in the specified folder. You must enter a
# tag that will be used when naming the (principal) result data file. This file will be named
# "tag + `_vectorized_data.pickle`" and will be used to uniquely identify the bundle of files that makes
# up the result. Note that if the `tag` already exists in the specified target folder then it will
# be overwritten. You can use the tag to describe the arguments of the computation i.e. PoS tags etc. If the `Create folder` option is checked, then the result bundle will be stored in a subfolder of the target folder named _tag_.
#
# #### Compute arguments
#
# | | Config element |  Description |
# | -- | :------------- | :------------- |
# | | Corpus type | Type of corpus, disabled since only text corpora are allowed in this notebook.
# | | Source corpus file | Select file (ZIP) or folder that contains the text documents.
# | | Output tag | String that will be prefix to result files. Only valid filename chars allowed.
# | | Output folder | Target folder for result files.
# | | Part-of-speech groups | Groups of tags to include in DTM given corpus PoS-schema
# | | Remove stopwords | Remove common stopwords using NLTK language specific stopwords
# | | Extra stopwords | Additional stopwords
# | | Filename fields | Specifies attribute values to be extracted from filenames
#
# N.B. Note that PoS schema (e.g. SUC, Universal, ON5 Penn Treebank tag sets) and language must be set for each corpus.
#  This, and other options, is specified in the _corpus configuration file_. For an example, please see _SSI.yaml_ inf the `resources` folder.
#
# #### Load a DTM corpus
#
# To load an existing corpus youmust first select a file, then press <b>`Load`</b>. To select the file:</b> <b>1)</b> press <b>`Change`</b> to open the file browser, <b>2)</b> find and select the file you want to open and <b>3)</b> press <b>`Change`</b> again to confirm the file and close the file browser. Then you can load the corpus by pressing <b>`Load`</b>.
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
#
#
# %%


import penelope.notebook.word_trends.main_gui as main_gui
from bokeh.plotting import output_notebook
from IPython.core.display import display

import __paths__

output_notebook()
main_gui.CLEAR_OUTPUT = True
gui = main_gui.create_to_dtm_gui(
    corpus_config="riksdagens-protokoll",
    corpus_folder=__paths__.corpus_folder,
    data_folder=__paths__.data_folder,
    resources_folder=__paths__.resources_folder,
)
display(gui)

# %%
