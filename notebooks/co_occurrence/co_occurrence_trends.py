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
# ### Overview
#
# This notebook implements a processing pipeline from computes word co-occurrences from plain text. The term-term co-occurrence matrix is transformed into a DTM corpus that has a vocabulary consisting of token-pairs. The co-occurring word trends can hence be xxplored using the ordinary word trends analysis tools.
#
# For large corpora the processing time can be considerable and in such a case you
# should consider using the CLI-version of the processing pipeline.
#
# ### Text Processing Pipeline
#
# | | Building block | Arguments | Description | Configuration
# | -- | :------------- | :------------- | :------------- | :------------- |
# | ⚙ | <b>SetTagger</b>~~SpacyModel~~ | tagger | Set PoS tagger | spaCy
# | 📜| <b>LoadText</b> | reader_opts, transform_opts | Load text stream | config.yaml
# | 🔎 | <b>Tqdm</b> | ⚪ | Progress indicator | ⚪
# | ⌛ | <b>Passthrough</b> | ⚪ | Passthrough  | ⚪
# | 🔨 | Spacy<b>ToTaggedFrame</b> | tagger service | PoS tagging |
# | 💾 | <b>Checkpoint</b> | checkpoint_filename | Checkpoint (tagged frames) to file |
# | 🔨 | TaggedFrame<b>ToTokens</b> | extract_tagged_tokens_opts, filter_opts | Tokens extractor | User
# | 🔨 | <b>TokensTransform</b> | tokens_transform_opts | Tokens transformer | User
# | 🔨 | <b>Vocabulary</b> | ⚪ | Generate a token to integer ID mapping | ⚪
# | 🔨 | <b>ToDocumentContentTuple</b> | ⚪ | API adapter| ⚪
# | 🔨 | <i>Partition</i> | ⚪ | Partition corpus into subsets based on predicate | 'year'
# | 🔎 | <i>ToTTM</i> | ⚪ | Transform each partition to TTM matrices | User
# | 🔨 | <b>ToCoOccurrence</b> | ⚪| Transform TTM into data frame with normalized values | ⚪
# | 💾 | <i>Checkpoint</i> | checkpoint_filename| Store co-occurrence data frame | ⚪
# | 🔨 | <b>ToDTM</b> | vectorize_opts| Transform data frame into DTM | ⚪
# | 💾 | <b>Checkpoint</b> | checkpoint_filename| Checkpoint (DTM) to file | ⚪
#
#
# ### How the co-occurrence counts are computed
#
# The co-occurrences are computed using a sliding window of size (D + 1 + D) that word by word moves through each document in the corpus, and keeps count of how many windows each pair of words co-occur in. Note that all windows will (currenty) always have an odd number of words, and the reason for this is the conditioned co-occurrence described below.
#
# The computation is done by first creating a (streamed) windowed corpus, consisting of all windows. From this corpus a DTM (document-term-matrix) is created, giving counts for each word in each window. This DTM is then used to compute a TTM (term-term-matrix) simply by multiplying the DTM with a transposed version of itself.
#
# Also note that the process of generating windows currently ignores sentences, paragraphs etc.
#
# #### GUI elements
#
# | | Config element |  Description |
# | -- | :------------- | :------------- |
# | | Corpus type | Type of corpus, disabled since only text corpora are allowed in this notebook.
# | | Source corpus file | Select file (ZIP) or folder that contains the text documents.
# | | Output tag | String that will be prefix to result files. Only valid filename chars are allowed.
# | | Output folder | Target folder for result files.
# | | Part-of-speech groups | Groups of tags to include in DTM given corpus PoS-schema
# | | Filename fields | Specifies attribute values to be extracted from filenames
# | | Lemmatize | Use word's lemma (base form)
# | | To lower | Lowercase all words
# | | Only alphabetic | Filter out words that has at least one non-alphabetic character
# | | Only alphanumeric | Filter out words that has at least one non-alphanumeric character
# | | No stopwords | Remove common stopwords using NLTK language specific stopwords
# | | Extra stopwords | Additional stopwords to remove
# | | Create subfolder | Sore result in subfolder named `output tag` in `target folder`
# | | Context distance | Max allowed distance to window's center-most word
# | | Concept | If specified, filter out windows having a center-most word not in list of specified words
# | | Remove concept | If specified, filter out concept word's co-occurrence pairs
# | | Normalize | Normalize output based on selected time period's corpus size
# | | Smooth | Smooth curve using PCHIP interpolations.
#
#
# N.B. Note that PoS schema (e.g. SUC, Universal, ON5 Penn Treebank tag sets) and language must be set for each corpus. This, and other options, is specified in the _corpus configuration file_. For an example, please see _SSI.yaml_ inf the `resources` folder.
#
#
# ### Concept co-occurrence
#
# The algorithm allows for computing a conditioned co-occurrence, where the set of windows are constrained so that the center-most word must one of a number of specified (concept) words. This results in a set of co-occurrences that occur in close proximity (i.e. the max distance of D) of the center-most word.
#
# ### Word trends
#
# Specifiy words of interest in the textbox in the text box. You can use both __wildcards__ ```*``` and __regular expressions__ to widen your search. The
# words in the vocabulary that matches what you have specified will be listed in the selection box. Since using wildcards and regexps can result
# in a large number of words, only the `Word count` most frequent words are displayed. Refine your search if you get to many matches.
#
# The words are plotted when selected in the selection box. You can select and plot __multiple words__ by pressing CTRL, or using CTRL + arrow keys. The words are listed  in descending order by global corpus frequency.
#
# The regular expressions must be surrounded by vertical bars ```|```. To find words ending with ```tion```
# you can enter ```|.*tion$|```  The vertical
# bars is specified only so that the system can distinguish the regexp from normal words. The actual expression is ```^.*tion$```.
# The dot and star ```.*``` matches any character (the dot) any number of times (the ```*```). The dollar sign ```$``` indicates the word ending.
# So this expression matches all words that begins with any number of characters follwoed, by the character sequence ```tion``` at the end of the word.
# To match all words starting with `info`you can enter ```|^info.*|``` where ```^``` specifies the start of the word.
#
# The "words" in this case are co-occurrence pairs and to find instances matching "information" you could enter ```information*```, ```*information``` or ```*information*``` to match pairs starting with information, ending with information or containing information respectively.

# %%

from bokeh.plotting import output_notebook
from IPython.core.display import display
from penelope.notebook.co_occurrence import main_gui

import __paths__

output_notebook()
gui = main_gui.MainGUI(
    corpus_config="riksdagens-protokoll",
    corpus_folder=__paths__.corpus_folder,
    data_folder=__paths__.data_folder,
    resources_folder=__paths__.resources_folder,
    global_count_threshold=250,
)
display(gui.layout())

# %%
