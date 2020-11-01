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
#     display_name: 'Python 3.7.5 64-bit (''welfare_state_analytics'': pipenv)'
#     language: python
#     name: python37564bitwelfarestateanalyticspipenvb8730e518d45450a918e95a98e055dbf
# ---

# %% [markdown]
# # Prepare Riksdagens Protokoll for Text Analysis
#
# ## Download Text from KB-LAB API (JSON)
#
# Use Python script `extract_json_text.py` to download `content.json` and `meta.json`. The script downloads queries packagers having "protocoll" tag (query `{ "tags": "protokoll" }`).
#
# ```bash
# cd source/westac_data
# pipenv shell
# cd src/kb_labb
# nohup python download_protocol_content_json.py >& run.log &
# ```
#
# The result is stored as a Zip archive.
#
# ### Extract text from JSON and store as a corpus of individual text files
#
# Use the script `extract_json_text.py` to extract the text from the JSON files.
#
# ```bash
# python extract_json_text.py --source-filename ~/tmp/riksdagens_protokoll_content.zip --target-filename ~/tmp/riksdagens_protokoll_content_corpus.zip
# ```
#
# The resulting Zip file contains the text files named as `prot_yyyyyy__NN.txt`. One file per protocoll.
#
# ### Vectorize the corpus to a BoW corpus (westac.VectorizedCorpus)
#
# Use the script `vectorize_protocols.py` to create a BoW corpus.
#
# ```bash
# python vectorize_protocols.py --source-filename ~/tmp/riksdagens_protokoll_content.zip --target-filename ~/tmp/riksdagens_protokoll_content_corpus.zip
# ```
#
# The script calls `generate_corpus` in `westac.corpus.corpus_vectorizer`:
#
# ```python
# import westac.corpus.corpus_vectorizer as corpus_vectorizer
#
# kwargs = dict( ...vectorize arguments...)
# corpus_filename = ...
# output_folder = ...
#
# corpus_vectorizer.generate_corpus(corpus_filename, output_folder=output_folder, **kwargs)
#
# ```
#
# The resulting corpus are stored in the specified output folder in two files; a numpy file containing the DTM and a Pythin pickled file with the dictionary and a document index.
#
# ### Prepare text files for Sparv
#
# The Sparv pipeline requires that the individual document are stored as (individual) XML files. The shell script `sparvit-to-xml` can be used to add a root tag to all text files in a Zip archive. The resulting XML files iare stored as a new Zip archive.
#
# ```bash
#  sparvit-to-xml --input riksdagens_protokoll_content_corpus.zip --output riksdagens_protokoll_content_corpus_xml.zip
#  ```

# %% tags=[] vscode={}
# %load_ext autoreload
# %autoreload 2

import os
import sys
import types

import bokeh.plotting
import notebooks.word_trends.corpus_gui as corpus_gui
import notebooks.word_trends.word_trends_gui as word_trends_gui

# from beakerx import *
# from beakerx.object import beakerx

root_folder = os.path.join(os.getcwd().split('welfare_state_analytics')[0], 'welfare_state_analytics')
corpus_folder = os.path.join(root_folder, 'data/riksdagens_protokoll')

sys.path = sys.path + [root_folder, globals()['_dh'][-1]]

bokeh.plotting.output_notebook(hide_banner=True)

container = types.SimpleNamespace(corpus=None, handle=None, data_source=None, data=None, figure=None)


# %% [markdown]
# # Load previously vectorized corpus
#
# The corpus was created with the following settings:
#  - Tokens were converted to lower case.
#  - Only tokens that contains at least one alphanumeric character (only_alphanumeric).
#  - Accents are ot removed (remove_accents)
#  - Min token length 2 (min_len)
#  - Max length not set (max_len)
#  - Numerals are removed (keep_numerals, -N)
#  - Symbols are removed (keep_symbols, -S)
#
# Use the `vectorize_protocol` script to create a new corpus with different settings.
#
# The corpus is processed in the following ways when loaded:
#
#  - Exclude tokens having a total word count less than `Min count`
#  - Include at most `Top count` most frequent words.
#  - Group and sum up documents by year.
#  - Normalize token distribution over years to 1.0
#

# %% tags=[] vscode={"end_execution_time": "2020-08-31T18:34:55.995Z", "start_execution_time": "2020-08-31T18:34:55.854Z"}


ui = corpus_gui.display_gui(corpus_folder, container=container)

# %%
word_trends_gui.display_gui(container)
