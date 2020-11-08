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
