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

# %%

# import math

# import bokeh
# import bokeh.models
# import bokeh.plotting

# bokeh.plotting.output_notebook()
# plot_width = 600
# plot_height = 400

# x_ticks = [1920, 1921, 1922]

# data_source = bokeh.models.ColumnDataSource(
#     {
#         'xs': [[1920, 1921, 1922]],
#         'ys': [[5, 10, 8]],
#         'label': ["red"],
#         'color': ['red'],
#     }
# )

# p = bokeh.plotting.figure(plot_width=plot_width, plot_height=plot_height)
# p.y_range.start = 0
# p.yaxis.axis_label = 'Frequency'

# p.xaxis.ticker = x_ticks
# p.xaxis.major_label_orientation = math.pi / 4
# p.xaxis.major_label_text_font_style = 'bold'
# p.yaxis.major_label_overrides = {1921: None}
# p.xgrid.grid_line_color = None
# p.ygrid.grid_line_color = None

# _ = p.multi_line(xs='xs', ys='ys', legend_field='label', line_color='color', source=data_source)

# p.legend.location = "top_left"
# p.legend.click_policy = "hide"
# p.legend.background_fill_alpha = 0.0

# bokeh.plotting.show(p)
# # %%
