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
# ## Word Significance using TF-IDF

# %%
# %load_ext autoreload
# %autoreload 2

# pylint: disable=wrong-import-position

import logging
import os
import sys
from typing import Mapping

import ipywidgets
import numpy as np
import pandas as pd
import penelope.corpus.dtm as vectorized_corpus
from IPython.display import display
from sklearn.feature_extraction.text import TfidfVectorizer

root_folder = os.path.abspath(os.path.join(globals()["_dh"][-1], "../.."))

corpus_folder = os.path.join(root_folder, "output")

sys.path = [root_folder] + sys.path


logger = logging.getLogger(__name__)


# %% [markdown]
# ## Load previously vectorized corpus
# # Use the `corpus_vectorizer` module to create a new corpus with different settings.
#
# The loaded corpus is processed in the following ways:
#
#  - Exclude tokens having a total word count less than 10
#  - Include at most 50000 most frequent words words.
#
# Compute a new TF-IDF weighted corpus
#
#  - Group document index by year
#  - Compute mean TF-IDF for each year
#
# Some references:
#
#  - [scikit-learn TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer)
#  - [Spark MLlib TF-IDF](https://spark.apache.org/docs/2.2.0/ml-features.html#tf-idf)
#
#

# %%

# pylint: disable=wrong-import-position

v_corpus: vectorized_corpus.VectorizedCorpus = (
    vectorized_corpus.VectorizedCorpus.load(tag="SOU_1945-1989_NN+VB+JJ_lemma_L0_+N_+S", folder=corpus_folder)
    .slice_by_n_count(10)
    .slice_by_n_top(500000)
)

tf_idf_corpus = v_corpus.tf_idf().group_by_year(aggregate="mean", fill_gaps=True)


# %%


def display_top_terms(data):

    if data is None:
        logger.info("No data to display!")
        return

    df = pd.DataFrame({k: [x[0] for x in v] for k, v in data.items()})

    display(df)


def compute_top_terms(x_corpus: vectorized_corpus.VectorizedCorpus, n_top: int, idx_groups=None) -> Mapping:

    data = {x["label"]: x_corpus.get_top_n_words(n=n_top, indices=x["indices"]) for x in idx_groups}
    return data


def display_gui(x_corpus: vectorized_corpus.VectorizedCorpus, x_documents_index: pd.DataFrame):

    lw = lambda w: ipywidgets.Layout(width=w)

    year_min, year_max = x_documents_index.year.min(), x_documents_index.year.max()
    years = list(range(year_min, year_max + 1))
    decades = [10 * decade for decade in range((year_min // 10), (year_max // 10) + 1)]
    lustrums = [lustrum for lustrum in range(year_min - year_min % 5, year_max - year_max % 5, 5)]

    groups = [
        (
            "year",
            [{"label": str(year), "indices": [year - year_min]} for year in years],
        ),
        (
            "decade",
            [
                {
                    "label": str(decade),
                    "indices": [year - year_min for year in range(decade, decade + 10) if year_min <= year <= year_max],
                }
                for decade in decades
            ],
        ),
        (
            "lustrum",
            [
                {
                    "label": str(lustrum),
                    "indices": [
                        year - year_min for year in range(lustrum, lustrum + 5) if year_min <= year <= year_max
                    ],
                }
                for lustrum in lustrums
            ],
        ),
    ]

    w_n_top = ipywidgets.IntSlider(
        description="#words",
        min=10,
        max=1000,
        value=100,
        tooltip="Number of words to compute",
    )
    w_compute = ipywidgets.Button(description="Compute", icon="", button_style="Success", layout=lw("120px"))
    w_output = ipywidgets.Output()  # layout={'border': '1px solid black'})
    w_groups = ipywidgets.Dropdown(options=groups, value=groups[0][1], description="Groups:")

    boxes = ipywidgets.VBox(
        [
            ipywidgets.HBox(
                [w_n_top, w_groups, w_compute],
                layout=ipywidgets.Layout(align_items="flex-end"),
            ),
            w_output,
        ]
    )

    display(boxes)

    def compute_callback_handler(*_):
        w_output.clear_output()
        with w_output:
            try:

                w_compute.disabled = True

                data = compute_top_terms(
                    x_corpus,
                    n_top=w_n_top.value,
                    idx_groups=w_groups.value,
                )

                display_top_terms(data)

            except Exception as ex:
                logger.error(ex)
            finally:
                w_compute.disabled = False

    w_compute.on_click(compute_callback_handler)


display_gui(tf_idf_corpus, tf_idf_corpus.document_index)


# %% jupyter={"source_hidden": true}

# %matplotlib inline


def plot_word(x_corpus: vectorized_corpus.VectorizedCorpus, word: str):
    wv = x_corpus.get_word_vector(word)

    df = pd.DataFrame({"count": wv, "year": x_corpus.document_index.year}).set_index("year")
    df.plot()


plot_word(v_corpus, "arbete")
plot_word(tf_idf_corpus, "arbete")
# plot_word(yearly_tf_idf_corpus, "arbete")


# %% jupyter={"source_hidden": true}


# np.nansum(a, axis=None, dtype=None, out=None, keepdims=<no value>)[source]
# np.nanmean(a, axis=None, dtype=None, out=None, keepdims=<no value>)[source]
docs = [
    "the house had a tiny little mouse",
    "the cat saw the mouse",
    "the mouse ran away from the house",
    "the cat finally ate the mouse",
    "the end of the mouse story",
]

tfidf_vectorizer = TfidfVectorizer(use_idf=True)
tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(docs)

tfidf = tfidf_vectorizer_vectors.todense()
# TFIDF of words not in the doc will be 0, so replace them with nan
tfidf[tfidf == 0] = np.nan
# Use nanmean of numpy which will ignore nan while calculating the mean
means = np.nansum(tfidf, axis=0)
# convert it into a dictionary for later lookup
means = dict(zip(tfidf_vectorizer.get_feature_names(), means.tolist()[0]))

tfidf = tfidf_vectorizer_vectors.todense()
# Argsort the full TFIDF dense vector
ordered = np.argsort(tfidf * -1)
words = tfidf_vectorizer.get_feature_names()

top_k = 5
for i, doc in enumerate(docs):
    result = {}
    # Pick top_k from each argsorted matrix for each doc
    for t in range(top_k):
        # Pick the top k word, find its average tfidf from the
        # precomputed dictionary using nanmean and save it to later use
        result[words[ordered[i, t]]] = means[words[ordered[i, t]]]
    print(result)
