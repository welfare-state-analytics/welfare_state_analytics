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
#     name: python37564bitwelfarestateanalyticspipenvb857bd21a5fc4575b483276067dc0241
# ---

# %% [markdown]
# ### Process R data files

# %%
# %load_ext autoreload
# %autoreload 2

# pylint: disable=redefined-outer-name

import os
import sys
import warnings

import numpy as np
import pandas as pd
from IPython.core.interactiveshell import InteractiveShell

root_folder = os.path.join(
    os.getcwd().split("welfare_state_analytics")[0], "welfare_state_analytics"
)

sys.path = list(set(sys.path + [root_folder]))

from . import corpus_data

corpus_folder = os.path.join(root_folder, "data/textblock_politisk")

InteractiveShell.ast_node_interactivity = "all"

warnings.simplefilter(action="ignore", category=FutureWarning)

flatten = lambda l: [item for sublist in l for item in sublist]


# %%


def load_meta_text_blocks_as_data_frame(folder):
    """ Load censored corpus data """

    filename = os.path.join(folder, corpus_data.meta_textblocks_filename)
    df_meta = pd.read_csv(
        filename, compression="zip", header=0, sep=",", quotechar='"', na_filter=False
    )
    # df_meta = df_meta[['id', 'pred_bodytext']].drop_duplicates()
    # df_meta.columns = ["doc_id", "pred_bodytext"]
    # df_meta = df_meta.set_index("doc_id")
    return df_meta


def load_reconstructed_text_corpus(folder):
    filename = os.path.join(folder, corpus_data.reconstructed_text_corpus_file)
    if not os.path.isfile(filename):
        df_corpus = corpus_data.load_corpus_dtm_as_data_frame(folder)
        df_vocabulary = corpus_data.load_vocabulary_file_as_data_frame(folder)
        id2token = df_vocabulary["token"].to_dict()
        df_reconstructed_text_corpus = (df_corpus.groupby("document_id")).apply(
            lambda x: " ".join(
                flatten(x["tf"] * (x["token_id"].apply(lambda y: [id2token[y]])))
            )
        )
        df_reconstructed_text_corpus.to_csv(
            filename, compression="zip", header=0, sep=",", quotechar='"'
        )
    else:
        df_reconstructed_text_corpus = pd.read_csv(
            filename, compression="zip", header=None, sep=",", quotechar='"'
        )
        df_reconstructed_text_corpus.columns = ["document_id", "test"]
        df_reconstructed_text_corpus.set_index("document_id")

    return df_reconstructed_text_corpus


def plot_document_size_distribution(df_document):

    df_term_counts = df_document.groupby("term_count").size()

    dx = pd.DataFrame(
        {"term_count": list(range(0, df_term_counts.index.max() + 1))}
    ).set_index("term_count")
    df_term_counts = (
        dx.join(df_term_counts.rename("x"), how="left").fillna(0).astype(np.int)
    )

    ax = df_term_counts.plot.bar(figsize=(20, 10), rot=45)

    ticks = ax.xaxis.get_ticklocs()
    ticklabels = [l.get_text() for l in ax.xaxis.get_ticklabels()]
    ax.xaxis.set_ticks(ticks[::100])
    ax.xaxis.set_ticklabels(ticklabels[::100])

    return df_term_counts


def unique_documents_per_year_and_publication(df_document):
    df = (
        df_document.groupby(["year", "publication"])
        .agg(document_count=("doc_id", "nunique"))
        .reset_index()
        .set_index(["year", "publication"])
    )
    return df


def mean_tokens_per_year(df_document):
    df = (
        df_document.groupby(["year", "publication"])
        .agg(term_count=("term_count", "mean"))
        .reset_index()
        .set_index(["year", "publication"])
        .unstack("publication")
    )
    return df


# %% [markdown]
# ### Load DTM, document index and vocabulary
# This data is loaded from CSV files exported from R (drm1)

# %%

# df = load_meta_text_blocks_as_data_frame(corpus_folder)
# rt = load_reconstructed_text_corpus(corpus_folder)

df_corpus, df_document, df_vocabulary = corpus_data.load(corpus_folder)
id2token = df_vocabulary["token"].to_dict()

df_tf = df_corpus.groupby(["document_id"]).agg(term_count=("tf", "sum"))
df_document = df_document.merge(df_tf, how="inner", right_index=True, left_index=True)


# %% [markdown]
# ### Document size distribution
# %%

_ = plot_document_size_distribution(df_document)

# print(df.describe())


# %% [markdown]
# ### Number of documents per year and publication

# %%


unique_yearly_docs = unique_documents_per_year_and_publication(df_document)

unique_yearly_docs.unstack("publication").plot(
    kind="bar", subplots=True, figsize=(20, 20), layout=(2, 2), rot=45
)


# %% [markdown]
# ### Numer of tokens per year and publication

# %%


df_mean_tokens_per_year = mean_tokens_per_year(df_document)
df_mean_tokens_per_year.to_excel("mean_tokens_per_year.xlsx")
# display(df_mean_tokens_per_year)
# df_mean_tokens_per_year.plot(kind='bar', subplots=True, figsize=(25,25), layout=(2,2), rot=45);


# %% [markdown]
# ### Print data sizes

# %%

print('Corpus metrics, source "dtm1.rds", arrays drm$i, drm$j, drm$v')
print("  {} max document ID".format(df_corpus.document_id.max()))
print("  {} unique document ID".format(df_corpus.document_id.unique().shape[0]))
print("  {} max token ID".format(df_corpus.token_id.max()))
print("  {} unique token ID".format(df_corpus.token_id.unique().shape[0]))

print('Document metrics, source "dtm1.rds", arrays drm$dimnames[1]')
print("  {} max ID".format(df_document.index.max()))
print("  {} unique ID".format(df_document.index.unique().shape[0]))
print("  {} unique names".format(df_document.doc_id.unique().shape[0]))

print('Vocabulary metrics, source "dtm1.rds", arrays drm$dimnames[2]')
print("  {} max ID".format(df_vocabulary.index.max()))
print("  {} unique ID".format(df_vocabulary.index.unique().shape[0]))
print("  {} unique token".format(df_vocabulary.token.unique().shape[0]))

# df_document.groupby('doc_id').filter(lambda x: len(x) > 1).head()
