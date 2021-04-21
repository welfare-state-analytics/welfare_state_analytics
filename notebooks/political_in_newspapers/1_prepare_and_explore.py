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

import __paths__  # isort:skip pylint: disable=import-error, unused-import

import os
import warnings
import zipfile

import numpy as np
import pandas as pd

from notebooks.political_in_newspapers import corpus_data

corpus_folder = '/data/westac/textblock_politisk'

warnings.simplefilter(action="ignore", category=FutureWarning)

flatten = lambda l: [item for sublist in l for item in sublist]


# %%


def load_meta_text_blocks_as_data_frame(folder):
    """ Load censored corpus data """

    filename = os.path.join(folder, corpus_data.meta_textblocks_filename)
    df_meta = pd.read_csv(filename, compression="zip", header=0, sep=",", quotechar='"', na_filter=False)
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
            lambda x: " ".join(flatten(x["tf"] * (x["token_id"].apply(lambda y: [id2token[y]]))))
        )
        df_reconstructed_text_corpus.to_csv(filename, compression="zip", header=0, sep=",", quotechar='"')
    else:
        df_reconstructed_text_corpus = pd.read_csv(filename, compression="zip", header=None, sep=",", quotechar='"')
        df_reconstructed_text_corpus.columns = ["document_id", "text"]
        df_reconstructed_text_corpus.set_index("document_id")

    return df_reconstructed_text_corpus


def plot_document_size_distribution(df_document):

    df_term_counts = df_document.groupby("term_count").size()

    dx = pd.DataFrame({"term_count": list(range(0, df_term_counts.index.max() + 1))}).set_index("term_count")
    df_term_counts = dx.join(df_term_counts.rename("x"), how="left").fillna(0).astype(np.int)

    ax = df_term_counts.plot.bar(figsize=(20, 10), rot=45)

    ticks = ax.xaxis.get_ticklocs()
    ticklabels = [lst.get_text() for lst in ax.xaxis.get_ticklabels()]
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
df_document = df_document.merge(df_tf, how="inner", right_index=True, left_on='document_id')


# %%

# Load DN 68, write Excel and ZP
dn68 = df_document[(df_document.publication == 'DAGENS NYHETER') & (df_document.year == 1968)]
rt = load_reconstructed_text_corpus(corpus_folder)

dn68_text = rt.merge(dn68, how='inner', left_index=True, right_on='document_id')[
    ['document_id', 'year', 'date', 'term_count', 'text']
]
dn68_text.columns = ['document_id', 'year', 'date', 'term_count', 'text']
dn68_text.to_excel('dn68_text.xlsx')
# dn68_text.to_csv('dn68_text.csv', sep='\t')


with zipfile.ZipFile('dn68.zip', 'w', zipfile.ZIP_DEFLATED) as out:
    i = 0
    for index, row in dn68_text.iterrows():
        i += 1
        filename = 'dn_{}_{}_{}.txt'.format(row['date'], index, 1)
        text = row['text']
        out.writestr(filename, text, zipfile.ZIP_DEFLATED)

# %% [markdown]
# ### Document size distribution
# %%

_ = plot_document_size_distribution(df_document)

# print(df.describe())


# %% [markdown]
# ### Number of documents per year and publication

# %%


unique_yearly_docs = unique_documents_per_year_and_publication(df_document)

unique_yearly_docs.unstack("publication").plot(kind="bar", subplots=True, figsize=(20, 20), layout=(2, 2), rot=45)


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
