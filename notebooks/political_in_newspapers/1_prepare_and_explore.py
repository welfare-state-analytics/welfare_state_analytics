# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
# ---

# %% [markdown]
# ### Process R data files

# %%
# %load_ext autoreload
# %autoreload 2

# pylint: disable=redefined-outer-name,no-member

import __paths__  # isort:skip pylint: disable=import-error, unused-import

from notebooks.political_in_newspapers import repository

CORPUS_FOLDER = '/data/westac/textblock_politisk'

# %% [markdown]
# ### Load DTM, document index and vocabulary
# Load data from CSV files exported from R (drm1)

# %%

source_corpus: repository.SourceCorpus = repository.SourceRepository.load(CORPUS_FOLDER)

# %%

_ = repository.plot_document_size_distribution(source_corpus.document_index)

source_corpus.corpus.head()
source_corpus.document_index.head()
source_corpus.info()

# %% [markdown]
# ### Document size distribution
# %%

# print(df.describe())


# %% [markdown]
# ### Number of documents per year and publication

# %%


unique_yearly_docs = repository.unique_documents_per_year_and_publication(source_corpus.document_index)

unique_yearly_docs.unstack("publication").plot(kind="bar", subplots=True, figsize=(20, 20), layout=(2, 2), rot=45)

mean_tokens_per_year = repository.mean_tokens_per_year(source_corpus.document_index)
mean_tokens_per_year.to_excel("mean_tokens_per_year.xlsx")
# display(mean_tokens_per_year)
mean_tokens_per_year.plot(kind='bar', subplots=True, figsize=(25, 25), layout=(2, 2), rot=45)


# %% [markdown]
# ### Extract text for DN 1968
# %%

# repository.ExtractDN68.extract(folder=CORPUS_FOLDER, document_index=source_corpus.document_index)
