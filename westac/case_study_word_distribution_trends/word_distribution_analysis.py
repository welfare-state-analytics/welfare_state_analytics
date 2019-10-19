# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Iteration #1: Create word frequency distribution over years

# +
# %load_ext autoreload
# %autoreload 2

import os, sys
sys.path = list(set(sys.path + [ '../common' ]))

import westac.common.corpus_vectorizer as corpus_vectorizer
import westac.common.text_corpus as text_corpus
import westac.common.utility as utility
import westac.common.file_text_reader as file_text_reader
import numpy as np
import sklearn

# -

# # Helpers

def create_corpus(filename):
    meta_extract = dict(year=r".{5}(\d{4})_.*", serial_no=r".{9}_(\d+).*")
    reader = file_text_reader.FileTextReader(filename, meta_extract=meta_extract, compress_whitespaces=True, dehyphen=True)
    kwargs = dict(isalnum=False, to_lower=False, deacc=False, min_len=2, max_len=None, numerals=False)
    corpus = text_corpus.ProcessedCorpus(reader, **kwargs)
    return corpus


# # Analysis
# https://github.com/davidmcclure/lint-analysis/tree/master/notebooks/2017
#

# ## Goodness-of-fit to uniform distribution (chi-square)
#
# See [scipy.stats.chisquare](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chisquare.html):
# "*When just f_obs is given, it is assumed that the expected frequencies are uniform...*"
#

# +
import numpy as np
from scipy import stats

#filename = './test/test_data/test_corpus.zip'
#filename = './data/Sample_1945-1989_1.zip'

filename = './data/SOU_1945-1989.zip'
dump_name = os.path.basename(filename).split('.')[0]

vectorizer = corpus_vectorizer.CorpusVectorizer()
corpus = create_corpus(filename)
X = vectorizer.fit_transform(corpus)
vectorizer.dump(dump_name, folder='./output')

#vectorizer.load(dump_name, folder='./output')


if False:

    Y         = vectorizer.group_by_year()
    Yn        = vectorizer.normalize(Y, axis=1, norm='l1')
    Ynw       = vectorizer.slice_tokens_by_count_threshold(Yn, 1)
    Yx2, imap = vectorizer.pick_by_top_variance(500)

    stats.chisquare(Ynw, f_exp=None, ddof=0, axis=0)

# -

# # Ward Clustering
#
# See [this](https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/) tutorial.
#

# +
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

linked = linkage(Z.todense(), 'ward')

labelList = tokens_of_interest

plt.figure(figsize=(10, 7))
dendrogram(linked, orientation='top', labels=labelList, distance_sort='descending', show_leaf_counts=True)
plt.show()


# +
# df_Zy.sum().where(lambda x: x>= 10000).sort_values().dropna()

# +
#Xn = normalize(X, axis=1, norm='l1')
#Y = group_by_year_matrix(X, df_documents)
#df = pd.DataFrame(Y, columns=list(vectorizer.get_feature_names()))
#df.to_excel('test.xlsx')

if False:

    df = pd.DataFrame(X.toarray(), columns=list(vectorizer.get_feature_names()))
    df['year'] = df.index + 45
    df = df.set_index('year')
    df['year'] =  pd.Series(df.index).apply(lambda x: documents[x][0])
    %matplotlib inline
    df[['krig']].plot() #.loc[df["000"]==49]


