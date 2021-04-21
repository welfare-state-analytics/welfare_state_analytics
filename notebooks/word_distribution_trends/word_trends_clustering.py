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
# https://docs.scipy.org/doc/scipy/reference/cluster.vq.html
#

# %% [markdown]
# ## Word Distribution Cluster Analysis

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

# pylint: disable=wrong-import-position, import-error

import __paths__  # isort: skip

import os
import warnings

import holoviews as hv
import ipywidgets
import penelope.common.goodness_of_fit as gof
import penelope.corpus.dtm as vectorized_corpus
import penelope.notebook.cluster_analysis.cluster_analysis_gui as cluster_analysis_gui
import penelope.notebook.word_trends as word_trends
from bokeh.plotting import output_notebook
from IPython.display import display

root_folder = __paths__.ROOT_FOLDER
corpus_folder = os.path.join(root_folder, "output")

warnings.filterwarnings("ignore", category=FutureWarning)


output_notebook()

hv.extension("bokeh")


# %% [markdown]
# ## Load previously vectorized corpus
#
# The corpus was created using the following settings:
#  - Tokens were converted to lower case.
#  - Only tokens that contains at least one alphanumeric character (only_alphanumeric).
#  - Accents are ot removed (remove_accents)
#  - Min token length 2 (min_len)
#  - Max length not set (max_len)
#  - Numerals are removed (keep_numerals, -N)
#  - Symbols are removed (keep_symbols, -S)
#
# Use the `corpus_vectorizer` module to create a new corpus with different settings.
#
# The loaded corpus is processed in the following ways:
#
#  - Exclude tokens having a total word count less than 10000
#  - Include at most 50000 most frequent words words.
#  - Group documents by year (y_corpus).
#  - Normalize token count based on each year's total token count.
#  - Normalize token distrubution over years to 1.0 (n_corpus).
#

# %% tags=[]

y_corpus = vectorized_corpus.load_corpus(
    tag="SOU_test_L0_+N_+S",
    folder=os.path.join(root_folder, "output"),
    n_count=5000,
    n_top=50000,
    axis=1,
    keep_magnitude=False,
)

n_corpus = y_corpus.normalize(axis=0)


# %% [markdown]
# ## Deviation metrics
#
# These metrics are used to identify tokens that have a distribution that deviates the most to a uniform distribution.
# The following metrics are computed for each token:
#
# | Metric |  | |
# | :------ | :------- | :------ |
# | L2-norm | Measures distance to unform distribution. Lower bound 1/sqrt(d) is uniformity and upper bound is a 1-hot vector. |
# | Linear regression | Curve fitting to f(x) = k * x + m, where k is the slope, and m is the intercept to y axis when x is 0. A uniform curve has slope equals to 0. |
# | χ2 test | [Khan](https://www.khanacademy.org/math/statistics-probability/inference-categorical-data-chi-square-tests/chi-square-goodness-of-fit-tests/v/pearson-s-chi-square-test-goodness-of-fit) [Blog](https://towardsdatascience.com/inferential-statistic-understanding-hypothesis-testing-using-chi-square-test-eacf9fcac533) [Wikipedia](https://en.wikipedia.org/wiki/Chi-squared_test) |
# | Statistics | The min, max, mean and variance of the distribution |
# | Earth mover distance | [PDF](http://infolab.stanford.edu/pub/cstr/reports/cs/tr/99/1620/CS-TR-99-1620.ch4.pdf) [Wikipedia](https://en.wikipedia.org/wiki/Earth_mover%27s_distance)  |
# | Kullback-Leibler divergence | S = sum(pk * log(pk / qk), where pk is the token distribution and qk is the uniform distribution |
# | Entropy | Basically the same as KLD |
# | Skew | A measure of the "skewness" of the token distribution. See [scipy.stats.skew](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html#scipy.stats.skew) |
#
# References:
#
#  - [StackExchange](stats.stackexchange.com/questions/25827/how-does-one-measure-the-non-uniformity-of-a-distribution)
#     "It just so happens, though, that the L2 norm has a simple algebraic connection to the χ2 statistic used in goodness of fit tests:
#     that's the reason it might be suitable to measure non-uniformity"
#

# %%

# pylint: disable=redefined-outer-name


def display_uniformity_metrics(x_corpus, df_gof, df_most_deviating):

    output_row = [ipywidgets.Output(), ipywidgets.Output()]
    display(ipywidgets.HBox(output_row))

    with output_row[0]:

        columns = [
            "token",
            "word_count",
            "l2_norm",
            "slope",
            "chi2_stats",
            "earth_mover",
            "kld",
            "skew",
        ]
        display(df_gof.nlargest(10000, columns=["word_count"])[columns])

    with output_row[1]:
        pass
        # columns = ['L2 token', 'L2', 'K token', 'K', 'X2 token', 'X2', 'EMD token', 'EMD', 'KLD token', 'KLD', 'Ent. token', 'Entropy']
        # df_most_deviating.columns = columns
        # display(df_most_deviating.head(100))

    gof.plot_metrics(df_gof)
    gof.plot_slopes(x_corpus, df_most_deviating, "l2_norm")

    # df_gof.to_csv('df_gof.txt', sep='\t')
    # df_most_deviating.to_csv('df_most_deviating.txt', sep='\t')


# n_corpus.data.shape[0]

df_gof = gof.compute_goddness_of_fits_to_uniform(n_corpus)
df_most_deviating = gof.compile_most_deviating_words(df_gof, n_count=10000)


display_uniformity_metrics(n_corpus, df_gof, df_most_deviating)


# %% [markdown]
# ## Word distributions

# %%

tokens = df_most_deviating["l2_norm_token"]
word_trends.TrendsWithPickTokensGUI.create(n_corpus, tokens).layout()


# %% [markdown]
# ### Cluster Analysis (Hierarchical Clustering & K-means)
#

# %%

container = cluster_analysis_gui.display_gui(n_corpus, df_gof)


# %% [markdown]
#
# ### Some references
#
# |  |  |
# | :------- | :------- |
# | Overview: | https://docs.scipy.org/doc/scipy-0.19.0/reference/cluster.hierarchy.html |
# | Linkage: |https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.cluster.hierarchy.linkage.html#scipy.cluster.hierarchy.linkage |
# | Reference | Daniel Mullner, “Modern hierarchical, agglomerative clustering algorithms”, [arXiv:1109.2378v1](https://arxiv.org/abs/1109.2378v1). |
# |  | [Link](https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn/) [Link](https://github.com/giusdp/Clustering-Techniques/tree/0c78d1a893995c4603ed07216d645801ab4fdb4d)
#
# %%
