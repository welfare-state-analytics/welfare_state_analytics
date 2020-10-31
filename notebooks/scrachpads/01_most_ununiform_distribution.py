# -*- coding: utf-8 -*-
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
# # Find word's that changes most over time
#
# ## Approach
#
# Compare word's distribution over time with a uniform distribution. Use as null hypothesis the belief that a word's distribution does not change over time. Filter out all the words for which there is no significance.
#
# https://en.wikipedia.org/wiki/Goodness_of_fit#Categorical_data
#
#
# ## Candidate methods
#
# See: https://stats.stackexchange.com/questions/25827/how-does-one-measure-the-non-uniformity-of-a-distribution
#
#
# ### DONE Chi-square test for a discrete uniform distribution
#
# A χ2 goodness-of-fit test is used to determine how (un)likely a data serie (i.e. the word's distribution over time) has been generate by a (discrete) uniform distribution. The actual word counts for each year are used since χ2 is not applicable to relative frequencies. As a rule of thumb, χ2 test requires each individual value to be greater or equal to 5.
#
# From [stats.stackexchange.com](stats.stackexchange.com/questions/25827/how-does-one-measure-the-non-uniformity-of-a-distribution):
#
# *If you have not only the frequencies but the actual counts, you can use a χ2 goodness-of-fit test for each data series. In particular, you wish to use the test for a discrete uniform distribution. This gives you a good test, which allows you to find out which data series are likely not to have been generated by a uniform distribution, but does not provide a measure of uniformity..... (I guess that the chi-squared statistic can be seen as a measure of uniformity, but it has some drawbacks, such as the lack of convergence, dependence on the arbitrarily placed bins, that the number of expected counts in the cells needs to be sufficiently large, etc. Which measure/test to use is a matter of taste though, and entropy is not without its problems either (in particular, there are many different estimators of the entropy of a distribution). To me, entropy seems like a less arbitrary measure and is easier to interpret.)*
#
#  $\tilde{\chi}^2=\frac{1}{d}\sum_{k=1}^{n} \frac{(O_k - E_k)^2}{E_k}$ (d degree of freedom, n samples, E expected, O observed)
#
# References:
#
#   - [Chi-squared test](https://en.wikipedia.org/wiki/Chi-squared_test)
#   - [comparing-two-word-distributions](stats.stackexchange.com/questions/236192/comparing-two-word-distributions)
#
# DONE ### Simple linear regression
#
# Use least-squares fit to compute a Compare word's distribution over time with a uniform distribution. Use as null hypothesis the belief that a word's distribution does not change over time. Filter out all the words for which there is no significance.
#
# | Slope | $y = k * x + m$ | Use linear regression to compute slope k. Select n word having highest absoulute value |
#
# ### G-test for a discrete uniform distribution
#
# en.wikipedia.org/wiki/G-test
#
# ### Kolmogorov-Smirnov test (KS-test)
#
# stackoverflow.com/questions/25208421/how-to-test-for-uniformity
#
# ### Entropy
#
# https://en.wikipedia.org/wiki/Entropy_%28information_theory%29
#
#
# *There are other possible approaches, such as computing the entropy of each series - the uniform distribution maximizes the entropy, so if the entropy is suspiciously low you would conclude that you probably don't have a uniform distribution. That works as a measure of uniformity in some sense.*
#
# ### Kullback-Leibler divergence (KS-test)
# Another suggestion would be to use a measure like the Kullback-Leibler divergence, which measures the similarity of two distributions.
#
#
# ### L2 norm
#
# *Here is a simple heuristic: if you assume elements in any vector sum to 1 (or simply normalize each element with the sum to achieve this), then uniformity can be represented by L2 norm, which ranges from 1d√ to 1, with d being the dimension of vectors. The lower bound 1d√ corresponds to uniformity and upper bound to the 1-hot vector. To scale this to a score between 0 and 1, you can use n∗d√−1d√−1, where n is the L2 norm.
#
# ```
# def gof_by_l2_norm(matrix, axis=1):
#     d = matrix.shape[int(not axis)]
#     l2_norm = (np.linalg.norm(matrix, axis=axis) * math.sqrt(d) - 1 ) / (math.sqrt(d) - 1)
#     return l2_norm
# ```
#
# https://stats.stackexchange.com/questions/248772/why-does-the-l2-norm-heuristic-work-in-measuring-uniformity-of-probability-distr?noredirect=1&lq=1
#
# ### Earth Mover Distance
#
# *The earth mover distance, also known as the Wasserstein metric, measures the distance between two histograms. Essentially, it considers one histogram as a number of piles of dirt and then assesses how much dirt one needs to move and how far (!) to turn this histogram into the other. You would measure the distance between your distribution and a uniform one over the days of the week.* (https://stats.stackexchange.com/a/178187)
#
#
# * [variance - How does one measure the non-uniformity of a distribution? - Cross Validated](https://stats.stackexchange.com/questions/25827/how-does-one-measure-the-non-uniformity-of-a-distribution)
# * [G-test - Wikipedia](https://en.wikipedia.org/wiki/G-test#Relation_to_the_chi-squared_test)
# * [g-test uniform distribution python - Sök på Google](https://www.google.com/search?q=g-test+uniform+distribution+python&oq=g-test+&aqs=chrome.0.69i59l2j69i57j0j69i60l2.3775j0j1&sourceid=chrome&ie=UTF-8)
# * [Python String Format Cookbook – mkaz.blog](https://mkaz.blog/code/python-string-format-cookbook/)
# * [The writing and reporting of assertions in tests](https://docs.pytest.org/en/2.8.7/assert.html)
#
# * [scipy.stats.entropy — SciPy v1.3.1 Reference Guide](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.entropy.html)
# * [python - Interpreting scipy.stats.entropy values - Stack Overflow](https://stackoverflow.com/questions/26743201/interpreting-scipy-stats-entropy-values)
# * [Plotting data with matplotlib — How to Think Like a Computer Scientist: Learning with Python 3](https://howtothink.readthedocs.io/en/latest/PvL_H.html)
# * [lint-analysis/33-cluster-auto.ipynb at master · davidmcclure/lint-analysis](https://github.com/davidmcclure/lint-analysis/blob/master/notebooks/2017/33-cluster-auto.ipynb)
# * [Search · sm.OLS fit.predict plot](https://github.com/search?p=2&q=sm.OLS+fit.predict+plot&type=Code)
# * [statsmodels.regression.linear_model.OLS — statsmodels](https://www.statsmodels.org/dev/generated/statsmodels.regression.linear_model.OLS.html#statsmodels.regression.linear_model.OLS)
# * [Scikit-learn Normalization mode (L1 vs L2 & Max) - Cross Validated](https://stats.stackexchange.com/questions/225564/scikit-learn-normalization-mode-l1-vs-l2-max)
# * [Pyplot tutorial — Matplotlib 3.1.1 documentation](https://matplotlib.org/3.1.1/tutorials/introductory/pyplot.html)
# * [matplotlib axis range - Sök på Google](https://www.google.com/search?q=matplotlib+axis+range&oq=matplotlib+axis&aqs=chrome.3.69i57j0l5.6542j0j4&sourceid=chrome&ie=UTF-8)
# * [patsy - Describing statistical models in Python — patsy 0.5.1+dev documentation](https://patsy.readthedocs.io/en/latest/)
# * [k-nearest neighbors algorithm - Wikipedia](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)
# * [chi squared - Comparing two word distributions - Cross Validated](https://stats.stackexchange.com/questions/236192/comparing-two-word-distributions)
# * [Chi-squared test - Wikipedia](https://en.wikipedia.org/wiki/Chi-squared_test)
# * [scipy - Calculating probability distribution from time series data in python - Stack Overflow](https://stackoverflow.com/questions/49293019/calculating-probability-distribution-from-time-series-data-in-python)
# * [scipy.stats.gaussian_kde — SciPy v1.3.1 Reference Guide](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gaussian_kde.html)
# * [What is the difference between trendline and regression line? - Quora](https://www.quora.com/What-is-the-difference-between-trendline-and-regression-line)
# * [How is the k-nearest neighbor algorithm different from k-means clustering? | Python Tutorial](https://pythonprogramminglanguage.com/How-is-the-k-nearest-neighbor-algorithm-different-from-k-means-clustering/)
# * [[1510.00012] Fast Discrete Distribution Clustering Using Wasserstein Barycenter with Sparse Support](https://arxiv.org/abs/1510.00012)
# * [Exceed Synonyms, Exceed Antonyms | Merriam-Webster Thesaurus](https://www.merriam-webster.com/thesaurus/exceed)
# * [python - Swapping 1 with 0 and 0 with 1 in a Pythonic way - Stack Overflow](https://stackoverflow.com/questions/1779286/swapping-1-with-0-and-0-with-1-in-a-pythonic-way/1779448#1779448)
#
# * [Issues · humlab-sead/sead_query_api](https://github.com/humlab-sead/sead_query_api/issues)
# * [scipy.stats.chi2_contingency — SciPy v0.14.0 Reference Guide](https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.chi2_contingency.html)
# * [Capturing of the stdout/stderr output — pytest documentation](https://docs.pytest.org/en/latest/capture.html)
# * [sm.OLS np matrix - Sök på Google](https://www.google.com/search?q=sm.OLS+np+matrix&oq=sm.OLS+np+matrix&aqs=chrome..69i57.11997j0j4&sourceid=chrome&ie=UTF-8)
# * [scipy.stats.chi2_contingency — SciPy v1.3.1 Reference Guide](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.chi2_contingency.html#scipy.stats.chi2_contingency)
# * [G-test - Wikipedia](https://en.wikipedia.org/wiki/G-test)
# * [OsloMovers/Web.csproj at bd8a6c91aa957bf8a4857eaccbc4b383c711ddfb · webmasterdevlin/OsloMovers](https://github.com/webmasterdevlin/OsloMovers/blob/bd8a6c91aa957bf8a4857eaccbc4b383c711ddfb/Web/Web.csproj)
#

# %%
# %load_ext autoreload
# %autoreload 2

# pylint: disable=wrong-import-position

import itertools
import sys

import bokeh
import pandas as pd
import penelope.utility as utility
from bokeh.io import output_file, output_notebook, show
from bokeh.plotting import figure
from penelope.common import goodness_of_fit as gof
from penelope.corpus import vectorized_corpus

sys.path = ['/home/roger/source/welfare_state_analytics'] + sys.path


logger = utility.setup_logger(filename='./westac.log')
output_notebook()

# %%

v_y_corpus = (
    vectorized_corpus.VectorizedCorpus.load(
        'SOU_1945-1989_L0_+N_+S', folder='/home/roger/source/welfare_state_analytics/output'
    )
    .group_by_year()
    .slice_by_n_count(10000)
    .slice_by_n_top(100000)
)

v_ny_corpus = v_y_corpus.normalize(axis=1).normalize(axis=0)

v_m_corpus = v_y_corpus.normalize(axis=1, keep_magnitude=True)

_ = v_ny_corpus.stats()


# %% [markdown]
# Compute (ordinary least square) linear regression on corpus nomalized by each years volume, but keeping the magnitude.callable
# The first year's values are kept, while rest of the years are scaled based on each year's token count compared to first years token count.

# %%

df_fits = gof.compute_goddness_of_fits_to_uniform(v_y_corpus)
df_fits.sort_values('l2_norm', ascending=False).head()


# %% [markdown]
# Two axes:
# https://stackoverflow.com/questions/25199665/one-chart-with-two-different-y-axis-ranges-in-bokeh
#

# %%


def plot_word_distribution(x_corpus, df):  # pylint: disable=too-many-locals

    tokens = list(df.token)

    colors = itertools.cycle(bokeh.palettes.Dark2[8])

    min_year = x_corpus.document_index.year.min()
    max_year = x_corpus.document_index.year.max()

    years = [str(y) for y in range(min_year, max_year + 1)]

    tokens_data = {token: x_corpus.get_word_vector(token) for token in tokens}

    tokens_data['year'] = years
    tokens_data['current'] = tokens_data[tokens[0]]

    source = bokeh.models.ColumnDataSource(tokens_data)

    # for token in tokens:
    #    color = next(colors)
    #    p.circle(x='year', y=token, size=5,source=source, color=color)
    #    p.line(x='year', y=token, source=source, color=color)

    color = next(colors)

    plot = figure(plot_width=800, plot_height=400, title="Word distribution", x_range=years, name="token_plot_01")

    plot.xgrid.grid_line_color = None
    plot.xaxis.axis_label = "Year"
    plot.xaxis.major_label_orientation = 1.2

    _ = plot.circle(x='year', y='current', size=5, source=source, color=color)
    _ = plot.line(x='year', y='current', source=source, color=color)

    # line_source = bokeh.models.ColumnDataSource({'x0': [], 'y0': [], 'x1': [], 'y1': []})
    # sr = p.segment(x0='x0', y0='y0', x1='x1', y1='y1', color='olive', alpha=0.6, line_width=3, source=line_source, )

    slider = bokeh.models.Slider(start=0, end=len(tokens) - 1, value=0, step=1, title="Word")
    next_button = bokeh.models.Button(label=">>")

    token_vectors = [tokens_data[w] for w in tokens]

    update_plot = bokeh.models.CustomJS(
        args=dict(source=source, tokens=tokens, token_vectors=token_vectors, title=plot.title),
        code="""
        try {
            // var plot = Bokeh.documents[0].get_model_by_name('token_plot_01');
            const data = source.data;
            const token_index = cb_obj.value;
            const y = data['current'];
            for (let i = 0; i < y.length; i++)
                y[i] = token_vectors[token_index][i];
            div.text = tokens[token_index];
            title.text = tokens[token_index].toUpperCase();
            //div.text = plot.title;
            source.change.emit();
        } catch (ex) {
            div.text = ex.toString();
        }
    """,
    )

    next_plot = bokeh.models.CustomJS(
        args=dict(slider=slider),
        code="""
        try {
            slider.value = slider.value + 1;
            slider.trigger('change');
        } catch (ex) {
            div.text = ex.toString();
        }
    """,
    )
    # find_token = CustomJS(args=dict(title=plot.title, token2id), code="""
    #     token = cb_obj.value
    # """)
    # text = TextInput(title='Enter title', value='my sine wave', callback=update_title)

    div = bokeh.models.Div(text="", width=400, height=10)
    # update_plot.args['div'] = div
    # update_plot.args['plot'] = plot
    slider.js_on_change('value', update_plot)
    next_button.js_on_click(next_plot)

    w = bokeh.layouts.grid([div, next_button, plot, slider])  # , sizing_mode='stretch_both')

    return w


widget = plot_word_distribution(v_ny_corpus, df_fits.head(2000))
output_file('word_browse.html', title='change title')
show(widget)


# %%

chi2_stats, chi2_pvalues = list(
    zip(*[gof.gof_chisquare_to_uniform(v_m_corpus.data[:, i]) for i in range(0, v_m_corpus.data.shape[1])])
)
dx = pd.DataFrame({'chisquare': chi2_stats, 'pvalues': chi2_pvalues})
dx.chisquare.hist(bins=1000)
