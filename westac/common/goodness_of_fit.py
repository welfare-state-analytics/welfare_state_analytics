
import scipy
import numpy as np
import math

import statsmodels.api as sm
import pandas as pd

from numpy.polynomial.polynomial import Polynomial as polyfit

def gof_by_l2_norm(matrix, axis=1):

    """ Computes L2 norm for rows (axis = 1) or columns (axis = 0).

    See stats.stackexchange.com/questions/25827/how-does-one-measure-the-non-uniformity-of-a-distribution

    Measures distance tp unform distribution (1/sqrt(d))

    "The lower bound 1d√ corresponds to uniformity and upper bound to the 1-hot vector."

    "It just so happens, though, that the L2 norm has a simple algebraic connection to the χ2 statistic used in goodness of fit tests:
    that's the reason it might be suitable to measure non-uniformity"

    """
    d = matrix.shape[int(not axis)]

    l2_norm = (np.linalg.norm(matrix, axis=axis) * math.sqrt(d) - 1 ) / (math.sqrt(d) - 1)

    return l2_norm


def fit_ordinary_least_square(ys, xs=None):
    """[summary]

    Parameters
    ----------
    ys : array-like
        observations
    xs : array-like, optional
        categories, domain, by default None

    Returns
    -------
    tuple m, k, p, (x1, x2), (y1, y2)
        where y = k * x + m
          and p is the p-value
    """
    if xs is None:
        xs = np.arange(len(ys)) #0.0, len(ys), 1.0))

    xs = sm.add_constant(xs)

    model = sm.OLS(endog=ys, exog=xs)
    result = model.fit()
    coeffs = result.params
    predicts = result.predict()
    (x1, x2), (y1, y2) = (xs[0][1], xs[-1][1]), (predicts[0],  predicts[-1])

    return coeffs[0], coeffs[1], result.pvalues, (x1, x2), (y1, y2)

def fit_ordinary_least_square_ravel(Y, xs):

    if xs is None:
        xs = np.arange(Y.shape[0])

    xsr = np.repeat(xs, Y.shape[1])
    ysr = Y.ravel()

    return fit_ordinary_least_square(ys=ysr, xs=xsr)

def fit_polynomial(ys, xs=None, deg=1):

    if xs is None:
        xs = np.arange(len(ys))

    return polyfit.fit(xs, ys, deg).convert().coef

def fit_polynomial_ravel(Y, xs):

    #xs = np.arange(x_corpus.document_index.year.min(), x_corpus.document_index.year.max() + 1, 1)

    # Layout columns as a single y-vector using ravel. Repeat x-vector for each column
    xsr = np.repeat(xs, Y.shape[1])
    ysr = Y.ravel()
    kx_m = np.polyfit(x=xsr, y=ysr, deg=1)

    return kx_m

def gof_chisquare_to_uniform(f_obs, axis=0):

    (chi2, p) = scipy.stats.chisquare(f_obs, axis=axis)

    return (chi2, p)

def earth_mover_distance(vs, ws=None):

    if ws is None:
        ws = np.full(len(vs), vs.mean())

    return scipy.stats.wasserstein_distance(vs, ws)

def entropy(pk, qk=None):

    if qk is None:
        qk = np.full(len(pk), pk.mean())

    return scipy.stats.entropy(pk, qk)

def kullback_leibler_divergence(p, q):
    if q is None:
        q = np.full(len(p), p.mean())
    e = 0.00001 # avoid div by zero
    kld = np.sum( (p + e) * np.log( (p + e) / (q + e)) )
    return kld

def kullback_leibler_divergence_to_uniform(p):
    q = p.mean()
    if q == 0:
        return np.nan
    kld = np.sum(p * np.log(p / q))
    return kld

def compute_goddness_of_fits_to_uniform(x_corpus):

    xs_years = np.arange(1945, 1990, 1)

    df = pd.DataFrame(
        {
            'token': [ x_corpus.id2token[i] for i in range(0, x_corpus.data.shape[1]) ],
            'word_count': [ x_corpus.word_counts[x_corpus.id2token[i]] for i in range(0, x_corpus.data.shape[1]) ],
            'l2_norm': gof_by_l2_norm(x_corpus.data, axis=0),
        }
    )

    chi2_stats, chi2_p    = list(zip(*[ gof_chisquare_to_uniform(x_corpus.data[:,i]) for i in range(0, x_corpus.data.shape[1]) ]))
    ks, ms                = list(zip(*[ fit_polynomial(x_corpus.data[:,i], xs_years, 1) for i in range(0, x_corpus.data.shape[1]) ]))

    df['slope']           = ks
    df['intercept']       = ms

    df['chi2_stats']      = chi2_stats
    df['chi2_p']          = chi2_p

    df['min']             = [ x_corpus.data[:,i].min() for i in range(0, x_corpus.data.shape[1]) ]
    df['max']             = [ x_corpus.data[:,i].max() for i in range(0, x_corpus.data.shape[1]) ]
    df['mean']            = [ x_corpus.data[:,i].mean() for i in range(0, x_corpus.data.shape[1]) ]
    df['var']             = [ x_corpus.data[:,i].var() for i in range(0, x_corpus.data.shape[1]) ]

    df['earth_mover']     = [ earth_mover_distance(x_corpus.data[:,i]) for i in range(0, x_corpus.data.shape[1]) ]
    df['entropy']         = [ entropy(x_corpus.data[:,i]) for i in range(0, x_corpus.data.shape[1]) ]
    df['kld']             = [ kullback_leibler_divergence_to_uniform(x_corpus.data[:,i]) for i in range(0, x_corpus.data.shape[1]) ]

    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.skew.html
    df['skew']             = [ scipy.stats.skew(x_corpus.data[:,i]) for i in range(0, x_corpus.data.shape[1]) ]

    # df['ols_m_k_p_xs_ys'] = [ gof.fit_ordinary_least_square(x_corpus.data[:,i], xs=xs_years) for i in range(0, x_corpus.data.shape[1]) ]
    # df['ols_k']           = [ m_k_p_xs_ys[1] for m_k_p_xs_ys in df.ols_m_k_p_xs_ys.values ]
    # df['ols_m']           = [ m_k_p_xs_ys[0] for m_k_p_xs_ys in df.ols_m_k_p_xs_ys.values ]

    df.sort_values(['l2_norm'], ascending=False, inplace=True)

    # uniform_constant = 1.0 / math.sqrt(float(x_corpus.data.shape[0]))

    return df

def get_most_deviating_words(df, metric, n_count=500, ascending=False, abs_value=False):

    sx = df.reindex(df[metric].abs().sort_values(ascending=False).index) if abs_value \
            else df.nlargest(n_count, columns=metric).sort_values(by=metric, ascending=ascending)

    return sx\
        .reset_index()[['token', metric]]\
        .rename(columns={'token': metric + '_token'})

def compile_most_deviating_words(df, n_count=500):

    xf =      get_most_deviating_words(df, 'l2_norm', n_count)\
        .join(get_most_deviating_words(df, 'slope', n_count, abs_value=True))\
        .join(get_most_deviating_words(df, 'chi2_stats', n_count))\
        .join(get_most_deviating_words(df, 'earth_mover', n_count))\
        .join(get_most_deviating_words(df, 'kld', n_count))\
        .join(get_most_deviating_words(df, 'entropy', n_count))

    return xf
