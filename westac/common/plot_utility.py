import scipy
import scipy.stats
# import numpy as np
# import pandas as pd
# import statsmodels.api as sm
# import matplotlib
import matplotlib.pyplot as plt

from   westac.common import goodness_of_fit as gof
import westac.common.utility as utility

def plot_word_distributions(x_corpus, *words, xs=None, plot_trend_line=False):

    indices = [ x_corpus.token2id[w] for w in words if w in x_corpus.token2id ]

    if len(indices) == 0:
        print("Not found! Please select other token(s)")
        return

    p_values = [
        scipy.stats.chisquare(x_corpus.bag_term_matrix[:,i], axis=0)[1]
            for i in indices
    ]

    ols_results = [
        gof.fit_ordinary_least_square(x_corpus.data[:, indices[i]])
            for i in range(0, len(indices))
    ]

    ols_slope = [ z[1] for z in ols_results ]

    if xs is None:
        xs = [ i for i in range(x_corpus.bag_term_matrix.shape[0]) ]

    labels = [
        '{} p={:+.5f}, k={:+.5f}'.format(x_corpus.id2token[indices[i]], p_values[i], ols_slope[i])
            for i in range(len(indices))
    ]

    data = tuple(utility.flatten([ (xs, x_corpus.bag_term_matrix[:,i]) for i in indices ]))

    #plt.figure(figsize=(15,7))

    plt.plot(*data)

    for ols_result in ols_results:
        m, k, p_values, (x1, x2), (y1, y2) = ols_result
        plt.plot((x1 + xs[0], x2 + xs[0]), (y1, y2))

    plt.ylim((0, x_corpus.bag_term_matrix[:,indices].max()))
    plt.gca().legend(tuple(labels))
    plt.show()


# def plot_histogram(ys, bins=50, w=4, h=2):

#     plt.figure(figsize=(w,h))

#     y, _, _ = plt.hist(ys, bins, (0,1))

#     fit = lin_reg(y)

#     slope = fit.params[1]
#     p = fit.pvalues[1]
#     print(slope, p)

#     x1 = 0
#     x2 = 1
#     y1 = fit.predict()[0]
#     y2 = fit.predict()[-1]

#     plt.plot([x1, x2], [y1, y2])
#     plt.show()


class IntStepper():

    def __init__(self, min, max, step=1, callback=None, value=None, data=None):
        self.min = min
        self.max = max
        self.step = step
        self.value = value or min
        self.data = data or {}
        self.callback = callback

    def trigger(self):
        if callable(self.callback):
            self.callback(self.value, self.data)
        return self.value

    def next(self):
        self.value = self.min + (self.value - self.min + self.step) % (self.max - self.min)
        return self.trigger()

    def previous(self):
        self.value = self.min + (self.value - self.min - self.step) % (self.max - self.min)
        return self.trigger()

    def reset(self):
        self.value = self.min
        return self.trigger()
