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

def plot_distribution(xs, ys, plot=None, title='', color='navy', ticker_labels=None, smoothers=None, **kwargs):

    if plot is None:

        p = figure(plot_width=kwargs.get('plot_width', 400), plot_height=kwargs.get('plot_height', 200))
        p.y_range.start = 0
        #p.y_range.end = 0.5
        #p.title.text = title.upper()
        # p.yaxis.axis_label = 'Frequency'
        p.toolbar.autohide = True
        if ticker_labels is not None:
            p.xaxis.ticker = ticker_labels
        p.xaxis.major_label_orientation = math.pi / 2
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None

    else:
        p = plot

    sd = p.scatter(xs, ys, size=2, color=color, alpha=1.0, legend_label=title)

    #p.line(xs, ys , line_width=2, color=color, alpha=0.5, legend_label=title)
    for smoother in smoothers or []:
        xs, ys = smoother(xs, ys)

    ld = p.line(xs, ys , line_width=2, color=color, alpha=0.5, legend_label=title)

    #p.vbar(x=xs, top=ys, width=0.5, color=color, alpha=0.1)
    #p.step(xs, ys, line_width=2, color=color, alpha=0.5)

    #_, _, _, lx, ly = gof.fit_ordinary_least_square(ys, xs)
    #p.line(x=lx, y=ly, line_width=1, color=color, alpha=0.6, legend_label=title)

    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    p.legend.background_fill_alpha = 0.0

    return p

def plot_distributions(x_corpus, xs, indices, metric, n_count=4, start=0, columns=4):

    smoothers = [ cf.rolling_average_smoother('nearest', 3), cf.pchip_spline ]
    colors = itertools.cycle(bokeh.palettes.Category10[10])

    plots = []
    plot = None
    for i in range(0, n_count):
        plot = plot_distribution(
            xs,
            x_corpus.data[:,indices[start+i]],
            plot=plot if columns is None else None,
            title=x_corpus.id2token[indices[start+i]].upper(),
            color=next(colors),
            plot_width=1000 if columns is None else 400,
            plot_height=600 if columns is None else 300,
            ticker_labels=xs if columns is None else None,
            smoothers=smoothers
        )
        plots.append(plot)

    if columns is not None:
        plot = gridplot([ plots[u:u+columns] for u in range(0,len(plots),columns) ])

    show(plot)
