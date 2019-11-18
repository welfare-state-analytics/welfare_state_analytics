import math
import scipy
import numpy as np
import bokeh
import itertools
import westac.common.curve_fit as cf

def plot_distribution(xs, ys, plot=None, title='', color='navy', ticker_labels=None, smoothers=None, **kwargs):

    if plot is None:

        p = bokeh.plotting.figure(plot_width=kwargs.get('plot_width', 400), plot_height=kwargs.get('plot_height', 200))
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

    _ = p.scatter(xs, ys, size=2, color=color, alpha=1.0, legend_label=title)

    #p.line(xs, ys , line_width=2, color=color, alpha=0.5, legend_label=title)
    for smoother in smoothers or []:
        xs, ys = smoother(xs, ys)

    _ = p.line(xs, ys , line_width=2, color=color, alpha=0.5, legend_label=title)

    #p.vbar(x=xs, top=ys, width=0.5, color=color, alpha=0.1)
    #p.step(xs, ys, line_width=2, color=color, alpha=0.5)

    #_, _, _, lx, ly = gof.fit_ordinary_least_square(ys, xs)
    #p.line(x=lx, y=ly, line_width=1, color=color, alpha=0.6, legend_label=title)

    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    p.legend.background_fill_alpha = 0.0

    return p

def plot_distributions(x_corpus, xs, indices, n_count=4, start=0, columns=3, width=1000, height=600, smoothers=None):

    smoothers = smoothers or [
        cf.rolling_average_smoother('nearest', 3),
        cf.pchip_spline
    ]
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
            plot_width=width if columns is None else max(int(width/columns), 400),
            plot_height=height if columns is None else  max(int(height/columns), 300),
            ticker_labels=xs if columns is None else None,
            smoothers=smoothers
        )
        plots.append(plot)

    if columns is not None:
        plot = bokeh.layouts.gridplot([ plots[u:u+columns] for u in range(0, len(plots), columns) ])

    return plot

def plot_distribution2(ys, window_size, mode='nearest'):

    xs = np.arange(0, len(ys), 1)
    yw = scipy.ndimage.filters.uniform_filter1d(ys, size=window_size, mode=mode)
    xw = np.arange(0, len(yw), 1)

    #xp, yp = cf.fit_curve(cf.fit_polynomial3, xs, ys, step=0.1)
    xp, yp = cf.fit_polynomial4(xs, ys)
    p = bokeh.plotting.figure(plot_width=800, plot_height=600, )
    p.scatter(xs, ys, size=6, color='red', alpha=0.6, legend_label='actual')
    p.line(x=xs, y=ys, line_width=1, color='red', alpha=0.6, legend_label='actual')
    p.line(x=xp, y=yp, line_width=1, color='green', alpha=0.6, legend_label='poly')
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    p.line(x=xw, y=yw, line_width=1, color='green', alpha=1, legend_label='rolling')

    bokeh.plotting.show(p)
