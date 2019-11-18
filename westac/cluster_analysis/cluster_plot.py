
import itertools
import sklearn.cluster
import numpy as np
import pandas as pd
import westac.common as common
import westac.common.curve_fit as cf
import holoviews as hv
from holoviews import dim

def noop(x=None, p=None, max=None): pass

def cluster_plot(x_corpus, token_clusters, n_cluster, tick=noop, **kwargs):

    palette = itertools.cycle(bokeh.palettes.Category20[20])
    assert n_cluster <= token_clusters.cluster.max()

    xs = np.arange(x_corpus.document_index.year.min(), x_corpus.document_index.year.max() + 1, 1)

    token_ids = list(token_clusters[token_clusters.cluster==n_cluster].index)
    tick(1,max=len(token_ids))

    p = figure(plot_width=kwargs.get('plot_width', 600), plot_height=kwargs.get('plot_height', 400))
    p.xaxis.axis_label = 'Year'
    p.yaxis.axis_label = 'Frequency'

    Y = x_corpus.data[:,token_ids]
    xsr = np.repeat(xs, Y.shape[1])
    ysr = Y.ravel()
    p.scatter(xsr, ysr, size=3, color='green', alpha=0.1)

    tick(1)
    _, k, _, lx, ly = gof.fit_ordinary_least_square_ravel(x_corpus.data[:,token_ids], xs)
    p.line(lx, ly, line_width=0.6, color='green', alpha=0.8)

    xsp, ysp = cf.fit_curve_ravel(cf.polynomial3, xs, x_corpus.data[:,token_ids])
    p.line(xsp, ysp, line_width=1.0, color='blue', alpha=1)

    ys_cluster_center = clusters.result.cluster_centers_[n_cluster, :]
    p.line(xs, ys_cluster_center, line_width=2.0, color='black')

    xs_spline, ys_spline = cf.pchip_spline(xs, ys_cluster_center)
    p.line(xs_spline, ys_spline, line_width=2.0, color='green')

    tick(2)

    return p

def cluster_boxplot(x_corpus, token_clusters, n_cluster):

    xs = np.arange(x_corpus.document_index.year.min(), x_corpus.document_index.year.max() + 1, 1)

    token_ids = list(token_clusters[token_clusters.cluster==n_cluster].index)

    Y = x_corpus.data[:,token_ids]
    xsr = np.repeat(xs, Y.shape[1])
    ysr = Y.ravel()

    data = pd.DataFrame(data={'year': xsr, 'frequency': ysr})

    kind = hv.BoxWhisker # hv.Violin
    violin = kind(data, ('year', 'Year'), ('frequency', 'Frequency'))

    violin_opts = {
        'height': 600,
        'width': 900,
        #'violin_fill_color': 'green',
        'xrotation': 45,
        #'violin_width': 0.8
    }
    return violin.opts(**violin_opts)

def clusters_plot(token_clusters, tick=noop):

    tick()
    n_clusters = len(token_clusters.cluster.unique())

    cluster_hist, edges = np.histogram(token_clusters['cluster'], bins=range(0, n_clusters))
    tick()

    # Put the information in a dataframe
    df = pd.DataFrame({'cluster': cluster_hist,  'left': edges[:-1],  'right': edges[1:]})
    tick()

    p = figure(plot_height = 600, plot_width = 600,  title = 'Histogram of clusters', x_axis_label = 'Cluster',  y_axis_label = 'Number of tokens')
    tick()

    p.quad(bottom=0, top=df['cluster'],  left=df['left'], right=df['right'],  fill_color='green', line_color='black')
    tick()

    return p

def display_cluster(x_corpus, token_clusters, n_cluster, output_type='table'):

    if output_type == 'table':
        display(token_clusters[token_clusters.cluster==n_cluster])

       # df_clusters.to_csv('k_means_clusters.txt', sep='\t')