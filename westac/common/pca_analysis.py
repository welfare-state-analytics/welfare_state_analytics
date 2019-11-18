
import itertools
import bokeh

from sklearn.decomposition import PCA

def compute(data, n_components=2):
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(data)
    return X_pca

def plot(X_pca, clusters=None):

    if clusters is not None:
        palette = itertools.cycle(bokeh.palettes.Colorblind8)
        cmap = {
            x: next(palette) for x in set(clusters)
        }
        colors = [ cmap[x] for x in clusters ]
    else:
        colors = 'navy'

    p = bokeh.plotting.figure(plot_width=600, plot_height=600)

    # p.legend(loc="best", shadow=False, scatterpoints=1)

    p.title.text = 'PCA decomposition'

    p.xaxis.axis_label, p.yaxis.axis_label = 'pca 0', 'pca 1'

    _ = p.scatter(X_pca[:, 0], X_pca[:, 1] , size=6, fill_color=colors,  color=colors, alpha=0.5)

    bokeh.plotting.show(p)

    return p
