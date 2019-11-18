import types
import ipywidgets
import bokeh
from IPython.display import display

import westac.notebooks_gui.cluster_plot as cluster_plot

import westac.cluster_analysis.cluster_k_means as k_means
import westac.cluster_analysis.cluster_hca as hca

CLUSTER_METHODS = {
    'k_means': types.SimpleNamespace(
        key='k_means',
        title='K-means',
        max_clusters=200,
        algorithm=k_means.compute
    ),
    'hca': types.SimpleNamespace(
        key='hca',
        title='HCA',
        max_clusters=None,
        algorithm=hca.compute,
        threshold=(0.0,1.0)
    )
}

def display_gui(x_corpus):

    data = None

    n_clusters = ipywidgets.IntSlider(description='Cluster count', min=1, max=200, step=1, value=20,  bar_style='info')
    method = ipywidgets.Dropdown(description='Method', values=[('k_means', 'K-means'), ('hca', 'HCA')], value='k_means')
    progress = ipywidgets.IntProgress(description='', min=0, max=10, step=1, value=0, continuous_update=False, layout=ipywidgets.Layout(width='98%'))
    compute = ipywidgets.Button(description='Compute', layout=ipywidgets.Layout(width='100px'))

    n_cluster = ipywidgets.IntSlider(description='Cluster index', min=0, max=200, step=1, value=20,  bar_style='info', disabled=True)
    forward = ipywidgets.Button(description=">>", layout=ipywidgets.Layout(width='40px', color='green'), disabled=True)
    back    = ipywidgets.Button(description="<<", layout=ipywidgets.Layout(width='40px', color='green'), disabled=True)

    def tick(x=None, p=progress, max=10):
        if p.max != max: p.max = max
        p.value = x if x is not None else p.value + 1

    def update_cluster(*args):

        output_cluster.clear_output()
        with output_cluster:

            #display(clusters.df[clusters.df.cluster==n_cluster.value])
            p = cluster_plot.plot_cluster(x_corpus, data.token_clusters, n_cluster.value, tick=tick)
            tick(1, max=2)
            bokeh.plotting.show(p)
            cluster_plot.cluster_boxplot(x_corpus, data.token_clusters, n_cluster.value,)
            tick(0)

    def on_goto(b):

        if b.description == "<<":
            n_cluster.value = max(n_cluster.value - 1, 0)

        if b.description == ">>":
            n_cluster.value = min(n_cluster.value + 1, n_cluster.max)

        update_cluster()

    forward.on_click(on_goto)
    back.on_click(on_goto)
    n_cluster.observe(update_cluster, 'value')
    output_clusters = ipywidgets.Output()
    output_cluster = ipywidgets.Output()

    def compute_on_click(*args):

        output_clusters.clear_output()
        output_cluster.clear_output()

        n_cluster.disabled = True
        n_cluster.max = n_clusters.value

        with output_clusters:
            tick(1, max=10)
            clusters = CLUSTER_METHODS[method.value](x_corpus, n_clusters.value)
            p = cluster_plot.plot_clusters(clusters.token_clusters, tick)
            bokeh.plotting.show(p)
            tick(0)

        n_cluster.disabled = False
        forward.disabled = False
        back.disabled = False

        n_cluster.value = 0

    compute.on_click(compute_on_click)

    widgets = ipywidgets.HBox([
        ipywidgets.HBox([
            ipywidgets.VBox([
                progress,
                ipywidgets.HBox([n_clusters, compute]),
                output_clusters
            ]),
        ]),
         ipywidgets.HBox([
            ipywidgets.VBox([
                ipywidgets.HBox([n_cluster, back, forward]),
                output_cluster
            ]),
        ])
    ])

    display(widgets)
