
import bokeh.models as bm
import bokeh.palettes
import networkx as nx

from bokeh.plotting import figure
from . utility import *

if 'extend' not in globals():
    extend = lambda a,b: a.update(b) or a

DFLT_NODE_OPTS = dict(color='green', level='overlay', alpha=1.0)
DFLT_EDGE_OPTS = dict(color='black', alpha=0.2)

layout_algorithms = {
    'Fruchterman-Reingold': lambda x,**args: nx.spring_layout(x,**args),
    'Eigenvectors of Laplacian':  lambda x,**args: nx.spectral_layout(x,**args),
    'Circular': lambda x,**args: nx.circular_layout(x,**args),
    'Shell': lambda x,**args: nx.shell_layout(x,**args),
    'Kamada-Kawai': lambda x,**args: nx.kamada_kawai_layout(x,**args)
}

class PlotNetworkUtility:

    @staticmethod
    def layout_args(layout_algorithm, network, scale):
        if layout_algorithm == 'Shell':
            if nx.is_bipartite(network):
                nodes, other_nodes = get_bipartite_node_set(network, bipartite=0)
                args = dict(nlist=[nodes, other_nodes])

        if layout_algorithm == 'Fruchterman-Reingold':
            k = scale #/ math.sqrt(network.number_of_nodes())
            args = dict(dim=2, k=k, iterations=20, weight='weight', scale=0.5)

        if layout_algorithm == 'Kamada-Kawai':
            args = dict(dim=2, weight='weight', scale=1.0)

        return dict()

    @staticmethod
    def get_layout_algorithm(layout_algorithm):
        if layout_algorithm not in layout_algorithms:
            raise Exception("Unknown algorithm {}".format(layout_algorithm))
        return layout_algorithms.get(layout_algorithm, None)

    @staticmethod
    def project_series_to_range(series, low, high):
        norm_series = series / series.max()
        return norm_series.apply(lambda x: low + (high - low) * x)

    @staticmethod
    def plot_network(
        network,
        layout_algorithm=None,
        scale=1.0,
        threshold=0.0,
        node_description=None,
        node_proportions=None,
        weight_scale=5.0,
        normalize_weights=True,
        node_opts=None,
        line_opts=None,
        element_id='nx_id3',
        figsize=(900,900)
    ):
        if threshold > 0:
            values = nx.get_edge_attributes(network, 'weight').values()
            max_weight = max(1.0, max(values))

            print('Max weigth: {}'.format(max_weight))
            print('Mean weigth: {}'.format(sum(values) / len(values)))

            filter_edges = [(u, v) for u, v, d in network.edges(data=True) \
                            if d['weight'] >= (threshold * max_weight)]

            sub_network = network.edge_subgraph(filter_edges)
        else:
            sub_network = network

        args = PlotNetworkUtility.layout_args(layout_algorithm, sub_network, scale)
        layout = (PlotNetworkUtility.get_layout_algorithm(layout_algorithm))(sub_network, **args)
        lines_source = NetworkUtility.get_edges_source(
            sub_network, layout, scale=weight_scale, normalize=normalize_weights
        )
        nodes_source = NetworkUtility.create_nodes_data_source(sub_network, layout)

        nodes_community = NetworkMetricHelper.compute_partition(sub_network)
        community_colors = NetworkMetricHelper.partition_colors(nodes_community, bokeh.palettes.Category20[20])

        nodes_source.add(nodes_community, 'community')
        nodes_source.add(community_colors, 'community_color')

        nodes_size = 5
        if node_proportions is not None:
            # NOTE!!! By pd index - not iloc!!
            nodes_weight = node_proportions.loc[list(sub_network.nodes)]
            nodes_weight = PlotNetworkUtility.project_series_to_range(nodes_weight, 20, 60)
            nodes_size = 'size'
            nodes_source.add(nodes_weight, nodes_size)

        node_opts = extend(DFLT_NODE_OPTS, node_opts or {})
        line_opts = extend(DFLT_EDGE_OPTS, line_opts or {})

        p = figure(plot_width=figsize[0], plot_height=figsize[1], x_axis_type=None, y_axis_type=None)
        #node_size = 'size' if node_proportions is not None else 5
        r_lines = p.multi_line('xs', 'ys', line_width='weights', source=lines_source, **line_opts)
        r_nodes = p.circle('x', 'y', size=nodes_size, source=nodes_source, **node_opts)

        text_opts = dict(x='x', y='y', text='name', level='overlay', text_align='center', text_baseline='middle')

        r_nodes.glyph.fill_color = 'lightgreen' # 'community_color'

        p.add_layout(bm.LabelSet(source=nodes_source, text_color='black', **text_opts))

        return p
