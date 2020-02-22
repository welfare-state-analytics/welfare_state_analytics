import math
import community # pip3 install python-louvain packages
from networkx.algorithms import bipartite
import networkx as nx
import bokeh.models as bm
import bokeh.palettes

from itertools import product

if 'extend' not in globals():
    extend = lambda a,b: a.update(b) or a

if 'filter_kwargs' not in globals():
    import inspect
    filter_kwargs = lambda f, args: { k:args[k] for k in args.keys() if k in inspect.getargspec(f).args }

DISTANCE_METRICS = {
    # 'Bray-Curtis': 'braycurtis',
    # 'Canberra': 'canberra',
    # 'Chebyshev': 'chebyshev',
    # 'Manhattan': 'cityblock',
    'Correlation': 'correlation',
    'Cosine': 'cosine',
    'Euclidean': 'euclidean',
    # 'Mahalanobis': 'mahalanobis',
    # 'Minkowski': 'minkowski',
    'Normalized Euclidean': 'seuclidean',
    'Squared Euclidean': 'sqeuclidean',
    'Kullback-Leibler': 'kullback–leibler',
    'Kullback-Leibler (SciPy)': 'scipy.stats.entropy'
}

class NetworkMetricHelper:

    @staticmethod
    def compute_centrality(network):
        centrality = nx.algorithms.centrality.betweenness_centrality(network)
        _, nodes_centrality = zip(*sorted(centrality.items()))
        max_centrality = max(nodes_centrality)
        centrality_vector = [7 + 10 * t / max_centrality for t in nodes_centrality]
        return centrality_vector

    @staticmethod
    def compute_partition(network):
        partition = community.best_partition(network)
        _, nodes_community = zip(*sorted(partition.items()))
        return nodes_community

    @staticmethod
    def partition_colors(nodes_community, color_palette=None):
        if color_palette is None:
            color_palette = ['#e41a1c','#377eb8','#4daf4a','#984ea3','#ff7f00',
                             '#ffff33','#a65628', '#b3cde3','#ccebc5','#decbe4','#fed9a6',
                             '#ffffcc','#e5d8bd','#fddaec','#1b9e77','#d95f02','#7570b3','#e7298a',
                             '#66a61e','#e6ab02','#a6761d','#666666']
        community_colors = [ color_palette[x % len(color_palette)] for x in nodes_community ]
        return community_colors

    @staticmethod
    def compute_alpha_vector(value_vector):
        max_value = max(value_vector)
        alphas = list(map(lambda h: 0.1 + 0.6 * (h / max_value), value_vector))
        return alphas

class NetworkUtility:

    @staticmethod
    def get_edge_layout_data(network, layout):

        data = [ (u, v, d['weight'], [layout[u][0], layout[v][0]], [layout[u][1], layout[v][1]])
                    for u, v, d in network.edges(data=True) ]

        return zip(*data)

    #FIXME Merge these two methods, return dict instead (lose bokeh dependency)
    @staticmethod
    def get_edges_source(network, layout, scale=1.0, normalize=False):

        _, _, weights, xs, ys = NetworkUtility.get_edge_layout_data(network, layout)
        norm = max(weights) if normalize else 1.0
        weights = [ scale * x / norm for x in  weights ]
        lines_source = bm.ColumnDataSource(dict(xs=xs, ys=ys, weights=weights))
        return lines_source

    @staticmethod
    def get_node_subset_source(network, layout, node_list = None):

        layout_items = layout.items() if node_list is None else [ x for x in layout.items() if x[0] in node_list ]

        nodes, nodes_coordinates = zip(*sorted(layout_items))
        xs, ys = list(zip(*nodes_coordinates))

        nodes_source = bm.ColumnDataSource(dict(x=xs, y=ys, name=nodes, node_id=nodes))
        return nodes_source

    @staticmethod
    def create_nodes_data_source(network, layout):

        nodes, nodes_coordinates = zip(*sorted([ x for x in layout.items() ])) # if x[0] in line_nodes]))
        nodes_xs, nodes_ys = list(zip(*nodes_coordinates))
        nodes_source = bm.ColumnDataSource(dict(x=nodes_xs, y=nodes_ys, name=nodes, node_id=nodes))
        return nodes_source

    @staticmethod
    def create_network(df, source_field='source', target_field='target', weight='weight'):

        G = nx.Graph()
        nodes = list(set(list(df[source_field].values) + list(df[target_field].values)))
        edges = [ (x, y, { weight: z })
                 for x, y, z in [ tuple(x) for x in df[[source_field, target_field, weight]].values]]
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)
        return G

    @staticmethod
    def create_bipartite_network(df, source_field='source', target_field='target', weight='weight'):

        G = nx.Graph()
        G.add_nodes_from(set(df[source_field].values), bipartite=0)
        G.add_nodes_from(set(df[target_field].values), bipartite=1)
        edges = list(zip(df[source_field].values,df[target_field].values,df[weight]\
                         .apply(lambda x: dict(weight=x))))
        G.add_edges_from(edges)
        return G

    @staticmethod
    def get_bipartite_node_set(network, bipartite=0):
        nodes = set(n for n,d in network.nodes(data=True) if d['bipartite']==bipartite)
        others = set(network) - nodes
        return list(nodes), list(others)

    @staticmethod
    def create_network_from_xyw_list(values, threshold=0.0):
        G = nx.Graph()
        G.add_weighted_edges_from(values)
        return G

    #@staticmethod
    #def create_network_from_correlation_matrix(matrix, threshold=0.0):

    #    G = nx.Graph()
    #    #G.add_nodes_from(range(0, max(x_dim,y_dim)))
    #    values = VectorSpaceHelper.symmetric_lower_left_iterator(matrix, threshold)
    #    G.add_weighted_edges_from(values)
    #    return G

    #@staticmethod
    #def matrix_weight_iterator(matrix, threshold=0.0):
    #    '''
    #    Iterates sparse matrix and reverses distance metric in range 0 to -1 i.e.
    #        weigh = 1.0 - distance
    #    A high distance value should be a low weight in the graph.
    #    The matrix i assumed to be symmetric, and only one edge is returned per node pair
    #    '''
    #    x_dim, y_dim = matrix.shape
    #    return ((i, j, 1.0 - matrix[i,j])
    #            for i, j in product(range(0,x_dim), range(0,y_dim))
    #                if i < j and (1.0 - matrix[i,j]) >= threshold)

    #@staticmethod
    #def df_stack_correlation_matrix(cm, threshold=0.0, n_top=100):
    #    items = NetworkUtility.matrix_weight_iterator(cm, threshold)
    #    return sorted(items, key=lambda x: x[2])[:n_top]

# pos = nx.graphviz_layout(G, prog="twopi") # twopi, neato, circo

get_edge_layout_data = NetworkUtility.get_edge_layout_data
get_edges_source = NetworkUtility.get_edges_source
get_node_subset_source = NetworkUtility.get_node_subset_source
create_nodes_data_source = NetworkUtility.create_nodes_data_source
create_network = NetworkUtility.create_network
create_bipartite_network = NetworkUtility.create_bipartite_network
get_bipartite_node_set = NetworkUtility.get_bipartite_node_set
create_network_from_xyw_list = NetworkUtility.create_network_from_xyw_list

