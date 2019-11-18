
import itertools
import sklearn
import numpy as np
import pandas as pd
import types

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

LINKAGE_METHODS = [ 'single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward' ]

LINKAGE_METRICS = {
    'braycurtis':     'Bray-Curtis distance.',
    'canberra':       'Canberra distance.',
    'chebyshev':      'Chebyshev distance.',
    'cityblock':      'Manhattan distance.',
    'correlation':    'Correlation distance.',
    'cosine':         'Cosine distance.',
    'euclidean':      'Euclidean distance.',
    'jensenshannon':  'Jensen-Shannon distance.',
    'mahalanobis':    'Mahalanobis distance.',
    'minkowski':      'Minkowski distance.',
    'seuclidean':     'Normalized Euclidean distance.',
    'sqeuclidean':    'Squared Euclidean distance.'
}


def compute(data, labels, linkage_method='ward', linkage_metric='euclidean'):

    linkage_matrix = linkage(data.T, method=linkage_method, metric=linkage_metric)

    """ from documentation

        A (n-1) by 4 matrix Z is returned. At the i-th iteration, clusters with indices Z[i, 0] and Z[i, 1] are combined to form cluster n + i.
        A cluster with an index less than n corresponds to one of the original observations.
        The distance between clusters Z[i, 0] and Z[i, 1] is given by Z[i, 2].
        The fourth value Z[i, 3] represents the number of original observations in the newly formed cluster.

    """

    N = len(labels)

    df = pd.DataFrame(data=linkage_matrix, columns=['a_id', 'b_id', 'distance', 'n_obs'])\
            .astype({'a_id': np.int64, 'b_id': np.int64, 'n_obs': np.int64 })

    df['a_cluster'] = df.a_id.apply(lambda i: labels[i] if i < N else '#{}#'.format(i))
    df['b_cluster'] = df.b_id.apply(lambda i: labels[i] if i < N else '#{}#'.format(i))
    df['cluster'] = [ '#{}#'.format(N+i) for i in df.index ]

    df = df[[ 'a_cluster', 'b_cluster', 'distance', 'cluster']] #, 'a_id', 'b_id', 'n_obs']]

    return df


def plot_dendogram(linkage_matrix, labels):

    plt.figure(figsize=(16, 40))

    dendrogram(
        linkage_matrix,
        truncate_mode="level",
        color_threshold = 1.8,
        show_leaf_counts = True,
        no_labels = False,
        orientation="right",
        labels = labels,
        leaf_rotation = 0,  # rotates the x axis labels
        leaf_font_size = 12,  # font size for the x axis labels
    )
    plt.show()

def clusters_at_threshold(df_clusters, labels, threshold):
    cluster2tokens = { x: set([x]) for x in labels }
    for i, r in df_clusters.iterrows():
        if r['distance'] > threshold:
            break
        cluster2tokens[r['cluster']] = set(cluster2tokens[r['a_cluster']]) | set(cluster2tokens[r['b_cluster']])
        del cluster2tokens[r['a_cluster']]
        del cluster2tokens[r['b_cluster']]

    token_clusters = clusters_to_df(cluster2tokens)
    return types.SimpleNamespace(
        result=cluster2tokens,
        token_clusters=token_clusters
    )

def clusters_to_df(clusters):
    cluster_lists = [ [ (x, y, i) for x, y in itertools.product([k], clusters[k])] for i, k in enumerate(clusters) ]
    df = pd.DataFrame(data=[ x for ws in cluster_lists for x in ws], columns=["cluster_name", "token", "cluster"])
    return df
