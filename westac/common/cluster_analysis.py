
import sklearn
import numpy as np

def compute_k_means(x_corpus, n_clusters=8, indices=None):

    m = (x_corpus.data if indices is None else x_corpus.data[:, indices])

    k_means = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=0, n_jobs=2).fit(m.T)

    return {
        x_corpus.id2token[i]: label for i, label in enumerate(k_means.labels_)
    }
