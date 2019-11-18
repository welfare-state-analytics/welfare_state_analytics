import types
import sklearn
import pandas as pd

def compute(x_corpus, n_clusters=8, indices=None):

    m = (x_corpus.data if indices is None else x_corpus.data[:, indices])

    clusters = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=0, n_jobs=2).fit(m.T)

    token2cluster = {
        x_corpus.id2token[i]: x for i, x in enumerate(clusters.labels_)
    }
    df_clusters = pd.DataFrame({
        'token': list(token2cluster.keys()),
        'cluster': list(token2cluster.values())
    })
    return types.SimpleNamespace(
        result=clusters,
        token2cluster=token2cluster,
        token_clusters=df_clusters
    )
