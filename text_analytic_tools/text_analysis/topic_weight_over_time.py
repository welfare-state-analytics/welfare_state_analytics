import pandas as pd
import numpy as np
import itertools

METHODS = [
    {
        'key': 'max_weight',
        'description': 'Max value',
        'tooltip': 'Use maximum value over documents'
    }, {
        'key': 'false_mean',
        'description': 'Mean where topic is relevant',
        'tooltip': 'Use mean value of all documents where topic is above certain treshold'
    }, {
        'key': 'true_mean',
        'description': 'Mean of all documents',
        'tooltip': 'Use mean value of all documents even those where topic is zero'
    }
]

def plot_topic(df, x):
    df = df.reset_index()
    df[df.topic_id==x].set_index('year').drop('topic_id', axis=1).plot()

def compute_weight_over_time(df):

    """ Initialize year/topic cross product data frame """
    cross_iter = itertools.product(range(df.year.min(), df.year.max() + 1), range(0, df.topic_id.max() + 1))
    dfs = pd.DataFrame(list(cross_iter), columns=['year', 'topic_id']).set_index(['year', 'topic_id'])

    """ Add the most basic stats """
    dfs = dfs.join(df.groupby(['year', 'topic_id'])['weight'].agg([np.max, np.sum, np.mean, len]), how='left').fillna(0)
    dfs.columns = ['max_weight', 'sum_weight', 'false_mean', 'n_topic_docs']
    dfs['n_topic_docs'] = dfs.n_topic_docs.astype(np.uint32)

    doc_counts = df.groupby('year').document_id.nunique().rename('n_total_docs')
    dfs = dfs.join(doc_counts, how='left').fillna(0)
    dfs['n_total_docs'] = dfs.n_total_docs.astype(np.uint32)
    dfs['true_mean'] = dfs.apply(lambda x: x['sum_weight'] / x['n_total_docs'], axis=1)

    return dfs.reset_index()

def normalize_weights(df):

    dfy = df.groupby(['year'])['weight'].sum().rename('sum_weight')
    df = df.merge(dfy, how='inner', left_on=['year'], right_index=True)
    df['weight'] = df.apply(lambda x: x['weight'] / x['sum_weight'], axis=1)
    df = df.drop(['sum_weight'], axis=1)
    return df

# def year_topic_weight_statistics(years, topics):

#     data = itertools.product(range(df.year.min(), df.year.max() + 1), range(0, df.topic_id.max() + 1))

#     xf = pd.DataFrame(list(data), columns=['year', 'topic_id']).set_index(['year', 'topic_id'])

#     return xf

# df = current_state().compiled_data.document_topic_weights

# xf = aggregate_weights(df)
# #plot_topic(xf[['true_mean']], 0)
# print(xf.head())

def get_weight_over_time(current_weight_over_time, document_topic_weights, publication_id):
    if current_weight_over_time.publication_id != publication_id:
        current_weight_over_time.publication_id = publication_id
        df = document_topic_weights
        if publication_id is not None:
            df = df[df.publication_id == publication_id]
        current_weight_over_time.weights = compute_weight_over_time(df).fillna(0)
    return current_weight_over_time.weights

