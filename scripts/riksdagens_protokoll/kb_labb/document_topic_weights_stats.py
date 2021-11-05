# %%
import pandas as pd

# pylint: disable=unsubscriptable-object, no-member
df: pd.DataFrame = pd.read_csv('../data/dtw.csv', index_col=0, sep='\t')

# %%
df_n_docs_per_year: pd.DataFrame = pd.DataFrame(df.groupby('year').document_id.nunique())
df_n_docs_per_year.columns = ['n_total_count']
# %%
df_topic_69: pd.DataFrame = df[df.topic_id == 68]
df_topic_69_by_year: pd.DataFrame = df_topic_69.groupby(['year', 'topic_id']).agg(
    {'document_id': 'size', 'weight': 'sum'}
)
df_topic_69_by_year = df_topic_69_by_year.merge(df_n_docs_per_year, left_index=True, right_index=True, how='inner')
df_topic_69_by_year = df_topic_69_by_year.reset_index()
df_topic_69_by_year.columns = ['year', 'topic_id', 'n_topic_docs', 'sum_weight', 'n_total_docs']
df_topic_69_by_year = df_topic_69_by_year[['year', 'topic_id', 'sum_weight', 'n_topic_docs', 'n_total_docs']]
df_topic_69_by_year.to_csv('df_topic_69_by_year.csv', sep='\t')
