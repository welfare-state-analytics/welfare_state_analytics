import pandas as pd

from ipyaggrid import Grid
from IPython.display import display

import notebooks.word_trends.displayers.data_compilers as data_compilers

NAME = "Table"

compile = data_compilers.compile_year_token_vector_data

def setup(container, **kwargs):
    pass

def plot(data, **kwargs):

    df = pd.DataFrame(data=data)
    df = df[['year']+[x for x in df.columns if x!= 'year']].set_index('year')
    display(df)