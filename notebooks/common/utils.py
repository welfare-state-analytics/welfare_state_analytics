import pandas as pd

def setup_pandas():

    pd.set_option("max_rows", None)
    pd.set_option("max_columns", None)
    pd.set_option('colheader_justify', 'left')
    pd.set_option('max_colwidth', 300)
