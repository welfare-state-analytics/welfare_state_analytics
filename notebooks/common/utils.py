import pandas as pd


def setup_pandas():

    pd.set_option("max_rows", None)
    pd.set_option("max_columns", None)
    pd.set_option('colheader_justify', 'left')
    pd.set_option('max_colwidth', 300)


def flatten(lst):
    return [x for ws in lst for x in ws]


def to_text(document, id2token):
    return ' '.join(flatten([f * [id2token[token_id]] for token_id, f in document]))
