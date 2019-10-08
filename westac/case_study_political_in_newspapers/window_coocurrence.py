# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py,md
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# %load_ext autoreload
# %autoreload 2
import pandas as pd
import numpy as np
import scipy
import os
import nltk

from sklearn import preprocessing

from westac.common import corpus_vectorizer
from westac.common import utility
from westac.common import text_corpus

def load_text_windows(filename: str):
    """Reads excel file "filename" and returns content as a Pandas DataFrame.
    The file is written to tsv the first time read for faster subsequent reads.

    Parameters
    ----------
    filename : str
        Name of excel file that has two columns: year and txt

    Returns
    -------
    [DataFrame]
        Content of filename as a DataFrame

    Raises
    ------
    FileNotFoundError
    """
    filepath = os.path(filename)

    if not os.path.isdir(filepath):
        raise FileNotFoundError("Path {filepath} does not exist!")

    filebase = os.path.basename(filename).split('.')[0]
    textfile = os.path.join(filepath, filebase + '.txt')

    if not os.path.isfile(textfile):
        df = pd.read_excel(filename)
        df.to_csv(textfile, sep='\t')

    df = pd.read_csv(textfile, sep='\t')[['year', 'txt']]

    return df

def compute_coocurrence_matrix(reader, min_count=1, **kwargs):
    """Computes a term-term coocurrence matrix for documents in reader.

    Parameters
    ----------
    reader : enumerable(list(str))
        Sequence of tokenized documents

    Returns
    -------
    [DataFrane]
        Upper diagonal of term-term frequency matrix (TTM). Note that diagonal (wi, wi) is not returned
    """
    corpus = text_corpus.ProcessedCorpus(reader, isalnum=False, **kwargs)
    vectorizer = corpus_vectorizer.CorpusVectorizer(lowercase=False)
    vectorizer.fit_transform(corpus)

    term_term_matrix = np.dot(vectorizer.X.T, vectorizer.X)
    term_term_matrix = scipy.sparse.triu(term_term_matrix, 1)

    id2token = {
        i: t for t,i in vectorizer.vocabulary.items()
    }

    cdf = pd.DataFrame({
        'w1_id': term_term_matrix.row,
        'w2_id': term_term_matrix.col,
        'value': term_term_matrix.data
    })[['w1_id', 'w2_id', 'value']].sort_values(['w1_id', 'w2_id'])\
        .reset_index(drop=True)

    if min_count > 1:
        cdf = cdf[cdf.value >= min_count]

    n_documents = len(corpus.get_metadata())
    n_tokens = sum(corpus.n_raw_tokens.values())

    cdf['value_n_d'] = cdf.value / float(n_documents)
    cdf['value_n_t'] = cdf.value / float(n_tokens)

    cdf['w1'] = cdf.w1_id.apply(lambda x: id2token[x])
    cdf['w2'] = cdf.w2_id.apply(lambda x: id2token[x])

    return cdf[['w1', 'w2', 'value', 'value_n_d', 'value_n_t']]

def compute_co_ocurrence_for_year(source_filename, years, target_filename, min_count=1, **options):

    columns = ['w1', 'w2', 'value', 'value_n_d', 'value_n_t']

    df   = pd.read_csv(source_filename, sep='\t')[['year', 'txt']]
    df_r = pd.DataFrame(columns=columns)

    for year in years:
        reader = utility.DfTextReader(df, year)
        df_y = compute_coocurrence_matrix(reader, min_count=min_count, **options)
        df_y['year'] = year
        df_r = df_r.append(df_y[columns], ignore_index=True)

    # min_max_scaler = preprocessing.MinMaxScaler()
    # # Scale a normalized data matrix to the [0, 1] range:
    # df_r['value_n_t'] = min_max_scaler.fit_transform(df_r.value_n_t)
    # df_r['value_n_d'] = min_max_scaler.fit_transform(df_r.value_n_d)

    # Scale a normalized data matrix to the [0, 1] range:
    df_r['value_n_t'] = df_r.value_n_t / df_r.value_n_t.max()
    df_r['value_n_d'] = df_r.value_n_d / df_r.value_n_d.max()

    df_r.to_csv(target_filename, sep='\t', index=False, )

if __name__ == "__main__":

    stopwords = set(nltk.corpus.stopwords.words('swedish')) + { "politisk", "politiska", "politiskt" }
    options   = dict(to_lower=True, deacc=False, min_len=2, max_len=None, numerals=False, filter_stopwords=False, stopwords=stopwords)

    compute_co_ocurrence_for_year('./data/year+text_window.txt', [1957], 'test_1957.xlsx', min_count=1, options=options)
