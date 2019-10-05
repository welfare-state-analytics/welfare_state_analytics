---
jupyter:
  jupytext:
    formats: ipynb,py,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.1'
      jupytext_version: 1.2.4
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

```python
%load_ext autoreload
%autoreload 2
import pandas as pd
import numpy as np
import scipy
import os, sys

sys.path = list(set(sys.path + [ '../common' ]))

import utility
import text_corpus
import corpus_vectorizer
import types

# df = pd.read_excel('./data/year+text_window.xlsx')
# df.to_csv('./data/year+text_window.txt', sep='\t')
```

```python
class DfTextReader:

    def __init__(self, df, year=None):
        
        self.df = df
        
        if year is not None:
            self.df = self.df[self.df.year == year]
                              
        if len(self.df[self.df.txt.isna()]) > 0:
            print('Warn: {} n/a rows encountered'.format(len(self.df[self.df.txt.isna()])))
            self.df = self.df.dropna()
            
        self.iterator = None
        self.metadata = [ types.SimpleNamespace(filename=str(i), year=r) for i, r in enumerate(self.df.year.values)]
        self.metadict = { x.filename: x for x in (self.metadata or [])}
        self.filenames = [ x.filename for x in self.metadata ]
        
    def __iter__(self):
                              
        self.iterator = None
        return self

    def __next__(self):
                              
        if self.iterator is None:
            self.iterator = self.get_iterator()
                              
        return next(self.iterator)

    def get_iterator(self):
        return ((str(i), x) for i,x in enumerate(self.df.txt))
```

```python
import numpy as np

def compute_coocurrence_matrix(reader, **kwargs):

    corpus = text_corpus.ProcessedCorpus(reader, isalnum=False, **kwargs)
    vectorizer = corpus_vectorizer.CorpusVectorizer(lowercase=False)
    vectorizer.fit_transform(corpus)
        
    term_term_matrix = np.dot(vectorizer.X.T, vectorizer.X)
        
    term_term_matrix = scipy.sparse.triu(term_term_matrix, 1)
        
    coo = term_term_matrix
    id2token = { i: t for t,i in vectorizer.vocabulary.items()}
    cdf = pd.DataFrame({
        'w1_id': coo.row,
        'w2_id': coo.col,
        'value': coo.data
    })[['w1_id', 'w2_id', 'value']].sort_values(['w1_id', 'w2_id'])\
        .reset_index(drop=True)
    cdf['w1'] = cdf.w1_id.apply(lambda x: id2token[x])
    cdf['w2'] = cdf.w2_id.apply(lambda x: id2token[x])
    
    return cdf[['w1', 'w2', 'value']]

def compute_co_ocurrence_for_year(source_filename, year, result_filename):
    
    df = pd.read_csv(source_filename, sep='\t')[['year', 'txt']]

    reader = DfTextReader(df, year)

    kwargs = dict(to_lower=True, deacc=False, min_len=1, max_len=None, numerals=False)

    coo_df = compute_coocurrence_matrix(reader, **kwargs)
    coo_df.to_excel(result_filename)

compute_co_ocurrence_for_year('./data/year+text_window.txt', 1957, 'test_1957.xlsx')

```
