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
import unittest
import corpus_vectorizer

class Test_DfTextReader(unittest.TestCase):
    
    def create_test_dataframe(self):
        data = [ 
            (2000, 'A B C'),
            (2000, 'B C D'), 
            (2001, 'C B'),
            (2003, 'A B F'),
            (2003, 'E B'),
            (2003, 'F E E')
        ]
        df = pd.DataFrame(data, columns=['year', 'txt'])
        return df
    
    def test_reader_with_all_documents(self):
        df = self.create_test_dataframe()
        reader = DfTextReader(df)
        result = [x for x in reader]
        expected = [('0', 'A B C'), ('1', 'B C D'), ('2', 'C B'), ('3', 'A B F'), ('4', 'E B'), ('5', 'F E E')]
        self.assertEqual(expected, result)
        self.assertEqual(['0', '1', '2', '3', '4', '5'], reader.filenames)
        self.assertEqual([
                types.SimpleNamespace(filename='0', year=2000),
                types.SimpleNamespace(filename='1', year=2000),
                types.SimpleNamespace(filename='2', year=2001),
                types.SimpleNamespace(filename='3', year=2003),
                types.SimpleNamespace(filename='4', year=2003),
                types.SimpleNamespace(filename='5', year=2003)
            ], reader.metadata
        )
        
    def test_reader_with_given_year(self):
        df = self.create_test_dataframe()
        reader = DfTextReader(df, 2003)
        result = [x for x in reader]
        expected = [('0', 'A B F'), ('1', 'E B'), ('2', 'F E E')]
        self.assertEqual(expected, result)
        self.assertEqual(['0', '1', '2'], reader.filenames)
        self.assertEqual([
                types.SimpleNamespace(filename='0', year=2003),
                types.SimpleNamespace(filename='1', year=2003),
                types.SimpleNamespace(filename='2', year=2003)
            ], reader.metadata
        )

class Test_DfVectorize(unittest.TestCase):
    
    def setUp(self):
        pass
    
    def create_test_dataframe(self):
        data = [ 
            (2000, 'A B C'),
            (2000, 'B C D'), 
            (2001, 'C B'),
            (2003, 'A B F'),
            (2003, 'E B'),
            (2003, 'F E E')
        ]
        df = pd.DataFrame(data, columns=['year', 'txt'])
        return df
    
    def create_corpus(self):
        df = self.create_test_dataframe()
        reader = DfTextReader(df)
        kwargs = dict(isalnum=False, to_lower=False, deacc=False, min_len=0, max_len=None, numerals=False)
        corpus = text_corpus.ProcessedCorpus(reader, **kwargs)
        return corpus
    
    def test_corpus_text_stream(self):
        df = self.create_test_dataframe()
        reader = DfTextReader(df)
        corpus = text_corpus.CorpusTextStream(reader)
        result = [ x for x in corpus.documents()]
        expected = [('0', 'A B C'), ('1', 'B C D'), ('2', 'C B'), ('3', 'A B F'), ('4', 'E B'), ('5', 'F E E')]
        self.assertEqual(expected, result)
        
    def test_corpus_token_stream(self):
        df = self.create_test_dataframe()
        reader = DfTextReader(df)
        corpus = text_corpus.CorpusTokenStream(reader)
        result = [ x for x in corpus.documents()]
        expected = [('0', ['A', 'B', 'C']), ('1', ['B', 'C', 'D']), ('2', ['C', 'B']), ('3', ['A', 'B', 'F']), ('4', ['E', 'B']), ('5', ['F', 'E', 'E'])]
        self.assertEqual(expected, result)

    def test_processed_corpus_token_stream(self):
        df = self.create_test_dataframe()
        reader = DfTextReader(df)
        kwargs = dict(isalnum=False, to_lower=False, deacc=False, min_len=0, max_len=None, numerals=False)
        corpus = text_corpus.ProcessedCorpus(reader, **kwargs)
        result = [ x for x in corpus.documents()]
        expected = [('0', ['A', 'B', 'C']), ('1', ['B', 'C', 'D']), ('2', ['C', 'B']), ('3', ['A', 'B', 'F']), ('4', ['E', 'B']), ('5', ['F', 'E', 'E'])]
        self.assertEqual(expected, result)
        
    def test_fit_transform_gives_document_term_matrix(self):
        reader = DfTextReader(self.create_test_dataframe())
        kwargs = dict(to_lower=False, deacc=False, min_len=1, max_len=None, numerals=False)
        corpus = text_corpus.ProcessedCorpus(reader, isalnum=False, **kwargs)
        vectorizer = corpus_vectorizer.CorpusVectorizer(lowercase=False)
        vectorizer.fit_transform(corpus)
        expected = np.asarray([
            [1, 1, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 1],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 0, 0, 2, 1]
        ])
        self.assertTrue((expected == vectorizer.X).all())
        results = vectorizer.vocabulary
        expected = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'F': 5, 'E': 4 }
        self.assertEqual(expected, results)

    def test_AxAt_of_document_term_matrix_gives_term_term_matrix(self):
        
        # Arrange
        reader = DfTextReader(self.create_test_dataframe())
        kwargs = dict(to_lower=False, deacc=False, min_len=1, max_len=None, numerals=False)
        corpus = text_corpus.ProcessedCorpus(reader, isalnum=False, **kwargs)
        vectorizer = corpus_vectorizer.CorpusVectorizer(lowercase=False)
        vectorizer.fit_transform(corpus)
        
        # Act
        term_term_matrix = np.dot(vectorizer.X.T, vectorizer.X)
        
        # Assert
        expected = np.asarray([
             [2, 2, 1, 0, 0, 1],
             [2, 5, 3, 1, 1, 1],
             [1, 3, 3, 1, 0, 0],
             [0, 1, 1, 1, 0, 0],
             [0, 1, 0, 0, 5, 2],
             [1, 1, 0, 0, 2, 2]
        ])
        self.assertTrue((expected == term_term_matrix).all())
        
        term_term_matrix = scipy.sparse.triu(term_term_matrix, 1)
        expected = np.asarray([
             [0, 2, 1, 0, 0, 1],
             [0, 0, 3, 1, 1, 1],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 2],
             [0, 0, 0, 0, 0, 0]
        ])
        
        #print(term_term_matrix.todense())
        #print(term_term_matrix)
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
        print(cdf[['w1', 'w2', 'value']])
        
unittest.main(argv=['first-arg-is-ignored'], exit=False)
        
```
