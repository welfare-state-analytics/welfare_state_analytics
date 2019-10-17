import os
import westac.common.corpus_vectorizer as corpus_vectorizer
import westac.common.text_corpus as text_corpus
import westac.common.utility as utility
import numpy as np
import sklearn
from scipy import stats

#filename = './test/test_data/test_corpus.zip'
#filename = './data/Sample_1945-1989_1.zip'

filename = './westac/case_study_word_distribution_trends/data/SOU_1945-1989.zip'

if not os.path.isfile(filename):
    print('error: no such file: {}'.format(filename))
    assert os.path.isfile(filename)

dump_name = os.path.basename(filename).split('.')[0]

vectorizer = corpus_vectorizer.CorpusVectorizer()

print('Creating corpus...')
corpus = text_corpus.create_corpus(filename)

X = vectorizer.fit_transform(corpus)

vectorizer.dump(dump_name, folder='./output')

#vectorizer.load(dump_name, folder='./output')

Y         = vectorizer.collapse_to_year()
Yn        = vectorizer.normalize(Y, axis=1, norm='l1')
Ynw       = vectorizer.slice_tokens_by_count_threshold(Yn, 1)
Yx2, imap = vectorizer.pick_by_top_variance(500)

data       = stats.chisquare(Ynw, f_exp=None, ddof=0, axis=0)

