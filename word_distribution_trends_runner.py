import os
import westac.common.corpus_vectorizer as corpus_vectorizer
import westac.common.text_corpus as text_corpus

from scipy import stats

filename = './data/SOU_1945-1989.zip'

if not os.path.isfile(filename):
    print('error: no such file: {}'.format(filename))
    assert os.path.isfile(filename)

dump_tag = os.path.basename(filename).split('.')[0]

vectorizer = corpus_vectorizer.CorpusVectorizer()

if not vectorizer.dump_exists(dump_tag):

    meta_extract = {
        'year': r"SOU (\d{4})\_.*",
        'serial_no': r"SOU \d{4}\_(\d+).*"
    }

    print('Creating new corpus...')
    corpus = text_corpus.create_corpus(filename, meta_extract)

    print('Creating document-term matrix...')
    DTM = vectorizer.fit_transform(corpus)

    print('Saving data matrix...')
    vectorizer.dump(tag=dump_tag, folder='./output')

else:

    print('Loading data matrix...')

    vectorizer.load(dump_tag, folder='./output')

#YTM       = vectorizer.group_by_year()
#Yn        = vectorizer.normalize(Y, axis=1, norm='l1')
#Ynw       = vectorizer.slice_tokens_by_count_threshold(Yn, 1)
#Yx2, imap = vectorizer.pick_by_top_variance(500)

#data       = stats.chisquare(Ynw, f_exp=None, ddof=0, axis=0)

