import os
from westac.common import corpus_vectorizer
from westac.common import text_corpus
from westac.common import vectorized_corpus
from westac.common import file_text_reader

def create_corpus(filename, meta_extract):
    reader = file_text_reader.FileTextReader(filename, meta_extract=meta_extract, compress_whitespaces=True, dehyphen=True)
    kwargs = dict(
        isalnum=False,
        to_lower=True,
        deacc=False,
        min_len=2,
        max_len=None,
        numerals=False,
        symbols=False
    )
    corpus = text_corpus.ProcessedCorpus(reader, **kwargs)
    return corpus

filename = './data/SOU_1945-1989.zip'

if not os.path.isfile(filename):
    print('error: no such file: {}'.format(filename))
    assert os.path.isfile(filename)

dump_tag = os.path.basename(filename).split('.')[0]

if not vectorized_corpus.VectorizedCorpus.dump_exists(dump_tag):

    meta_extract = {
        'year': r"SOU (\d{4})\_.*",
        'serial_no': r"SOU \d{4}\_(\d+).*"
    }

    print('Creating new corpus...')
    corpus = text_corpus.create_corpus(filename, meta_extract)

    print('Creating document-term matrix...')
    vectorizer = corpus_vectorizer.CorpusVectorizer()
    v_corpus = vectorizer.fit_transform(corpus)

    print('Saving data matrix...')
    v_corpus.dump(tag=dump_tag, folder='./output')

else:

    print('Loading data matrix...')

    v_corpus = vectorized_corpus.VectorizedCorpus.load(dump_tag, folder='./output')

#YTM       = vectorizer.group_by_year()
#Yn        = vectorizer.normalize(Y, axis=1, norm='l1')
#Ynw       = vectorizer.slice_tokens_by_count_threshold(Yn, 1)
#Yx2, imap = vectorizer.pick_by_top_variance(500)

#data       = stats.chisquare(Ynw, f_exp=None, ddof=0, axis=0)

