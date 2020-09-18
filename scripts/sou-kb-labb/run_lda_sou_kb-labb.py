import os, sys
import click

root_folder = os.path.join(os.getcwd().split('welfare_state_analytics')[0], 'welfare_state_analytics')

sys.path = list(set(sys.path + [ root_folder ]))

import notebooks.political_in_newspapers.corpus_data as corpus_data
import text_analytic_tools.text_analysis.topic_model as topic_model
import westac.corpus.vectorized_corpus as vectorized_corpus
import westac.corpus.corpus_vectorizer as corpus_vectorizer
import westac.corpus.iterators.sparv_xml_corpus_source_reader as sparv_reader

import types
import pickle
import logging

CORPUS_FOLDER = os.path.join(root_folder, "data")

ENGINE_OPTIONS = [
    ('MALLET LDA', 'gensim_mallet-lda'),
    ('gensim LDA', 'gensim_lda'),
    ('gensim LSI', 'gensim_lsi'),
    ('gensim HDP', 'gensim_hdp'),
    ('gensim DTM', 'gensim_dtm'),
    ('scikit LDA', 'sklearn_lda'),
    ('scikit NMF', 'sklearn_nmf'),
    ('scikit LSA', 'sklearn_lsa'),
    ('STTM   LDA', 'gensim_sttm-lda'),
    ('STTM   BTM', 'gensim_sttm-btm'),
    ('STTM   PTM', 'gensim_sttm-ptm'),
    ('STTM  SATM', 'gensim_sttm-satm'),
    ('STTM   DMM', 'gensim_sttm-dmm'),
    ('STTM  WATM', 'gensim_sttm-watm'),
]

def store_model(data, filename):

    data = types.SimpleNamespace(
        topic_model=data.topic_model,
        id2term=data.id2term,
        bow_corpus=data.bow_corpus,
        doc_term_matrix=None, #doc_term_matrix,
        doc_topic_matrix=None, #doc_topic_matrix,
        vectorizer=None, #vectorizer,
        processed=data.processed,
        coherence_scores=data.coherence_scores
    )

    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

@click.command()
@click.argument('name') #, help='Model name.')
@click.option('--n-topics', default=50, help='Number of topics.')
@click.option('--data-folder', default=CORPUS_FOLDER, help='Corpus folder.')
@click.option('--corpus-filename', default=CORPUS_FOLDER, help='Corpus filename (if text corpus or Sparv XML). Corpus tag if vectorized corpus.')
@click.option('--engine', default="gensim_lda-multicore", help='LDA implementation')
@click.option('--passes', default=None, help='Number of passes.')
@click.option('--alpha', default='symmetric', help='Prior belief of topic probability.')
@click.option('--random-seed', default=None, help="Random seed value")
@click.option('--workers', default=None, help='Number of workers (if applicable).')
@click.option('--max-iter', default=None, help='Max number of iterations.')
@click.option('--prefix', default=None, help='Prefix.')
def _run_model(name, n_topics, data_folder, corpus_filename, engine, passes, random_seed, alpha, workers, max_iter, prefix):
    run_model(name, n_topics, data_folder, corpus_filename, engine, passes, random_seed, alpha, workers, max_iter, prefix)

def run_model(name, n_topics, data_folder, corpus_filename, engine, passes, random_seed, alpha, workers, max_iter, prefix):
    """ runner """

    if engine not in [ y for x, y in ENGINE_OPTIONS ]:
        logging.error("Unknown method {}".format(engine))

    opts = dict(pos_includes='|NN|', lemmatize=True, chunk_size=None)

    reader = sparv_reader.SparvXmlCorpusSourceReader(corpus_filename, **opts)

    kwargs = dict(n_topics=n_topics)

    if workers is not None:
        kwargs.update(dict(workers=workers))

    if passes is not None:
        kwargs.update(dict(passes=passes))

    if max_iter is not None:
        kwargs.update(dict(max_iter=max_iter))

    if prefix is not None:
        kwargs.update(dict(prefix=prefix))

    if random_seed is not None:
        kwargs.update(dict(random_seed=random_seed))

    target_folder = os.path.join(data_folder, name)
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)

    m_data, c_data = topic_model.compute(
        terms=terms,
        doc_term_matrix=dtm,
        id2word=id2token,
        documents=documents,
        method=engine,
        engine_args=kwargs
    )

    target_name = os.path.join(target_folder, 'gensim.model')
    m_data.topic_model.save(target_name)

    topic_model.store_model(m_data, data_folder, name)

    c_data.document_topic_weights = corpus_data.extend_with_document_info(
        c_data.document_topic_weights,
        corpus_data.slim_documents(documents)
    )

    c_data.store(data_folder, name)

if __name__ == '__main__':
    _run_model()
