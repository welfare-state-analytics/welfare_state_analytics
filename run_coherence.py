import os, sys
import click

sys.path = [ os.path.abspath("../../..") ] + sys.path

import westac.notebooks.political_in_newspapers.corpus_data as corpus_data
import text_analytic_tools.text_analysis.topic_model as topic_model
import types
import pickle
import logging

CORPUS_FOLDER = "/home/roger/source/welfare_state_analytics/data/textblock_politisk"

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
@click.option('--engine', default="gensim_lda-multicore", help='LDA implementation')
@click.option('--passes', default=20, help='Number of passes.')
@click.option('--alpha', default='symmetric', help='Prior belief of topic probability.')
@click.option('--workers', default=None, help='Number of workers (if applicable).')
def run_model(name, n_topics, data_folder, engine, passes, alpha, workers):
    """ runner """

    if engine not in [ y for x, y in ENGINE_OPTIONS ]:
        logging.error("Unknown method {}".format(engine))

    #dtm, documents, id2token = corpus_data.load_as_dtm2(data_folder, [1, 3])

    dtm, documents, id2token = corpus_data.load_dates_subset_as_dtm(data_folder, ["1949-06-16", "1959-06-16", "1969-06-16", "1979-06-16", "1989-06-16"])

    kwargs = dict(n_topics=n_topics)

    if workers is not None:
        kwargs.update(dict(workers=workers))

    m_data, c_data = topic_model.compute(
        doc_term_matrix=dtm,
        id2word=id2token,
        documents=documents,
        method=engine,
        engine_args=kwargs
    )

    target_folder = os.path.join(data_folder, name)
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)

    target_name = os.path.join(target_folder, 'gensim.model')
    m_data.topic_model.save(target_name)

    topic_model.store_model(m_data, data_folder, name)

    c_data.document_topic_weights = corpus_data.extend_with_document_info(
        c_data.document_topic_weights,
        corpus_data.slim_documents(documents)
    )

    c_data.store(data_folder, name)

if __name__ == '__main__':
    run_model()
