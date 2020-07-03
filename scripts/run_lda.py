import os, sys
import click

sys.path = [ os.path.abspath("..") ] + sys.path

import westac.notebooks.political_in_newspapers.corpus_data as corpus_data
import text_analytic_tools.text_analysis.topic_model as topic_model
import westac.corpus.vectorized_corpus as vectorized_corpus
import westac.corpus.corpus_vectorizer as corpus_vectorizer

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
@click.option('--passes', default=None, help='Number of passes.')
@click.option('--alpha', default='symmetric', help='Prior belief of topic probability.')
@click.option('--random-seed', default=None, help="Random seed value")
@click.option('--workers', default=None, help='Number of workers (if applicable).')
@click.option('--max-iter', default=None, help='Max number of iterations.')
@click.option('--prefix', default=None, help='Prefix.')
@click.option('--corpus-type', default='R', type=click.Choice(['R', 'vectorized'], case_sensitive=False))
@click.option('--vectorized-corpus-dump-tag', default=None, help="Vectorized corpus dump (prefix) tag")
def _run_model(name, n_topics, data_folder, engine, passes, random_seed, alpha, workers, max_iter, prefix, corpus_type, vectorized_corpus_dump_tag):
    run_model(name, n_topics, data_folder, engine, passes, random_seed, alpha, workers, max_iter, prefix, corpus_type, vectorized_corpus_dump_tag)

def run_model(name, n_topics, data_folder, engine, passes, random_seed, alpha, workers, max_iter, prefix, corpus_type, vectorized_corpus_dump_tag):
    """ runner """

    if engine not in [ y for x, y in ENGINE_OPTIONS ]:
        logging.error("Unknown method {}".format(engine))

    if corpus_type == 'vectorized':

        assert vectorized_corpus_dump_tag is not None, "error: dump tag not specified for vectorized corpus"
        assert vectorized_corpus.VectorizedCorpus.dump_exists(vectorized_corpus_dump_tag, data_folder), "error: no dump for given tag exists"

        v_corpus = vectorized_corpus.VectorizedCorpus\
            .load(vectorized_corpus_dump_tag, data_folder)

        dtm = v_corpus.data
        id2token = v_corpus.id2token
        documents = v_corpus.document_index
        documents['publication_id'] = 1

    else:
        dtm, documents, id2token = corpus_data.load_as_dtm2(data_folder, [1, 3])
        # dtm, documents, id2token = corpus_data.load_dates_subset_as_dtm(data_folder, ["1949-06-16", "1959-06-16", "1969-06-16", "1979-06-16", "1989-06-16"])

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

# engine = "gensim_mallet-lda"
# workers = 4
# max_iter = 4000
# passes = 1
# random_seed = None
# alpha = None
# corpus_type = "vectorized"
# vectorized_corpus_dump_tag = 'tCoIR_en_45-72_renamed_L0_+N_+S'
# data_folder = "/home/roger/source/welfare_state_analytics/data/tCoIR/"

# for n_topics in [50, 100, 150, 200, 250, 300, 350, 400]:
#     name = "treaties.{}".format(n_topics)
#     prefix = os.path.join(data_folder, "treaties.{}/".format(n_topics))
#     run_model(name, n_topics, data_folder, engine, passes, random_seed, alpha, workers, max_iter, prefix, corpus_type, vectorized_corpus_dump_tag)