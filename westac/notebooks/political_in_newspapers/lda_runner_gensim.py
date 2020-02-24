import os, sys
import click

sys.path = [ os.path.abspath("../../..") ] + sys.path

import westac.notebooks.political_in_newspapers.corpus_data as corpus_data
import text_analytic_tools.text_analysis.topic_model as topic_model
import text_analytic_tools.text_analysis.topic_model_utility as topic_model_utility
import types
import pickle

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
@click.argument('model_name') #, help='Model name.')
@click.option('--n-topics', default=50, help='Number of topics.')
@click.option('--data-folder', default=CORPUS_FOLDER, help='Corpus folder.')
@click.option('--passes', default=20, help='Number of passes.')
@click.option('--alpha', default='symmetric', help='Prior belief of topic probability.')
def run_model(model_name, n_topics, data_folder, passes, alpha):
    """ runner """
    method = 'gensim_lda-multicore'

    # dtm, documents, id2token = corpus_data.load_as_dtm(data_folder)

    # dtm, documents, id2token = corpus_data.load_dates_subset_as_dtm(data_folder, ["1949-06-16", "1959-06-16", "1969-06-16", "1979-06-16"])
    dtm, documents, id2token = corpus_data.load_dates_subset_as_dtm(data_folder, ["1949-06-16", "1959-06-16", "1969-06-16", "1979-06-16", "1989-06-16"])

    model_data, compiled_data = topic_model.compute(
        doc_term_matrix=dtm,
        id2word=id2token,
        documents=documents,
        method=method,
        tm_args=dict(n_topics=n_topics, workers=7)
    )

    target_folder = os.path.join(data_folder, model_name)
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)

    target_name = os.path.join(target_folder, 'gensim.model')
    model_data.topic_model.save(target_name)

    topic_model.store_model(model_data, data_folder, model_name)

    compiled_data.document_topic_weights = corpus_data.extend_with_document_info(
        compiled_data.document_topic_weights,
        corpus_data.slim_documents(documents)
    )

    topic_model_utility.store_compiled_data(compiled_data, data_folder, model_name)

if __name__ == '__main__':
    run_model()
