import types
import textacy
import gensim
import os
import pickle
import json

import text_analytic_tools.utility as utility
import text_analytic_tools.text_analysis.derived_data_compiler as derived_data_compiler
import text_analytic_tools.text_analysis.compute_coherence as coherence

from . import engine_options as options
from pprint import pprint as pp

logger = utility.getLogger("text_analytic_tools")

TEMP_PATH = './tmp/'

# OBS OBS! https://scikit-learn.org/stable/auto_examples/applications/plot_topics_extraction_with_nmf_lda.html
DEFAULT_VECTORIZE_PARAMS = dict(tf_type='linear', apply_idf=False, idf_type='smooth', norm='l2', min_df=1, max_df=0.95)

def compute(
    terms=None,
    documents=None,
    doc_term_matrix=None,
    id2word=None,
    method='sklearn_lda',
    vectorizer_args=None,
    engine_args=None,
    **args
):

    vectorizer_args = utility.extend({}, DEFAULT_VECTORIZE_PARAMS, vectorizer_args or {})

    perplexity_score = None
    coherence_score = 0
    doc_topic_matrix = None
    #doc_term_matrix = None

    if not os.path.exists(TEMP_PATH):
        os.makedirs(TEMP_PATH)

    if method.startswith('sklearn'):

        if doc_term_matrix is None:
            assert terms is not None
            doc_term_matrix, id2word = utility.vectorize_terms(terms, vectorizer_args)

        model = textacy.TopicModel(method.split('_')[1], **engine_args)
        model.fit(doc_term_matrix)

        doc_topic_matrix = model.transform(doc_term_matrix)

        corpus = gensim.matutils.Sparse2Corpus(doc_term_matrix, documents_columns=False)
        # assert corpus.sparse.shape[0] == doc_term_matrix.shape[0]

        perplexity_score = None
        coherence_score = None

    elif method.startswith('gensim_'):

        vectorizer_args = None
        algorithm_name = method.split('_')[1].upper()

        if doc_term_matrix is None:
            id2word = gensim.corpora.Dictionary(terms)
            corpus = [ id2word.doc2bow(tokens) for tokens in terms ]
        else:
            assert id2word is not None
            corpus = gensim.matutils.Sparse2Corpus(doc_term_matrix, documents_columns=False)
            # assert corpus.sparse.shape[0] == doc_term_matrix.shape[0]
        if args.get('tfidf_weiging', False):
            # assert algorithm_name != 'MALLETLDA', 'MALLET training model cannot (currently) use TFIDF weighed corpus'
            tfidf_model = gensim.models.tfidfmodel.TfidfModel(corpus)
            corpus = [ tfidf_model[d] for d in corpus ]

        algorithm = options.engine_options(algorithm_name, corpus, id2word, engine_args)

        engine = algorithm['engine']
        engine_options = algorithm['options']

        model = engine(**engine_options)

        if hasattr(model, 'log_perplexity'):
            perplexity_score = 2 ** model.log_perplexity(corpus, len(corpus))

        coherence_score = coherence.compute_score(id2word, model, corpus)

    c_data = derived_data_compiler.compile_data(
        model,
        corpus,
        id2word,
        documents,
        doc_topic_matrix=doc_topic_matrix,
        n_tokens=200
    )

    m_data = types.SimpleNamespace(
        topic_model=model,
        id2term=id2word,
        corpus=corpus,
        options=dict(
            metrics=dict(
                coherence_score=coherence_score,
                perplexity_score=perplexity_score
            ),
            method=method,
            vec_args=vectorizer_args,
            tm_args=engine_options,
            **args
        ),
        coherence_scores=None
    )

    return m_data, c_data

def store_model(model_data, data_folder, model_name):

    target_folder = os.path.join(data_folder, model_name)
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)

    model_data.doc_term_matrix = None
    model_data.options['tm_args']['id2word'] = None
    model_data.options['tm_args']['corpus'] = None

    filename = os.path.join(target_folder, "model_data.pickle")

    with open(filename, 'wb') as fp:
        pickle.dump(model_data, fp, pickle.HIGHEST_PROTOCOL)

    filename = os.path.join(target_folder, "model_options.json")
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    with open(filename, 'w') as fp:
        json.dump(model_data.options, fp, indent=4, default=default)

def load_model(data_folder, model_name):
    filename = os.path.join(data_folder, model_name, "model_data.pickle")
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data
