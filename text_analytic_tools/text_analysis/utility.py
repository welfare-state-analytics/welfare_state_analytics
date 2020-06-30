import os
import glob
import json
import pandas as pd
import gensim
import textacy
import scipy
import numpy as np

from gensim.models import LdaModel

#from sklearn.preprocessing import normalize

def normalize_array(x: np.ndarray, ord: int=1):
    """
    function that normalizes an ndarray of dim 1d

    Args:
     ``x``: A numpy array

    Returns:
     ``x``: The normalize darray.
    """
    norm = np.linalg.norm(x, ord=ord)
    return x / (norm if norm != 0 else 1.0)

# FIXME: Bug somewhere...
def n_gram_detector(doc_iter, n_gram_size=2, min_count=5, threshold=100):

    for n_span in range(2, n_gram_size+1):
        print('Applying {}_gram detector'.format(n_span))
        n_grams = gensim.models.Phrases(doc_iter(), min_count=min_count, threshold=threshold)
        ngram_modifier = gensim.models.phrases.Phraser(n_grams)
        ngram_doc_iter = lambda: ( ngram_modifier[doc] for doc in doc_iter() )
        doc_iter = ngram_doc_iter

    return doc_iter

def vectorize_terms(terms, vectorizer_args):
    vectorizer = textacy.Vectorizer(**vectorizer_args)
    doc_term_matrix = vectorizer.fit_transform(terms)
    id2word = vectorizer.id_to_term
    return doc_term_matrix, id2word

def create_dictionary(id2word):
    dictionary = gensim.corpora.Dictionary()
    dictionary.id2token = id2word
    dictionary.token2id = dict((v, k) for v, k in id2word.items())
    return dictionary

def compute_topic_proportions_deprecated(document_topic_weights, doc_length_series, n_terms_column='words'):

    '''
    Topic proportions are computed in the same as in LDAvis.

    Computes topic proportions over entire corpus.
    The document topic weight (pivot) matrice is multiplies by the length of each document
      i.e. each weight are multiplies ny the document's length.
    The topic frequency is then computed by summing up all values over each topic
    This value i then normalized by total sum of matrice

    theta matrix: with each row containing the probability distribution
      over topics for a document, with as many rows as there are documents in the
      corpus, and as many columns as there are topics in the model.

    doc_length integer vector containing token count for each document in the corpus

    '''
    # compute counts of tokens across K topics (length-K vector):
    # (this determines the areas of the default topic circles when no term is highlighted)
    # topic.frequency <- colSums(theta * doc.length)
    # topic.proportion <- topic.frequency/sum(topic.frequency)

    if document_topic_weights.index.name == 'document_id':
        document_topic_weights.index.name = 'id'

    theta = pd.pivot_table(
        document_topic_weights,
        values='weight',
        index=['document_id'],
        columns=['topic_id']
    ) #.set_index('document_id')

    theta_mult_doc_length = theta.mul(doc_length_series[n_terms_column], axis=0)

    topic_frequency = theta_mult_doc_length.sum()
    topic_proportion = topic_frequency / topic_frequency.sum()

    return topic_proportion

def compute_topic_proportions(document_topic_weights, doc_length_series):
    """Computes topic proportations as LDAvis. Fast version
    Parameters
    ----------
    document_topic_weights : :class:`~pandas.DataFrame`
        Document Topic Weights
    doc_length_series : numpy.ndarray
        Document lengths
    Returns
    -------
    numpy array
    """
    theta = scipy.sparse.coo_matrix((document_topic_weights.weight, (document_topic_weights.document_id, document_topic_weights.topic_id)))
    theta_mult_doc_length = theta.T.multiply(doc_length_series).T
    topic_frequency = theta_mult_doc_length.sum(axis=0).A1
    topic_proportion = topic_frequency / topic_frequency.sum()
    return topic_proportion

def malletmodel2ldamodel(mallet_model, gamma_threshold=0.001, iterations=50):
    """Convert :class:`~gensim.models.wrappers.ldamallet.LdaMallet` to :class:`~gensim.models.ldamodel.LdaModel`.
    This works by copying the training model weights (alpha, beta...) from a trained mallet model into the gensim model.
    Parameters
    ----------
    mallet_model : :class:`~gensim.models.wrappers.ldamallet.LdaMallet`
        Trained Mallet model
    gamma_threshold : float, optional
        To be used for inference in the new LdaModel.
    iterations : int, optional
        Number of iterations to be used for inference in the new LdaModel.
    Returns
    -------
    :class:`~gensim.models.ldamodel.LdaModel`
        Gensim native LDA.
    """
    model_gensim = LdaModel(
        id2word=mallet_model.id2word, num_topics=mallet_model.num_topics,
        alpha=mallet_model.alpha, eta=0,
        iterations=iterations,
        gamma_threshold=gamma_threshold,
        dtype=np.float64  # don't loose precision when converting from MALLET
    )
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim

def read_json(path):
    with open(path) as fp:
        return json.load(fp)

def find_models(corpus_folder):
    folders = [ os.path.split(x)[0] for x in glob.glob(os.path.join(corpus_folder, "*", "model_data.pickle")) ]
    models = [
        {
            'folder': x,
            'name': os.path.split(x)[1],
            'options': read_json(os.path.join(x, "model_options.json"))
        }
        for x in folders
    ]
    return models


def display_termite_plot(model, id2term, doc_term_matrix):
    #tm = get_current_model().tm_model
    #id2term = get_current_model().tm_id2term
    #dtm = get_current_model().doc_term_matrix

    if hasattr (model, 'termite_plot'):
        # doc_term_matrix [ dictionary.doc2bow(doc) for doc in doc_clean ]
        model.termite_plot(
            doc_term_matrix,
            id2term,
            topics=-1,
            sort_topics_by='index',
            highlight_topics=None,
            n_terms=50,
            rank_terms_by='topic_weight',
            sort_terms_by='seriation',
            save=False
        )
