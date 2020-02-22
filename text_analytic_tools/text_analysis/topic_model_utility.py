import types
import pandas as pd
import text_analytic_tools.utility as utility
import gensim

logger = utility.getLogger('corpus_text_analysis')

class TopicModelException(Exception):
    pass

class TopicModelContainer():
    """Class for current (last) computed or loaded model
    """
    _singleton = None

    def __init__(self):
        self._data = None

    @staticmethod
    def singleton():
        TopicModelContainer._singleton = TopicModelContainer._singleton or TopicModelContainer()
        return TopicModelContainer._singleton

    @property
    def data(self):
        if self._data is None:
            raise TopicModelException('Model not loaded or computed')
        return self._data

    @data.setter
    def data(self, value):
        self._data = value

    @property
    def topic_model(self):
        return self.data.topic_model

    @property
    def id2term(self):
        return self.data.id2term

    @property
    def processed(self):
        return self.data.processed

    @property
    def num_topics(self):
        tm = self.topic_model
        if tm is None:
            return 0
        if hasattr(tm, 'num_topics'):
            return tm.num_topics
        if hasattr(tm, 'n_topics'):
            return tm.n_topics

        return 0

    @property
    def relevant_topics(self):
        return self.processed.relevant_topic_ids

def compile_dictionary(model, vectorizer=None):
    logger.info('Compiling dictionary...')
    if hasattr(model, 'id2word'):
        # Gensim LDA model
        id2word = model.id2word
        dfs = list(id2word.dfs.values()) or 0 if hasattr(id2word, 'dfs') else 0
    else:
        # Textacy/scikit-learn model
        assert vectorizer is not None, 'vectorizer is empty'
        id2word = vectorizer.id_to_term
        dfs = 0
    token_ids, tokens = list(zip(*id2word.items()))
    dictionary = pd.DataFrame({
        'token_id': token_ids,
        'token': tokens,
        'dfs': dfs
    }).set_index('token_id')[['token', 'dfs']]
    return dictionary

def compile_topic_token_weights(model, dictionary, n_tokens=200):

    logger.info('Compiling topic-tokens weights...')

    id2term = dictionary.token.to_dict()
    term2id = { v: k for k,v in id2term.items() }

    if hasattr(model, 'show_topics'):
        # Gensim LDA model
        #topic_data = model.show_topics(model.num_topics, num_words=n_tokens, formatted=False)
        topic_data = model.show_topics(num_topics=-1, num_words=n_tokens, formatted=False)
    else:
        # Textacy/scikit-learn model
        topic_data = model.top_topic_terms(id2term, topics=-1, top_n=n_tokens, weights=True)

    df_topic_weights = pd.DataFrame(
        [ (topic_id, token, weight)
            for topic_id, tokens in topic_data
                for token, weight in tokens if weight > 0.0 ],
        columns=['topic_id', 'token', 'weight']
    )

    term2id = { v: k for k,v in id2term.items() }
    df_topic_weights['token_id'] = df_topic_weights.token.apply(lambda x: term2id[x])

    return df_topic_weights[['topic_id', 'token_id', 'token', 'weight']]

def compile_topic_token_overview(topic_token_weights, alpha=None, n_tokens=200):
    """
    Group by topic_id and concatenate n_tokens words within group sorted by weight descending.
    There must be a better way of doing this...
    """
    logger.info('Compiling topic-tokens overview...')

    df = topic_token_weights.groupby('topic_id')\
        .apply(lambda x: sorted(list(zip(x["token"], x["weight"])), key=lambda z: z[1], reverse=True))\
        .apply(lambda x: ' '.join([z[0] for z in x][:n_tokens])).reset_index()
    df.columns = ['topic_id', 'tokens']
    df['alpha'] = df.topic_id.apply(lambda topic_id: alpha[topic_id]) if alpha is not None else 0.0

    return df.set_index('topic_id')

def compile_document_topics(model, corpus, documents, doc_topic_matrix=None, minimum_probability=0.001):

    try:
        def document_topics_iter(model, corpus, minimum_probability=0.0):

            if isinstance(model, gensim.models.LsiModel):
                # Gensim LSI Model
                #logger.warning('FIXME!!! Gensim LSI Model is to big!!! ')
                data_iter = enumerate(model[corpus])
            elif hasattr(model, 'get_document_topics'):
                # Gensim LDA Model
                data_iter = enumerate(model.get_document_topics(corpus, minimum_probability=minimum_probability))
            elif hasattr(model, 'load_document_topics'):
                # Gensim MALLET wrapper
                data_iter = enumerate(model.load_document_topics())
            elif hasattr(model, 'top_doc_topics'):
                # scikit-learn
                assert doc_topic_matrix is not None, "doc_topic_matrix not supplied"
                data_iter = model.top_doc_topics(doc_topic_matrix, docs=-1, top_n=1000, weights=True)
            else:
                data_iter = ( (document_id, model[corpus[document_id]]) for document_id in range(0, len(corpus)) )

                # assert False, 'compile_document_topics: Unknown topic model'

            for document_id, topic_weights in data_iter:
                for (topic_id, weight) in ((topic_id, weight) for (topic_id, weight) in topic_weights if weight >= minimum_probability):
                    yield (document_id, topic_id, weight)

        '''
        Get document topic weights for all documents in corpus
        Note!  minimum_probability=None filters less probable topics, set to 0 to retrieve all topcs

        If gensim model then use 'get_document_topics', else 'load_document_topics' for mallet model
        '''
        logger.info('Compiling document topics...')
        logger.info('  Creating data iterator...')
        data = document_topics_iter(model, corpus, minimum_probability)

        logger.info('  Creating frame from iterator...')
        df_doc_topics = pd.DataFrame(data, columns=[ 'document_id', 'topic_id', 'weight' ]).set_index('document_id')

        logger.info('  Merging data...')
        df = pd.merge(documents, df_doc_topics, how='inner', left_index=True, right_index=True)
        logger.info('  DONE!')
        return df
    except Exception as ex:
        logger.error(ex)
        return None

# FIXME VARYING ASPECTS: year_column='signed_year' for tCoIR
def compile_metadata(model, corpus, id2term, documents, vectorizer=None, doc_topic_matrix=None, n_tokens=200, year_column='year'):
    '''
    Compile metadata associated to given model and corpus
    '''
    dictionary = compile_dictionary(model, vectorizer)
    topic_token_weights = compile_topic_token_weights(model, dictionary, n_tokens=n_tokens)
    alpha = model.alpha if 'alpha' in model.__dict__ else None
    topic_token_overview = compile_topic_token_overview(topic_token_weights, alpha)
    document_topic_weights = compile_document_topics(model, corpus, documents, doc_topic_matrix=doc_topic_matrix, minimum_probability=0.001)

    # PATCH: take care of case when year is 0
    assert year_column in documents.columns
    years_series = documents[year_column]
    year_period = (years_series[years_series > 0].min(), years_series.max())
    #year_period = (documents[year_column].min(), documents[year_column].max()) if year_column in documents.columns else (None, None)

    relevant_topic_ids = list(document_topic_weights.topic_id.unique())

    return types.SimpleNamespace(
        dictionary=dictionary,
        documents=documents,
        topic_token_weights=topic_token_weights,
        topic_token_overview=topic_token_overview,
        document_topic_weights=document_topic_weights,
        year_period=year_period,
        relevant_topic_ids=relevant_topic_ids
    )

def get_topic_titles(topic_token_weights, topic_id=None, n_tokens=100):
    df_temp = topic_token_weights if topic_id is None else topic_token_weights[(topic_token_weights.topic_id==topic_id)]
    df = df_temp\
            .sort_values('weight', ascending=False)\
            .groupby('topic_id')\
            .apply(lambda x: ' '.join(x.token[:n_tokens].str.title()))
    return df

def get_topic_title(topic_token_weights, topic_id, n_tokens=100):
    return get_topic_titles(topic_token_weights, topic_id, n_tokens=n_tokens).iloc[0]

def get_topic_tokens(topic_token_weights, topic_id=None, n_tokens=100):
    df_temp = topic_token_weights if topic_id is None else topic_token_weights[(topic_token_weights.topic_id == topic_id)]
    df = df_temp.sort_values('weight', ascending=False)[:n_tokens]
    return df

def get_topics_unstacked(model, n_tokens=20, id2term=None, topic_ids=None):

    if hasattr(model, 'num_topics'):
        # Gensim LDA model
        show_topic = lambda topic_id: model.show_topic(topic_id, topn=n_tokens)
        n_topics = model.num_topics
    elif hasattr(model, 'm_T'):
        # Gensim HDP model
        show_topic = lambda topic_id: model.show_topic(topic_id, topn=n_tokens)
        n_topics = model.m_T
    else:
        # Textacy/scikit-learn model
        def scikit_learn_show_topic(topic_id):
            topic_words = list(model.top_topic_terms(id2term, topics=(topic_id,), top_n=n_tokens, weights=True))
            if len(topic_words) == 0:
                return []
            else:
                return topic_words[0][1]
        show_topic = scikit_learn_show_topic
        n_topics = model.n_topics

    topic_ids = topic_ids or range(n_topics)

    return pd.DataFrame({
        'Topic#{:02d}'.format(topic_id+1) : [ word[0] for word in show_topic(topic_id) ]
            for topic_id in topic_ids
    })

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

from gensim.models import LdaModel
import numpy

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
        dtype=numpy.float64  # don't loose precision when converting from MALLET
    )
    model_gensim.state.sstats[...] = mallet_model.wordtopics
    model_gensim.sync_state()
    return model_gensim
