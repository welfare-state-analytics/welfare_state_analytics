# -*- coding: utf-8 -*-
"""Topic Modeling(LDA) & Word2Vec.
"""

import os
import re
import sys
import json
import zipfile
import urllib
import random
import warnings
import subprocess
import itertools as it
import functools as ft
import collections
from glob import glob
from pprint import pprint
import numpy as np
import pandas as pd

import gensim
import pyLDAvis
import pyLDAvis.gensim as gensimvis

import unidecode
from unicodedata import normalize

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


__all__ = []
__all__ += [
    # 'compute_coherence_values',
    # 'pick_best_topics',
    # 'groupby_top_n',
    # 'get_terminfo_table',
    'TopicModeler',
]

def compute_coherence_values(
        dictionary,
        corpus,
        id2word,
        texts,
        num_topic_list=[5, 10],
        lda_type='default',  # {'default', 'mallet'}
        workers_n=2,
        random_seed=1,
        ):
    """
    Compute c_v coherence for various number of topics

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    texts : List of input texts
    limit : Max num of topics

    Returns:
    -------
    model_list : List of LDA topic models
    coherence_value_list : Coherence values corresponding to the LDA model,
        with respective number of topics
    """
    model_list = []
    coherence_list = []

    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)
    if lda_type == 'default':
        for num_topics in num_topic_list:
            model = gensim.models.LdaMulticore(
                corpus,
                num_topics=num_topics,
                id2word=id2word,
                passes=2,
                workers=workers_n,
                eta='symmetric',
                decay=.8,  # {.5, 1.}
                per_word_topics=False,
                offset=1.,
                iterations=30,
                gamma_threshold=.001,  # 0.001,
                minimum_probability=.05,  # .01,
                minimum_phi_value=.01,
                random_state=random_seed,
            )
            coherence_model = gensim.models.CoherenceModel(
                model=model,
                texts=texts,
                dictionary=id2word,
                coherence='c_v',
            )

            model_list += [model]
            coherence_list += [coherence_model.get_coherence()]

    elif lda_type == 'hdp':
        for num_topics in num_topic_list:
            model = gensim.models.HdpModel(
                corpus,
                id2word=id2word,
                T=3,
                # alpha=,
                K=num_topics,
                # gamma=,
                # decay=.5, # {.5, 1.}
                # per_word_topics=True,
                # minimum_probability=.1,
                # minimum_phi_value=.01,
                random_state=random_seed,
            )
            coherence_model = gensim.models.CoherenceModel(
                model=model,
                texts=texts,
                dictionary=id2word,
                coherence='c_v',
            )

            model_list += [model]
            coherence_list += [coherence_model.get_coherence()]

    elif lda_type == 'mallet':
        # Download File: http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip
        mallet_url = 'http://mallet.cs.umass.edu/dist/mallet-2.0.8.zip'
        mallet_filename = mallet_url.split('/')[-1]
        mallet_unzipped_dirname = mallet_filename.split('.zip')[0]
        mallet_path = f'{mallet_unzipped_dirname}/bin/mallet'


        if not os.path.exists(mallet_path):
            # download the url contents in binary format
            urllib.request.urlretrieve(mallet_url, mallet_filename)

            # open method to open a file on your system and write the contents
            with zipfile.ZipFile(mallet_filename, "r") as zip_ref:
                zip_ref.extractall(mallet_unzipped_dirname)

        for num_topics in num_topic_list:
            model = gensim.models.wrappers.LdaMallet(
                mallet_path,
                corpus=corpus,
                num_topics=num_topics,
                id2word=id2word,
            )
            coherence_model = gensim.models.CoherenceModel(
                model=model,
                texts=texts,
                dictionary=id2word,
                coherence='c_v',
            )

            model_list += [model]
            # coherence_list += [coherence_model.get_coherence()]

    return model_list, coherence_list


# Print the coherence scores
def pick_best_topics(
        dictionary,
        corpus,
        texts,
        num_topic_list=[5, 7, 10, 12, 15, 17, 20],
        lda_type='default',
        workers_n=2,
        random_seed=1,
        ):
    model_list, coherence_value_list = compute_coherence_values(
        dictionary=dictionary,
        corpus=corpus,
        id2word=dictionary,
        texts=texts,
        num_topic_list=num_topic_list,
        lda_type=lda_type,
        workers_n=workers_n,
        random_seed=random_seed,
        #  start=2, limit=40, step=6,
    )

    paired = zip(model_list, coherence_value_list)
    ordered = sorted(paired, key=lambda x: x[1], reverse=True)
    best_model = ordered[0][0]

    model_coh_list = []
    model_topicnum_list = []
    for i, (m, cv) in enumerate(zip(model_list, coherence_value_list)):
        topic_num = m.num_topics
        coh_value = round(cv, 4)
        print(
            f'[{i}] Num Topics ({topic_num:2})' +
            f' has Coherence Value of {coh_value}'
        )
        # model_topicnum_list += [(topic_num, m)]
        model_coh_list += [(topic_num, m, coh_value)]
        model_topicnum_list += [
            (topic_num, {'model': m, 'coherence': coh_value})
        ]

    model_dict = dict(model_topicnum_list)
    print(f'Best N topics: {best_model.num_topics}')

    return best_model, model_coh_list, model_dict, coherence_value_list


def get_saliency(tinfo_df):
    r"""Calculate Saliency for terms within a topic.

    .. math::
        distinctiveness(w) = \sum P(t \vert w) log\frac{P(t \vert w)}{P(w)}
        saliency(w) = P(w) \times distinctiveness(w)
    <div align="right">(Chuang, J., 2012. Termite: Visualization techniques for assessing textual topic models)</div>

    Parameters
    ----------
    tinfo: pandas.DataFrame
        `pyLDAvis.gensim.prepare`.to_dict()['tinfo'] containing
        ['Category', 'Freq', 'Term', 'Total', 'loglift', 'logprob']

    Return
    ------
    saliency: float

    """

    saliency = tinfo_df['Freq'] / tinfo_df['Total']

    return saliency


def get_relevance(tinfo_df, lambda_val=.6):
    r"""Calculate Relevances with a given lambda value.

    .. math::
        relevance(t,w) = \lambda \cdot P(w \vert t) + (1 - \lambda) \cdot \frac{P(w \vert t)}{P(w)}
        Recommended \lambda = 0.6
    <div align="right">(Sievert, C., 2014. LDAvis: A method for visualizing and interpreting topics)</div>

    Parameters
    ----------
    tinfo: pandas.DataFrame
        `pyLDAvis.gensim.prepare`.to_dict()['tinfo'] containing
        ['Category', 'Freq', 'Term', 'Total', 'loglift', 'logprob']

    lambda_val: float
        lambda_ratio between {0-1}. default is .6 (recommended from its paper)

    Return
    ------
    relevance: float

    """

    relevance = l * tinfo_df['logprob'] + (1 - l) * tinfo_df['loglift']

    return relevance


def groupby_top_n(
        dataframe,
        group_by=None,
        order_by=None,
        ascending=False,
        n=5,
        ):

    res_df = (
        dataframe
        .groupby(group_by)
        [dataframe.columns.drop(group_by)]
        .apply(
            lambda x: x.sort_values(order_by, ascending=ascending).head(n)
        )
    )
    return res_df


def _df_with_names(data, index_name, columns_name):
    """A renaming function from `pyLDAvis._prepare`.
    """
    if type(data) == pd.DataFrame:
        # we want our index to be numbered
        df = pd.DataFrame(data.values)
    else:
        df = pd.DataFrame(data)
    df.index.name = index_name
    df.columns.name = columns_name
    return df


def _series_with_name(data, name):
    """A renaming function from `pyLDAvis._prepare`.
    """
    if type(data) == pd.Series:
        data.name = name
        # ensures a numeric index
        return data.reset_index()[name]
    else:
        return pd.Series(data, name=name)


def get_terminfo_table(
        lda_model,
        corpus: list = None,
        dictionary: gensim.corpora.dictionary.Dictionary = None,
        doc_topic_dists=None,
        use_gensim_prepared=True,
        top_n=10,
        workers_n=-1,
        r_normalized=False,
        relevence_lambda_val=.6,
        random_seed=1,
        ):

    if random_seed:
        random.seed(random_seed)
        np.random.seed(random_seed)

    if use_gensim_prepared:
        _prepared = gensimvis.prepare(
            topic_model=lda_model,
            corpus=corpus,
            dictionary=dictionary,
            doc_topic_dist=None,
            R=len(dictionary),
            # lambda_step=0.2,
            mds='tsne',
            # mds=<function js_PCoA>,
            n_jobs=workers_n,
            plot_opts={'xlab': 'PC1', 'ylab': 'PC2'},
            sort_topics=True,
        )
        tinfo_df = pd.DataFrame(_prepared.to_dict()['tinfo'])

        tinfo_df['topic_term_dists'] = np.exp(tinfo_df['logprob'])
        tinfo_df['term_proportion'] = (
            np.exp(tinfo_df['logprob']) / np.exp(tinfo_df['loglift'])
        )
        tinfo_df['saliency'] = get_saliency(tinfo_df)
        tinfo_df['relevance'] = get_relevance(
            tinfo_df,
            lambda_val=relevence_lambda_val,
        )

        tinfo_df['term_prob'] = np.exp(tinfo_df['logprob'])
        tinfo_df['term_r_prob'] = np.exp(tinfo_df['relevance'])
        tinfo_df['term_r_adj_prob'] = (
            tinfo_df
            .groupby(['Category'])
            ['term_r_prob']
            .apply(lambda x: x / x.sum())
        )

        if r_normalized:
            r_colname = 'term_r_adj_prob'
        else:
            r_colname = 'term_r_prob'

        relevance_score_df = (
            tinfo_df[tinfo_df['Category'] != 'Default']
            .groupby(['Category', 'Term'])
            [[r_colname]]
            .sum()
            .reset_index()
        )

        # corpus_dict_df = pd.DataFrame(
        #     # It is possible
        #     # because the keys of this dictionary generated from range(int).
        #     # Usually the dictionary is iterable but not ordered.
        #     list(dictionary.values()),
        #     # [dictionary[i] for i, _ in enumerate(dictionary)],
        #     columns=['Term'],
        # )
        # corpus_dict_df['term_id'] = corpus_dict_df.index
        corpus_dict_df = pd.DataFrame(
            list(dictionary.items()),
            columns=['term_id', 'Term'],
        )
        corpus_dict_df.set_index('term_id', drop=False, inplace=True)

        r_score_df = pd.merge(
            relevance_score_df,
            corpus_dict_df,
            on=['Term'],
            how='left',
        )
        r_score_df['category_num'] = (
            r_score_df['Category']
            .str
            .replace('Topic', '')
            .astype(int) - 1
        ).astype('category')
        r_score_df.set_index(['category_num', 'term_id'], inplace=True)
        ixs = pd.IndexSlice

        topic_list = r_score_df.index.levels[0]
        equal_prob = 1. / len(topic_list)
        empty_bow_case_list = list(
            zip(topic_list, [equal_prob] * len(topic_list))
        )

        def get_bow_score(
                bow_chunk,
                score_df=r_score_df,
                colname=r_colname,
                ):

            bow_chunk_arr = np.array(bow_chunk)
            word_id_arr = bow_chunk_arr[:, 0]
            word_cnt_arr = bow_chunk_arr[:, 1]

            # normed_word_cnt_arr = (word_cnt_arr / word_cnt_arr.sum()) * 10
            clipped_word_cnt_arr = np.clip(word_cnt_arr, 0, 3)

            score_series = (score_df.loc[ixs[:, word_id_arr], :]
                            .groupby(level=0)
                            [colname]
                            .apply(lambda x: x @ clipped_word_cnt_arr)
                            )
            score_list = list(score_series.iteritems())
            # normed_score_series = score_series / score_series.sum()
            # score_list = list(normed_score_series.iteritems())

            return score_list

        bow_score_list = [
            get_bow_score(bow_chunk)
            if bow_chunk not in (None, [])
            else empty_bow_case_list
            for bow_chunk in corpus
        ]

        relevant_terms_df = groupby_top_n(
            tinfo_df,
            group_by=['Category'],
            order_by=['relevance'],
            ascending=False,
            n=top_n,
        )
        relevant_terms_df['rank'] = (
            relevant_terms_df
            .groupby(['Category'])
            ['relevance']
            # .rank(method='max')
            .rank(method='max', ascending=False)
            .astype(int)
        )

    else:
        vis_attr_dict = gensimvis._extract_data(
            topic_model=lda_model,
            corpus=corpus,
            dictionary=dictionary,
            doc_topic_dists=None,
        )
        topic_term_dists = _df_with_names(
            vis_attr_dict['topic_term_dists'],
            'topic', 'term',
        )
        doc_topic_dists = _df_with_names(
            vis_attr_dict['doc_topic_dists'],
            'doc', 'topic',
        )
        term_frequency = _series_with_name(
            vis_attr_dict['term_frequency'],
            'term_frequency',
        )
        doc_lengths = _series_with_name(
            vis_attr_dict['doc_lengths'],
            'doc_length',
        )
        vocab = _series_with_name(
            vis_attr_dict['vocab'],
            'vocab',
        )

        ## Topic
        # doc_lengths @ doc_topic_dists
        topic_freq = (doc_topic_dists.T * doc_lengths).T.sum()
        topic_proportion = (topic_freq / topic_freq.sum())

        ## reorder all data based on new ordering of topics
        # topic_proportion = (topic_freq / topic_freq.sum()).sort_values(ascending=False)
        # topic_order = topic_proportion.index
        # topic_freq = topic_freq[topic_order]
        # topic_term_dists = topic_term_dists.iloc[topic_order]
        # doc_topic_dists = doc_topic_dists[topic_order]

        # token counts for each term-topic combination
        term_topic_freq = (topic_term_dists.T * topic_freq).T
        term_frequency = np.sum(term_topic_freq, axis=0)

        ## Term
        term_proportion = term_frequency / term_frequency.sum()

        # compute the distinctiveness and saliency of the terms
        topic_given_term = topic_term_dists / topic_term_dists.sum()
        kernel = (topic_given_term *
                  np.log((topic_given_term.T / topic_proportion).T))
        distinctiveness = kernel.sum()
        saliency = term_proportion * distinctiveness

        default_tinfo_df = pd.DataFrame(
            {
                'saliency': saliency,
                'term': vocab,
                'freq': term_frequency,
                'total': term_frequency,
                'category': 'default',
                'logprob': np.arange(len(vocab), 0, -1),
                'loglift': np.arange(len(vocab), 0, -1),
            }
        )

        log_lift = np.log(topic_term_dists / term_proportion)
        log_prob = log_ttd = np.log(topic_term_dists)

    return _prepared, tinfo_df, relevant_terms_df, r_score_df, bow_score_list


class TopicModeler(object):
    """Topic Modeling via LDA(Latent Diriclet Allocation).

    Get tokenized from text.

    Parameters
    ----------

    sentence_list: list
        A list of raw sentences.

    tokenized_sentence_list: list
        A nested list of tokenized sentences.

    Attributes
    ----------

    After `__init__`:
        self.sentences:
            A list of raw sentences.
        self.tokenized: list
            A nested list of tokenized sentences.

        self.corpora_dict: `gensim.corpora.dictionary.Dictionary`
            A token dictionary from a given text.

        self.bow_corpus_idx: list
            A nested list, which contains converted documents into a list of token indices.

        self.bow_corpus_doc: list
            A nested list, which contains converted documents into a list of token words.

    After `train_lda` or `load_lda`:
        self.best_lda_model: dict
            A dict contains the best model & its coherence value.
            `{'coherence': int, 'model':gensim.models.ldamulticore.LdaMulticore}`

        self.lda_model_list = model_coh_list
            A nested list of `[topic_num, model, coherence_value]`

        self.lda_model_dict:
            A nested dict as `{topic_num: {'coherence': int, 'model': `gensim.models.ldamulticore.LdaMulticore`}}`

        self.trained: bool
            `True` If trained or properly loaded.

    After `visualize_lda_to_html`:
        self.selected_topic_num: int
            A int of selected topic number.

        self.selected_model: `gensim.models.ldamulticore.LdaMulticore`

        self.vis_prepared: `pyLDAvis.prepared_data.PreparedData`

        self.total_terms_df
            `tinfo_table`, `'Default'` removed.

        self.top_relevant_terms_df: `pandas.DataFrame`
            A rank table of `Category`.

        self.r_adj_score_df: `pandas.DataFrame`
            A tinfo table, considering saliency and relevence score.

        self.bow_score_list: list
            Scores of each sentence, based on bow_corpus, clipped by (0, 3).

    After `estimate_topics_by_documents` or `load_estimated`:
        self.dominant_topic_estimation_df: `pandas.DataFrame`
            A dataframe contains `['lda_prob', 'dominant_topic', 'contribution', 'topic_keywords']`

        self.topic_freq_df: `pandas.DataFrame`
            A rank table by topic frequency.

    After `get_representitive_documents` or `load_representitive_documents`:
        self.representitive_docs: `pandas.DataFrame`

    After `get_representitive_candidates`:
        return `repr_sentences, repr_bow_corpus_doc, repr_bow_corpus_idx`


    Methods
    -------

    train_lda

    save_lda

    load_lda

    pick_best_lda_topics

    visualize_lda_to_html

    estimate_topics_by_documents

    get_representitive_documents

    See Also
    --------
    Preprocessing
        ``unipy_nlp.preprocessing.Preprocessor``

    POS-Tagging
        ``konlpy.tag.Mecab``

    Byte-Pair Encoding
        ``sentencepiece``

    Examples
    --------

    >>> import unipy_nlp.data_collector as udcl
    >>> import unipy_nlp.preprocessing as uprc
    >>> import unipy_nlp.analyze.topic_modeling as utpm
    >>> from pprint import pprint
    >>> prep = uprc.Preprocessor()
    >>> prep.read_json('./data/_tmp_dump/prep/rawdata_collected.json')
    >>> sentence_for_pos_list = [
    ...     "무궁화 꽃이 피었습니다."
    ...     "우리는 민족중흥의 역사적 사명을 띠고 이 땅에 태어났다.",
    ... ]
    >>> tokenized = prep.pos_tag(
    ...     input_text=sentence_for_pos_list,
    ...     tag_type=[
    ...         '체언 접두사', '명사', '한자', '외국어',
    ...         '수사', '구분자',
    ...         '동사',
    ...         '부정 지정사', '긍정 지정사',
    ...     ]
    ... )
    >>> print(tokenized)
    [['무궁화'], ['우리', '민족중흥', '역사', '사명']]
    >>> tpm = utpm.TopicModeler(sentence_list, tokenized)
    >>> tpm.train_lda(
    ...     num_topic=5,
    ...     workers_n=8,
    ...     random_seed=1,
    ... )
    >>> tpm.save_lda(savepath='data/_tmp_dump/topic_modeling', affix='lda')
    >>> tpm.load_lda('data/_tmp_dump/topic_modeling')
    >>> tpm.pick_best_lda_topics(
    ...     num_topic_list=[5, 7, 10],
    ...     workers_n=8,
    ...     random_seed=1,
    ... )
    >>> tpm.visualize_lda_to_html(
    ...     7,
    ...     top_n=10,
    ...     r_normalized=False,
    ...     relevence_lambda_val=.6,
    ...     workers_n=8,
    ...     random_seed=1,
    ...     savepath='data/_tmp_dump/topic_modeling',
    ...     filename_affix='lda',
    ...     # save_type='html',  # {'html', 'json'}
    ...     save_relevent_terms_ok=True,
    ...     save_html_ok=True,
    ...     display_ok=False,
    ... )

    >>> sentence_labeled = tpm.estimate_topics_by_documents(
    ...     7,
    ...     # sentence_list=tokenized,
    ...     random_seed=1,
    ...     save_ok=True,
    ...     savepath='data/_tmp_dump/topic_modeling',
    ...     filename_affix='lda',
    ... )
    >>> sentence_repr = tpm.get_representitive_documents(
    ...     7,
    ...     len_range=(10, 30),
    ...     top_n=10,
    ...     save_ok=True,
    ...     savepath='data/_tmp_dump/topic_modeling',
    ...     filename_affix='lda',
    ... )

    """
    def __init__(
            self,
            sentence_list,
            tokenized_sentence_list,
            ):

        self.trained = False  # {'trained', 'loaded'}
        self.selected_model = None
        self.lda_model_list = None

        self.sentences = sentence_list
        self.tokenized = tokenized_sentence_list
        self.corpora_dict = gensim.corpora.Dictionary(
            tokenized_sentence_list
        )
        self.corpora_dict.filter_extremes(
            no_below=30, no_above=.5, keep_n=100000,
        )

        self.bow_corpus_idx = [
            self.corpora_dict.doc2idx(doc)
            for doc in tokenized_sentence_list
        ]
        self.bow_corpus_doc = [
            self.corpora_dict.doc2bow(doc)
            for doc in tokenized_sentence_list
        ]

    def train_lda(
            self,
            num_topic=5,
            lda_type='default',
            workers_n=2,
            random_seed=1,
            ):
        """
        Train a single LDA Topic Model.

        Parameters
        ----------

        num_topics: int (default: 5)
            A number of topics.

        lda_type: str (default: `'default'`, `{'default', 'hdp', 'mallet'}`)
            A type of LDA model.
            Use `'default'` for now. Other options are working in progress.

        workers_n: int (default: 2)
            A number of CPU core to train.

        random_seed: int (default: 1)
            A random seed int.

        Example
        -------

        >>> import unipy_nlp.data_collector as udcl
        >>> import unipy_nlp.preprocessing as uprc
        >>> import unipy_nlp.analyze.topic_modeling as utpm
        >>> from pprint import pprint
        >>> prep = uprc.Preprocessor()
        >>> prep.read_json('./data/_tmp_dump/prep/rawdata_collected.json')
        >>> sentence_for_pos_list = [
        ...     "무궁화 꽃이 피었습니다."
        ...     "우리는 민족중흥의 역사적 사명을 띠고 이 땅에 태어났다.",
        ... ]
        >>> tokenized = prep.pos_tag(
        ...     input_text=sentence_for_pos_list,
        ...     tag_type=[
        ...         '체언 접두사', '명사', '한자', '외국어',
        ...         '수사', '구분자',
        ...         '동사',
        ...         '부정 지정사', '긍정 지정사',
        ...     ]
        ... )
        >>> tpm = utpm.TopicModeler(sentence_list, tokenized)
        >>> tpm.train_lda(
        ...     num_topic=5,
        ...     workers_n=8,
        ...     random_seed=1,
        ... )

        """
        self.pick_best_lda_topics(
            num_topic_list=[num_topic],
            lda_type=lda_type,
            workers_n=workers_n,
            random_seed=random_seed,
        )
        self.trained = True

    def pick_best_lda_topics(
            self,
            num_topic_list=[5, 7, 10, 12, 15, 17, 20],
            lda_type='default',
            workers_n=2,
            random_seed=1,
            ):
        """
        Train multiple LDA Topic Models by given topic numbers.

        Parameters
        ----------

        num_topic_list: list (default: `[5, 7, 10, 12, 15, 17, 20]`)
            A list of topic numbers.

        lda_type: str (default: `'default'`, `{'default', 'hdp', 'mallet'}`)
            A type of LDA model.
            Use `'default'` for now. Other options are working in progress.

        workers_n: int (default: 2)
            A number of CPU core to train.

        random_seed: int (default: 1)
            A random seed int.

        Example
        -------

        >>> import unipy_nlp.analyze.topic_modeling as utpm
        >>> tpm = utpm.TopicModeler(sentence_list, tokenized)
        >>> tpm.pick_best_lda_topics(
        ...     num_topic=5,
        ...     workers_n=8,
        ...     random_seed=1,
        ... )

        """

        (best_lda_model, model_coh_list,
         model_dict, coh_list) = pick_best_topics(
            dictionary=self.corpora_dict,
            corpus=self.bow_corpus_doc,
            texts=self.tokenized,
            num_topic_list=num_topic_list,
            lda_type=lda_type,
            workers_n=workers_n,
            random_seed=random_seed,
        )

        self.best_lda_model = best_lda_model
        self.lda_model_list = model_coh_list
        self.lda_model_dict = model_dict

        self.trained = True

    def save_lda(self, savepath='./', affix='lda'):
        """
        Save trained lda model(s).

        Parameters
        ----------

        savepath: str (default: `'./'`)
            A dirpath to save.

        affix: str (default: `'lda'`)
            An affix of filename. Its ext will be `.ldamodel`.

        Example
        -------

        >>> import unipy_nlp.analyze.topic_modeling as utpm
        >>> tpm = utpm.TopicModeler(sentence_list, tokenized)
        >>> tpm.pick_best_lda_topics(
        ...     num_topic=5,
        ...     workers_n=8,
        ...     random_seed=1,
        ... )
        >>> tpm.save_lda(savepath='data/_tmp_dump/topic_modeling', affix='lda')

        """
        os.makedirs(savepath, exist_ok=True)

        corpora_filename = os.path.join(
            savepath,
            f'{affix}.cdict',
        )
        self.corpora_dict.save_as_text(
            corpora_filename,
            sort_by_word=False,
        )

        for _topic_num, _inner_dict in self.lda_model_dict.items():

            _model = _inner_dict['model']
            _coh_val = _inner_dict['coherence']

            model_name_str = '_'.join([
                f'{affix}',
                f'topics-{_topic_num}',
                f'coh-{_coh_val}.ldamodel',
            ])
            print(f'{savepath:2}: {model_name_str}')

            _filename = os.path.join(
                savepath,
                model_name_str,
            )
            _model.save(_filename)

    def load_lda(self, filepath):
        """
        Load trained lda model(s).

        Parameters
        ----------

        filepath: str
            A dirpath to load. It contains `.ldamodel`.

        Example
        -------

        >>> import unipy_nlp.analyze.topic_modeling as utpm
        >>> tpm = utpm.TopicModeler(sentence_list, tokenized)
        >>> tpm.load_lda('data/_tmp_dump/topic_modeling')

        """
        if os.path.isfile(filepath):

            self.lda_model_list = []
            self.lda_model_dict = {}

            path_str = '/'.join(filepath.split('/')[:-1])
            model_name_str = filepath.split('/')[-1]
            affix, _topic_num, _coh_val = re.findall(
                r'^(.+)_topics\-(\d+)_coh\-([-+]?\d*\.\d+|\d+)\.ldamodel',
                model_name_str,
            )[0]

            _topic_num, _coh_val = int(_topic_num), int(_coh_val)
            _model = gensim.models.LdaMulticore.load(filepath)
            self.lda_model_list = [(_topic_num, _model, _coh_val)]
            self.lda_model_dict.__setitem__(
                _topic_num,
                {'model': _model, 'coherence': _coh_val},
            )

            corpora_filename = os.path.join(
                path_str,
                f'{affix}.cdict',
            )
            self.corpora_dict = gensim.corpora.Dictionary.load_from_text(
                corpora_filename,
            )
            self.best_lda_model = self.lda_model_dict[_topic_num]
            print(f'Model loaded: topics={_topic_num}, coh={_coh_val}')

        elif os.path.isdir(filepath):

            self.lda_model_list = []
            self.lda_model_dict = {}

            path_str = filepath
            filepath_list = glob(os.path.join(filepath, '*.ldamodel'))

            for filename in filepath_list:
                model_name_str = filename.split('/')[-1]
                affix, _topic_num, _coh_val = re.findall(
                    r'^(.+)_topics\-(\d+)_coh\-([-+]?\d*\.\d+|\d+).ldamodel',
                    model_name_str,
                )[0]
                _topic_num, _coh_val = int(_topic_num), int(_coh_val)
                _model = gensim.models.LdaMulticore.load(filename)
                self.lda_model_list += [(_topic_num, _model, _coh_val)]
                self.lda_model_dict.__setitem__(
                    _topic_num,
                    {'model': _model, 'coherence': _coh_val},
                )
                print(f'Model loaded: topics={_topic_num}, coh={_coh_val}')

            corpora_filename = os.path.join(
                path_str,
                f'{affix}.cdict',
            )
            self.corpora_dict = gensim.corpora.Dictionary.load_from_text(
                corpora_filename,
            )
            best_topic_num, best_model, best_coh_val = max(
                self.lda_model_list,
                key=lambda x: x[-1],
            )
            self.best_lda_model = self.lda_model_dict[int(best_topic_num)]

        self.trained = True

    def _get_terminfo_table(self, *args, **kwargs):
        return get_terminfo_table(*args, **kwargs)

    def visualize_lda_to_html(
            self,
            target_topic_num,
            top_n=10,
            r_normalized=False,
            relevence_lambda_val=.6,
            workers_n=2,
            random_seed=1,
            savepath='./',
            filename_affix='lda',
            # save_type='html',  # {'html', 'json'}
            save_relevent_terms_ok=True,
            save_html_ok=True,
            display_ok=False,
            ):
        """
        Run `pyLDAvis.prepare` & get adjusted scores(use saliency & relevence) of terms by each topic.

        Parameters
        ----------

        target_topic_num: int
            A topic number of LDA model to visualize.

        top_n: int (default: `10`)
            A number of the most relevent terms in a topic.

        r_normalized: bool (default: `False`)
            Use normalized probabilities when it is `True`. (not recommended in most cases.)

        relevence_lambda_val: float (defautl: `.6`).
            A lambda value(ratio) to calculate relevence.

        workers_n: int (default: `2`)
            A number of CPU cores to calculate(`pyLDAvis.prepare`)

        random_seed: int (default: `1`)
            A random seed number.

        savepath: str (default: `'./'`)
            A dirpath to save `pyLDAvis` or other `pandas.DataFrame`s.

        filename_affix: str (default: `'lda'`)
            An affix of filename to save `pyLDAvis` html or json.

        save_relevent_terms_ok: bool (default: `True`)
            An option to save `pandas.DataFrame` of `top_relevent_terms`.

        save_html_ok: bool (default: `True`)
            An option to save html.

        display_ok: bool (default: `False`)
            Call `pyLDAvis.display` when it is `True`.

        References
        ----------

        Saliency:
            `Chuang, J., 2012. Termite: Visualization techniques for assessing textual topic models`

        Relevence:
            `Sievert, C., 2014. LDAvis: A method for visualizing and interpreting topics`

        Example
        -------

        >>> import unipy_nlp.analyze.topic_modeling as utpm
        >>> tpm = utpm.TopicModeler(sentence_list, tokenized)
        >>> tpm.pick_best_lda_topics(
        ...     num_topic=5,
        ...     workers_n=8,
        ...     random_seed=1,
        ... )
        >>> tpm.visualize_lda_to_html(
        ...     7,
        ...     top_n=10,
        ...     r_normalized=False,
        ...     relevence_lambda_val=.6,
        ...     workers_n=8,
        ...     random_seed=1,
        ...     savepath='data/_tmp_dump/topic_modeling',
        ...     filename_affix='lda',
        ...     save_relevent_terms_ok=True,
        ...     save_html_ok=True,
        ...     display_ok=False,
        ... )

        """
        if target_topic_num in self.lda_model_dict.keys():
            self.selected_topic_num = target_topic_num
            self.selected_model = (
                self.lda_model_dict[target_topic_num]['model']
            )
        else:
            raise KeyError("Model doesn't exist. Select a proper number.")

        (vis_prepared,
         total_terms_df,
         top_relevant_terms_df,
         r_adj_score_df,
         bow_score_list) = self._get_terminfo_table(
            self.selected_model,
            corpus=self.bow_corpus_doc,
            dictionary=self.corpora_dict,
            doc_topic_dists=None,
            use_gensim_prepared=True,
            top_n=top_n,
            r_normalized=r_normalized,
            relevence_lambda_val=relevence_lambda_val,
            workers_n=workers_n,
            random_seed=random_seed,
        )

        self.vis_prepared = vis_prepared
        self.total_terms_df = total_terms_df
        self.top_relevant_terms_df = top_relevant_terms_df
        self.r_adj_score_df = r_adj_score_df
        self.bow_score_list = bow_score_list

        if save_html_ok:
            os.makedirs(savepath, exist_ok=True)
            ldavis_filename_html_str = os.path.join(
                savepath,
                f'{filename_affix}_topics-{target_topic_num}.html',
            )
            pyLDAvis.save_html(
                self.vis_prepared,
                ldavis_filename_html_str,
            )
            print(f"LDAVIS HTML Saved: '{ldavis_filename_html_str}'")

        if save_relevent_terms_ok:
            os.makedirs(savepath, exist_ok=True)
            ldavis_filename_rdf_str = os.path.join(
                savepath,
                '_'.join([
                    f'{filename_affix}',
                    f'topics-{target_topic_num}',
                    f'top{top_n}_relevent_terms_df.csv',
                ]),
            )
            self.top_relevant_terms_df.to_csv(
                ldavis_filename_rdf_str,
                index=True,
                header=True,
                encoding='utf-8',
            )
            print(f"LDAVIS DF Saved: '{ldavis_filename_rdf_str}'")

        if display_ok:
            pyLDAvis.display(self.vis_prepared, local=False)

    def estimate_topics_by_documents(
            self,
            target_topic_num,
            # sentence_list=None,
            random_seed=1,
            save_ok=True,
            savepath='./',
            filename_affix='lda',
            ):
        """
        Get dominant topics & its contribution scores from each documents.

        Parameters
        ----------

        target_topic_num: int
            A topic number of LDA model to use.

        random_seed: int (default: `1`)
            A random seed number.

        save_ok: bool (default: `True`)
            Save return `pandas.DataFrame`.

        savepath: str (default: `'./'`)
            A dirpath to save the topic-labeled sentences.

        filename_affix: str (default: `'lda'`)
            An affix of filename to save the topic-labeled sentences.

        Return
        ------

        dominant_topic_estimation_df: `pandas.DataFrame`
            Topic-labeled given(trained) sentences.

        topic_freq_df: `pandas.DataFrame`
            A rank table of topics by frequency.

        Example
        -------

        >>> import unipy_nlp.analyze.topic_modeling as utpm
        >>> tpm = utpm.TopicModeler(sentence_list, tokenized)
        >>> tpm.pick_best_lda_topics(
        ...     num_topic=5,
        ...     workers_n=8,
        ...     random_seed=1,
        ... )
        >>> sentence_labeled = tpm.estimate_topics_by_documents(
        ...     7,
        ...     random_seed=1,
        ...     save_ok=True,
        ...     savepath='data/_tmp_dump/topic_modeling',
        ...     filename_affix='lda',
        ... )

        """

        if target_topic_num != self.selected_topic_num:
            raise ValueError(
                'You should run `visualize_lda_to_html` first.'
            )

        lda_model = self.selected_model
        corpus = self.bow_corpus_doc
        docs = self.sentences
        bow_r_score_list = self.bow_score_list
        top_r_terms_df = self.r_adj_score_df

        res_df = pd.DataFrame(
            columns=[
                'dominant_topic',
                'contribution',
                'topic_keywords',
                'documents',
                'lda_prob',
            ]
        )

        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)

        r_colname = top_r_terms_df.columns.drop(['Category', 'Term'])[0]

        if top_r_terms_df is not None:
            top_sorted_words = groupby_top_n(
                top_r_terms_df.reset_index(),
                group_by=['category_num'],
                order_by=[r_colname],
                ascending=False,
                n=10,
            )
            top_word_str = (
                top_sorted_words
                .groupby(level=0)
                ['Term']
                .apply(lambda x: ', '.join(x.tolist()))
            )

        def normalize_prob(prob_row):
            total_prob = sum([prob for topic, prob in prob_row])
            normed_prob_row = [
                (topic, prob / total_prob) for topic, prob in prob_row
            ]
            return normed_prob_row

        def sort_prob(prob_row, bow_r_score_list=bow_r_score_list):
            return sorted(prob_row, key=lambda x: x[1], reverse=True)

        def get_dominant_prob(prob_row):
            return pd.Series(prob_row[0])

        def get_topic_keywords(
                dom_topic_num,
                lda_model=lda_model,
                top_r_terms_df=top_r_terms_df,
                ):
            if top_r_terms_df is not None:
                return top_word_str[int(dom_topic_num)]
            else:
                return ', '.join(
                    np.array(lda_model.show_topic(int(dom_topic_num)))[:, 0]
                )

        res_df['documents'] = docs

        if bow_r_score_list is not None:
            # bow_score_series = (
            #     pd.Series(bow_r_score_list).apply(normalize_prob)
            # )
            bow_score_series = pd.Series(bow_r_score_list)
        else:
            bow_score_series = pd.Series(lda_model[corpus])

        res_df['lda_prob'] = bow_score_series.apply(sort_prob)
        res_df[['dominant_topic', 'contribution']] = (
            res_df['lda_prob']
            .apply(get_dominant_prob)
        )
        res_df['dominant_topic'] = (
            res_df['dominant_topic']
            .astype(int)
            .astype('category')
        )
        res_df['topic_keywords'] = (
            res_df['dominant_topic']
            .apply(get_topic_keywords)
        )
        res_df['lda_prob'] = res_df['lda_prob'].apply(dict)
        res_df.index.name = 'doc_num'
        res_df.reset_index(inplace=True)

        self.dominant_topic_estimation_df = res_df
        self.topic_freq_df = (
            self.dominant_topic_estimation_df
            .groupby('dominant_topic')
            ['doc_num']
            .count()
            .reset_index()
            .sort_values('doc_num', ascending=False)
        )

        if save_ok:
            os.makedirs(savepath, exist_ok=True)
            filename_str = os.path.join(
                savepath,
                '_'.join([
                    f'{filename_affix}',
                    f'topics-{target_topic_num}',
                    f'dominant_topic_estimation_df.csv',
                ]),
            )
            res_df.to_csv(
                filename_str,
                index=False,
                header=True,
                encoding='utf-8',
            )
        return self.dominant_topic_estimation_df, self.topic_freq_df

    def load_estimated(
            self,
            target_topic_num,
            savepath='./',
            filename_affix='lda'
            ):
        """
        Load the result of `self.estimate_topics_by_documents`.

        Parameters
        ----------

        target_topic_num: int
            A topic number of LDA model to use.

        savepath: str (default: `'./'`)
            A dirpath to load the topic-labeled sentences.

        filename_affix: str (default: `'lda'`)
            An affix of filename to load the topic-labeled sentences.

        Return
        ------

        dominant_topic_estimation_df: `pandas.DataFrame`
            Topic-labeled given(trained) sentences.

        topic_freq_df: `pandas.DataFrame`
            A rank table of topics by frequency.

        Example
        -------

        >>> import unipy_nlp.analyze.topic_modeling as utpm
        >>> tpm = utpm.TopicModeler(sentence_list, tokenized)
        >>> tpm.pick_best_lda_topics(
        ...     num_topic=5,
        ...     workers_n=8,
        ...     random_seed=1,
        ... )
        >>> sentence_labeled = tpm.estimate_topics_by_documents(
        ...     7,
        ...     random_seed=1,
        ...     save_ok=True,
        ...     savepath='data/_tmp_dump/topic_modeling',
        ...     filename_affix='lda',
        ... )
        >>> sentence_labeled, topic_freq = tpm.load_estimated(
        ...     target_topic_num=7,
        ...     savepath='data/_tmp_dump/topic_modeling',
        ...     filename_affix='lda',
        ... )

        """
        filename_str = os.path.join(
            savepath,
            '_'.join([
                f'{filename_affix}',
                f'topics-{target_topic_num}',
                f'dominant_topic_estimation_df.csv',
            ]),
        )
        res_df = pd.read_csv(
            filename_str,
            encoding='utf-8',
        )
        self.dominant_topic_estimation_df = res_df
        self.topic_freq_df = (
            self.dominant_topic_estimation_df
            .groupby('dominant_topic')
            ['doc_num']
            .count()
            .reset_index()
            .sort_values('doc_num', ascending=False)
        )

        return self.dominant_topic_estimation_df, self.topic_freq_df

    def get_best_n_terms(self):
        pass

    def get_representitive_candidates(
            self,
            len_range=(10, 30),
            ):
        """
        Get representitive candidates by length. It is for to use `unipy_nlp.network_plot`.

        Parameters
        ----------

        len_range: `list` or `tuple` (default: `(10, 30)`)
            A candidate threshold by length.

        Return
        ------

        repr_sentences: `list`
            A list of sentences.

        repr_bow_corpus_doc: `list`
            A nested list, which contains converted documents into a list of token words.

        repr_bow_corpus_idx: `list`
            A nested list, which contains converted documents into a list of token indices..

        Example
        -------

        >>> import unipy_nlp.analyze.topic_modeling as utpm
        >>> tpm = utpm.TopicModeler(sentence_list, tokenized)
        >>> tpm.pick_best_lda_topics(
        ...     num_topic=5,
        ...     workers_n=8,
        ...     random_seed=1,
        ... )
        >>> sentence_labeled = tpm.estimate_topics_by_documents(
        ...     7,
        ...     random_seed=1,
        ...     save_ok=True,
        ...     savepath='data/_tmp_dump/topic_modeling',
        ...     filename_affix='lda',
        ... )
        >>> (repr_sentenced,
        >>>  repr_bow_corpus_doc,
        >>>  repr_bow_corpus_idx) = tpm.get_representitive_candidates(
        ...     len_range=(12, 30),
        ... )

        """
        len_min, len_max = len_range
        bool_mask = mask_to_filter_document_by_len = list(
            map(lambda x: len_min <= len(x) < len_max, self.sentences)
        )

        repr_bow_corpus_idx = list(
            it.compress(self.bow_corpus_idx, bool_mask)
        )
        repr_bow_corpus_doc = list(
            it.compress(self.bow_corpus_doc, bool_mask)
        )
        repr_sentences = list(
            it.compress(self.sentences, bool_mask)
        )
        return repr_sentences, repr_bow_corpus_doc, repr_bow_corpus_idx

    def _clip_document_len(
            self,
            topic_kwd_df,
            len_range=(10, 30),
            ):

        len_min, len_max = len_range
        len_series = topic_kwd_df['documents'].apply(len)
        # mask = np.where((len_series >= 100) & (len_series < 300))
        res = topic_kwd_df.loc[
            (len_series >= len_min) & (len_series < len_max),
            :
        ]
        return res

    def get_representitive_documents(
            self,
            target_topic_num,
            len_range=(10, 30),
            top_n=10,
            save_ok=True,
            savepath='./',
            filename_affix='lda',
            ):
        """
        List-up the most representitive documents by topic.

        Parameters
        ----------

        target_topic_num: int
            A topic number of LDA model to use.

        len_range: `list` or `tuple` (default: `(10, 30)`)
            A candidate threshold by length.

        top_n: int (default: `10`)
            A document number to list-up, by topic.

        save_ok: bool (default: `True`)
            An option to save.

        savepath: str (default: `'./'`)
            A dirpath to load the topic-labeled sentences.

        filename_affix: str (default: `'lda'`)
            An affix of filename to load the topic-labeled sentences.

        Return
        ------

        reordered: `pandas.DataFrame`
            Representitive documents, group by topic, ordery by its rank.

        Example
        -------

        >>> import unipy_nlp.analyze.topic_modeling as utpm
        >>> tpm = utpm.TopicModeler(sentence_list, tokenized)
        >>> tpm.pick_best_lda_topics(
        ...     num_topic=5,
        ...     workers_n=8,
        ...     random_seed=1,
        ... )
        >>> sentence_labeled = tpm.estimate_topics_by_documents(
        ...     7,
        ...     random_seed=1,
        ...     save_ok=True,
        ...     savepath='data/_tmp_dump/topic_modeling',
        ...     filename_affix='lda',
        ... )
        >>> sentence_repr = tpm.get_representitive_documents(
        ...     7,
        ...     len_range=(10, 30),
        ...     top_n=10,
        ...     save_ok=True,
        ...     savepath='data/_tmp_dump/topic_modeling',
        ...     filename_affix='lda',
        ... )

        """
        if target_topic_num != self.selected_topic_num:
            raise ValueError(
                'You should run `visualize_lda_to_html` first.'
            )

        if self.dominant_topic_estimation_df is None:
            raise ValueError(
                "You should run `estimate_topics_by_documents` first."
            )

        target_topic_num = len(self.topic_freq_df)
        repr_candidates_df = self._clip_document_len(
            self.dominant_topic_estimation_df,
            len_range=len_range,
        )

        repr_docs_df = groupby_top_n(
            repr_candidates_df,
            group_by=['dominant_topic'],
            order_by=['contribution'],
            ascending=False,
            n=top_n,
        )

        grouped_dict = dict(list(repr_docs_df.groupby(level=0)))
        reordered = pd.concat(
            [
                grouped_dict[i]
                for i in self.topic_freq_df['dominant_topic']
            ]
        )
        self.representitive_docs = reordered

        if save_ok:
            os.makedirs(savepath, exist_ok=True)
            filename_str = os.path.join(
                savepath,
                '_'.join([
                    f'{filename_affix}',
                    f'topics-{target_topic_num}',
                    f'top{top_n}_repr_docs_df.csv',
                ]),
            )
            reordered.to_csv(
                filename_str,
                index=True,
                header=True,
                encoding='utf-8',
            )

        return reordered

    def load_representitive_documents(
            self,
            target_topic_num,
            top_n=10,
            savepath='./',
            filename_affix='lda',
            ):
        """
        Load the result of `self.get_representitive_documents`.

        Parameters
        ----------

        target_topic_num: int
            A topic number of LDA model to use.

        top_n: int (default: `10`)
            A document number to list-up, by topic.
            The upper bound depends on how many documents saved.

        savepath: str (default: `'./'`)
            A dirpath to load the topic-labeled sentences.

        filename_affix: str (default: `'lda'`)
            An affix of filename to load the topic-labeled sentences.

        Return
        ------

        dominant_topic_estimation_df: `pandas.DataFrame`
            Topic-labeled given(trained) sentences.

        topic_freq_df: `pandas.DataFrame`
            A rank table of topics by frequency.

        Example
        -------

        >>> import unipy_nlp.analyze.topic_modeling as utpm
        >>> tpm = utpm.TopicModeler(sentence_list, tokenized)
        >>> tpm.pick_best_lda_topics(
        ...     num_topic=5,
        ...     workers_n=8,
        ...     random_seed=1,
        ... )
        >>> sentence_labeled = tpm.estimate_topics_by_documents(
        ...     7,
        ...     random_seed=1,
        ...     save_ok=True,
        ...     savepath='data/_tmp_dump/topic_modeling',
        ...     filename_affix='lda',
        ... )
        >>> sentence_labeled, topic_freq = tpm.load_estimated(
        ...     target_topic_num=7,
        ...     savepath='data/_tmp_dump/topic_modeling',
        ...     filename_affix='lda',
        ... )

        """
        filename_str = os.path.join(
            savepath,
            '_'.join([
                f'{filename_affix}',
                f'topics-{target_topic_num}',
                f'top{top_n}_repr_docs_df.csv',
            ]),
        )
        self.representitive_docs = pd.read_csv(
            filename_str,
            encoding='utf-8',
        )

        return self.representitive_docs