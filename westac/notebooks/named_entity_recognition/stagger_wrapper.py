#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# This source code is based on ...
#
# Copyright (C) ....
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html
#
r"""Wrapper for `The Stockholm NER tagger(Stagger) NER tagger <https://www.ling.su.se/english/nlp/tools/stagger>`_

Notes
-----

Installation
------------

Examples
--------

"""
import logging
import os
import random
import warnings
import tempfile
import xml.etree.ElementTree as et
import zipfile
from itertools import chain

import numpy
import scipy
import io

from smart_open import smart_open

from gensim import utils, matutils
from gensim.models import basemodel
from gensim.models.ldamodel import LdaModel
from gensim.utils import check_output, revdict

import datetime

logger = logging.getLogger(__name__)

class STTMTopicModel(utils.SaveLoad, basemodel.BaseTopicModel):
    """Wrapper for The Stockholm NER tagger(Stagger) NER tagger <https://www.ling.su.se/english/nlp/tools/stagger>.
    """

    def __init__(self, stagger_jar_path, stagger_model, corpus, prefix, name):
        """
        Parameters
        ----------
        stagger_jar_path : str
            Path to the Stagger jar file.
        stagger_model : str
            Path to the Stagger model file.
        corpus : iterable of iterable of (str, str), optional
            Collection of texts.
        prefix : str, optional
            Result files prefix.
        name : str, optional
            Name of execution, timestamp if None.

        """
        self.stagger_jar_path = stagger_jar_path
        self.stagger_model = stagger_model.upper()

        self.prefix = prefix or ""
        self.name = name or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        self.stagger_home = os.environ.get('STAGGER_HOME', None)
        self.java_home = os.environ.get('JAVA_HOME', None)

        assert self.stagger_home is not None, "You must set STAGER_HOME"
        assert self.java_home_home is not None, "You must set JAVA_HOME"

        if corpus is not None:
            self.stagit(corpus)

    def stagit(self, corpus):
        """Tag corpus.

        Parameters
        ----------
        corpus : iterable of iterable of (str, str)

        """

        # java -Xmx4G -jar stagger.jar -modelfile models/swedish.bin -tag test.txt > test.conll

        self.convert_input(corpus)

        self.java_opts = '-Xmx1G'

        cmd = 'java {} -jar {} -modelfile {} -lang sv -tag {} '.format

        cmd = cmd.format(
            self.java_opts,
            self.stagger_jar_path,
            self.stagger_model,
            self.fcorpustxt(),
            self.num_topics,
            self.alpha[0],
            self.beta,
            self.iterations,
            self.twords,
            self.name,
            self.sstep
        )

        if self.vectors is not None:
            cmd += ' -vectors {}'.format(self.vectors)

        logger.info("training STTM model with %s", cmd)
        check_output(args=cmd, shell=True)
        self.word_topics = self.load_word_topics()
        self.wordtopics = self.word_topics

    def ftopickeys(self):
        """Get path to topic keys text file.

        Returns
        -------
        str
            Path to topic keys text file.

        """
        return self.prefix + self.name + '.phi'

    def fdoctopics(self):
        """Get path to document topic text file.

        Returns
        -------
        str
            Path to document topic text file.

        """
        return self.prefix + self.name + '.theta'

    def fcorpustxt(self):
        """Get path to corpus text file.

        Returns
        -------
        str
            Path to corpus text file.

        """
        return self.prefix + self.name + '.corpus'

    def fvocabulary(self):
        """Get path to vocabulary text file.

        Returns
        -------
        str
            Path to vocabulary text file.

        """
        return self.prefix + self.name + '.vocabulary'

    def ftopwords(self):
        """Get path to topwords text file.

        Returns
        -------
        str
            Path to topwords text file.

        """
        return self.prefix + self.name + '.topWords'

    def fwordweights(self):
        """Get path to word weight file.

        Returns
        -------
        str
            Path to word weight file.

        """
        return self.prefix + 'wordweights.txt'

    def corpus2sttm(self, corpus, file_like):
        """Convert `corpus` to STTM format and write it to `file_like` descriptor.

        Format ::

            whitespace delimited utf8-encoded tokens[NEWLINE]

        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Collection of texts in BoW format.
        file_like : file-like object
            Opened file.

        """
        for _, doc in enumerate(corpus):
            if self.id2word:
                tokens = chain.from_iterable([self.id2word[tokenid]] * int(cnt) for tokenid, cnt in doc)
            else:
                tokens = chain.from_iterable([str(tokenid)] * int(cnt) for tokenid, cnt in doc)
            file_like.write(utils.to_utf8("{}\n".format(' '.join(tokens))))

    def convert_input(self, corpus):
        """Convert corpus to SSTM format and save it to a temporary text file.

        Parameters
        ----------
        corpus : iterable of iterable of (int, int)
            Collection of texts in BoW format.
        """
        logger.info("serializing temporary corpus to %s", self.fcorpustxt())
        with smart_open(self.fcorpustxt(), 'wb') as fout:
            self.corpus2sttm(corpus, fout)


    def __getitem__(self, bow, iterations=100):
        raise ValueError('infer-topics for new, unseen documents not implemented (or even possible?)')

    def load_word_topics(self):
        """Load words X topics matrix from :meth:`gensim.models.wrappers.ldamallet.LdaMallet.fstate` file.

        Returns
        -------
        numpy.ndarray
            Matrix words X topics.

        """
        logger.info("loading assigned topics from %s", self.ftopickeys())

        #with open(self.ftopickeys(), 'r') as f:
        #    text = f.read().replace(' \n', '\n')
        #word_topics = np.loadtxt(io.StringIO(text), delimiter=' ', dtype=numpy.float64)

        word_topics = numpy.loadtxt(self.ftopickeys(), delimiter=' ', usecols=range(0,self.num_terms), dtype=numpy.float64)

        assert word_topics.shape == (self.num_topics, self.num_terms)

        return word_topics

    def load_document_topics(self):
        """Load document topics from :meth:`gensim.models.wrappers.ldamallet.LdaMallet.fdoctopics` file.
        Shortcut for :meth:`gensim.models.wrappers.ldamallet.LdaMallet.read_doctopics`.

        Returns
        -------
        iterator of list of (int, float)
            Sequence of LDA vectors for documents.

        """
        return self.read_doctopics(self.fdoctopics())

    def get_topics(self):
        """Get topics X words matrix.

        Returns
        -------
        numpy.ndarray
            Topics X words matrix, shape `num_topics` x `vocabulary_size`.

        """
        topics = self.word_topics
        return topics / topics.sum(axis=1)[:, None]

    def show_topics(self, num_topics=10, num_words=10, log=False, formatted=True):
        """Get the `num_words` most probable words for `num_topics` number of topics.

        Parameters
        ----------
        num_topics : int, optional
            Number of topics to return, set `-1` to get all topics.
        num_words : int, optional
            Number of words.
        log : bool, optional
            If True - write topic with logging too, used for debug proposes.
        formatted : bool, optional
            If `True` - return the topics as a list of strings, otherwise as lists of (weight, word) pairs.

        Returns
        -------
        list of str
            Topics as a list of strings (if formatted=True) **OR**
        list of (float, str)
            Topics as list of (weight, word) pairs (if formatted=False)

        """
        if num_topics < 0 or num_topics >= self.num_topics:
            num_topics = self.num_topics
            chosen_topics = range(num_topics)
        else:
            num_topics = min(num_topics, self.num_topics)
            # add a little random jitter, to randomize results around the same alpha
            sort_alpha = self.alpha + 0.0001 * numpy.random.rand(len(self.alpha))
            sorted_topics = list(matutils.argsort(sort_alpha))
            chosen_topics = sorted_topics[: num_topics // 2] + sorted_topics[-num_topics // 2:]
        shown = []
        for i in chosen_topics:
            if formatted:
                topic = self.print_topic(i, topn=num_words)
            else:
                topic = self.show_topic(i, topn=num_words)
            shown.append((i, topic))
            if log:
                logger.info("topic #%i (%.3f): %s", i, self.alpha[i], topic)
        return shown

    def show_topic(self, topicid, topn=10):
        """Get `num_words` most probable words for the given `topicid`.

        Parameters
        ----------
        topicid : int
            Id of topic.
        topn : int, optional
            Top number of topics that you'll receive.

        Returns
        -------
        list of (str, float)
            Sequence of probable words, as a list of `(word, word_probability)` for `topicid` topic.

        """

        if self.word_topics is None:
            logger.warning("Run train or load_word_topics before showing topics.")

        topic = self.word_topics[topicid]
        topic = topic / topic.sum()  # normalize to probability dist
        bestn = matutils.argsort(topic, topn, reverse=True)
        beststr = [(self.id2word[idx], topic[idx]) for idx in bestn]
        return beststr

    def read_doctopics(self, fname, eps=1e-6, renorm=True):
        """Get document topic vectors from MALLET's "doc-topics" format, as sparse gensim vectors.

        Parameters
        ----------
        fname : str
            Path to input file with document topics.
        eps : float, optional
            Threshold for probabilities.
        renorm : bool, optional
            If True - explicitly re-normalize distribution.

        Raises
        ------
        RuntimeError
            If any line in invalid format.

        Yields
        ------
        list of (int, float)
            LDA vectors for document.

        """
        doc_topics = numpy.loadtxt(fname, delimiter=' ', usecols=range(0,self.num_topics), dtype=numpy.float64)
        doc_topics[doc_topics < eps] = 0
        m = scipy.sparse.coo_matrix(doc_topics)

        for i in range(0, m.shape[0]):
            row = m.getrow(i)
            values = list(zip(row.indices, row.data))
            if renorm:
                total_weight = sum([w for _, w in values])
                if total_weight != 0:
                    values = [ (i, w / total_weight) for (i, w) in values]
            yield values