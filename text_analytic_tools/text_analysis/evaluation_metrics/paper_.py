
# extract from https://github.com/cscorley/mud2014-modeling-changeset-topics

import csv
import sys
import os.path
import random
import math
from collections import namedtuple

import numpy
from gensim.corpora import MalletCorpus, Dictionary
from gensim.models import LdaModel

import logging

logger = logging.getLogger('test')

def error(msg, errorno=1):
    logger.error(msg)
    sys.exit(errorno)

def kullback_leibler_divergence(q_dist, p_dist, filter_by=0.001):
    assert len(q_dist) == len(p_dist)
    z = zip(q_dist, p_dist)
    divergence = 0.0
    for q, p in z:
        if q < filter_by and p < filter_by:
            continue

        if q > 0.0 and p > 0.0:
            divergence += q * math.log10(q / p)

    return divergence

def hellinger_distance(q_dist, p_dist, filter_by=0.001):
    assert len(q_dist) == len(p_dist)
    distance = 0.0
    z = zip(q_dist, p_dist)
    for q, p in z:
        if q < filter_by and p < filter_by:
            continue

        inner = math.sqrt(q) - math.sqrt(p)
        distance += (inner * inner)

    distance /= 2
    distance = math.sqrt(distance)
    return distance


def cosine_distance(q_dist, p_dist, filter_by=0.001):
    assert len(q_dist) == len(p_dist)
    z = zip(q_dist, p_dist)
    numerator = 0.0
    denominator_a = 0.0
    denominator_b = 0.0
    for q, p in z:
        if q < filter_by and p < filter_by:
            continue

        numerator += (q * p)
        denominator_a += (q * q)
        denominator_b += (p * p)

    denominator = math.sqrt(denominator_a) * math.sqrt(denominator_b)
    similarity = (numerator / denominator)
    return 1.0 - similarity


def jensen_shannon_divergence(q_dist, p_dist, filter_by=0.001):
    assert len(q_dist) == len(p_dist)
    z = zip(q_dist, p_dist)
    q_dist, p_dist, M = list(), list(), list()
    for q, p in z:
        if q < filter_by and p < filter_by:
            continue

        M.append((q + p) / 2)
        q_dist.append(q)
        p_dist.append(p)

    divergence_a = (kullback_leibler_divergence(q_dist, M) / 2)
    divergence_b = (kullback_leibler_divergence(p_dist, M) / 2)
    return divergence_a + divergence_b


def total_variation_distance(q_dist, p_dist, filter_by=0.001):
    z = zip(q_dist, p_dist)
    distance = 0.0
    for q, p in z:
        if q < filter_by and p < filter_by:
            continue

        distance += math.fabs(q - p)

    distance /= 2
    return distance


def score(model, fn):
    # thomas et al 2011 msr
    #
    scores = list()
    for a, topic_a in norm_phi(model):
        score = 0.0
        for b, topic_b in norm_phi(model):
            if a == b:
                continue

            score += fn(topic_a, topic_b)

        score *= (1.0 / (model.num_topics - 1))
        logger.debug("topic %d score %f" % (a, score))
        scores.append((a, score))

    return scores


def norm_phi(model):
    for topicid in range(model.num_topics):
        topic = model.state.get_lambda()[topicid]
        topic = topic / topic.sum()  # normalize to probability dist
        yield topicid, topic

def create_evaluation_distinctiveness(config, Kind):
    model_fname = config.model_fname % Kind.__name__

    try:
        model = LdaModel.load(model_fname)
        logger.info('Opened previously created model at file %s' % model_fname)
    except:
        error('Cannot evalutate LDA models not built yet!')

    scores = score(model, kullback_leibler_divergence)
    total = sum([x[1] for x in scores])

    logger.info("%s model KL: %f" % (model_fname, total))
    with open(config.path + 'evaluate-results.csv', 'a') as f:
        w = csv.writer(f)
        w.writerow([model_fname, total])

    etas = list()
    for topic in model.state.get_lambda():
        topic_eta = list()
        for p_w in topic:
            topic_eta.append(p_w * numpy.log2(p_w))
            etas.append(-sum(topic_eta))

    entropy = sum(etas) / len(etas)

    logger.info("%s model entropy mean: %f" % (model_fname, entropy))
    with open(config.path + 'evaluate-entropy-results.csv', 'a') as f:
        w = csv.writer(f)
        w.writerow([model_fname, entropy])
