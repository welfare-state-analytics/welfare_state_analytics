import numpy as np
from text_analytic_tools.text_analysis.utility import normalize_array

# source: https://github.com/baali/TopicModellingExperiments

def compute_topic_metrics(ldamodel, dictionary, n_words=1000):
    """Returns weights for top n_words words in all topics
    """
    if n_words is None:
        # Set to 0.1 % of total number of terms
        n_words = len(dictionary.items()) // 100

    topic_info = []

    for index in range(ldamodel.num_topics):
        topic_weight = sum([prob_score for term, prob_score in ldamodel.show_topic(index, topn=n_words) ])
        topic_info.append( {
            'topic' : ldamodel.show_topic(index, topn=n_words),
            'weight' : topic_weight
        } )

    return topic_info

def computeTermFreq(corpus):
    from collections import Counter
    term_freq = Counter()
    for doc in corpus:
        for (term, freq) in doc:
            term_freq[term] += freq
    return term_freq

def computeTermInfo(ldamodel, dictionary, corpus):
    """Iterate over the list of terms. Compute frequency, distinctiveness, saliency."""
    topic_info = compute_topic_metrics(ldamodel, dictionary)
    topic_marginal = normalize_array( [ d['weight'] for d in topic_info ] )
    term_freq = computeTermFreq(corpus)
    term_info = []
    for (tid, term) in dictionary.items():
        frequency = term_freq[tid]
        probs = []
        for index in range(ldamodel.num_topics):
            probs.append(ldamodel.expElogbeta[index][tid])
        probs = normalize_array( probs )
        distinctiveness = getKLDivergence( probs, topic_marginal )
        saliency = frequency * distinctiveness
        term_info.append({
            'term' : term,
            'saliency' : saliency,
            'frequency' : frequency,
            'distinctiveness' : distinctiveness,
            'rank' : None,
            'visibility' : 'default'
        })
    return term_info

def getKLDivergence( P, Q ):
    """Compute KL-divergence from P to Q"""
    import math
    divergence = 0
    assert len(P) == len(Q)
    for i in range(len(P)):
        p = P[i]
        q = Q[i]
        assert p >= 0
        assert q >= 0
        if p > 0:
            divergence += p * math.log( p / q )
    return divergence
