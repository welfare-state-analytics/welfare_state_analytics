import logging
import collections

logger = logging.getLogger(__name__)

def build_vocab(corpus):
    ''' Iterates corpus and add distict terms to vocabulary '''
    logger.info('Builiding vocabulary...')
    token2id = collections.defaultdict()
    token2id.default_factory = token2id.__len__
    for doc in corpus:
        for term in doc:
            token2id[term] # pylint: disable=pointless-statement
    logger.info('Vocabulary of size {} built.'.format(len(token2id)))
    return token2id
