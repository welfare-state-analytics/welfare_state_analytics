# -*- coding: utf-8 -*-
import itertools
import re

import pandas as pd
import spacy
import textacy

from spacy.language import Language

import text_analytic_tools.utility as utility

logger = utility.getLogger('corpus_text_analysis')

LANGUAGE_MODEL_MAP = { 'en': 'en_core_web_sm', 'fr': 'fr_core_news_sm', 'it': 'it_core_web_sm', 'de': 'de_core_web_sm' }

def create_textacy_corpus(corpus_reader, nlp, tick=utility.noop, n_chunk_threshold = 100000):

    corpus = textacy.Corpus(nlp)
    counter = 0

    for filename, document_id, text, metadata in corpus_reader:

        metadata = utility.extend(metadata, dict(filename=filename, document_id=document_id))

        if len(text) > n_chunk_threshold:
            doc = textacy.spacier.utils.make_doc_from_text_chunks(text, lang=nlp, chunk_size=n_chunk_threshold)
            corpus.add_doc(doc)
            doc._.meta = metadata
        else:
            corpus.add((text, metadata))

        counter += 1
        if counter % 100 == 0:
            logger.info('%s documents added...', counter)
        tick(counter)

    return corpus

@utility.timecall
def save_corpus(corpus, filename, lang=None, include_tensor=False):
    if not include_tensor:
        for doc in corpus:
            doc.tensor = None
    corpus.save(filename)

@utility.timecall
def load_corpus(filename, lang, document_id='document_id'):
    corpus = textacy.Corpus.load(lang, filename)
    return corpus

def keep_hyphen_tokenizer(nlp):
    infix_re = re.compile(r'''[.\,\?\:\;\...\‘\’\`\“\”\"\'~]''')
    prefix_re = spacy.util.compile_prefix_regex(nlp.Defaults.prefixes)
    suffix_re = spacy.util.compile_suffix_regex(nlp.Defaults.suffixes)

    return spacy.tokenizer.Tokenizer(nlp.vocab, prefix_search=prefix_re.search, suffix_search=suffix_re.search, infix_finditer=infix_re.finditer, token_match=None)

_load_spacy = textacy.load_spacy_lang if hasattr(textacy, 'load_spacy_lang') else textacy.load_spacy # pylint: disable=no-member

@utility.timecall
def setup_nlp_language_model(language, **nlp_args):

    if (len(nlp_args.get('disable', [])) == 0):
        nlp_args.pop('disable')

    def remove_whitespace_entities(doc):
        doc.ents = [ e for e in doc.ents if not e.text.isspace() ]
        return doc

    logger.info('Loading model: %s...', language)

    Language.factories['remove_whitespace_entities'] = lambda nlp, **cfg: remove_whitespace_entities
    model_name = LANGUAGE_MODEL_MAP[language]
    #if not model_name.endswith('lg'):
    #    logger.warning('Selected model is not the largest availiable.')

    nlp = _load_spacy(model_name, **nlp_args)
    nlp.tokenizer = keep_hyphen_tokenizer(nlp)

    pipeline = lambda: [ x[0] for x in nlp.pipeline ]

    logger.info('Using pipeline: ' + ' '.join(pipeline()))

    return nlp

def store_tokens_to_file(corpus, filename):
    doc_tokens = lambda d: (
        dict(
            i=t.i,
            token=t.lower_,
            lemma=t.lemma_,
            pos=t.pos_,
            year=d._.meta['year'],
            document_id=d._.meta['document_id']
        ) for t in d )
    tokens = pd.DataFrame(list(itertools.chain.from_iterable(doc_tokens(d) for d in corpus )))

    if filename.endswith('.xlxs'):
        tokens.to_excel(filename)
    else:
        tokens['token'] = tokens.token.str.replace('\t', ' ')
        tokens['token'] = tokens.token.str.replace('\n', ' ')
        tokens['token'] = tokens.token.str.replace('"', ' ')
        tokens['lemma'] = tokens.token.str.replace('\t', ' ')
        tokens['lemma'] = tokens.token.str.replace('\n', ' ')
        tokens['lemma'] = tokens.token.str.replace('"', ' ')
        tokens.to_csv(filename, sep='\t')

