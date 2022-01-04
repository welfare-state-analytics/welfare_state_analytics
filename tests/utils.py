# type: ignore

import os

import numpy as np
import pandas as pd
from penelope import pipeline
from penelope.corpus import ExtractTaggedTokensOpts, TextReaderOpts, TokensTransformOpts, VectorizedCorpus
from penelope.workflows import interface
from penelope.workflows.vectorize import dtm as workflow

os.makedirs('./tests/output', exist_ok=True)

# pylint: disable=too-many-arguments

if __file__ in globals():
    this_file = os.path.dirname(__file__)
    this_path = os.path.abspath(this_file)


def create_smaller_vectorized_corpus():
    bag_term_matrix = np.array([[2, 1, 4, 1], [2, 2, 3, 0], [2, 3, 2, 0], [2, 4, 1, 1], [2, 0, 1, 1]])
    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    document_index = pd.DataFrame({'year': [2013, 2013, 2014, 2014, 2014]})
    v_corpus = VectorizedCorpus(bag_term_matrix, token2id=token2id, document_index=document_index)
    return v_corpus


def create_bigger_vectorized_corpus(
    corpus_source: str,
    output_tag: str = "xyz_nnvb_lemma",
    output_folder: str = "./tests/output",
    tf_threshold: int = 5,
    tf_threshold_mask: bool = False,
):

    args: interface.ComputeOpts = interface.ComputeOpts(
        corpus_type=pipeline.CorpusType.SparvCSV,
        corpus_source=corpus_source,
        target_folder=output_folder,
        corpus_tag=f"{output_tag}_nnvb_lemma",
        transform_opts=TokensTransformOpts(
            only_alphabetic=False,
            only_any_alphanumeric=False,
            to_lower=True,
            to_upper=False,
            min_len=1,
            max_len=None,
            remove_accents=False,
            remove_stopwords=True,
            stopwords=None,
            extra_stopwords=["Örn"],
            language="swedish",
            keep_numerals=True,
            keep_symbols=True,
        ),
        text_reader_opts=TextReaderOpts(
            filename_pattern='*.csv',
            filename_fields=r"year:prot\_(\d{4}).*",
            index_field=None,  # use filename
            as_binary=False,
        ),
        extract_opts=ExtractTaggedTokensOpts(
            pos_includes="|NN|PM|VB|",
            pos_excludes="|MAD|MID|PAD|",
            pos_paddings=None,
            passthrough_tokens=[],
            block_tokens=[],
            lemmatize=True,
            append_pos=False,
            global_tf_threshold=tf_threshold,
            global_tf_threshold_mask=tf_threshold_mask,
            text_column='token',
            lemma_column='baseform',
            pos_column='pos',
        ),
        tf_threshold=tf_threshold,
        tf_threshold_mask=tf_threshold_mask,
        create_subfolder=True,
        persist=True,
        vectorize_opts=None,
    )
    corpus: VectorizedCorpus = workflow.compute(args=args, corpus_config=None)

    return corpus
