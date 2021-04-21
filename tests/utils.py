import os

import numpy as np
import pandas as pd
import penelope.corpus.dtm as dtm
from penelope import pipeline, workflows
from penelope.corpus import ExtractTaggedTokensOpts, TextReaderOpts, TokensTransformOpts
from penelope.notebook import interface

TEST_CORPUS_FILENAME = './tests/test_data/test_corpus.zip'
OUTPUT_FOLDER = './tests/output'
TEST_DATA_FOLDER = './tests/test_data'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# pylint: disable=too-many-arguments

if __file__ in globals():
    this_file = os.path.dirname(__file__)
    this_path = os.path.abspath(this_file)
    TEST_CORPUS_FILENAME = os.path.join(this_path, TEST_CORPUS_FILENAME)


def create_smaller_vectorized_corpus():
    bag_term_matrix = np.array([[2, 1, 4, 1], [2, 2, 3, 0], [2, 3, 2, 0], [2, 4, 1, 1], [2, 0, 1, 1]])
    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    document_index = pd.DataFrame({'year': [2013, 2013, 2014, 2014, 2014]})
    v_corpus = dtm.VectorizedCorpus(bag_term_matrix, token2id, document_index)
    return v_corpus


def create_bigger_vectorized_corpus(
    corpus_filename: str,
    output_tag: str = "xyz_nnvb_lemma",
    output_folder: str = "./tests/output",
    count_threshold: int = 5,
):

    args: interface.ComputeOpts = interface.ComputeOpts(
        corpus_type=pipeline.CorpusType.SparvCSV,
        corpus_filename=corpus_filename,
        target_folder=output_folder,
        corpus_tag=f"{output_tag}_nnvb_lemma",
        tokens_transform_opts=TokensTransformOpts(
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
        extract_tagged_tokens_opts=ExtractTaggedTokensOpts(
            pos_includes="|NN|PM|VB|",
            pos_excludes="|MAD|MID|PAD|",
            pos_paddings=None,
            passthrough_tokens=[],
            lemmatize=True,
            append_pos=False,
        ),
        count_threshold=count_threshold,
        create_subfolder=True,
        persist=True,
    )
    corpus: dtm.VectorizedCorpus = workflows.document_term_matrix.compute(args=args, corpus_config=None)

    return corpus
