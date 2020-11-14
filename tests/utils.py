import os
from typing import Callable

import numpy as np
import pandas as pd
from penelope.corpus import TextTransformOpts, TokensTransformOpts
from penelope.corpus.readers import AnnotationOpts, TextTokenizer
from penelope.corpus.vectorized_corpus import VectorizedCorpus
from penelope.workflows import vectorize_corpus_workflow

TEST_CORPUS_FILENAME = './tests/test_data/test_corpus.zip'
OUTPUT_FOLDER = './tests/output'
TEST_DATA_FOLDER = './tests/test_data'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# pylint: disable=too-many-arguments

if __file__ in globals():
    this_file = os.path.dirname(__file__)
    this_path = os.path.abspath(this_file)
    TEST_CORPUS_FILENAME = os.path.join(this_path, TEST_CORPUS_FILENAME)


def create_text_tokenizer(
    source_path=TEST_CORPUS_FILENAME,
    transforms=None,
    chunk_size: int = None,
    filename_pattern: str = "*.txt",
    filename_filter: str = None,
    filename_fields=None,
    fix_whitespaces=False,
    fix_hyphenation=True,
    as_binary: bool = False,
    tokenize: Callable = None,
):
    kwargs = dict(
        transforms=transforms,
        chunk_size=chunk_size,
        filename_pattern=filename_pattern,
        filename_filter=filename_filter,
        as_binary=as_binary,
        tokenize=tokenize,
        filename_fields=filename_fields,
        text_transform_opts=TextTransformOpts(fix_whitespaces=fix_whitespaces, fix_hyphenation=fix_hyphenation),
    )
    reader = TextTokenizer(source_path, **kwargs)
    return reader


def create_smaller_vectorized_corpus():
    bag_term_matrix = np.array([[2, 1, 4, 1], [2, 2, 3, 0], [2, 3, 2, 0], [2, 4, 1, 1], [2, 0, 1, 1]])
    token2id = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
    df = pd.DataFrame({'year': [2013, 2013, 2014, 2014, 2014]})
    v_corpus = VectorizedCorpus(bag_term_matrix, token2id, df)
    return v_corpus


def create_bigger_vectorized_corpus(
    corpus_filename: str,
    output_tag: str = "xyz_nnvb_lemma",
    output_folder: str = "./tests/output",
    count_threshold: int = 5,
):
    filename_field = r"year:prot\_(\d{4}).*"
    count_threshold = 5
    output_tag = f"{output_tag}_nnvb_lemma"
    annotation_opts = AnnotationOpts(
        pos_includes="|NN|PM|UO|PC|VB|",
        pos_excludes="|MAD|MID|PAD|",
        passthrough_tokens=[],
        lemmatize=True,
        append_pos=False,
    )
    tokens_transform_opts = TokensTransformOpts(
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
    )
    corpus = vectorize_corpus_workflow(
        corpus_type="sparv4-csv",
        input_filename=corpus_filename,
        output_folder=output_folder,
        output_tag=output_tag,
        create_subfolder=True,
        filename_field=filename_field,
        count_threshold=count_threshold,
        annotation_opts=annotation_opts,
        tokens_transform_opts=tokens_transform_opts,
    )

    return corpus
