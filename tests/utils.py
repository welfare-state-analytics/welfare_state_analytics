import os
from typing import Callable

from penelope.corpus import TextTransformOpts
from penelope.corpus.readers import TextTokenizer

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
