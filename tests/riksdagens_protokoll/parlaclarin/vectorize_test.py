import os
import shutil
import uuid

import pytest
from scripts.riksdagens_protokoll.parlaclarin.vectorize import process

jj = os.path.join


CORPUS_SOURCE: str = './tests/test_data/riksdagens_protokoll/parlaclarin/tagged_corpus_01'

VECTORIZE_OPTS: dict = dict(
    corpus_config='./resources/riksdagens-protokoll-parlaclarin.yml',
    corpus_source=CORPUS_SOURCE,
    create_subfolder=True,
    pos_includes='NN|PM|VB',
    pos_excludes='MAD|MID|PAD',
    pos_paddings="AB|DT|HA|HD|HP|HS|IE|IN|JJ|KN|PC|PL|PN|PP|PS|RG|RO|SN|UO",
    to_lower=True,
    lemmatize=True,
    remove_stopwords=None,
    min_word_length=1,
    max_word_length=None,
    keep_symbols=True,
    keep_numerals=True,
    only_any_alphanumeric=False,
    only_alphabetic=False,
    tf_threshold=1,
    merge_speeches=False,
)

@pytest.mark.long_running
def test_vectorize_cli():

    output_folder: str = './tests/output'
    output_tag: str = f'{uuid.uuid4()}'

    target_folder: str = jj(output_folder, output_tag)

    opts: dict = dict(output_folder=output_folder, output_tag=output_tag, **VECTORIZE_OPTS)

    shutil.rmtree(target_folder,ignore_errors=True)

    process(**opts)

    assert os.path.isdir(target_folder)
    assert os.path.isfile(jj(target_folder, f'{output_tag}_vectorizer_data.json'))
