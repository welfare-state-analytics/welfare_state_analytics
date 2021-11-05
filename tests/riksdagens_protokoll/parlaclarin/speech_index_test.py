import os
import uuid
from os.path import isfile, join

import pytest
from numpy import dtype
from westac.riksdagens_protokoll.parlaclarin.speech_index import (
    SPEECH_INDEX_BASENAME,
    SpeechIndex,
    create_speech_index,
    read_speech_index,
    speech_index_exists,
    store_speech_index,
)

SPEECH_INDEX_FOLDER = "./tests/test_data/riksdagens_protokoll/tagged-1"
# SPEECH_INDEX_FOLDER = "/data/riksdagen_corpus_data/annotated"

# @pytest.fixture(scope='session')
# def speech_index() -> SpeechIndex:
#     return create_speech_index(folder=SPEECH_INDEX_FOLDER)


@pytest.mark.skip("Not implemented")
def test_create_speech_index():
    speech_index: SpeechIndex = create_speech_index(folder=SPEECH_INDEX_FOLDER)
    assert speech_index is not None
    # assert len(speech_index) == 12
    assert set(speech_index.columns.tolist()) == {
        'speech_id',
        'speaker',
        'speech_date',
        'speech_index',
        'document_id',
        'document_name',
        'filename',
        'n_tokens',
        'year',
    }

    assert speech_index.dtypes.to_dict() == {
        'speech_id': dtype('O'),
        'speaker': dtype('O'),
        'speech_date': dtype('O'),
        'speech_index': dtype('int16'),
        'document_name': dtype('O'),
        'filename': dtype('O'),
        'n_tokens': dtype('int32'),
        'document_id': dtype('int32'),
        'year': dtype('int16'),
    }


@pytest.mark.skip("Not implemented")
@pytest.mark.parametrize('extension', ['feather', 'excel', 'csv'])
def test_store_speech_index(extension: str):
    target_folder = './tests/output'
    speech_index: SpeechIndex = create_speech_index(folder=SPEECH_INDEX_FOLDER)
    store_speech_index(folder=target_folder, speech_index=speech_index, extension="feather")
    assert isfile(join(target_folder, f"{SPEECH_INDEX_BASENAME}.{extension}"))


@pytest.mark.skip("Not implemented")
def test_speech_index_exists():

    target_folder = f'./tests/output/{uuid.uuid1()}'

    os.makedirs(target_folder, exist_ok=True)

    assert not speech_index_exists(target_folder)

    speech_index: SpeechIndex = create_speech_index(folder=SPEECH_INDEX_FOLDER)
    store_speech_index(target_folder, speech_index)

    assert speech_index_exists(target_folder)


@pytest.mark.skip("Not implemented")
def test_read_speech_index():
    folder: str = SPEECH_INDEX_FOLDER
    speech_index: SpeechIndex = read_speech_index(folder)
    assert speech_index is not None
