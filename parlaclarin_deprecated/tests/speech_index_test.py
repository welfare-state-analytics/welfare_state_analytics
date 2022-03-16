import os
import uuid
from os.path import isfile, join

import pytest
from numpy import dtype
from westac.riksdagens_protokoll.parlaclarin.speech_index import SPEECH_INDEX_BASENAME, SpeechIndex, SpeechIndexHelper

SPEECH_INDEX_FOLDER = "tests/test_data/riksdagens_protokoll/v0.4.0/tagged_corpus"


def test_create_speech_index():
    speech_index: SpeechIndex = SpeechIndexHelper.create(folder=SPEECH_INDEX_FOLDER).value
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
        'document_id': dtype('int64'),
        'year': dtype('int16'),
    }


@pytest.mark.parametrize('extension', ['feather', 'xlsx', 'csv', 'zip'])
def test_store_speech_index(extension: str):
    target_folder = './tests/output'
    SpeechIndexHelper.create(folder=SPEECH_INDEX_FOLDER).store(folder=target_folder, extension=extension)
    assert isfile(join(target_folder, f"{SPEECH_INDEX_BASENAME}.{extension.split('_')[0]}"))


@pytest.mark.parametrize('extension', ['feather', 'xlsx', 'csv', 'zip'])
def test_speech_index_store_load(extension: str):

    target_folder = f'./tests/output/{uuid.uuid1()}'

    os.makedirs(target_folder, exist_ok=True)

    assert not SpeechIndexHelper.exists(target_folder)

    helper: SpeechIndexHelper = SpeechIndexHelper.create(folder=SPEECH_INDEX_FOLDER)
    helper.store(target_folder, extension)

    assert SpeechIndexHelper.exists(target_folder)

    helper2: SpeechIndexHelper = SpeechIndexHelper.load(target_folder)

    assert (helper.value == helper2.value).all().all()
