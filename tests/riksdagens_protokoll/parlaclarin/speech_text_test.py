from __future__ import annotations

import os

import pandas as pd
import pytest
from penelope import topic_modelling as tm

import westac.riksprot.parlaclarin.speech_text as sr
from westac.riksprot.parlaclarin import codecs as md

jj = os.path.join

DATA_FOLDER: str = "./tests/test_data/riksprot/main"
MODEL_FOLDER: str = jj(DATA_FOLDER, "tm_test.5files.mallet")
DATABASE_FILENAME: str = jj(DATA_FOLDER, 'riksprot_metadata.db')
TAGGED_CORPUS_FOLDER: str = jj(DATA_FOLDER, "tagged_frames")
SPEECH_INDEX_FILENAME: str = jj(DATA_FOLDER, "tagged_frames_speeches.feather/document_index.feather")

# pylint: disable=redefined-outer-name


@pytest.fixture
def speech_repository(person_codecs: md.PersonCodecs, speech_index: pd.DataFrame) -> sr.SpeechTextRepository:
    repository: sr.SpeechTextRepository = sr.SpeechTextRepository(
        source=TAGGED_CORPUS_FOLDER, person_codecs=person_codecs, document_index=speech_index
    )
    return repository


@pytest.fixture
def speech_index() -> pd.DataFrame:
    di: pd.DataFrame = pd.read_feather(SPEECH_INDEX_FILENAME)
    return di


@pytest.fixture
def person_codecs() -> md.PersonCodecs:
    data: md.PersonCodecs = md.PersonCodecs().load(source=DATABASE_FILENAME)
    return data


@pytest.fixture
def inferred_topics() -> tm.InferredTopicsData:
    data: tm.InferredTopicsData = tm.InferredTopicsData.load(folder=MODEL_FOLDER, slim=True)
    return data


def test_get_github_tags(speech_repository: sr.SpeechTextRepository):
    release_tags: list[str] = speech_repository.release_tags
    assert "main" in release_tags
    github_urls = speech_repository.get_github_xml_urls("prot-1920--ak--1.xml")
    assert len(github_urls) > 0
    links = speech_repository.to_parla_clarin_urls("prot-1920--ak--1.xml")
    assert len(links) > 0


def test_loader():
    document_name: str = 'prot-1933--fk--5'
    loader: sr.Loader = sr.ZipLoader(folder=TAGGED_CORPUS_FOLDER)
    metadata, utterances = loader.load(document_name)
    assert metadata.get('name') == document_name
    assert metadata.get('date') == '1933-01-21'
    assert len(utterances) == 2


# speech_index.groupby('protocol_name').agg(n_utterances=('n_utterances', sum), n_speeches=('n_utterances', len)).to_dict("index")
@pytest.mark.parametrize(
    'protocol_name, n_utterances, n_speeches',
    [
        ('prot-1933--fk--5', 2, 1),
        ('prot-1955--ak--22', 414, 151),
        ('prot-199192--127', 274, 222),
        ('prot-199192--21', 28, 20),
        ('prot-199596--35', 396, 21),
    ],
)
def test_speech_text_service(speech_index: pd.DataFrame, protocol_name: str, n_utterances: int, n_speeches: int):
    loader: sr.Loader = sr.ZipLoader(folder=TAGGED_CORPUS_FOLDER)

    strategy: sr.SpeechTextService = sr.SpeechTextService(speech_index)
    metadata, utterances = loader.load(protocol_name)
    assert len(utterances) == n_utterances

    speeches: list[dict] = strategy.speeches(metadata=metadata, utterances=utterances)
    assert len(speeches) == n_speeches


@pytest.mark.parametrize(
    'protocol_name', ['prot-1933--fk--5', 'prot-1955--ak--22', 'prot-199192--127', 'prot-199192--21', 'prot-199596--35']
)
def test_speech_to_dict(person_codecs: md.PersonCodecs, speech_index: pd.DataFrame, protocol_name: str):
    repository: sr.SpeechTextRepository = sr.SpeechTextRepository(
        source=TAGGED_CORPUS_FOLDER, person_codecs=person_codecs, document_index=speech_index
    )

    speech_infos: list[str] = speech_index[speech_index['document_name'].str.startswith(protocol_name)].to_dict(
        'records'
    )

    for speech_info in speech_infos:
        speech = repository.speech(speech_name=speech_info['document_name'], mode='dict')

        assert all(speech[k] == speech_info[k] for k in set(speech.keys()).intersection(speech_info.keys()))
        assert speech.get("speaker_note")

        html_speech: str = repository.speech(speech_info['document_name'], mode='html')

        assert isinstance(html_speech, str)
        assert "Protokoll:" in html_speech
        assert speech.get("speaker_note", "") in html_speech

        text_speech = repository.speech(speech_info['document_name'], mode='text')

        assert isinstance(text_speech, str)


# default_template: Template = Template(
#     """
# <b>Protokoll:</b> {{protocol_name}} sidan {{ page_number }}, {{ chamber }} <br/>
# <b>Källa (XML):</b> {{parlaclarin_links}} <br/>
# <b>Talare:</b> {{name}}, {{ party }}, {{ district }} ({{ gender}}) <br/>
# <b>Antal tokens:</b> {{ num_tokens }} ({{ num_words }}),  uid: {{u_id}}, who: {{who}} ) <br/>
# <h3> Anförande av {{ office_type }} {{ sub_office_type }} {{ name }} ({{ party_abbrev }}) {{ date }}</h3>
# <h2> {{ speaker_note }} </h2>
# <span style="color: blue;line-height:50%;">
# {% for n in paragraphs %}
# {{n}}
# {% endfor %}
# </span>
# """
# )
