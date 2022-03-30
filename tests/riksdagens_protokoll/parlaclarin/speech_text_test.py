from __future__ import annotations

import os
import shutil
import uuid

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


@pytest.fixture
def speech_repository(person_codecs: md.PersonCodecs, speech_index: pd.DataFrame) -> sr.SpeechTextRepository:
    repository: sr.SpeechTextRepository = sr.SpeechTextRepository(
        source=TAGGED_CORPUS_FOLDER,
        person_codecs=person_codecs,
        document_index=speech_index,
    )
    return repository


def test_get_github_tags(speech_repository: sr.SpeechTextRepository):
    release_tags: list[str] = speech_repository.release_tags
    assert len(release_tags) > 2
    assert "main" in release_tags
    github_urls = speech_repository.get_github_xml_urls("prot-1920--ak--1.xml")
    assert len(github_urls) > 0
    links = speech_repository.to_parla_clarin_urls("prot-1920--ak--1.xml")
    assert len(links) > 0


def test_speech_repository(person_codecs: md.PersonCodecs, inferred_topics: tm.InferredTopicsData):

    protocol_name: str = "prot-199192--127"

    """loader wants data in sub folders"""
    work_folder: str = f"tests/output/{str(uuid.uuid1())[:8]}"
    sub_folder: str = protocol_name.split("-")[1]
    os.makedirs(jj(work_folder, sub_folder))
    shutil.copy(
        jj(TAGGED_CORPUS_FOLDER, f"{protocol_name}.zip"),
        jj(work_folder, sub_folder, f"{protocol_name}.zip"),
    )

    loader: sr.Loader = sr.ZipLoader(folder=work_folder)
    repository1: sr.SpeechTextRepository = sr.SpeechTextRepository(
        source=loader,
        person_codecs=person_codecs,
        document_index=inferred_topics.document_index,
    )
    repository2: sr.SpeechTextRepository = sr.SpeechTextRepository(
        source=work_folder,
        person_codecs=person_codecs,
        document_index=inferred_topics.document_index,
    )

    for metadata, utterances in [
        loader.load(protocol_name),
        repository1.load_protocol(protocol_name),
        repository2.load_protocol(protocol_name),
    ]:
        assert isinstance(metadata, dict)
        assert isinstance(utterances, list)
        assert metadata['name'] == protocol_name
        assert len(utterances) == 274
        assert utterances[7]['who'] == 'Q4946998'
        assert utterances[7]["u_id"] == "i-4946a79dbcd35844-0"

    metadata, utterances = repository1.load_protocol(protocol_name)

    merger: sr.DefaultMergeStrategy = sr.DefaultMergeStrategy()
    groups: list[tuple[str, list[dict]]] = merger.groups(utterances)
    assert len(groups) == 222

    assert [
        ('6cb28761', 1),
        ('e8dc8b2f', 1),
        ('8028ef00', 1),
        ('353b2727', 1),
        ('0b05f8f4', 1),
        ('59437d17', 1),
        ('3b5aaf83', 2),
    ]

    speech: dict = merger.to_speech(groups[5][1], metadata=metadata)
    assert speech is not None
    assert speech['who'] == 'Q5571477'
    assert len(speech['paragraphs']) == 3
    assert speech['page_number'] == '?'
    assert speech['page_number2'] == '?'

    speech = repository1.speech(f'{protocol_name}_004', mode='dict')
    assert speech['who'] == 'Q5571477'
    assert speech['name'] == 'Leif Bergdahl'
    assert speech['num_words'] == 74
    assert speech['protocol_name'] == protocol_name
    assert speech['office_type'] == 'Ledamot'
    assert speech['office_type_id'] == 1
    assert speech['sub_office_type'] == 'ledamot av Sveriges riksdag'
    assert speech['gender'] == 'man'
    assert speech['page_number'] == '?'
    assert speech['page_number2'] == '?'

    speech = repository1.speech(f'{protocol_name}_004', mode='html')
    assert isinstance(speech, str)

    speech = repository1.speech(f'{protocol_name}_004', mode='text')
    assert isinstance(speech, str)
