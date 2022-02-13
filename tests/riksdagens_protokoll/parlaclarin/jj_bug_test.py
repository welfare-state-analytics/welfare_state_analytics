
from __future__ import annotations

import os
from unittest import mock
import pandas as pd

from penelope import topic_modelling as tm

import westac.riksprot.parlaclarin.speech_text as sr
from notebooks.riksdagens_protokoll import topic_modeling as wtm_ui
from westac.riksprot.parlaclarin import metadata as md

jj = os.path.join

DATA_FOLDER: str = "/data/westac/riksdagen_corpus_data/"
MODEL_FOLDER: str = jj(DATA_FOLDER, "tm_1920-2020_500-TF5-MP0.02.500000.lemma.mallet/")


def get_riksprot_metadata() -> md.ProtoMetaData:
    person_filename: str = jj(DATA_FOLDER, 'dtm_1920-2020_v0.3.0.tf20', 'person_index.zip')
    data: md.ProtoMetaData = md.ProtoMetaData(members=person_filename)
    return data


def get_inferred_topics(riksprot_metadata: md.ProtoMetaData) -> tm.InferredTopicsData:
    data: tm.InferredTopicsData = tm.InferredTopicsData.load(
        # folder="/data/westac/riksdagen_corpus_data/tm_1920-2020_500-topics-mallet/", slim=True
        folder=jj(DATA_FOLDER, "tm_1920-2020_500-TF5-MP0.02.500000.lemma.mallet/"),
        slim=True,
    )
    data.document_index = riksprot_metadata.overload_by_member_data(data.document_index, encoded=True, drop=True)
    return data


def get_speech_repository(riksprot_metadata: md.ProtoMetaData) -> sr.SpeechTextRepository:
    repository: sr.SpeechTextRepository = sr.SpeechTextRepository(
        source=jj(DATA_FOLDER, "tagged_frames_v0.3.0_20201218"),
        riksprot_metadata=riksprot_metadata,
    )
    return repository


def test_find_documents_gui( ):

    riksprot_metadata: md.ProtoMetaData = get_riksprot_metadata()
    speech_repository: sr.SpeechTextRepository = get_speech_repository(riksprot_metadata)
    inferred_topics: tm.InferredTopicsData = get_inferred_topics(riksprot_metadata)
    state = dict(inferred_topics=inferred_topics)

    ui: wtm_ui.RiksprotFindTopicDocumentsGUI = wtm_ui.RiksprotFindTopicDocumentsGUI(
        riksprot_metadata, speech_repository, state
    )
    """
    Protokoll: prot-198586--152 sidan 5, Enkammarriksdagen
    Källa (XML): main  (main)  dev  (dev)
    Talare: Göthe Knutson, Moderata samlingspartiet, Värmlands län (man)
    Antal tokens: 82 (82) (i-78c036a7f9e08229-0)
    Anförande av riksdagsman Göthe Knutson (M) 1986-05-27

    ok, men sök på "information" under load model, 200 tokens. då kommer tre topics upp.
    sen under BROWSE Find topic's documents by token, tid 1920-2020, threshold 0.1,
    toplist threshold 100, sökord "information", och inga andra filter.

    """
    ui.setup()
    ui._auto_compute.value = False
    ui._year_range.value = (1920, 2020)
    ui._n_top_token.value = 200
    ui._threshold.value = 0.10
    ui._find_text.value = "information"

    # ui._compute.click()
    data: pd.DataFrame = ui.update()

    assert data is not None

    """

    RIKSPROT_METADATA:
        Source: dtm_1920-2020_v0.3.0.tf20, person_index.zip
        riksprot_metadata.members.loc['ingemund_bengtsson_talman']
            role_type                   talman
            born                             0
            chamber          Enkammarriksdagen
            district                       NaN
            start                         1979
            end                           1988
            gender                     unknown
            name            Ingemund Bengtsson
            occupation                     NaN
            party                            s
            party_abbrev               unknown
            Name: ingemund_bengtsson_talman, dtype: object
        riksprot_metadata.encoded_members.loc['ingemund_bengtsson_talman']
            born                                0
            chamber             Enkammarriksdagen
            district                          NaN
            start                            1979
            end                              1988
            name               Ingemund Bengtsson
            occupation                        NaN
            party                               s
            who_id                            NaN
            gender_id                           0
            party_abbrev_id                    10
            role_type_id                        1

    DOCUMENT_INDEX:
        Source: tm_1920-2020_500-TF5-MP0.02.500000.lemma.mallet/document_index.feather
        Length: 697950
        document_index.loc[324275]
            year                                  1985
            document_name         prot-198586--152_005
            n_tokens                                 1
            who              ingemund_bengtsson_talman
            n_raw_tokens                             1

        Overloaded and encoded:
            Invalid ROW:
                year                               1985
                document_name      prot-198586--152_005
                n_tokens                              1  <===== WRONG!
                n_raw_tokens                          1  <===== WRONG!
                who_id                                0  <===== WRONG!
                gender_id                             0
                party_abbrev_id                      10
                role_type_id                          1

    SPEECH_REPOSITORY:
        Source: tagged_frames_v0.3.0_20201218


"""
    # bug_row: pd.Series = data.loc[324275]

    # text: str = speech_repository.speech(speech_name='prot-198586--152_005', mode='html')
    # assert text is not None

    # assert bug_row.gender == "man"

    # document_item: pd.Series = inferred_topics.document_index.loc[324275]

    # assert document_item.who_id > 0
