import os
from typing import Callable

import penelope.notebook.topic_modelling as tm_ui
import pytest
from penelope import pipeline as pp
from penelope import topic_modelling as tm

from notebooks.riksdagens_protokoll import topic_modeling as wtm_ui
from westac.riksprot.parlaclarin import metadata as md
import westac.riksprot.parlaclarin.speech_text as sr

DATA_FOLDER: str = "/data/westac/riksdagen_corpus_data/"

jj = os.path.join

# pylint: disable=protected-access,redefined-outer-name

# import pandas as pd
# import numpy as np
# import zipfile

# def test_investigate_load():
#     data = {}

#     """Load previously stored aggregate"""
#     folder = "/data/westac/riksdagen_corpus_data/tm_1920-2020_500-topics-mallet/"

#     # document_index: pd.DataFrame = (
#     #     pd.read_feather(jj(folder, "documents.feather")).set_index('document_name', drop=False).rename_axis('')
#     # )
#     # assert document_index is not None

#     # data['dictionary'] = pd.read_feather(jj(folder, "dictionary.feather")).set_index('token_id', drop=True)
#     # data['dictionary'].drop(columns='dfs', inplace=True)
#     # assert data['dictionary'] is not None
#     with zipfile.ZipFile(jj(folder, 'topic_token_weights.zip'), "r") as zp:
#         ts = zp.read('topic_token_weights.csv')

#     csv_opts: dict = dict(sep='\t', header=0, index_col=0, na_filter=False)

#     topic_token_weights=pd.read_csv(jj(folder, 'topic_token_weights.zip'), **csv_opts)
#     data['topic_token_weights'] = pd.read_feather(jj(folder, "topic_token_weights.feather")).set_index('token_id', drop=True)

#     if 'token_id' not in  data['topic_token_weights'].columns:
#         data['topic_token_weights'] = data['topic_token_weights'].head().reset_index().set_index('topic_id')
#     if 'token' in data['topic_token_weights']:
#         data['topic_token_weights'].drop(columns='token', inplace=True)
#     # FIXME Varför är Weights så stora tal???

#     assert data['topic_token_weights'] is not None

#     # topic_token_weights=pd.read_feather(jj(folder, "topic_token_weights.feather"))
#     # document_topic_weights=pd.read_feather(jj(folder, "document_topic_weights.feather"))
#     # topic_token_overview=pd.read_feather(jj(folder, "topic_token_overview.feather")).set_index( 'topic_id', drop=True )

# FIXME: Use **way** smaller corpus!


@pytest.fixture
def riksprot_metadata() -> md.ProtoMetaData:
    metadata_folder: str = jj(DATA_FOLDER, 'dtm_1920-2020_v0.3.0.tf20')
    data: md.ProtoMetaData = md.ProtoMetaData.load_from_same_folder(metadata_folder)
    return data


@pytest.fixture
def inferred_topics(riksprot_metadata: md.ProtoMetaData) -> tm.InferredTopicsData:
    data: tm.InferredTopicsData = tm.InferredTopicsData.load(
        folder="/data/westac/riksdagen_corpus_data/tm_1920-2020_500-topics-mallet/", slim=True
    )
    data.document_index = riksprot_metadata.overload_by_member_data(data.document_index, encoded=True, drop=True)
    return data


@pytest.fixture
def speech_repository(riksprot_metadata: md.ProtoMetaData) -> sr.SpeechTextRepository:
    repository: sr.SpeechTextRepository = sr.SpeechTextRepository(
        folder="/data/westac/riksdagen_corpus_data/tagged_frames_v0.3.0_20201218",
        riksprot_metadata=riksprot_metadata,
    )
    return repository


def test_find_documents_gui(
    riksprot_metadata: md.ProtoMetaData,
    speech_repository: sr.SpeechTextRepository,
    inferred_topics: tm.InferredTopicsData,
):

    state = dict(inferred_topics=inferred_topics)
    ui: wtm_ui.RiksprotFindTopicDocumentsGUI = wtm_ui.RiksprotFindTopicDocumentsGUI(
        riksprot_metadata, speech_repository, state
    )

    ui.setup()
    ui._year_range.value = (1990, 1992)
    ui._n_top_token.value = 3
    ui._threshold.value = 0.20

    assert ui.n_top_token == 3
    assert ui.pivot_keys is not None
    assert ui.years == (1990, 1992)
    assert ui.threshold == 0.20
    assert ui.filter_opts.data == {'year': (1990, 1992)}

    """Assert no selection is made"""
    assert ui.pivot_keys_id_names == []
    assert ui.pivot_keys_text_names == []

    """Make a selection"""
    ui._pivot_keys_text_names.value = ["gender"]
    assert ui.pivot_keys_id_names == ["gender_id"]
    assert ui.pivot_keys_text_names == ["gender"]

    """No filter added yet"""
    assert ui.filter_opts.data == {'year': (1990, 1992)}

    assert set(ui.filter_key_values) == {"gender: unknown", "gender: woman", "gender: man"}

    was_called: bool = False

    def handler(self, *_):  # pylint: disable=unused-argument
        nonlocal was_called
        was_called = True

    ui.observe(handler=handler, value=True)
    ui._filter_keys.value = ["gender: woman"]
    ui._find_text.value = "film"

    assert was_called
    assert ui.filter_opts.data == {'gender_id': [2], 'year': (1990, 1992)}
    assert ui.layout() is not None

    ui._auto_compute.value = False
    ui._find_text.value = "film"
    ui._max_count.value = 10

    ui._compute.click()
    _ = ui.update()


def test_browse_documents_gui(
    riksprot_metadata: md.ProtoMetaData,
    speech_repository: sr.SpeechTextRepository,
    inferred_topics: tm.InferredTopicsData,
):

    state = dict(inferred_topics=inferred_topics)
    ui: wtm_ui.RiksprotBrowseTopicDocumentsGUI = wtm_ui.RiksprotBrowseTopicDocumentsGUI(
        riksprot_metadata, speech_repository, state
    )

    ui.setup()
    ui._year_range.value = (1990, 1992)
    ui._n_top_token.value = 3
    ui._threshold.value = 0.20

    assert ui.n_top_token == 3
    assert ui.pivot_keys is not None
    assert ui.years == (1990, 1992)
    assert ui.threshold == 0.20
    assert ui.filter_opts.data == {'year': (1990, 1992)}

    """Assert no selection is made"""
    assert ui.pivot_keys_id_names == []
    assert ui.pivot_keys_text_names == []

    """Make a selection"""
    ui._pivot_keys_text_names.value = ["gender"]
    assert ui.pivot_keys_id_names == ["gender_id"]
    assert ui.pivot_keys_text_names == ["gender"]

    """No filter added yet"""
    assert ui.filter_opts.data == {'year': (1990, 1992)}

    assert set(ui.filter_key_values) == {"gender: unknown", "gender: woman", "gender: man"}

    was_called: bool = False

    def handler(self, *_):  # pylint: disable=unused-argument
        nonlocal was_called
        was_called = True

    ui.observe(handler=handler, value=True)
    ui._filter_keys.value = ["gender: woman"]

    assert was_called
    assert ui.filter_opts.data == {'gender_id': [2], 'year': (1990, 1992)}
    assert ui.layout() is not None

    ui._auto_compute.value = False
    ui._find_text.value = "film"
    ui._max_count_slider.value = 10

    _ = ui.update()


def test_topic_trends_overview(
    riksprot_metadata: md.ProtoMetaData,
    speech_repository: sr.SpeechTextRepository,
    inferred_topics: tm.InferredTopicsData,
):

    state = dict(inferred_topics=inferred_topics)

    # ui = tm_ui.TopicTrendsOverviewGUI(state=state, calculator=calculator).setup()
    ui: wtm_ui.RiksprotTopicTrendsOverviewGUI = wtm_ui.RiksprotTopicTrendsOverviewGUI(
        riksprot_metadata=riksprot_metadata, speech_repository=speech_repository, state=state
    )

    ui.setup()

    ui.update_handler()

    assert ui is not None


def test_topic_trends(
    riksprot_metadata: md.ProtoMetaData,
    speech_repository: sr.SpeechTextRepository,
    inferred_topics: tm.InferredTopicsData,
):

    state = dict(inferred_topics=inferred_topics)

    # ui = tm_ui.TopicTrendsOverviewGUI(state=state, calculator=calculator).setup()
    ui: wtm_ui.RiksprotTopicTrendsGUI = wtm_ui.RiksprotTopicTrendsGUI(
        riksprot_metadata=riksprot_metadata, speech_repository=speech_repository, state=state
    )

    ui.setup()

    ui.update_handler()

    assert ui is not None
