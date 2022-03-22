from __future__ import annotations

import os
import uuid
from unittest import mock

import pandas as pd
import penelope.notebook.topic_modelling as tm_ui
import pytest
from penelope import topic_modelling as tm
from penelope.notebook.topic_modelling import mixins as mx

import westac.riksprot.parlaclarin.speech_text as sr
from notebooks.riksdagens_protokoll import topic_modeling as wtm_ui
from westac.riksprot.parlaclarin import codecs as md

jj = os.path.join

# pylint: disable=redefined-outer-name,protected-access

# DATA_FOLDER: str = "/data/riksdagen_corpus_data/"
# MODEL_FOLDER: str = jj(DATA_FOLDER, "tm_1920-2020_500-TF5-MP0.02.500000.lemma.mallet/")
# DATABASE_FILENAME: str = jj(DATA_FOLDER, 'tagged_frames_v0.4.1_speeches.feather', 'riksprot_metadata.v0.4.1.db')
# TAGGED_CORPUS_FOLDER: str = jj(DATA_FOLDER, "tagged_frames_v0.4.1")

DATA_FOLDER: str = "./tests/test_data/riksprot/main"
MODEL_FOLDER: str = jj(DATA_FOLDER, "tm_test.5files.mallet")
DATABASE_FILENAME: str = jj(DATA_FOLDER, 'riksprot_metadata.db')
TAGGED_CORPUS_FOLDER: str = jj(DATA_FOLDER, "tagged_frames")
SPEECH_INDEX_FILENAME: str = jj(DATA_FOLDER, "tagged_frames_speeches.feather/document_index.feather")


def test_db():
    data: md.PersonCodecs = md.PersonCodecs().load(source=DATABASE_FILENAME)
    assert data


@pytest.fixture
def person_codecs() -> md.PersonCodecs:
    data: md.PersonCodecs = md.PersonCodecs().load(source=DATABASE_FILENAME)
    return data


@pytest.fixture
def inferred_topics() -> tm.InferredTopicsData:
    data: tm.InferredTopicsData = tm.InferredTopicsData.load(folder=MODEL_FOLDER, slim=True)
    return data


@pytest.fixture
def speech_index() -> pd.DataFrame:
    di: pd.DataFrame = pd.read_feather(SPEECH_INDEX_FILENAME)
    return di


@pytest.fixture
def speech_repository(person_codecs: md.PersonCodecs, speech_index: pd.DataFrame) -> sr.SpeechTextRepository:
    repository: sr.SpeechTextRepository = sr.SpeechTextRepository(
        source=TAGGED_CORPUS_FOLDER,
        person_codecs=person_codecs,
        document_index=speech_index,
    )
    return repository


def test_load_gui(person_codecs: md.PersonCodecs, inferred_topics: tm.InferredTopicsData):
    state = dict(inferred_topics=inferred_topics)
    ui = wtm_ui.RiksprotLoadGUI(person_codecs, corpus_folder=DATA_FOLDER, corpus_config=None, state=state, slim=True)
    assert ui is not None
    ui.setup()
    ui.load()


def test_loaded_gui(
    inferred_topics: tm.InferredTopicsData,
):
    topic_tokens_overview: pd.DataFrame = inferred_topics.topic_token_overview
    topic_tokens_overview['tokens'] = inferred_topics.get_topic_titles(n_tokens=500)
    topic_proportions = inferred_topics.calculator.topic_proportions()
    if topic_proportions is not None:
        topic_tokens_overview['score'] = topic_proportions

    search_text: str = "riksdag"
    n_top: int = 200
    truncate_tokens: bool = False

    _: pd.DataFrame = tm.filter_topic_tokens_overview(
        topic_tokens_overview, search_text=search_text, n_top=n_top, truncate_tokens=truncate_tokens
    )

    # ui = tm_ui.PandasTopicTitlesGUI(topics=topic_tokens_overview, n_tokens=n_top)


@mock.patch('bokeh.plotting.show', lambda *_, **__: None)
def test_find_documents_gui(
    person_codecs: md.PersonCodecs,
    speech_repository: sr.SpeechTextRepository,
    inferred_topics: tm.InferredTopicsData,
):

    state = dict(inferred_topics=inferred_topics)
    ui: wtm_ui.RiksprotFindTopicDocumentsGUI = wtm_ui.RiksprotFindTopicDocumentsGUI(
        person_codecs, speech_repository, state
    )
    """
    Protokoll: prot-198586--152 sidan 5, Enkammarriksdagen
    Källa (XML): main  (main)  dev  (dev)
    Talare: Göthe Knutson, Moderata samlingspartiet, Värmlands län (man)
    Antal tokens: 82 (82) (i-78c036a7f9e08229-0)
    Anförande av riksdagsman Göthe Knutson (M) 1986-05-27
    """
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
    ui._multi_pivot_keys_picker.value = ["gender"]
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


@mock.patch('bokeh.plotting.show', lambda *_, **__: None)
def test_browse_documents_gui(
    person_codecs: md.PersonCodecs,
    speech_repository: sr.SpeechTextRepository,
    inferred_topics: tm.InferredTopicsData,
):

    state = dict(inferred_topics=inferred_topics)
    ui: wtm_ui.RiksprotBrowseTopicDocumentsGUI = wtm_ui.RiksprotBrowseTopicDocumentsGUI(
        person_codecs, speech_repository, state
    )

    ui.setup()
    ui._year_range.value = (1990, 1992)
    ui._threshold.value = 0.20

    assert ui.pivot_keys is not None
    assert ui.years == (1990, 1992)
    assert ui.threshold == 0.20
    assert ui.filter_opts.data == {'year': (1990, 1992)}

    """Assert no selection is made"""
    assert ui.pivot_keys_id_names == []
    assert ui.pivot_keys_text_names == []

    """Make a selection"""
    ui._multi_pivot_keys_picker.value = ["gender"]
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
    ui._max_count.value = 10

    _ = ui.update()


@mock.patch('bokeh.plotting.show', lambda *_, **__: None)
def test_topic_trends_overview(
    person_codecs: md.PersonCodecs,
    speech_repository: sr.SpeechTextRepository,
    inferred_topics: tm.InferredTopicsData,
):

    state = dict(inferred_topics=inferred_topics)

    # ui = tm_ui.TopicTrendsOverviewGUI(state=state, calculator=calculator).setup()
    ui: wtm_ui.RiksprotTopicTrendsOverviewGUI = wtm_ui.RiksprotTopicTrendsOverviewGUI(
        person_codecs=person_codecs, speech_repository=speech_repository, state=state
    )

    ui.setup()
    ui._year_range.value = (1920, 2020)
    ui.update_handler()

    assert ui._alert.value == "✅"


@mock.patch('bokeh.plotting.show', lambda *_, **__: None)
def test_topic_trends(
    person_codecs: md.PersonCodecs,
    speech_repository: sr.SpeechTextRepository,
    inferred_topics: tm.InferredTopicsData,
):

    state = dict(inferred_topics=inferred_topics)

    # ui = tm_ui.TopicTrendsOverviewGUI(state=state, calculator=calculator).setup()
    ui: wtm_ui.RiksprotTopicTrendsGUI = wtm_ui.RiksprotTopicTrendsGUI(
        person_codecs=person_codecs, speech_repository=speech_repository, state=state
    )

    ui.setup()

    ui.update_handler()

    assert ui is not None


@mock.patch('bokeh.plotting.show', lambda *_, **__: None)
def test_topic_topic_network(
    person_codecs: md.PersonCodecs,
    speech_repository: sr.SpeechTextRepository,
    inferred_topics: tm.InferredTopicsData,
):

    state = dict(inferred_topics=inferred_topics)

    # ui: tm_ui.TopicTopicGUI = tm_ui.TopicTopicGUI(state=state).setup()
    ui: wtm_ui.RiksprotTopicTopicGUI = wtm_ui.RiksprotTopicTopicGUI(
        person_codecs=person_codecs, speech_repository=speech_repository, state=state
    )

    ui.setup()
    _ = ui.layout()

    # ui.pacify(False)
    ui._threshold.value = 0.05
    ui._year_range.value = (1920, 2020)
    ui._n_docs.value = 2
    # ui.pacify(True)

    ui.update_handler()
    # ui._compute.click()

    assert ui is not None


@mock.patch('bokeh.plotting.show', lambda *_, **__: None)
def test_pivot_topic_network(
    person_codecs: md.PersonCodecs,
    inferred_topics: tm.InferredTopicsData,
):

    state = dict(inferred_topics=inferred_topics)

    # ui: tm_ui.TopicTopicGUI = tm_ui.TopicTopicGUI(state=state).setup()
    ui: tm_ui.PivotTopicNetworkGUI = tm_ui.PivotTopicNetworkGUI(
        pivot_key_specs=person_codecs.member_property_specs, state=state
    )

    ui.setup()

    ui.update_handler()
    _ = ui.layout()
    ui._compute.click()

    assert ui.network_data is not None
    assert len(ui.network_data) > 0


def test_topic_labels_gui(inferred_topics: tm.InferredTopicsData):

    folder: str = f'tests/output/{str(uuid.uuid4())[:6]}'
    os.makedirs(folder)

    expected_filename: str = jj(folder, 'topic_token_overview_label.csv')
    assert not os.path.isfile(expected_filename)

    topic_labels: pd.DataFrame = inferred_topics.load_topic_labels(folder=folder, sep='\t', header=0, index_col=0)

    assert topic_labels is not None
    assert 'label' in topic_labels.columns
    assert (topic_labels.label == inferred_topics.topic_token_overview.label).all()
    assert inferred_topics.is_satisfied_topic_token_overview(topic_labels)

    state = dict(inferred_topics=inferred_topics)
    ui = tm_ui.EditTopicLabelsGUI(folder=folder, state=state)
    ui.setup()

    assert topic_labels is not None

    ui.save()
    assert os.path.isfile(expected_filename)


class TNextPrevTopicMixIn(mx.NextPrevTopicMixIn):
    def __init__(self, inferred_topics: tm.InferredTopicsData):
        self.inferred_topics: tm.InferredTopicsData = inferred_topics
        super().__init__()


def test_NextPrevTopicMixIn(inferred_topics: tm.InferredTopicsData):
    ...
    ctrl: TNextPrevTopicMixIn = TNextPrevTopicMixIn(inferred_topics)
    assert ctrl is not None
    ctrl.topic_id = (0, inferred_topics.n_topics - 1, inferred_topics.topic_labels)
    assert ctrl is not None


def test_get_github_tags(speech_repository: sr.SpeechTextRepository):
    release_tags: list[str] = speech_repository.release_tags
    assert len(release_tags) > 2
    assert "main" in release_tags
    github_urls = speech_repository.get_github_xml_urls("prot-1920--ak--1.xml")
    assert len(github_urls) > 0
    links = speech_repository.to_parla_clarin_urls("prot-1920--ak--1.xml")
    assert len(links) > 0


def test_speech_repository(person_codecs: md.PersonCodecs, inferred_topics: tm.InferredTopicsData):
    loader: sr.Loader = sr.ZipLoader(folder=TAGGED_CORPUS_FOLDER)
    repository1: sr.SpeechTextRepository = sr.SpeechTextRepository(
        source=loader,
        person_codecs=person_codecs,
        document_index=inferred_topics.document_index,
    )
    repository2: sr.SpeechTextRepository = sr.SpeechTextRepository(
        source=TAGGED_CORPUS_FOLDER,
        person_codecs=person_codecs,
        document_index=inferred_topics.document_index,
    )

    for metadata, utterances in [
        loader.load('prot-198586--152'),
        repository1.load_protocol('prot-198586--152'),
        repository2.load_protocol('prot-198586--152'),
    ]:
        assert isinstance(metadata, dict)
        assert isinstance(utterances, list)
        assert metadata['name'] == "prot-198586--152"
        assert len(utterances) == 270
        assert utterances[7]['who'] == 'gothe_knutson_2fa077'
        assert utterances[7]["u_id"] == "i-78c036a7f9e08229-3"

    metadata, utterances = repository1.load_protocol('prot-198586--152')

    merger: sr.DefaultMergeStrategy = sr.DefaultMergeStrategy()
    groups: list[str, list[dict]] = merger.groups(utterances)
    assert len(groups) == 70

    assert [(g[0], len(g[1])) for g in groups][:7] == [
        ('karl-anders_petersson_8c3cb6', 2),
        ('inger_hestvik_ddc26f', 3),
        ('gothe_knutson_2fa077', 3),
        ('elver_jonsson_e67379', 7),
        ('ingemund_bengtsson_talman', 1),
        ('per-olof_strindberg_a2107a', 10),
        ('ulla_orring_dac7cf', 12),
    ]

    speech: dict = merger.to_speech(groups[5][1], metadata=metadata)
    assert speech is not None
    assert speech['who'] == 'per-olof_strindberg_a2107a'
    assert len(speech['paragraphs']) == 35
    assert int(speech['page_number']) == 15
    assert int(speech['page_number2']) == 19

    speech = repository1.speech('prot-198586--152_005', mode='dict')
    assert speech['who'] == 'ingemund_bengtsson_talman'
    assert speech['num_words'] == 1
    assert speech['protocol_name'] == 'prot-198586--152'
    assert speech['office_type'] == 'talman'
    assert int(speech['page_number']) == 15
    assert int(speech['page_number2']) == 15

    speech = repository1.speech('prot-198586--152_004', mode='html')
    assert isinstance(speech, str)

    speech = repository1.speech('prot-198586--152_002', mode='text')
    assert isinstance(speech, str)


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
