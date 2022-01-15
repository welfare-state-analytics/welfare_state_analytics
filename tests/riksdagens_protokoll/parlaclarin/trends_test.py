import os
from typing import Set

import pytest
from penelope import corpus as pc
from penelope.common.keyness.metrics import KeynessMetric
from penelope.corpus.dtm.corpus import VectorizedCorpus
from tests.riksdagens_protokoll.parlaclarin.fixtures import sample_riksprot_metadata
from westac.riksprot.parlaclarin import metadata as md

from notebooks.riksdagens_protokoll.word_trends import word_trends_gui as wt

# pylint: disable=protected-access

# FIXME: Create smaller test data
TEST_FOLDER: str = '/data/riksdagen_corpus_data/dtm_1920-2020_v0.3.0.tf20'

@pytest.fixture
def riksprot_metadata() -> md.ProtoMetaData:
    return sample_riksprot_metadata()


@pytest.mark.long_running
def test_trends_gui_create_without_pivot_keys():

    gui: wt.RiksProtTrendsGUI = wt.RiksProtTrendsGUI(pivot_key_specs=None, riksprot_metadata=None, default_folder=TEST_FOLDER, encoded=True)

    gui = gui.setup()

    assert os.path.isdir(gui.source_folder)
    assert gui.source_folder == gui.options.source_folder == gui._source_folder.selected_path

    assert gui.riksprot_metadata is None
    assert isinstance(gui.pivot_keys_text_names, list)
    assert isinstance(gui.pivot_keys_id_names, list)
    assert len(gui.pivot_keys_text_names) == 0
    assert len(gui.pivot_keys_id_names) == 0
    assert gui.data is None
    assert not gui.defaults
    assert gui.encoded

@pytest.mark.long_running
def test_trends_gui_create_with_pivot_keys(riksprot_metadata: md.ProtoMetaData):

    expected_keys: Set[str] = {'role_type', 'who', 'None', 'party_abbrev', 'gender'}

    gui: wt.RiksProtTrendsGUI = wt.RiksProtTrendsGUI(default_folder=TEST_FOLDER, riksprot_metadata=riksprot_metadata)

    assert set(gui._pivot_keys_text_names.options) == expected_keys
    assert set(gui.pivot_keys_id_names) == set() == set(gui.options.pivot_keys)


def test_trends_gui_setup(riksprot_metadata):

    gui: wt.RiksProtTrendsGUI = wt.RiksProtTrendsGUI(default_folder=TEST_FOLDER, riksprot_metadata=riksprot_metadata)

    it_was_called = False
    def patch_update_picker(self, *_):  # pylint: disable=unused-argument
        nonlocal it_was_called
        it_was_called = True

    gui._update_picker = patch_update_picker

    gui = gui.setup()

    assert gui._displayers is not None
    assert len(gui._displayers) == len(gui._tab.children)

    gui.observe(True)
    gui._words.value = "APA"
    assert it_was_called

    it_was_called = False
    gui.observe(False)
    gui._words.value = "BANAN"
    assert not it_was_called

    gui.observe(True)
    gui.observe(False)

    assert gui.options is not None
    assert gui.options.source_folder.endswith(gui.default_folder)
    assert gui.options.unstack_tabular == gui._unstack_tabular.value
    assert gui.options.pivot_keys == []


@pytest.mark.long_running
def test_trends_gui_load():

    gui = wt.RiksProtTrendsGUI(default_folder=TEST_FOLDER).setup()

    metadata: md.ProtoMetaData = gui.load_metadata(gui.source_folder)

    assert metadata is not None
    assert metadata.document_index is not None
    assert metadata.members is not None

    corpus: pc.VectorizedCorpus = gui.load_corpus()

    assert corpus is not None
    assert corpus.shape[0] == len(metadata.document_index)
    assert len(corpus.document_index) == len(metadata.document_index)

    corpus = gui.assign_metadata(corpus, metadata)

    assert 'gender_id' in corpus.document_index.columns
    assert 'party_abbrev_id' in corpus.document_index.columns
    assert 'role_type_id' in corpus.document_index.columns

    gui.load()
    gui.observe(True)
    gui.observe(False)
    assert gui.proto_metadata is not None
    assert gui.trends_data.corpus is not None
    assert 'gender_id' in gui.trends_data.corpus.document_index.columns


@pytest.mark.long_running
def test_trends_gui_transform():

    default_opts: dict = dict(
        normalize=False,
        keyness=KeynessMetric.TF,
        time_period="decade",
        temporal_key="decade",
        fill_gaps=False,
        smooth=False,
        top_count=100,
        pivot_keys=[],
        unstack_tabular=False,
    )
    folder: str = TEST_FOLDER
    tag: str = os.path.basename(folder)

    metadata: md.ProtoMetaData = md.ProtoMetaData.load_from_same_folder(folder=folder)
    corpus: VectorizedCorpus = pc.VectorizedCorpus.load(folder=folder, tag=tag)
    corpus.replace_document_index(metadata.overload_by_member_data(corpus.document_index, encoded=True))
    trends_data: wt.TrendsData = wt.TrendsData(corpus, n_top=1000)

    assert trends_data is not None

    opts: wt.ComputeOpts = wt.ComputeOpts(**default_opts)

    trends_data.transform(opts)

    assert trends_data.transformed_corpus is not None
    assert set(trends_data.corpus.document_index.columns).intersection({'year', 'time_period'}) == {
        'year',
        'time_period',
    }
    assert set(trends_data.transformed_corpus.document_index.columns).intersection({'year', 'time_period'}) == {
        'year',
        'time_period',
    }
    # assert 'year' not in trends_data.transformed_corpus.document_index.columns (exists, start year of peropd)
    assert len(trends_data.transformed_corpus.document_index) == 11
    assert all(x in trends_data.transformed_corpus.document_index.columns for x in opts.pivot_keys)


def test_trends_gui_bugcheck():
    gui: wt.RiksProtTrendsGUI = wt.RiksProtTrendsGUI(default_folder='/data/riksdagen_corpus_data/dtm_1920-2020_v0.3.0.tf20').setup()

    gui.load()
    gui.compute_keyness()

    gui._invalidate(False)

    gui.plot()
    gui._pivot_keys.value = ['gender', 'role_type']
    gui._time_period.value = 'decade'

    words = ['herr']
    gui.observe(False)
    gui._picker.options = words
    gui._picker.value = words
    for i in range(0, len(gui._displayers)):
        gui._tab.selected_index = i
        gui.plot()


# @pytest.mark.long_running
# @pytest.mark.parametrize(
#     'folder,encoded,expected_columns_added',
#     [
#         (TEST_FOLDER, True, set({'who_id', 'gender_id', 'party_abbrev_id', 'role_type_id'})),
#         (TEST_FOLDER, False, set({'gender', 'party_abbrev', 'role_type'})),
#     ],
# )
# def test_pos_count_gui_prepare(folder: str, encoded: bool, expected_columns_added: Set[str]):

#     gui = tc.PoSCountGUI(default_folder=folder, encoded=encoded).setup(load_data=False).load(folder)

#     columns_after_load: set = set(gui.document_index.columns)

#     assert set(columns_after_load).intersection(expected_columns_added) == set()

#     gui.prepare()

#     assert gui.document_index is not None
#     assert set(gui.document_index.columns).intersection(expected_columns_added) == set(expected_columns_added)


# @pytest.mark.long_running
# @pytest.mark.parametrize(
#     'folder,temporal_key,encoded,expected_count',
#     [
#         (TEST_FOLDER, 'year', True, 101),
#         (TEST_FOLDER, 'decade', True, 11),
#         (TEST_FOLDER, 'lustrum', True, 21),
#         (TEST_FOLDER, 'year', False, 101),
#     ],
# )
# def test_pos_count_gui_compute_without_pivot_keys(folder: str, temporal_key: str, encoded: bool, expected_count: int):

#     gui = tc.PoSCountGUI(default_folder=folder, encoded=encoded).setup(load_data=False).load(folder).prepare()
#     gui.observe(False)

#     gui._temporal_key.value = temporal_key
#     data: pd.DataFrame = gui.compute(gui.document_index, gui.opts)
#     assert len(data) == expected_count
#     assert set(data.columns) == set(['Total', temporal_key])


# # FIXME: Compute fails if encode is False
# @pytest.mark.long_running
# @pytest.mark.parametrize(
#     'folder,temporal_key,encoded,pivot_keys,expected_count',
#     [
#         (TEST_FOLDER, 'decade', True, [], 11),
#         (TEST_FOLDER, 'decade', True, ['gender_id'], 11),
#         (TEST_FOLDER, 'decade', True, ['gender_id', 'party_abbrev_id'], 11),
#     ],
# )
# def test_pos_count_gui_compute_and_plot_with_pivot_keys_and_unstacked(
#     folder: str, temporal_key: str, encoded: bool, pivot_keys: List[str], expected_count: int
# ):

#     #expected_pivot_keys = [x.rstrip('_id') for x in pivot_keys]

#     gui = tc.PoSCountGUI(default_folder=folder, encoded=encoded).setup(load_data=False).load(folder).prepare()

#     gui.observe(False)

#     gui._temporal_key.value = temporal_key
#     gui._pivot_keys.value = [gui.pivot_key_idname2name.get(x) for x in pivot_keys]

#     data: pd.DataFrame = gui.compute()
#     unstacked_data = gui.unstack_pivot_keys(data)

#     assert len(unstacked_data) == expected_count

#     gui.plot(data)


# @pytest.mark.long_running
# @pytest.mark.parametrize('folder,encoded', [(TEST_FOLDER, True)])
# def test_pos_count_gui_display(folder: str, encoded: bool):

#     gui = tc.PoSCountGUI(default_folder=folder, encoded=encoded).setup(load_data=False).load(folder).prepare()
#     gui.tab.display_content = lambda *_, **__: None

#     layout = gui.layout()
#     assert isinstance(layout, widgets.VBox)

#     gui = gui.display()

#     assert gui._status.value == '✔'
