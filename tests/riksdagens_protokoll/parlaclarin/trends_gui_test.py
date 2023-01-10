import os
from types import MethodType
from unittest import mock

import pytest
from penelope import corpus as pc
from penelope import utility as pu
from penelope.common.keyness.metrics import KeynessMetric

from notebooks.riksdagens_protokoll.word_trends import word_trends_gui as wt
from westac.riksprot.parlaclarin import codecs as md

# pylint: disable=protected-access,redefined-outer-name,(too-many-locals

TEST_FOLDER: str = "./tests/test_data/riksprot/main/dtm_test.5files"
METADATA_FILENAME: str = "./tests/test_data/riksprot/main/riksprot_metadata.db"


@pytest.fixture
def person_codecs() -> md.PersonCodecs:
    return md.PersonCodecs().load(source=METADATA_FILENAME)


@pytest.mark.long_running
def test_trends_gui_create_without_pivot_keys():

    gui: wt.RiksProtTrendsGUI = wt.RiksProtTrendsGUI(
        pivot_key_specs=None, person_codecs=None, default_folder=TEST_FOLDER, encoded=True
    )

    gui = gui.setup()

    assert os.path.isdir(gui.source_folder)
    assert gui.source_folder == gui.options.source_folder == gui._source_folder.selected_path

    assert gui.person_codecs is None
    assert isinstance(gui.pivot_keys_text_names, list)
    assert isinstance(gui.pivot_keys_id_names, list)
    assert len(gui.pivot_keys_text_names) == 0
    assert len(gui.pivot_keys_id_names) == 0
    assert gui.data is None
    assert not gui.defaults
    assert gui.encoded


@pytest.mark.long_running
def test_trends_gui_create_with_metadata(person_codecs: md.PersonCodecs):

    gui: wt.RiksProtTrendsGUI = wt.RiksProtTrendsGUI(default_folder=TEST_FOLDER, person_codecs=person_codecs).setup()

    assert gui.person_codecs is not None


@pytest.mark.long_running
def test_trends_gui_corpus_assign_metadata(person_codecs: md.PersonCodecs):

    gui: wt.RiksProtTrendsGUI = wt.RiksProtTrendsGUI(default_folder=TEST_FOLDER, person_codecs=person_codecs).setup()

    corpus: pc.VectorizedCorpus = gui.load_corpus(overload=False)
    # corpus = gui.assign_metadata(corpus, gui.person_codecs)

    assert 'gender_id' in corpus.document_index.columns
    assert 'party_id' in corpus.document_index.columns
    assert 'office_type_id' in corpus.document_index.columns


@pytest.mark.long_running
def test_trends_gui_create_with_pivot_keys(person_codecs: md.PersonCodecs):

    expected_keys: set[str] = {'sub_office_type', 'office_type', 'name', 'None', 'party_abbrev', 'gender'}

    gui: wt.RiksProtTrendsGUI = wt.RiksProtTrendsGUI(default_folder=TEST_FOLDER, person_codecs=person_codecs)

    assert set(gui._multi_pivot_keys_picker.options) == expected_keys
    assert set(gui.pivot_keys_id_names) == set() == set(gui.options.pivot_keys_id_names)


def test_trends_gui_update_picker(person_codecs: md.PersonCodecs):

    gui: wt.RiksProtTrendsGUI = wt.RiksProtTrendsGUI(default_folder=TEST_FOLDER, person_codecs=person_codecs)

    it_was_called = False

    def patch_update_picker(self, *_):  # pylint: disable=unused-argument
        nonlocal it_was_called
        it_was_called = True

    gui._update_picker_handler = MethodType(patch_update_picker, gui)

    gui = gui.setup()

    assert gui._displayers is not None
    assert len(gui._displayers) == len(gui._tab.children)

    gui.load()
    assert gui._alert.value == '✅'
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
    assert gui.options.source_folder.endswith(os.path.normpath(gui.default_folder))
    assert gui.options.unstack_tabular == gui._unstack_tabular.value
    assert gui.options.pivot_keys_id_names == []


@pytest.mark.long_running
def test_trends_gui_load(person_codecs: md.PersonCodecs):

    gui: wt.RiksProtTrendsGUI = wt.RiksProtTrendsGUI(default_folder=TEST_FOLDER, person_codecs=person_codecs).setup()

    gui.load()

    assert gui._alert.value == "✅"


@pytest.mark.long_running
@pytest.mark.parametrize(
    'temporal_key,normalize,smooth,fill_gaps,pivot_keys_id_names,filter_opts,picked_tokens',
    [
        # ([], None),
        ('decade', False, False, False, ['gender_id'], None, ["information"]),
        ('decade', False, False, False, ['gender_id', 'party_id'], None, ["information"]),
        (
            'decade',
            False,
            False,
            False,
            ['gender_id', 'party_id'],
            pu.PropertyValueMaskingOpts(gender_id=2),
            ["information"],
        ),
    ],
)
def test_trends_gui_transform(
    person_codecs: md.PersonCodecs,
    temporal_key: str,
    normalize: bool,
    smooth: bool,
    fill_gaps: bool,
    pivot_keys_id_names: list[str],
    filter_opts: pu.PropertyValueMaskingOpts,
    picked_tokens: list[str],
):

    class_name: str = "notebooks.riksdagens_protokoll.word_trends.word_trends_gui.RiksProtTrendsGUI"

    opts: wt.ComputeOpts = wt.ComputeOpts(
        fill_gaps=fill_gaps,
        keyness=KeynessMetric.TF,
        normalize=normalize,
        pivot_keys_id_names=pivot_keys_id_names,
        filter_opts=filter_opts,
        smooth=smooth,
        temporal_key=temporal_key,
        top_count=100,
        unstack_tabular=False,
    )

    with mock.patch(f'{class_name}.options', new_callable=mock.PropertyMock) as mocked_options:

        with mock.patch(f'{class_name}.picked_indices', new_callable=mock.PropertyMock) as mocked_picked_indices:

            opts.pivot_keys_id_names = pivot_keys_id_names
            opts.filter_opts = filter_opts
            opts.temporal_key = temporal_key
            pivot_keys_text_names: list[str] = person_codecs.translate_key_names(pivot_keys_id_names)

            mocked_options.return_value = opts

            gui: wt.RiksProtTrendsGUI = wt.RiksProtTrendsGUI(
                default_folder=TEST_FOLDER, person_codecs=person_codecs
            ).setup()

            gui.load(compute=False)
            gui.observe(False)

            picked_indices: list[int] = [gui.trends_data.corpus.token2id[t] for t in picked_tokens]
            mocked_picked_indices.return_value = picked_indices

            gui.transform()

            stacked, unstacked = gui.extract()

            """Stacked data: Temporal key must exist"""
            assert gui.temporal_key in stacked.columns

            """Stacked data: Words should exist as columns in stacked data"""
            assert all(x in stacked.columns for x in picked_tokens)

            """Stacked data: Pivot key's ID columns should be removed"""
            assert set(stacked.columns).intersection(pivot_keys_id_names) == set()

            """Stacked data: Pivot key's NAME columns must exist"""
            assert set(stacked.columns).intersection(pivot_keys_text_names) == set(pivot_keys_text_names)

            """Unstacked data: Temporal key must be index"""
            assert unstacked.index.name == gui.temporal_key

            """Stacked data: No pivot key column is allowed to exists in unstacked data"""
            assert set(unstacked.columns).intersection(pivot_keys_id_names + pivot_keys_text_names) == set()

            """DOCUMENT INDEX"""
            result_columns: set[str] = set(gui.trends_data.transformed_corpus.document_index.columns)

            assert set(pivot_keys_id_names).intersection(set(result_columns)) == set(pivot_keys_id_names)
            assert set(pivot_keys_text_names).intersection(set(result_columns)) == set(pivot_keys_text_names)


@mock.patch('bokeh.plotting.show', lambda *_, **__: None)
@mock.patch('bokeh.io.push_notebook', lambda *_, **__: None)
def test_trends_gui_bugcheck(person_codecs: md.PersonCodecs):
    gui: wt.RiksProtTrendsGUI = wt.RiksProtTrendsGUI(default_folder=TEST_FOLDER, person_codecs=person_codecs).setup()

    gui.load()
    gui.transform()

    gui.invalidate(False)

    gui.plot()
    gui._multi_pivot_keys_picker.value = ['gender', 'office_type']
    gui._temporal_key.value = 'decade'

    words = ['herr']
    gui.observe(False)
    gui._words_picker.options = words
    gui._words_picker.value = words
    for i in range(0, len(gui._displayers)):
        gui._tab.selected_index = i
        gui.plot()


# @pytest.mark.long_running
# @pytest.mark.parametrize(
#     'folder,encoded,expected_columns_added',
#     [
#         (TEST_FOLDER, True, set({'person_id', 'gender_id', 'party_id', 'office_type_id'})),
#         (TEST_FOLDER, False, set({'gender', 'party_abbrev', 'office_type'})),
#     ],
# )
# def test_pos_count_gui_prepare(folder: str, encoded: bool, expected_columns_added: set[str]):

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
#         (TEST_FOLDER, 'decade', True, ['gender_id', 'party_id'], 11),
#     ],
# )
# def test_pos_count_gui_compute_and_plot_with_pivot_keys_and_unstacked(
#     folder: str, temporal_key: str, encoded: bool, pivot_keys: list[str], expected_count: int
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
