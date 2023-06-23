import os
from types import MethodType
from typing import List, Set
from unittest.mock import patch

import ipywidgets as widgets
import pandas as pd
import pytest

from notebooks.riksdagens_protokoll.token_counts import pos_statistics_gui as tc
from westac.riksprot.parlaclarin import codecs as md

# pylint: disable=protected-access,redefined-outer-name

TEST_FOLDER: str = 'tests/test_data/riksprot/main/dtm_test.5files'

METADATA_FILENAME: str = "tests/test_data/riksprot/main/riksprot_metadata.db"


@pytest.fixture
def person_codecs():
    return md.PersonCodecs().load(source=METADATA_FILENAME)


@pytest.mark.long_running
@pytest.mark.parametrize('folder,encoded', [(TEST_FOLDER, True), (TEST_FOLDER, False)])
def test_pos_count_gui_load_create(folder: str, encoded: bool, person_codecs: md.PersonCodecs):
    """See also tc.BaseTokenCountGUI test case in penelope"""

    gui = tc.PoSCountGUI(default_folder=folder, encoded=encoded, person_codecs=person_codecs)
    assert gui is not None

    gui = gui.setup(load_data=False)

    assert gui is not None
    assert os.path.isdir(gui.source_folder)
    assert gui.source_folder == gui.opts.source_folder == gui._source_folder.selected_path
    assert len(gui._pos_groups.options) > 0
    assert gui._temporal_key.options == ('decade', 'lustrum', 'year')

    gui.observe(False)

    assert gui.PoS_tag_groups is not None
    assert list(gui.selected_pos_groups) == ['Total']
    assert gui.document_index is None

    assert gui.opts.document_index is None
    assert set(gui._filter_keys_picker.options) == set(['None'] + list(gui.pivot_keys.text_names))
    assert set(gui.pivot_keys_id_names) == set() == set(gui.opts.pivot_keys_id_names)


@pytest.mark.long_running
@pytest.mark.parametrize('folder,encoded', [(TEST_FOLDER, True)])
def test_pos_count_gui_load(folder: str, encoded: bool, person_codecs: md.PersonCodecs):
    gui = tc.PoSCountGUI(default_folder=folder, encoded=encoded, person_codecs=person_codecs).setup(load_data=False)

    gui.load(gui.source_folder)

    assert gui.document_index is not None
    assert len(gui.document_index) > 0
    assert 'Noun' in gui.document_index
    assert all(key in gui.document_index for key in ['who', 'gender_id', 'party_id', 'office_type_id'])

    member_columns: set = set({'person_id', 'pid', 'gender', 'party_abbrev', 'office_type'})

    assert set(gui.document_index.columns).intersection(member_columns) == set()


@pytest.mark.long_running
@pytest.mark.parametrize(
    'folder,encoded,expected_columns_added',
    [
        (TEST_FOLDER, True, {'who', 'gender_id', 'party_id', 'office_type_id'}),
        (TEST_FOLDER, False, {'gender', 'party_abbrev', 'office_type'}),
    ],
)
def test_pos_count_gui_prepare(
    folder: str, encoded: bool, expected_columns_added: Set[str], person_codecs: md.PersonCodecs
):
    gui: tc.PoSCountGUI = (
        tc.PoSCountGUI(default_folder=folder, encoded=encoded, person_codecs=person_codecs)
        .setup(load_data=False)
        .load(folder)
    )

    columns_after_load: set = set(gui.document_index.columns)

    assert set(columns_after_load).intersection(expected_columns_added) == expected_columns_added

    gui.prepare()

    assert gui.document_index is not None
    assert set(gui.document_index.columns).intersection(expected_columns_added) == set(expected_columns_added)


@pytest.mark.long_running
@pytest.mark.parametrize(
    'folder,temporal_key,encoded,expected_count',
    [
        (TEST_FOLDER, 'year', True, 4),
        (TEST_FOLDER, 'decade', True, 3),
        (TEST_FOLDER, 'lustrum', True, 4),
        # (TEST_FOLDER, 'year', False, 101),
    ],
)
def test_pos_count_gui_compute_without_pivot_keys(
    folder: str, temporal_key: str, encoded: bool, expected_count: int, person_codecs: md.PersonCodecs
):
    gui = (
        tc.PoSCountGUI(default_folder=folder, encoded=encoded, person_codecs=person_codecs)
        .setup(load_data=False)
        .load(folder)
        .prepare()
    )
    gui.observe(False)

    gui._temporal_key.value = temporal_key
    data: pd.DataFrame = gui.compute()
    assert len(data) == expected_count
    assert set(data.columns) == set(['Total', temporal_key])


# FIXME: Compute fails if encode is False
@pytest.mark.long_running
@pytest.mark.parametrize(
    'folder,temporal_key,encoded,pivot_keys,expected_stacked_count,expected_unstacked_count',
    [
        (TEST_FOLDER, 'decade', True, [], 3, 3),
        (TEST_FOLDER, 'decade', True, ['gender_id'], 6, 3),
        (TEST_FOLDER, 'decade', True, ['gender_id', 'party_id'], 29, 3),
    ],
)
@patch('penelope.plot.plot_multiline', lambda *_, **__: None)
@patch('penelope.plot.plot_stacked_bar', lambda *_, **__: None)
def test_pos_count_gui_compute_and_plot_with_pivot_keys_and_unstacked(
    folder: str,
    temporal_key: str,
    encoded: bool,
    pivot_keys: List[str],
    expected_stacked_count: int,
    expected_unstacked_count: int,
    person_codecs: md.PersonCodecs,
):
    # expected_pivot_keys = [x.rstrip('_id') for x in pivot_keys]

    gui: tc.PoSCountGUI = (
        tc.PoSCountGUI(default_folder=folder, encoded=encoded, person_codecs=person_codecs)
        .setup(load_data=False)
        .load(folder)
        .prepare()
    )

    gui.observe(False)

    gui._temporal_key.value = temporal_key
    gui._filter_keys_picker.value = [gui.pivot_keys.key_id2key_name.get(x) for x in pivot_keys]

    data: pd.DataFrame = gui.compute()
    assert len(data) == expected_stacked_count

    unstacked_data = gui.unstack_pivot_keys(data)
    assert len(unstacked_data) == expected_unstacked_count

    gui.plot(data)


@pytest.mark.long_running
@patch('penelope.notebook.plot.plot_multiline', lambda *_, **__: None)
@patch('penelope.notebook.plot.plot_stacked_bar', lambda *_, **__: None)
def test_pos_count_gui_with_filter_keys(person_codecs: md.PersonCodecs):
    computed_data: pd.DataFrame = None
    compute_calls: int = 0

    def patch_display(self):
        nonlocal computed_data, compute_calls
        compute_calls += 1
        computed_data = self.compute()

    gender_value_pairs: Set[str] = {"gender: unknown", "gender: man", "gender: woman"}
    gui: tc.PoSCountGUI = (
        tc.PoSCountGUI(default_folder=TEST_FOLDER, person_codecs=person_codecs, encoded=True)
        .setup(load_data=False)
        .load(source=TEST_FOLDER)
        .prepare()
    )

    gui.display = MethodType(patch_display, gui)
    gui.observe(True)

    assert compute_calls == 0
    assert computed_data is None

    gui._temporal_key.value = "decade"
    gui._filter_keys_picker.value = ['gender']

    """No display triggered by _filter_keys_picker"""
    assert compute_calls == 0

    assert set(gui._filter_values_picker.options) == gender_value_pairs
    assert set(gui._filter_values_picker.value) == set()

    gui._filter_values_picker.value = gui._filter_values_picker.options
    assert set(gui.filter_opts.opts.keys()) == {'gender_id'} and set(gui.filter_opts.gender_id) == {0, 1, 2}

    gui._filter_keys_picker.value = ['None']

    assert set(gui._filter_values_picker.options) == set()
    assert set(gui._filter_values_picker.value) == set()

    assert compute_calls == 2
    assert computed_data is not None
    assert len(computed_data) == 3  # Unstacked

    gui._filter_keys_picker.value = ['gender']
    assert compute_calls == 2

    gui._filter_values_picker.value = ['gender: woman']
    assert compute_calls == 3


@pytest.mark.long_running
@pytest.mark.parametrize('folder,encoded', [(TEST_FOLDER, True)])
def test_pos_count_gui_display(folder: str, encoded: bool, person_codecs: md.PersonCodecs):
    gui: tc.PoSCountGUI = (
        tc.PoSCountGUI(default_folder=folder, encoded=encoded, person_codecs=person_codecs)
        .setup(load_data=False)
        .load(folder)
        .prepare()
    )
    gui._tab.display_content = lambda *_, **__: None

    layout = gui.layout()
    assert isinstance(layout, widgets.VBox)

    data: pd.DataFrame = gui.compute()
    gui.plot(data)
    assert gui._status.value == '✔'
