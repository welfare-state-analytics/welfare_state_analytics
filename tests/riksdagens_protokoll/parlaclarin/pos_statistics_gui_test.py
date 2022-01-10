import os
from typing import List, Set

import ipywidgets as widgets
import pandas as pd
import pytest

from notebooks.riksdagens_protokoll.token_counts import pos_statistics_gui as tc

# pylint: disable=protected-access

# FIXME: Create smaller test data
TEST_FOLDER: str = '/data/riksdagen_corpus_data/dtm_1920-2020_v0.3.0.tf20'


@pytest.fixture
def prepared_gui() -> tc.PoSCountGUI:
    return tc.PoSCountGUI(default_folder=TEST_FOLDER, encoded=True).setup(load_data=False).load(TEST_FOLDER).prepare()


@pytest.mark.long_running
@pytest.mark.parametrize(
    'folder,encoded',
    [
        (TEST_FOLDER, True),
        (TEST_FOLDER, False),
    ],
)
def test_pos_count_gui_load_create(folder: str, encoded: bool):
    """See also tc.BaseDTMGUI test case in penelope"""

    gui = tc.PoSCountGUI(default_folder=folder, encoded=encoded)
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
    assert set(gui._pivot_keys.options) == set(['None'] + list(gui.pivot_keys_map.keys()))
    assert set(gui.selected_pivot_keys) == set() == set(gui.opts.pivot_keys)


@pytest.mark.long_running
@pytest.mark.parametrize('folder,encoded', [(TEST_FOLDER, True)])
def test_pos_count_gui_load(folder: str, encoded: bool):

    gui = tc.PoSCountGUI(default_folder=folder, encoded=encoded).setup(load_data=False)

    gui.load(gui.source_folder)

    assert gui.document_index is not None
    assert len(gui.document_index) > 0
    assert 'Noun' in gui.document_index
    assert 'who' in gui.document_index

    member_columns: set = set(
        {'who_id', 'gender_id', 'party_abbrev_id', 'role_type_id', 'gender', 'party_abbrev', 'role_type'}
    )

    assert set(gui.document_index.columns).intersection(member_columns) == set()


@pytest.mark.long_running
@pytest.mark.parametrize(
    'folder,encoded,expected_columns_added',
    [
        (TEST_FOLDER, True, set({'who_id', 'gender_id', 'party_abbrev_id', 'role_type_id'})),
        (TEST_FOLDER, False, set({'gender', 'party_abbrev', 'role_type'})),
    ],
)
def test_pos_count_gui_prepare(folder: str, encoded: bool, expected_columns_added: Set[str]):

    gui = tc.PoSCountGUI(default_folder=folder, encoded=encoded).setup(load_data=False).load(folder)

    columns_after_load: set = set(gui.document_index.columns)

    assert set(columns_after_load).intersection(expected_columns_added) == set()

    gui.prepare()

    assert gui.document_index is not None
    assert set(gui.document_index.columns).intersection(expected_columns_added) == set(expected_columns_added)


@pytest.mark.long_running
@pytest.mark.parametrize(
    'folder,temporal_key,encoded,expected_count',
    [
        (TEST_FOLDER, 'year', True, 101),
        (TEST_FOLDER, 'decade', True, 11),
        (TEST_FOLDER, 'lustrum', True, 21),
        (TEST_FOLDER, 'year', False, 101),
    ],
)
def test_pos_count_gui_compute_without_pivot_keys(folder: str, temporal_key: str, encoded: bool, expected_count: int):

    gui = tc.PoSCountGUI(default_folder=folder, encoded=encoded).setup(load_data=False).load(folder).prepare()
    gui.observe(False)

    gui._temporal_key.value = temporal_key
    data: pd.DataFrame = gui.compute(gui.document_index, gui.opts)
    assert len(data) == expected_count
    assert set(data.columns) == set(['Total', temporal_key])


# FIXME: Compute fails if encode is False
@pytest.mark.long_running
@pytest.mark.parametrize(
    'folder,temporal_key,encoded,pivot_keys,expected_count',
    [
        (TEST_FOLDER, 'decade', True, ['gender_id'], 33),
        # (TEST_FOLDER, 'decade', False, ['gender'], 33),
    ],
)
def test_pos_count_gui_compute_with_pivot_keys(
    folder: str, temporal_key: str, encoded: bool, pivot_keys: List[str], expected_count: int
):

    gui = tc.PoSCountGUI(default_folder=folder, encoded=encoded).setup(load_data=False).load(folder).prepare()
    gui.observe(False)

    gui._temporal_key.value = temporal_key

    """Translates pivot key names to pivot key IDs if encoded = False"""
    # FIXME: GUI only accepts ID options
    gui._pivot_keys.value = tuple()
    gui._pivot_keys.options = pivot_keys
    gui._pivot_keys.value = pivot_keys

    data: pd.DataFrame = gui.compute(gui.document_index, gui.opts)

    """Decode pivot key names since returned data always has decoded values"""
    """Translates pivot key IDs to pivot key names if encoded = True"""
    expected_pivot_keys = [x.rstrip('_id') for x in pivot_keys]

    assert len(data) == expected_count
    assert set(data.columns) == set(['Total', temporal_key] + expected_pivot_keys)


@pytest.mark.long_running
@pytest.mark.parametrize('folder,encoded', [(TEST_FOLDER, True)])
def test_pos_count_gui_display(folder: str, encoded: bool):

    gui = tc.PoSCountGUI(default_folder=folder, encoded=encoded).setup(load_data=False).load(folder).prepare()
    gui.tab.display_content = lambda *_, **__: None

    layout = gui.layout()
    assert isinstance(layout, widgets.VBox)

    gui = gui.display()

    assert gui._status.value == '✔'
