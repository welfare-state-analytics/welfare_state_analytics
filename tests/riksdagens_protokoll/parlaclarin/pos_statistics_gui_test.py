import os

import ipywidgets as widgets
import pandas as pd
import pytest
from notebooks.riksdagens_protokoll.token_counts import pos_statistics_gui as tc

# pylint: disable=protected-access


@pytest.mark.long_running
def test_create_gui():
    """See also tc.BaseDTMGUI test case in penelope"""
    gui = tc.PoSCountGUI(default_folder='/data/riksdagen_corpus_data/dtm_1920-2020_v0.3.0.tf20', encoded=True)
    assert gui is not None

    gui = gui.setup()

    assert gui is not None
    assert os.path.isdir(gui.source_folder)
    assert gui.source_folder == gui.opts.source_folder == gui._source_folder.selected_path
    assert len(gui._pos_groups.options) > 0
    assert len(gui._temporal_key.options) > 0

    gui.observe(False)

    gui._smooth.value = True
    assert gui.smooth
    assert gui.opts.smooth

    gui._smooth.value = False
    assert not gui.smooth
    assert not gui.opts.smooth

    gui._normalize.value = True
    assert gui.normalize
    assert gui.opts.normalize

    gui._normalize.value = False
    assert not gui.normalize
    assert not gui.opts.normalize

    assert gui.PoS_tag_groups is not None
    assert len(gui.selected_pos_groups) == 0
    assert gui.document_index is None

    assert gui.opts.document_index is None
    assert gui.opts.grouping_keys == gui.grouping_keys
    assert gui.opts.normalize == gui.normalize

    gui.load(gui.source_folder)
    assert len(gui.document_index) > 0
    assert 'Noun' in gui.document_index.columns

    gui.prepare()
    assert gui.document_index is not None
    assert 'Total' in gui.document_index.columns

    gui._temporal_key.value = 'year'
    data: pd.DataFrame = gui.compute(gui.document_index, gui.opts)
    assert len(data) == 101

    gui._temporal_key.value = 'decade'
    data: pd.DataFrame = gui.compute(gui.document_index, gui.opts)
    assert len(data) == 11

    layout = gui.layout()
    assert isinstance(layout, widgets.VBox)

    gui = gui.display()
    assert gui._status.value == '✔'
