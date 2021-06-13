import os

import pandas as pd
import pytest
from penelope import workflows
from penelope.notebook.dtm import ComputeGUI, create_compute_gui

from ..utils import OUTPUT_FOLDER, TEST_DATA_FOLDER

# pylint: disable=protected-access


def monkey_patch(*_, **__):
    ...


def test_compute_gui_compute_dry_run():

    compute_called = False

    def compute_patch(*_, **__):
        nonlocal compute_called
        compute_called = True

    gui: ComputeGUI = create_compute_gui(
        corpus_config="riksdagens-protokoll",
        data_folder=TEST_DATA_FOLDER,
        corpus_folder=TEST_DATA_FOLDER,
        compute_callback=compute_patch,  # type: ignore
        done_callback=monkey_patch,
    )

    gui._corpus_tag.value = 'CERES'
    gui._target_folder.reset(path=OUTPUT_FOLDER)  # , filename='output.txt')
    gui._target_folder._apply_selection()
    gui._corpus_filename.reset(path=TEST_DATA_FOLDER, filename='proto.2files.2sentences.sparv4.csv.zip')
    gui._corpus_filename._apply_selection()
    gui._compute_button.click()
    gui._cli_button.click()

    assert compute_called
    # display(compute_gui.layout())


@pytest.mark.long_running
def test_compute_gui_compute_hot_run():

    corpus_tag = 'CERES'
    gui: ComputeGUI = create_compute_gui(
        corpus_config="riksdagens-protokoll",
        data_folder=TEST_DATA_FOLDER,
        corpus_folder=TEST_DATA_FOLDER,
        compute_callback=workflows.document_term_matrix.compute,
        done_callback=monkey_patch,  # type: ignore
    )

    gui._corpus_tag.value = corpus_tag
    gui._target_folder.reset(path=OUTPUT_FOLDER)  # , filename='output.txt')
    gui._target_folder._apply_selection()
    gui._corpus_filename.reset(path=TEST_DATA_FOLDER, filename='proto.2files.2sentences.sparv4.csv.zip')
    gui._corpus_filename._apply_selection()
    gui._compute_handler(gui._compute_button)

    assert os.path.isfile(os.path.join(OUTPUT_FOLDER, corpus_tag, f'{corpus_tag}_vectorizer_data.pickle'))


def test_allo_allo():

    df = pd.DataFrame(
        data={'token': ['apor', 'kaninen', 'hundar'], 'baseform': ['|apa|xxx|', '|kanin|xxx|', '|hund|xxx|']}
    )
    # ((x[_tok] if x[_lem].strip('|') == '' else x[_lem].strip('|').split('|')[0]) for x in data)
    assert df is not None
