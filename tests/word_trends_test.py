import pandas as pd
import pytest
from penelope.common.curve_fit import pchip_spline, rolling_average_smoother
from penelope.notebook.utility import OutputsTabExt

from notebooks.word_trends.displayers.data_compilers import (
    PenelopeBugCheck,
    compile_multiline_data,
    compile_year_token_vector_data,
)
from notebooks.word_trends.gui_callback import State, build_layout, update_state
from tests.utils import create_bigger_vectorized_corpus, create_smaller_vectorized_corpus

BIGGER_CORPUS_FILENAME = './tests/test_data/riksdagens-protokoll.1950-1959.ak.sparv4.csv.zip'
OUTPUT_FOLDER = './tests/output'


def xtest_loaded_callback():
    pass


def test_build_layout():

    # FIXME: Use mocks/patches!
    state = State(
        compute_options={},
        corpus=None,
        corpus_folder="",
        corpus_tag="",
        goodness_of_fit=pd.DataFrame(
            data={
                k: []
                for k in [
                    "token",
                    "word_count",
                    "l2_norm",
                    "slope",
                    "chi2_stats",
                    "earth_mover",
                    "kld",
                    "skew",
                    "entropy",
                ]
            }
        ),
        most_deviating=pd.DataFrame(data={'l2_norm_token': [], 'l2_norm': [], 'abs_l2_norm': []}),
        most_deviating_overview=pd.DataFrame(data={'l2_norm_token': [], 'l2_norm': [], 'abs_l2_norm': []}),
    )

    w: OutputsTabExt = build_layout(state=state)

    assert w is not None and isinstance(w, OutputsTabExt)
    assert 2 == len(w.children)
    # assert 4 == len(w.children[1].children)


def test_vectorize_workflow():

    corpus_filename = './tests/test_data/riksdagens-protokoll.1950-1959.ak.sparv4.csv.zip'
    output_folder = "./tests/output"
    output_tag = "xyz"
    count_threshold = 5
    n_count = 10000

    corpus = create_bigger_vectorized_corpus(
        corpus_filename, output_tag=output_tag, output_folder=output_folder, count_threshold=count_threshold
    )

    state = State()

    _ = update_state(
        state,
        corpus=corpus,
        corpus_folder=output_folder,
        corpus_tag=output_tag,
        n_count=n_count,
    )

    assert isinstance(state.goodness_of_fit, pd.DataFrame)
    assert isinstance(state.most_deviating_overview, pd.DataFrame)
    assert isinstance(state.goodness_of_fit, pd.DataFrame)
    assert isinstance(state.most_deviating, pd.DataFrame)


def test_vectorize_workflow2():
    output_tag = "xyz"
    count_threshold = 5
    corpus = create_bigger_vectorized_corpus(
        BIGGER_CORPUS_FILENAME, output_tag=output_tag, output_folder=OUTPUT_FOLDER, count_threshold=count_threshold
    )
    n_count = 10000
    assert corpus is not None

    state = State()

    _ = update_state(
        state,
        corpus=corpus,
        corpus_folder=OUTPUT_FOLDER,
        corpus_tag=output_tag,
        n_count=n_count,
    )

    assert state.goodness_of_fit is not None
    assert state.most_deviating_overview is not None
    assert state.goodness_of_fit is not None
    assert state.most_deviating is not None


def test_compile_multiline_data_with_no_smoothers():
    corpus = create_smaller_vectorized_corpus().group_by_year()
    indices = [0, 1]
    multiline_data = compile_multiline_data(corpus, indices, smoothers=None)

    assert isinstance(multiline_data, dict)
    assert ["A", "B"] == multiline_data['label']
    assert [[2013, 2014], [2013, 2014]] == multiline_data['xs']
    assert 2 == len(multiline_data['color'])
    assert 2 == len(multiline_data['ys'])
    assert [[4.0, 6.0], [3.0, 7.0]] == multiline_data['ys']


def test_compile_multiline_data_with_smoothers():
    corpus = create_smaller_vectorized_corpus().group_by_year()
    indices = [0, 1, 2, 3]
    smoothers = [pchip_spline, rolling_average_smoother('nearest', 3)]
    multiline_data = compile_multiline_data(corpus, indices, smoothers=smoothers)

    assert isinstance(multiline_data, dict)
    assert ["A", "B", "C", "D"] == multiline_data['label']
    assert 4 == len(multiline_data['xs'])
    assert 4 == len(multiline_data['ys'])
    assert 4 == len(multiline_data['color'])
    assert 4 == len(multiline_data['ys'])
    assert 2 < len(multiline_data['xs'][0])  # interpolated coordinates added
    assert len(multiline_data['ys'][0]) == len(multiline_data['xs'][0])  # interpolated coordinates added


def test_compile_year_token_vector_data_when_corpus_is_grouped_by_year_succeeds():
    corpus = create_smaller_vectorized_corpus().group_by_year()
    indices = [0, 1, 2, 3]
    data = compile_year_token_vector_data(corpus, indices)
    assert isinstance(data, dict)
    assert all(token in data.keys() for token in ["A", "B", "C", "D"])
    assert 2 == len(data["B"])


def test_compile_year_token_vector_data_when_corpus_is_not_grouped_by_year_fails():
    corpus = create_smaller_vectorized_corpus()
    indices = [0, 1, 2, 3]
    with pytest.raises(PenelopeBugCheck):
        _ = compile_year_token_vector_data(corpus, indices)
