import uuid

import numpy as np
import pandas as pd
import pytest
from penelope.common.curve_fit import pchip_spline, rolling_average_smoother
from penelope.notebook.utility import OutputsTabExt
from penelope.notebook.word_trends.displayers._displayer import (
    MultiLineDataMixin,
    PenelopeBugCheck,
    WordTrendData,
    YearTokenDataMixin,
)
from penelope.notebook.word_trends.loaded_callback import build_layout, update_trend_data

from tests.utils import create_bigger_vectorized_corpus, create_smaller_vectorized_corpus

BIGGER_CORPUS_FILENAME = './tests/test_data/riksdagens-protokoll.1950-1959.ak.sparv4.csv.zip'
OUTPUT_FOLDER = './tests/output'


def xtest_loaded_callback():
    pass


@pytest.fixture
def bigger_corpus():
    corpus_filename = BIGGER_CORPUS_FILENAME
    output_folder = OUTPUT_FOLDER
    output_tag = str(uuid.uuid1())
    count_threshold = 5
    corpus = create_bigger_vectorized_corpus(
        corpus_filename, output_tag=output_tag, output_folder=output_folder, count_threshold=count_threshold
    )
    return corpus


def test_build_layout():

    # FIXME: Use mocks/patches!
    data = WordTrendData(
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

    w: OutputsTabExt = build_layout(data=data)

    assert w is not None and isinstance(w, OutputsTabExt)
    assert len(w.children) == 2
    # assert 4 == len(w.children[1].children)


def test_vectorize_workflow(bigger_corpus):  # pylint: disable=redefined-outer-name

    n_count = 10000

    data = update_trend_data(
        corpus=bigger_corpus,
        corpus_folder=OUTPUT_FOLDER,
        corpus_tag="dummy",
        n_count=n_count,
    )

    assert isinstance(data.goodness_of_fit, pd.DataFrame)
    assert isinstance(data.most_deviating_overview, pd.DataFrame)
    assert isinstance(data.goodness_of_fit, pd.DataFrame)
    assert isinstance(data.most_deviating, pd.DataFrame)


def test_compile_multiline_data_with_no_smoothers():
    corpus = create_smaller_vectorized_corpus().group_by_year()
    indices = [0, 1]
    multiline_data = MultiLineDataMixin().compile(corpus, indices, smoothers=None)

    assert isinstance(multiline_data, dict)
    assert ["A", "B"] == multiline_data['label']
    assert all([(x == y).all() for x, y in zip([[2013, 2014], [2013, 2014]], multiline_data['xs'])])
    assert len(multiline_data['color']) == 2
    assert len(multiline_data['ys']) == 2
    assert all([np.allclose(x, y) for x, y in zip([[4.0, 6.0], [3.0, 7.0]], multiline_data['ys'])])


def test_compile_multiline_data_with_smoothers():
    corpus = create_smaller_vectorized_corpus().group_by_year()
    indices = [0, 1, 2, 3]
    smoothers = [pchip_spline, rolling_average_smoother('nearest', 3)]
    multiline_data = MultiLineDataMixin().compile(corpus, indices, smoothers=smoothers)

    assert isinstance(multiline_data, dict)
    assert ["A", "B", "C", "D"] == multiline_data['label']
    assert len(multiline_data['xs']) == 4
    assert len(multiline_data['ys']) == 4
    assert len(multiline_data['color']) == 4
    assert len(multiline_data['ys']) == 4
    assert len(multiline_data['xs'][0]) > 2  # interpolated coordinates added
    assert len(multiline_data['ys'][0]) == len(multiline_data['xs'][0])  # interpolated coordinates added


def test_compile_year_token_vector_data_when_corpus_is_grouped_by_year_succeeds():
    corpus = create_smaller_vectorized_corpus().group_by_year()
    indices = [0, 1, 2, 3]
    data = YearTokenDataMixin().compile(corpus, indices)
    assert isinstance(data, dict)
    assert all(token in data.keys() for token in ["a", "b", "c", "d"])
    assert len(data["b"]) == 2


def test_compile_year_token_vector_data_when_corpus_is_not_grouped_by_year_fails():
    corpus = create_smaller_vectorized_corpus()
    indices = [0, 1, 2, 3]
    with pytest.raises(PenelopeBugCheck):
        _ = YearTokenDataMixin().compile(corpus, indices)
