import os
from unittest.mock import patch

import pandas as pd
import pytest
from penelope.co_occurrence import filter_co_coccurrences_by_global_threshold, load_co_occurrences
from penelope.co_occurrence.concept_co_occurrence import to_vectorized_corpus
from penelope.common.goodness_of_fit import GoodnessOfFitComputeError

from notebooks.concept_co_occurrences.gui_callback import State, build_layout, update_state
from tests.utils import TEST_DATA_FOLDER


def simple_co_occurrences():

    co_occurrences = pd.DataFrame(
        [
            ('a', 'b', 2, 2.0 / 6.0, 1976),
            ('a', 'c', 1, 2.0 / 1.0, 1976),
            ('b', 'c', 3, 2.0 / 3.0, 1976),
            ('a', 'b', 4, 4.0 / 11.0, 1977),
            ('a', 'c', 2, 2.0 / 11.0, 1977),
            ('b', 'c', 5, 5.0 / 11.0, 1977),
        ],
        columns=['w1', 'w2', 'value', 'value_n_t', 'year'],
        index=[0, 1, 2, 3, 4, 5],
    )
    return co_occurrences


def test_filter_co_coccurrences_by_global_threshold():
    co_occurrences = simple_co_occurrences()
    threshold = 4

    result = filter_co_coccurrences_by_global_threshold(co_occurrences, threshold)

    expected_result = pd.DataFrame(
        [
            ('a', 'b', 2, 1976),
            ('b', 'c', 3, 1976),
            ('a', 'b', 4, 1977),
            ('b', 'c', 5, 1977),
        ],
        columns=['w1', 'w2', 'value', 'year'],
        index=[0, 1, 4, 5],
    )
    assert result['w1'].tolist() == expected_result['w1'].tolist()
    assert result['w2'].tolist() == expected_result['w2'].tolist()
    assert result['value'].tolist() == expected_result['value'].tolist()
    assert result['year'].tolist() == expected_result['year'].tolist()


jj = os.path.join


def generic_patch(*x, **y):  # pylint: disable=unused-argument
    return 42


@patch('penelope.common.goodness_of_fit.compute_goddness_of_fits_to_uniform', generic_patch)
@patch('penelope.common.goodness_of_fit.compile_most_deviating_words', generic_patch)
@patch('penelope.common.goodness_of_fit.get_most_deviating_words', generic_patch)
def test_update_state_with_corpus_passed_succeeds():

    filename = jj(TEST_DATA_FOLDER, 'partitioned_concept_co_occurrences_data.zip')
    compute_options = {}
    co_occurrences = load_co_occurrences(filename)
    corpus = to_vectorized_corpus(co_occurrences, 'value_n_t')

    state = State()
    _ = update_state(
        state,
        corpus=corpus,
        corpus_folder=None,
        corpus_tag=None,
        concept_co_occurrences=co_occurrences,
        compute_options=compute_options,
    )

    assert state.goodness_of_fit == 42
    assert state.most_deviating == 42
    assert state.most_deviating_overview == 42


@patch('penelope.common.goodness_of_fit.compile_most_deviating_words', generic_patch)
@patch('penelope.common.goodness_of_fit.get_most_deviating_words', generic_patch)
def test_update_state_when_only_one_year_should_fail():
    co_occurrences = simple_co_occurrences().query('year == 1976')
    corpus = to_vectorized_corpus(co_occurrences, 'value_n_t')

    state = State()

    with pytest.raises(GoodnessOfFitComputeError):
        _ = update_state(
            state,
            corpus=corpus,
            corpus_folder=None,
            corpus_tag=None,
            concept_co_occurrences=co_occurrences,
            compute_options=dict(),
        )


@patch('penelope.notebook.ipyaggrid_utility.display_grid', generic_patch)
def test_build_layout():
    co_occurrences = simple_co_occurrences()
    corpus = to_vectorized_corpus(co_occurrences, 'value_n_t')
    state = State(
        corpus=corpus,
        concept_co_occurrences=co_occurrences,
        compute_options={},
        goodness_of_fit=pd.DataFrame(
            data={
                "token": [],
                "word_count": [],
                "l2_norm": [],
                "slope": [],
                "chi2_stats": [],
                "earth_mover": [],
                "kld": [],
                "skew": [],
                "entropy": [],
            },
        ),
        most_deviating=pd.DataFrame(data=dict(l2_norm_token=[], l2_norm=[], abs_l2_norm=[])),
        most_deviating_overview=pd.DataFrame(data=dict(l2_norm_token=[], l2_norm=[], abs_l2_norm=[])),
    )
    gui = build_layout(state)
    assert gui is not None
    assert len(gui.children) == 4


# from penelope.corpus import AnnotationOpts, TokensTransformOpts
# from penelope.workflows import concept_co_occurrence_workflow
# import pathlib
# @pytest.mark.skip("FIXME: improve test fixture")
# def test_concept_co_occurrence_workflow_():

#     tokens_transform_opts = TokensTransformOpts(
#         **{
#             "only_alphabetic": False,
#             "only_any_alphanumeric": True,
#             "to_lower": True,
#             "to_upper": False,
#             "min_len": 1,
#             "max_len": None,
#             "remove_accents": False,
#             "remove_stopwords": True,
#             "stopwords": None,
#             "extra_stopwords": None,
#             "language": "swedish",
#             "keep_numerals": False,
#             "keep_symbols": False,
#         }
#     )
#     annotation_opts = AnnotationOpts(
#         pos_includes="|NN|PM|UO|PC|VB|",
#         pos_excludes="|MAD|MID|PAD|",
#         passthrough_tokens=["valv"],
#         lemmatize=True,
#         append_pos=False,
#     )
#     input_filename = "./tests/test_data/tranströmer_corpus_export.csv.zip"
#     output_filename = "./tests/output/valv_concept.csv.zip"
#     concept_opts = ConceptContextOpts(concept=["valv"], ignore_concept=True, context_width=2)
#     partition_keys = "year"
#     count_threshold = None
#     filename_field = dict(year=r"prot\_(\d{4}).*")

#     p = pathlib.Path(output_filename)

#     p.unlink(missing_ok=True)

#     _ = concept_co_occurrence_workflow(
#         input_filename=input_filename,
#         output_filename=output_filename,
#         concept_opts=concept_opts,
#         count_threshold=count_threshold,
#         partition_keys=partition_keys,
#         filename_field=filename_field,
#         annotation_opts=annotation_opts,
#         tokens_transform_opts=tokens_transform_opts,
#         store_vectorized=False,
#     )

#     assert os.path.isfile(output_filename)

#     p.unlink(missing_ok=True)