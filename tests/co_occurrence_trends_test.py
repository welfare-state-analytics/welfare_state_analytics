import os
from unittest.mock import patch

import pandas as pd
import pytest
from penelope.co_occurrence import load_co_occurrences, to_vectorized_corpus
from penelope.co_occurrence.partitioned import ComputeResult
from penelope.co_occurrence.partitioned import (
    _filter_co_coccurrences_by_global_threshold as filter_co_coccurrences_by_global_threshold,
)
from penelope.common.goodness_of_fit import GoodnessOfFitComputeError
from penelope.corpus import load_document_index

from notebooks.concept_co_occurrences.loaded_callback import CoOccurrenceData, build_layout, compile_data
from tests.utils import TEST_DATA_FOLDER


def simple_co_occurrences():

    document_index = (
        pd.DataFrame(
            data={
                'document_id': [0, 1],
                'year': [1976, 1977],
                'document_name': ["1976", "1977"],
                'filename': ["1976.coo", "1977.coo"],
                'n_raw_tokens': [20, 30],
            }
        )
        .set_index('document_id', drop=False)
        .sort_index()
        .rename_axis('')
    )

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
    return ComputeResult(co_occurrences=co_occurrences, document_index=document_index)


def test_filter_co_coccurrences_by_global_threshold():
    computation = simple_co_occurrences()
    threshold = 4

    result = filter_co_coccurrences_by_global_threshold(computation.co_occurrences, threshold)

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
    index_filename = jj(TEST_DATA_FOLDER, 'partitioned_concept_co_occurrences_document_index.csv')
    compute_options = {}
    co_occurrences = load_co_occurrences(filename)
    document_index = load_document_index(index_filename, key_column=None, sep='\t')
    corpus = to_vectorized_corpus(co_occurrences, document_index, 'value')

    data = compile_data(
        corpus=corpus,
        corpus_folder=None,
        corpus_tag=None,
        co_occurrences=co_occurrences,
        compute_options=compute_options,
    )

    assert data.goodness_of_fit == 42
    assert data.most_deviating == 42
    assert data.most_deviating_overview == 42


@patch('penelope.common.goodness_of_fit.compile_most_deviating_words', generic_patch)
@patch('penelope.common.goodness_of_fit.get_most_deviating_words', generic_patch)
def test_update_state_when_only_one_year_should_fail():
    computation: ComputeResult = simple_co_occurrences()
    data_1976 = computation.co_occurrences.query('year == 1976')
    corpus = to_vectorized_corpus(computation.co_occurrences, computation.document_index, 'value')

    with pytest.raises(GoodnessOfFitComputeError):
        _ = compile_data(
            corpus=corpus,
            corpus_folder=None,
            corpus_tag=None,
            co_occurrences=data_1976.co_occurrences,
            compute_options=dict(),
        )


@patch('penelope.notebook.ipyaggrid_utility.display_grid', generic_patch)
def test_build_layout():
    computation: ComputeResult = simple_co_occurrences()
    corpus = to_vectorized_corpus(computation.co_occurrences, computation.document_index, 'value')
    state = CoOccurrenceData(
        corpus=corpus,
        co_occurrences=computation.co_occurrences,
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


# from penelope.corpus import ExtractTaggedTokensOpts, TokensTransformOpts
# from penelope.workflows import co_occurrence_workflow
# import pathlib
# @pytest.mark.skip("FIXME: improve test fixture")
# def test_co_occurrence_workflow_():

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
#     extract_tokens_opts = ExtractTaggedTokensOpts(
#         pos_includes="|NN|PM|UO|PC|VB|",
#         pos_excludes="|MAD|MID|PAD|",
#         passthrough_tokens=["valv"],
#         lemmatize=True,
#         append_pos=False,
#     )
#     input_filename = "./tests/test_data/tranströmer_corpus_export.csv.zip"
#     output_filename = "./tests/output/valv_concept.csv.zip"
#     context_opts = ContextOpts(concept=["valv"], ignore_concept=True, context_width=2)
#     partition_keys = "year"
#     count_threshold = None
#     filename_field = dict(year=r"prot\_(\d{4}).*")

#     p = pathlib.Path(output_filename)

#     p.unlink(missing_ok=True)

#     _ = co_occurrence_workflow(
#         input_filename=input_filename,
#         output_filename=output_filename,
#         context_opts=context_opts,
#         count_threshold=count_threshold,
#         partition_keys=partition_keys,
#         filename_field=filename_field,
#         extract_tokens_opts=extract_tokens_opts,
#         tokens_transform_opts=tokens_transform_opts,
#         store_vectorized=False,
#     )

#     assert os.path.isfile(output_filename)

#     p.unlink(missing_ok=True)
