import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping

import ipywidgets
import pandas as pd
import penelope.common.goodness_of_fit as gof
import penelope.notebook.utility as notebook_utility
import penelope.notebook.word_trends as word_trends
from penelope.co_occurrence import to_vectorized_corpus
from penelope.common.goodness_of_fit import GoodnessOfFitComputeError
from penelope.corpus import VectorizedCorpus
from penelope.notebook.ipyaggrid_utility import display_grid
from penelope.utility import getLogger

logger = getLogger()

# FIXME: Consolidate with penelope!!!!


@dataclass
class CoOccurrenceData:  # pylint: disable=too-many-instance-attributes
    corpus: VectorizedCorpus = None
    corpus_folder: str = None
    corpus_tag: str = None
    co_occurrences: pd.DataFrame = None
    compute_options: Dict = None
    co_occurrence_metadata: Dict = None
    goodness_of_fit: pd.DataFrame = None
    most_deviating_overview: pd.DataFrame = None
    most_deviating: pd.DataFrame = None


def compile_data(
    *,
    corpus: VectorizedCorpus = None,
    corpus_folder: str = None,
    corpus_tag: str = None,
    n_count: int = 10000,
    co_occurrences: pd.DataFrame = None,
    compute_options: Mapping[str, Any] = None,
) -> CoOccurrenceData:

    data = CoOccurrenceData()

    data.corpus = corpus
    data.corpus_folder = corpus_folder
    data.corpus_tag = corpus_tag
    data.co_occurrences = co_occurrences
    data.compute_options = compute_options

    if corpus is None:

        if data.co_occurrences is None:
            raise ValueError("Both corpus and co_occurrences cannot be None")

        data.corpus = to_vectorized_corpus(co_occurrences=data.co_occurrences, value_column='value_n_t').group_by_year()

    data.goodness_of_fit = gof.compute_goddness_of_fits_to_uniform(data.corpus, None, verbose=False)
    data.most_deviating_overview = gof.compile_most_deviating_words(data.goodness_of_fit, n_count=n_count)
    data.most_deviating = gof.get_most_deviating_words(
        data.goodness_of_fit, 'l2_norm', n_count=n_count, ascending=False, abs_value=True
    )

    return data


def build_layout(
    data: CoOccurrenceData,  # pylint: disable=redefined-outer-name
):
    trends_with_pick_gui: word_trends.TrendsWithPickTokensGUI = word_trends.TrendsWithPickTokensGUI.create(
        data.corpus, tokens=data.most_deviating
    )

    tab_gof = (
        notebook_utility.OutputsTabExt(["GoF", "GoF (abs)", "Plots", "Slopes"], layout={'width': '98%'})
        .display_fx_result(0, display_grid, data.goodness_of_fit)
        .display_fx_result(1, display_grid, data.most_deviating_overview[['l2_norm_token', 'l2_norm', 'abs_l2_norm']])
        .display_fx_result(2, gof.plot_metrics, data.goodness_of_fit, plot=False, lazy=True)
        .display_fx_result(
            3, gof.plot_slopes, data.corpus, data.most_deviating, "l2_norm", 600, 600, plot=False, lazy=True
        )
    )

    layout = (
        notebook_utility.OutputsTabExt(["Data", "Explore", "Options", "GoF"], layout={'width': '98%'})
        .display_fx_result(0, display_grid, data.co_occurrences)
        .display_content(1, what=trends_with_pick_gui.layout(), clear=True)
        .display_as_yaml(2, data.compute_options, clear=True, width='800px', height='600px')
        .display_content(3, tab_gof, clear=True)
    )

    return layout


def loaded_callback(
    *,
    output: ipywidgets.Output,
    corpus: VectorizedCorpus = None,
    corpus_folder: str = None,
    corpus_tag: str = None,
    n_count: int = 10000,
    **kwargs,
):

    with output:
        try:

            output.clear_output()

            if os.environ.get('VSCODE_LOGS', None) is not None:
                logger.error("bug-check: vscode detected, aborting plot...")
                return

            data = compile_data(
                corpus=corpus,
                corpus_folder=corpus_folder,
                corpus_tag=corpus_tag,
                n_count=n_count,
                **kwargs,
            )

            build_layout(data).display()

        except GoodnessOfFitComputeError as ex:
            logger.info(f"Unable to compute GoF: {str(ex)}")
        except Exception as ex:
            logger.exception(ex)
