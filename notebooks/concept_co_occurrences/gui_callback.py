from dataclasses import dataclass
from typing import Dict

import ipywidgets
import pandas as pd
import penelope.common.goodness_of_fit as gof
import penelope.notebook.utility as notebook_utility
import penelope.notebook.word_trend_plot_gui as word_trend_plot_gui
from penelope.co_occurrence.concept_co_occurrence import to_vectorized_corpus
from penelope.corpus import VectorizedCorpus
from penelope.notebook.ipyaggrid_utility import display_grid as display_as_grid
from tqdm import tqdm


@dataclass
class State:
    concept_co_occurrences: pd.DataFrame = None
    concept_co_occurrences_metadata: Dict = None
    corpus: VectorizedCorpus = None
    corpus_tag: str = None
    metadata: Dict = None
    goodness_of_fit: pd.DataFrame = None
    most_deviating_overview: pd.DataFrame = None
    most_deviating: pd.DataFrame = None


state = State()


def loaded_callback(
    output: ipywidgets.Output,
    *,
    concept_co_occurrences: pd.DataFrame = None,
    metadata: Dict = None,
    corpus: VectorizedCorpus = None,
    corpus_tag: str = None,
):
    global state

    pbar = tqdm(total=5)
    output.clear_output()
    with output:

        pbar.update(1)

        state.concept_co_occurrences = concept_co_occurrences
        state.metadata = metadata

        if corpus is None:

            if state.concept_co_occurrences is None:
                raise ValueError("Both corpus and concept_co_occurrences cannot be None")

            pbar.set_description("Vectorizing co-occurrences...")
            state.corpus = to_vectorized_corpus(
                co_occurrences=state.concept_co_occurrences, value_column='value_n_t'
            ).group_by_year()

            state.corpus_tag = corpus_tag
        pbar.update(1)

        pbar.set_description("Computing goodness of fit...")
        state.goodness_of_fit = gof.compute_goddness_of_fits_to_uniform(state.corpus, None, verbose=False)
        pbar.update(1)
        pbar.set_description("Most deviating...")
        state.most_deviating_overview = gof.compile_most_deviating_words(state.goodness_of_fit, n_count=10000)
        pbar.update(1)
        state.most_deviating = gof.get_most_deviating_words(
            state.goodness_of_fit, 'l2_norm', n_count=5000, ascending=False, abs_value=True
        )
        pbar.update(1)
        pbar.set_description("Setting upp GUI...")

        wtf = word_trend_plot_gui.display_gui(state.corpus, tokens=state.most_deviating, display_widgets=False)

        tab_gof = (
            notebook_utility.OutputsTabExt(["GoF", "GoF (abs)", "Plots", "Slopes"])
            .display_fx_result(0, display_as_grid, state.goodness_of_fit)
            .display_fx_result(
                1, display_as_grid, state.most_deviating_overview[['l2_norm_token', 'l2_norm', 'abs_l2_norm']]
            )
            .display_fx_result(2, gof.plot_metrics, state.goodness_of_fit, plot=False, lazy=True)
            .display_fx_result(
                3, gof.plot_slopes, state.corpus, state.most_deviating, "l2_norm", 600, 600, plot=False, lazy=True
            )
        )

        _ = (
            notebook_utility.OutputsTabExt(["Data", "Explore", "Options", "GoF"])
            .display()
            .display_fx_result(0, display_as_grid, concept_co_occurrences)
            .display_content(1, what=wtf, clear=True)
            .display_as_yaml(2, state.metadata, clear=True, width='800px', height='600px')
            .display_content(3, tab_gof, clear=True)
        )
        pbar.update(2)

        pbar.close()
