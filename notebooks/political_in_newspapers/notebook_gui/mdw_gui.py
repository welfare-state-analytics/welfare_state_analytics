from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

import ipywidgets
import pandas as pd
from IPython.display import display
from loguru import logger
from penelope.corpus import VectorizedCorpus
from penelope.vendor.textacy import compute_most_discriminating_terms

from notebooks.political_in_newspapers import repository

# pylint: disable=too-many-instance-attributes


def get_group_indicies(
    document_index: pd.DataFrame, period: Sequence[int], pub_ids: Sequence[int] | int = None
) -> pd.Index:

    if isinstance(pub_ids, int):
        pub_ids = list(pub_ids)

    group_index: pd.DataFrame = document_index[document_index.year.between(*period)]

    if len(pub_ids or []) > 0:
        # TODO: #90 Make groupings more generic and move to penelope
        group_index = group_index[group_index.publication_id.isin(pub_ids)]

    return group_index.index


def load_vectorized_corpus(corpus_folder: str, publication_ids) -> VectorizedCorpus:
    logger.info("Loading DTM corpus...")
    source_corpus: repository.SourceCorpus = (
        repository.SourceRepository.load(corpus_folder).to_coo_corpus().slice_by_publications(list(publication_ids))
    )
    token2id: Mapping[str, int] = {v: k for k, v in source_corpus.id2token.items()}
    corpus: VectorizedCorpus = VectorizedCorpus(
        source_corpus.corpus, token2id=token2id, document_index=source_corpus.document_index
    )
    return corpus


@dataclass
class ComputeOpts:
    @dataclass
    class GroupOpts:
        pub_ids: Sequence[int]
        period: Tuple[int, int]

    @dataclass
    class FilterOpts:
        min_df: float
        max_df: float
        top_n_terms: int
        max_n_terms: int

    group1: GroupOpts
    group2: GroupOpts
    filter_opts: FilterOpts


def compile_CLI_opts(opts: ComputeOpts):
    return (
        f"python mdw_runner.py "
        f"--group {','.join([repository.ID2PUB[x] for x in opts.group1.pub_ids])} {opts.group1.period[0]} {opts.group1.period[1]} "
        f"--group {','.join([repository.ID2PUB[x] for x in opts.group2.pub_ids])} {opts.group2.period[0]} {opts.group2.period[1]} "
        f"--top-n-terms {opts.filter_opts.top_n_terms} "
        f"--max-n-terms {opts.filter_opts.max_n_terms} "
        f"--min-df {opts.filter_opts.min_df} "
        f"--max-df {opts.filter_opts.max_df} "
    )


def compute_mdw(corpus: VectorizedCorpus, opts: ComputeOpts) -> pd.DataFrame:
    """Compute most discriminating wors between two non-overlapping subsets."""
    filter_opts: ComputeOpts.FilterOpts = opts.filter_opts

    logger.info(f"Using min DF {filter_opts.min_df} and max DF{filter_opts.max_df}")
    logger.info("Slicing corpus...")

    sliced_corpus: VectorizedCorpus = corpus.slice_by_document_frequency(
        min_df=filter_opts.min_df,
        max_df=filter_opts.max_df,
        max_n_terms=filter_opts.max_n_terms,
    )

    mdw_data: pd.DataFrame = compute_most_discriminating_terms(
        sliced_corpus,
        group1_indices=get_group_indicies(sliced_corpus.document_index, opts.group1.period, opts.group1.pub_ids),
        group2_indices=get_group_indicies(sliced_corpus.document_index, opts.group2.period, opts.group2.pub_ids),
        top_n_terms=filter_opts.top_n_terms,
        max_n_terms=filter_opts.max_n_terms,
    )

    return mdw_data


def display_mwd(mdw_data: pd.DataFrame):
    """Display result"""
    # logger.info(compile_CLI_opts(self))
    if mdw_data is not None:
        display(mdw_data)
    else:
        logger.info("No data for selected groups or periods.")


class MDWGUI:
    def __init__(self, corpus: VectorizedCorpus, document_index: pd.DataFrame):

        self.corpus: VectorizedCorpus = corpus
        self.document_index: pd.DataFrame = document_index

        self.progress = ipywidgets.IntProgress(value=0, min=0, max=5, step=1, description="", layout={'width': "90%"})
        self.top_n_terms = ipywidgets.IntSlider(
            description="#terms",
            min=10,
            max=1000,
            value=200,
            tooltip="The total number of tokens to return for each group",
        )
        self.max_n_terms = ipywidgets.IntSlider(
            description="#top",
            min=1,
            max=2000,
            value=100,
            tooltip="Only consider tokens whose DF is within the top # terms out of all terms",
        )
        self.min_df = ipywidgets.FloatSlider(
            description="Min DF%",
            min=0.0,
            max=100.0,
            value=3.0,
            step=0.1,
            layout={'width': "250px"},
        )
        self.max_df = ipywidgets.FloatSlider(
            description="Max DF%",
            min=0.0,
            max=100.0,
            value=95.0,
            step=0.1,
            layout={'width': "250px"},
        )
        self.period1 = ipywidgets.IntRangeSlider(description="Period", min=0, max=100, layout={'width': "250px"})
        self.period2 = ipywidgets.IntRangeSlider(description="Period", layout={'width': "250px"})
        self.publication_ids1 = ipywidgets.SelectMultiple(
            description="Publication",
            options=dict(repository.PUBLICATION2ID),
            rows=4,
            value=(1,),
            layout=ipywidgets.Layout(width="250px"),
        )
        self.publication_ids2 = ipywidgets.SelectMultiple(
            description="Publication",
            options=dict(repository.PUBLICATION2ID),
            rows=4,
            value=(3,),
            layout=ipywidgets.Layout(width="250px"),
        )
        self.compute = ipywidgets.Button(
            description="Compute", icon="", button_style="Success", layout={'width': "120px"}
        )
        self.output = ipywidgets.Output(layout={"border": "1px solid black"})

    def setup(self) -> "MDWGUI":

        min_year, max_year = (self.document_index.year.min(), self.document_index.year.max())

        self.period1.min, self.period1.max = min_year, max_year
        self.period1.value = (min_year, max_year + 4)

        self.period2.min, self.period2.max = min_year, max_year
        self.period2.value = (max_year + 4, max_year)

        def compute_callback_handler(*_args):
            self.output.clear_output()
            with self.output:
                try:
                    self.compute.disabled = True
                    mdw_data: pd.DataFrame = compute_mdw(self.corpus, self.opts)
                    display_mwd(mdw_data)
                except Exception as ex:
                    logger.error(ex)
                finally:
                    self.compute.disabled = False

        self.compute.on_click(compute_callback_handler)

        return self

    def layout(self) -> ipywidgets.VBox:

        return ipywidgets.VBox(
            children=[
                ipywidgets.HBox(
                    children=[
                        ipywidgets.VBox(children=[self.period1, self.publication_ids1]),
                        ipywidgets.VBox(children=[self.period2, self.publication_ids2]),
                        ipywidgets.VBox(
                            children=[
                                self.top_n_terms,
                                self.max_n_terms,
                                self.min_df,
                                self.max_df,
                            ],
                            layout=ipywidgets.Layout(align_items="flex-end"),
                        ),
                        ipywidgets.VBox(children=[self.compute]),
                    ]
                ),
                self.output,
            ]
        )

    @property
    def opts(self) -> ComputeOpts:
        return ComputeOpts(
            group1=ComputeOpts.GroupOpts(
                pub_ids=self.publication_ids1.value,
                period=self.period1.value,
            ),
            group2=ComputeOpts.GroupOpts(
                pub_ids=self.publication_ids2.value,
                period=self.period2.value,
            ),
            filter_opts=ComputeOpts.FilterOpts(
                min_df=self.min_df.value / 100.0,
                max_df=self.max_df.value / 100.0,
                top_n_terms=self.top_n_terms.value,
                max_n_terms=self.max_n_terms.value,
            ),
        )


def display_gui(corpus: VectorizedCorpus, documents: pd.DataFrame):

    gui = MDWGUI(corpus, documents).setup()

    display(gui.layout())

    return gui
