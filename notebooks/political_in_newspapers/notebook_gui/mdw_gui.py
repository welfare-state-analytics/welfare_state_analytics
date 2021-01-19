import logging
import types
from typing import Mapping, Sequence

import ipywidgets
import pandas as pd
from IPython.display import display
from penelope.corpus import VectorizedCorpus
from penelope.vendor.textacy import compute_most_discriminating_terms

import notebooks.political_in_newspapers.corpus_data as corpus_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("westac")


def year_range_group_indicies(
    document_index: pd.DataFrame, period: Sequence[int], pub_ids: Sequence[int] = None
) -> pd.Index:
    """[summary]

    Parameters
    ----------
    document_index : pd.DataFrame
        Documents meta data
    period : Sequence[int]
        Year range for group
    pub_ids : : Sequence[int], optional
        [description], by default None

    Returns
    -------
    pd.Index[int]
        [description]
    """
    assert "year" in document_index.columns

    docs = document_index[document_index.year.between(*period)]

    if isinstance(pub_ids, int):
        pub_ids = list(pub_ids)

    if len(pub_ids or []) > 0:
        # TODO: #90 Make groupings mot generic and move to penelope
        docs = docs[(docs.publication_id.isin(pub_ids))]

    return docs.index


def load_vectorized_corpus(corpus_folder: str, publication_ids) -> VectorizedCorpus:
    logger.info("Loading DTM corpus...")
    bag_term_matrix, document_index, id2token = corpus_data.load_as_dtm2(corpus_folder, list(publication_ids))
    token2id: Mapping[str, int] = {v: k for k, v in id2token.items()}
    v_corpus: VectorizedCorpus = VectorizedCorpus(bag_term_matrix, token2id, document_index)
    return v_corpus


id2pubs = lambda pubs: [corpus_data.PUB2ID[x] for x in pubs]


def compile_opts(gui):
    return (
        f"python mdw_runner.py "
        f"--group {','.join([corpus_data.ID2PUB[x] for x in gui.publication_ids1.value])} {gui.period1.value[0]} {gui.period1.value[1]} "
        f"--group {','.join([corpus_data.ID2PUB[x] for x in gui.publication_ids2.value])} {gui.period2.value[0]} {gui.period2.value[1]} "
        f"--top-n-terms {gui.top_n_terms.value} "
        f"--max-n-terms {gui.max_n_terms.value} "
        f"--min-df {gui.min_df.value/100.0} "
        f"--max-df {gui.max_df.value/100.0} "
    )


def display_gui(v_corpus: VectorizedCorpus, v_documents: pd.DataFrame):

    publications = dict(corpus_data.PUBLICATION2ID)

    lw = lambda w: ipywidgets.Layout(width=w)
    year_span = (v_documents.year.min(), v_documents.year.max())
    gui = types.SimpleNamespace(
        progress=ipywidgets.IntProgress(value=0, min=0, max=5, step=1, description="", layout=lw("90%")),
        top_n_terms=ipywidgets.IntSlider(
            description="#terms",
            min=10,
            max=1000,
            value=200,
            tooltip="The total number of tokens to return for each group",
        ),
        max_n_terms=ipywidgets.IntSlider(
            description="#top",
            min=1,
            max=2000,
            value=100,
            tooltip="Only consider tokens whose DF is within the top # terms out of all terms",
        ),
        min_df=ipywidgets.FloatSlider(
            description="Min DF%",
            min=0.0,
            max=100.0,
            value=3.0,
            step=0.1,
            layout=lw("250px"),
        ),
        max_df=ipywidgets.FloatSlider(
            description="Max DF%",
            min=0.0,
            max=100.0,
            value=95.0,
            step=0.1,
            layout=lw("250px"),
        ),
        period1=ipywidgets.IntRangeSlider(
            description="Period",
            min=year_span[0],
            max=year_span[1],
            value=(v_documents.year.min(), v_documents.year.min() + 4),
            layout=lw("250px"),
        ),
        period2=ipywidgets.IntRangeSlider(
            description="Period",
            min=year_span[0],
            max=year_span[1],
            value=(v_documents.year.max() - 4, v_documents.year.max()),
            layout=lw("250px"),
        ),
        publication_ids1=ipywidgets.SelectMultiple(
            description="Publication",
            options=publications,
            rows=4,
            value=(1,),
            layout=ipywidgets.Layout(width="250px"),
        ),
        publication_ids2=ipywidgets.SelectMultiple(
            description="Publication",
            options=publications,
            rows=4,
            value=(3,),
            layout=ipywidgets.Layout(width="250px"),
        ),
        compute=ipywidgets.Button(description="Compute", icon="", button_style="Success", layout=lw("120px")),
        output=ipywidgets.Output(layout={"border": "1px solid black"}),
    )

    boxes = ipywidgets.VBox(
        [
            ipywidgets.HBox(
                [
                    ipywidgets.VBox([gui.period1, gui.publication_ids1]),
                    ipywidgets.VBox([gui.period2, gui.publication_ids2]),
                    ipywidgets.VBox(
                        [
                            gui.top_n_terms,
                            gui.max_n_terms,
                            gui.min_df,
                            gui.max_df,
                        ],
                        layout=ipywidgets.Layout(align_items="flex-end"),
                    ),
                    ipywidgets.VBox([gui.compute]),
                ]
            ),
            gui.output,
        ]
    )

    display(boxes)

    def compute_callback_handler(*_args):
        gui.output.clear_output()
        with gui.output:
            try:
                gui.compute.disabled = True

                logger.info("Using min DF %s and max DF %s", gui.min_df.value, gui.max_df.value)

                logger.info("Slicing corpus...")

                x_corpus: VectorizedCorpus = v_corpus.slice_by_df(
                    max_df=gui.max_df.value / 100.0,
                    min_df=gui.min_df.value / 100.0,
                    max_n_terms=gui.max_n_terms.value,
                )

                logger.info("Corpus size after DF trim %s x %s.", *x_corpus.data.shape)

                logger.info(compile_opts(gui))

                df = compute_most_discriminating_terms(
                    x_corpus,
                    group1_indices=year_range_group_indicies(
                        x_corpus.document_index,
                        gui.period1.value,
                        gui.publication_ids1.value,
                    ),
                    group2_indices=year_range_group_indicies(
                        x_corpus.document_index,
                        gui.period2.value,
                        gui.publication_ids2.value,
                    ),
                    top_n_terms=gui.top_n_terms.value,
                    max_n_terms=gui.max_n_terms.value,
                )
                if df is not None:
                    display(df)
                else:
                    logger.info("No data for selected groups or periods.")

            except Exception as ex:
                logger.error(ex)
            finally:
                gui.compute.disabled = False

    gui.compute.on_click(compute_callback_handler)
    return gui
