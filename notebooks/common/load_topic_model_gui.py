import types
import warnings
from os.path import join as jj
from typing import Any, Dict, List

import ipywidgets as widgets
import numpy as np
import penelope.topic_modelling as topic_modelling
import penelope.utility as utility
from ipyaggrid.grid import Grid
from IPython.display import display
from penelope.notebook.ipyaggrid_utility import DEFAULT_GRID_OPTIONS, DEFAULT_GRID_STYLE

from notebooks.common.model_container import TopicModelContainer
from notebooks.political_in_newspapers.corpus_data import extend_with_document_info

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = utility.get_logger()


def display_as_grid(df):

    column_defs = [
        {"headerName": "Topic", "field": "topic_id"},
        {"headerName": "Tokens", "field": "tokens"},
        {
            "headerName": "alpha",
            "field": "alpha",
            "cellRenderer": "function(params) { return params.value.toFixed(6); }",
        },
    ]

    grid_options = dict(columnDefs=column_defs, **DEFAULT_GRID_OPTIONS)
    g = Grid(grid_data=df, grid_options=grid_options, **DEFAULT_GRID_STYLE)
    display(g)


# FIXME: #94 Column 'year' is missing in `documents` in model metadata (InferredTopicsData)
def temporary_bug_fixupdate_documents(inferred_topics):

    logger.info("applying temporary bug fix of missing year in documents...done!")
    documents = inferred_topics.documents
    document_topic_weights = inferred_topics.document_topic_weights

    if "year" not in documents.columns:
        documents["year"] = documents.filename.str.split("_").apply(lambda x: x[1]).astype(np.int)

    if "year" not in document_topic_weights.columns:
        document_topic_weights = extend_with_document_info(document_topic_weights, documents)

    inferred_topics.documents = documents
    inferred_topics.document_topic_weights = document_topic_weights

    assert "year" in inferred_topics.documents.columns
    assert "year" in inferred_topics.document_topic_weights.columns

    return inferred_topics


def load_model(
    corpus_folder: str,
    state: TopicModelContainer,
    model_name: str,
    model_infos: List[Dict[str, Any]] = None,
):

    model_infos = model_infos or topic_modelling.find_models(corpus_folder)
    model_info = next(x for x in model_infos if x["name"] == model_name)

    inferred_model = topic_modelling.load_model(model_info["folder"])
    inferred_topics = topic_modelling.InferredTopicsData.load(jj(corpus_folder, model_info["name"]))

    inferred_topics = temporary_bug_fixupdate_documents(inferred_topics)

    state.set_data(inferred_model, inferred_topics)

    topics = inferred_topics.topic_token_overview
    # topics.style.set_properties(**{'text-align': 'left'}).set_table_styles(
    #     [dict(selector='td', props=[('text-align', 'left')])]
    # )

    display_as_grid(topics)


def display_gui(corpus_folder: str, state: TopicModelContainer):

    model_infos = topic_modelling.find_models(corpus_folder)
    model_names = list(x["name"] for x in model_infos)

    gui = types.SimpleNamespace(
        model_name=widgets.Dropdown(description="Model", options=model_names, layout=widgets.Layout(width="40%")),
        load=widgets.Button(
            description="Load",
            button_style="Success",
            layout=widgets.Layout(width="80px"),
        ),
        output=widgets.Output(),
    )

    def load_handler(*_):  # pylint: disable=unused-argument
        gui.output.clear_output()
        try:
            gui.load.disabled = True
            with gui.output:
                if gui.model_name.value is None:
                    print("Please specify which model to load.")
                    return
                load_model(corpus_folder, state, gui.model_name.value, model_infos)
        finally:
            gui.load.disabled = False

    gui.load.on_click(load_handler)

    display(widgets.VBox([widgets.HBox([gui.model_name, gui.load]), widgets.VBox([gui.output])]))
