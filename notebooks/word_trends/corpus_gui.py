import glob
import os
import types

import ipywidgets as widgets
import penelope.corpus.dtm as vectorized_corpus
from IPython.display import display
from penelope.utility import get_logger

logger = get_logger()


def get_corpus_tags(corpus_folder):

    filenames = [os.path.basename(x) for x in glob.glob(os.path.join(corpus_folder, "*_vectorizer_data.pickle"))]
    tags = [x[0 : len(x) - len("_vectorizer_data.pickle")] for x in filenames]
    return tags


def load_vectorized_corpus(corpus_folder, corpus_tag, n_count, n_top, normalize_axis=None, year_range=(1922, 1989)):

    try:
        year_filter = lambda x: year_range[0] <= x['year'] <= year_range[1]
        x_corpus = (
            vectorized_corpus.VectorizedCorpus.load(tag=corpus_tag, folder=corpus_folder)
            .filter(year_filter)
            .group_by_year()
            .slice_by_n_count(n_count)
            .slice_by_n_top(n_top)
        )

        for axis in normalize_axis or []:
            x_corpus = x_corpus.normalize(axis=axis, keep_magnitude=False)

        return x_corpus

    except Exception as ex:
        logger.exception(ex)
        return None


def display_gui(corpus_folder, container=None):

    corpus_tags = get_corpus_tags(corpus_folder)

    if len(corpus_tags) == 0:
        logger.info("Please install at least one vectorized corpus")
        return None

    year_range = [1922, 1989]
    normalize_options = {'None': [], 'Over year': [0], 'Over word': [1], 'Over year and word': [0, 1]}
    gui = types.SimpleNamespace(
        corpus_tag=widgets.Dropdown(
            description="Corpus",
            options=corpus_tags,
            value=corpus_tags[0],
            layout=widgets.Layout(width='420px', color='green'),
        ),
        n_min_count=widgets.IntSlider(
            description="Min count", min=100, max=10000, value=500, layout=widgets.Layout(width='400px', color='green')
        ),
        n_top_count=widgets.IntSlider(
            description="Top count",
            min=100,
            max=1000000,
            value=5000,
            layout=widgets.Layout(width='400px', color='green'),
        ),
        normalize=widgets.Dropdown(
            description="Normalize",
            options=normalize_options,
            value=[0],
            disabled=False,
            layout=widgets.Layout(width='420px', color='green'),
        ),
        load=widgets.Button(description="Load", disabled=False, layout=widgets.Layout(width='80px', color='green')),
        year_range=widgets.IntRangeSlider(
            value=year_range, min=year_range[0], max=year_range[1], step=1, description='Period:'
        ),
        output=widgets.Output(layout=widgets.Layout(width='500px')),
    )

    def load(*args):  # pylint: disable=unused-argument

        gui.output.clear_output()

        with gui.output:

            gui.load.disabled = True
            gui.load.description = 'Wait...'
            gui.corpus_tag.disabled = True

            container.corpus = load_vectorized_corpus(
                corpus_folder,
                gui.corpus_tag.value,
                n_count=gui.n_min_count.value,
                n_top=gui.n_top_count.value,
                normalize_axis=gui.normalize.value,
                year_range=gui.year_range.value,
            )

            gui.load.disabled = False
            gui.corpus_tag.disabled = False
            gui.load.description = 'Load'

        # gui.output.clear_output()

        # with gui.output:

        #    logger.exception("Corpus loaded.")

    gui.load.on_click(load)

    display(
        widgets.HBox(
            [
                widgets.VBox([gui.corpus_tag, gui.normalize, gui.n_min_count, gui.n_top_count, gui.year_range]),
                widgets.VBox([gui.load]),
                gui.output,
            ]
        )
    )

    return gui
