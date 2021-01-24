import types
import warnings
from typing import Any, Dict

import ipywidgets as widgets
import pandas as pd
import penelope.notebook.widgets_utils as widgets_utils
import penelope.topic_modelling as topic_modelling
import penelope.utility as utility
from IPython.display import display
from penelope.corpus import bow_to_text
from penelope.notebook.topic_modelling import TopicModelContainer, filter_document_topic_weights

import notebooks.political_in_newspapers.corpus_data as corpus_data

logger = utility.get_logger()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def reconstitue_texts_for_topic(df: pd.DataFrame, corpus, id2token, n_top=500):

    df['text'] = df.document_id.apply(lambda x: bow_to_text(corpus[x], id2token))
    df['pub'] = df.publication_id.apply(lambda x: corpus_data.ID2PUB[x])
    df = df.drop(['topic_id', 'year', 'publication_id'], axis=1).set_index('document_id')
    df.index.name = 'id'
    return df.sort_values('weight', ascending=False).head(n_top)


def display_texts(
    state: TopicModelContainer,
    filters: Dict[str, Any],
    threshold: float = 0.0,
    output_format: str = 'Table',
    n_top: int = 500,
):

    if state.inferred_model.train_corpus is None:
        print("Corpus is not avaliable. Please store model with corpus!")
        return

    corpus = state.inferred_model.train_corpus.corpus
    id2token = state.inferred_model.train_corpus.id2word
    document_topic_weights = state.inferred_topics.document_topic_weights

    df = filter_document_topic_weights(document_topic_weights, filters=filters, threshold=threshold)

    df = reconstitue_texts_for_topic(df, corpus, id2token, n_top=n_top)

    if len(df) == 0:
        print('No data to display for this topic and theshold')
    elif output_format == 'Table':
        display(df)


def display_gui(state: TopicModelContainer):

    year_min, year_max = state.inferred_topics.year_period
    year_options = [(x, x) for x in range(year_min, year_max + 1)]

    text_id = 'topic_document_text'
    publications = utility.extend(dict(corpus_data.PUBLICATION2ID), {'(ALLA)': None})

    gui = types.SimpleNamespace(
        n_topics=state.num_topics,
        text_id=text_id,
        text=widgets_utils.text_widget(text_id),
        year=widgets.Dropdown(
            description='Year', options=year_options, value=year_options[0][0], layout=widgets.Layout(width="200px")
        ),
        publication_id=widgets.Dropdown(
            description='Publication', options=publications, value=None, layout=widgets.Layout(width="200px")
        ),
        topic_id=widgets.IntSlider(
            description='Topic ID', min=0, max=state.num_topics - 1, step=1, value=0, continuous_update=False
        ),
        n_top=widgets.IntSlider(description='#Docs', min=5, max=500, step=1, value=75),
        threshold=widgets.FloatSlider(
            description='Threshold', min=0.0, max=1.0, step=0.01, value=0.20, continues_update=False
        ),
        output_format=widgets.Dropdown(
            description='Format', options=['Table'], value='Table', layout=widgets.Layout(width="200px")
        ),
        progress=widgets.IntProgress(min=0, max=4, step=1, value=0),
        output=widgets.Output(),
        prev_topic_id=None,
        next_topic_id=None,
    )

    gui.prev_topic_id = widgets_utils.button_with_previous_callback(gui, 'topic_id', state.num_topics)
    gui.next_topic_id = widgets_utils.button_with_next_callback(gui, 'topic_id', state.num_topics)

    def on_topic_change_update_gui(topic_id):

        if gui.n_topics != state.num_topics:
            gui.n_topics = state.num_topics
            gui.topic_id.value = 0
            gui.topic_id.max = state.num_topics - 1

        tokens = topic_modelling.get_topic_title(state.inferred_topics.topic_token_weights, topic_id, n_tokens=200)

        gui.text.value = 'ID {}: {}'.format(topic_id, tokens)

    def update_handler(*_):

        gui.output.clear_output()

        with gui.output:

            on_topic_change_update_gui(gui.topic_id.value)

            display_texts(
                state=state,
                filters=dict(year=gui.year.value, topic_id=gui.topic_id.value, publication_id=gui.publication_id.value),
                threshold=gui.threshold.value,
                n_top=gui.n_top.value,
                output_format=gui.output_format.value,
            )

    gui.topic_id.observe(update_handler, names='value')
    gui.year.observe(update_handler, names='value')
    gui.publication_id.observe(update_handler, names='value')
    gui.threshold.observe(update_handler, names='value')
    gui.n_top.observe(update_handler, names='value')
    gui.output_format.observe(update_handler, names='value')

    display(
        widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                widgets.HBox([gui.prev_topic_id, gui.next_topic_id]),
                                gui.progress,
                            ]
                        ),
                        widgets.VBox([gui.topic_id, gui.threshold, gui.n_top]),
                        widgets.VBox([gui.publication_id, gui.year]),
                        widgets.VBox([gui.output_format]),
                    ]
                ),
                gui.text,
                gui.output,
            ]
        )
    )

    update_handler()
