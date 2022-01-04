import warnings
from typing import Any, Dict

import pandas as pd
import penelope.notebook.widgets_utils as widgets_utils
import penelope.topic_modelling as topic_modelling
import penelope.utility as utility
from IPython.display import display
from ipywidgets import HTML, Button, Dropdown, FloatSlider, HBox, IntProgress, IntSlider, Output, VBox
from penelope.corpus import bow2text
from penelope.notebook.topic_modelling import TopicModelContainer

import notebooks.political_in_newspapers.repository as repository

TEXT_ID = 'topic_document_text'

# pylint: disable=too-many-instance-attributes)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def reconstitue_texts_for_topic(df: pd.DataFrame, corpus, id2token, n_top=500) -> pd.DataFrame:

    df['text'] = df.document_id.apply(lambda x: bow2text(corpus[x], id2token))
    df['pub'] = df.publication_id.apply(lambda x: repository.ID2PUB[x])
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

    if state.train_corpus is None:
        print("Corpus is not avaliable. Please store model with corpus!")
        return

    corpus = state.trained_model.corpus
    id2token = state.trained_model.id2word

    df: pd.DataFrame = state.inferred_topics.filter_document_topic_weights(filters=filters, threshold=threshold)

    df = reconstitue_texts_for_topic(df, corpus, id2token, n_top=n_top)

    if len(df) == 0:
        print('No data to display for this topic and theshold')
    elif output_format == 'Table':
        display(df)


class TopicDocumentTextGUI:
    def __init__(self, state: TopicModelContainer):

        year_min, year_max = state.inferred_topics.year_period
        year_options = [(x, x) for x in range(year_min, year_max + 1)]

        publications = utility.extend(dict(repository.PUBLICATION2ID), {'(ALLA)': None})

        self.state: TopicModelContainer = state
        self.n_topics: int = state.num_topics
        self.text_id: str = TEXT_ID
        self.text: HTML = widgets_utils.text_widget(TEXT_ID)
        self.year: Dropdown = Dropdown(
            description='Year', options=year_options, value=year_options[0][0], layout=dict(width="200px")
        )
        self.publication_id: Dropdown = Dropdown(
            description='Publication', options=publications, value=None, layout=dict(width="200px")
        )
        self.topic_id: IntSlider = IntSlider(
            description='Topic ID', min=0, max=state.num_topics - 1, step=1, value=0, continuous_update=False
        )
        self.n_top: IntSlider = IntSlider(description='#Docs', min=5, max=500, step=1, value=75)
        self.threshold: FloatSlider = FloatSlider(
            description='Threshold', min=0.0, max=1.0, step=0.01, value=0.20, continues_update=False
        )
        self.output_format: Dropdown = Dropdown(
            description='Format', options=['Table'], value='Table', layout=dict(width="200px")
        )
        self.progress: IntProgress = IntProgress(min=0, max=4, step=1, value=0)
        self.output: Output = Output()
        self.prev_topic_id: Button = widgets_utils.button_with_previous_callback(self, 'topic_id', state.num_topics)
        self.next_topic_id: Button = widgets_utils.button_with_next_callback(self, 'topic_id', state.num_topics)

    def setup(self) -> "TopicDocumentTextGUI":

        self.topic_id.observe(self.update_handler, names='value')
        self.year.observe(self.update_handler, names='value')
        self.publication_id.observe(self.update_handler, names='value')
        self.threshold.observe(self.update_handler, names='value')
        self.n_top.observe(self.update_handler, names='value')
        self.output_format.observe(self.update_handler, names='value')

        return self

    def on_topic_change_update_gui(self, topic_id: int):

        if self.n_topics != self.state.num_topics:
            self.n_topics = self.state.num_topics
            self.topic_id.value = 0
            self.topic_id.max = self.state.num_topics - 1

        tokens = topic_modelling.get_topic_title(self.state.inferred_topics.topic_token_weights, topic_id, n_tokens=200)

        self.text.value = 'ID {}: {}'.format(topic_id, tokens)

    def update_handler(self, *_):

        self.output.clear_output()

        with self.output:

            self.on_topic_change_update_gui(self.topic_id.value)

            display_texts(
                state=self.state,
                filters=dict(
                    year=self.year.value, topic_id=self.topic_id.value, publication_id=self.publication_id.value
                ),
                threshold=self.threshold.value,
                n_top=self.n_top.value,
                output_format=self.output_format.value,
            )

    def layout(self) -> VBox:

        return VBox(
            [
                HBox(
                    [
                        VBox([HBox([self.prev_topic_id, self.next_topic_id]), self.progress]),
                        VBox([self.topic_id, self.threshold, self.n_top]),
                        VBox([self.publication_id, self.year]),
                        VBox([self.output_format]),
                    ]
                ),
                self.text,
                self.output,
            ]
        )


def display_gui(state: TopicModelContainer):

    gui: TopicDocumentTextGUI = TopicDocumentTextGUI(state).setup()

    display(gui.layout())

    gui.update_handler()
