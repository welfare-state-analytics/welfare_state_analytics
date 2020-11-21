import warnings
from dataclasses import dataclass

import ipywidgets as widgets
import penelope.notebook.widgets_utils as widgets_utils
import penelope.topic_modelling as topic_modelling
import penelope.utility as utility
from IPython.display import display
from penelope.notebook.topic_modelling import TopicModelContainer, display_topic_trends

import notebooks.political_in_newspapers.corpus_data as corpus_data

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

TEXT_ID = 'topic_share_plot'


@dataclass
class GUI:  # pylint: disable=too-many-instance-attributes

    text = widgets_utils.text_widget(TEXT_ID)
    n_topics = None
    text_id = TEXT_ID

    publication_id = widgets.Dropdown(
        description='Publication',
        options=utility.extend(dict(corpus_data.PUBLICATION2ID), {'(ALLA)': None}),
        value=None,
        layout=widgets.Layout(width="200px"),
    )
    aggregate = widgets.Dropdown(
        description='Aggregate',
        options=[(x['description'], x['key']) for x in topic_modelling.YEARLY_MEAN_COMPUTE_METHODS],
        value='true_mean',
        layout=widgets.Layout(width="200px"),
    )
    normalize = widgets.ToggleButton(description='Normalize', value=True, layout=widgets.Layout(width="120px"))
    topic_id = widgets.IntSlider(description='Topic ID', min=0, max=999, step=1, value=0, continuous_update=False)
    output_format = widgets.Dropdown(
        description='Format', options=['Chart', 'Table'], value='Chart', layout=widgets.Layout(width="200px")
    )
    progress = widgets.IntProgress(min=0, max=4, step=1, value=0)
    output = widgets.Output()
    prev_topic_id = None
    next_topic_id = None

    def layout(self):
        return widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                widgets.HBox([self.prev_topic_id, self.next_topic_id]),
                                self.progress,
                            ]
                        ),
                        widgets.VBox([self.topic_id]),
                        widgets.VBox([self.publication_id]),
                        widgets.VBox([self.aggregate, self.output_format]),
                        widgets.VBox([self.normalize]),
                    ]
                ),
                self.text,
                self.output,
            ]
        )


def display_gui(state: TopicModelContainer, extra_filter=None):  # pylint: disable=unused-argument

    gui = GUI()
    gui.topic_id.max = state.num_topics - 1
    gui.prev_topic_id = widgets_utils.button_with_previous_callback(gui, 'topic_id', state.num_topics)
    gui.next_topic_id = widgets_utils.button_with_next_callback(gui, 'topic_id', state.num_topics)

    def on_topic_change_update_gui(topic_id):

        if gui.n_topics != state.num_topics:
            gui.n_topics = state.num_topics
            gui.topic_id.value = 0
            gui.topic_id.max = state.num_topics - 1

        tokens = topic_modelling.get_topic_title(state.inferred_topics.topic_token_weights, topic_id, n_tokens=200)

        gui.text.value = 'ID {}: {}'.format(topic_id, tokens)

    _current_weight_over_time = dict(publication_id=-1, weights=None)

    def weight_over_time(document_topic_weights, publication_id):
        """Cache weight over time due to the large number of ocuments"""
        if _current_weight_over_time["publication_id"] != publication_id:
            _current_weight_over_time["publication_id"] = publication_id
            df = document_topic_weights
            if publication_id is not None:
                df = df[df.publication_id == publication_id]
            _current_weight_over_time["weights"] = topic_modelling.compute_topic_yearly_means(df).fillna(0)

        return _current_weight_over_time["weights"]

    def update_handler(*_):

        gui.output.clear_output()

        with gui.output:

            on_topic_change_update_gui(gui.topic_id.value)

            weights = weight_over_time(state.inferred_topics.document_topic_weights, gui.publication_id.value)

            display_topic_trends(
                weight_over_time=weights,
                topic_id=gui.topic_id.value,
                year_range=state.inferred_topics.year_period,
                aggregate=gui.aggregate.value,
                normalize=gui.normalize.value,
                output_format=gui.output_format.value,
            )

    gui.topic_id.observe(update_handler, names='value')
    gui.publication_id.observe(update_handler, names='value')
    gui.normalize.observe(update_handler, names='value')
    gui.aggregate.observe(update_handler, names='value')
    gui.output_format.observe(update_handler, names='value')

    display(gui.layout)

    update_handler()
