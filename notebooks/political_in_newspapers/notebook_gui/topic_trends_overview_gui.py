import types

import ipywidgets as widgets
import penelope.notebook.widgets_utils as widgets_utils
import penelope.topic_modelling as topic_modelling
import penelope.utility as utility
from IPython.display import display
from penelope.notebook.topic_modelling import TopicModelContainer, display_topic_trends_heatmap

import notebooks.political_in_newspapers.corpus_data as corpus_data


def display_gui(state: TopicModelContainer):

    lw = lambda w: widgets.Layout(width=w)

    text_id = 'topic_relevance'

    publications = utility.extend(dict(corpus_data.PUBLICATION2ID), {'(ALLA)': None})
    titles = topic_modelling.get_topic_titles(state.inferred_topics.topic_token_weights, n_tokens=100)
    weighings = [(x['description'], x['key']) for x in topic_modelling.YEARLY_MEAN_COMPUTE_METHODS]

    gui = types.SimpleNamespace(
        text_id=text_id,
        text=widgets_utils.text_widget(text_id),
        flip_axis=widgets.ToggleButton(value=True, description='Flip', icon='', layout=lw("80px")),
        publication_id=widgets.Dropdown(
            description='Publication', options=publications, value=None, layout=widgets.Layout(width="200px")
        ),
        aggregate=widgets.Dropdown(
            description='Aggregate', options=weighings, value='true_mean', layout=widgets.Layout(width="250px")
        ),
        output_format=widgets.Dropdown(
            description='Output', options=['Heatmap', 'Table'], value='Heatmap', layout=lw("180px")
        ),
        output=widgets.Output(),
    )

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
        gui.flip_axis.disabled = True
        gui.flip_axis.description = 'Wait!'
        with gui.output:

            weights = weight_over_time(state.inferred_topics.document_topic_weights, gui.publication_id.value)

            display_topic_trends_heatmap(
                weights,
                titles,
                flip_axis=gui.flip_axis.value,
                aggregate=gui.aggregate.value,
                output_format=gui.output_format.value,
            )
        gui.flip_axis.disabled = False
        gui.flip_axis.description = 'Flip'

    gui.publication_id.observe(update_handler, names='value')
    gui.aggregate.observe(update_handler, names='value')
    gui.output_format.observe(update_handler, names='value')

    display(
        widgets.VBox(
            [
                widgets.HBox([gui.aggregate, gui.publication_id, gui.output_format, gui.flip_axis]),
                widgets.HBox([gui.output]),
                gui.text,
            ]
        )
    )

    update_handler()
