import types

import ipywidgets as widgets
import text_analytic_tools.text_analysis.derived_data_compiler as derived_data_compiler
import text_analytic_tools.text_analysis.topic_weight_over_time as topic_weight_over_time
import text_analytic_tools.utility.widgets as widgets_helper
import westac.common.utility as utility
from IPython.display import display

import notebooks.common.topic_trends_overview_display as topic_trends_overview_display
import notebooks.political_in_newspapers.corpus_data as corpus_data

# from beakerx import *
# from beakerx.object import beakerx
# beakerx.pandas_display_table()


def display_gui(state):

    lw = lambda w: widgets.Layout(width=w)

    text_id = 'topic_relevance'

    publications = utility.extend(dict(corpus_data.PUBLICATION2ID), {'(ALLA)': None})
    titles = derived_data_compiler.get_topic_titles(state.compiled_data.topic_token_weights, n_tokens=100)
    weighings = [(x['description'], x['key']) for x in topic_weight_over_time.METHODS]

    gui = types.SimpleNamespace(
        text_id=text_id,
        text=widgets_helper.text(text_id),
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
            _current_weight_over_time["weights"] = topic_weight_over_time.compute(df).fillna(0)

        return _current_weight_over_time["weights"]

    def update_handler(*args):

        gui.output.clear_output()
        gui.flip_axis.disabled = True
        gui.flip_axis.description = 'Wait!'
        with gui.output:

            weights = weight_over_time(state.compiled_data.document_topic_weights, gui.publication_id.value)

            topic_trends_overview_display.display_heatmap(
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
