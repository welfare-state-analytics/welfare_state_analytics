import warnings

import ipywidgets as widgets
from IPython.display import display

import text_analytic_tools.text_analysis.derived_data_compiler as derived_data_compiler
import text_analytic_tools.text_analysis.topic_weight_over_time as topic_weight_over_time
import text_analytic_tools.utility.widgets as widgets_helper
import text_analytic_tools.utility.widgets_utility as widgets_utility
import notebooks.common.topic_trend_display as topic_trend_display

# from beakerx import *
# from beakerx.object import beakerx
# beakerx.pandas_display_table()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def display_gui(state):

    text_id = 'topic_share_plot'

    weighings = [(x['description'], x['key']) for x in topic_weight_over_time.METHODS]

    gui = widgets_utility.WidgetUtility(
        n_topics=state.num_topics,
        text_id=text_id,
        text=widgets_helper.text(text_id),
        aggregate=widgets.Dropdown(
            description='Aggregate', options=weighings, value='true_mean', layout=widgets.Layout(width="200px")
        ),
        normalize=widgets.ToggleButton(description='Normalize', value=True, layout=widgets.Layout(width="120px")),
        topic_id=widgets.IntSlider(
            description='Topic ID', min=0, max=state.num_topics - 1, step=1, value=0, continuous_update=False
        ),
        output_format=widgets.Dropdown(
            description='Format', options=['Chart', 'Table'], value='Chart', layout=widgets.Layout(width="200px")
        ),
        progress=widgets.IntProgress(min=0, max=4, step=1, value=0),
        output=widgets.Output(),
    )

    gui.prev_topic_id = gui.create_prev_id_button('topic_id', state.num_topics)
    gui.next_topic_id = gui.create_next_id_button('topic_id', state.num_topics)

    def on_topic_change_update_gui(topic_id):

        if gui.n_topics != state.num_topics:
            gui.n_topics = state.num_topics
            gui.topic_id.value = 0
            gui.topic_id.max = state.num_topics - 1

        tokens = derived_data_compiler.get_topic_title(state.compiled_data.topic_token_weights, topic_id, n_tokens=200)

        gui.text.value = 'ID {}: {}'.format(topic_id, tokens)

    def update_handler(*_):

        gui.output.clear_output()

        with gui.output:

            on_topic_change_update_gui(gui.topic_id.value)

            weights = topic_weight_over_time.compute(state.compiled_data.document_topic_weights)

            topic_trend_display.display(
                weight_over_time=weights,
                topic_id=gui.topic_id.value,
                year_range=state.compiled_data.year_period,
                aggregate=gui.aggregate.value,
                normalize=gui.normalize.value,
                output_format=gui.output_format.value,
            )

    gui.topic_id.observe(update_handler, names='value')
    gui.normalize.observe(update_handler, names='value')
    gui.aggregate.observe(update_handler, names='value')
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
                        widgets.VBox([gui.topic_id]),
                        widgets.VBox([gui.aggregate, gui.output_format]),
                        widgets.VBox([gui.normalize]),
                    ]
                ),
                gui.text,
                gui.output,
            ]
        )
    )

    update_handler()
