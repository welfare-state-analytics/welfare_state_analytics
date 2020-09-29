import types

import ipywidgets as widgets
from IPython.display import display

import notebooks.common.topic_trends_overview_display as topic_trends_overview_display
import text_analytic_tools.text_analysis.derived_data_compiler as derived_data_compiler
import text_analytic_tools.text_analysis.topic_weight_over_time as topic_weight_over_time
import text_analytic_tools.utility.widgets as widgets_helper
import westac.common.utility as utility

# from beakerx import *
# from beakerx.object import beakerx
# beakerx.pandas_display_table()

logger = utility.setup_logger()

def display_gui(state):

    lw = lambda w: widgets.Layout(width=w)

    text_id = 'topic_relevance'

    # year_min, year_max = state.compiled_data.year_period

    titles = derived_data_compiler.get_topic_titles(state.compiled_data.topic_token_weights, n_tokens=100)
    weighings = [ (x['description'], x['key']) for x in topic_weight_over_time.METHODS ]

    gui = types.SimpleNamespace(
        text_id=text_id,
        text=widgets_helper.text(text_id),
        flip_axis=widgets.ToggleButton(value=True, description='Flip', icon='', layout=lw("80px")),
        aggregate=widgets.Dropdown(description='Aggregate', options=weighings, value='true_mean', layout=widgets.Layout(width="250px")),
        output_format=widgets.Dropdown(description='Output', options=['Heatmap', 'Table'], value='Heatmap', layout=lw("180px")),
        output=widgets.Output()
    )

    def update_handler(*_):

        gui.output.clear_output()

        with gui.output:

            weights = topic_weight_over_time.compute(state.compiled_data.document_topic_weights).fillna(0)

            topic_trends_overview_display.display_heatmap(
                weights,
                titles,
                flip_axis=gui.flip_axis.value,
                aggregate=gui.aggregate.value,
                output_format=gui.output_format.value
            )

    gui.aggregate.observe(update_handler, names='value')
    gui.output_format.observe(update_handler, names='value')

    display(widgets.VBox([
        widgets.HBox([gui.aggregate, gui.output_format, gui.flip_axis ]),
        widgets.HBox([gui.output]), gui.text
    ]))

    update_handler()
