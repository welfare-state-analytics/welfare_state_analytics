import warnings
import math
import types
import numpy as np
import ipywidgets as widgets
import bokeh
import bokeh.plotting
import text_analytic_tools.utility.widgets_utility as widgets_utility
import text_analytic_tools.text_analysis.derived_data_compiler as derived_data_compiler
import text_analytic_tools.utility.widgets as widgets_helper
import text_analytic_tools.text_analysis.topic_weight_over_time as topic_weight_over_time
import westac.common.utility as utility
import westac.notebooks.political_in_newspapers.corpus_data as corpus_data

from IPython.display import display

# from beakerx import *
# from beakerx.object import beakerx
# beakerx.pandas_display_table()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def plot_topic_trend(df, category_column, value_column, x_label=None, y_label=None, **figopts):

    xs = df[category_column].astype(np.str)
    ys = df[value_column]

    y_max = ys.max() # max(ys.max(), 0.1)

    figopts = utility.extend(dict(title='', toolbar_location="right",y_range = (0.0, y_max)), figopts)

    p = bokeh.plotting.figure(**figopts)

    glyph = p.vbar(x=xs, top=ys, width=0.5, fill_color="#b3de69")

    p.xaxis.major_label_orientation = math.pi/4
    p.xgrid.grid_line_color = None
    p.xaxis[0].axis_label = (x_label or category_column.title().replace('_', ' ')).title()
    p.yaxis[0].axis_label = (y_label or value_column.title().replace('_', ' ')).title()
    #p.y_range.start = 0.0
    p.y_range.start = 0.0
    p.x_range.range_padding = 0.01

    return p

def display_topic_trend(
    weight_over_time,
    topic_id,
    year_range,
    aggregate,
    output_format='Chart',
    normalize=True
):

    figopts = dict(plot_width=1000, plot_height=400, title='', toolbar_location="right")

    df = weight_over_time[(weight_over_time.topic_id == topic_id)]

    min_year, max_year = year_range
    figopts['x_range'] = list(map(str, range(min_year, max_year+1)))

    if len(df) == 0:
        print('No data to display for this topic and theshold')
    elif output_format == 'Table':
        print(df)
    else:
        p = plot_topic_trend(df, 'year', aggregate, **figopts)
        bokeh.plotting.show(p)

current_weight_over_time = types.SimpleNamespace(
    publication_id=-1,
    weights=None
)

def get_weight_over_time(document_topic_weights, publication_id):
    global current_weight_over_time
    if current_weight_over_time.publication_id != publication_id:
        current_weight_over_time.publication_id = publication_id
        df = document_topic_weights
        if publication_id is not None:
            df = df[df.publication_id == publication_id]
        print('Recomputing..."')
        current_weight_over_time.weights = topic_weight_over_time.compute_weight_over_time(df).fillna(0)

    return current_weight_over_time.weights

def display_gui(state):

    text_id = 'topic_share_plot'
    publications = utility.extend(dict(corpus_data.PUBLICATION2ID), {'(ALLA)': None})

    weighings = [ (x['description'], x['key']) for x in topic_weight_over_time.METHODS ]

    gui = widgets_utility.WidgetUtility(
        n_topics=state.num_topics,
        text_id=text_id,
        text=widgets_helper.text(text_id),
        publication_id=widgets.Dropdown(description='Publication', options=publications, value=None, layout=widgets.Layout(width="200px")),
        aggregate=widgets.Dropdown(description='Aggregate', options=weighings, value='true_mean', layout=widgets.Layout(width="200px")),
        normalize=widgets.ToggleButton(description='Normalize', value=True, layout=widgets.Layout(width="120px")),
        topic_id=widgets.IntSlider(description='Topic ID', min=0, max=state.num_topics - 1, step=1, value=0, continuous_update=False),
        output_format=widgets.Dropdown(description='Format', options=['Chart', 'Table'], value='Chart', layout=widgets.Layout(width="200px")),
        progress=widgets.IntProgress(min=0, max=4, step=1, value=0),
        output=widgets.Output()
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

    def update_handler(*args):

        gui.output.clear_output()

        with gui.output:

            on_topic_change_update_gui(gui.topic_id.value)

            weights = topic_weight_over_time.get_weight_over_time(
                current_weight_over_time,
                state.compiled_data.document_topic_weights,
                gui.publication_id.value
            )

            display_topic_trend(
                weight_over_time=weights,
                topic_id=gui.topic_id.value,
                year_range=state.compiled_data.year_period,
                aggregate=gui.aggregate.value,
                normalize=gui.normalize.value,
                output_format=gui.output_format.value
            )

    gui.topic_id.observe(update_handler, names='value')
    gui.publication_id.observe(update_handler, names='value')
    gui.normalize.observe(update_handler, names='value')
    gui.aggregate.observe(update_handler, names='value')
    gui.output_format.observe(update_handler, names='value')

    display(widgets.VBox([
        widgets.HBox([
            widgets.VBox([
                widgets.HBox([gui.prev_topic_id, gui.next_topic_id]),
                gui.progress,
            ]),
            widgets.VBox([gui.topic_id]),
            widgets.VBox([gui.publication_id]),
            widgets.VBox([gui.aggregate, gui.output_format]),
            widgets.VBox([gui.normalize]),
        ]),
        gui.text,
        gui.output
    ]))

    update_handler()
