import warnings
import math
import numpy as np
import ipywidgets as widgets
import bokeh
import bokeh.plotting
import text_analytic_tools.utility.widgets_utility as widgets_utility
import text_analytic_tools.text_analysis.topic_model_utility as topic_model_utility
import text_analytic_tools.utility.widgets as widgets_helper
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

    y_max = max(ys.max(),0.3)

    figopts = utility.extend(dict(title='', toolbar_location="right",y_range = (0.0, y_max+0.05)), figopts)

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
    document_topic_weights,
    topic_id,
    year,
    year_aggregate,
    threshold=0.01,
    output_format='Chart',
    year_column = 'year'
):

    figopts = dict(plot_width=1000, plot_height=400, title='', toolbar_location="right")

    pivot_column = year_column if year is None else None
    value_column = year_aggregate if year is None else 'weight'

    df = document_topic_weights[(document_topic_weights.topic_id == topic_id)]

    if year is not None:
        df = df[(df[year_column] == year)]

    if pivot_column is not None:

        df = df[df['weight'] >= threshold]

        df = df.groupby([pivot_column, 'topic_id']).agg([np.mean, np.max])['weight'].reset_index()
        df.columns = [pivot_column, 'topic_id', 'mean', 'max']

        #df = df[(df[year_aggregate] > threshold)].reset_index()

        category_column = pivot_column
        min_year = document_topic_weights[year_column].min()
        max_year = document_topic_weights[year_column].max()
        figopts['x_range'] = list(map(str, range(min_year, max_year+1)))
    else:
        df = df[(df.weight > threshold)].reset_index()
        category_column = 'treaty'
        figopts['x_range'] = df['treaty'].unique()

    if len(df) == 0:
        print('No data to display for this topic and theshold')
    elif output_format == 'Table':
        print(df)
    else:
        p = plot_topic_trend(df, category_column, value_column, **figopts)
        bokeh.plotting.show(p)

def display_gui(state):

    year_options = [ ('all years', None) ] + [ (x,x) for x in range(state.compiled_data.year_period[0], state.compiled_data.year_period[1] + 1)]

    text_id = 'topic_share_plot'
    publications = utility.extend(dict(corpus_data.PUBLICATION2ID), {'(ALLA)': None})
    gui = widgets_utility.WidgetUtility(
        n_topics=state.num_topics,
        text_id=text_id,
        text=widgets_helper.text(text_id),
        year=widgets.Dropdown(description='Year', options=year_options, value=None, layout=widgets.Layout(width="200px")),
        publication_id=widgets.Dropdown(description='Publication', options=publications, value=None, layout=widgets.Layout(width="200px")),
        year_aggregate=widgets.Dropdown(description='Aggregate', options=['mean', 'max'], value='mean', layout=widgets.Layout(width="200px")),
        threshold=widgets.FloatSlider(description='Threshold', min=0.0, max=0.25, step=0.01, value=0.0, continuous_update=False),
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

        tokens = topic_model_utility.get_topic_title(state.compiled_data.topic_token_weights, topic_id, n_tokens=200)

        gui.text.value = 'ID {}: {}'.format(topic_id, tokens)

    def update_handler(*args):

        gui.output.clear_output()

        with gui.output:

            on_topic_change_update_gui(gui.topic_id.value)

            document_topic_weights = state.compiled_data.document_topic_weights

            if gui.publication_id.value is not None:
                document_topic_weights = document_topic_weights[document_topic_weights.publication_id == gui.publication_id.value]

            display_topic_trend(
                document_topic_weights=document_topic_weights,
                topic_id=gui.topic_id.value,
                year=gui.year.value,
                year_aggregate=gui.year_aggregate.value,
                threshold=gui.threshold.value,
                output_format=gui.output_format.value
            )

    gui.year.disabled = True

    gui.topic_id.observe(update_handler, names='value')
    gui.year.observe(update_handler, names='value')
    gui.publication_id.observe(update_handler, names='value')
    gui.threshold.observe(update_handler, names='value')
    gui.year_aggregate.observe(update_handler, names='value')
    gui.output_format.observe(update_handler, names='value')

    display(widgets.VBox([
        widgets.HBox([
            widgets.VBox([
                widgets.HBox([gui.prev_topic_id, gui.next_topic_id]),
                gui.progress,
            ]),
            widgets.VBox([gui.topic_id, gui.threshold]),
            widgets.VBox([gui.publication_id, gui.year]),
            widgets.VBox([gui.year_aggregate, gui.output_format]),
        ]),
        gui.text,
        gui.output
    ]))

    update_handler()
