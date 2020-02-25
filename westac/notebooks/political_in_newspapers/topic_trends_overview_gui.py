import warnings
import types
import numpy as np
import ipywidgets as widgets
import bokeh
import bokeh.plotting
import bokeh.transform
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

logger = utility.setup_logger()

def setup_glyph_coloring(df, color_high=0.3):

    #colors = list(reversed(bokeh.palettes.Greens[9]))
    colors = ['#ffffff', '#f7fcf5', '#e5f5e0', '#c7e9c0', '#a1d99b', '#74c476', '#41ab5d', '#238b45', '#006d2c', '#00441b']
    mapper = bokeh.models.LinearColorMapper(palette=colors, low=0.0, high=color_high)
    color_transform = bokeh.transform.transform('weight', mapper)
    color_bar = bokeh.models.ColorBar(color_mapper=mapper, location=(0, 0),
                         ticker=bokeh.models.BasicTicker(desired_num_ticks=len(colors)),
                         formatter=bokeh.models.PrintfTickFormatter(format=" %5.2f"))
    return color_transform, color_bar

def compute_int_range_categories(values):
    categories = values.unique()
    if all(map(utility.isint, categories)):
        categories = sorted(list(map(int, categories)))
        return list(map(str, categories))
    else:
        return sorted(list(categories))

HEATMAP_FIGOPTS = dict(title="Topic heatmap", toolbar_location="right",  x_axis_location="above", plot_width=1000)

def plot_topic_relevance_by_year(df, xs, ys, flip_axis, titles, text_id, **figopts):

    line_height = 7
    if flip_axis is True:
        xs, ys = ys, xs
        line_height = 10

    x_range = compute_int_range_categories(df[xs])
    y_range = compute_int_range_categories(df[ys])

    color_high = max(df.weight.max(), 0.3)
    color_transform, color_bar = setup_glyph_coloring(df, color_high=color_high)

    source = bokeh.models.ColumnDataSource(df)

    if x_range is not None:
        figopts['x_range'] = x_range

    if y_range is not None:
        figopts['y_range'] = y_range
        figopts['plot_height'] = max(len(y_range) * line_height, 500)

    p = bokeh.plotting.figure(**figopts)

    args = dict(x=xs, y=ys, source=source, alpha=1.0, hover_color='red')

    cr = p.rect(width=1, height=1, line_color=None, fill_color=color_transform, **args)

    p.x_range.range_padding = 0
    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "8pt"
    p.axis.major_label_standoff = 0
    p.xaxis.major_label_orientation = 1.0
    p.add_layout(color_bar, 'right')

    p.add_tools(bokeh.models.HoverTool(tooltips=None, callback=widgets_utility.WidgetUtility.glyph_hover_callback(
        source, 'topic_id', titles.index, titles, text_id), renderers=[cr]))

    return p

def display_heatmap(document_topic_weights, titles, key='max', flip_axis=False, glyph='Circle', year=None, aggregate=None, output_format=None):
    try:

        category_column = 'year' if year is None else 'document_id'

        df = document_topic_weights.copy()

        if year is not None:
            df = df[(df.year == year)]

        if year is None:

            ''' Display aggregate value grouped by year  '''
            df = df.groupby(['year', 'topic_id']).agg([np.mean, np.max])['weight'].reset_index()
            df.columns = ['year', 'topic_id', 'mean', 'max']
            df['weight'] = df[aggregate]

        else:
            assert False, "Not implemented"
            ''' Display individual documents for selected year  '''
            # df[category_column] = df.index

        df[category_column] = df[category_column].astype(str)
        df['topic_id'] = df.topic_id.astype(str)

        if output_format.lower() == 'heatmap':

            p = plot_topic_relevance_by_year(
                df,
                xs=category_column,
                ys='topic_id',
                flip_axis=flip_axis,
                titles=titles,
                text_id='topic_relevance',
                **HEATMAP_FIGOPTS)

            bokeh.plotting.show(p)

        else:
            display(df)

    except Exception as ex:
        raise
        # logger.error(ex)

def display_gui(state):

    lw = lambda w: widgets.Layout(width=w)

    text_id = 'topic_relevance'

    year_min, year_max = state.compiled_data.year_period
    year_options = [ ('all years', None) ] + [ (x,x) for x in range(year_min, year_max + 1)]

    publications = utility.extend(dict(corpus_data.PUBLICATION2ID), {'(ALLA)': None})
    titles = topic_model_utility.get_topic_titles(state.compiled_data.topic_token_weights, n_tokens=100)

    gui = types.SimpleNamespace(
        text_id=text_id,
        text=widgets_helper.text(text_id),
        flip_axis=widgets.ToggleButton(value=True, description='Flip', icon='', layout=lw("80px")),
        year=widgets.Dropdown(description='Year', options=year_options, value=None, layout=lw("160px")),
        publication_id=widgets.Dropdown(description='Publication', options=publications, value=None, layout=widgets.Layout(width="200px")),
        aggregate=widgets.Dropdown(description='Aggregate', options=['mean', 'max'], value='max', layout=lw("160px")),
        output_format=widgets.Dropdown(description='Output', options=['Heatmap', 'Table'], value='Heatmap', layout=lw("180px")),
        output=widgets.Output()
    )

    def update_handler(*args):

        gui.output.clear_output()

        with gui.output:

            document_topic_weights = state.compiled_data.document_topic_weights

            if gui.publication_id.value is not None:
                document_topic_weights = document_topic_weights[document_topic_weights.publication_id == gui.publication_id.value]

            display_heatmap(
                document_topic_weights,
                titles,
                flip_axis=gui.flip_axis.value,
                year=gui.year.value,
                aggregate=gui.aggregate.value,
                output_format=gui.output_format.value
            )

    gui.year.disabled = True

    gui.year.observe(update_handler, names='value')
    gui.publication_id.observe(update_handler, names='value')
    gui.aggregate.observe(update_handler, names='value')
    gui.output_format.observe(update_handler, names='value')

    display(widgets.VBox([
        widgets.HBox([gui.year, gui.aggregate, gui.publication_id, gui.output_format, gui.flip_axis ]),
        widgets.HBox([gui.output]), gui.text
    ]))

    update_handler()
