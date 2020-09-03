import ipywidgets as widgets
from IPython.display import display
import bokeh.plotting
import bokeh.io
import notebooks.common.distributions_plot as plotter
import numpy as np
import bokeh
import westac.common.curve_fit as cf
import math
import itertools
import pandas as pd
#import qgrid

from pprint import pprint as pp

def compile_multiline_data(x_corpus, indices, smoothers=None):

    xs = x_corpus.xs_years()

    if len(smoothers or []) > 0:
        xs_data = []
        ys_data = []
        for j in indices:
            xs_j = xs
            ys_j = x_corpus.bag_term_matrix[:, j]
            for smoother in smoothers:
                xs_j, ys_j = smoother(xs_j, ys_j)
            xs_data.append(xs_j)
            ys_data.append(ys_j)
    else:
        xs_data = [ xs.tolist() ] * len(indices)
        ys_data = [ x_corpus.bag_term_matrix[:, token_id].tolist() for token_id in indices ]

    data = {
        'xs':    xs_data,
        'ys':    ys_data,
        'label': [ x_corpus.id2token[token_id].upper() for token_id in indices ],
        'color': take(len(indices), itertools.cycle(bokeh.palettes.Category10[10]))
    }
    return data

def compile_year_token_vector_data(x_corpus, indices, *args):

    xs = x_corpus.xs_years()
    data = {
        x_corpus.id2token[token_id]: x_corpus.bag_term_matrix[:, token_id]
            for token_id in indices
    }
    data['year'] = xs

    return data

def setup_plot(container, x_ticks=None, plot_width=1000, plot_height=800, **kwargs):

    data = { 'xs': [ [0] ],  'ys': [ [0] ], 'label': [ "" ], 'color': ['red'] } #, 'token_id': [ 0 ] }

    data_source = bokeh.models.ColumnDataSource(data)

    p = bokeh.plotting.figure(plot_width=plot_width, plot_height=plot_height)
    p.y_range.start = 0
    p.yaxis.axis_label = 'Frequency'
    p.toolbar.autohide = True

    if x_ticks is not None:
        p.xaxis.ticker = x_ticks

    p.xaxis.major_label_orientation = math.pi / 4
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None

    r = p.multi_line(xs='xs', ys='ys', legend_field='label', line_color='color', source=data_source)

    p.legend.location = "top_left"
    p.legend.click_policy="hide"
    p.legend.background_fill_alpha = 0.0

    container.figure = p
    container.handle = bokeh.plotting.show(p, notebook_handle=True)
    container.data_source = data_source

def display_bar_plot(data, **kwargs):

    years = [ str(y) for y in data['year'] ]

    data['year'] = years

    tokens = [ w for w in data.keys() if w != 'year' ]

    source = bokeh.models.ColumnDataSource(data=data)

    max_value = max([ max(data[key]) for key in data if key != 'year']) + 0.005

    p = bokeh.plotting.figure(x_range=years, y_range=(0, max_value), plot_height=400, plot_width=1000, title="Word frequecy by year")

    colors = itertools.islice(itertools.cycle(bokeh.palettes.d3['Category20b'][20]), len(tokens))

    offset = -0.25
    v = []
    for token in tokens:
        w = p.vbar(x=bokeh.transform.dodge('year', offset, range=p.x_range), top=token, width=0.2, source=source, color=next(colors)) #, legend_label=token)
        offset += 0.25
        v.append(w)

    p.x_range.range_padding = 0.04
    p.xaxis.major_label_orientation = math.pi / 4
    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = None
    p.legend.location = "top_right"
    p.legend.orientation = "vertical"

    legend = bokeh.models.Legend(items=[(x, [v[i]]) for i, x in enumerate(tokens)])

    p.add_layout(legend, 'left')

    bokeh.io.show(p)

def display_as_table(data, **kwargs):
    df = pd.DataFrame(data=data)
    df = df[['year']+[x for x in df.columns if x!= 'year']].set_index('year')
    
    display(df)

# def display_as_qgrid(data, **kwargs):
#     df = pd.DataFrame(data=data).set_index('year')
#     qgrid_widget = qgrid.show_grid(df, show_toolbar=False)
#     display(qgrid_widget)

def display_multiline_plot(data, **kwargs):
    container = kwargs['container']
    container.data_source.data.update(data)
    bokeh.io.push_notebook(handle=container.handle)

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, n))

def display_gui(container):

    output_widget       = widgets.Output(layout=widgets.Layout(width='600px', height='200px'))
    words_widget        = widgets.Textarea(description="", rows=4, value="och eller hur", layout=widgets.Layout(width='600px', height='200px'))
    tab_widget          = widgets.Tab()
    tab_widget.children = [ widgets.Output(), widgets.Output(), widgets.Output() ]

    tab_plot_types      = [ "Table", "Line", "Bar" ]
    data_compilers      = [ compile_year_token_vector_data, compile_multiline_data, compile_year_token_vector_data ]
    data_displayers     = [ display_as_table, display_multiline_plot, display_bar_plot ]
    clear_output        = [ True, False, True ]
    _                   = [ tab_widget.set_title(i, x) for i,x in enumerate(tab_plot_types) ]

    smooth    = False
    smoothers = [] if not smooth else [
        # cf.rolling_average_smoother('nearest', 3),
        cf.pchip_spline
    ]

    z_corpus = None
    x_corpus = None

    def update_plot(*args):

        nonlocal z_corpus, x_corpus

        if container.corpus is None:

            with output_widget:
                print("Please load a corpus!")

            return

        if z_corpus is None or not z_corpus is container.corpus:
            with output_widget:
                print("Corpus changed...")
            z_corpus = container.corpus
            x_corpus = z_corpus.todense()

            tab_widget.children[1].clear_output()
            with tab_widget.children[1]:
                setup_plot(container, x_ticks=[ x for x in x_corpus.xs_years() ], plot_width=1000, plot_height=500)

        tokens = '\n'.join(words_widget.value.split()).split()
        index = tab_widget.selected_index
        indices = [ x_corpus.token2id[token] for token in tokens if token in x_corpus.token2id ]

        with output_widget:

            if len(indices) == 0:
                print("Nothing to show!")
                return

            missing_tokens = [ token for token in tokens if token not in x_corpus.token2id ]
            if len(missing_tokens) > 0:
                print("Not in corpus subset: {}".format(' '.join(missing_tokens)))

        if clear_output[index]:
            tab_widget.children[index].clear_output()

        with tab_widget.children[index]:

            data = data_compilers[index](x_corpus, indices)
            container.data = data
            data_displayers[index](data, container=container)

    words_widget.observe(update_plot, names='value')
    tab_widget.observe(update_plot, 'selected_index')

    display(widgets.VBox([
        widgets.HBox([words_widget, output_widget]),
        tab_widget
    ]))


    update_plot()
