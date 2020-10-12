import itertools
import math

import bokeh

from . import data_compilers

NAME = "Line"

compile = data_compilers.compile_multiline_data  # pylint: disable=redefined-builtin


def setup(container, **kwargs):

    x_ticks = kwargs.get('x_ticks', None)
    plot_width = kwargs.get('plot_width', 1000)
    plot_height = kwargs.get('plot_height', 800)

    data = {'xs': [[0]], 'ys': [[0]], 'label': [""], 'color': ['red']}  # , 'token_id': [ 0 ] }

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

    _ = p.multi_line(xs='xs', ys='ys', legend_field='label', line_color='color', source=data_source)

    p.legend.location = "top_left"
    p.legend.click_policy = "hide"
    p.legend.background_fill_alpha = 0.0

    container.figure = p
    container.handle = bokeh.plotting.show(p, notebook_handle=True)
    container.data_source = data_source


def plot(data, **_):

    years = [str(y) for y in data['year']]

    data['year'] = years

    tokens = [w for w in data.keys() if w != 'year']

    source = bokeh.models.ColumnDataSource(data=data)

    max_value = max([max(data[key]) for key in data if key != 'year']) + 0.005

    p = bokeh.plotting.figure(
        x_range=years, y_range=(0, max_value), plot_height=400, plot_width=1000, title="Word frequecy by year"
    )

    colors = itertools.islice(itertools.cycle(bokeh.palettes.d3['Category20b'][20]), len(tokens))

    offset = -0.25
    v = []
    for token in tokens:
        w = p.vbar(
            x=bokeh.transform.dodge('year', offset, range=p.x_range),
            top=token,
            width=0.2,
            source=source,
            color=next(colors),
        )  # , legend_label=token)
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
