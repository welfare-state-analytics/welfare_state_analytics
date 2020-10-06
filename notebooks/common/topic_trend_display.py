import math
import warnings

import bokeh
import bokeh.plotting
import numpy as np
import penelope.utility as utility

# from beakerx import *
# from beakerx.object import beakerx
# beakerx.pandas_display_table()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def _plot(df, category_column, value_column, x_label=None, y_label=None, **figopts):

    xs = df[category_column].astype(np.str)
    ys = df[value_column]

    y_max = ys.max()  # max(ys.max(), 0.1)

    figopts = utility.extend(dict(title='', toolbar_location="right", y_range=(0.0, y_max)), figopts)

    p = bokeh.plotting.figure(**figopts)

    _ = p.vbar(x=xs, top=ys, width=0.5, fill_color="#b3de69")

    p.xaxis.major_label_orientation = math.pi / 4
    p.xgrid.grid_line_color = None
    p.xaxis[0].axis_label = (x_label or category_column.title().replace('_', ' ')).title()
    p.yaxis[0].axis_label = (y_label or value_column.title().replace('_', ' ')).title()
    # p.y_range.start = 0.0
    p.y_range.start = 0.0
    p.x_range.range_padding = 0.01

    return p


def display(weight_over_time, topic_id, year_range, aggregate, output_format='Chart', normalize=True):

    figopts = dict(plot_width=1000, plot_height=400, title='', toolbar_location="right")

    df = weight_over_time[(weight_over_time.topic_id == topic_id)]

    min_year, max_year = year_range
    figopts['x_range'] = list(map(str, range(min_year, max_year + 1)))

    if len(df) == 0:
        print('No data to display for this topic and theshold')
    elif output_format == 'Table':
        print(df)
    else:
        p = _plot(df, 'year', aggregate, **figopts)
        bokeh.plotting.show(p)
