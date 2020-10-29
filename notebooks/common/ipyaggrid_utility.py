import numpy as np
import pandas as pd
from ipyaggrid import Grid

DEFAULT_GRID_STYLE = dict(
    quick_filter=False,
    show_toggle_edit=False,
    export_mode="buttons",
    export_csv=True,
    export_excel=False,
    theme="ag-theme-balham",
    show_toggle_delete=False,
    columns_fit="auto",
    index=True,
    keep_multiindex=False,
    menu={"buttons": [{"name": "Export Grid", "hide": True}]},
)

DEFAULT_GRID_OPTIONS = dict(
    enableSorting=True,
    enableFilter=True,
    enableColResize=True,
    enableRangeSelection=False,
)


def default_column_defs(df):
    column_defs = [
        {
            "headerName": column.title(),
            "field": column,
            # 'rowGroup':False,
            # 'hide':False,
            "cellRenderer": "function(params) { return params.value.toFixed(6); }" if isinstance(df.dtypes[i], np.floating) else None,
            # 'type': 'numericColumn'
        }
        for i, column in enumerate(df.columns)
    ]
    return column_defs


def display_grid(data, column_defs=None, grid_options=None, grid_style=None):

    if isinstance(data, dict):
        df = pd.DataFrame(data=data)  # .set_index('year')
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("Data must be dict or pandas.DataFrame")

    column_defs = default_column_defs(df) if column_defs is None else column_defs

    grid_options = dict(
        columnDefs=column_defs,
        **DEFAULT_GRID_OPTIONS,
        **grid_options,
    )

    grid_style = {**DEFAULT_GRID_STYLE, **grid_style}

    g = Grid(grid_data=df, grid_options=grid_options, **grid_style)

    return g


def simple_plot(df):

    g = Grid(
        grid_data=df,
        grid_options=DEFAULT_GRID_OPTIONS,
        **DEFAULT_GRID_STYLE,
    )

    return g
