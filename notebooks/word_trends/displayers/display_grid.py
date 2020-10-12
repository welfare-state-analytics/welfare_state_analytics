import pandas as pd
from ipyaggrid import Grid
from IPython.display import display

from . import data_compilers

NAME = "Grid"

compile = data_compilers.compile_year_token_vector_data  # pylint: disable=redefined-builtin


def setup(container, **kwargs):  # pylint: disable=unused-argument
    pass


def default_column_defs(df):
    column_defs = [
        {
            'headerName': column.title(),
            'field': column,
            # 'rowGroup':False,
            # 'hide':False,
            'cellRenderer': "function(params) { return params.value.toFixed(6); }" if column != 'year' else None,
            # 'type': 'numericColumn'
        }
        for column in df.columns
    ]
    return column_defs


def plot(data, **kwargs):  # pylint: disable=unused-argument

    df = pd.DataFrame(data=data).set_index('year')
    column_defs = default_column_defs(df)
    grid_options = {
        'columnDefs': column_defs,
        'enableSorting': True,
        'enableFilter': True,
        'enableColResize': True,
        'enableRangeSelection': False,
    }

    g = Grid(
        grid_data=df,
        grid_options=grid_options,
        quick_filter=False,
        show_toggle_edit=False,
        export_mode="buttons",
        export_csv=True,
        export_excel=False,
        theme='ag-theme-balham',
        show_toggle_delete=False,
        columns_fit='auto',
        index=True,
        keep_multiindex=False,
        menu={'buttons': [{'name': 'Export Grid', 'hide': True}]},
    )

    display(g)
