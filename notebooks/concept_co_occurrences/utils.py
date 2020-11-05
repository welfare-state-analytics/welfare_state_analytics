import notebooks.common.ipyaggrid_utility as ipyaggrid_utility


def display_as_grid(df):

    # if os.environ.get('VSCODE_AGENT_FOLDER', None) is not None:
    #    display(df)
    #    return

    column_defs = [
        {
            "headerName": field,
            "field": field,
            "cellRenderer": "function(params) { return params.value.toFixed(6); }"
            if not field.endswith("token")
            else None,
        }
        for field in df.columns
    ]
    grid_options = {'enableSorting': True, 'enableFilter': True, 'enableColResize': True, 'rowSelection': 'multiple'}
    G = ipyaggrid_utility.display_grid(df, column_defs=column_defs, grid_options=grid_options)
    return G
