from ipyaggrid import Grid

css_rules = """
.ag-cell {
    white-space: normal !important;
}
"""

DEFAULT_GRID_OPTIONS = dict(
    enableSorting=True,
    enableFilter=True,
    enableColResize=True,
    enableRangeSelection=False,
    rowSelection='multiple',
    defaultColDef={
        "flex": 1,
        "wrapText": True,
        "autoHeight": True,
        "sortable": True,
        "resizable": True,
        "editable": False,
    },
    rowHeight=80,
    onColumnResized="function onColumnResized(params) {params.api.resetRowHeights();}",
    onColumnVisible="function onColumnResized(params) {params.api.resetRowHeights();}",
)


DEFAULT_GRID_STYLE = dict(
    export_csv=True,
    export_excel=True,
    export_mode="buttons",
    index=True,
    keep_multiindex=False,
    menu={"buttons": [{"name": "Export Grid", "hide": True}]},
    quick_filter=True,
    show_toggle_delete=False,
    show_toggle_edit=False,
    theme="ag-theme-balham",
)


def display_as_grid(df):

    column_defs = [
        {"headerName": "Topic", "field": "topic_id", "maxWidth": 80},
        {"headerName": "Tokens", "field": "tokens"},
        {
            "headerName": "alpha",
            "field": "alpha",
            "cellRenderer": "function(params) { return params.value.toFixed(6); }",
            "maxWidth": 80,
        },
    ]

    grid_options = dict(columnDefs=column_defs, **DEFAULT_GRID_OPTIONS)
    g = Grid(grid_data=df, grid_options=grid_options, **DEFAULT_GRID_STYLE, css_rules=css_rules)
    return g

# FIXME
# from ipyaggrid import Grid
# import ipywidgets
# import pandas as pd

# count_slider = ipywidgets.IntSlider(description="Tokens", min=25, max=200, value=50)

# css_rules = """
# .ag-cell {
#     white-space: normal !important;
# }
# """

# DEFAULT_GRID_OPTIONS = dict(
#     enableSorting=True,
#     enableFilter=True,
#     enableColResize=True,
#     enableRangeSelection=False,
#     rowSelection='multiple',
#     defaultColDef={
#         "flex": 1,
#         "wrapText": True,
#         "autoHeight": True,
#         "sortable": True,
#         "resizable": True,
#         "editable": False,
#     },
#     onColumnResized="function onColumnResized(params) {params.api.resetRowHeights();params.api.setDomLayout('autoHeight');}",
#     onColumnVisible="function onColumnResized(params) {params.api.resetRowHeights();params.api.setDomLayout('autoHeight');}",
# )


# DEFAULT_GRID_STYLE = dict(
#     export_csv=True,
#     export_excel=True,
#     export_mode="buttons",
#     index=True,
#     keep_multiindex=False,
#     menu={"buttons": [{"name": "Export Grid", "hide": True}]},
#     quick_filter=True,
#     show_toggle_delete=False,
#     show_toggle_edit=False,
#     theme="ag-theme-balham",
# )

# def display_as_grid(df_tokens, n_count):
    
#     df = pd.DataFrame(data={'tokens': df_tokens.tokens.apply(lambda x: " ".join(x.split()[:n_count]))}, index=df_tokens.index)
    
#     column_defs = [
#         {"headerName": "Topic", "field": "topic_id", "maxWidth": 80},
#         {"headerName": "Tokens", "field": "tokens"},
#         #{
#         #    "headerName": "alpha",
#         #    "field": "alpha",
#         #    "cellRenderer": "function(params) { return params.value.toFixed(6); }",
#         #    "maxWidth": 80,
#         #},
#     ]

#     grid_options = dict(columnDefs=column_defs, **DEFAULT_GRID_OPTIONS) #, gridAutoHeight=True, pagination=True, paginationAutoPageSize=True)
#     g = Grid(grid_data=df, grid_options=grid_options, **DEFAULT_GRID_STYLE, css_rules=css_rules)#, height=1100)
#     return g

# #pd.options.display.max_colwidth = None
# #pd.set_option('colheader_justify', 'left')

# topics = current_state().inferred_topics.topic_token_overview.copy(deep=True)
# output=ipywidgets.Output() #layout=ipywidgets.Layout(height="1000px"))

# def update_grid(*_):
#     output.clear_output()
#     with output:
#        display(display_as_grid(topics, count_slider.value))
    
# count_slider.observe(update_grid, "value")
# display(ipywidgets.VBox([count_slider, output]))