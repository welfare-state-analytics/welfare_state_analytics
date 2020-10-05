
# Visualize topic co-occurrence
import types
import ipywidgets as widgets
from IPython.display import display

import notebooks.common.topic_topic_network_display as topic_topic_network_display
import text_analytic_tools.text_analysis.derived_data_compiler as derived_data_compiler
import text_analytic_tools.text_analysis.utility as tmutility
import text_analytic_tools.utility.widgets_utility as widgets_utility

#bokeh.plotting.output_notebook()

def display_gui(state):

    lw = lambda w: widgets.Layout(width=w)
    n_topics = state.num_topics

    text_id = 'nx_topic_topic'
    layout_options = [ 'Circular', 'Kamada-Kawai', 'Fruchterman-Reingold']
    output_options = { 'Network': 'network', 'Table': 'table', 'Excel': 'excel', 'CSV': 'csv' }
    ignore_options = [('', None)] + [ ('Topic #'+str(i), i) for i in range(0, n_topics) ]
    year_min, year_max = state.compiled_data.year_period

    topic_proportions = tmutility.compute_topic_proportions(state.compiled_data.document_topic_weights, state.compiled_data.documents.n_terms.values)
    titles = derived_data_compiler.get_topic_titles(state.compiled_data.topic_token_weights)

    gui = types.SimpleNamespace(
        n_topics=n_topics,
        text=widgets_utility.wf.create_text_widget(text_id),
        period=widgets.IntRangeSlider(description='Time', min=year_min, max=year_max, step=1, value=(year_min, year_min+5), continues_update=False),
        scale=widgets.FloatSlider(description='Scale', min=0.0, max=1.0, step=0.01, value=0.1, continues_update=False),
        n_docs=widgets.IntSlider(description='n-docs', min=10, max=100, step=1, value=1, continues_update=False),
        threshold=widgets.FloatSlider(description='Threshold', min=0.01, max=1.0, step=0.01, value=0.20, continues_update=False),
        output_format=widgets.Dropdown(description='Output', options=output_options, value='network', layout=lw('200px')),
        layout=widgets.Dropdown(description='Layout', options=layout_options, value='Fruchterman-Reingold', layout=lw('250px')),
        progress=widgets.IntProgress(min=0, max=4, step=1, value=0, layout=widgets.Layout(width="99%")),
        ignores=widgets.SelectMultiple(description='Ignore', options=ignore_options, value=[], rows=5, layout=lw('250px')),
        node_range=widgets.IntRangeSlider(description='Node size', min=10, max=100, step=1, value=(20, 60), continues_update=False),
        edge_range=widgets.IntRangeSlider(description='Edge size', min=1, max=20, step=1, value=(2, 6), continues_update=False),
        output=widgets.Output()
    )

    def tick(x=None):
        gui.progress.value = gui.progress.value + 1 if x is None else x

    def compute_handler(*_):

        gui.output.clear_output()
        tick(1)
        with gui.output:

            topic_topic_network_display.display_topic_topic_network(
                c_data=state.compiled_data,
                filters=dict(),
                period=gui.period.value,
                ignores=gui.ignores.value,
                threshold=gui.threshold.value,
                layout=gui.layout.value,
                n_docs=gui.n_docs.value,
                scale=gui.scale.value,
                node_range=gui.node_range.value,
                edge_range=gui.edge_range.value,
                output_format=gui.output_format.value,
                text_id=text_id,
                titles=titles,
                topic_proportions=topic_proportions
            )
        tick(0)

    gui.threshold.observe(compute_handler, names='value')
    gui.n_docs.observe(compute_handler, names='value')
    gui.period.observe(compute_handler, names='value')
    gui.scale.observe(compute_handler, names='value')
    gui.node_range.observe(compute_handler, names='value')
    gui.edge_range.observe(compute_handler, names='value')
    gui.threshold.observe(compute_handler, names='value')
    gui.output_format.observe(compute_handler, names='value')
    gui.layout.observe(compute_handler, names='value')
    gui.ignores.observe(compute_handler, names='value')

    display(widgets.VBox([
        widgets.HBox([
            widgets.VBox([gui.layout, gui.threshold, gui.n_docs, gui.period]),
            widgets.VBox([gui.ignores]),
            widgets.VBox([gui.node_range, gui.edge_range, gui.scale]),
            widgets.VBox([widgets.HBox([gui.output_format]), gui.progress]),
        ]),
        gui.output,
        gui.text
    ]))

    compute_handler()