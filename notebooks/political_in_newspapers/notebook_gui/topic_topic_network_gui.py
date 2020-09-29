
# Visualize topic co-occurrence
import types
import warnings

import bokeh
import bokeh.plotting
import ipywidgets as widgets
from IPython.display import display

import notebooks.political_in_newspapers.corpus_data as corpus_data
import text_analytic_tools.common.network.plot_utility as plot_utility
import text_analytic_tools.common.network.utility as network_utility
import text_analytic_tools.text_analysis.derived_data_compiler as derived_data_compiler
import text_analytic_tools.text_analysis.utility as tmutility
import text_analytic_tools.utility.widgets_utility as widgets_utility
import westac.common.utility as utility

#bokeh.plotting.output_notebook()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = utility.setup_logger()

def get_topic_titles(topic_token_weights, topic_id=None, n_words=100):
    df_temp = topic_token_weights if topic_id is None else topic_token_weights[(topic_token_weights.topic_id==topic_id)]
    df = df_temp\
        .sort_values('weight', ascending=False)\
        .groupby('topic_id')\
        .apply(lambda x: ' '.join(x.token[:n_words].str.title()))
    return df

def display_topic_topic_network(
    c_data,
    publication_id,
    period=None,
    ignores=None,
    threshold=0.10,
    layout='Fruchterman-Reingold',
    n_docs=1,
    scale=1.0,
    output_format='table',
    text_id='',
    titles=None,
    topic_proportions=None,
    node_range=(20, 60),
    edge_range=(1, 10)
):
    try:

        documents = c_data.documents

        if 'document_id' not in documents.columns:
            documents['document_id'] = documents.index

        df = c_data.document_topic_weights
        df = df[(df.weight >= threshold)]

        if ignores is not None:
            df = df[~df.topic_id.isin(ignores)]

        if publication_id is not None:
            df = df[df.publication_id == publication_id]

        if len(period or []) == 2:
            df = df[(df.year>=period[0]) & (df.year<=period[1])]

        if isinstance(period, int):
            df = df[df.year == period]

        df = df.merge(df, how='inner', left_on='document_id', right_on='document_id')
        df = df[(df.topic_id_x < df.topic_id_y)]

        df = df.groupby([df.topic_id_x, df.topic_id_y]).size().reset_index()

        df.columns = ['source', 'target', 'n_docs']

        if n_docs > 1:
            df = df[df.n_docs >= n_docs]

        if len(df) == 0:
            print('No data. Please change selections.')
            return

        if output_format == 'network':
            network = network_utility.create_network(df, source_field='source', target_field='target', weight='n_docs')
            p = plot_utility.plot_network(
                network=network,
                layout_algorithm=layout,
                scale=scale,
                threshold=0.0,
                node_description=titles,
                node_proportions=topic_proportions,
                weight_scale=1.0,
                normalize_weights=False,
                element_id=text_id,
                figsize=(1200,800),
                node_range=node_range,
                edge_range=edge_range
            )
            bokeh.plotting.show(p)
        else:
            df.columns = ['Source', 'Target', 'DocCount']
            if output_format == 'table':
                display(df)
            if output_format == 'excel':
                filename = utility.timestamp("{}_topic_topic_network.xlsx")
                df.to_excel(filename)
                print('Data stored in file {}'.format(filename))
            if output_format == 'csv':
                filename = utility.timestamp("{}_topic_topic_network.csv")
                df.to_csv(filename, sep='\t')
                print('Data stored in file {}'.format(filename))

    except Exception as x:
        raise
        #print("No data: please adjust filters")

def display_gui(state):

    lw = lambda w: widgets.Layout(width=w)
    n_topics = state.num_topics

    text_id = 'nx_topic_topic'
    publications = utility.extend(dict(corpus_data.PUBLICATION2ID), {'(ALLA)': None})
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
        publication_id=widgets.Dropdown(description='Publication', options=publications, value=None, layout=widgets.Layout(width="250px")),
        progress=widgets.IntProgress(min=0, max=4, step=1, value=0, layout=widgets.Layout(width="99%")),
        ignores=widgets.SelectMultiple(description='Ignore', options=ignore_options, value=[], rows=5, layout=lw('250px')),
        node_range=widgets.IntRangeSlider(description='Node size', min=10, max=100, step=1, value=(20, 60), continues_update=False),
        edge_range=widgets.IntRangeSlider(description='Edge size', min=1, max=20, step=1, value=(2, 6), continues_update=False),
        output=widgets.Output()
    )

    def tick(x=None):
        gui.progress.value = gui.progress.value + 1 if x is None else x

    def compute_handler(*args):

        gui.output.clear_output()
        tick(1)
        with gui.output:

            display_topic_topic_network(
                c_data=state.compiled_data,
                publication_id=gui.publication_id.value,
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
    gui.publication_id.observe(compute_handler, names='value')
    gui.ignores.observe(compute_handler, names='value')

    display(widgets.VBox([
        widgets.HBox([
            widgets.VBox([gui.layout, gui.threshold, gui.n_docs, gui.period]),
            widgets.VBox([gui.publication_id, gui.ignores]),
            widgets.VBox([gui.node_range, gui.edge_range, gui.scale]),
            widgets.VBox([widgets.HBox([gui.output_format]), gui.progress]),
        ]),
        gui.output,
        gui.text
    ]))

    compute_handler()
