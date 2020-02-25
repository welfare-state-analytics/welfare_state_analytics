
# Visualize topic co-occurrence
import warnings
import types
import pandas as pd
import ipywidgets as widgets
import bokeh
import bokeh.plotting
import text_analytic_tools.utility.widgets_utility as widgets_utility
import text_analytic_tools.text_analysis.topic_model_utility as topic_model_utility
import text_analytic_tools.text_analysis.topic_model as topic_model
import text_analytic_tools.common.network.utility as network_utility
import text_analytic_tools.common.network.plot_utility as plot_utility
import westac.common.utility as utility
import westac.notebooks.political_in_newspapers.corpus_data as corpus_data

from IPython.display import display

bokeh.plotting.output_notebook()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def get_topic_titles(topic_token_weights, topic_id=None, n_words=100):
    df_temp = topic_token_weights if topic_id is None else topic_token_weights[(topic_token_weights.topic_id==topic_id)]
    df = df_temp\
            .sort_values('weight', ascending=False)\
            .groupby('topic_id')\
            .apply(lambda x: ' '.join(x.token[:n_words].str.title()))
    return df

# FIXME: add doc token length to df_documents
def get_topic_proportions(corpus_documents, document_topic_weights):
    topic_proportion = topic_model.compute_topic_proportions(document_topic_weights, corpus_documents)
    return topic_proportion

def display_topic_co_occurrence_network(
    compiled_data,
    documents,
    publication_id,
    period=None,
    ignores=None,
    threshold=0.10,
    layout='Fruchterman-Reingold',
    scale=1.0,
    output_format='table'
):
    try:

        titles = topic_model_utility.get_topic_titles(compiled_data.topic_token_weights)
        df = compiled_data.document_topic_weights
        df['document_id'] = df.index

        node_sizes = topic_model.compute_topic_proportions(df, documents)

        if ignores is not None:
            df = df[~df.topic_id.isin(ignores)]

        if publication_id is not None:
            df = df[df.publication_id == publication_id]

        df = df.loc[(df.weight >= threshold)]
        df = pd.merge(df, df, how='inner', left_on='document_id', right_on='document_id')
        df = df.loc[(df.topic_id_x < df.topic_id_y)]
        df = df.groupby([df.topic_id_x, df.topic_id_y]).size().reset_index()

        df.columns = ['source', 'target', 'weight']

        if len(df) == 0:
            print('No data. Please change selections.')
            return

        if output_format == 'table':
            display(df)
        else:
            network = network_utility.NetworkUtility.create_network(df, source_field='source', target_field='target', weight='weight')
            p = plot_utility.PlotNetworkUtility.plot_network(
                network=network,
                layout_algorithm=layout,
                scale=scale,
                threshold=0.0,
                node_description=titles,
                node_proportions=node_sizes,
                weight_scale=10.0,
                normalize_weights=True,
                element_id='cooc_id',
                figsize=(900,500)
            )
            bokeh.plotting.show(p)

    except Exception as x:
        raise
        print("No data: please adjust filters")

def display_gui(state, documents):

    lw = lambda w: widgets.Layout(width=w)
    n_topics = state.num_topics

    text_id = 'cooc_id'

    publications = utility.extend(dict(corpus_data.PUBLICATION2ID), {'(ALLA)': None})
    layout_options = [ 'Circular', 'Kamada-Kawai', 'Fruchterman-Reingold']
    year_min, year_max = state.compiled_data.year_period

    gui = types.SimpleNamespace(
        n_topics=n_topics,
        text=widgets_utility.wf.create_text_widget(text_id),
        period=widgets.IntRangeSlider(description='Time', min=year_min, max=year_max, step=1, value=(year_min, year_max), continues_update=False),
        scale=widgets.FloatSlider(description='Scale', min=0.0, max=1.0, step=0.01, value=0.1, continues_update=False),
        threshold=widgets.FloatSlider(description='Threshold', min=0.0, max=1.0, step=0.01, value=0.20, continues_update=False),
        output_format=widgets.Dropdown(description='Output', options={ 'Network': 'network', 'Table': 'table' }, value='network', layout=lw('200px')),
        layout=widgets.Dropdown(description='Layout', options=layout_options, value='Fruchterman-Reingold', layout=lw('250px')),
        publication_id=widgets.Dropdown(description='Publication', options=publications, value=None, layout=widgets.Layout(width="200px")),
        progress=widgets.IntProgress(min=0, max=4, step=1, value=0, layout=widgets.Layout(width="99%")),
        ignores=widgets.SelectMultiple(description='Ignore', options=[('', None)] + [ ('Topic #'+str(i), i) for i in range(0, n_topics) ], value=[], rows=8, layout=lw('180px')),
        output=widgets.Output()
    )

    def tick(x=None):
        gui.progress.value = gui.progress.value + 1 if x is None else x

    def update_handler(*args):

        gui.output.clear_output()

        with gui.output:

            document_topic_weights = state.compiled_data.document_topic_weights

            if gui.publication_id.value is not None:
                document_topic_weights = document_topic_weights[document_topic_weights.publication_id == gui.publication_id.value]

            display_topic_co_occurrence_network(
                compiled_data=state.compiled_data,
                documents=documents,
                publication_id=gui.publication_id.value,
                period=gui.period.value,
                ignores=gui.ignores.value,
                threshold=gui.threshold.value,
                layout=gui.layout.value,
                scale=gui.scale.value,
                output_format=gui.output_format.value
            )

    gui.period.disabled = True
    gui.ignores.disabled = True
    gui.threshold.observe(update_handler, names='value')
    gui.layout.observe(update_handler, names='value')
    gui.scale.observe(update_handler, names='value')
    gui.output_format.observe(update_handler, names='value')
    gui.publication_id.observe(update_handler, names='value')
    gui.output_format.observe(update_handler, names='value')

    display(widgets.VBox([
        gui.text,
        widgets.HBox([
            widgets.VBox([gui.layout, gui.threshold, gui.scale, gui.period]),
            widgets.VBox([gui.publication_id]),
            widgets.VBox([gui.ignores]),
            widgets.VBox([gui.output_format, gui.progress]),
        ]),
        gui.output
    ]))

    update_handler()
