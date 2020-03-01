
# Visualize topic co-occurrence
import warnings
import types
import pandas as pd
import ipywidgets as widgets
import bokeh
import bokeh.plotting
import text_analytic_tools.utility.widgets_utility as widgets_utility
import text_analytic_tools.text_analysis.derived_data_compiler as derived_data_compiler
import text_analytic_tools.text_analysis.utility as tmutility
import text_analytic_tools.text_analysis.topic_model as topic_model
import text_analytic_tools.common.network.utility as network_utility
import text_analytic_tools.common.network.plot_utility as plot_utility
import westac.common.utility as utility
import westac.notebooks.political_in_newspapers.corpus_data as corpus_data

from IPython.display import display

#bokeh.plotting.output_notebook()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def get_topic_titles(topic_token_weights, topic_id=None, n_words=100):
    df_temp = topic_token_weights if topic_id is None else topic_token_weights[(topic_token_weights.topic_id==topic_id)]
    df = df_temp\
        .sort_values('weight', ascending=False)\
        .groupby('topic_id')\
        .apply(lambda x: ' '.join(x.token[:n_words].str.title()))
    return df

def display_topic_co_occurrence_network(
    c_data,
    publication_id,
    period=None,
    ignores=None,
    threshold=0.10,
    layout='Fruchterman-Reingold',
    scale=1.0,
    output_format='table',
    text_id=''
):
    try:
        documents = c_data.documents
        titles = derived_data_compiler.get_topic_titles(c_data.topic_token_weights)
        df = c_data.document_topic_weights
        df['document_id'] = df.index

        node_sizes = tmutility.compute_topic_proportions(df, documents, n_terms_column='n_terms')

        if ignores is not None:
            df = df[~df.topic_id.isin(ignores)]

        if publication_id is not None:
            df = df.merge(documents[documents.publication_id == publication_id]['year'], how='inner', left_on="document_id", right_index=True)
        else:
            df = df.merge(documents['year'], how='inner', left_on="document_id", right_index=True)

        if len(period or []) == 2:
            df = df[(df.year>=period[0]) & (df.year<=period[1])]

        if isinstance(period, int):
            df = df[df.year == period]

        df = df.loc[(df.weight >= threshold)]

        df = pd.merge(df, df, how='inner', left_on='document_id', right_on='document_id')
        df = df.loc[(df.topic_id_x < df.topic_id_y)]

        df_coo = df.groupby([df.topic_id_x, df.topic_id_y]).size().reset_index()

        df_coo.columns = ['source', 'target', 'n_docs']

        if len(df_coo) == 0:
            print('No data. Please change selections.')
            return

        if output_format == 'network':
            network = network_utility.create_network(df_coo, source_field='source', target_field='target', weight='n_docs')
            p = plot_utility.plot_network(
                network=network,
                layout_algorithm=layout,
                scale=scale,
                threshold=0.0,
                node_description=titles,
                node_proportions=node_sizes,
                weight_scale=5.0,
                normalize_weights=False,
                element_id=text_id,
                figsize=(900,500)
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

def display_gui(state, documents):

    lw = lambda w: widgets.Layout(width=w)
    n_topics = state.num_topics

    text_id = 'nx_topic_topic'
    publications = utility.extend(dict(corpus_data.PUBLICATION2ID), {'(ALLA)': None})
    layout_options = [ 'Circular', 'Kamada-Kawai', 'Fruchterman-Reingold']
    year_min, year_max = state.compiled_data.year_period

    gui = types.SimpleNamespace(
        n_topics=n_topics,
        text=widgets_utility.wf.create_text_widget(text_id),
        period=widgets.IntRangeSlider(description='Time', min=year_min, max=year_max, step=1, value=(year_min, year_max), continues_update=False),
        scale=widgets.FloatSlider(description='Scale', min=0.0, max=1.0, step=0.01, value=0.1, continues_update=False),
        threshold=widgets.FloatSlider(description='Threshold', min=0.0, max=1.0, step=0.01, value=0.20, continues_update=False),
        output_format=widgets.Dropdown(description='Output', options={ 'Network': 'network', 'Table': 'table', 'Excel': 'excel', 'CSV': 'csv' }, value='network', layout=lw('200px')),
        layout=widgets.Dropdown(description='Layout', options=layout_options, value='Fruchterman-Reingold', layout=lw('250px')),
        publication_id=widgets.Dropdown(description='Publication', options=publications, value=None, layout=widgets.Layout(width="250px")),
        progress=widgets.IntProgress(min=0, max=4, step=1, value=0, layout=widgets.Layout(width="99%")),
        ignores=widgets.SelectMultiple(description='Ignore', options=[('', None)] + [ ('Topic #'+str(i), i) for i in range(0, n_topics) ], value=[], rows=5, layout=lw('250px')),
        output=widgets.Output()
    )

    def tick(x=None):
        gui.progress.value = gui.progress.value + 1 if x is None else x

    def update_handler(*args):

        gui.output.clear_output()
        tick(1)
        with gui.output:

            display_topic_co_occurrence_network(
                c_data=state.compiled_data,
                publication_id=gui.publication_id.value,
                period=gui.period.value,
                ignores=gui.ignores.value,
                threshold=gui.threshold.value,
                layout=gui.layout.value,
                scale=gui.scale.value,
                output_format=gui.output_format.value,
                text_id=text_id
            )
        tick(0)

    gui.threshold.observe(update_handler, names='value')
    gui.layout.observe(update_handler, names='value')
    gui.scale.observe(update_handler, names='value')
    gui.output_format.observe(update_handler, names='value')
    gui.publication_id.observe(update_handler, names='value')
    gui.output_format.observe(update_handler, names='value')

    display(widgets.VBox([
        widgets.HBox([
            widgets.VBox([gui.layout, gui.threshold, gui.scale, gui.period]),
            widgets.VBox([gui.publication_id, gui.ignores]),
            widgets.VBox([gui.output_format, gui.progress]),
        ]),
        gui.output,
        gui.text
    ]))

    update_handler()
