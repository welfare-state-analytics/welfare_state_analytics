import warnings

import ipywidgets as widgets
from IPython.display import display

import text_analytic_tools.text_analysis.derived_data_compiler as derived_data_compiler
import text_analytic_tools.utility.widgets as widgets_helper
import text_analytic_tools.utility.widgets_utility as widgets_utility
import westac.common.utility as utility
from notebooks.common import filter_document_topic_weights, to_text

logger = utility.setup_logger()
# from beakerx import *
# from beakerx.object import beakerx
# beakerx.pandas_display_table()

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def reconstitue_texts_for_topic(df, corpus, id2token, n_top=500):

    df['text'] = df.document_id.apply(lambda x: to_text(corpus[x], id2token))
    df = df.drop(['topic_id', 'year'], axis=1).set_index('document_id')
    df.index.name = 'id'
    return df.sort_values('weight', ascending=False).head(n_top)

def display_texts(
    state,
    filters,
    threshold=0.0,
    output_format='Table',
    n_top=500
):
 
    corpus = state.model_data.corpus
    id2token = state.model_data.id2term
    document_topic_weights = state.compiled_data.document_topic_weights

    df = filter_document_topic_weights(document_topic_weights, filters=filters, threshold=threshold)

    df = reconstitue_texts_for_topic(df, corpus, id2token, n_top=n_top)

    if len(df) == 0:
        print('No data to display for this topic and threshold')
    elif output_format == 'Table':
        display(df)

def display_gui(state):

    year_min, year_max = state.compiled_data.year_period
    year_options =  [ (x,x) for x in range(year_min, year_max + 1)]

    text_id = 'topic_document_text'

    gui = widgets_utility.WidgetUtility(
        n_topics=state.num_topics,
        text_id=text_id,
        text=widgets_helper.text(text_id),
        year=widgets.Dropdown(description='Year', options=year_options, value=year_options[0][0], layout=widgets.Layout(width="200px")),
        topic_id=widgets.IntSlider(description='Topic ID', min=0, max=state.num_topics - 1, step=1, value=0, continuous_update=False),
        n_top=widgets.IntSlider(description='#Docs', min=5, max=500, step=1, value=75),
        threshold=widgets.FloatSlider(description='Threshold', min=0.0, max=1.0, step=0.01, value=0.20, continues_update=False),
        output_format=widgets.Dropdown(description='Format', options=['Table'], value='Table', layout=widgets.Layout(width="200px")),
        progress=widgets.IntProgress(min=0, max=4, step=1, value=0),
        output=widgets.Output()
    )

    gui.prev_topic_id = gui.create_prev_id_button('topic_id', state.num_topics)
    gui.next_topic_id = gui.create_next_id_button('topic_id', state.num_topics)

    def on_topic_change_update_gui(topic_id):

        if gui.n_topics != state.num_topics:
            gui.n_topics = state.num_topics
            gui.topic_id.value = 0
            gui.topic_id.max = state.num_topics - 1

        tokens = derived_data_compiler.get_topic_title(state.compiled_data.topic_token_weights, topic_id, n_tokens=200)

        gui.text.value = 'ID {}: {}'.format(topic_id, tokens)

    def update_handler(*_):

        gui.output.clear_output()

        with gui.output:

            on_topic_change_update_gui(gui.topic_id.value)

            display_texts(
                state=state,
                filters=dict(year=gui.year.value, topic_id=gui.topic_id.value),
                threshold=gui.threshold.value,
                n_top=gui.n_top.value,
                output_format=gui.output_format.value
            )

    gui.topic_id.observe(update_handler, names='value')
    gui.year.observe(update_handler, names='value')
    gui.threshold.observe(update_handler, names='value')
    gui.n_top.observe(update_handler, names='value')
    gui.output_format.observe(update_handler, names='value')

    display(widgets.VBox([
        widgets.HBox([
            widgets.VBox([
                widgets.HBox([gui.prev_topic_id, gui.next_topic_id]),
                gui.progress,
            ]),
            widgets.VBox([gui.topic_id, gui.threshold, gui.n_top]),
            widgets.VBox([gui.year]),
            widgets.VBox([gui.output_format])
        ]),
        gui.text,
        gui.output
    ]))

    update_handler()
