import warnings
import numpy as np
import ipywidgets as widgets
import bokeh
import bokeh.plotting
import text_analytic_tools.utility.widgets_utility as widgets_utility
import text_analytic_tools.utility.widgets as widgets_helper
import text_analytic_tools.text_analysis.derived_data_compiler as derived_data_compiler

from IPython.display import display

warnings.filterwarnings("ignore", category=DeprecationWarning)

def plot_topic_word_distribution(tokens, **args):

    source = bokeh.models.ColumnDataSource(tokens)

    p = bokeh.plotting.figure(toolbar_location="right", **args)

    cr = p.circle(x='xs', y='ys', source=source)

    label_style = dict(level='overlay', text_font_size='8pt', angle=np.pi/6.0)

    text_aligns = ['left', 'right']
    for i in [0, 1]:
        label_source = bokeh.models.ColumnDataSource(tokens.iloc[i::2])
        labels = bokeh.models.LabelSet(x='xs', y='ys', text_align=text_aligns[i], text='token', text_baseline='middle',
                          y_offset=5*(1 if i == 0 else -1),
                          x_offset=5*(1 if i == 0 else -1),
                          source=label_source, **label_style)
        p.add_layout(labels)

    p.xaxis[0].axis_label = 'Token #'
    p.yaxis[0].axis_label = 'Probability%'
    p.ygrid.grid_line_color = None
    p.xgrid.grid_line_color = None
    p.axis.axis_line_color = None
    p.axis.major_tick_line_color = None
    p.axis.major_label_text_font_size = "6pt"
    p.axis.major_label_standoff = 0
    return p

def display_topic_tokens(state, topic_id=0, n_words=100, output_format='Chart', gui=None):

    def tick(n=None):
        if gui is not None:
            gui.progress.value = (gui.progress.value + 1) if n is None else n

    if gui is not None and gui.n_topics != state.num_topics:
        gui.n_topics = state.num_topics
        gui.topic_id.value = 0
        gui.topic_id.max=state.num_topics - 1

    tick(1)

    tokens = derived_data_compiler.get_topic_tokens(state.compiled_data.topic_token_weights, topic_id=topic_id, n_tokens=n_words).\
        copy()\
        .drop('topic_id', axis=1)\
        .assign(weight=lambda x: 100.0 * x.weight)\
        .sort_values('weight', axis=0, ascending=False)\
        .reset_index()\
        .head(n_words)

    if len(tokens) == 0:
        print("No data! Please change selection.")
        return

    if output_format == 'Chart':
        tick()
        tokens = tokens.assign(xs=tokens.index, ys=tokens.weight)
        p = plot_topic_word_distribution(tokens, plot_width=1200, plot_height=500, title='', tools='box_zoom,wheel_zoom,pan,reset')
        bokeh.plotting.show(p)
        tick()
    else:
        display(tokens)

    tick(0)

def display_gui(state):

    text_id = 'wc01'
    output_options = ['Chart', 'Table']

    gui = widgets_utility.WidgetUtility(
        n_topics=state.num_topics,
        text_id=text_id,
        text=widgets_helper.text(text_id),
        topic_id=widgets.IntSlider(description='Topic ID', min=0, max=state.num_topics - 1, step=1, value=0),
        n_words=widgets.IntSlider(description='#Words', min=5, max=500, step=1, value=75),
        output_format=widgets.Dropdown(description='Format', options=output_options, value=output_options[0], layout=widgets.Layout(width="200px")),
        progress=widgets.IntProgress(min=0, max=4, step=1, value=0, layout=widgets.Layout(width="95%"))
    )

    gui.prev_topic_id = gui.create_prev_id_button('topic_id', state.num_topics)
    gui.next_topic_id = gui.create_next_id_button('topic_id', state.num_topics)

    iw = widgets.interactive(
        display_topic_tokens,
        state=widgets.fixed(state),
        topic_id=gui.topic_id,
        n_words=gui.n_words,
        output_format=gui.output_format,
        gui=widgets.fixed(gui)
    )

    display(widgets.VBox([
        gui.text,
        widgets.HBox([gui.prev_topic_id, gui.next_topic_id, gui.topic_id, gui.n_words, gui.output_format]),
        gui.progress,
        iw.children[-1]
    ]))

    iw.update()

