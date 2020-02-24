import wordcloud
import matplotlib.pyplot as plt
import ipywidgets as widgets
import text_analytic_tools.utility.widgets_utility as widgets_utility
import text_analytic_tools.utility.widgets as widgets_helper
import text_analytic_tools.text_analysis.topic_model_utility as topic_model_utility

from IPython.display import display

opts = { 'max_font_size': 100, 'background_color': 'white', 'width': 900, 'height': 600 }

def plot_wordcloud(df, token='token', weight='weight', figsize=(14, 14/1.618), **args):
    token_weights = dict({ tuple(x) for x in df[[token, weight]].values })
    image = wordcloud.WordCloud(**args,)
    image.fit_words(token_weights)
    plt.figure(figsize=figsize) #, dpi=100)
    plt.imshow(image, interpolation='bilinear')
    plt.axis("off")
    plt.show()

def display_wordcloud(
    state,
    topic_id=0,
    n_words=100,
    output_format='Wordcloud',
    gui=None
):
    def tick(n=None):
        gui.progress.value = (gui.progress.value + 1) if n is None else n

    if gui.n_topics != state.num_topics:
        gui.n_topics = state.num_topics
        gui.topic_id.value = 0
        gui.topic_id.max=state.num_topics - 1

    tick(1)

    try:
        topic_token_weights = state.compiled_data.topic_token_weights

        df = topic_token_weights.loc[(topic_token_weights.topic_id == topic_id)]

        tokens = topic_model_utility.get_topic_title(topic_token_weights, topic_id, n_tokens=n_words)
        gui.text.value = 'ID {}: {}'.format(topic_id, tokens)

        tick()

        if output_format == 'Wordcloud':
            plot_wordcloud(df, 'token', 'weight', max_words=n_words, **opts)
        else:
            tick()
            df = topic_model_utility.get_topic_tokens(topic_token_weights, topic_id=topic_id, n_tokens=n_words)
            tick()
            display(df)
    except IndexError:
        print('No data for topic')
    tick(0)

def display_gui(state):

    output_options = ['Wordcloud', 'Table']
    text_id = 'tx02'

    gui = widgets_utility.WidgetUtility(
        n_topics=state.num_topics,
        text_id=text_id,
        text=widgets_helper.text(text_id),
        topic_id=widgets.IntSlider(description='Topic ID', min=0, max=state.num_topics - 1, step=1, value=0, continuous_update=False),
        word_count=widgets.IntSlider(description='#Words', min=5, max=250, step=1, value=25, continuous_update=False),
        output_format=widgets.Dropdown(description='Format', options=output_options, value=output_options[0], layout=widgets.Layout(width="200px")),
        progress=widgets.IntProgress(min=0, max=4, step=1, value=0, layout=widgets.Layout(width="95%"))
    )

    gui.prev_topic_id = gui.create_prev_id_button('topic_id', state.num_topics)
    gui.next_topic_id = gui.create_next_id_button('topic_id', state.num_topics)

    iw = widgets.interactive(
        display_wordcloud,
        state=widgets.fixed(state),
        topic_id=gui.topic_id,
        n_words=gui.word_count,
        output_format=gui.output_format,
        gui=widgets.fixed(gui)
    )

    display(widgets.VBox([
        gui.text,
        widgets.HBox([gui.prev_topic_id, gui.next_topic_id, gui.topic_id, gui.word_count, gui.output_format]),
        gui.progress,
        iw.children[-1]
    ]))

    iw.update()
