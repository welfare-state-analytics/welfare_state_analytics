from pprint import pprint as pp

import bokeh
import ipywidgets as widgets
import westac.common.curve_fit as cf
from IPython.display import display

from notebooks.word_trends.displayers import (display_bar, display_grid,
                                              display_line, display_table)


def display_gui(container):

    output_widget = widgets.Output(layout=widgets.Layout(width='600px', height='200px'))
    words_widget = widgets.Textarea(
        description="", rows=4, value="och eller hur", layout=widgets.Layout(width='600px', height='200px')
    )
    tab_widget = widgets.Tab()

    data_handlers = [display_table, display_line, display_bar, display_grid]
    tab_widget.children = [widgets.Output() for _ in data_handlers]
    _ = [tab_widget.set_title(i, x.NAME) for i, x in enumerate(data_handlers)]

    # smooth = False
    # smoothers = (
    #     []
    #     if not smooth
    #     else [
    #         # cf.rolling_average_smoother('nearest', 3),
    #         cf.pchip_spline
    #     ]
    # )

    z_corpus = None
    x_corpus = None

    def update_plot(*_):

        nonlocal z_corpus, x_corpus

        if container.corpus is None:

            with output_widget:
                print("Please load a corpus!")

            return

        if z_corpus is None or z_corpus is not container.corpus:

            with output_widget:
                print("Corpus changed...")

            z_corpus = container.corpus
            x_corpus = z_corpus.todense()

            tab_widget.children[1].clear_output()
            with tab_widget.children[1]:
                display_line.setup(
                    container, x_ticks=[x for x in x_corpus.xs_years()], plot_width=1000, plot_height=500
                )

        tokens = '\n'.join(words_widget.value.split()).split()
        tab_index = tab_widget.selected_index
        data_handler = data_handlers[tab_index]
        indices = [x_corpus.token2id[token] for token in tokens if token in x_corpus.token2id]

        with output_widget:

            if len(indices) == 0:
                print("Nothing to show!")
                return

            missing_tokens = [token for token in tokens if token not in x_corpus.token2id]
            if len(missing_tokens) > 0:
                print("Not in corpus subset: {}".format(' '.join(missing_tokens)))

        if data_handler.NAME != "Line":
            tab_widget.children[tab_index].clear_output()

        with tab_widget.children[tab_index]:

            data = data_handler.compile(x_corpus, indices)
            container.data = data
            data_handler.plot(data, container=container)

    words_widget.observe(update_plot, names='value')
    tab_widget.observe(update_plot, 'selected_index')

    display(widgets.VBox([widgets.HBox([words_widget, output_widget]), tab_widget]))

    update_plot()
