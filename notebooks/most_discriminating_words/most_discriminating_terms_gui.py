import logging
import types

import ipywidgets
from IPython.display import display

logger = logging.getLogger(__name__)


def display_most_discriminating_terms(df):
    display(df)


def display_gui(x_corpus, x_documents, compute_callback, display_callback):

    lw = lambda w: ipywidgets.Layout(width=w)
    year_span = (x_documents.year.min(), x_documents.year.max())
    gui = types.SimpleNamespace(
        progress=ipywidgets.IntProgress(value=0, min=0, max=5, step=1, description='', layout=lw('90%')),
        top_n_terms=ipywidgets.IntSlider(
            description='#terms',
            min=10,
            max=1000,
            value=100,
            tooltip='The total number of most discriminating terms to return for each group',
        ),
        max_n_terms=ipywidgets.IntSlider(
            description='#top',
            min=1,
            max=2000,
            value=2000,
            tooltip='Only consider terms whose document frequency is within the top # terms out of all terms',
        ),
        period1=ipywidgets.IntRangeSlider(
            description='Period',
            min=year_span[0],
            max=year_span[1],
            value=(x_documents.year.min(), x_documents.year.min() + 4),
            layout=lw('250px'),
        ),
        period2=ipywidgets.IntRangeSlider(
            description='Period',
            min=year_span[0],
            max=year_span[1],
            value=(x_documents.year.max() - 4, x_documents.year.max()),
            layout=lw('250px'),
        ),
        compute=ipywidgets.Button(description='Compute', icon='', button_style='Success', layout=lw('120px')),
        output=ipywidgets.Output(layout={'border': '1px solid black'}),
    )

    boxes = ipywidgets.VBox(
        [
            ipywidgets.HBox(
                [
                    ipywidgets.VBox([gui.period1, gui.period2]),
                    ipywidgets.VBox(
                        [
                            gui.top_n_terms,
                            gui.max_n_terms,
                        ],
                        layout=ipywidgets.Layout(align_items='flex-end'),
                    ),
                    ipywidgets.VBox([gui.compute]),
                ]
            ),
            gui.output,
        ]
    )

    display(boxes)

    def compute_callback_handler(*_):
        gui.output.clear_output()
        with gui.output:
            try:

                gui.compute.disabled = True

                df = compute_callback(
                    x_corpus,
                    x_documents,
                    top_n_terms=gui.top_n_terms.value,
                    max_n_terms=gui.max_n_terms.value,
                    period1=gui.period1.value,
                    period2=gui.period2.value,
                )

                if df is not None:
                    display_callback(df)
                else:
                    logger.info('No data for selected groups or periods.')

            except Exception as ex:
                logger.error(ex)
            finally:
                gui.compute.disabled = False

    gui.compute.on_click(compute_callback_handler)
    return gui
