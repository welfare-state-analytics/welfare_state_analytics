import logging
from dataclasses import dataclass
from typing import Callable

import ipywidgets
import pandas as pd
from IPython.display import display

logger = logging.getLogger(__name__)


def display_most_discriminating_terms(df):
    display(df)

 # pylint: disable=too-many-instance-attributes

@dataclass
class MDW_GUI:

    progress: ipywidgets.IntProgress = ipywidgets.IntProgress(
        value=0, min=0, max=5, step=1, description='', layout={'width': '90%'}
    )
    top_n_terms: ipywidgets.IntSlider = ipywidgets.IntSlider(
        description='#terms',
        min=10,
        max=1000,
        value=100,
        tooltip='The total number of most discriminating terms to return for each group',
    )
    max_n_terms: ipywidgets.IntSlider = ipywidgets.IntSlider(
        description='#top',
        min=1,
        max=2000,
        value=2000,
        tooltip='Only consider terms whose document frequency is within the top # terms out of all terms',
    )
    period1: ipywidgets.IntRangeSlider = ipywidgets.IntRangeSlider(
        description='Period',
        min=1900,
        max=2099,
        value=(2001, 2002),
        layout={'width': '250px'},
    )
    period2 = ipywidgets.IntRangeSlider(
        description='Period',
        min=1900,
        max=2099,
        value=(2001, 2002),
        layout={'width': '250px'},
    )
    compute = ipywidgets.Button(description='Compute', icon='', button_style='Success', layout={'width': '120px'})
    output = ipywidgets.Output(layout={'border': '1px solid black'})

    compute_handler: Callable = None

    def setup(self, document_index: pd.DataFrame, compute_handler: Callable) -> "MDW_GUI":

        self.period1.min, self.period1.max = (document_index.year.min(), document_index.year.max())
        self.period2.min, self.period2.max = (document_index.year.min(), document_index.year.max())
        self.period1.value = (document_index.year.min(), document_index.year.min() + 4)
        self.period2.value = (document_index.year.max() - 4, document_index.year.max())

        self.compute_handler = compute_handler
        self.compute.on_click(self._compute_handler)

        return self

    def layout(self):

        layout = ipywidgets.VBox(
            [
                ipywidgets.HBox(
                    [
                        ipywidgets.VBox([self.period1, self.period2]),
                        ipywidgets.VBox(
                            [
                                self.top_n_terms,
                                self.max_n_terms,
                            ],
                            layout=ipywidgets.Layout(align_items='flex-end'),
                        ),
                        ipywidgets.VBox([self.compute]),
                    ]
                ),
                self.output,
            ]
        )

        return layout

    def _compute_handler(self, *_):

        if self.compute_handler is None:
            return
        try:
            self.compute.disabled = True
            self.output.clear_output()

            with self.output:

                self.compute_handler(self)

        except Exception as ex:
            logger.error(ex)
        finally:
            self.compute.disabled = False


def display_gui(corpus, document_index, compute_callback, display_callback):
    def _compute_callback(args: MDW_GUI):

        mdw = compute_callback(
            corpus,
            document_index,
            top_n_terms=args.top_n_terms.value,
            max_n_terms=args.max_n_terms.value,
            period1=args.period1.value,
            period2=args.period2.value,
        )

        if mdw is not None:
            display_callback(mdw)
        else:
            logger.info('No data for selected groups or periods.')

    gui = MDW_GUI().setup(document_index=document_index, compute_handler=_compute_callback)
    layout = gui.layout()

    display(layout)

    return gui
