from dataclasses import dataclass
from typing import Callable, Optional

import ipywidgets as widgets
import penelope.co_occurrence as co_occurrence
import penelope.notebook.co_occurrence as co_occurrence_gui
from IPython.core.display import display
from penelope.notebook.word_trends.trends_data import TrendsData
from penelope.pipeline import CorpusConfig

import __paths__

view = widgets.Output(layout={'border': '2px solid green'})


def create(
    data_folder: str,
    filename_pattern: str = co_occurrence.CO_OCCURRENCE_FILENAME_PATTERN,
    loaded_callback: Callable[[co_occurrence.Bundle], None] = None,
) -> co_occurrence_gui.LoadGUI:

    # @debug_view.capture(clear_output=True)
    def load_callback(filename: str):
        co_occurrence.load_bundle(filename, loaded_callback)

    gui: co_occurrence_gui.LoadGUI = co_occurrence_gui.LoadGUI(default_path=data_folder).setup(
        filename_pattern=filename_pattern, load_callback=load_callback
    )
    return gui


@view.capture(clear_output=True)
def compute_co_occurrence_callback(
    corpus_config: CorpusConfig,
    args: co_occurrence_gui.ComputeGUI,
    partition_key: str,
    done_callback: Callable,
    checkpoint_file: Optional[str] = None,
):
    compute_co_occurrence = co_occurrence_gui.compute_pipeline_factory(corpus_config.corpus_type)
    compute_co_occurrence(
        corpus_config=corpus_config,
        args=args,
        partition_key=partition_key,
        done_callback=done_callback,
        checkpoint_file=checkpoint_file,
    )


@dataclass
class MainGUI:
    def __init__(
        self,
        corpus_config_name: str,
        corpus_folder: str = __paths__.data_folder,
    ) -> widgets.VBox:

        self.trends_data: TrendsData = None
        self.config = CorpusConfig.find(corpus_config_name, __paths__.resources_folder).folder(corpus_folder)

        self.gui_compute: co_occurrence_gui.ComputeGUI = co_occurrence_gui.ComputeGUI.create(
            corpus_folder=corpus_folder,
            corpus_config=self.config,
            compute_callback=compute_co_occurrence_callback,
            done_callback=self.display_explorer,
        )

        self.gui_load: co_occurrence_gui.LoadGUI = co_occurrence_gui.create_load_gui(
            data_folder=corpus_folder,
            filename_pattern=co_occurrence.CO_OCCURRENCE_FILENAME_PATTERN,
            loaded_callback=self.display_explorer,
        )

        self.gui_explore: co_occurrence_gui.ExploreGUI = None

    def layout(self):

        accordion = widgets.Accordion(
            children=[
                widgets.VBox(
                    [
                        self.gui_load.layout(),
                    ],
                    layout={'border': '1px solid black', 'padding': '16px', 'margin': '4px'},
                ),
                widgets.VBox(
                    [
                        self.gui_compute.layout(),
                    ],
                    layout={'border': '1px solid black', 'padding': '16px', 'margin': '4px'},
                ),
            ]
        )

        accordion.set_title(0, "LOAD AN EXISTING CO-OCCURRENCE COMPUTATION")
        accordion.set_title(1, '...OR COMPUTE A NEW CO-OCCURRENCE')
        # accordion.set_title(2, '...OR LOAD AND EXPLORE A CO-OCCURRENCE DTM')
        # accordion.set_title(3, '...OR COMPUTE OR DOWNLOAD CO-OCCURRENCES AS EXCEL')

        return widgets.VBox([accordion, view])

    @view.capture(clear_output=True)
    def display_explorer(self, bundle: co_occurrence.Bundle):

        self.trends_data = co_occurrence.to_trends_data(bundle).update()

        self.gui_explore = co_occurrence_gui.ExploreGUI().setup().display(trends_data=self.trends_data)

        display(self.gui_explore.layout())
