import penelope.utility as utility
from IPython.display import display
from ipywidgets import Dropdown, HBox, VBox
from penelope.notebook.topic_modelling import TopicModelContainer, TopicOverviewGUI
from penelope.topic_modelling import prevelance

import notebooks.political_in_newspapers.repository as repository


class PoliticalTopicOverviewGUI(TopicOverviewGUI):
    def __init__(self):
        super().__init__(prevelance.AverageTopicPrevalenceOverTimeCalculator())

        publications = utility.extend(dict(repository.PUBLICATION2ID), {'(ALLA)': None})
        self.publication_id: Dropdown = Dropdown(
            description='Publication', options=publications, value=None, layout=dict(width="200px")
        )

    def setup(self, state: TopicModelContainer) -> "PoliticalTopicOverviewGUI":
        super().setup(state)
        self.publication_id.observe(self.update_handler, names='value')
        return self

    def layout(self) -> VBox:

        return VBox(
            [
                HBox([self.aggregate, self.publication_id, self.output_format, self.flip_axis]),
                HBox([self.output]),
                self.text,
            ]
        )


def display_gui(state: TopicModelContainer):

    gui: PoliticalTopicOverviewGUI = PoliticalTopicOverviewGUI().setup(state)
    display(gui.layout())
    gui.update_handler()
