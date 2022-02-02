import penelope.utility as utility
from IPython.display import display
from ipywidgets import Dropdown, HBox, VBox
from penelope.notebook.topic_modelling import TopicModelContainer, TopicTrendsOverviewGUI

import notebooks.political_in_newspapers.repository as repository


class PoliticalTopicOverviewGUI(TopicTrendsOverviewGUI):
    def __init__(self, state: TopicModelContainer):
        super().__init__(state=state)

        publications = utility.extend(dict(repository.PUBLICATION2ID), {'(ALLA)': None})
        self._publication_id: Dropdown = Dropdown(
            description='Publication', options=publications, value=None, layout=dict(width="200px")
        )

    def setup(self, **kwargs) -> "PoliticalTopicOverviewGUI":
        super().setup(**kwargs)
        self._publication_id.observe(self.update_handler, names='value')
        return self

    def layout(self) -> VBox:

        return VBox(
            [
                HBox([self._aggregate, self._publication_id, self._output_format, self._flip_axis]),
                HBox([self._output]),
                self._text,
            ]
        )


def display_gui(state: TopicModelContainer):

    gui: PoliticalTopicOverviewGUI = PoliticalTopicOverviewGUI(state=state).setup()
    display(gui.layout())
    gui.update_handler()
