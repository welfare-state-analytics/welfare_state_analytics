import penelope.utility as utility
from IPython.display import display
from ipywidgets import Dropdown
from penelope.notebook.topic_modelling import TopicModelContainer, TopicTrendsGUI

import notebooks.political_in_newspapers.repository as repository


class PoliticalTopicTrendsGUI(TopicTrendsGUI):
    def __init__(self):
        super().__init__()

        self.publication_id = Dropdown(
            description='Publication',
            options=utility.extend(dict(repository.PUBLICATION2ID), {'(ALLA)': None}),
            value=None,
            layout=dict(width="200px"),
        )

    def layout(self):
        self.extra_placeholder.children = [self.publication_id]
        return super().layout()

    def setup(self, state: TopicModelContainer) -> "PoliticalTopicTrendsGUI":
        super().setup(state)

        self.publication_id.observe(self.update_handler, names='value')

        return self

    def data_filter(self) -> dict:
        return {'publication_id': self.publication_id.value}


def display_gui(state: TopicModelContainer, extra_filter=None):  # pylint: disable=unused-argument

    gui = PoliticalTopicTrendsGUI().setup(state)

    display(gui.layout())

    gui.update_handler()
