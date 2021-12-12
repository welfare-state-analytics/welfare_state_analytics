from IPython.display import display
from ipywidgets import HTML, Dropdown, VBox  # type: ignore
from penelope.notebook.topic_modelling import TopicModelContainer
from penelope.notebook.topic_modelling.topic_topic_network_gui import TopicTopicGUI

import notebooks.political_in_newspapers.repository as repository


class PoliticalTopicTopicGUI(TopicTopicGUI):
    def __init__(self, state: TopicModelContainer):
        super().__init__(state)

        publications = {**repository.PUBLICATION2ID, **{'(ALLA)': None}}

        self.publication_id: Dropdown = Dropdown(
            description='Publication', options=publications, value=None, layout={'width': '250px'}
        )

    def get_data_filter(self) -> dict:
        return dict(publication_id=self.publication_id.value)

    def setup(self) -> "TopicTopicGUI":
        super().setup()
        self.publication_id.observe(self.update_handler, names='value')
        return self

    def extra_widgets(self) -> VBox:
        return VBox(
            [
                HTML("<b>Publication</b>"),
                self.publication_id,
            ]
        )


def display_gui(state: TopicModelContainer):

    gui: PoliticalTopicTopicGUI = PoliticalTopicTopicGUI(state).setup()
    display(gui.layout())
    gui.update_handler()
