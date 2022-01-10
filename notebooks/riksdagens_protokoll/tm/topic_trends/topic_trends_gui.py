import pandas as pd
import penelope.topic_modelling as tm
from IPython.display import display
from ipywidgets import Dropdown, VBox
from penelope.notebook.topic_modelling import TopicModelContainer
from penelope.notebook.topic_modelling.topic_trends_gui import TopicTrendsGUI


class ProtocolTopicTrendsGUI(TopicTrendsGUI):
    def __init__(self):
        super().__init__()

        self.party_id = Dropdown(
            description='Party',
            options=[],
            value=None,
            layout=dict(width="200px"),
        )
        self.speaker_id = Dropdown(
            description='Speaker',
            options=[],
            value=None,
            layout=dict(width="200px"),
        )
        self.gender_id = Dropdown(
            description='Gender',
            options=[],
            value=None,
            layout=dict(width="200px"),
        )
        self.current_party_id: int = -1
        self.current_weights: pd.DataFrame = None

    def layout(self):
        placeholder: VBox = self.extra_placeholder
        placeholder.children = [self.party_id, self.gender_id, self.speaker_id]
        return super().layout()

    def setup(self, state: TopicModelContainer) -> "ProtocolTopicTrendsGUI":
        super().setup(state)

        self.party_id.observe(self.update_handler, names='value')
        self.gender_id.observe(self.update_handler, names='value')
        self.speaker_id.observe(self.update_handler, names='value')

        return self

    def compute_weights(self) -> pd.DataFrame:
        """Cache weight over time due to the large number of documents"""

        if self.current_party_id != self.party_id.value:

            filters: dict = {'party_id': self.party_id.value}

            self.current_party_id = self.party_id.value
            self.current_weights = (
                tm.FilterDocumentTopicWeights(self.state.inferred_topics).filter_by_keys(filters).value
            )

        return self.current_weights


def display_gui(state: TopicModelContainer, extra_filter=None):  # pylint: disable=unused-argument

    gui = ProtocolTopicTrendsGUI().setup(state)

    display(gui.layout())

    gui.update_handler()
