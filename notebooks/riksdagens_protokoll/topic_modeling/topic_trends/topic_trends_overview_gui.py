from IPython.display import display
from ipywidgets import Dropdown, HBox, VBox
from penelope.notebook.topic_modelling import TopicModelContainer, TopicOverviewGUI


class PoliticalTopicOverviewGUI(TopicOverviewGUI):
    def __init__(self):
        super().__init__(calculator=None)

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

    def setup(self, state: TopicModelContainer) -> "PoliticalTopicOverviewGUI":
        super().setup(state)
        self.party_id.observe(self.update_handler, names='value')
        self.speaker_id.observe(self.update_handler, names='value')
        self.gender_id.observe(self.update_handler, names='value')
        return state

    def layout(self) -> VBox:

        return VBox(
            [
                HBox(
                    [self.aggregate, self.party_id, self.gender_id, self.speaker_id, self.output_format, self.flip_axis]
                ),
                HBox([self.output]),
                self.text,
            ]
        )

    def data_filters(self) -> dict:
        return {
            'party_id': self.party_id.value,
            'speaker_id': self.speaker_id.value,
        }


def display_gui(state: TopicModelContainer):

    gui: PoliticalTopicOverviewGUI = PoliticalTopicOverviewGUI().setup(state)
    display(gui.layout())
    gui.update_handler()
