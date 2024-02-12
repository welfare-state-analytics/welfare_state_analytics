from __future__ import annotations

from penelope.notebook import topic_modelling as ntm

from notebooks.riksdagens_protokoll.topic_modeling.utility import metadata

from .container import TopicModelContainer


class RiksprotLoadGUI(ntm.LoadGUI):
    def __init__(
        self, data_folder: str, state: TopicModelContainer, slim: bool = False
    ):  # pylint: disable=useless-parent-delegation
        super().__init__(data_folder, state, slim)

    def load(self) -> None:
        self.state.store(**metadata.load_metadata(self.data_folder, self.model_info.folder))
        return super().load()

    @property
    def corpus_version(self) -> str:
        return self.state.corpus_version
