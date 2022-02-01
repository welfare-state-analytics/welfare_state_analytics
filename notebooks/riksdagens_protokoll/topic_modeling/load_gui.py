from __future__ import annotations

from penelope import pipeline as pp
from penelope import topic_modelling as tm
from penelope.notebook import topic_modelling as ntm

from westac.riksprot.parlaclarin import metadata as md


class RiksprotLoadGUI(ntm.LoadGUI):
    def __init__(
        self,
        riksprot_metadata: md.ProtoMetaData,
        corpus_folder: str,
        state: ntm.TopicModelContainer,
        corpus_config: pp.CorpusConfig | None = None,
        slim: bool = False,
    ):
        super().__init__(corpus_folder, state, corpus_config, slim)
        self.riksprot_metadata: md.ProtoMetaData = riksprot_metadata

    def load(self):
        super().load()
        inferred_topics: tm.InferredTopicsData = self.state["inferred_topics"]
        inferred_topics.document_index = self.riksprot_metadata.overload_by_member_data(
            inferred_topics.document_index, encoded=True, drop=True
        )
