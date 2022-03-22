from __future__ import annotations

from penelope import pipeline as pp
from penelope import topic_modelling as tm
from penelope.notebook import topic_modelling as ntm

from westac.riksprot.parlaclarin import codecs as md


class RiksprotLoadGUI(ntm.LoadGUI):
    def __init__(
        self,
        person_codecs: md.PersonCodecs,
        corpus_folder: str,
        state: ntm.TopicModelContainer,
        corpus_config: pp.CorpusConfig | None = None,
        slim: bool = False,
    ):
        super().__init__(corpus_folder, state, corpus_config, slim)
        self.person_codecs: md.PersonCodecs = person_codecs

    def load(self):
        super().load()

        if 'gender_id' not in self.document_index.columns:
            raise ValueError("gender_id missing in document index, might be a config.yml error")
