import pandas as pd
from penelope.notebook import topic_modelling as ntm

from westac.riksprot.parlaclarin import PersonCodecs, SpeechTextRepository


class TopicModelContainer(ntm.TopicModelContainer):

    @property
    def corpus_version(self) -> str:
        return self.get('corpus_version')

    @property
    def data_folder(self) -> str:
        return self.get('data_folder')

    @property
    def person_codecs(self) -> PersonCodecs:
        return self.get('person_codecs')

    @property
    def speech_repository(self) -> SpeechTextRepository:
        return self.get('speech_repository')

    @property
    def speech_index(self) -> pd.DataFrame:
        return self.get('speech_index')
