from __future__ import annotations
from os.path import join as jj, isdir, isfile
from penelope.notebook import topic_modelling as ntm

import westac.riksprot.parlaclarin.speech_text as sr
import pandas as pd
from westac.riksprot.parlaclarin import codecs as md


class RiksprotLoadGUI(ntm.LoadGUI):
    def __init__(
        self, person_codecs: md.PersonCodecs, data_folder: str, state: ntm.TopicModelContainer, slim: bool = False
    ):
        super().__init__(data_folder, state, slim)
        self.person_codecs: md.PersonCodecs = person_codecs
        self.version: str = None

    def load(self) -> None:
        self.update_metadata(self.version, self.data_folder)
        return super().load()
    


    def update_metadata(self, version: str) -> None:

        self.version = version

        metadata_filename: str = self.probe_metadata_filename()
        document_index_filename: str = jj(self.data_folder, version, 'tagged_frames.feather/document_index.feather')
        corpus_folder: str = jj(self.data_folder, version, 'tagged_frames')

        person_codecs: md.PersonCodecs = md.PersonCodecs().load(source=metadata_filename)
        document_index: pd.DataFrame = pd.read_feather(document_index_filename)
        speech_repository: sr.SpeechTextRepository = sr.SpeechTextRepository(
            source=corpus_folder, person_codecs=person_codecs, document_index=document_index
        )

        self.state.put('person_codecs', person_codecs)
        self.state.put('speech_repository', speech_repository)
        self.state.put('version', version)
        self.state.put('data_folder', self.data_folder)
        self.state.put('speech_index', document_index)


    def probe_metadata_filename(self):

        return self.probe_filenames([
            jj(self.data_folder, f"metadata/riksprot_metadata.{self.version}.db"),
            jj(self.data_folder, f"metadata/{self.version}/riksprot_metadata.{self.version}.db"),
            jj(self.data_folder, f"metadata/{self.version}/riksprot_metadata..db"),
        ])
            
    
    def probe_vrt_corpus_folder(self):

        return self.probe_filenames([
            jj(self.data_folder, self.version, 'tagged_frames'),
            jj(self.data_folder, self.version, 'corpus/tagged_frames'),
            jj(self.data_folder, self.version, f'corpus/tagged_frames_{self.version}'),
            jj(self.data_folder, f'corpus/tagged_frames_{self.version}'),
        ])
    

    def probe_filenames(self, candidates: str):
        for candidate in candidates:
            if isfile(candidate):
                return candidate
        raise FileNotFoundError(f"Could not find any of {candidates}")
