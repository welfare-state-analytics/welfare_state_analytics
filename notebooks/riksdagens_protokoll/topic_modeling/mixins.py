from __future__ import annotations

import ipywidgets as w

from westac.riksprot.parlaclarin import metadata as md
from westac.riksprot.parlaclarin import speech_text as st

# pylint: disable=too-many-instance-attributes


class SpeechTextMixin:
    def __init__(self, riksprot_metadata: md.ProtoMetaData, speech_repository: st.SpeechTextRepository, **kwargs):
        pivot_key_specs = riksprot_metadata.member_property_specs
        super().__init__(pivot_key_specs=pivot_key_specs, **kwargs)

        self.riksprot_metadata: md.ProtoMetaData = riksprot_metadata
        self.speech_repository: st.SpeechTextRepository = speech_repository
        self._content: w.HTML = w.HTML(layout={'width': '48%', 'background-color': 'lightgreen'})
        self._content_placeholder: w.VBox = self._content
        self.click_handler = self.on_row_select

    def on_row_select(self, args: dict):

        try:
            if self.speech_repository is None:
                raise ValueError("no repo!")

            if args.get('column', '') != 'document_name':
                raise ValueError(f"click on wrong column {args.get('column', '')}")

            speech_name: str = args.get('cell_value', '')

            if not speech_name.startswith("prot-"):
                raise ValueError(f"WTF! {speech_name}")

            self._content.value = self.speech_repository.speech(speech_name, mode="html")
        except Exception as ex:
            self._content.value = str(ex) + " " + str(args)
