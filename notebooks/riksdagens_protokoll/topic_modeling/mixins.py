from __future__ import annotations

from typing import Any

import ipywidgets as w
import pandas as pd

from westac.riksprot.parlaclarin import codecs as md
from westac.riksprot.parlaclarin import speech_text as st

# pylint: disable=too-many-instance-attributes


class RiksProtMetaDataMixIn:
    def __init__(self, person_codecs: md.PersonCodecs, speech_repository: st.SpeechTextRepository, **kwargs):
        super().__init__(**kwargs)

        self.person_codecs: md.PersonCodecs = person_codecs
        self.speech_repository: st.SpeechTextRepository = speech_repository

        """Display speech text stuff"""
        self._content: w.HTML = w.HTML(layout={'width': '48%', 'background-color': 'lightgreen'})
        self._content_placeholder: w.VBox = self._content
        self.click_handler = self.on_row_click

    def on_row_click(self, item: pd.Series, g: Any):  # pylint: disable=unused-argument
        try:
            if self.speech_repository is None:
                raise ValueError("no repo!")

            speech_name: str = item['document_name']
            if not speech_name.startswith("prot-"):
                raise ValueError(f"WTF! {speech_name}")

            self._content.value = self.speech_repository.speech(speech_name, mode="html")

        except Exception as ex:
            self._content.value = str(ex)
