from __future__ import annotations

from typing import Any

import ipywidgets as w
import pandas as pd

from westac.riksprot.parlaclarin import codecs as md
from westac.riksprot.parlaclarin import speech_text as st

# pylint: disable=too-many-instance-attributes


class RiksProtMetaDataMixIn:
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        """Display speech text stuff"""
        self._content: w.HTML = w.HTML(layout={'width': '48%', 'background-color': 'lightgreen'})
        self._content_placeholder: w.VBox = self._content
        self._document_click_handler = self.on_row_click

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

    @property
    def person_codecs(self) -> md.PersonCodecs:
        return self.state.person_codecs  # pylint: disable=no-member

    @property
    def speech_repository(self) -> st.SpeechTextRepository:
        return self.state.speech_repository  # pylint: disable=no-member
