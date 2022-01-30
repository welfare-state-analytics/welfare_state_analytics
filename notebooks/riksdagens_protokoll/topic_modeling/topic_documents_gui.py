from __future__ import annotations

import ipywidgets as w
import pandas as pd
from penelope import topic_modelling as tx
from penelope import utility as pu
from penelope.notebook import mixins as mx
from penelope.notebook import topic_modelling as tm

from westac.riksprot.parlaclarin import metadata as md
from westac.riksprot.parlaclarin import speech_text as st

from .mixins import SpeechTextMixin

# pylint: disable=too-many-instance-attributes


class RiksprotBrowseTopicDocumentsGUI(SpeechTextMixin, mx.PivotKeysMixIn, tm.BrowseTopicDocumentsGUI):
    def __init__(
        self,
        riksprot_metadata: md.ProtoMetaData,
        speech_repository: st.SpeechTextRepository,
        state: tm.TopicModelContainer | dict,
    ):
        super().__init__(riksprot_metadata=riksprot_metadata, speech_repository=speech_repository, state=state)

        self._filter_keys.rows = 8
        self._filter_keys.layout = {'width': '180px'}
        self._threshold.value = 0.20
        self._year_range.value = (1990, 1992)
        self._extra_placeholder = w.HBox(
            [
                w.VBox([w.HTML("<b>Filter by</b>"), self._pivot_keys_text_names]),
                w.VBox([w.HTML("<b>Value</b>"), self._filter_keys]),
            ]
        )

    def setup(self, **kwargs):  # pylint: disable=useless-super-delegation
        return super().setup(**kwargs)

    @property
    def filter_opts(self) -> pu.PropertyValueMaskingOpts:
        options: dict = super(RiksprotBrowseTopicDocumentsGUI, self).filter_opts  # pylint: disable=super-with-arguments
        return options

    def update(self) -> pd.DataFrame:
        _ = super().update()
        """note: at this point dtw is equal to calculator.data"""
        calculator: tx.DocumentTopicsCalculator = self.inferred_topics.calculator
        data: pd.DataFrame = self.riksprot_metadata.decode_members_data(
            calculator.overload(includes="protocol_name,document_name,gender_id,party_abbrev_id,who_id").value,
            drop=True,
        )
        return data


class RiksprotFindTopicDocumentsGUI(mx.PivotKeysMixIn, tm.FindTopicDocumentsGUI):
    def __init__(
        self,
        riksprot_metadata: md.ProtoMetaData,
        speech_repository: st.SpeechTextRepository,
        state: tm.TopicModelContainer | dict,
    ):
        pivot_key_specs = riksprot_metadata.member_property_specs
        super().__init__(pivot_key_specs=pivot_key_specs, state=state)

        self.riksprot_metadata: md.ProtoMetaData = riksprot_metadata
        self.speech_repository: st.SpeechTextRepository = speech_repository

        self._content: w.HTML = w.HTML(layout={'width': '48%', 'background-color': 'lightgreen'})
        self._filter_keys.rows = 8
        self._filter_keys.layout = {'width': '180px'}
        self._threshold.value = 0.20
        self._find_text.value = "film"
        self._year_range.value = (1990, 1992)
        self._extra_placeholder = w.HBox(
            [
                w.VBox([w.HTML("<b>Filter by</b>"), self._pivot_keys_text_names]),
                w.VBox([w.HTML("<b>Value</b>"), self._filter_keys]),
            ]
        )
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

    def setup(self, **kwargs):  # pylint: disable=useless-super-delegation
        return super().setup(**kwargs)

    @property
    def filter_opts(self) -> pu.PropertyValueMaskingOpts:
        options: dict = super(RiksprotFindTopicDocumentsGUI, self).filter_opts  # pylint: disable=super-with-arguments
        return options

    def update(self) -> pd.DataFrame:
        _ = super().update()
        """note: at this point dtw is equal to calculator.data"""
        calculator: tx.DocumentTopicsCalculator = self.inferred_topics.calculator
        data: pd.DataFrame = self.riksprot_metadata.decode_members_data(
            calculator.overload(includes="protocol_name,document_name,gender_id,party_abbrev_id,who_id").value,
            drop=True,
        )
        return data
