from __future__ import annotations

import ipywidgets as w
import pandas as pd
from penelope import topic_modelling as tx
from penelope import utility as pu
from penelope.notebook import mixins as mx
from penelope.notebook import topic_modelling as tm

from westac.riksprot.parlaclarin import metadata as md
from westac.riksprot.parlaclarin import speech_text as st

from .mixins import RiksProtMetaDataMixIn

# pylint: disable=too-many-instance-attributes


class RiksprotBrowseTopicDocumentsGUI(RiksProtMetaDataMixIn, mx.PivotKeysMixIn, tm.BrowseTopicDocumentsGUI):
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


class RiksprotFindTopicDocumentsGUI(RiksProtMetaDataMixIn, mx.PivotKeysMixIn, tm.FindTopicDocumentsGUI):
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
        self._find_text.value = "film"
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
