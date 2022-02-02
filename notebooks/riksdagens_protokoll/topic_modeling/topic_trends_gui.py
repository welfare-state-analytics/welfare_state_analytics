from __future__ import annotations

import pandas as pd
from penelope import utility as pu
from penelope.notebook import mixins as mx
from penelope.notebook import topic_modelling as tm

from westac.riksprot.parlaclarin import metadata as md
from westac.riksprot.parlaclarin import speech_text as st

from .mixins import RiksProtMetaDataMixIn

# pylint: disable=too-many-instance-attributes

# FIXME #162 [BUG] {TopicTrends} Nothing happens when pressing prev/next
# FIXME #163 [BUG] {TopicTrends} Empty chart when filtering on gender (woman)
# FIXME #164  [ENHANCEMENT] {TopicTrends} Enable display of protocol text for output format table


class RiksprotTopicTrendsGUI(RiksProtMetaDataMixIn, mx.PivotKeysMixIn, tm.TopicTrendsGUI):
    def __init__(
        self,
        riksprot_metadata: md.ProtoMetaData,
        speech_repository: st.SpeechTextRepository,
        state: tm.TopicModelContainer | dict,
    ):
        super(RiksprotTopicTrendsGUI, self).__init__(  # pylint: disable=super-with-arguments
            pivot_key_specs=riksprot_metadata.member_property_specs,
            riksprot_metadata=riksprot_metadata,
            speech_repository=speech_repository,
            state=state,
        )
        # self._threshold.value = 0.20
        self._extra_placeholder = self.default_pivot_keys_layout(layout={'width': '180px'}, rows=8)

    def setup(self, **kwargs):  # pylint: disable=useless-super-delegation
        return super().setup(**kwargs)

    @property
    def filter_opts(self) -> pu.PropertyValueMaskingOpts:
        options: dict = super(RiksprotTopicTrendsGUI, self).filter_opts  # pylint: disable=super-with-arguments
        return options

    def update(self) -> pd.DataFrame:
        data: pd.DataFrame = super().update()
        # todo: if grouped by pivo, then overload with decoded values
        # calculator: tx.DocumentTopicsCalculator = self.inferred_topics.calculator
        # dtw: pd.DataFrame = calculator.overload(includes="protocol_name,document_name,who").value
        return data


class RiksprotTopicTrendsOverviewGUI(mx.PivotKeysMixIn, RiksProtMetaDataMixIn, tm.TopicTrendsOverviewGUI):
    def __init__(
        self,
        riksprot_metadata: md.ProtoMetaData,
        speech_repository: st.SpeechTextRepository,
        state: tm.TopicModelContainer | dict,
    ):
        super(RiksprotTopicTrendsOverviewGUI, self).__init__(  # pylint: disable=super-with-arguments
            pivot_key_specs=riksprot_metadata.member_property_specs,
            riksprot_metadata=riksprot_metadata,
            speech_repository=speech_repository,
            state=state,
        )

        self._threshold.value = 0.02
        self._year_range.value = (1990, 1992)
        self._extra_placeholder = self.default_pivot_keys_layout(layout={'width': '180px'}, rows=8)

    def setup(self, **kwargs):  # pylint: disable=useless-super-delegation
        return super().setup(**kwargs)

    @property
    def filter_opts(self) -> pu.PropertyValueMaskingOpts:
        options: dict = super(RiksprotTopicTrendsOverviewGUI, self).filter_opts  # pylint: disable=super-with-arguments
        return options
