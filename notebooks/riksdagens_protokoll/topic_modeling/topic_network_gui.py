from __future__ import annotations

import pandas as pd
from penelope import utility as pu
from penelope.notebook import mixins as mx
from penelope.notebook import topic_modelling as tm

from westac.riksprot.parlaclarin import codecs as md
from westac.riksprot.parlaclarin import speech_text as st

from .mixins import RiksProtMetaDataMixIn

# pylint: disable=too-many-instance-attributes


class RiksprotTopicTopicGUI(RiksProtMetaDataMixIn, mx.PivotKeysMixIn, tm.TopicTopicGUI):
    def __init__(
        self,
        person_codecs: md.PersonCodecs,
        speech_repository: st.SpeechTextRepository,
        state: tm.TopicModelContainer | dict,
    ):
        super(RiksprotTopicTopicGUI, self).__init__(  # pylint: disable=super-with-arguments
            pivot_key_specs=person_codecs.property_values_specs,
            person_codecs=person_codecs,
            speech_repository=speech_repository,
            state=state,
        )
        # self._threshold.value = 0.20
        self._year_range.value = self.inferred_topics.startspan(5)
        self._extra_placeholder = self.default_pivot_keys_layout(layout={'width': '180px'}, rows=8)

    def setup(self, **kwargs) -> "RiksprotTopicTopicGUI":  # pylint: disable=useless-super-delegation
        super().setup(**kwargs)
        return self

    @property
    def filter_opts(self) -> pu.PropertyValueMaskingOpts:
        options: dict = super(RiksprotTopicTopicGUI, self).filter_opts  # pylint: disable=super-with-arguments
        return options

    def update(self) -> pd.DataFrame:
        data: pd.DataFrame = super().update()
        # todo: if grouped by pivo, then overload with decoded values
        # calculator: tx.DocumentTopicsCalculator = self.inferred_topics.calculator
        # dtw: pd.DataFrame = calculator.overload(includes="protocol_name,document_name,who").value
        return data
