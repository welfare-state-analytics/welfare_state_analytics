from __future__ import annotations

import pandas as pd
from penelope import topic_modelling as tx
from penelope import utility as pu
from penelope.notebook import mixins as mx
from penelope.notebook import topic_modelling as ntm

from westac.riksprot.parlaclarin import codecs as md
from westac.riksprot.parlaclarin import speech_text as st

from .mixins import RiksProtMetaDataMixIn

# pylint: disable=too-many-instance-attributes


class RiksprotBrowseTopicDocumentsGUI(RiksProtMetaDataMixIn, mx.PivotKeysMixIn, ntm.BrowseTopicDocumentsGUI):
    def __init__(
        self,
        person_codecs: md.PersonCodecs,
        speech_repository: st.SpeechTextRepository,
        state: ntm.TopicModelContainer | dict,
    ):
        super().__init__(
            person_codecs=person_codecs,
            speech_repository=speech_repository,
            pivot_key_specs=person_codecs.property_values_specs,
            state=state,
        )

        self._threshold.value = 0.20
        self._year_range.value = (1990, 1992)
        self._extra_placeholder = self.default_pivot_keys_layout(layout={'width': '180px'}, rows=8)

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
        data: pd.DataFrame = self.person_codecs.decode(
            calculator.overload(includes="protocol_name,document_name,gender_id,party_id,person_id").value,
            drop=True,
        )
        return data


class RiksprotFindTopicDocumentsGUI(RiksProtMetaDataMixIn, mx.PivotKeysMixIn, ntm.FindTopicDocumentsGUI):
    def __init__(
        self,
        person_codecs: md.PersonCodecs,
        speech_repository: st.SpeechTextRepository,
        state: ntm.TopicModelContainer | dict,
    ):
        super().__init__(
            person_codecs=person_codecs,
            speech_repository=speech_repository,
            pivot_key_specs=person_codecs.property_values_specs,
            state=state,
        )

        self._threshold.value = 0.20
        self._extra_placeholder = self.default_pivot_keys_layout(layout={'width': '180px'}, rows=8)

    def setup(self, **kwargs):  # pylint: disable=useless-super-delegation
        return super().setup(**kwargs)

    @property
    def filter_opts(self) -> pu.PropertyValueMaskingOpts:
        options: dict = super(RiksprotFindTopicDocumentsGUI, self).filter_opts  # pylint: disable=super-with-arguments
        return options

    def update(self) -> pd.DataFrame:
        _ = super().update()
        return overload_decoded_member_data(self.person_codecs, self.inferred_topics.calculator)


def overload_decoded_member_data(
    person_codecs: md.PersonCodecs, calculator: tx.DocumentTopicsCalculator
) -> pd.DataFrame:
    """note: at this point dtw is equal to calculator.data"""
    data: pd.DataFrame = person_codecs.decode(
        calculator.overload(includes="protocol_name,document_name,gender_id,party_id,person_id").value,
        drop=True,
    )
    return data
