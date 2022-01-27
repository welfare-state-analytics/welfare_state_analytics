from __future__ import annotations


import ipywidgets as w
import pandas as pd
from penelope.notebook import topic_modelling as tm
from penelope.notebook import mixins as mx
from penelope import utility as pu
from penelope import topic_modelling as tx
from westac.riksprot.parlaclarin import metadata as md
from IPython.display import display

# pylint: disable=too-many-instance-attributes


class RiksprotFindTopicDocumentsGUI(mx.PivotKeysMixIn, tm.FindTopicDocumentsGUI):
    def __init__(
        self,
        riksprot_metadata: md.ProtoMetaData,
        state: tm.TopicModelContainer | dict,
    ):
        pivot_key_specs = riksprot_metadata.member_property_specs
        super().__init__(pivot_key_specs=pivot_key_specs, state=state)

        self.riksprot_metadata: md.ProtoMetaData = riksprot_metadata
        self._filter_keys.rows = 8
        self._filter_keys.layout = {'width': '180px'}
        self._threshold_slider.value = 0.20
        self._find_text.value = "film"
        self._year_range.value = (1990, 1992)
        self._extra_placeholder = w.HBox(
            [
                w.VBox([w.HTML("<b>Filter by</b>"), self._pivot_keys_text_names]),
                w.VBox([w.HTML("<b>Value</b>"), self._filter_keys]),
            ]
        )
        self.inferred_topics: tx.InferredTopicsData = self.state["inferred_topics"]

    def setup(self, **kwargs):
        return super().setup(**kwargs)

    @property
    def filter_opts(self) -> pu.PropertyValueMaskingOpts:
        options: dict = super(RiksprotFindTopicDocumentsGUI, self).filter_opts
        return options

    def update(self) -> pd.DataFrame:
        _ = super().update()
        calculator: tx.DocumentTopicsCalculator = self.dtw_calculator
        dtw: pd.DataFrame = calculator.overload(includes="protocol_name,document_name,who").value
        """note: dtw == calculator.data"""
        return dtw
