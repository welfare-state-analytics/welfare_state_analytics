from __future__ import annotations

from typing import List, Union

import pandas as pd
from ipydatagrid import DataGrid
from penelope.notebook import token_counts as tc
from westac.riksprot.parlaclarin import metadata

# pylint: disable=unused-argument


class PoSCountGUI(tc.BasicDTMGUI):
    def __init__(self, *, default_folder: str, riksprot_metadata: metadata.ProtoMetaData, encoded: bool = True):
        self.DATA = None
        self.encoded: bool = encoded
        self.riksprot_metadata: metadata.ProtoMetaData = riksprot_metadata
        member_property_spec: dict = self.riksprot_metadata.member_property_specs
        super().__init__(default_folder=default_folder, pivot_key_specs=member_property_spec)

    def keep_columns(self) -> List[str]:
        return super().keep_columns() + ['who']

    def prepare(self) -> "PoSCountGUI":
        super().prepare()
        self.document_index = self.riksprot_metadata.overload_by_member_data(self.document_index, encoded=self.encoded)
        return self

    def load(self, source: Union[str, pd.DataFrame]) -> "PoSCountGUI":
        super().load(source)
        return self

    def compute(self) -> pd.DataFrame:
        data: pd.DataFrame = super().compute()
        if data is not None and self.encoded:
            data = self.riksprot_metadata.decode_members_data(data)
        self.DATA = data
        return data

    def plot_tabular(self, df: pd.DataFrame, opts: tc.ComputeOpts) -> DataGrid:
        return df
