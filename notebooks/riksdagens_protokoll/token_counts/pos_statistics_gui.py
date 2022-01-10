from typing import List, Union

import pandas as pd
from ipydatagrid import DataGrid
from loguru import logger
from penelope.notebook import token_counts as tc
from penelope.notebook.token_counts.plot import plot_by_bokeh as plot_dataframe
from westac.riksprot.parlaclarin import metadata

# pylint: disable=unused-argument


class PoSCountGUI(tc.BasicDTMGUI):
    def __init__(self, *, default_folder: str, encoded: bool = True):
        super().__init__(
            default_folder=default_folder,
            pivot_keys={'gender': 'gender_id', 'party_abbrev': 'party_abbrev_id', 'role_type': 'role_type_id'},
        )
        self.encoded: bool = encoded
        self.protocol_metadata: metadata.ProtoMetaData = None

    def keep_columns(self) -> List[str]:
        return super().keep_columns() + ['who']

    def prepare(self) -> "tc.BasicDTMGUI":
        super().prepare()
        self.document_index = self.protocol_metadata.overload_by_member_data(self.document_index, encoded=self.encoded)
        return self

    def load(self, source: Union[str, pd.DataFrame]) -> "PoSCountGUI":
        super().load(source)
        self.protocol_metadata = metadata.ProtoMetaData.load_from_same_folder(source)
        return self

    def compute(self) -> pd.DataFrame:
        data: pd.DataFrame = super().compute()
        if data is not None and self.encoded:
            data = self.protocol_metadata.decode_members_data(data)
        return data

    def plot_tabular(self, df: pd.DataFrame, opts: tc.ComputeOpts) -> DataGrid:
        return df
