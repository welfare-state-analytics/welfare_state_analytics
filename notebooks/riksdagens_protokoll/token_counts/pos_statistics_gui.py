from __future__ import annotations

from typing import List, Union

import pandas as pd
from ipydatagrid import DataGrid
from penelope import utility as pu
from penelope.notebook import token_counts as tc

from westac.riksprot.parlaclarin import codecs as md

# pylint: disable=unused-argument


class PoSCountGUI(tc.BasicDTMGUI):
    def __init__(self, *, default_folder: str, person_codecs: md.PersonCodecs, encoded: bool = True):
        self.DATA = None
        self.encoded: bool = encoded
        self.person_codecs: md.PersonCodecs = person_codecs
        pivot_keys: dict = self.person_codecs.property_values_specs
        self._keep_columns: list[str] = pu.flatten([[k['text_name'], k['id_name']] for k in pivot_keys]) + ['who']
        super().__init__(default_folder=default_folder, pivot_key_specs=pivot_keys)

    def keep_columns(self) -> List[str]:
        return super().keep_columns() + self._keep_columns

    def prepare(self) -> "PoSCountGUI":
        super().prepare()
        return self

    def load(self, source: Union[str, pd.DataFrame]) -> "PoSCountGUI":
        super().load(source)
        if not self.encoded:
            self.document_index = self.person_codecs.decode(self.document_index, drop=True)
        return self

    def compute(self) -> pd.DataFrame:
        data: pd.DataFrame = super().compute()
        if data is not None and self.encoded:
            data = self.person_codecs.decode(data)
        self.DATA = data
        return data

    def plot_tabular(self, df: pd.DataFrame, opts: tc.ComputeOpts) -> DataGrid:
        return df
