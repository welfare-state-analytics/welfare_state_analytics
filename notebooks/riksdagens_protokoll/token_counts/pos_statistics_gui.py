from typing import Union
import pandas as pd
from penelope.notebook.token_counts import BasicDTMGUI, ComputeOpts
from westac.riksprot.parlaclarin import metadata


class PoSCountGUI(BasicDTMGUI):
    def __init__(self, default_folder: str, encoded: bool = True):
        super().__init__(default_folder, ['gender_id', 'party_abbrev_id'])
        self.encoded: bool = encoded
        self.protocol_metadata: metadata.ProtoMetaData = None

    def load(self, source: Union[str, pd.DataFrame]) -> None:
        self.protocol_metadata = metadata.ProtoMetaData.load_from_same_folder(source)
        self.document_index = self.protocol_metadata.overload_by_member_data(
            self.protocol_metadata.document_index, encoded=self.encoded
        )

    def compute(self, df: pd.DataFrame, opts: ComputeOpts) -> pd.DataFrame:
        data: pd.DataFrame = super().compute(df, opts)
        if self.encoded:
            data = self.protocol_metadata.decode_members_data(data)
        return data
