from functools import cached_property
import pandas as pd
from os.path import join as jj


def load_speech_index(index_path: str, members_path: str) -> pd.DataFrame:
    """Load speech index. Merge with person index (parla. members, ministers, speakers)"""
    speech_index: pd.DataFrame = pd.read_feather(index_path)
    members: pd.DataFrame = pd.read_csv(members_path, delimiter='\t').set_index('id')
    speech_index['protocol_name'] = speech_index.filename.str.split('_').str[0]
    speech_index = speech_index.merge(members, left_on='who', right_index=True, how='inner').fillna('')
    speech_index.loc[speech_index['gender'] == '', 'gender'] = 'unknown'
    return speech_index, members


class ProtoMetaData:

    DOCUMENT_INDEX_NAME: str = 'document_index.feather'
    MEMBERS_NAME: str = 'person_index.csv'

    def __init__(self, members: pd.DataFrame, document_index: pd.DataFrame):
        self.document_index: pd.DataFrame = document_index
        self.members: pd.DataFrame = members

    @staticmethod
    def load(folder: str) -> "ProtoMetaData":

        document_index: pd.DataFrame = pd.read_feather(jj(folder, ProtoMetaData.DOCUMENT_INDEX_NAME))
        document_index.assign(protocol_name=document_index.filename.str.split('_').str[0])
        members: pd.DataFrame = pd.read_csv(jj(folder, ProtoMetaData.MEMBERS_NAME), delimiter='\t').set_index('id')

        return ProtoMetaData(document_index=document_index, members=members)

    @cached_property
    def full_index(self) -> pd.DataFrame:
        si: pd.DataFrame = self.document_index.merge(self.members, left_on='who', right_index=True, how='inner').fillna(
            ''
        )
        si.loc[si['gender'] == '', 'gender'] = 'unknown'
        return si

    @cached_property
    def simple_index(self) -> pd.DataFrame:
        return self.full_index[
            [
                'year',
                'document_name',
                'n_tokens',
                'who',
                'document_id',
                'role_type',
                'born',
                'gender',
                'name',
                'party_abbrev',
            ]
        ]
