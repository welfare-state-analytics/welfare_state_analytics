from __future__ import annotations
from functools import cached_property
from io import StringIO
from typing import List, Mapping
from loguru import logger
import pandas as pd
import numpy as np
from os.path import isfile, isdir, join
from penelope import utility as pu
from penelope import corpus as pc

ROLE_TYPE2ID: dict = {
    'unknown': 0,
    'talman': 1,
    'minister': 2,
    'member': 3,
}

GENDER2ID: dict = {
    'unknown': 0,
    'man': 1,
    'woman': 2,
}

MEMBER_NAME2IDNAME_MAPPING: Mapping[str, str] = {
    'gender': 'gender_id',
    'party_abbrev': 'party_abbrev_id',
    'role_type': 'role_type_id',
    'who': 'who_id',
}

PARTY_COLORS = [
    (0, 'S', '#E8112d'),
    (1, 'M', '#52BDEC'),
    (2, 'gov', '#000000'),
    (3, 'C', '#009933'),
    (4, 'L', '#006AB3'),
    (5, 'V', '#DA291C'),
    (6, 'MP', '#83CF39'),
    (7, 'KD', '#000077'),
    (8, 'NYD', '#007700'),
    (9, 'SD', '#DDDD00'),
    # {'party_abbrev_id': 0, 'party_abbrev': 'PP', 'party': 'Piratpartiet', 'color_name': 'Lila', 'color': '#572B85'},
    # {'party_abbrev_id': 0, 'party_abbrev': 'F', 'party': 'Feministiskt', 'color_name': 'initiativ	Rosa', 'color': '#CD1B68'},
]

PARTY_COLOR_BY_ID = {x[0]: x[2] for x in PARTY_COLORS}
PARTY_COLOR_BY_ABBREV = {x[1]: x[2] for x in PARTY_COLORS}

MEMBER_IDNAME2NAME_MAPPING: Mapping[str, str] = pu.revdict(MEMBER_NAME2IDNAME_MAPPING)


def load_speech_index(index_path: str, members_path: str) -> pd.DataFrame:
    """Load speech index. Merge with person index (parla. members, ministers, speakers)"""
    speech_index: pd.DataFrame = pd.read_feather(index_path)
    members: pd.DataFrame = pd.read_csv(members_path, delimiter='\t').set_index('id')
    speech_index['protocol_name'] = speech_index.filename.str.split('_').str[0]
    speech_index = speech_index.merge(members, left_on='who', right_index=True, how='inner').fillna('')
    speech_index.loc[speech_index['gender'] == '', 'gender'] = 'unknown'
    return speech_index, members

class MemberNotFoundError(ValueError):
    ...

class ProtoMetaData:

    DOCUMENT_INDEX_NAME: str = 'document_index.feather'
    MEMBERS_NAME: str = 'person_index.csv'
    MEMBER_NAME2IDNAME_MAPPING: Mapping[str, str] = MEMBER_NAME2IDNAME_MAPPING
    MEMBER_IDNAME2NAME_MAPPING: Mapping[int, str] = MEMBER_IDNAME2NAME_MAPPING
    PARTY_COLOR_BY_ID: Mapping[int, str] = PARTY_COLOR_BY_ID
    PARTY_COLOR_BY_ABBREV: Mapping[str, str] = PARTY_COLOR_BY_ABBREV

    def __init__(self, *, members: pd.DataFrame, document_index: pd.DataFrame, verbose: bool = False):

        self.document_index: pd.DataFrame = (
            document_index if isinstance(document_index, pd.DataFrame) else self.load_document_index(document_index)
        )
        self.members: pd.DataFrame = members if isinstance(members, pd.DataFrame) else self.load_members(members)
        self.members['party_abbrev'] = self.members['party_abbrev'].fillna('unknown')
        self.members['gender'] = self.members['gender'].fillna('unknown')
        self.members.loc[~self.members['gender'].isin(GENDER2ID.keys()), 'gender'] = 'unknown'
        self.members.loc[~self.members['role_type'].isin(ROLE_TYPE2ID.keys()), 'role_type'] = 'unknown'

        if verbose:
            logger.info(f"size of mop's metadata: {pu.size_of(self.members, 'MB', total=True)}")
            logger.info(f"size of document_index: {pu.size_of(self.document_index, 'MB', total=True)}")

    def map_id2text_names(self, id_names: List[str]) -> List[str]:
        return [MEMBER_IDNAME2NAME_MAPPING.get(x) for x in id_names]

    @property
    def gender2id(self) -> dict:
        return GENDER2ID

    @property
    def gender2name(self) -> dict:
        return pu.revdict(GENDER2ID)

    @property
    def role_type2id(self) -> dict:
        return ROLE_TYPE2ID

    @property
    def role_type2name(self) -> dict:
        return pu.revdict(ROLE_TYPE2ID)

    @cached_property
    def member_property_specs(self) -> List[Mapping[str, str | Mapping[str, int]]]:
        return [
            dict(text_name='gender', id_name='gender_id', values=self.gender2id),
            dict(text_name='role_type', id_name='role_type_id', values=self.role_type2id),
            dict(text_name='party_abbrev', id_name='party_abbrev_id', values=self.party_abbrev2id),
            dict(text_name='who', id_name='who_id', values=self.who2id),
        ]

    @cached_property
    def genders(self) -> pd.DataFrame:
        return pd.DataFrame(data={'gender': self.gender2id.values()}, index=self.gender2id.keys())

    # @cached_property
    # def gender2name(self) -> pd.DataFrame:
    #     return self.genders['gender'].to_dict()

    # @cached_property
    # def gender2id(self) -> dict:
    #     return revdict(self.gender2name)

    @cached_property
    def party_abbrevs(self) -> pd.DataFrame:
        return pd.DataFrame({'party_abbrev': self.members.party_abbrev.unique()})

    @cached_property
    def party_abbrev2name(self) -> dict:
        return self.party_abbrevs['party_abbrev'].to_dict()

    @cached_property
    def party_abbrev2id(self) -> dict:
        return pu.revdict(self.party_abbrev2name)

    @cached_property
    def whos(self) -> pd.DataFrame:
        return pd.DataFrame({'who': self.members.index.unique()})

    @cached_property
    def who2name(self) -> dict:
        return self.whos['who'].to_dict()

    @cached_property
    def who2id(self) -> dict:
        return pu.revdict(self.who2name)

    def get_member(self, who: str) -> dict:
        try:
            return self.members.loc[who].to_dict()
        except:
            MemberNotFoundError(f"ID={who}")

    @staticmethod
    def load_from_same_folder(folder: str, verbose: bool = False) -> "ProtoMetaData":
        """Loads members and document index from `folder`"""
        try:
            return ProtoMetaData(document_index=folder, members=folder, verbose=verbose)
        except Exception as ex:
            raise FileNotFoundError(
                "unable to load data from {folder}, please make sure both document index and members index reside in same folder."
            ) from ex

    @staticmethod
    def load_document_index(folder: str) -> pd.DataFrame:
        document_index: pd.DataFrame = None
        try:
            document_index = pc.DocumentIndexHelper.load(
                filename=folder if not isdir(folder) else join(folder, ProtoMetaData.DOCUMENT_INDEX_NAME)
            ).document_index
        except FileNotFoundError:
            """Try load from DTM folder"""
            document_index = pc.DocumentIndexHelper.load(filename=folder).document_index

        # document_index = pd.read_feather(join(folder, ProtoMetaData.DOCUMENT_INDEX_NAME))
        document_index.assign(protocol_name=document_index.filename.str.split('_').str[0])
        return document_index

    @staticmethod
    def read_members(filename: str, sep: str = '\t') -> pd.DataFrame:
        members: pd.DataFrame = (
            pd.read_feather(filename)
            if not isinstance(filename, StringIO) and filename.endswith('feather')
            else pd.read_csv(filename, sep=sep)
        )
        if 'unknown' not in members.id:
            members = members.append(unknown_member(), ignore_index=True)
        members = members.set_index('id')
        return members

    @staticmethod
    def load_members(source: str, sep: str = '\t') -> pd.DataFrame:

        if isinstance(source, StringIO) or isfile(source):
            return ProtoMetaData.read_members(source, sep=sep)

        probe_extension: List[str] = ['feather', 'csv.feather', 'csv', 'zip', 'csv.gz', 'csv.zip']
        for extension in probe_extension:
            filename: str = join(
                source,
                pu.replace_extension(ProtoMetaData.MEMBERS_NAME, extension=extension),
            )
            if not isfile(filename):
                continue
            return ProtoMetaData.read_members(filename, sep=sep)

        raise FileNotFoundError(f"Parliamentary members file not found in {source}, probed {','.join(probe_extension)}")

    def overload_by_member_data(
        self, df: pd.DataFrame, *, encoded: bool = True, drop: bool = True, columns: List[str] = None
    ) -> pd.DataFrame:

        if 'who' not in df.columns:
            raise ValueError("cannot merge member data, `who` is missing in target")

        columns = (
            columns
            if columns is not None
            else ['who_id', 'gender_id', 'party_abbrev_id', 'role_type_id']
            if encoded
            else ['gender', 'party_abbrev', 'role_type']
        )

        target: pd.DataFrame = self.encoded_members if encoded else self.members
        target = target[columns]
        xi: pd.DataFrame = df.merge(target, left_on='who', right_index=True, how='left')

        if drop:
            xi.drop(columns='who', inplace=True)

        xi = self.as_slim_types(xi)

        return xi

    @cached_property
    def overloaded_document_index(self) -> pd.DataFrame:
        return self.overload_by_member_data(self.document_index, encoded=True, drop=True)

    @cached_property
    def simple_members(self) -> pd.DataFrame:
        return self.members[
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

    @cached_property
    def encoded_members(self) -> pd.DataFrame:
        mx: pd.DataFrame = self.members
        mx = mx.assign(
            who_id=pd.Series(mx.index).apply(self.who2id.get).astype(np.int16),
            gender_id=mx['gender'].apply(self.gender2id.get).astype(np.int8),
            party_abbrev_id=mx['party_abbrev'].apply(self.party_abbrev2id.get).astype(np.int8),
            role_type_id=mx['role_type'].apply(self.role_type2id.get),
        )
        mx = mx.drop(columns=['gender', 'party_abbrev', 'role_type'])
        return mx

    def as_slim_types(self, df: pd.DataFrame) -> pd.DataFrame:
        df = as_slim_types(df, columns=['role_type_id', 'gender_id', 'party_abbrev_id'], dtype=np.int8)
        df = as_slim_types(df, columns=['who_id'], dtype=np.int16)
        return df

    def decode_members_data(self, df: pd.DataFrame, drop: bool = True) -> pd.DataFrame:
        if 'role_type_id' in df.columns:
            df['role_type'] = df['role_type_id'].apply(self.role_type2name.get)
        if 'gender_id' in df.columns:
            df['gender'] = df['gender_id'].apply(self.gender2name.get)
        if 'party_abbrev_id' in df.columns:
            df['party_abbrev'] = df['party_abbrev_id'].apply(self.party_abbrev2name.get)
        if 'who_id' in df.columns:
            df['who'] = df['who_id'].apply(self.who2name.get)
        if drop:
            df.drop(columns=['who_id', 'gender_id', 'party_abbrev_id', 'role_type_id'], inplace=True, errors='ignore')
        return df


def unknown_member() -> dict:
    return dict(
        id='unknown',
        role_type='unknown',
        born=0,
        chamber=np.nan,
        district=np.nan,
        start=0,
        end=0,
        gender='unknown',
        name='unknown',
        occupation='unknown',
        party='unknown',
        party_abbrev='unknown',
    )


def as_slim_types(df: pd.DataFrame, columns: List[str], dtype: np.dtype) -> pd.DataFrame:
    if df is None:
        return None
    if isinstance(columns, str):
        columns = [columns]
    for column in columns:
        if column in df.columns:
            df[column] = df[column].fillna(0).astype(dtype)
    return df
