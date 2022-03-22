from __future__ import annotations
import abc
from functools import cached_property
from typing import Any, List, Mapping
import pandas as pd
import numpy as np
from penelope import utility as pu
import sqlite3


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
]

PARTY_COLOR_BY_ID = {x[0]: x[2] for x in PARTY_COLORS}
PARTY_COLOR_BY_ABBREV = {x[1]: x[2] for x in PARTY_COLORS}

NAME2IDNAME_MAPPING: Mapping[str, str] = {
    'gender': 'gender_id',
    'office_type': 'office_type_id',
    'sub_office_type': 'sub_office_type_id',
    'person_id': 'pid',
}
IDNAME2NAME_MAPPING: Mapping[str, str] = pu.revdict(NAME2IDNAME_MAPPING)


def read_sql_table(table_name: str, con: Any) -> pd.DataFrame:
    return pd.read_sql(f"select * from {table_name}", con)


class IRiksprotMetaData(abc.ABC):

    NAME2IDNAME_MAPPING: Mapping[str, str] = NAME2IDNAME_MAPPING
    IDNAME2NAME_MAPPING: Mapping[int, str] = IDNAME2NAME_MAPPING

    PARTY_COLOR_BY_ID: Mapping[int, str] = PARTY_COLOR_BY_ID
    PARTY_COLOR_BY_ABBREV: Mapping[str, str] = PARTY_COLOR_BY_ABBREV

    def __init__(self, data: dict):
        self.data = data

    @staticmethod
    def load(database_filename: str) -> IRiksprotMetaData:

        data: dict = IRiksprotMetaData.load_data(database_filename=database_filename)
        riksprot_metadata: IRiksprotMetaData = IRiksprotMetaData(data).slim_types().set_indexes()
        return riksprot_metadata

    @staticmethod
    def load_data(database_filename: str) -> dict:

        with sqlite3.connect(database=database_filename) as db:
            data: dict = {
                'chamber': read_sql_table("chamber", db),
                'gender': read_sql_table("gender", db),
                'office_type': read_sql_table("office_type", db),
                'sub_office_type': read_sql_table("sub_office_type", db),
                'person': read_sql_table("persons_of_interest", db),
                'terms_of_office': read_sql_table("terms_of_office", db),
                'government': read_sql_table("government", db),
            }

            return data

    def set_indexes(self) -> IRiksprotMetaData:

        self.chamber.set_index("chamber_id", drop=True, inplace=True)
        self.gender.set_index("gender_id", drop=True, inplace=True)
        self.office_type.set_index("office_type_id", drop=True, inplace=True)
        self.sub_office_type.set_index("sub_office_type_id", drop=True, inplace=True)
        self.terms_of_office.set_index("terms_of_office_id", drop=True, inplace=True)
        self.person.rename_axis("pid", inplace=True)

        return self

    def slim_types(self) -> IRiksprotMetaData:

        self.person['gender_id'].fillna(0, inplace=True)
        self.person['party_id'].fillna(0, inplace=True)

        self.terms_of_office.loc[
            ~self.terms_of_office['office_type_id'].isin(self.office_type.index), 'office_type_id'
        ] = 0
        self.terms_of_office.loc[
            ~self.terms_of_office['sub_office_type_id'].isin(self.sub_office_type.index), 'sub_office_type_id'
        ] = 0

        as_slim_types(self.person, ['year_of_birth', 'year_of_death'], np.int16)
        as_slim_types(self.person, ['gender_id'], np.int8)
        as_slim_types(self.gender, ['gender_id'], np.int8)
        as_slim_types(self.chamber, ['chamber_id'], np.int8)
        as_slim_types(self.office_type, ['office_type_id'], np.int8)
        as_slim_types(self.sub_office_type, ['office_type_id', 'sub_office_type_id'], np.int8)
        as_slim_types(self.terms_of_office, ['start_year', 'end_year', 'district_id'], np.int16)
        as_slim_types(self.terms_of_office, ['office_type_id', 'sub_office_type_id'], np.int8)

        return self

    @property
    def chamber(self) -> pd.DataFrame:
        return self.data['chamber']

    @property
    def person(self) -> pd.DataFrame:
        return self.data['person']

    @property
    def terms_of_office(self) -> pd.DataFrame:
        return self.data['terms_of_office']

    @property
    def gender(self) -> pd.DataFrame:
        return self.data['gender']

    @property
    def office_type(self) -> pd.DataFrame:
        return self.data['office_type']

    @property
    def sub_office_type(self) -> pd.DataFrame:
        return self.data['sub_office_type']

    @property
    def government(self) -> pd.DataFrame:
        return self.data['government']

    @cached_property
    def gender2name(self) -> dict:
        return self.gender['gender'].to_dict()

    @cached_property
    def gender2id(self) -> dict:
        return pu.revdict(self.gender2name)

    @cached_property
    def office_type2name(self) -> dict:
        return self.office_type['office'].to_dict()

    @cached_property
    def office_type2id(self) -> dict:
        return pu.revdict(self.office_type2name)

    @cached_property
    def sub_office_type2name(self) -> dict:
        return self.sub_office_type['description'].to_dict()

    @cached_property
    def sub_office_type2id(self) -> dict:
        return pu.revdict(self.sub_office_type2name)

    @cached_property
    def pid2person_id(self) -> dict:
        return self.person['person_id'].to_dict()

    @cached_property
    def person_id2pid(self) -> dict:
        return pu.revdict(self.pid2person_id)

    @cached_property
    def pid2person_name(self) -> dict:
        return self.person['name'].to_dict()

    @cached_property
    def party_abbrev2id(self) -> dict:
        return {}

    @cached_property
    def property_values_specs(self) -> List[Mapping[str, str | Mapping[str, int]]]:
        return [
            dict(text_name='gender', id_name='gender_id', values=self.gender2id),
            dict(text_name='office_type', id_name='office_type_id', values=self.office_type2id),
            dict(text_name='sub_office_type', id_name='sub_office_type_id', values=self.sub_office_type2id),
            dict(text_name='party_abbrev', id_name='party_id', values=self.party_abbrev2id),
            dict(text_name='person_id', id_name='pid', values=self.person_id2pid),
        ]

    def overload_by_member_data(
        self, df: pd.DataFrame, *, encoded: bool = True, drop: bool = True, columns: List[str] = None
    ) -> pd.DataFrame:

        if 'who' not in df.columns:
            raise ValueError("cannot merge member data, `who` is missing in target")

        columns = (
            columns
            if columns is not None
            else ['person_id', 'gender_id', 'party_id', 'role_type_id']
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
    def encoded_members(self) -> pd.DataFrame:
        mx: pd.DataFrame = self.members
        mx['who'] = mx.index
        mx = mx.assign(
            person_id=mx['who'].apply(self.who2id.get).astype(np.int32),
            gender_id=mx['gender'].apply(self.gender2id.get).astype(np.int8),
            party_id=mx['party_abbrev'].apply(self.party_abbrev2id.get).astype(np.int8),
            role_type_id=mx['role_type'].apply(self.role_type2id.get),
        )
        mx = mx.drop(columns=['who', 'gender', 'party_abbrev', 'role_type'])
        return mx

    def decode_members_data(self, df: pd.DataFrame, drop: bool = True) -> pd.DataFrame:
        if 'role_type_id' in df.columns:
            df['role_type'] = df['role_type_id'].apply(self.role_type2name.get)
        if 'gender_id' in df.columns:
            df['gender'] = df['gender_id'].apply(self.gender2name.get)
        if 'party_id' in df.columns:
            df['party_abbrev'] = df['party_id'].apply(self.party_abbrev2name.get)
        if 'person_id' in df.columns:
            df['who'] = df['person_id'].apply(self.id2who.get)
        if drop:
            df.drop(columns=['person_id', 'gender_id', 'party_id', 'role_type_id'], inplace=True, errors='ignore')
        return df


class MemberNotFoundError(ValueError):
    ...


def as_slim_types(df: pd.DataFrame, columns: List[str], dtype: np.dtype) -> pd.DataFrame:
    if df is None:
        return None
    if isinstance(columns, str):
        columns = [columns]
    for column in columns:
        if column in df.columns:
            df[column] = df[column].fillna(0).astype(dtype)
    return df
