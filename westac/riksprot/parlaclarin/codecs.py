from __future__ import annotations

import sqlite3
from contextlib import nullcontext
from dataclasses import dataclass
from functools import cached_property
from typing import Callable, Literal, Mapping

import pandas as pd
from penelope import utility as pu
from .utility import load_tables


CODE_TABLENAMES: dict[str, str] = {
    'chamber': 'chamber_id',
    'gender': 'gender_id',
    'government': 'government_id',
    'office_type': 'office_type_id',
    'party': 'party_id',
    'sub_office_type': 'sub_office_type_id',
    'person_of_interest': 'person_id',
}


@dataclass
class Codec:
    type: Literal['encoder', 'decoder']
    from_column: str
    to_column: str
    fx: Callable[[int], str]
    default: str = None


null_frame: pd.DataFrame = pd.DataFrame()


class Codecs:
    def __init__(self):
        self.chamber: pd.DataFrame = null_frame
        self.gender: pd.DataFrame = null_frame
        self.government: pd.DataFrame = null_frame
        self.office_type: pd.DataFrame = null_frame
        self.party: pd.DataFrame = null_frame
        self.sub_office_type: pd.DataFrame = null_frame
        self.persons_of_interest: pd.DataFrame = null_frame
        self.extra_codecs: list[Codec] = []

    def load(self, source: str | sqlite3.Connection | dict) -> Codecs:
        with (sqlite3.connect(database=source) if isinstance(source, str) else nullcontext(source)) as db:
            tables: dict[str, pd.DataFrame] = load_tables(self.tablenames(), db=db)
            for table_name, table in tables.items():
                setattr(self, table_name, table)
        return self

    def tablenames(self) -> dict[str, str]:
        return CODE_TABLENAMES

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
    def party_abbrev2name(self) -> dict:
        return self.party['party_abbrev'].to_dict()

    @cached_property
    def party_abbrev2id(self) -> dict:
        return pu.revdict(self.party_abbrev2name)

    @property
    def codecs(self) -> list[Codec]:
        return self.extra_codecs + [
            Codec('decode', 'gender_id', 'gender', self.gender2name.get),
            Codec('decode', 'office_type_id', 'office_type', self.office_type2name.get),
            Codec('decode', 'party_id', 'party_abbrev', self.party_abbrev2name.get),
            Codec('decode', 'sub_office_type_id', 'sub_office_type', self.sub_office_type2name.get),
            Codec('encode', 'gender', 'gender_id', self.gender2id.get),
            Codec('encode', 'office_type', 'office_type_id', self.office_type2id.get),
            Codec('encode', 'party', 'party_id', self.party_abbrev2id.get),
            Codec('encode', 'sub_office_type', 'sub_office_type_id', self.sub_office_type2id.get),
        ]

    @property
    def decoders(self) -> list[Codec]:
        return [c for c in self.codecs if c.type == 'decode']

    @property
    def encoders(self) -> list[dict]:
        return [c for c in self.codecs if c.type == 'encode']

    def apply_codec(self, df: pd.DataFrame, codecs: list[Codec], drop: bool = True) -> pd.DataFrame:

        for codec in codecs:
            if codec.from_column in df.columns:
                if codec.to_column not in df:
                    df[codec.to_column] = df[codec.from_column].apply(codec.fx)
                if codec.default is not None:
                    df[codec.to_column] = df[codec.to_column].fillna(codec.default)
            if drop:
                df.drop(columns=[codec.from_column], inplace=True, errors='ignore')
        return df

    def decode(self, df: pd.DataFrame, drop: bool = True) -> pd.DataFrame:
        return self.apply_codec(df, self.decoders, drop=drop)

    def encode(self, df: pd.DataFrame, drop: bool = True) -> pd.DataFrame:
        return self.apply_codec(df, self.encoders, drop=drop)

    @cached_property
    def property_values_specs(self) -> list[Mapping[str, str | Mapping[str, int]]]:
        return [
            dict(text_name='gender', id_name='gender_id', values=self.gender2id),
            dict(text_name='office_type', id_name='office_type_id', values=self.office_type2id),
            dict(text_name='party_abbrev', id_name='party_id', values=self.party_abbrev2id),
            dict(text_name='sub_office_type', id_name='sub_office_type_id', values=self.sub_office_type2id),
        ]


class PersonCodecs(Codecs):
    def __init__(self):
        super().__init__()
        self.persons_of_interest: pd.DataFrame = null_frame

    def tablenames(self) -> dict[str, str]:
        tables: dict[str, str] = dict(CODE_TABLENAMES)
        tables['persons_of_interest'] = 'person_id'
        return tables

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
    def property_values_specs(self) -> list[Mapping[str, str | Mapping[str, int]]]:
        return super().property_values_specs + [
            dict(text_name='person_id', id_name='pid', values=self.name2id),
        ]

    @property
    def person(self) -> pd.DataFrame:
        return self.persons_of_interest

    @property
    def codecs(self) -> list[Codec]:
        return (
            self.extra_codecs
            + super().codecs
            + [
                Codec('decode', 'person_id', 'name', self.person_id2name.get),
                Codec('decode', 'pid', 'person_id', self.pid2person_id.get),
                Codec('encode', 'person_id', 'pid', self.person_id2pid.get),
            ]
        )
