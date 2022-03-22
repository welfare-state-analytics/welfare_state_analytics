from __future__ import annotations

import os
import sqlite3
from typing import Any, Mapping, Type

import numpy as np
import pandas as pd
import requests


def revdict(d: dict) -> dict:
    return {v: k for k, v in d.items()}


COLUMN_TYPES = {
    'year_of_birth': np.int16,
    'year_of_death': np.int16,
    'gender_id': np.int8,
    'party_id': np.int8,
    'chamber_id': np.int8,
    'office_type_id': np.int8,
    'sub_office_type_id': np.int8,
    'start_year': np.int16,
    'end_year': np.int16,
    'district_id': np.int16,
}

COLUMN_DEFAULTS = {
    'gender_id': 0,
    'year_of_birth': 0,
    'year_of_death': 0,
    'district_id': 0,
    'party_id': 0,
    'chamber_id': 0,
    'office_type_id': 0,
    'sub_office_type_id': 0,
    'start_year': 0,
    'end_year': 0,
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
]

PARTY_COLOR_BY_ID = {x[0]: x[2] for x in PARTY_COLORS}
PARTY_COLOR_BY_ABBREV = {x[1]: x[2] for x in PARTY_COLORS}

NAME2IDNAME_MAPPING: Mapping[str, str] = {
    'gender': 'gender_id',
    'office_type': 'office_type_id',
    'sub_office_type': 'sub_office_type_id',
    'person_id': 'pid',
}
IDNAME2NAME_MAPPING: Mapping[str, str] = revdict(NAME2IDNAME_MAPPING)


def read_sql_table(table_name: str, con: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql(f"select * from {table_name}", con)


def read_sql_tables(tables: list[str] | dict, db: sqlite3.Connection) -> dict[str, pd.DataFrame]:
    return tables if isinstance(tables, dict) else {table_name: read_sql_table(table_name, db) for table_name in tables}


def load_tables(
    tables: dict[str, str],
    *,
    db: sqlite3.Connection,
    defaults: dict[str, Any] = None,
    types: dict[str, Any] = None,
):
    """Loads tables as pandas dataframes, slims types, fills NaN, sets pandas index"""
    data: dict[str, pd.DataFrame] = read_sql_tables(list(tables.keys()), db)
    slim_table_types(data.values(), defaults=defaults, types=types)
    for table_name, table in data.items():
        if tables.get(table_name):
            table.set_index(tables.get(table_name), drop=True, inplace=True)
    return data


def slim_table_types(
    tables: list[pd.DataFrame] | pd.DataFrame,
    defaults: dict[str, Any] = None,
    types: dict[str, Any] = None,
) -> None:

    """Slims types and sets default value for NaN entries"""

    if isinstance(tables, pd.DataFrame):
        tables = [tables]

    defaults: dict[str, Any] = COLUMN_DEFAULTS if defaults is None else defaults
    types: dict[str, Any] = COLUMN_TYPES if types is None else types

    for table in tables:

        for column_name, value in defaults.items():
            if column_name in table.columns:
                table[column_name].fillna(value, inplace=True)

        for column_name, dt in types.items():
            if column_name in table.columns:
                if table[column_name].dtype != dt:
                    table[column_name] = table[column_name].astype(dt)


def group_to_list_of_records2(df: pd.DataFrame, key: str) -> dict[str | int, list[dict]]:
    """Groups `df` by `key` and aggregates each group to list of row records (dicts)"""
    return {q: df.loc[ds].to_dict(orient='records') for q, ds in df.groupby(key).groups.items()}


def group_to_list_of_records(
    df: pd.DataFrame, key: str, properties: list[str] = None, ctor: Type = None
) -> dict[str | int, list[dict]]:
    """Groups `df` by `key` and aggregates each group to list of row records (dicts)"""
    key_rows: pd.DataFrame = pd.DataFrame(
        data={
            key: df[key],
            'data': (df[properties] if properties else df).to_dict("records"),
        }
    )
    if ctor is not None:
        key_rows['data'] = key_rows['data'].apply(lambda x: ctor(**x))

    return key_rows.groupby(key)['data'].apply(list).to_dict()


def download_url_to_file(url: str, target_name: str, force: bool = False) -> None:

    if os.path.isfile(target_name):
        if not force:
            raise ValueError("File exists, use `force=True` to overwrite")
        os.unlink(target_name)

    ensure_path(target_name)

    with open(target_name, 'w', encoding="utf-8") as fp:
        data: str = requests.get(url, allow_redirects=True).content.decode("utf-8")
        fp.write(data)


def probe_filename(filename: list[str], exts: list[str] = None) -> str | None:
    """Probes existence of filename with any of given extensions in folder"""
    for probe_name in set([filename] + ([replace_extension(filename, ext) for ext in exts] if exts else [])):
        if os.path.isfile(probe_name):
            return probe_name
    raise FileNotFoundError(filename)


def replace_extension(filename: str, extension: str) -> str:
    if filename.endswith(extension):
        return filename
    base, _ = os.path.splitext(filename)
    return f"{base}{'' if extension.startswith('.') else '.'}{extension}"


def ensure_path(f: str) -> None:
    os.makedirs(os.path.dirname(f), exist_ok=True)
