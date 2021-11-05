from dataclasses import dataclass
from enum import Enum
from typing import List, Mapping, Union
import pandas as pd
import numpy as np

GITHUB_DATA_URL = (
    "https://raw.githubusercontent.com/welfare-state-analytics/riksdagen-corpus/{}/corpus/members_of_parliament.csv"
)


def encode(df: pd.DataFrame, codes: pd.DataFrame, columns: Union[str, List[str]] = None) -> pd.DataFrame:
    """Replace column values `columns` in `df` with corresponding code in `codes`."""
    id_name: str = codes.index.name
    columns = id_name.rstrip('_id') if columns is None else columns
    if isinstance(columns, str):
        columns = [columns]
    data: pd.Series = (
        df.merge(codes.reset_index()[[id_name] + columns], on=columns, how='left')
        .set_index(df.index)[id_name]
        .fillna(0)
        .astype(np.int32)
    )
    df[id_name] = data
    df.drop(columns=columns, inplace=True)
    return df


def codify(df: pd.DataFrame, column_names: Union[str, List[str]], id_column_name: str = None) -> pd.DataFrame:
    """Create a dataframe with integer codes for each unique value in `column_name`. Return data frame."""

    if id_column_name is None:
        id_column_name = f'{column_names if isinstance(column_names, str) else "_".join(column_names)}_id'

    return (
        pd.DataFrame(df[column_names])
        .groupby(column_names)
        .size()
        .reset_index()
        .rename({0: 'items'}, axis=1)
        .rename_axis(id_column_name)
    )


def party_to_abbrev(df: pd.DataFrame) -> pd.DataFrame:
    """Create party_name, party_abbrev mapping."""
    """Some parties have multiple abbevs of which one is empty string. Hence sort and take last."""
    mapping = df.groupby(['party'])['party_abbrev'].agg(lambda x: sorted(list(set(x)))[-1]).to_dict()
    return mapping


class Gender(Enum):
    Undefined = ''
    Male = 'M'
    Female = 'K'

    @staticmethod
    def convert(x: str) -> "Gender":
        if x.lower() in ('man', 'm'):
            return Gender.Male
        if x.lower() in ('woman', 'w', 'kvinna', 'k'):
            return Gender.Female
        return Gender.Undefined

    @staticmethod
    def codes() -> pd.DataFrame:
        return pd.DataFrame({'gender': [e.name for e in Gender]}, index=[e.value for e in Gender])


@dataclass
class ParliamentaryData:

    source_data: pd.DataFrame

    members: pd.DataFrame
    genders: pd.DataFrame
    parties: pd.DataFrame
    party_abbrev: pd.DataFrame
    districts: pd.DataFrame
    chambers: pd.DataFrame
    terms_of_office: pd.DataFrame

    party_to_abbrev_map: Mapping[str, str]

    @staticmethod
    def load(source_path: str = GITHUB_DATA_URL, tag: str = None) -> "ParliamentaryData":

        if "{}" in source_path:
            source_path = source_path.format(tag or "dev")

        member_data: pd.DataFrame = (
            pd.read_csv(source_path, sep=',', index_col=None).set_index('id', drop=False).rename_axis('')
        )

        member_data = member_data[~(member_data[['id', 'name']].isna().any(axis=1))]

        member_data['party'] = member_data.party.fillna('')
        member_data['district'] = member_data.district.fillna('')
        member_data['gender'] = member_data.gender.fillna('').apply(lambda x: Gender.convert(x).value)
        member_data['chamber'] = member_data.chamber.fillna('')
        member_data['occupation'] = member_data.occupation.fillna('')
        member_data['party_abbrev'] = member_data.party_abbrev.fillna('')
        member_data['riksdagen_id'] = member_data.riksdagen_id.fillna(0).astype(np.int64)
        member_data['born'] = member_data.born.fillna(0).astype(np.int16)

        """Update party_abbrev (some mebers have empty code for encoded `party`)"""
        party_map: Mapping[str, str] = party_to_abbrev(member_data)
        member_data['party_abbrev'] = member_data.party.apply(party_map.get)

        """We don't use numeric codes for party abbrev"""
        party_abbrev = (
            codify(member_data, column_names='party_abbrev').set_index('party_abbrev', drop=False).rename_axis('')
        )

        parties = codify(member_data, column_names='party')
        districts = codify(member_data, column_names='district')
        chambers = codify(member_data, column_names='chamber')
        occupation = codify(member_data, column_names='occupation')
        terms_of_office = codify(
            member_data, column_names=['chamber', 'start', 'end'], id_column_name='terms_of_office_id'
        )
        terms_of_office = encode(terms_of_office, chambers)

        members: pd.DataFrame = pd.DataFrame(member_data)

        members = encode(members, parties)
        members = encode(members, districts)
        members = encode(members, chambers)
        members = encode(members, occupation)
        members = encode(members, terms_of_office, columns=['chamber_id', 'start', 'end'])

        data: ParliamentaryData = ParliamentaryData(
            source_data=member_data,
            members=members,
            parties=parties,
            party_abbrev=party_abbrev,
            districts=districts,
            chambers=chambers,
            terms_of_office=terms_of_office,
            party_to_abbrev_map=party_map,
            genders=Gender.codes(),
        )
        return data


__parliamentary_metadata: ParliamentaryData = None


def get__parliamentary_metadata() -> ParliamentaryData:
    global __parliamentary_metadata
    if __parliamentary_metadata is None:
        __parliamentary_metadata = ParliamentaryData.load()
    return __parliamentary_metadata
