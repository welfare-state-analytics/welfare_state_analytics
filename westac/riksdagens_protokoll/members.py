from dataclasses import dataclass, field
from enum import IntEnum
from typing import Mapping
import pandas as pd

GITHUB_DATA_URL = (
    "https://raw.githubusercontent.com/welfare-state-analytics/riksdagen-corpus/dev/corpus/members_of_parliament.csv"
)


class Gender(IntEnum):
    Undefined = 0
    Male = 1
    Female = 2


@dataclass
class ParliamentaryData:

    members: pd.DataFrame
    parties: pd.DataFrame
    districts: pd.DataFrame
    chambers: pd.DataFrame
    terms_of_office: pd.DataFrame
    genders: Mapping[str, str]

    @staticmethod
    def load(source_path: str = GITHUB_DATA_URL, branch: str='dev') -> "ParliamentaryData":

        members = pd.read_csv(source_path, sep=',', index_col=None).set_index('id', drop=False)

        parties = members.groupby('party').size().reset_index()
        districts = members.groupby('district').size().reset_index()
        chambers = members.groupby('chamber').size().reset_index()
        terms_of_office = members.groupby(['chamber', 'start', 'end']).size().reset_index()
        genders = list(members.gender.unique())

        data = ParliamentaryData(
            members=members,
            parties=parties,
            districts=districts,
            chambers=chambers,
            terms_of_office=terms_of_office,
            genders=genders,
        )
        return data


@dataclass
class LazyLoader:

    _data: ParliamentaryData = field(init=False, default=None)

    @property
    def data(self):
        if self._data is None:
            self._data = ParliamentaryData.load()
        return self._data

DATA =
def get_parla_data():

parla_data: LazyLoader = LazyLoader()
