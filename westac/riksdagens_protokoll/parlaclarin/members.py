from dataclasses import dataclass
from enum import IntEnum
from typing import Mapping
import pandas as pd

GITHUB_DATA_URL = (
    "https://raw.githubusercontent.com/welfare-state-analytics/riksdagen-corpus/{}/corpus/members_of_parliament.csv"
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
    def load(source_path: str = GITHUB_DATA_URL) -> "ParliamentaryData":

        members = pd.read_csv(source_path, sep=',', index_col=None).set_index('id', drop=False)

        parties = members.groupby('party').size().reset_index()
        districts = members.groupby('district').size().reset_index()
        chambers = members.groupby('chamber').size().reset_index()
        terms_of_office = members.groupby(['chamber', 'start', 'end']).size().reset_index()
        genders = list(members.gender.unique())

        data: ParliamentaryData = ParliamentaryData(
            members=members,
            parties=parties,
            districts=districts,
            chambers=chambers,
            terms_of_office=terms_of_office,
            genders=genders,
        )
        return data


__parliamentary_metadata: ParliamentaryData = None


def get__parliamentary_metadata() -> ParliamentaryData:
    global __parliamentary_metadata
    if __parliamentary_metadata is None:
        __parliamentary_metadata = ParliamentaryData.load()
    return __parliamentary_metadata
