from io import StringIO

import pandas as pd
import pytest

from westac.riksprot.parlaclarin.metadata import ProtoMetaData

from .fixtures import ID_COLUMNS, NAME_COLUMNS, SAMPLE_MEMBERS, sample_document_index

# pylint: disable=redefined-outer-name


def sample_members(folder: str) -> None:
    filename: str = 'tests/test_data/person_index.feather'
    mps: pd.DataFrame = ProtoMetaData.load_members(folder)
    df = pd.concat(
        [
            mps[mps.role_type == 'talman'].sample(n=2),
            mps[mps.role_type == 'minister'].sample(n=2),
            mps[mps.role_type == 'member'].sample(n=10),
        ]
    )
    df.reset_index().to_feather(filename)


@pytest.fixture
def members() -> pd.DataFrame:
    filename: str = 'tests/test_data/person_index.feather'
    return ProtoMetaData.read_members(filename)


@pytest.fixture
def document_index() -> pd.DataFrame:
    return sample_document_index()


def test_members_load():

    mps: pd.DataFrame = ProtoMetaData.load_members(StringIO('\n'.join(SAMPLE_MEMBERS)))
    assert isinstance(mps, pd.DataFrame)
    assert len(mps) == 14

    mps: pd.DataFrame = ProtoMetaData.load_members('tests/test_data/person_index.feather')
    assert isinstance(mps, pd.DataFrame)
    assert len(mps) == 14

    mps: pd.DataFrame = ProtoMetaData.load_members('tests/test_data')
    assert isinstance(mps, pd.DataFrame)
    assert len(mps) == 14


def test_create_metadata(members, document_index):

    md: ProtoMetaData = ProtoMetaData(members=members, document_index=document_index)

    assert md is not None
    assert len(md.members) == 14
    assert len(md.document_index) == 24
    assert len(md.role_type2id) == 4 == len(md.role_type2name)
    assert len(md.gender2id) == 3 == len(md.gender2name) == len(md.genders)
    assert len(md.party_abbrevs) == 5

    assert not md.members.party_abbrev.isna().any()
    assert not md.members.gender.isna().any()
    assert not md.members.role_type.isna().any()
    assert not md.party_abbrevs.party_abbrev.isna().any()
    assert not md.genders.gender.isna().any()

    assert not md.document_index.who.isna().any()

    assert md.who2id['hans_blix_minister_1978'] == 3
    assert md.who2name[3] == 'hans_blix_minister_1978'

    assert md.gender2id['woman'] == 2
    assert md.gender2name[2] == 'woman'

    assert md.role_type2id['talman'] == 1
    assert md.role_type2name[1] == 'talman'


def test_encoded_members(members, document_index):

    md: ProtoMetaData = ProtoMetaData(members=members, document_index=document_index)

    em: pd.DataFrame = md.encoded_members

    assert em is not None

    assert len(em) == len(md.members)
    assert set(ID_COLUMNS).intersection(set(em.columns)) == set(ID_COLUMNS)
    assert set(NAME_COLUMNS).intersection(set(em.columns)) == set()

    em_back = md.decode_members_data(em, drop=True)

    assert len(em_back) == len(md.members)
    assert set(ID_COLUMNS).intersection(set(em_back.columns)) == set()
    assert set(NAME_COLUMNS).intersection(set(em_back.columns)) == set(NAME_COLUMNS)
    assert em_back.equals(md.members[em_back.columns.to_list()])


def test_overload_by_member_data(members, document_index):

    md: ProtoMetaData = ProtoMetaData(members=members, document_index=document_index)

    dx: pd.DataFrame = md.overloaded_document_index

    assert dx is not None

    assert set(ID_COLUMNS).intersection(set(dx.columns)) == set(ID_COLUMNS)

    dx: pd.DataFrame = md.overload_by_member_data(md.document_index, encoded=True, drop=True, columns='gender_id')
    assert set(ID_COLUMNS).intersection(set(dx.columns)) == set(['gender_id'])

    dx: pd.DataFrame = md.overload_by_member_data(md.document_index, encoded=False, drop=True, columns='gender')
    assert set(NAME_COLUMNS).intersection(set(dx.columns)) == set(['gender'])
