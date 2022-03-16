# from io import StringIO

# import pandas as pd
# import pytest

# from westac.riksprot.parlaclarin import metadata as md

# from .fixtures import ID_COLUMNS, NAME_COLUMNS, SAMPLE_MEMBERS, sample_document_index

# # pylint: disable=redefined-outer-name


# def sample_members(folder: str) -> None:
#     filename: str = 'tests/test_data/person_index.feather'
#     mps: pd.DataFrame = md.IRiksprotMetaData.load_members(folder)
#     df = pd.concat(
#         [
#             mps[mps.role_type == 'talman'].sample(n=2),
#             mps[mps.role_type == 'minister'].sample(n=2),
#             mps[mps.role_type == 'member'].sample(n=10),
#         ]
#     )
#     df.reset_index().to_feather(filename)


# @pytest.fixture
# def members() -> pd.DataFrame:
#     filename: str = 'tests/test_data/person_index.feather'
#     return md.IRiksprotMetaData.read_members(filename)


# @pytest.fixture
# def document_index() -> pd.DataFrame:
#     return sample_document_index()


# def test_members_load():

#     mps: pd.DataFrame = md.IRiksprotMetaData.load_members(StringIO('\n'.join(SAMPLE_MEMBERS)))
#     assert isinstance(mps, pd.DataFrame)
#     assert len(mps) == 15

#     mps: pd.DataFrame = md.IRiksprotMetaData.load_members('tests/test_data/person_index.feather')
#     assert isinstance(mps, pd.DataFrame)
#     assert len(mps) == 15

#     mps: pd.DataFrame = md.IRiksprotMetaData.load_members('tests/test_data')
#     assert isinstance(mps, pd.DataFrame)
#     assert len(mps) == 15


# def test_create_metadata(members):

#     riksprot_metadata: md.IRiksprotMetaData = md.IRiksprotMetaData.load(members=members)

#     assert riksprot_metadata is not None
#     assert len(riksprot_metadata.members) == 15
#     assert len(riksprot_metadata.role_type2id) == 4 == len(riksprot_metadata.role_type2name)
#     assert len(riksprot_metadata.gender2id) == 3 == len(riksprot_metadata.gender2name) == len(riksprot_metadata.genders)
#     assert len(riksprot_metadata.party_abbrevs) == 5

#     assert not riksprot_metadata.members.party_abbrev.isna().any()
#     assert not riksprot_metadata.members.gender.isna().any()
#     assert not riksprot_metadata.members.role_type.isna().any()
#     assert not riksprot_metadata.party_abbrevs.party_abbrev.isna().any()
#     assert not riksprot_metadata.genders.gender.isna().any()

#     assert riksprot_metadata.who2id['hans_blix_minister_1978'] == 3
#     assert riksprot_metadata.id2who[3] == 'hans_blix_minister_1978'

#     assert riksprot_metadata.gender2id['woman'] == 2
#     assert riksprot_metadata.gender2name[2] == 'woman'

#     assert riksprot_metadata.role_type2id['talman'] == 1
#     assert riksprot_metadata.role_type2name[1] == 'talman'


# def test_encoded_members(members):

#     riksprot_metadata: md.IRiksprotMetaData = md.IRiksprotMetaData.load(members=members)

#     em: pd.DataFrame = riksprot_metadata.encoded_members

#     assert em is not None

#     assert len(em) == len(riksprot_metadata.members)
#     assert set(ID_COLUMNS).intersection(set(em.columns)) == set(ID_COLUMNS)
#     assert set(NAME_COLUMNS).intersection(set(em.columns)) == set()

#     em_back = riksprot_metadata.decode_members_data(em, drop=True)

#     assert len(em_back) == len(riksprot_metadata.members)
#     assert set(ID_COLUMNS).intersection(set(em_back.columns)) == set()
#     assert set(NAME_COLUMNS).intersection(set(em_back.columns)) == set(NAME_COLUMNS)


# def test_overload_by_member_data(members, document_index):

#     riksprot_metadata: md.IRiksprotMetaData = md.IRiksprotMetaData.load(members=members)

#     dx: pd.DataFrame = riksprot_metadata.overload_by_member_data(document_index)

#     assert dx is not None

#     assert set(ID_COLUMNS).intersection(set(dx.columns)) == set(ID_COLUMNS)

#     dx: pd.DataFrame = riksprot_metadata.overload_by_member_data(
#         document_index, encoded=True, drop=True, columns='gender_id'
#     )
#     assert set(ID_COLUMNS).intersection(set(dx.columns)) == set(['gender_id'])

#     dx: pd.DataFrame = riksprot_metadata.overload_by_member_data(
#         document_index, encoded=False, drop=True, columns='gender'
#     )
#     assert set(NAME_COLUMNS).intersection(set(dx.columns)) == set(['gender'])
