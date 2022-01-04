import pandas as pd
from westac.riksdagens_protokoll.parlaclarin import members


def test_reset_index_base():

    df: pd.DataFrame = pd.DataFrame({'col': ['a', 'a', 'b', 'c']})

    assert members.reset_index_base(df).index.tolist() == [0, 1, 2, 3]
    assert members.reset_index_base(df, base=1).index.tolist() == [1, 2, 3, 4]
    assert members.reset_index_base(pd.DataFrame({'col': []}), base=0).index.tolist() == []
    assert members.reset_index_base(pd.DataFrame({'col': []}), base=1).index.tolist() == []


def test_codify():

    df: pd.DataFrame = pd.DataFrame({'col': ['a', 'a', 'b', 'c']}, index=[1, 2, 3, 4])

    codes = members.codify(df, 'col', base=0)
    assert codes is not None
    assert codes.index.tolist() == [0, 1, 2]
    assert codes.col.tolist() == ['a', 'b', 'c']
    assert codes.index.name == 'col_id'

    codes = members.codify(df, 'col', base=1)
    assert codes.index.tolist() == [1, 2, 3]

    assert members.codify(pd.DataFrame({'col': []}), 'col', base=1).index.tolist() == []


def test_encode():

    df: pd.DataFrame = pd.DataFrame({'col': ['a', 'a', 'b', 'c']}, index=[1, 2, 3, 4])
    codes = members.codify(df, 'col', base=0)
    df = members.encode(df, codes)
    assert df.columns == ['col_id']
    assert df.col_id.tolist() == [0, 0, 1, 2]


def test_gender():

    assert members.Gender.convert("man") == members.Gender.Male
    assert members.Gender.codes().to_dict('records') == [
        {'gender': 'Undefined'},
        {'gender': 'Male'},
        {'gender': 'Female'},
    ]


def test_load_members():

    parliament_data = members.ParliamentaryData.load(members.GITHUB_DATA_URL, tag=None)

    assert parliament_data is not None
    parliament_data = members.ParliamentaryData.load(members.GITHUB_DATA_URL)

    assert len(parliament_data.members) == len(parliament_data.source_data)

    assert set(parliament_data.members.columns) == {
        'riksdagen_id',
        'gender',
        'twittername',
        'terms_of_office_id',
        'born',
        'id',
        'district_id',
        'name',
        'specifier',
        'party_abbrev',
        'party_id',
        'occupation_id',
    }

    df = parliament_data.members[['name', 'gender', 'id', 'party_abbrev']]

    assert not df.isna().any().any()

    assert set(parliament_data.members.gender.unique()) == {'', 'M', 'K'}
    assert set(parliament_data.members.gender.unique()) == set(parliament_data.genders.index)
    assert set(parliament_data.parties.index) == set(parliament_data.members.party_id)
    assert set(parliament_data.party_abbrev.index) == set(parliament_data.members.party_abbrev)

    assert members.get_parliamentary_metadata() is not None
