from westac.riksdagens_protokoll.parlaclarin import members


def test_load_members():

    parliament_data = members.ParliamentaryData.load(members.GITHUB_DATA_URL, tag=None)

    assert parliament_data is not None
    parliament_data = members.ParliamentaryData.load(members.GITHUB_DATA_URL)

    assert set(parliament_data.members.gender.unique()) == {'', 'M', 'K'}
