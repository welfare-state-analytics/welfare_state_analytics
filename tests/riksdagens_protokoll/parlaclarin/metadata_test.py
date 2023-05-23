import pandas as pd

from westac.riksprot.parlaclarin import codecs as md

METADATA_FILENAME: str = './tests/test_data/riksprot/main/riksprot_metadata.db'
EXPECTED_TABLES: set[str] = {
    'chamber',
    'gender',
    'office_type',
    'sub_office_type',
    'persons_of_interest',
    'government',
    "party",
}


def test_load_data():
    data: md.Codecs = md.Codecs().load(source=METADATA_FILENAME)
    assert data is not None
    assert set(data.tablenames().keys()) == (EXPECTED_TABLES - {'persons_of_interest'})

    data: md.Codecs = md.PersonCodecs().load(source=METADATA_FILENAME)
    assert data is not None
    assert set(data.tablenames().keys()) == EXPECTED_TABLES


def test_load_riksprot_netadata():
    person_codecs: md.PersonCodecs = md.PersonCodecs().load(source=METADATA_FILENAME)

    assert isinstance(person_codecs, md.PersonCodecs)
    assert all(isinstance(getattr(person_codecs, k), pd.DataFrame) for k in EXPECTED_TABLES)

    assert set(person_codecs.gender.columns) == {"gender"}
    assert person_codecs.gender.index.name == "gender_id"
    # assert person_codecs.gender.index.dtype == np.int8
    assert set(person_codecs.chamber.columns) == {"chamber"}
    assert set(person_codecs.office_type.columns) == {"role", "office"}
    assert set(person_codecs.sub_office_type.columns) == {
        'identifier',
        'description',
        'chamber_id',
        'office_type_id',
    }

    assert person_codecs.pid2person_id.get(2) == 'Q5005171'
    assert person_codecs.person_id2pid.get('Q5005171') == 2
    assert person_codecs.pid2person_name.get(2) == 'Börje Hörnlund'
    assert person_codecs.person_id2name.get('Q5005171') == 'Börje Hörnlund'

    # Q5005171      Börje Hörnlund    2

    # def person_id2name(self) -> dict[str, str]:
    #     fg = self.pid2person_id.get
    #     return {fg(pid): name for pid, name in self.pid2person_name.items()}

    # assert set(person_codecs..columns) == {
    #     'end_year',
    #     'office_type_id',
    #     'sub_office_type_id',
    #     'start_year',
    #     'person_id',
    #     'end_date',
    #     'district_id',
    #     'start_date',
    # }
