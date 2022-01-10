from io import StringIO
from typing import List

import pandas as pd
import pytest
from penelope import corpus as pc
from westac.riksprot.parlaclarin.metadata import ProtoMetaData

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


SAMPLE_MEMBERS: str = [
    'id\trole_type\tborn\tchamber\tdistrict\tstart\tend\tgender\tname\toccupation\tparty\tparty_abbrev',
    'urban_ahlin_talman\ttalman\t0\tEnkammarriksdagen\t\t2014\t2018\tunknown\tUrban Ahlin\t\ts\t',
    'axel_vennersten_talman\ttalman\t0\tFörsta kammaren\t\t1928\t1936\tunknown\tAxel Vennersten\t\tm\t',
    'ingemar_mundebo_minister_1979\tminister\t0\tgov\t\t1979\t1980\tunknown\tIngemar Mundebo\t\tgovernment\tgov',
    'camilla_odhnoff_minister_1967\tminister\t0\tgov\t\t1967\t1969\tunknown\tCamilla Odhnoff\t\tgovernment\tgov',
    'gosta_tore_edvin_bengtsson_a62bb1\tmember\t0\t\t\t1954\t1954\tman\tGösta Tore Edvin Bengtsson\t\tunknown\tgov',
    'ingrid_andersson_89b1a7\tmember\t0\tEnkammarriksdagen\tUppsala län\t1994\t1998\twoman\tIngrid Andersson\t\tSocialdemokraterna\tS',
    'lennart_blom_ef2fae\tmember\t0\tFörsta kammaren\tSkaraborgs län\t1966\t1972\tman\tLennart Blom\t\tFolkpartiet Liberalerna\tL',
    'gustaf_nilsson_c84a21\tmember\t0\tFörsta kammaren\tKristianstads län\t1918\t1924\tman\tGustaf Nilsson\t\tSocialdemokraterna\tS',
    'ernst_johannes_wigforss_72ff72\tmember\t0\t\t\t1929\t1929\tman\tErnst Johannes Wigforss\t\tunknown\tgov',
    'bertil_georg_rubin_bb6619\tmember\t0\t\t\t1968\t1968\tman\tBertil Georg Rubin\t\tunknown\tgov',
    'oscar_ludvig_carlstrom_67d31b\tmember\t0\t\t\t1945\t1945\tman\tOscar Ludvig Carlström\t\tunknown\tgov',
    'albert_ramberg_9ae24e\tmember\t0\t\t\t1951\t1951\tman\tAlbert Ramberg\t\tunknown\tgov',
    'ernst_trygger_59d75d\tmember\t0\t\t\t1926\t1926\tman\tErnst Trygger\t\tunknown\tgov',
    'harry_walfrid_august_weibull_db2b0e\tmember\t0\t\t\t1928\t1928\tman\tHarry Walfrid August Weibull\t\tunknown\tgov',
]

DOCUMENT_INDEX: str = """\tyear\tdocument_name\tfilename\tn_tokens\twho\tdocument_id\tAdjective\tAdverb\tConjunction\tDelimiter\tNoun\tNumeral\tOther\tPreposition\tPronoun\tVerb\tn_raw_tokens
1959_karl_jonsson_c519a7\t1959\t1959_karl_jonsson_c519a7\t1959_karl_jonsson_c519a7.csv\t458\tkarl_jonsson_c519a7\t11239\t19\t50\t25\t0\t110\t16\t0\t53\t81\t104\t458
1960_karl_jonsson_c519a7\t1960\t1960_karl_jonsson_c519a7\t1960_karl_jonsson_c519a7.csv\t1110\tkarl_jonsson_c519a7\t11513\t52\t124\t77\t0\t255\t21\t2\t114\t196\t269\t1110
1963_lars_lindahl_d4fc9f\t1963\t1963_lars_lindahl_d4fc9f\t1963_lars_lindahl_d4fc9f.csv\t376\tlars_lindahl_d4fc9f\t12549\t19\t44\t19\t0\t96\t4\t0\t49\t74\t71\t376
1976_elvy_olsson_minister_1976\t1976\t1976_elvy_olsson_minister_1976\t1976_elvy_olsson_minister_1976.csv\t9487\telvy_olsson_minister_1976\t16134\t587\t1044\t617\t0\t2452\t69\t0\t1265\t1506\t1947\t9487
1977_elvy_olsson_minister_1976\t1977\t1977_elvy_olsson_minister_1976\t1977_elvy_olsson_minister_1976.csv\t3395\telvy_olsson_minister_1976\t16499\t242\t272\t173\t0\t1073\t134\t1\t525\t342\t633\t3395
1978_elvy_olsson_minister_1976\t1978\t1978_elvy_olsson_minister_1976\t1978_elvy_olsson_minister_1976.csv\t927\telvy_olsson_minister_1976\t17009\t69\t98\t50\t0\t245\t7\t0\t142\t118\t198\t927
1987_christer_eirefelt_2_vice_talman\t1987\t1987_christer_eirefelt_2_vice_talman\t1987_christer_eirefelt_2_vice_talman.csv\t178\tchrister_eirefelt_2_vice_talman\t20417\t2\t26\t10\t0\t48\t2\t0\t25\t20\t45\t178
1988_christer_eirefelt_2_vice_talman\t1988\t1988_christer_eirefelt_2_vice_talman\t1988_christer_eirefelt_2_vice_talman.csv\t4920\tchrister_eirefelt_2_vice_talman\t20627\t295\t685\t378\t0\t1163\t53\t3\t566\t864\t913\t4920
1989_christer_eirefelt_2_vice_talman\t1989\t1989_christer_eirefelt_2_vice_talman\t1989_christer_eirefelt_2_vice_talman.csv\t2889\tchrister_eirefelt_2_vice_talman\t21124\t182\t369\t207\t0\t693\t23\t2\t331\t520\t562\t2889
1990_christer_eirefelt_2_vice_talman\t1990\t1990_christer_eirefelt_2_vice_talman\t1990_christer_eirefelt_2_vice_talman.csv\t513\tchrister_eirefelt_2_vice_talman\t21506\t13\t49\t39\t0\t154\t14\t0\t65\t52\t127\t513
1991_christer_eirefelt_2_vice_talman\t1991\t1991_christer_eirefelt_2_vice_talman\t1991_christer_eirefelt_2_vice_talman.csv\t1154\tchrister_eirefelt_2_vice_talman\t21790\t52\t145\t68\t0\t280\t17\t0\t126\t192\t274\t1154
1992_christer_eirefelt_2_vice_talman\t1992\t1992_christer_eirefelt_2_vice_talman\t1992_christer_eirefelt_2_vice_talman.csv\t1585\tchrister_eirefelt_2_vice_talman\t22140\t66\t167\t88\t0\t461\t26\t0\t205\t253\t319\t1585
1993_christer_eirefelt_2_vice_talman\t1993\t1993_christer_eirefelt_2_vice_talman\t1993_christer_eirefelt_2_vice_talman.csv\t1128\tchrister_eirefelt_2_vice_talman\t22521\t48\t129\t62\t0\t322\t29\t0\t179\t140\t219\t1128
2003_bjorn_von_sydow_talman\t2003\t2003_bjorn_von_sydow_talman\t2003_bjorn_von_sydow_talman.csv\t28\tbjorn_von_sydow_talman\t25802\t2\t7\t2\t0\t5\t0\t0\t3\t3\t6\t28
2004_bjorn_von_sydow_talman\t2004\t2004_bjorn_von_sydow_talman\t2004_bjorn_von_sydow_talman.csv\t243\tbjorn_von_sydow_talman\t26191\t12\t19\t13\t0\t64\t2\t0\t32\t45\t56\t243
2005_bjorn_von_sydow_talman\t2005\t2005_bjorn_von_sydow_talman\t2005_bjorn_von_sydow_talman.csv\t483\tbjorn_von_sydow_talman\t26587\t32\t48\t22\t0\t130\t5\t0\t61\t81\t104\t483
2006_bjorn_von_sydow_talman\t2006\t2006_bjorn_von_sydow_talman\t2006_bjorn_von_sydow_talman.csv\t99\tbjorn_von_sydow_talman\t26718\t3\t11\t10\t0\t21\t0\t0\t9\t20\t25\t99
2015_anna_kinberg_batra_68e88e\t2015\t2015_anna_kinberg_batra_68e88e\t2015_anna_kinberg_batra_68e88e.csv\t13245\tanna_kinberg_batra_68e88e\t29793\t797\t1724\t868\t0\t3179\t186\t4\t1444\t2228\t2815\t13245
2015_jimmie_akesson_5a69d8\t2015\t2015_jimmie_akesson_5a69d8\t2015_jimmie_akesson_5a69d8.csv\t19931\tjimmie_akesson_5a69d8\t29794\t1135\t2771\t1374\t0\t4350\t125\t12\t2035\t3974\t4155\t19931
2016_anna_kinberg_batra_68e88e\t2016\t2016_anna_kinberg_batra_68e88e\t2016_anna_kinberg_batra_68e88e.csv\t11558\tanna_kinberg_batra_68e88e\t30089\t692\t1486\t870\t0\t2716\t136\t17\t1251\t1981\t2409\t11558
2016_jimmie_akesson_5a69d8\t2016\t2016_jimmie_akesson_5a69d8\t2016_jimmie_akesson_5a69d8.csv\t17247\tjimmie_akesson_5a69d8\t30127\t962\t2387\t1163\t0\t3804\t137\t21\t1763\t3399\t3611\t17247
2017_jimmie_akesson_5a69d8\t2017\t2017_jimmie_akesson_5a69d8\t2017_jimmie_akesson_5a69d8.csv\t11574\tjimmie_akesson_5a69d8\t30439\t693\t1494\t807\t0\t2604\t96\t4\t1111\t2247\t2518\t11574
2017_anna_kinberg_batra_68e88e\t2017\t2017_anna_kinberg_batra_68e88e\t2017_anna_kinberg_batra_68e88e.csv\t1400\tanna_kinberg_batra_68e88e\t30454\t85\t228\t96\t0\t295\t31\t0\t154\t234\t277\t1400
2018_jimmie_akesson_5a69d8\t2018\t2018_jimmie_akesson_5a69d8\t2018_jimmie_akesson_5a69d8.csv\t12960\tjimmie_akesson_5a69d8\t30563\t816\t1685\t907\t0\t2976\t91\t22\t1296\t2576\t2591\t12960
"""

ID_COLUMNS: List[str] = ['who_id', 'gender_id', 'party_abbrev_id', 'role_type_id']
NAME_COLUMNS: List[str] = ['gender', 'party_abbrev', 'role_type']


@pytest.fixture
def members() -> pd.DataFrame:
    filename: str = 'tests/test_data/person_index.feather'
    return ProtoMetaData.read_members(filename)


@pytest.fixture
def document_index() -> pd.DataFrame:
    return pc.DocumentIndexHelper.load(StringIO(DOCUMENT_INDEX)).document_index


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
