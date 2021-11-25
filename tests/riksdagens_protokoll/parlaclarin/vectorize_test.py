from scripts.riksdagens_protokoll.parlaclarin.vectorize import process


def test_vectorize():

    process(
        corpus_config='./resources/riksdagens-protokoll-parlaclarin.yml',
        input_folder='./tests/test_data/riksdagens_protokoll/parlaclarin/tagged_corpus_01',
        output_folder='./tests/output',
        output_tag="NEPTUNUS",
        create_subfolder=True,
        pos_includes='NN|PM|VB',
        pos_excludes='MAD|MID|PAD',
        pos_paddings="AB|DT|HA|HD|HP|HS|IE|IN|JJ|KN|PC|PL|PN|PP|PS|RG|RO|SN|UO",
        to_lower=True,
        lemmatize=True,
        remove_stopwords=None,
        min_word_length=1,
        max_word_length=None,
        keep_symbols=True,
        keep_numerals=True,
        only_any_alphanumeric=False,
        only_alphabetic=False,
        tf_threshold=None,
        merge_speeches=True,
    )
