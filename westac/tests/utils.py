import os
from westac.common import utility
from westac.common import file_text_reader

TEST_CORPUS_FILENAME = './westac/tests/test_data/test_corpus.zip'

if __file__ in globals():
    this_file = os.path.dirname(__file__)
    this_path = os.path.abspath(this_file)
    TEST_CORPUS_FILENAME = os.path.join(this_path, TEST_CORPUS_FILENAME)

def create_text_files_reader(
    filename=TEST_CORPUS_FILENAME,
    pattern="*.txt",
    itemfilter=None,
    compress_whitespaces=False,
    dehyphen=True,
    meta_extract=None
):
    kwargs = dict(
        pattern=pattern,
        itemfilter=itemfilter,
        compress_whitespaces=compress_whitespaces,
        dehyphen=dehyphen,
        meta_extract=meta_extract
    )
    reader = file_text_reader.FileTextReader(filename, **kwargs)
    return reader

