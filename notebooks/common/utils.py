from typing import Dict, List, Tuple

import pandas as pd
from penelope.utility import flatten


def setup_pandas():

    pd.set_option("max_rows", None)
    pd.set_option("max_columns", None)
    pd.set_option('colheader_justify', 'left')
    pd.set_option('max_colwidth', 300)


def to_text(document: List[Tuple[int, int]], id2token: Dict[int, str]):
    return ' '.join(flatten([f * [id2token[token_id]] for token_id, f in document]))


# from zipfile import ZipFile, ZipInfo

# class ZipFileExt(ZipFile):

#     class ZipFileIterator:

#         def __init__(self, zipfile: ZipFile):
#             self.zipfile = zipfile
#             self.zipinfos: List[ZipInfo] = zipfile.infolist()
#             self.zipiter: Iterator[ZipInfo] = iter(self.zipinfos)

#         def __iter__(self):
#             return self

#         def __next__(self):
#             next_info = next(self.zipiter)
#             if next_info is None:
#                 raise StopIteration
#             return self._zipfile.open(next_info)

#     def __iter__(self):
#         return self.ZipFileIterator(self)
