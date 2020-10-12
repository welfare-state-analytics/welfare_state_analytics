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
