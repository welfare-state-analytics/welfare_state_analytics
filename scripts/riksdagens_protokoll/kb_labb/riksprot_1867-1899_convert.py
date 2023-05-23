import csv
import io
import zipfile
from typing import List, Optional

import pandas as pd
from tqdm import tqdm

# pylint: disable=unsubscriptable-object)


def baseform(x: dict) -> Optional[str]:
    hist_baseform = x.get('hist.baseform')
    saldo_baseform = x.get('saldo.baseform')
    if hist_baseform:
        hist_baseform = hist_baseform.replace('dalinm--', '').replace('swedbergm--', '')
        return hist_baseform
    return saldo_baseform


def convert(source_filename: str, target_filename: str, compresslevel: int = 9):

    with zipfile.ZipFile(source_filename, "r") as sfp:

        with zipfile.ZipFile(
            target_filename, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=compresslevel
        ) as tfp:

            filenames: List[str] = sfp.namelist()

            for filename in tqdm(filenames):
                content: str = sfp.read(filename).decode('utf-8')
                tagged_frame: pd.DataFrame = pd.read_csv(
                    io.StringIO(content), sep='\t', comment='#', quoting=csv.QUOTE_NONE
                )

                tagged_frame['baseform'] = tagged_frame.apply(baseform, axis=1)
                tagged_frame = tagged_frame[['token', 'pos', 'baseform']]

                data: str = tagged_frame.to_csv(sep='\t', index=False)

                tfp.writestr(filename, data)


if __name__ == "__main__":
    source_file: str = '/home/roger/source/penelope/data/riksprot_1800.csv.zip'
    target_file: str = '/home/roger/source/penelope/data/riksdagens_protokoll_1867-1899.csv.zip'

    convert(source_file, target_file)
