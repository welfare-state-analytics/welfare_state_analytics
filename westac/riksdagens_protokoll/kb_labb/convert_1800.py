import io
import zipfile
from typing import List

import pandas as pd


def convert(source_filename: str, target_filename: str):

    with zipfile.ZipFile(source_filename, "r") as sfp:

        with zipfile.ZipFile(target_filename, "w") as tfp:

            filenames: List[str] = sfp.namelist()

            for filename in filenames:

                tagged_frame: pd.DataFrame = pd.read_csv(
                    io.StringIO(sfp.read(filename).decode('utf-8')), sep='\t', index_col=None
                )

                tagged_frame['baseform'] = tagged_frame.apply(
                    lambda x: x.get('hist.baseform') if x.get('hist.baseform') else x.get('saldo.baseform')
                )

                data: str = tagged_frame[['token', 'pos', 'baseform']].to_csv(sep='\t', index=False)

                tfp.writestr(filename, data, compresslevel=zipfile.ZIP_DEFLATED)
