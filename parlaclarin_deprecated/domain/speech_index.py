from dataclasses import dataclass
import os
from typing import List, Literal, Tuple, get_args
import numpy as np
import pandas as pd
from penelope.pipeline import (
    CorruptCheckpointError,
    EmptyCheckpointError,
    find_checkpoints,
    read_document_index,
)
from penelope.utility import replace_extension

from .utility import to_xlsx, to_zip, to_csv, to_feather

jj = os.path.join

SpeechIndex = pd.DataFrame
SpeechIndexExtension = Literal['xlsx', 'zip', 'csv', 'feather']
SPEECH_INDEX_BASENAME = "speech_index"
SPEECH_INDEX_WRITERS: dict = dict(xlsx=to_xlsx, zip=to_zip, csv=to_csv, feather=to_feather)


@dataclass
class SpeechIndexHelper:
    value: pd.DataFrame = None

    @staticmethod
    def create(folder: str, file_pattern: str = "*.zip") -> "SpeechIndexHelper":
        value: pd.DataFrame = _create(folder, file_pattern)
        return SpeechIndexHelper(value)

    @staticmethod
    def load(folder: str) -> "SpeechIndexHelper":
        value: pd.DataFrame = _load(folder)
        return SpeechIndexHelper(value)

    def store(self, folder: str, extension: SpeechIndexExtension) -> "SpeechIndexHelper":
        _store(folder, self.value, extension)
        return self

    @staticmethod
    def exists(folder: str) -> bool:
        return any(_exists(folder, x) for x in get_args(SpeechIndexExtension))

    def overload(self, data: pd.DataFrame, field: str, key: str, right_index: bool=False) -> "SpeechIndexHelper":
        # FIXME: Make sure index stays the same
        # FIXME: If key is index
        join_keys = { 'on': key} if not right_index else {'left_in': key, 'right_index': True}
        self.value = self.value.merge(data, **join_keys, how='left')
        return self

def _create(folder: str, file_pattern: str = "*.zip") -> pd.DataFrame:
    def _parse_year(speech_date: str) -> int:
        try:
            return int(speech_date[:4])
        except:  # pylint: disable=bare-except
            return 0

    filenames: List[str] = find_checkpoints(folder, file_pattern)

    empty: List[str] = []
    failed: List[str] = []
    success: List[Tuple[str, pd.DataFrame]] = []

    for filename in filenames:
        try:
            df: pd.DataFrame = read_document_index(archive_filename=filename)
            success.append((filename, df))
        except EmptyCheckpointError:
            empty.append(filename)
        except CorruptCheckpointError:
            failed.append(filename)

    speech_index = pd.concat([x[1] for x in success], ignore_index=True)
    speech_index['document_id'] = speech_index.index.astype(np.int32)
    speech_index = speech_index.set_index('document_id', drop=False).rename_axis('')

    # speech_index['year'] = speech_index.speech_date.str[:4].astype(np.int16)
    speech_index['year'] = speech_index.speech_date.apply(_parse_year).astype(np.int16)
    speech_index['n_tokens'] = speech_index.num_tokens

    speech_index.drop(columns=['num_tokens', 'num_words'], inplace=True)

    return speech_index


def _store(folder: str, speech_index: SpeechIndex, extension: SpeechIndexExtension = "feather") -> None:
    """Store speech document index to disk"""
    target_filename: str = jj(folder, f"{SPEECH_INDEX_BASENAME}.{extension}")
    if SPEECH_INDEX_WRITERS.get(extension):
        SPEECH_INDEX_WRITERS.get(extension)(speech_index, target_filename)


def _exists(folder: str, extension: SpeechIndexExtension = "feather") -> bool:
    filename: str = jj(folder, f"{SPEECH_INDEX_BASENAME}.{extension}")
    return os.path.isfile(filename)


def _load(folder: str) -> SpeechIndex:
    loaders: dict = dict(
        feather=pd.read_feather,
        xlsx=lambda x: pd.read_excel(x, index_col=0),
        csv=lambda x: pd.read_csv(x, sep='\t', decimal=',', index_col=0),
        zip=lambda x: pd.read_csv(x, sep='\t', decimal=',', index_col=0)
    )
    for extension in loaders.keys():
        filename = jj(folder, replace_extension(SPEECH_INDEX_BASENAME, extension))
        if os.path.exists(filename):
            df: pd.DataFrame = loaders[extension](filename)
            if '' in df.columns:
                df = df.drop(columns='')
            return df
    return None
