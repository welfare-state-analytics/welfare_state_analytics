import os
from typing import List, Literal, Tuple
import numpy as np
import pandas as pd
from penelope.pipeline import (
    CorruptCheckpointError,
    EmptyCheckpointError,
    find_checkpoints,
    read_document_index,
)
from penelope.utility import replace_extension, strip_path_and_extension

SpeechIndex = pd.DataFrame
SpeechIndexExtension = Literal["feather", "csv", "excel"]

SPEECH_INDEX_BASENAME = "speech_index"


def create_speech_index(folder: str, file_pattern: str = "*.zip") -> pd.DataFrame:

    def _parse_year(speech_date: str) -> int:
        try:
            return int(speech_date[:4])
        except: # pylint: disable=bare-except
            return 0

    filenames: List[str] = find_checkpoints(folder, file_pattern)

    empty: List[str] = []
    failed: List[str] = []
    success: List[Tuple[str, pd.DataFrame]] = []

    for filename in filenames:
        try:
            df: pd.DataFrame = read_document_index(checkpoint_filename=filename)
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


def store_speech_index(folder: str, speech_index: SpeechIndex, extension: SpeechIndexExtension = "feather") -> None:
    """Store speech index to disk"""
    target_filename: str = os.path.join(folder, f"{SPEECH_INDEX_BASENAME}.{extension}")

    if extension == "excel":
        speech_index.to_excel(target_filename)

    elif extension == "csv":
        """Store as zipped tab-seperated file"""

        archive_name: str = f"{strip_path_and_extension(target_filename)}.csv"
        compression: dict = dict(method='zip', archive_name=archive_name)
        target_filename = replace_extension(target_filename, "zip")
        speech_index.to_csv(target_filename, compression=compression, sep='\t', header=True, decimal=',')

    else:
        if len(speech_index) == 0:
            speech_index.reset_index(drop=True, inplace=True)

        speech_index.to_feather(target_filename, compression="lz4")


def speech_index_exists(folder: str, extension: SpeechIndexExtension = "feather") -> SpeechIndex:
    filename: str = os.path.join(folder, f"{SPEECH_INDEX_BASENAME}.{extension}")
    return os.path.isfile(filename)


def read_speech_index(folder: str) -> SpeechIndex:
    filename = os.path.join(folder, replace_extension(SPEECH_INDEX_BASENAME, ".feather"))
    speech_index: SpeechIndex = pd.read_feather(filename)
    return speech_index
