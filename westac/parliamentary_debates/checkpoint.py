import os
import zipfile
from io import StringIO
from multiprocessing import Pool
from pathlib import Path
from typing import Callable, Iterable, List, Tuple

import pandas as pd
from loguru import logger
from penelope.corpus import TextReaderOpts
from penelope.pipeline import DocumentPayload, checkpoint as cp
from tqdm.auto import tqdm


def process_document_file(args: List[Tuple]) -> DocumentPayload:

    filename, content, serializer, checkpoint_opts = args
    return DocumentPayload(
        content_type=checkpoint_opts.content_type,
        content=serializer.deserialize(content, checkpoint_opts),
        filename=filename,
    )


def parallell_deserialized_payload_stream(
    source_name: str, checkpoint_opts: cp.CheckpointOpts, filenames: List[str]
) -> Iterable[DocumentPayload]:
    """Yields a deserialized payload stream read from given source"""

    serializer: cp.IContentSerializer = cp.create_serializer(checkpoint_opts)

    with zipfile.ZipFile(source_name, mode="r") as zf:
        args: str = [
            (filename, zf.read(filename).decode(encoding='utf-8'), serializer, checkpoint_opts)
            for filename in filenames
        ]

    with Pool(processes=4) as executor:
        payloads_futures: Iterable[DocumentPayload] = executor.map(process_document_file, args)

        for payload in payloads_futures:
            yield payload


class ParlaCsvContentSerializer(cp.IContentSerializer):
    def serialize(self, content: pd.DataFrame, options: cp.CheckpointOpts) -> str:
        return content.to_csv(sep=options.sep, header=True)

    def deserialize(self, content: str, options: cp.CheckpointOpts) -> pd.DataFrame:
        data: pd.DataFrame = pd.read_csv(
            StringIO(content),
            sep=options.sep,
            quoting=options.quoting,
            index_col=options.index_column,
            keep_default_na=False,
            dtype=str,
            engine="c",
            usecols=[0, 1, 2],
        )
        data.fillna("", inplace=True)
        if any(x not in data.columns for x in options.columns):
            raise ValueError(f"missing columns: {', '.join([x for x in options.columns if x not in data.columns])}")
        return data[options.columns]


def load_checkpoints(
    source_folder: str,
    file_pattern: str,
    checkpoint_opts: cp.CheckpointOpts,
    checkpoint_filter: Callable[[str], bool] = None,
    reader_opts: TextReaderOpts = None,
    show_progress: bool = False,
) -> Iterable[cp.CheckpointData]:
    """Returns a CheckpointData stream of files (recursively) in `source_folder` that matches `file_pattern`
        and for which `checkpoint_filter`, given filename, returns true. Empty files are skipped.

    Args:
        source_folder (str): Source root folder
        file_pattern (str): Pattern to match
        checkpoint_opts (CheckpointOpts): Checkpoint (serialization) options
        checkpoint_filter (Callable[[str], bool]): Checkpoint filename predicate filter
        payload_filter (Callable[[DocumentPayload], bool]): Document payload filter

    Returns:
        Iterable[cp.CheckpointData]: Stream of Checkpoints

    """
    filenames: Iterable[str] = sorted(Path(source_folder).rglob(file_pattern))
    if show_progress:
        filenames: List[str] = list(filenames)
        filenames = tqdm(filenames, total=len(filenames))

    for path in filenames:

        if checkpoint_filter is not None:
            if not checkpoint_filter(path):
                continue

        if not zipfile.is_zipfile(path):
            if os.path.getsize(str(path)) != 0:
                logger.warning(f"skipping {path} (not a ZIP file)")
            continue

        checkpoint: cp.CheckpointDat = cp.load_checkpoint(
            path,
            checkpoint_opts=checkpoint_opts,
            reader_opts=reader_opts,
            deserialize_stream=parallell_deserialized_payload_stream,
        )
        yield checkpoint