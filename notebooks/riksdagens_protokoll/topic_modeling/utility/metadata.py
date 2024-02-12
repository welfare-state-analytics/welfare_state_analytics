from __future__ import annotations

import re
from os.path import basename, dirname, exists, isfile
from os.path import join as jj
from typing import Any

import pandas as pd
from penelope.pipeline import CorpusConfig

import westac.riksprot.parlaclarin.speech_text as sr
from westac.riksprot.parlaclarin import codecs as md


def _probe_corpus_version(folder: str) -> str:
    """Probe version of corpus from folder name or version in config file"""

    for filename in ['version', 'VERSION', 'corpus_version', 'CORPUS_VERSION']:
        if isfile(jj(folder, filename)):
            with open(jj(folder, filename), 'r') as fp:
                return fp.read().strip()

    config: CorpusConfig | None = None
    for config_filename in [jj(folder, 'corpus.yml'), jj(folder, 'config.yml')]:
        if isfile(config_filename):
            config = CorpusConfig.load(config_filename)

    if config:

        if config and config.corpus_version:
            return config.corpus_version

        if isinstance(config.pipeline_payload.source, str):
            if match := re.match(r'.*_(v\d+\.\d+\.\d+)_?.*', config.pipeline_payload.source):
                return match.group(1)

    if re.match(r'v\d+\.\d+\.\d+', dirname(str(folder))):
        return dirname(folder)

    if match := re.match(r'.*_(v\d+\.\d+\.\d+)_?.*', basename(str(folder))):
        return match.group(1)

    return None


def _probe_paths(candidates: list[str], raise_if_missing: bool = True) -> str:
    path: str = next((x for x in candidates if exists(x)), None)
    if raise_if_missing and path is None:
        raise FileNotFoundError(f"Could not find any of {candidates}")
    return path


def _probe_vrt_folder(folder: str, version: str) -> str:

    return _probe_paths(
        [
            jj(folder, 'corpus', version, f"tagged_frames_{version}"),
            jj(folder, 'corpus', version, "tagged_frames"),
            jj(folder, 'corpus' f'tagged_frames_{version}'),
            jj(folder, version, 'tagged_frames'),
            jj(folder, version, 'corpus/tagged_frames'),
            jj(folder, version, f'corpus/tagged_frames_{version}'),
        ]
    )


def _probe_metadata_filename(folder: str, version: str) -> str:

    return _probe_paths(
        [
            jj(folder, f"metadata/riksprot_metadata.{version}.db"),
            jj(folder, f"metadata/{version}/riksprot_metadata.{version}.db"),
            jj(folder, f"metadata/{version}/riksprot_metadata.db"),
        ]
    )


def _probe_document_index(folder: str, version: str) -> pd.DataFrame:
    # FIXME: better probing
    filename: str = _probe_paths(
        [
            jj(folder, "corpus", version, f"tagged_frames_{version}_speeches.feather/document_index.feather"),
            jj(folder, "corpus", version, 'tagged_frames_speeches.feather/document_index.feather'),
            jj(folder, f'tagged_frames.feather_{version}_speeches/document_index.feather'),
        ]
    )
    di: pd.DataFrame = pd.read_feather(filename)
    return di


def load_metadata(root_folder: str, folder: str) -> dict[str, Any]:

    version: str = _probe_corpus_version(folder)
    if version is None:
        raise FileNotFoundError(f"Could not find version for topic model in {folder}")

    metadata_filename: str = _probe_metadata_filename(root_folder, version)
    document_index: pd.DataFrane = _probe_document_index(root_folder, version)
    vrt_folder: str = _probe_vrt_folder(root_folder, version)
    if vrt_folder is None:
        raise FileNotFoundError(f"Could not find vrt folder for topic model in {folder}")

    person_codecs: md.PersonCodecs = md.PersonCodecs().load(source=metadata_filename)
    speech_repository: sr.SpeechTextRepository = sr.SpeechTextRepository(
        source=vrt_folder, person_codecs=person_codecs, document_index=document_index
    )

    return {
        'corpus_version': version,
        'data_folder': root_folder,
        'person_codecs': person_codecs,
        'speech_repository': speech_repository,
        'speech_index': document_index,
    }
