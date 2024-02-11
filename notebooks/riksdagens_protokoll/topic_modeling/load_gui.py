from __future__ import annotations
from functools import cached_property

import re
from os.path import dirname, basename, isfile, exists
from os.path import join as jj
from typing import Any

import pandas as pd
from penelope.notebook import topic_modelling as ntm
from penelope.pipeline import CorpusConfig

import westac.riksprot.parlaclarin.speech_text as sr
from westac.riksprot.parlaclarin import codecs as md


def probe_corpus_version(folder: str) -> str:
    """Probe version of corpus from folder name or version in config file"""
    config_filename: str = jj(folder, 'config.yml')

    if isfile(jj(folder, 'version')):
        with open(jj(folder, 'version'), 'r') as fp:
            return fp.read().strip()

    config = CorpusConfig.load(config_filename) if isfile(config_filename) else None
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


def probe_paths(candidates: list[str]) -> str:
    path: str = next((x for x in candidates if exists(x)), None)
    if path is None:
        raise FileNotFoundError(f"Could not find any of {candidates}")
    return path


def probe_vrt_folder(folder: str, version: str) -> str:

    return probe_paths(
        [
            jj(folder, version, 'tagged_frames'),
            jj(folder, version, 'corpus/tagged_frames'),
            jj(folder, version, f'corpus/tagged_frames_{version}'),
            jj(folder, f'corpus/tagged_frames_{version}'),
        ]
    )


def probe_metadata_filename(folder: str, version: str) -> str:

    return probe_paths(
        [
            jj(folder, f"metadata/riksprot_metadata.{version}.db"),
            jj(folder, f"metadata/{version}/riksprot_metadata.{version}.db"),
            jj(folder, f"metadata/{version}/riksprot_metadata.db"),
        ]
    )


def probe_document_index(folder: str, version: str) -> pd.DataFrame:
    # FIXME: better probing
    filename: str = probe_paths(
        [
            jj(folder, version, 'tagged_frames.feather/document_index.feather'),
        ]
    )[0]
    di: pd.DataFrame = pd.read_feather(filename)
    return di


def load_metadata(root_folder: str, folder: str) -> dict[str, Any]:

    version: str = probe_corpus_version(folder)
    if version is None:
        raise FileNotFoundError(f"Could not find version for topic model in {folder}")

    metadata_filename: str = probe_metadata_filename(root_folder, version)
    document_index: pd.DataFrane = probe_document_index(root_folder, version)
    vrt_folder: str = probe_vrt_folder(root_folder, version)
    if vrt_folder is None:
        raise FileNotFoundError(f"Could not find vrt folder for topic model in {folder}")

    person_codecs: md.PersonCodecs = md.PersonCodecs().load(source=metadata_filename)
    speech_repository: sr.SpeechTextRepository = sr.SpeechTextRepository(
        source=vrt_folder, person_codecs=person_codecs, document_index=document_index
    )

    return {
        'version': version,
        'data_folder': root_folder,
        'person_codecs': person_codecs,
        'speech_repository': speech_repository,
        'speech_index': document_index,
    }


class RiksprotLoadGUI(ntm.LoadGUI):
    def __init__(self, data_folder: str, state: ntm.TopicModelContainer, slim: bool = False):
        super().__init__(data_folder, state, slim)

    def load(self) -> None:
        for k, v in load_metadata(self.data_folder, self.model_info.folder).items():
            self.state.set(k, v)

        return super().load()

    @cached_property
    def version(self) -> str:
        return self.state.get('version')
