from __future__ import annotations

import abc
import json
import os
import re
import sqlite3
import types
import zipfile
from collections import namedtuple
from functools import cached_property
from os.path import join as jj
from typing import Iterable, Literal

import numpy as np
import pandas as pd
from jinja2 import Template
from loguru import logger
from penelope import utility as pu

from . import codecs as md
from .utility import read_sql_table

try:
    import github as gh
except ImportError:
    Github = lambda t: types.SimpleNamespace()


default_template: Template = Template(
    """
<b>Protokoll:</b> {{protocol_name}} sidan {{ page_number }}, {{ chamber }}, {{ date }} <br/>
<b>Källor:</b> {{parlaclarin_links}} {{ wikidata_link }} {{ kb_labb_link }} <br/>
<b>Talare:</b> {{name}}, {{ party_abbrev }}, {{ office_type }}, {{ sub_office_type }}, {{ district }}, {{ gender}}<br/>
<b>Antal tokens:</b> {{ num_tokens }} ({{ num_words }}), uid: {{u_id}}, who: {{who}} <br/>
<h3> {{ speaker_note }} </h3>
<span style="color: blue;line-height:50%;">
{% for n in paragraphs %}
{{n}}
{% endfor %}
</span>
"""
)

GithubUrl = namedtuple('GithubUrl', 'name url')

# pylint: disable=unused-argument


class SpeechTextService:
    """Reconstitute text using information stored in the document (speech) index"""

    def __init__(self, document_index: pd.DataFrame):
        self.speech_index: pd.DataFrame = document_index
        self.speech_index['protocol_name'] = self.speech_index['document_name'].str.split('_').str[0]
        self.speech_index.rename(columns={'speach_index': 'speech_index'}, inplace=True, errors='ignore')

        """Name of speaker note reference was changed from v0.4.3 (speaker_hash => speaker_note_id)"""
        self.id_name = 'speaker_note_id' if 'speaker_note_id' in self.speech_index.columns else 'speaker_hash'

    @cached_property
    def name2info(self) -> dict[str, dict]:
        """Create a map protcol name to list of dict of relevant properties"""
        si: pd.DataFrame = self.speech_index.set_index('protocol_name', drop=True)[
            ['u_id', 'speech_index', self.id_name, 'n_utterances']
        ]
        return si.assign(data=si.to_dict('records')).groupby(si.index).agg(list)['data'].to_dict()

    def speeches(self, *, metadata: dict, utterances: list[dict]) -> list[dict]:
        """Create list of speeches for all speeches in protocol"""
        speech_infos: dict = self.name2info.get(metadata.get("name"))
        speech_lengths: np.ndarray = np.array([s.get("n_utterances", 0) for s in speech_infos])
        speech_starts: np.ndarray = np.append([0], np.cumsum(speech_lengths))
        speeches = [
            self._create_speech(metadata=metadata, utterances=utterances[speech_starts[i] : speech_starts[i + 1]])
            for i in range(0, len(speech_infos))
        ]
        return speeches

    def nth(self, *, metadata: dict, utterances: list[dict], n: int) -> dict:
        # speech_infos: dict = self.name2info.get(metadata.get("name"))
        # u_idx: int = [u['u_id'] for u in utterances].index(speech_infos['u_id'])
        # self._create_speech(metadata, utterances[u_idx:u_idx+speech_infos['n_utterances']])
        return self.speeches(metadata=metadata, utterances=utterances)[n]

    def _create_speech(self, *, metadata: dict, utterances: list[dict]) -> dict:
        return (
            {}
            if len(list(utterances or [])) == 0
            else dict(
                speaker_note_id=utterances[0][self.id_name],
                who=utterances[0]['who'],
                u_id=utterances[0]['u_id'],
                paragraphs=[p for u in utterances for p in u['paragraphs']],
                num_tokens=sum(x['num_tokens'] for x in utterances),
                num_words=sum(x['num_words'] for x in utterances),
                page_number=utterances[0]["page_number"] or "?",
                page_number2=utterances[-1]["page_number"] or "?",
                protocol_name=(metadata or {}).get("name", "?"),
                date=(metadata or {}).get("date", "?"),
            )
        )


class Loader(abc.ABC):
    @abc.abstractmethod
    def load(self, protocol_name: str) -> tuple[dict, list[dict]]:
        ...


class ZipLoader(Loader):
    def __init__(self, folder: str):
        self.folder: str = folder

    def load(self, protocol_name: str) -> tuple[dict, list[dict]]:
        """Loads tagged protocol data from archive"""
        sub_folder: str = protocol_name.split('-')[1]
        for filename in [
            jj(self.folder, sub_folder, f"{protocol_name}.zip"),
            jj(self.folder, f"{protocol_name}.zip"),
        ]:
            if not os.path.isfile(filename):
                continue
            with zipfile.ZipFile(filename, "r") as fp:
                json_str: str = fp.read(f"{protocol_name}.json")
                metadata_str: str = fp.read("metadata.json")
            metadata: dict = json.loads(metadata_str)
            utterances: list[dict] = json.loads(json_str)
            return metadata, utterances
        raise FileNotFoundError(protocol_name)


class SpeechTextRepository:

    GITHUB_REPOSITORY_URL: str = "https://github.com/welfare-state-analytics/riksdagen-corpus"
    GITHUB_REPOSITORY_RAW_URL = "https://raw.githubusercontent.com/welfare-state-analytics/riksdagen-corpus"

    def __init__(
        self,
        *,
        source: str | Loader,
        person_codecs: md.PersonCodecs,
        document_index: pd.DataFrame,
        template: Template = None,
        service: SpeechTextService = None,
    ):

        self.template: Template = template or default_template
        self.source: Loader = source if isinstance(source, Loader) else ZipLoader(source)
        self.person_codecs: md.PersonCodecs = person_codecs
        self.document_index: pd.DataFrame = document_index
        self.subst_puncts = re.compile(r'\s([,?.!"%\';:`](?:\s|$))')
        self.release_tags: list[str] = self.get_github_tags()
        self.service: SpeechTextService = service or SpeechTextService(self.document_index)
        self.document_name2id: dict[str, int] = (
            document_index.reset_index().set_index('document_name')['document_id'].to_dict()
        )

    def load_protocol(self, protocol_name: str) -> tuple[dict, list[dict]]:
        return self.source.load(protocol_name)

    def speeches(self, protocol_name: str) -> Iterable[dict]:
        metadata, utterances = self.source.load(protocol_name)
        return self.service.speeches(utterances=utterances, metadata=metadata)

    def _get_speech_info(self, speech_id: int | str) -> dict:
        """Get speaker-info from document index and person table"""

        speech_id: int = self.document_name2id.get(speech_id) if isinstance(speech_id, str) else speech_id

        try:
            speech_info: dict = self.document_index.loc[speech_id].to_dict()
        except KeyError as ex:
            raise KeyError(f"Speech {speech_id} not found in index") from ex

        try:
            speaker_name: str = self.person_codecs.person.loc[speech_info['who']]['name'] if speech_info else "unknown"
        except KeyError:
            speaker_name: str = speech_info['who']

        speech_info.update(name=speaker_name)

        speech_info["speaker_note"] = self.speaker_note_id2note.get(
            speech_info.get(self.service.id_name), "(introductory note not found)"
        )

        return speech_info

    @cached_property
    def speaker_note_id2note(self) -> dict:
        try:
            if not self.person_codecs.source_filename:
                return {}
            with sqlite3.connect(database=self.person_codecs.source_filename) as db:
                speaker_notes: pd.DataFrame = read_sql_table("speaker_notes", db)
                speaker_notes.set_index(self.service.id_name, inplace=True)
                return speaker_notes['speaker_note'].to_dict()
        except Exception as ex:
            logger.error(f"unable to read speaker_notes: {ex}")
            return {}

    def speech(self, speech_name: str, mode: Literal['dict', 'text', 'html']) -> dict | str:
        try:
            """Load speech data from speech corpus"""
            protocol_name: str = speech_name.split('_')[0]
            speech_nr: int = int(speech_name.split('_')[1])

            metadata, utterances = self.source.load(protocol_name)
            speech: dict = self.service.nth(metadata=metadata, utterances=utterances, n=speech_nr - 1)

            speech_info: dict = self._get_speech_info(speech_name)
            speech.update(**speech_info)
            speech.update(protocol_name=protocol_name)

            speech["office_type"] = self.person_codecs.office_type2name.get(speech["office_type_id"], "unknown")
            speech["sub_office_type"] = self.person_codecs.sub_office_type2name.get(
                speech["sub_office_type_id"], "unknown"
            )
            speech["gender"] = self.person_codecs.gender2name.get(speech["gender_id"], "unknown")
            speech["party_abbrev"] = self.person_codecs.party_abbrev2name.get(speech["party_id"], "unknown")

        except Exception as ex:  # pylint: disable=bare-except
            speech = {"name": "speech not found", "error": str(ex)}

        if mode == 'html':
            return self.to_html(speech)

        if mode == 'text':
            return self.to_text(speech)

        return speech

    def to_text(self, speech: dict) -> str:
        paragraphs: list[str] = speech.get('paragraphs', [])
        text: str = self.fix_whitespace('\n'.join(paragraphs))
        return text

    def fix_whitespace(self, text: str) -> str:
        return self.subst_puncts.sub(r'\1', text)

    def to_html(self, speech: dict) -> str:
        try:
            speech['parlaclarin_links'] = self.to_parla_clarin_urls(speech["protocol_name"])
            speech['wikidata_link'] = self.to_wikidata_link(speech["who"])
            speech['kb_labb_link'] = self.to_kb_labb_link(speech["protocol_name"], speech["page_number"])
            return self.template.render(speech)
        except Exception as ex:
            return f"render failed: {ex}"

    def to_parla_clarin_urls(self, protocol_name: str, ignores: str = 'alpha') -> str:
        return ' '.join(
            map(
                lambda x: f'<a href="{x.url}" target="_blank" style="font-weight: bold;color: blue;">{x.name}</a>&nbsp;',
                self.get_github_xml_urls(protocol_name, ignores=ignores),
            )
        )

    def to_wikidata_link(self, who: str) -> str:
        if not bool(who) or who == "unknown":
            return ""
        height, width = 20, int(20 * 1.41)
        img_src = f'<img width={width} heigh={height} src="https://upload.wikimedia.org/wikipedia/commons/f/ff/Wikidata-logo.svg"/>'
        return f'<a href="https://www.wikidata.org/wiki/{who}" target="_blank" style="font-weight: bold;color: blue;">{img_src}</a>&nbsp;'

    def to_kb_labb_link(self, protocol_name: str, page_number: str) -> str:

        if not bool(protocol_name):
            return ""

        page_url: str = (
            f"{protocol_name.replace('-', '_')}-{str(page_number).zfill(3)}.jp2/" if page_number.isnumeric() else ""
        )

        url: str = f"https://betalab.kb.se/{protocol_name}/{page_url}_view"

        return f'<a href="{url}" target="_blank" style="font-weight: bold;color: blue;">KB</a>&nbsp;'

    def get_github_tags(self, github_access_token: str = None) -> list[str]:
        release_tags: list[str] = ["main", "dev"]
        try:

            access_token: str = github_access_token or os.environ.get("GITHUB_ACCESS_TOKEN", None)

            # if access_token is None:
            #    logger.info("GITHUB_ACCESS_TOKEN not set")

            github: gh.Github = gh.Github(access_token)

            riksdagen_corpus = github.get_repo("welfare-state-analytics/riksdagen-corpus")
            release_tags = release_tags + [x.title for x in riksdagen_corpus.get_releases()]

        except:  # pylint: disable=bare-except
            ...
        return release_tags

    def get_github_xml_urls(self, protocol_name: str, ignores: str = None, n: int = 2) -> list[GithubUrl]:
        protocol_name: str = pu.strip_extensions(protocol_name)
        sub_folder: str = protocol_name.split('-')[1]
        xml_urls: list[GithubUrl] = []
        tags: list[str] = [t for t in self.release_tags if ignores not in t] if ignores else self.release_tags
        for tag in tags[:n]:
            url: str = f"{self.GITHUB_REPOSITORY_URL}/blob/{tag}/corpus/protocols/{sub_folder}/{protocol_name}.xml"
            raw_url: str = f"{self.GITHUB_REPOSITORY_RAW_URL}/{tag}/corpus/protocols/{sub_folder}/{protocol_name}.xml"
            xml_urls.append(GithubUrl(name=tag, url=url))
            xml_urls.append(GithubUrl(name=f"({tag})", url=raw_url))
        return xml_urls
