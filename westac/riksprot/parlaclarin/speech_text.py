from __future__ import annotations
import abc
import os
import re
import types
from typing import Iterable, Literal
from collections import namedtuple
from itertools import groupby

import zipfile
import json
from os.path import join as jj
from loguru import logger

from jinja2 import Template
from . import metadata as md
from penelope import utility as pu

try:
    import github as gh
except ImportError:
    Github = lambda t: types.SimpleNamespace()

default_template: Template = Template(
    """
<b>Protokoll:</b> {{protocol_name}} sidan {{ page_number }}, {{ chamber }} <br/>
<b>Källa (XML):</b> {{parlaclarin_links}} <br/>
<b>Talare:</b> {{name}}, {{ party }}, {{ district }} ({{ gender}}) <br/>
<b>Antal tokens:</b> {{ num_tokens }} ({{ num_words }}),  uid: {{u_id}}, who: {{who}} ) <br/>
<h3> Anförande av {{ role_type }} {{ name }} ({{ party_abbrev }}) {{ date }}</h3>
<span style="color: blue;line-height:50%;">
{% for n in paragraphs %}
{{n}}
{% endfor %}
</span>
"""
)

GithubUrl = namedtuple('GithubUrl', 'name url')


class IMergeStrategy(abc.ABC):
    @abc.abstractmethod
    def merge(self, utterances: list[dict], metadata: dict) -> list[dict]:
        ...

    def nth(self, utterances: list[dict], n: int, metadata: dict) -> dict:
        ...

class DefaultMergeStrategy(IMergeStrategy):

    def merge(self, utterances: list[dict], metadata: dict) -> Iterable[dict]:
        return [self.to_speech(us, metadata=metadata) for _, us in self.groups(utterances)]

    def groups(self, utterances: list[dict]) -> list[tuple[str, list[dict]]]:
        return [(who, [u for u in us]) for who, us in groupby(utterances, key=lambda x: x['who'])]

    def nth(self, utterances: list[dict], n: int, metadata: dict) -> dict:
        return self.to_speech(self.groups(utterances=utterances)[n][1], metadata=metadata)

    def to_speech(self, utterances: list[dict], metadata: dict) -> dict:
        utterances = list(utterances or [])
        metadata = metadata or {}
        if len(utterances) == 0:
            return {}
        speech: dict = dict(
            who=utterances[0]['who'],
            u_id=utterances[0]['u_id'],
            paragraphs=[p for u in utterances for p in u['paragraphs']],
            num_tokens=sum(x['num_tokens'] for x in utterances),
            num_words=sum(x['num_words'] for x in utterances),
            page_number=utterances[0]["page_number"] or "?",
            page_number2=utterances[-1]["page_number"] or "?",
            protocol_name=metadata.get("name", "?"),
            date=metadata.get("date", "?"),
        )
        return speech


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
        filename: str = jj(self.folder, sub_folder, f"{protocol_name}.zip")
        with zipfile.ZipFile(filename, "r") as fp:
            json_str: str = fp.read(f"{protocol_name}.json")
            metadata_str: str = fp.read(f"metadata.json")
        metadata: dict = json.loads(metadata_str)
        utterances: list[dict] = json.loads(json_str)
        return metadata, utterances

class SpeechTextRepository:

    GITHUB_REPOSITORY_URL: str = "https://github.com/welfare-state-analytics/riksdagen-corpus"
    GITHUB_REPOSITORY_RAW_URL = "https://raw.githubusercontent.com/welfare-state-analytics/riksdagen-corpus"

    def __init__(
        self,
        source: str | Loader,
        riksprot_metadata: md.ProtoMetaData = None,
        template: Template = None,
        merger: IMergeStrategy = None,
    ):

        self.template: Template = template or default_template
        self.source: Loader = source if isinstance(source, Loader) else ZipLoader(source)
        self.riksprot_metadata: md.ProtoMetaData = riksprot_metadata
        self.role_type_translation: dict = {
            'member': 'riksdagsman',
            'speaker': 'talman',
            'minister': 'minister',
        }
        self.subst_puncts = re.compile(r'\s([,?.!"%\';:`](?:\s|$))')
        self.release_tags: list[str] = self.get_github_tags()
        self.merger: IMergeStrategy = merger or DefaultMergeStrategy()

    def load_protocol(self, protocol_name: str) -> tuple[dict, list[dict]]:
        return self.source.load(protocol_name)

    def speeches(self, protocol_name: str) -> Iterable[dict]:
        metadata, utterances = self.source.load(protocol_name)
        return self.merger.merge(utterances=utterances, metadata=metadata)

    def speech(self, speech_name: str, mode: Literal['dict', 'text', 'html']) -> dict | str:
        try:
            protocol_name: str = speech_name.split('_')[0]
            speech_index: int = int(speech_name.split('_')[1])
            metadata, utterances = self.source.load(protocol_name)
            speech: dict = self.merger.nth(utterances, n=speech_index-1, metadata=metadata)
            try:
                speaker: dict = self.riksprot_metadata.get_member(speech.get('who'))
                speech.update(**speaker)
            except md.MemberNotFoundError:
                speech.update(md.unknown_member())

            speech["role_type"] = self.role_type_translation.get(speech["role_type"], speech["role_type"])
        except:
            speech = {"name": "speech not found"}

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

    def get_github_tags(self, github_access_token: str = None) -> list[str]:
        release_tags: list[str] = ["main", "dev"]
        try:

            access_token: str = github_access_token or os.environ.get("GITHUB_ACCESS_TOKEN", None)

            if access_token is None:
                logger.info("GITHUB_ACCESS_TOKEN not set")

            github: gh.Github = gh.Github(access_token)

            riksdagen_corpus = github.get_repo("welfare-state-analytics/riksdagen-corpus")
            release_tags = release_tags + [x.title for x in riksdagen_corpus.get_releases()]

        except:  # pylint: disable=bare-except
            ...
        return release_tags

    def get_github_xml_urls(self, protocol_name: str, ignores: str = None) -> list[GithubUrl]:
        protocol_name: str = pu.strip_extensions(protocol_name)
        sub_folder: str = protocol_name.split('-')[1]
        xml_urls: list[GithubUrl] = []
        tags: list[str] = [t for t in self.release_tags if ignores not in t] if ignores else self.release_tags
        for tag in tags:
            url: str = f"{self.GITHUB_REPOSITORY_URL}/blob/{tag}/corpus/{sub_folder}/{protocol_name}.xml"
            raw_url: str = f"{self.GITHUB_REPOSITORY_RAW_URL}/{tag}/corpus/{sub_folder}/{protocol_name}.xml"
            xml_urls.append(GithubUrl(name=tag, url=url))
            xml_urls.append(GithubUrl(name=f"({tag})", url=raw_url))
        return xml_urls
