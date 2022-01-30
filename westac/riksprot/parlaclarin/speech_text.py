from __future__ import annotations
import re
from typing import Literal

import zipfile
import json
from os.path import join as jj

from jinja2 import Template
from . import metadata as md

default_template: Template = Template("""
<b>Protokoll:</b> {{protocol_name}} sidan {{ page_number }}, {{ chamber }}<br/>
<b>Talare:</b> {{name}}, {{ party }}, {{ district }} ({{ gender}}) <br/>
<b>Antal tokens:</b> {{ num_tokens }} ({{ num_words }}) ({{u_id}}) <br/>
<h3> Anförande av {{ role_type }} {{ name }} ({{ party_abbrev }}) {{ date }}</h3>
<color='blue'>
{% for n in paragraphs %}
{{n}}
{% endfor %}
</color>
"""
)

class SpeechTextRepository:
    def __init__(self, folder: str, riksprot_metadata: md.ProtoMetaData=None, template: Template = None):

        self.template: Template = template or default_template
        self.folder: str = folder
        self.riksprot_metadata: md.ProtoMetaData = riksprot_metadata
        self.role_type_translation: dict = {
            'member': 'riksdagsman',
            'speaker': 'talman',
            'minister': 'minister',
        }
        self.subst_puncts = re.compile(r'\s([,?.!"%\';:`](?:\s|$))')

    def speech(self, speech_name: str, mode: Literal['dict', 'text', 'html']) -> dict | str:

        try:
            sub_folder: str = speech_name.split('-')[1]
            protocol_name: str = speech_name.split('_')[0]
            speech_index: int = int(speech_name.split('_')[1])

            filename: str = jj(self.folder, sub_folder, f"{protocol_name}.zip")

            with zipfile.ZipFile(filename, "r") as fp:
                json_str: str = fp.read(f"{protocol_name}.json")
                metadata_str: str = fp.read(f"metadata.json")

            metadata: dict = json.loads(metadata_str)
            protocol_data: dict = json.loads(json_str)

            try:
                speech: dict = protocol_data[speech_index]
            except IndexError:
                return {"name": "speech not found (index-out-of-bound!"}

            if metadata:
                speech.update(protocol_name=metadata.get("name", "?"), date=metadata.get("date"))

            try:
                speaker: dict = self.riksprot_metadata.get_member(speech.get('who'))
                speech.update(**speaker)
            except md.MemberNotFoundError:
                speech.update(md.unknown_member())

            speech["page_number"] = speech["page_number"] or "?"
            speech["role_type"] = self.role_type_translation.get(speech["role_type"], speech["role_type"] )

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
        return self.subst_puncts.sub(r'\s([,?.!"%\';:`](?:\s|$))', r'\1', text)

    def to_html(self, speech: dict) -> str:
        try:
            return self.template.render(speech)
        except Exception as ex:
            return f"render failed: {ex}"
