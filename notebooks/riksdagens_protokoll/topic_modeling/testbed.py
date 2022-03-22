# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +


from __future__ import annotations

import __paths__
from os.path import join as jj

import IPython
import pandas as pd
from bokeh.io import output_notebook
from IPython.display import display
from penelope import topic_modelling as tm
from penelope import utility as pu
from penelope.notebook import topic_modelling as ntm

import westac.riksprot.parlaclarin.codecs as md
import westac.riksprot.parlaclarin.speech_text as sr
from notebooks.riksdagens_protokoll import topic_modeling as wtm  # pylint: disable=unused-import

IPython.Application.instance().kernel.do_shutdown(True)


output_notebook(hide_banner=True)
pu.set_default_options()

data_folder: str = jj(__paths__.data_folder, "riksdagen_corpus_data")
model_folder: str = jj(data_folder, "tm_v041.1920-2020_500-TF5-MP0.02.500000.lemma.mallet")
person_codecs: md.PersonCodecs = md.PersonCodecs().load(source="./metadata/riksprot_metadata.main.db")
# FIXME #166 Load fails if topic models if topic model is slimmed-
inferred_topics: tm.InferredTopicsData = tm.InferredTopicsData.load(folder=model_folder, slim=True)
speech_index: pd.DataFrame = pd.read_feather(
    jj(data_folder, 'tagged_frames_v0.4.1_speeches.feather/document_index.feather')
)

speech_repository: sr.SpeechTextRepository = sr.SpeechTextRepository(
    source=jj(data_folder, "tagged_frames_v0.4.1"),
    person_codecs=person_codecs,
    document_index=inferred_topics.document_index,
)

state = dict(inferred_topics=inferred_topics)
# ui = wtm.RiksprotLoadGUI(person_codecs, corpus_folder="/data/westac/riksdagen_corpus_data/", corpus_config=None,state=state, slim=True).setup()
# ui = ntm.WordcloudGUI(state).setup()
# ui = wtm.RiksprotFindTopicDocumentsGUI(person_codecs, speech_repository=speech_repository, state=state).setup()
# ui = wtm.RiksprotBrowseTopicDocumentsGUI(person_codecs, speech_repository=speech_repository, state=state).setup()
ui = ntm.TopicWordDistributionGUI(state=state).setup()
# ui = wtm.RiksprotTopicTrendsGUI(person_codecs, speech_repository=speech_repository, state=state).setup()
# ui = ntm.TopicOverviewGUI(state=state).setup()
# ui = wtm.RiksprotTopicTrendsOverviewGUI(person_codecs, speech_repository=speech_repository, state=state).setup()
# ui = ntm.TopicTrendsGUI(state=state).setup()
# ui = wtm.RiksprotTopicTrendsGUI(person_codecs, speech_repository=speech_repository, state=state).setup()
# ui = ntm.TopicTopicGUI(state=state).setup()
# ui = wtm.RiksprotTopicTopicGUI(person_codecs, speech_repository=speech_repository, state=state).setup()
# ui = ntm.PivotTopicNetworkGUI(pivot_key_specs=person_codecs.property_values_specs, state=state).setup()
# ui = wtm.RiksprotTopicTopicGUI(person_codecs, speech_repository=speech_repository, state=state).setup()
# ui = ntm.EditTopicLabelsGUI(folder=model_folder, state=state).setup()
# ui: ntm.DefaultTopicDocumentNetworkGui = ntm.DefaultTopicDocumentNetworkGui(
#     pivot_key_specs=person_codecs.property_values_specs, state=state
# ).setup()
# ui: ntm.FocusTopicDocumentNetworkGui = ntm.FocusTopicDocumentNetworkGui(
#     pivot_key_specs=person_codecs.property_values_specs, state=state
# ).setup()
# ui._find_text.value = "byggnad"
display(ui.layout())
ui.update_handler()
# -
