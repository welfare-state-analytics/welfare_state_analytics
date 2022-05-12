# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

from __future__ import annotations

import __paths__  # pylint: disable=unused-import
from os.path import join as jj

import pandas as pd  # pylint: disable=unused-import
from bokeh.io import output_notebook
from bokeh.models import FuncTickFormatter

# %%
from bokeh.plotting import figure, show
from IPython.display import display
from penelope import topic_modelling as tm
from penelope import utility as pu

import westac.riksprot.parlaclarin.codecs as md
import westac.riksprot.parlaclarin.speech_text as sr
from notebooks.riksdagens_protokoll import topic_modeling as wtm  # pylint: disable=unused-import
from notebooks.riksdagens_protokoll.topic_modeling.multitrends_gui import RiksprotTopicMultiTrendsGUI

# pylint: disable=protected-access
output_notebook(hide_banner=True)
pu.set_default_options()

corpus_version: str = "v0.4.1"
data_folder: str = "/data/westac/riksdagen_corpus_data"
codecs_filename: str = jj(data_folder, f"metadata/riksprot_metadata.{corpus_version}.db")
model_folder: str = jj(data_folder, "tm_v041.1920-2020_100-TF5-MP0.02.500000.lemma.mallet")
tagged_frames_folder: str = jj(data_folder, f"tagged_frames_{corpus_version}")

# data_folder: str = jj(__paths__.root_folder, "tests/test_data/riksprot/main")
# codecs_filename: str = jj(data_folder, "riksprot_metadata.db")
# model_folder: str = jj(data_folder, "tm_test.5files.mallet")
# tagged_frames_folder: str = jj(data_folder, "tagged_frames")

person_codecs: md.PersonCodecs = md.PersonCodecs().load(source=codecs_filename)

state = {
    'inferred_model': tm.InferredModel.load(folder=model_folder, lazy=True),
    'inferred_topics': tm.InferredTopicsData.load(
        folder=model_folder, filename_fields=r'year:prot\_(\d{4}).*', slim=True
    ),
}
speech_repository: sr.SpeechTextRepository = sr.SpeechTextRepository(
    source=tagged_frames_folder,
    person_codecs=person_codecs,
    document_index=state.get('inferred_topics').document_index,
)

ui: RiksprotTopicMultiTrendsGUI = RiksprotTopicMultiTrendsGUI(
    person_codecs, speech_repository=speech_repository, state=state
).setup()

display(ui.layout())

ui._topic_id.value = 1
ui._year_range.value = (ui._year_range.min, ui._year_range.max + 1)
ui.add_line(name="(S)", color='red', values=["party_abbrev: S"])
ui.add_line(name="(M)", color='blue', values=["party_abbrev: M"])
ui.add_line(name="(C)", color='green', values=["party_abbrev: C"])

# ui.add_line(name="(L)", values=["party_abbrev: L"])
# ui.add_line(name="(SD)", values=["party_abbrev: SD"])


# %%
fruits = ['Apples', 'Pears', 'Nectarines', 'Plums', 'Grapes', 'Strawberries']
years = ["2015", "2016", "2017"]
colors = ["#c9d9d3", "#718dbf", "#e84d60"]

data = {'fruits': fruits, '2015': [2, 1, 4, 3, 2, 4], '2016': [5, 3, 4, 2, 4, 6], '2017': [3, 2, 4, 4, 5, 3]}

p = figure(
    x_range=fruits,
    height=250,
    title="Fruit Counts by Year",
    toolbar_location=None,
    tools="hover",
    tooltips="$name @fruits: @$name",
)

p.vbar_stack(years, x='fruits', width=0.9, color=colors, source=data, legend_label=years)

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.outline_line_color = None
p.legend.location = "top_left"
p.legend.orientation = "horizontal"
p.axis.formatter = FuncTickFormatter(
    code="""
    return (index % 2 == 0) ? tick : "";
"""
)
show(p)
