from __future__ import annotations

import os
from unittest import mock

import pandas as pd
from penelope import topic_modelling as tm
from penelope.plot import plot_multiple_value_series

import westac.riksprot.parlaclarin.speech_text as sr
from notebooks.riksdagens_protokoll.topic_modeling.multitrends_gui import RiksprotTopicMultiTrendsGUI
from westac.riksprot.parlaclarin import codecs as md

jj = os.path.join

# pylint: disable=redefined-outer-name,protected-access

DATA_FOLDER: str = "/data/riksdagen_corpus_data/"
MODEL_FOLDER: str = jj(DATA_FOLDER, "/data/riksdagen_corpus_data/tm_v041.1920-2020_100-TF5-MP0.02.500000.lemma.mallet")
DATABASE_FILENAME: str = jj(DATA_FOLDER, 'metadata/riksprot_metadata.main.db')
TAGGED_CORPUS_FOLDER: str = jj(DATA_FOLDER, "tagged_frames_v0.4.1")
SPEECH_INDEX_FILENAME: str = jj(DATA_FOLDER, 'tagged_frames_v0.4.1_speeches.feather/document_index.feather')

# DATA_FOLDER: str = "./tests/test_data/riksprot/main"
# MODEL_FOLDER: str = jj(DATA_FOLDER, "tm_test.5files.mallet")
# DATABASE_FILENAME: str = jj(DATA_FOLDER, 'riksprot_metadata.db')
# TAGGED_CORPUS_FOLDER: str = jj(DATA_FOLDER, "tagged_frames")
# SPEECH_INDEX_FILENAME: str = jj(DATA_FOLDER, "tagged_frames_speeches.feather/document_index.feather")



@mock.patch('bokeh.io.show', lambda *_, **__: None)
def test_topic_multitrends():
    speech_index: pd.DataFrame = pd.read_feather(SPEECH_INDEX_FILENAME)
    person_codecs: md.PersonCodecs = md.PersonCodecs().load(source=DATABASE_FILENAME)
    speech_repository: sr.SpeechTextRepository = sr.SpeechTextRepository(
        source=TAGGED_CORPUS_FOLDER,
        person_codecs=person_codecs,
        document_index=speech_index,
    )
    inferred_topics: tm.InferredTopicsData = tm.InferredTopicsData.load(folder=MODEL_FOLDER, slim=True)

    state = dict(inferred_topics=inferred_topics)

    ui: RiksprotTopicMultiTrendsGUI = RiksprotTopicMultiTrendsGUI(
        person_codecs=person_codecs, speech_repository=speech_repository, state=state
    )

    ui.setup()
    ui.add_line(name="man", values=["gender: man"])
    ui.add_line(name="kvinna", values=["gender: woman"])
    ui._year_range.value = (1920, 2020)
    ui._topic_id.value = 8

    # ytw: pd.DataFrame = ui.update()

    # startregion: ui.update()
    years_range: tuple[int, int] = (ui.years[0], ui.years[1] + 1)
    ytw: pd.DataFrame = pd.DataFrame(data={'year': range(*years_range)}).set_index('year')

    calculator: tm.DocumentTopicsCalculator = ui.inferred_topics.calculator.reset()
    calculator.filter_by_data_keys(topic_id=ui.topic_id, year=list(range(*years_range)))
    calculator.threshold(ui.threshold)

    # .overload(includes="gender_id,party_id,office_type_id,sub_office_type_id,person_id")

    topic_data: pd.DataFrame = calculator.value
    for name, opts in ui.lines_filter_opts.items():
        calculator.reset(topic_data)
        calculator.filter_by_keys(**opts.opts)
        calculator.yearly_topic_weights(ui.get_result_threshold(), n_top_relevance=None, topic_ids=ui.topic_id)
        ytw_line: pd.DataFrame = calculator.value[['year', ui.aggregate]].set_index('year').fillna(0)
        ytw_line.columns = [name]
        ytw = ytw.merge(ytw_line[name], how='left', left_index=True, right_index=True)
        ytw[name].fillna(0, inplace=True)
    # endregion

    plot_multiple_value_series(kind='multi_line', data=ytw.reset_index(), category_name='year', columns=None)
    # ui.update_handler()
    # ui.display_handler()

    assert ui._alert.value.startswith("✅")
