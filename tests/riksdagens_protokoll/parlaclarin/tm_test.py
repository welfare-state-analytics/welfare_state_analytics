import os
from typing import Callable

import penelope.notebook.topic_modelling as tm_ui
import penelope.topic_modelling as tm
from penelope.pipeline.config import CorpusConfig


def test_bugcheck():
    current_state: Callable[[], tm_ui.TopicModelContainer] = tm_ui.TopicModelContainer.singleton
    corpus_folder: str = "/data/riksdagen_corpus_data/"
    corpus_config: CorpusConfig = CorpusConfig.load(
        os.path.join(corpus_folder, "dtm_1920-2020_v0.3.0.tf20", 'corpus.yml')
    )

    load_gui = tm_ui.create_load_topic_model_gui(corpus_config, corpus_folder, current_state())
    # display(load_gui.layout())
    load_gui.load.click()

    find_ui = tm_ui.find_topic_documents_gui(current_state())
    find_ui.find_text.value = "film"
    find_ui.update_handler()

    assert True


def test_tm_speech():

    folder: str = "/data/riksdagen_corpus_data/tm_1920-2020_500-topics"
    folder = "/data/riksdagen_corpus_data/tm_1920-2020_500-topics.id.year.who/"
    inferred_data: tm.InferredTopicsData = tm.InferredTopicsData.load(folder=folder, pickled=False)

    assert inferred_data is not None
