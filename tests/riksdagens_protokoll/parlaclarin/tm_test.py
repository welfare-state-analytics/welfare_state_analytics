import os
from typing import Callable

import penelope.notebook.topic_modelling as tm_ui
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
