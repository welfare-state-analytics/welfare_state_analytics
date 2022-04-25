import __paths__  # pylint: disable=unused-import
import os

from penelope import topic_modelling as tm
from penelope.notebook import topic_modelling as ntm

from westac.riksprot.parlaclarin import codecs as md

jj = os.path.join

DATA_FOLDER: str = "/data/westac/riksdagen_corpus_data"
CODECS_FILENAME: str = jj(DATA_FOLDER, "metadata/riksprot_metadata.main.db")
MODEL_NAME: str = "tm_v041.1920-2020_500-TF5-MP0.02.500000.lemma.mallet"

# pylint: disable=protected-access


def xtest_focus_topics():
    model_folder: str = jj(DATA_FOLDER, MODEL_NAME)
    person_codecs: md.PersonCodecs = md.PersonCodecs().load(source=CODECS_FILENAME)
    trained_model: tm.InferredModel = tm.InferredModel.load(folder=model_folder, lazy=True)
    inferred_topics: tm.InferredTopicsData = tm.InferredTopicsData.load(
        folder=model_folder, filename_fields=r'year:prot\_(\d{4}).*', slim=True
    )
    state = {'inferred_model': trained_model, 'inferred_topics': inferred_topics}

    gui: ntm.FocusTopicDocumentNetworkGui = ntm.FocusTopicDocumentNetworkGui(
        pivot_key_specs=person_codecs.property_values_specs, state=state
    )
    gui._topic_ids.value = [10]
    gui._year_range.value = (gui._year_range.min, gui._year_range.max)
    gui.setup()
    gui.observe(False)

    _ = gui.layout()
    gui.update_handler()

    # gui._threshold.value = 0.20
    # gui._compute.click()


def xtest_sou():

    model_folder: str = jj("/data/westac/sou_kb_labb", "gensim_mallet-lda.topics.100.sou_kb-labb_1945-1989_nn")
    trained_model: tm.InferredModel = tm.InferredModel.load(folder=model_folder, lazy=True)
    inferred_topics: tm.InferredTopicsData = tm.InferredTopicsData.load(
        folder=model_folder, filename_fields=r'year:sou\_(\d{4}).*', slim=True
    )
    state = {'inferred_model': trained_model, 'inferred_topics': inferred_topics}

    assert inferred_topics

    # ntm.display_topic_topic_network_gui(state);
    gui: ntm.TopicTopicGUI = ntm.TopicTopicGUI(state=state).setup()
    _ = gui.layout()
    gui.update_handler()
    gui.display_handler()

    calculator = gui.inferred_topics.calculator.reset()
    calculator.filter_by_keys(**gui.filter_opts.opts)
    calculator.threshold(threshold=gui.threshold)
    calculator.filter_by_topics(topic_ids=gui.topic_ids, negate=gui.exclude_mode)
    calculator.to_topic_topic_network(gui.n_docs, topic_labels=gui.topic_labels)
    _ = calculator.value

    # data = (
    #     gui.inferred_topics.calculator.reset()
    #     .filter_by_keys(**gui.filter_opts.opts)
    #     .threshold(threshold=gui.threshold)
    #     .filter_by_topics(topic_ids=gui.topic_ids, negate=gui.exclude_mode)
    #     .to_topic_topic_network(gui.n_docs, topic_labels=gui.topic_labels)
    # ).value
