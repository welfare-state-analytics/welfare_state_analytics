
import os
import penelope.notebook.dtm.to_DTM_gui as to_DTM_gui
import penelope.notebook.dtm.compute_DTM_corpus as compute_DTM_corpus

from ..utils import TEST_DATA_FOLDER, OUTPUT_FOLDER

 # pylint: disable=protected-access

def monkey_patch(*_, **__):
    ...

def test_compute_gui_compute_dry_run():

    compute_called = False

    def compute_patch(*_, **__):
        nonlocal compute_called
        compute_called = True

    gui: to_DTM_gui.ComputeGUI = to_DTM_gui.create_gui(
        corpus_config="riksdagens-protokoll",
        compute_document_term_matrix=compute_patch,
        corpus_folder=TEST_DATA_FOLDER,
        pipeline_factory=None,
        done_callback=monkey_patch,
    )

    gui._corpus_tag.value = 'CERES'
    gui._target_folder.reset(path=OUTPUT_FOLDER) #, filename='output.txt')
    gui._target_folder._apply_selection()
    gui._corpus_filename.reset(path=TEST_DATA_FOLDER, filename='proto.2files.2sentences.sparv4.csv.zip')
    gui._corpus_filename._apply_selection()
    gui._vectorize_button.click()

    assert compute_called
    #display(compute_gui.layout())

def test_compute_gui_compute_hot_run():

    corpus_tag = 'CERES'
    gui: to_DTM_gui.ComputeGUI = to_DTM_gui.create_gui(
        corpus_config="riksdagens-protokoll",
        compute_document_term_matrix=compute_DTM_corpus.compute_document_term_matrix,
        corpus_folder=TEST_DATA_FOLDER,
        pipeline_factory=None,
        done_callback=monkey_patch,
    )

    gui._corpus_tag.value = corpus_tag
    gui._target_folder.reset(path=OUTPUT_FOLDER) #, filename='output.txt')
    gui._target_folder._apply_selection()
    gui._corpus_filename.reset(path=TEST_DATA_FOLDER, filename='proto.2files.2sentences.sparv4.csv.zip')
    gui._corpus_filename._apply_selection()
    gui.compute_callback(gui)

    assert os.path.isfile(os.path.join(OUTPUT_FOLDER, corpus_tag, f'{corpus_tag}_vectorizer_data.pickle'))

    #display(compute_gui.layout())



# def done_callback(corpus: VectorizedCorpus, corpus_tag: str, corpus_folder: str):

#     trends_data: word_trends.TrendsData = word_trends.TrendsData(
#         corpus=corpus,
#         corpus_folder=corpus_folder,
#         corpus_tag=corpus_tag,
#         n_count=25000,
#     ).update()

#     gui = word_trends.GofTrendsGUI(
#         gofs_gui=word_trends.GoFsGUI().setup(),
#         trends_gui=word_trends.TrendsGUI().setup(),
#     )

#     display(gui.layout())
#     gui.display(trends_data=trends_data)
