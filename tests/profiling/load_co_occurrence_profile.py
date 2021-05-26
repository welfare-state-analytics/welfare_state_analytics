import penelope.co_occurrence as co_occurrence
import penelope.notebook.co_occurrence as co_occurrence_gui
from penelope.notebook.word_trends import TrendsData

filename: str = '/data/westac/shared/information_w3_NNPM_lemma_no_stops_NEW/information_w3_NNPM_lemma_no_stops_NEW_co-occurrence.csv.zip'

bundle: co_occurrence.Bundle = co_occurrence.Bundle.load(filename)

trends_data: TrendsData = TrendsData(
    compute_options=bundle.compute_options,
    corpus=bundle.corpus,
    corpus_folder=bundle.folder,
    corpus_tag=bundle.tag,
    n_count=25000,
).update()

co_occurrence_gui.ExploreGUI(bundle).setup().display(trends_data=trends_data)
