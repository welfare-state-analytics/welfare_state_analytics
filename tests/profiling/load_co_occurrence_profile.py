import penelope.co_occurrence as co_occurrence
import penelope.notebook.co_occurrence as co_occurrence_gui
from penelope.notebook.word_trends import BundleTrendsService

filename: str = (
    '/data/westac/shared/information_w3_NNPM_lemma_no_stops_NEW/information_w3_NNPM_lemma_no_stops_NEW_co-occurrence.csv.zip'
)

bundle: co_occurrence.Bundle = co_occurrence.Bundle.load(filename)

trends_service: BundleTrendsService = BundleTrendsService(bundle=bundle)

co_occurrence_gui.ExploreGUI(bundle).setup().display(trends_service=trends_service)
