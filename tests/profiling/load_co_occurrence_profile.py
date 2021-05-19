import penelope.co_occurrence as co_occurrence
import penelope.notebook.co_occurrence as co_occurrence_gui

filename: str = '/data/westac/shared/information_w3_NNPM_lemma_no_stops_NEW/information_w3_NNPM_lemma_no_stops_NEW_co-occurrence.csv.zip'

bundle: co_occurrence.Bundle = co_occurrence.Bundle.load(filename)

trends_data = co_occurrence.to_trends_data(bundle).update()

co_occurrence_gui.ExploreGUI().setup().display(trends_data=trends_data)
