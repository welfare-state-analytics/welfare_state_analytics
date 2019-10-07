import nltk

from westac.case_study_political_in_newspapers import window_coocurrence as political

stopwords = set(nltk.corpus.stopwords.words('swedish')).union({ "politisk", "politiska", "politiskt" })
options   = dict(to_lower=True, deacc=False, min_len=2, max_len=None, numerals=False, filter_stopwords=False, stopwords=stopwords)
years = list(range(1945, 1990))
political.compute_co_ocurrence_for_year('./westac/case_study_political_in_newspapers/data/year+text_window.txt', years, 'co_occurrence_1945-1989.xlsx', options)
