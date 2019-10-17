
from westac.case_study_political_in_newspapers import window_coocurrence as political
from westac.case_study_political_in_newspapers.stopwords import stopwords

years      = list(range(1945, 1990))
newspapers = [ 'AFTONBLADET', 'EXPRESSEN', 'DAGENS NYHETER', 'SVENSKA DAGBLADET' ]
min_count  = 3

options   = dict(to_lower=True, deacc=False, min_len=2, max_len=None, numerals=False, filter_stopwords=False, stopwords=stopwords)
source_filename = './westac/case_study_political_in_newspapers/data/year+newspaper+text_2019-10-16.txt'
target_filename = 'co_occurrence_{}-{}.csv'.format(min(years), max(years))

political.compute_co_ocurrence_for_year(source_filename, newspapers, years, target_filename, min_count=min_count, **options)
