import time

import penelope.vendor.nltk as nltk_utility

from notebooks.textblock_politiskt.pandas_co_occurrence import compute_co_occurrence_for_periods

stopwords = nltk_utility.extended_stopwords()

# pylint: disable=too-many-function-args


def run():
    # periods    = list(range(1945, 1990))
    # periods = [(1945, 1954), (1955, 1964), (1965, 1974), (1975, 1984)]
    # newspapers = ['AFTONBLADET', 'EXPRESSEN', 'DAGENS NYHETER', 'SVENSKA DAGBLADET']
    min_count = 0

    options = dict(
        to_lower=True,
        remove_accents=False,
        min_len=2,
        max_len=None,
        keep_numerals=False,
        keep_symbols=False,
        filter_stopwords=False,
        stopwords=stopwords,
    )
    source_filename = './data/year+newspaper+text_2019-10-16.txt'
    target_filename = './output/political_co_occurrence_{}.csv'.format(time.strftime("%Y%m%d_%H%M%S"))

    # political.compute_co_ocurrence_for_periods(source_filename, newspapers, periods, target_filename, min_count=min_count, **options)

    for year in [1948, 1958, 1968, 1978, 1988]:
        target_filename = './output/political_co_occurrence_{}.csv.zip'.format(year)
        compute_co_occurrence_for_periods(
            source_filename,
            # newspapers,
            [year],
            target_filename,
            min_count=min_count,
            **options,
        )


if __name__ == "__main__":
    run()
