from functools import wraps

import pandas as pd
import penelope.topic_modelling as tm


def memoize(f):
    memo: dict = {
        'filters': {},
        'threshold': {},
        'data': {},
    }

    @wraps(f)
    def wrapper(x):
        if x not in memo:
            memo[x] = f(x)
        return memo[x]

    return wrapper


class YearlyMeanTopicWeightsProxy:
    def __init__(self):

        self._current_data: pd.DataFrame = None
        self._current_filters: dict = {}
        self._current_threshold: float = 0.0

    def compute(self, inferred_topics: pd.DataFrame, filters: dict, threshold: float = 0.0) -> pd.DataFrame:

        if self._current_filters != filters or threshold != self._current_threshold:
            self._current_filters = filters
            self._current_threshold = threshold
            self._current_data = self._compute_data(inferred_topics, filters, threshold)

        return self._current_data

    def _compute_data(self, inferred_topics, filters, threshold):
        return tm.FilterDocumentTopicWeights(inferred_topics).threshold(threshold).filter_by_keys(filters).value
