import itertools
from typing import Callable, Dict, List, Sequence

import bokeh
from penelope.corpus.vectorized_corpus import VectorizedCorpus
from penelope.utility import take


class PenelopeBugCheck(Exception):
    pass


def compile_multiline_data(corpus: VectorizedCorpus, indices: List[int], smoothers: List[Callable] = None) -> Dict:
    """Compile multiline plot data for token ids `indicies`, optionally applying `smoothers` functions"""
    xs = corpus.xs_years()
    # FIXME #107 Error when occurs when compiling multiline data
    if len(smoothers or []) > 0:
        xs_data = []
        ys_data = []
        for j in indices:
            xs_j = xs
            ys_j = corpus.bag_term_matrix[:, j]
            for smoother in smoothers:
                xs_j, ys_j = smoother(xs_j, ys_j)
            xs_data.append(xs_j)
            ys_data.append(ys_j)
    else:
        xs_data = [[x for x in xs]] * len(indices)
        ys_data = corpus.bag_term_matrix[:, indices].T.toarray().tolist()

    data = {
        'xs': xs_data,
        'ys': ys_data,
        'label': [corpus.id2token[token_id].upper() for token_id in indices],
        'color': take(len(indices), itertools.cycle(bokeh.palettes.Category10[10])),
    }
    return data


def compile_year_token_vector_data(corpus: VectorizedCorpus, indices: Sequence[int], *_) -> Dict:
    """Extracts token's vectors for tokens ´indices` and returns a dict keyed by token"""
    # FIXME: Logic assumes that vector is grouped_by_year
    xs = corpus.xs_years()
    if len(xs) != corpus.data.shape[0]:
        raise PenelopeBugCheck(f"DTM shape{corpus.data.shape} is not compatible with year range {corpus.year_range()}")
    data = {corpus.id2token[token_id]: corpus.bag_term_matrix[:, token_id] for token_id in indices}
    data['year'] = xs
    return data
