import itertools

import bokeh


def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(itertools.islice(iterable, n))


def compile_multiline_data(x_corpus, indices, smoothers=None):

    xs = x_corpus.xs_years()

    if len(smoothers or []) > 0:
        xs_data = []
        ys_data = []
        for j in indices:
            xs_j = xs
            ys_j = x_corpus.bag_term_matrix[:, j]
            for smoother in smoothers:
                xs_j, ys_j = smoother(xs_j, ys_j)
            xs_data.append(xs_j)
            ys_data.append(ys_j)
    else:
        xs_data = [xs.tolist()] * len(indices)
        ys_data = [x_corpus.bag_term_matrix[:, token_id].tolist() for token_id in indices]

    data = {
        'xs': xs_data,
        'ys': ys_data,
        'label': [x_corpus.id2token[token_id].upper() for token_id in indices],
        'color': take(len(indices), itertools.cycle(bokeh.palettes.Category10[10])),
    }
    return data


def compile_year_token_vector_data(x_corpus, indices, *args):  # pylint: disable=unused-argument

    xs = x_corpus.xs_years()
    data = {x_corpus.id2token[token_id]: x_corpus.bag_term_matrix[:, token_id] for token_id in indices}
    data['year'] = xs

    return data
