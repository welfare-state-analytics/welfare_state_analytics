
import numpy as np
import matplotlib.pyplot as plt

def mpl_plot_distributions(x_corpus, df_most_deviating, metric, columns=2, rows=2):

    min_year = x_corpus.document_index.year.min()
    max_year = x_corpus.document_index.year.max()

    token_column = metric + '_token'
    indices = [  x_corpus.token2id[token] for token in df_most_deviating[token_column] ]

    fig = plt.figure(figsize=(20,8))
    xs = np.arange(min_year, max_year + 1, 1)

    for i in range(0, columns*rows):

        ys = x_corpus.data[:,indices[i]]

        ax_left = fig.add_subplot(rows, columns, i+1)
        ax_left.set_xticks(xs)
        ax_left.set_xticklabels([ str(x) for x in xs])
        ax_left.scatter(xs, ys)
        ax_left.set_ylim(0.0, None)
        ax_left.set_xlim(xs[0], xs[-1])

        #box = ax_left.get_position()
        #ax_left.set_position([box.x0, box.y0, box.width * 0.8, box.height])

        if ax_left.get_xticklabels() is not None:
            for tick in ax_left.get_xticklabels():
                tick.set_rotation(45)

    plt.tight_layout()
    plt.show()
