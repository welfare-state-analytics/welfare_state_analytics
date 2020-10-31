import logging
import os

import click
from penelope.vendor.textacy.mdw_modified import compute_most_discriminating_terms

import notebooks.political_in_newspapers.corpus_data as corpus_data
from notebooks.political_in_newspapers.notebook_gui import mdw_gui

# pylint: disable=too-many-locals, too-many-arguments

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("westac")


def get_ancestor_folder(ancestor):
    parts = os.getcwd().split(os.path.sep)
    parts = parts[: parts.index(ancestor) + 1]
    return os.path.join("/", *parts)


ROOT_FOLDER = get_ancestor_folder("welfare_state_analytics")
CORPUS_FOLDER = os.path.join(ROOT_FOLDER, "data/textblock_politisk")

PUB_IDS = ["AB", "DN"]  # list(corpus_data.ID2PUB.values()) + ['ALL']

pubs2ids = lambda pubs: [corpus_data.PUB2ID[x] for x in pubs]


@click.command()
@click.option("--corpus-folder", default=CORPUS_FOLDER, help="Corpus folder")
@click.option(
    "--max-n-terms",
    default=2000,
    help="Filter out token if DF is not within the top max_n_terms",
    type=int,
)
@click.option(
    "--min-df",
    default=None,
    help="Filter out tokens if DF fraction < ``min_df``.",
    type=click.FloatRange(0.0, 1.0),
)
@click.option(
    "--max-df",
    default=None,
    help="Filter out tokens if DF fraction > ``max_df``.",
    type=click.FloatRange(0.0, 1.0),
)
@click.option(
    "--min-abs-df",
    default=None,
    help="Filter out tokens if DF < ``min_abs_df``.",
    type=int,
)
@click.option(
    "--max-abs-df",
    default=None,
    help="Filter out tokens where DF > ``max_abs_df``.",
    type=int,
)
@click.option(
    "--top-n-terms",
    default=250,
    help="The total number of most discriminating terms to return for each group",
    type=int,
)
@click.option(
    "--group",
    "-g",
    multiple=True,
    metavar="PUBS YEAR-FROM YEAR-TO",
    help="A group e.g --group AB,DN 1945 1950.",
    type=click.Tuple([str, click.IntRange(1945, 1989), click.IntRange(1945, 1989)]),
)
def mdw_run(
    corpus_folder,
    max_n_terms,
    min_df,
    max_df,
    min_abs_df,
    max_abs_df,
    top_n_terms,
    group,
):

    if len(group) != 2:
        print("please specify two groups")
        return

    min_df = min_df or min_abs_df or 0.01
    max_df = max_df or max_abs_df or 0.95

    pubs1, pubs2 = (g[0].upper().split(",") for g in group)

    pubs_ids1, pubs_ids2 = pubs2ids(pubs1), pubs2ids(pubs2)

    if any((x not in PUB_IDS for x in pubs1 + pubs2)):
        print("Error: unknown publication(s):{}".format(",".join(list(set(pubs1 + pubs2) - set(PUB_IDS)))))
        return

    period1, period2 = ((g[1], g[2]) for g in group)

    logger.info("Using min DF %s and max DF %s", min_df, max_df)

    logger.info("Reading corpus...")

    # TODO: #92 Implement VectorizedCorpus.slice_by_df
    v_corpus = mdw_gui.load_vectorized_corpus(corpus_folder, pubs2ids(PUB_IDS)).slice_by_document_frequency(
        max_df=max_df, min_df=min_df, max_n_terms=max_n_terms
    )

    logger.info("Corpus size after DF trim %s x %s.", *v_corpus.data.shape)

    df = compute_most_discriminating_terms(
        v_corpus,
        top_n_terms=top_n_terms,
        max_n_terms=max_n_terms,
        group1_indices=mdw_gui.year_range_group_indicies(v_corpus.documents, period1, pubs_ids1),
        group2_indices=mdw_gui.year_range_group_indicies(v_corpus.documents, period2, pubs_ids2),
    )

    if df is not None:

        filename = "mdw_{}_{}-{}_vs_{}_{}-{}.xlsx".format(*group[0], *group[1])
        df.to_excel(filename)

        logger.info("Result saved to %s.", filename)


if __name__ == "__main__":
    mdw_run()  # pylint: disable=no-value-for-parameter
