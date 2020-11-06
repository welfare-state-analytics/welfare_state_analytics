from typing import Set

from penelope.co_occurrence import partitioned_corpus_concept_co_occurrence, to_vectorized_corpus
from penelope.co_occurrence.concept_co_occurrence import ConceptContextOpts
from penelope.corpus import AnnotationOpts, SparvTokenizedCsvCorpus


def generate_test_vectorized_corpus(
    filename: str,
    concepts: Set[str],
    *,
    context_width: int = 2,
    annotation_opts: AnnotationOpts = None,
    n_count_threshold: int = None,
    partition_keys: str = "year",
    no_concept: bool = False,
):
    annotation_opts = annotation_opts or AnnotationOpts()
    corpus = SparvTokenizedCsvCorpus(
        filename,
        tokenizer_opts=dict(
            filename_fields={"year": r"prot\_(\d{4}).*"},
        ),
        annotation_opts=annotation_opts,
    )

    if not any([concept in corpus.token2id for concept in concepts]):
        print("concept not found in corpus")
        return None

    coo_df = partitioned_corpus_concept_co_occurrence(
        corpus,
        concept_opts=ConceptContextOpts(concepts=concepts, no_concept=no_concept, n_context_width=context_width),
        n_count_threshold=n_count_threshold,
        partition_keys=partition_keys,
    )

    # store_co_occurrences('/data/riksdagens-protokoll.1920-2019.test', coo_df)
    _v_corpus = to_vectorized_corpus(coo_df, value_column="value_n_t")

    _v_corpus.dump(
        tag=f"coo_vectors_{'_'.join(concepts)}_{''.join(annotation_opts.get_pos_includes())}_{context_width}",
        folder="/data/westac/",
    )

    return _v_corpus


v_corpus_riksdagens_protokoll = generate_test_vectorized_corpus(
    "/data/westac/riksdagens-protokoll.1920-2019.test.zip",
    concepts={"arbetare"},
    context_width=2,
    annotation_opts=AnnotationOpts(lemmatize=True),
)

# %%
