import os
import sys
from os.path import join as jj

import click
import penelope.topic_modelling as topic_modelling
from penelope.corpus import VectorizedCorpus

import notebooks.political_in_newspapers.corpus_data as corpus_data

# pylint: disable=unused-argument, too-many-locals, too-many-arguments

ENGINE_OPTIONS = [
    ('MALLET LDA', 'gensim_mallet-lda'),
    ('gensim LDA', 'gensim_lda'),
    ('gensim LSI', 'gensim_lsi'),
    ('gensim HDP', 'gensim_hdp'),
    ('gensim DTM', 'gensim_dtm'),
    ('scikit LDA', 'sklearn_lda'),
    ('scikit NMF', 'sklearn_nmf'),
    ('scikit LSA', 'sklearn_lsa'),
    ('STTM   LDA', 'gensim_sttm-lda'),
    ('STTM   BTM', 'gensim_sttm-btm'),
    ('STTM   PTM', 'gensim_sttm-ptm'),
    ('STTM  SATM', 'gensim_sttm-satm'),
    ('STTM   DMM', 'gensim_sttm-dmm'),
    ('STTM  WATM', 'gensim_sttm-watm'),
]


@click.command()
@click.argument('name')  # , help='Model name.')
@click.option('--n-topics', default=50, help='Number of topics.', type=click.INT)
@click.option('--corpus-folder', default='.', help='Corpus folder.')
@click.option(
    '--corpus-type',
    default='R',
    type=click.Choice(['R', 'vectorized', 'sparv-xml'], case_sensitive=False),
)
@click.option(
    '--corpus-name',
    default='.',
    help='Corpus filename (if text corpus or Sparv XML). Corpus tag if vectorized corpus.',
)
@click.option('--engine', default="gensim_lda-multicore", help='LDA implementation')
@click.option('--passes', default=None, help='Number of passes.', type=click.INT)
@click.option('--alpha', default='asymmetric', help='Prior belief of topic probability. symmetric/asymmertic/auto')
@click.option('--random-seed', default=None, help="Random seed value", type=click.INT)
@click.option('--workers', default=None, help='Number of workers (if applicable).', type=click.INT)
@click.option('--max-iter', default=None, help='Max number of iterations.', type=click.INT)
@click.option('--prefix', default=None, help='Prefix.')
def compute_topic_model(
    name,
    n_topics,
    corpus_folder,
    corpus_type,
    corpus_name,
    engine,
    passes,
    random_seed,
    alpha,
    workers,
    max_iter,
    prefix,
):
    run_model(
        name=name,
        n_topics=n_topics,
        corpus_folder=corpus_folder,
        corpus_type=corpus_type,
        corpus_tag=corpus_name,
        engine=engine,
        passes=passes,
        random_seed=random_seed,
        alpha=alpha,
        workers=workers,
        max_iter=max_iter,
        prefix=prefix,
    )


def run_model(
    name=None,
    n_topics=50,
    corpus_folder=None,
    corpus_type=None,
    corpus_tag=None,
    engine="gensim_lda-multicore",
    passes=None,
    random_seed=None,
    alpha='asymmetric',  # pylint: disable=unused-argument
    workers=None,
    max_iter=None,
    prefix=None,
):
    """ runner """

    if corpus_tag is None and corpus_folder is None:
        click.echo("usage: either corpus-folder or corpus filename must be specified")
        sys.exit(1)
    call_arguments = dict(locals())
    topic_modeling_opts = {
        k: v
        for k, v in call_arguments.items()
        if k in ['n_topics', 'passes', 'random_seed', 'alpha', 'workers', 'max_iter', 'prefix'] and v is not None
    }

    if engine not in [y for _, y in ENGINE_OPTIONS]:
        click.echo('Unknown method {}'.format(engine))
        sys.exit(1)

    if corpus_type == 'vectorized':

        assert corpus_tag is not None, 'error: Corpus dump name-tag not specified for vectorized corpus'
        assert VectorizedCorpus.dump_exists(tag=corpus_tag, folder=corpus_folder), 'error: no dump for given tag exists'

        v_corpus = VectorizedCorpus.load(tag=corpus_tag, folder=corpus_folder)

        dtm = v_corpus.data
        id2token = v_corpus.id2token
        document_index = v_corpus.document_index
        # documents['publication_id'] = 1

    # elif corpus_type == 'text':
    #     opts = dict(pos_includes='|NN|', lemmatize=True, chunk_size=None)
    # elif corpus_type == 'sparv-xml':
    #     reader = sparv_reader.SparvXmlCorpusSourceReader(corpus_tag, **opts)

    else:

        dtm, document_index, id2token = corpus_data.load_as_dtm2(corpus_folder, [1, 3])

    target_folder = os.path.join(corpus_folder, name)
    if not os.path.isdir(target_folder):
        os.mkdir(target_folder)

    train_corpus = topic_modelling.TrainingCorpus(
        terms=None,
        doc_term_matrix=dtm,
        id2word=id2token,  # type: ignore
        document_index=document_index,
    )

    inferred_model = topic_modelling.infer_model(
        train_corpus=train_corpus,
        method=engine,
        engine_args=topic_modeling_opts,
    )

    inferred_model.topic_model.save(jj(target_folder, 'gensim.model'))

    topic_modelling.store_model(inferred_model, target_folder)

    inferred_topics = topic_modelling.compile_inferred_topics_data(
        inferred_model.topic_model, train_corpus.corpus, train_corpus.id2word, train_corpus.document_index
    )
    inferred_topics.store(target_folder)


if __name__ == '__main__':
    compute_topic_model()  # pylint: disable=no-value-for-parameter
