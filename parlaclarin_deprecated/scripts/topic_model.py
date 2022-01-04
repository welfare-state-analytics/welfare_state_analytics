import os
import sys

import click
import penelope.corpus as penelope
import yaml
from penelope import pipeline
from penelope.corpus import TextTransformOpts, remove_hyphens

# pylint: disable=unused-argument, too-many-arguments,


@click.command()
@click.argument('config-filename', required=True)
@click.argument('target-name', required=False)
@click.option('--options-filename', default=None, help='Use values in YAML file as command line options.')
@click.option('--target-folder', default=None, help='Target folder, if none then corpus-folder/target-name.')
@click.option('--corpus-source', default=None, help='Corpus source (overrides config)')
@click.option('--fix-hyphenation/--no-fix-hyphenation', default=True, is_flag=True, help='Fix hyphens')
@click.option('--fix-accents/--no-fix-accents', default=True, is_flag=True, help='Fix accents')
@click.option('-b', '--lemmatize/--no-lemmatize', default=True, is_flag=True, help='Use word baseforms')
@click.option('-i', '--pos-includes', default='', help='POS tags to include e.g. "|NN|JJ|".', type=click.STRING)
@click.option('-x', '--pos-excludes', default='', help='POS tags to exclude e.g. "|MAD|MID|PAD|".', type=click.STRING)
@click.option('-l', '--to-lower/--no-to-lower', default=True, is_flag=True, help='Lowercase words')
@click.option('--min-word-length', default=1, type=click.IntRange(1, 99), help='Min length of words to keep')
@click.option('--max-word-length', default=None, type=click.IntRange(10, 99), help='Max length of words to keep')
@click.option('--keep-symbols/--no-keep-symbols', default=True, is_flag=True, help='Keep symbols')
@click.option('--keep-numerals/--no-keep-numerals', default=True, is_flag=True, help='Keep numerals')
@click.option('--remove-stopwords', default=None, type=click.Choice(['swedish', 'english']), help='Remove stopwords')
@click.option('--only-alphabetic', default=False, is_flag=False, help='Remove tokens with non-alphabetic character(s)')
@click.option('--only-any-alphanumeric', default=False, is_flag=True, help='Remove tokes with no alphanumeric char')
@click.option('--n-topics', default=50, help='Number of topics.', type=click.INT)
@click.option('--engine', default="gensim_lda-multicore", help='LDA implementation')
@click.option('--passes', default=None, help='Number of passes.', type=click.INT)
@click.option('--alpha', default='asymmetric', help='Prior belief of topic probability. symmetric/asymmertic/auto')
@click.option('--random-seed', default=None, help="Random seed value", type=click.INT)
@click.option('--workers', default=None, help='Number of workers (if applicable).', type=click.INT)
@click.option('--max-iter', default=None, help='Max number of iterations.', type=click.INT)
@click.option('--store-corpus/--no-store-corpus', default=True, is_flag=True, help='')
@click.option('--store-compressed/--no-store-compressed', default=True, is_flag=True, help='')
@click.option('--force-checkpoint/--no-force-checkpoint', default=False, is_flag=True, help='')
@click.option('--enable-checkpoint/--no-enable-checkpoint', default=True, is_flag=True, help='')
def click_main(
    config_filename: str = None,
    target_name: str = None,
    options_filename: str = None,
    corpus_source: str = None,
    target_folder: str = None,
    fix_hyphenation: bool = True,
    fix_accents: bool = True,
    lemmatize: bool = True,
    pos_includes: str = '',
    pos_excludes: str = '',
    to_lower: bool = True,
    remove_stopwords: str = None,
    min_word_length: int = 2,
    max_word_length: int = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    only_any_alphanumeric: bool = False,
    only_alphabetic: bool = False,
    n_topics: int = 50,
    engine: str = "gensim_lda-multicore",
    passes: int = None,
    random_seed: int = None,
    alpha: str = 'asymmetric',
    workers: int = None,
    max_iter: int = None,
    store_corpus: bool = True,
    store_compressed: bool = True,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
):
    arguments: dict = locals()
    del arguments['options_filename']

    if options_filename is not None:
        with open(options_filename, "r") as fp:
            options: dict = yaml.load(fp, Loader=yaml.FullLoader)
        arguments.update(options)

    if not os.path.isfile(config_filename):
        click.echo(f"error: file {config_filename} not found")
        sys.exit(1)

    if arguments.get('target_name') is None:
        click.echo("error: target_name not specified")
        sys.exit(1)

    _main(**arguments)


def _main(
    config_filename: str = None,
    target_name: str = None,
    corpus_source: str = None,
    target_folder: str = None,
    fix_hyphenation: bool = True,
    fix_accents: bool = True,
    lemmatize: bool = True,
    pos_includes: str = '',
    pos_excludes: str = '',
    to_lower: bool = True,
    remove_stopwords: str = None,
    min_word_length: int = 2,
    max_word_length: int = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    only_any_alphanumeric: bool = False,
    only_alphabetic: bool = False,
    n_topics: int = 50,
    engine: str = "gensim_lda-multicore",
    passes: int = None,
    random_seed: int = None,
    alpha: str = 'asymmetric',
    workers: int = None,
    max_iter: int = None,
    store_corpus: bool = True,
    store_compressed: bool = True,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
):
    config: pipeline.CorpusConfig = pipeline.CorpusConfig.load(path=config_filename)

    text_transform_opts: TextTransformOpts = TextTransformOpts()

    text_transform_opts = TextTransformOpts()

    if fix_accents:
        text_transform_opts.fix_accents = True

    if fix_hyphenation:
        """Replace default dehyphen function"""
        # fix_hyphens: Callable[[str], str] = (
        #     remove_hyphens_fx(config.text_reader_opts.dehyphen_expr)
        #     if config.text_reader_opts.dehyphen_expr is not None
        #     else remove_hyphens
        # )
        text_transform_opts.fix_hyphenation = False
        text_transform_opts.extra_transforms.append(remove_hyphens)

    engine_args = {
        k: v
        for k, v in {
            'n_topics': n_topics,
            'passes': passes,
            'random_seed': random_seed,
            'alpha': alpha,
            'workers': workers,
            'max_iter': max_iter,
            'work_folder': os.path.join(target_folder, target_name),
        }.items()
        if v is not None
    }

    transform_opts: penelope.TokensTransformOpts = penelope.TokensTransformOpts(
        to_lower=to_lower,
        to_upper=False,
        min_len=min_word_length,
        max_len=max_word_length,
        remove_accents=False,
        remove_stopwords=(remove_stopwords is not None),
        stopwords=None,
        extra_stopwords=None,
        language=remove_stopwords,
        keep_numerals=keep_numerals,
        keep_symbols=keep_symbols,
        only_alphabetic=only_alphabetic,
        only_any_alphanumeric=only_any_alphanumeric,
    )

    extract_opts = penelope.ExtractTaggedTokensOpts(
        lemmatize=lemmatize,
        pos_includes=pos_includes,
        pos_excludes=pos_excludes,
        **config.pipeline_payload.tagged_columns_names,
    )

    main(
        config=config,
        target_name=target_name,
        corpus_source=corpus_source,
        target_folder=target_folder,
        text_transform_opts=text_transform_opts,
        extract_opts=extract_opts,
        transform_opts=transform_opts,
        engine=engine,
        engine_args=engine_args,
        store_corpus=store_corpus,
        store_compressed=store_compressed,
        enable_checkpoint=enable_checkpoint,
        force_checkpoint=force_checkpoint,
    )


def main(
    *,
    config: pipeline.CorpusConfig,
    target_name: str,
    corpus_source: str = None,
    target_folder: str = None,
    text_transform_opts: TextTransformOpts = None,
    extract_opts: penelope.ExtractTaggedTokensOpts = None,
    transform_opts: penelope.TokensTransformOpts = None,
    engine: str = "gensim_lda-multicore",
    engine_args: dict = None,
    store_corpus: bool = False,
    store_compressed: bool = True,
    enable_checkpoint: bool = True,
    force_checkpoint: bool = False,
):
    """ runner """

    corpus_source: str = corpus_source or config.pipeline_payload.source

    if corpus_source is None:
        click.echo("usage: either corpus source must be specified")
        sys.exit(1)

    _: dict = (
        config.get_pipeline(
            "tagged_frame_pipeline",
            corpus_source=corpus_source,
            enable_checkpoint=enable_checkpoint,
            force_checkpoint=force_checkpoint,
            text_transform_opts=text_transform_opts,
        )
        .tagged_frame_to_tokens(
            extract_opts=extract_opts,
            transform_opts=transform_opts,
        )
        .to_topic_model(
            corpus_source=None,
            target_folder=target_folder,
            target_name=target_name,
            engine=engine,
            engine_args=engine_args,
            store_corpus=store_corpus,
            store_compressed=store_compressed,
        )
    ).value()


if __name__ == '__main__':
    click_main()  # pylint: disable=no-value-for-parameter
