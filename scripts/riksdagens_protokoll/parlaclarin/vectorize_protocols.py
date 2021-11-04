# pylint: disable=too-many-arguments, too-many-locals, unused-import

import os

import click
import loguru
import penelope.notebook.interface as interface
import westac.riksdagens_protokoll.pipelines as rp_pipeline
from penelope.corpus import ExtractTaggedTokensOpts, TokensTransformOpts, VectorizeOpts
from penelope.pipeline import CorpusConfig, CorpusPipelineBase  # ,  Token2Id

# from westac.parliamentary_debates.members import ParliamentaryData

logger = loguru.logger

# FIXME Align with changes in penelope scripts (diff)


@click.command()
@click.pass_context
def profile_vectorize(ctx: click.Context):
    input_folder = '/data/riksdagen_corpus_data/annotated'
    config = './resources/parliamentary-debates.yml'
    output_folder = './tmp'
    output_tag = 'ZYGON'
    _ = ctx.invoke(vectorize, content=ctx.forward(input_folder, output_folder, config=config, output_tag=output_tag))  # type: ignore


@click.command()
@click.argument('input_folder', type=click.STRING)
@click.argument('output_folder', type=click.STRING)
@click.option('-c', '--config', type=click.STRING, required=True)
@click.option('-t', '--output-tag', type=click.STRING, required=True)
@click.option(
    '--tf-threshold',
    default=1,
    type=click.IntRange(1, 99),
    help='Globoal TF threshold filter (words below filtered out)',
)
@click.option(
    '--tf-threshold-mask',
    default=False,
    is_flag=True,
    help='If true, then low TF words are kept, but masked as "__low_tf__"',
)
@click.option(
    '-i',
    '--pos-includes',
    default=None,
    help='List of POS tags to include e.g. "|NOUN|JJ|".',
    type=click.STRING,
)
@click.option(
    '-i',
    '--pos-paddings',
    default=None,
    help='List of POS tags to include in output as padded tokens.',
    type=click.STRING,
)
@click.option(
    '-x',
    '--pos-excludes',
    default='|MAD|MID|PAD|',
    help='List of POS tags to exclude e.g. "|MAD|MID|PAD|".',
    type=click.STRING,
)
@click.option('-b', '--lemmatize/--no-lemmatize', default=True, is_flag=True, help='Use word baseforms')
@click.option('-l', '--to-lower/--no-to-lower', default=True, is_flag=True, help='Lowercase words')
@click.option(
    '-r',
    '--remove-stopwords',
    default=None,
    type=click.Choice(['swedish', 'english']),
    help='Remove stopwords using given language',
)
@click.option('--min-word-length', default=1, type=click.IntRange(1, 99), help='Min length of words to keep')
@click.option('--max-word-length', default=None, type=click.IntRange(10, 99), help='Max length of words to keep')
@click.option('--keep-symbols/--no-keep-symbols', default=True, is_flag=True, help='Keep symbols')
@click.option('--keep-numerals/--no-keep-numerals', default=True, is_flag=True, help='Keep numerals')
@click.option(
    '--only-alphabetic',
    default=False,
    is_flag=True,
    help='Keep only tokens having only alphabetic characters',
)
@click.option(
    '--only-any-alphanumeric',
    default=False,
    is_flag=True,
    help='Keep tokens with at least one alphanumeric char',
)
@click.option(
    '--merge-speeches',
    default=False,
    is_flag=True,
    help='Merge speeches in each protocol into a single document',
)
def _vectorize(
    input_folder: str = "",
    output_folder: str = None,
    config: str = "",
    output_tag: str = None,
    create_subfolder: bool = True,
    pos_includes: str = '|NN|',
    pos_excludes: str = '|MAD|MID|PAD|',
    pos_paddings: str = None,
    to_lower: bool = True,
    lemmatize: bool = True,
    remove_stopwords: str = None,
    min_word_length: int = 2,
    max_word_length: int = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    only_any_alphanumeric: bool = False,
    only_alphabetic: bool = False,
    tf_threshold: int = None,
    tf_threshold_mask: bool = None,
    merge_speeches: bool = False,
):

    vectorize(
        input_folder=input_folder,
        output_folder=output_folder,
        config=config,
        output_tag=output_tag,
        create_subfolder=create_subfolder,
        pos_includes=pos_includes,
        pos_excludes=pos_excludes,
        pos_paddings=pos_paddings,
        to_lower=to_lower,
        lemmatize=lemmatize,
        remove_stopwords=remove_stopwords,
        min_word_length=min_word_length,
        max_word_length=max_word_length,
        keep_symbols=keep_symbols,
        keep_numerals=keep_numerals,
        only_any_alphanumeric=only_any_alphanumeric,
        only_alphabetic=only_alphabetic,
        tf_threshold=tf_threshold,
        tf_threshold_mask=tf_threshold_mask,
        merge_speeches=merge_speeches,
    )


def vectorize(
    input_folder: str = "",
    output_folder: str = None,
    config: str = "",
    output_tag: str = None,
    create_subfolder: bool = True,
    pos_includes: str = '|NN|',
    pos_excludes: str = '|MAD|MID|PAD|',
    pos_paddings: str = None,
    to_lower: bool = True,
    lemmatize: bool = True,
    remove_stopwords: str = None,
    min_word_length: int = 2,
    max_word_length: int = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    only_any_alphanumeric: bool = False,
    only_alphabetic: bool = False,
    tf_threshold: int = None,
    tf_threshold_mask: bool = False,
    merge_speeches: bool = False,
):

    try:
        corpus_config: CorpusConfig = CorpusConfig.load(config)

        if not os.path.isdir(input_folder):
            raise FileNotFoundError(input_folder)

        args: interface.ComputeOpts = interface.ComputeOpts(
            corpus_type=corpus_config.corpus_type,
            corpus_filename=input_folder,
            target_folder=output_folder,
            corpus_tag=output_tag,
            transform_opts=TokensTransformOpts(
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
            ),
            text_reader_opts=corpus_config.text_reader_opts,
            extract_opts=ExtractTaggedTokensOpts(
                pos_includes=pos_includes,
                pos_excludes=pos_excludes,
                pos_paddings=pos_paddings,
                lemmatize=lemmatize,
                tf_threshold=tf_threshold,
                tf_threshold_mask=tf_threshold_mask,
                **corpus_config.pipeline_payload.tagged_columns_names,
            ),
            filter_opts=None,
            vectorize_opts=VectorizeOpts(already_tokenized=True),
            tf_threshold=tf_threshold,
            tf_threshold_mask=tf_threshold_mask,
            create_subfolder=create_subfolder,
            persist=True,
        )
        # parliament_data = ParliamentaryData.load()
        _: CorpusPipelineBase = (
            rp_pipeline.to_tagged_frame_pipeline(
                source_folder=input_folder,
                corpus_config=corpus_config,
                checkpoint_filter=None,
                filename_filter=None,
                filename_pattern=None,
                show_progress=True,
                merge_speeches=merge_speeches,
            )
            .tagged_frame_to_tokens(
                extract_opts=args.extract_opts,
                filter_opts=args.filter_opts,
                transform_opts=None,
            )
            .exhaust(100)
        )
        # for payload in pipeline.take(1000): pass

        # pipeline.payload.document_index.to_csv('./parliamentary_debates_document_index.csv', sep='\t')
        logger.info('Done!')

    except Exception as ex:
        raise ex


if __name__ == '__main__':
    _vectorize()  # pylint: disable=no-value-for-parameter
