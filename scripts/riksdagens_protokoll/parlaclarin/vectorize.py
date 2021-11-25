import os
import sys
from typing import Sequence

import click
import penelope.notebook.interface as interface
import penelope.workflows.document_term_matrix as workflow
import westac.riksdagens_protokoll.parlaclarin as parlaclarin
from loguru import logger
from penelope.corpus import ExtractTaggedTokensOpts, TextReaderOpts, TokensTransformOpts, VectorizeOpts
from penelope.pipeline import CorpusConfig, CorpusPipelineBase  # ,  Token2Id
from penelope.pipeline.phrases import parse_phrases
from penelope.utility import PropertyValueMaskingOpts, pos_tags_to_str

# pylint: disable=too-many-arguments, too-many-locals, unused-import, unused-argument


@click.command()
@click.argument('corpus_config', type=click.STRING)
@click.argument('corpus_source', type=click.STRING)
@click.argument('output_folder', type=click.STRING)
@click.argument('output_tag')
@click.option('-g', '--filename-pattern', default=None, help='Filename pattern', type=click.STRING)
@click.option('-i', '--pos-includes', default='', help='POS tags to include e.g. "|NN|JJ|".', type=click.STRING)
@click.option('-m', '--pos-paddings', default='', help='POS tags to replace with a padding marker.', type=click.STRING)
@click.option('-x', '--pos-excludes', default='', help='POS tags to exclude.', type=click.STRING)
@click.option('-a', '--append-pos', default=False, is_flag=True, help='Append PoS to tokems')
@click.option('-m', '--phrase', default=None, help='Phrase', multiple=True, type=click.STRING)
@click.option('-z', '--phrase-file', default=None, help='Phrase filename', multiple=False, type=click.STRING)
@click.option('-b', '--lemmatize/--no-lemmatize', default=True, is_flag=True, help='Use word baseforms')
@click.option('-l', '--to-lower/--no-to-lower', default=True, is_flag=True, help='Lowercase words')
@click.option(
    '-r',
    '--remove-stopwords',
    default=None,
    type=click.Choice(['swedish', 'english']),
    help='Remove stopwords using given language',
)
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
@click.option('--min-word-length', default=1, type=click.IntRange(1, 99), help='Min length of words to keep')
@click.option('--max-word-length', default=None, type=click.IntRange(10, 99), help='Max length of words to keep')
@click.option('--keep-symbols/--no-keep-symbols', default=True, is_flag=True, help='Keep symbols')
@click.option('--keep-numerals/--no-keep-numerals', default=True, is_flag=True, help='Keep numerals')
@click.option(
    '--only-alphabetic', default=False, is_flag=True, help='Keep only tokens having only alphabetic characters'
)
@click.option(
    '--only-any-alphanumeric', default=False, is_flag=True, help='Keep tokens with at least one alphanumeric char'
)
@click.option(
    '--merge-speeches',
    default=False,
    is_flag=True,
    help='Merge speeches in each protocol into a single document',
)
@click.option(
    '-n',
    '--deserialize-processes',
    default=4,
    type=click.IntRange(1, 99),
    help='Number of processes during deserialization',
)
def main(
    corpus_config: str = None,
    corpus_source: str = None,
    output_folder: str = None,
    output_tag: str = None,
    filename_pattern: str = None,
    phrase: Sequence[str] = None,
    phrase_file: str = None,
    create_subfolder: bool = True,
    pos_includes: str = '',
    pos_paddings: str = '',
    pos_excludes: str = '',
    append_pos: bool = False,
    to_lower: bool = True,
    lemmatize: bool = True,
    remove_stopwords: str = None,
    min_word_length: int = 2,
    max_word_length: int = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    only_any_alphanumeric: bool = False,
    only_alphabetic: bool = False,
    tf_threshold: int = 1,
    tf_threshold_mask: bool = False,
    deserialize_processes: int = 4,
    merge_speeches: bool = False,
):

    process(
        corpus_config=corpus_config,
        corpus_source=corpus_source,
        output_folder=output_folder,
        output_tag=output_tag,
        filename_pattern=filename_pattern,
        phrase=phrase,
        phrase_file=phrase_file,
        create_subfolder=create_subfolder,
        pos_includes=pos_includes,
        pos_paddings=pos_paddings,
        pos_excludes=pos_excludes,
        append_pos=append_pos,
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
        deserialize_processes=deserialize_processes,
        merge_speeches=merge_speeches,
    )


def process(
    corpus_config: str = None,
    corpus_source: str = None,
    output_folder: str = None,
    output_tag: str = None,
    filename_pattern: str = None,
    phrase: Sequence[str] = None,
    phrase_file: str = None,
    create_subfolder: bool = True,
    pos_includes: str = None,
    pos_paddings: str = None,
    pos_excludes: str = None,
    append_pos: bool = False,
    to_lower: bool = True,
    lemmatize: bool = True,
    remove_stopwords: str = None,
    min_word_length: int = 2,
    max_word_length: int = None,
    keep_symbols: bool = False,
    keep_numerals: bool = False,
    only_any_alphanumeric: bool = False,
    only_alphabetic: bool = False,
    tf_threshold: int = 1,
    tf_threshold_mask: bool = False,
    deserialize_processes: int = 4,
    merge_speeches: bool = False,
):
    opts: dict = locals()

    try:
        corpus_config: CorpusConfig = CorpusConfig.load(corpus_config)
        phrases = parse_phrases(phrase_file, phrase)

        if pos_excludes is None:
            pos_excludes = pos_tags_to_str(corpus_config.pos_schema.Delimiter)

        if pos_paddings.upper() in ["FULL", "ALL", "PASSTHROUGH"]:
            pos_paddings = pos_tags_to_str(corpus_config.pos_schema.all_types_except(pos_includes))
            logger.info(f"PoS paddings expanded to: {pos_paddings}")

        text_reader_opts: TextReaderOpts = corpus_config.text_reader_opts.copy()

        if filename_pattern is not None:
            text_reader_opts.filename_pattern = filename_pattern

        corpus_config.checkpoint_opts.deserialize_processes = max(1, deserialize_processes)

        tagged_columns: dict = corpus_config.pipeline_payload.tagged_columns_names

        if not os.path.isdir(corpus_source):
            raise FileNotFoundError(corpus_source)

        args: interface.ComputeOpts = interface.ComputeOpts(
            corpus_type=corpus_config.corpus_type,
            corpus_source=corpus_source,
            target_folder=output_folder,
            corpus_tag=output_tag,
            transform_opts=TokensTransformOpts().ingest(opts),
            text_reader_opts=text_reader_opts,
            extract_opts=ExtractTaggedTokensOpts(
                phrases=phrases,
                **tagged_columns,
            ).ingest(opts),
            vectorize_opts=VectorizeOpts(already_tokenized=True),
            tf_threshold=tf_threshold,
            tf_threshold_mask=tf_threshold_mask,
            create_subfolder=create_subfolder,
            persist=True,
            filter_opts=PropertyValueMaskingOpts(),
            enable_checkpoint=False,
            force_checkpoint=False,
        )

        # parliament_data = ParliamentaryData.load()

        tagged_frame_pipeline: CorpusPipelineBase = parlaclarin.load_tagged_frame_pipeline(
            corpus_source=corpus_source,
            corpus_config=corpus_config,
            checkpoint_filter=None,
            filename_filter=None,
            filename_pattern=None,
            show_progress=True,
            merge_speeches=merge_speeches,
        )

        workflow.compute(args=args, corpus_config=corpus_config, tagged_frame_pipeline=tagged_frame_pipeline)

        # pipeline.payload.document_index.to_csv('./parliamentary_debates_document_index.csv', sep='\t')
        logger.info('Done!')

    except Exception as ex:  # pylint: disable=try-except-raise
        logger.exception(ex)
        click.echo(ex)
        sys.exit(1)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
