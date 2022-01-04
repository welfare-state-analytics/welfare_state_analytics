from typing import Callable, Optional
from penelope import corpus, utility, pipeline
from . import tasks


def load_tagged_frame_pipeline(
    corpus_config: pipeline.CorpusConfig,
    corpus_source: Optional[str] = None,
    checkpoint_filter: Optional[Callable[[str], bool]] = None,
    filename_filter: Optional[corpus.FilenameFilterSpec] = None,
    filename_pattern: Optional[str] = None,
    show_progress: bool = False,
    merge_speeches: bool = False,
    **_,
) -> pipeline.CorpusPipeline:
    """Loads multiple checkpoint files from `corpus_source` into a single sequential payload stream.

    The checkpoints can be filtered by callable `checkpoint_filter` that takes the filename as argument.

    Args:
        corpus_config (config.CorpusConfig): [description]
        corpus_source (str, optional): [description]. Defaults to None.
        filename_filter (Optional[FilenameFilterSpec], optional): [description]. Defaults to None.
        filename_pattern (str, optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    p: pipeline.CorpusPipeline = pipeline.CorpusPipeline(config=corpus_config)
    reader_opts: corpus.TextReaderOpts = corpus_config.text_reader_opts.copy(
        filename_filter=filename_filter,
        filename_pattern=filename_pattern,
    )
    task: pipeline.ITask = tasks.LoadTaggedFrame(
        corpus_source=str(corpus_source or corpus_config.pipeline_payload.source),
        checkpoint_filter=checkpoint_filter,
        checkpoint_opts=corpus_config.checkpoint_opts,
        reader_opts=reader_opts,
        show_progress=show_progress,
        merge_speeches=merge_speeches,
    )

    p.add(task)
    return p


# FIXME: to_tagged_frame_pipeline is specified in corpus config, but has other signature than those used in generic penelope-scripts
# Currently ParlaClarin pipelines have their own scripts
to_tagged_frame_pipeline = load_tagged_frame_pipeline


def to_id_tagged_frame_pipeline(
    corpus_config: pipeline.CorpusConfig,
    corpus_source: str = None,
    checkpoint_filter: Callable[[str], bool] = None,
    filename_filter: Optional[corpus.FilenameFilterSpec] = None,
    filename_pattern: str = None,
    token2id: corpus.Token2Id = None,
    lemmatize: bool = False,
    show_progress: bool = False,
    **_,
):
    """Loads multiple checkpoint files from `corpus_source` into a single sequential payload stream.
    The tokens are mapped to integer values using Token2Id class.

    The checkpoints can be filtered by callable `checkpoint_filter` that takes the filename as argument.

    Args:
        corpus_config (config.CorpusConfig): [description]
        corpus_source (str, optional): [description]. Defaults to None.
        filename_filter (Optional[FilenameFilterSpec], optional): [description]. Defaults to None.
        filename_pattern (str, optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    p: pipeline.CorpusPipeline = pipeline.CorpusPipeline(config=corpus_config)
    reader_opts: corpus.TextReaderOpts = corpus_config.text_reader_opts.copy(
        filename_filter=filename_filter,
        filename_pattern=filename_pattern,
    )
    task: pipeline.ITask = tasks.ToIdTaggedFrame(
        corpus_source=str(corpus_source or corpus_config.pipeline_payload.source),
        checkpoint_filter=checkpoint_filter,
        checkpoint_opts=corpus_config.checkpoint_opts,
        reader_opts=reader_opts,
        token2id=token2id or corpus.Token2Id(lowercase=True),
        lemmatize=lemmatize,
        show_progress=show_progress,
    )

    p.add(task)
    return p


def to_topic_model_pipeline(
    *,
    config: pipeline.CorpusConfig,
    corpus_source: str,
    engine: str,
    engine_args: dict,
    target_folder: str,
    target_name: str,
    transform_opts: corpus.TokensTransformOpts,
    extract_opts: corpus.ExtractTaggedTokensOpts,
    checkpoint_filter: Optional[Callable[[str], bool]] = None,
    filename_filter: Optional[corpus.FilenameFilterSpec] = None,
    filename_pattern: Optional[str] = None,
):
    p: pipeline.CorpusPipeline = (
        to_tagged_frame_pipeline(
            corpus_source=corpus_source,
            corpus_config=config,
            checkpoint_filter=checkpoint_filter,
            filename_filter=filename_filter,
            filename_pattern=filename_pattern,
        )
        .tagged_frame_to_tokens(extract_opts=extract_opts, transform_opts=transform_opts)
        .to_topic_model(
            corpus_source=None,
            target_folder=target_folder,
            target_name=target_name,
            engine=engine,
            engine_args=engine_args,
            store_corpus=True,
            store_compressed=True,
        )
    )
    return p
