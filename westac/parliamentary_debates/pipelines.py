from typing import Callable, Optional
from penelope.corpus.readers.interfaces import FilenameOrCallableOrSequenceFilter, TextReaderOpts
from penelope.corpus.token2id import Token2Id
from penelope.pipeline import config, interfaces, pipelines
from . import tasks


def to_tagged_frame_pipeline(
    corpus_config: config.CorpusConfig,
    source_folder: Optional[str] = None,
    checkpoint_filter: Optional[Callable[[str], bool]] = None,
    filename_filter: Optional[FilenameOrCallableOrSequenceFilter] = None,
    filename_pattern: Optional[str] = None,
    show_progress: bool = False,
    merge_speeches: bool=False,
    **_,
) -> pipelines.CorpusPipeline:
    """Loads multiple checkpoint files from `source_folder` into a single sequential payload stream.

    The checkpoints can be filtered by callable `checkpoint_filter` that takes the filename as argument.

    Args:
        corpus_config (config.CorpusConfig): [description]
        source_folder (str, optional): [description]. Defaults to None.
        filename_filter (Optional[FilenameOrCallableOrSequenceFilter], optional): [description]. Defaults to None.
        filename_pattern (str, optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    p: pipelines.CorpusPipeline = pipelines.CorpusPipeline(config=corpus_config)
    reader_opts: TextReaderOpts = corpus_config.text_reader_opts.copy(
        filename_filter=filename_filter,
        filename_pattern=filename_pattern,
    )
    task: interfaces.ITask = tasks.ToIdTaggedFrame(
        source_folder=str(source_folder or corpus_config.pipeline_payload.source),
        checkpoint_filter=checkpoint_filter,
        checkpoint_opts=corpus_config.checkpoint_opts,
        reader_opts=reader_opts,
        show_progress=show_progress,
        merge_speeches=merge_speeches,
    )

    p.add(task)
    return p


def to_id_tagged_frame_pipeline(
    corpus_config: config.CorpusConfig,
    source_folder: str = None,
    checkpoint_filter: Callable[[str], bool] = None,
    filename_filter: Optional[FilenameOrCallableOrSequenceFilter] = None,
    filename_pattern: str = None,
    token2id: interfaces.Token2Id = None,
    lemmatize: bool = False,
    show_progress: bool = False,
    **_,
):
    """Loads multiple checkpoint files from `source_folder` into a single sequential payload stream.
    The tokens are mapped to integer values using Token2Id class.

    The checkpoints can be filtered by callable `checkpoint_filter` that takes the filename as argument.

    Args:
        corpus_config (config.CorpusConfig): [description]
        source_folder (str, optional): [description]. Defaults to None.
        filename_filter (Optional[FilenameOrCallableOrSequenceFilter], optional): [description]. Defaults to None.
        filename_pattern (str, optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """

    p: pipelines.CorpusPipeline = pipelines.CorpusPipeline(config=corpus_config)
    reader_opts: TextReaderOpts = corpus_config.text_reader_opts.copy(
        filename_filter=filename_filter,
        filename_pattern=filename_pattern,
    )
    task: interfaces.ITask = tasks.ToIdTaggedFrame(
        source_folder=str(source_folder or corpus_config.pipeline_payload.source),
        checkpoint_filter=checkpoint_filter,
        checkpoint_opts=corpus_config.checkpoint_opts,
        reader_opts=reader_opts,
        token2id=token2id or Token2Id(lowercase=True),
        lemmatize=lemmatize,
        show_progress=show_progress,
    )

    p.add(task)
    return p