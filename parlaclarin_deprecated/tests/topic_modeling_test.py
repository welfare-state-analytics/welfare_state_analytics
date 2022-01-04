import uuid
from typing import Callable, Optional

import pandas as pd
import pytest
from penelope import corpus, pipeline
from westac.riksdagens_protokoll import parlaclarin

# pylint: disable=redefined-outer-name, too-many-function-args, too-many-locals

CONFIG_FILENAME = './tests/test_data/riksdagens_protokoll/parlaclarin/riksdagens-protokoll.yml'
TAGGED_CORPUS_FOLDER = './tests/test_data/riksdagens_protokoll/parlaclarin/tagged_corpus_01'
DEFAULT_ENGINE_ARGS = {
    'n_topics': 4,
    'passes': 1,
    'random_seed': 42,
    'alpha': 'symmetric',
    'workers': 1,
    'max_iter': 100,
    'work_folder': './tests/output/',
}


@pytest.fixture
def config() -> pipeline.CorpusConfig:
    return pipeline.CorpusConfig.load(CONFIG_FILENAME)


def topic_model_payload(engine: str, config: pipeline.CorpusConfig) -> pipeline.DocumentPayload:
    extract_opts: corpus.ExtractTaggedTokensOpts = corpus.ExtractTaggedTokensOpts(
        lemmatize=True, pos_includes='', pos_excludes='MAD|MID|PAD', **config.pipeline_payload.tagged_columns_names
    )
    target_name: str = f'{uuid.uuid1()}'
    p = parlaclarin.to_topic_model_pipeline(
        config=config,
        corpus_source=TAGGED_CORPUS_FOLDER,
        engine=engine,
        engine_args=DEFAULT_ENGINE_ARGS,
        target_folder='./tests/output',
        target_name=target_name,
        transform_opts=corpus.TokensTransformOpts(),
        extract_opts=extract_opts,
    )

    payload: pipeline.DocumentPayload = p.single()

    return payload


@pytest.mark.parametrize('method', ["gensim_lda-multicore", "gensim_mallet-lda"])
def test_create_payload(method: str, config: pipeline.CorpusConfig):
    payload: pipeline.DocumentPayload = topic_model_payload(method, config)
    assert payload is not None


@pytest.mark.long_running
@pytest.mark.parametrize(
    'corpus_source, engine',
    [
        (TAGGED_CORPUS_FOLDER, "gensim_lda-multicore"),
        (TAGGED_CORPUS_FOLDER, "gensim_mallet-lda"),
    ],
)
def test_to_topic_model_pipeline(corpus_source: str, engine: str, config: pipeline.CorpusConfig):

    target_folder: str = './tests/output'
    target_name: str = f'{uuid.uuid4()}'
    transform_opts: corpus.TokensTransformOpts = corpus.TokensTransformOpts()
    extract_opts: corpus.ExtractTaggedTokensOpts = corpus.ExtractTaggedTokensOpts(
        lemmatize=True,
        pos_includes='',
        pos_excludes='|MAD|MID|PAD|',
        global_tf_threshold=1,
        **config.pipeline_payload.tagged_columns_names,
    )
    engine_args: dict = {
        'n_topics': 4,
        'passes': 1,
        'random_seed': 42,
        'alpha': 'symmetric',
        'workers': 1,
        'max_iter': 100,
        'work_folder': './tests/output/',
    }
    store_corpus: bool = True
    store_compressed: bool = True
    checkpoint_filter: Optional[Callable[[str], bool]] = None
    filename_filter: Optional[corpus.FilenameFilterSpec] = None
    filename_pattern: Optional[str] = None

    # pp: pipeline.CorpusPipeline = parlaclarin.to_topic_model_pipeline(
    #     config=config,
    #     corpus_source=corpus_source,
    #     engine=engine,
    #     engine_args=engine_args,
    #     target_folder=target_folder,
    #     target_name=target_name,
    #     transform_opts=transform_opts,
    #     extract_opts=extract_opts,
    #     checkpoint_filter=checkpoint_filter,
    #     filename_filter=filename_filter,
    #     filename_pattern=filename_pattern,
    # )
    pp: pipeline.CorpusPipeline = (
        config.get_pipeline(
            "tagged_frame_pipeline",
            # corpus_source=corpus_source,
            # enable_checkpoint=enable_checkpoint,
            # force_checkpoint=force_checkpoint,
            # text_transform_opts=text_transform_opts,
            corpus_source=corpus_source,
            checkpoint_filter=checkpoint_filter,
            filename_filter=filename_filter,
            filename_pattern=filename_pattern,
        )
        # parlaclarin.to_tagged_frame_pipeline(
        #     corpus_config=config,
        #     corpus_source=corpus_source,
        #     checkpoint_filter=checkpoint_filter,
        #     filename_filter=filename_filter,
        #     filename_pattern=filename_pattern,
        # )
        .tagged_frame_to_tokens(extract_opts=extract_opts, transform_opts=transform_opts).to_topic_model(
            corpus_source=None,
            target_folder=target_folder,
            target_name=target_name,
            engine=engine,
            engine_args=engine_args,
            store_corpus=store_corpus,
            store_compressed=store_compressed,
        )
    )

    value: dict = pp.value()

    assert isinstance(value, dict)
    assert value.get('target_name') == target_name
    assert value.get('target_folder') == target_folder

    document_index: pd.DataFrame = pp.payload.document_index

    assert len(document_index) == 15


# @pytest.mark.long_running
# @pytest.mark.parametrize('method', ["gensim_lda-multicore", "gensim_mallet-lda"])
# def test_predict_topics(method: str):

#     payload: DocumentPayload = tranströmer_topic_model_payload(method=method)
#     config: CorpusConfig = CorpusConfig.load('./tests/test_data/tranströmer.yml')
#     corpus_source: str = './tests/test_data/tranströmer_corpus_pos_csv.zip'

#     target_folder: str = './tests/output'
#     target_name: str = f'{uuid.uuid1()}'

#     model_folder: str = payload.content.get("target_folder")
#     model_name: str = payload.content.get("target_name")

#     transform_opts = TokensTransformOpts()
#     extract_opts = ExtractTaggedTokensOpts(
#         lemmatize=True,
#         pos_includes='',
#         pos_excludes='MAD|MID|PAD',
#         **config.checkpoint_opts.tagged_columns,
#     )
#     payload: DocumentPayload = (
#         CorpusPipeline(config=config)
#         .load_tagged_frame(
#             filename=corpus_source,
#             checkpoint_opts=config.checkpoint_opts,
#             extra_reader_opts=config.text_reader_opts,
#         )
#         .tagged_frame_to_tokens(extract_opts=extract_opts, transform_opts=transform_opts)
#         .predict_topics(
#             model_folder=model_folder,
#             model_name=model_name,
#             target_folder=target_folder,
#             target_name=target_name,
#         )
#     ).single()

#     assert payload is not None

#     model_infos = find_models('./tests/output')
#     assert any(m['name'] == target_name for m in model_infos)
#     model_info = next(m for m in model_infos if m['name'] == target_name)
#     assert 'method' in model_info['options']


# @pytest.mark.long_running
# @pytest.mark.parametrize("method", ["gensim_lda-multicore", "gensim_mallet-lda"])
# def test_topic_model_task_with_token_stream_and_document_index(method):

#     target_name: str = f'{uuid.uuid1()}'
#     corpus = TranströmerCorpus()

#     payload_stream = lambda: [
#         DocumentPayload(content_type=ContentType.TOKENS, filename=filename, content=tokens)
#         for filename, tokens in corpus
#     ]

#     pipeline = Mock(
#         spec=CorpusPipeline,
#         **{'payload.memory_store': SPARV_TAGGED_COLUMNS, 'payload.document_index': corpus.document_index},
#     )

#     prior = MagicMock(
#         spec=ITask,
#         outstream=payload_stream,
#         content_stream=lambda: ContentStream(payload_stream),
#         out_content_type=ContentType.TOKENS,
#     )

#     task: ToTopicModel = ToTopicModel(
#         pipeline=pipeline,
#         prior=prior,
#         corpus_source=None,
#         target_folder="./tests/output",
#         target_name=target_name,
#         engine=method,
#         engine_args=utility.DEFAULT_ENGINE_ARGS,
#         store_corpus=True,
#         store_compressed=True,
#     )

#     task.setup()
#     task.enter()
#     payload: DocumentPayload = next(task.process_stream())

#     assert payload is not None
#     assert payload.content_type == ContentType.TOPIC_MODEL
#     assert isinstance(payload.content, dict)

#     output_models = find_models('./tests/output')
#     assert any(m['name'] == target_name for m in output_models)
