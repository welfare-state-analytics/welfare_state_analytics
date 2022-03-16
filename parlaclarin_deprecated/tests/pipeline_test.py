import os
from typing import List

import pandas as pd
import pytest
from penelope import pipeline
from westac.riksdagens_protokoll import parlaclarin

# pylint: disable=redefined-outer-name, too-many-function-args

CONFIG_FILENAME = './tests/test_data/riksdagens_protokoll/riksdagens-protokoll.yml'
TAGGED_CORPUS_FOLDER = './tests/test_data/riksdagens_protokoll/v0.4.0/tagged_corpus'
INDEX_COLUMNS = [
    'speech_id',
    'speaker',
    'speech_date',
    'speech_index',
    'document_name',
    'filename',
    'num_tokens',
    'num_words',
    'document_id',
]


@pytest.fixture
def config() -> pipeline.CorpusConfig:
    return pipeline.CorpusConfig.load(CONFIG_FILENAME)


@pytest.mark.long_running
@pytest.mark.parametrize('corpus_source', [TAGGED_CORPUS_FOLDER])
def test_load_tagged_frame_pipeline(corpus_source: str, config: pipeline.CorpusConfig):

    pp: pipeline.CorpusPipeline = parlaclarin.load_tagged_frame_pipeline(
        corpus_config=config,
        corpus_source=corpus_source,
        checkpoint_filter=None,
        filename_filter=None,
        filename_pattern=None,
    )

    payloads: List[pipeline.DocumentPayload] = [x for x in pp.resolve()]
    document_index: pd.DataFrame = pp.payload.document_index

    assert len(payloads) == 15 == len(document_index)
    assert all(x in document_index.columns for x in INDEX_COLUMNS)


@pytest.mark.long_running
@pytest.mark.parametrize('corpus_source', [TAGGED_CORPUS_FOLDER])
def test_load_tagged_frame_pipeline_with_filename_filter(corpus_source: str, config: pipeline.CorpusConfig):
    def checkpoint_filter(path: str) -> bool:
        basename: str = os.path.basename(path)
        return basename.startswith('prot-1932')

    pp: pipeline.CorpusPipeline = parlaclarin.load_tagged_frame_pipeline(
        corpus_config=config,
        corpus_source=corpus_source,
        checkpoint_filter=checkpoint_filter,
        filename_filter=None,
        filename_pattern=None,
    )

    payloads: List[pipeline.DocumentPayload] = [x for x in pp.resolve()]
    document_index: pd.DataFrame = pp.payload.document_index

    assert len(payloads) == 1 == len(document_index)
    assert all(x in document_index.columns for x in INDEX_COLUMNS)

    # parliamentary_data = members.ParliamentaryData.load(members.GITHUB_DATA_URL)
    # assert parliamentary_data is not None


@pytest.mark.long_running
@pytest.mark.parametrize('corpus_source', [TAGGED_CORPUS_FOLDER])
def test_to_id_tagged_frame_pipeline(corpus_source: str, config: pipeline.CorpusConfig):

    pp: pipeline.CorpusPipeline = parlaclarin.to_id_tagged_frame_pipeline(
        corpus_config=config,
        corpus_source=corpus_source,
        checkpoint_filter=None,
        filename_filter=None,
        filename_pattern=None,
    )

    payloads: List[pipeline.DocumentPayload] = [x for x in pp.resolve()]
    document_index: pd.DataFrame = pp.payload.document_index

    assert len(payloads) == 15 == len(document_index)

    tagged_document: pd.DataFrame = payloads[0].content
    # assert all(x in tagged_document.columns for x in ['token', 'token_id', 'pos_id'])
    assert (~tagged_document.pos_id.isna()).all()

    assert all(x in document_index.columns for x in INDEX_COLUMNS)
