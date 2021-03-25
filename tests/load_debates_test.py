from typing import Any, Dict, Iterator, List

import pytest
from penelope import corpus, pipeline
from penelope.pipeline import checkpoint
from westac.parliamentary_debates import LoadToTaggedFrame


@pytest.fixture
def config() -> pipeline.CorpusConfig:
    config = pipeline.CorpusConfig.load('./tests/test_data/parliamentary-debates.yml').files(
        source='./tests/test_data/annotated_parliamentary_debates',
        index_source=None,
    )
    return config


def test_load_task_protocol_stream(config: pipeline.CorpusConfig):
    source_folder = config.pipeline_payload.source
    attributes: List[str] = None
    attribute_value_filters: Dict[str, Any] = None
    reader_opts: corpus.TextReaderOpts = config.text_reader_opts
    serializer_opts: pipeline.CorpusSerializeOpts = config.serialize_opts

    token2id: pipeline.Token2Id = None
    task: LoadToTaggedFrame = LoadToTaggedFrame(
        source_folder=source_folder,
        attribute_value_filters=attribute_value_filters,
        attributes=attributes,
        reader_opts=reader_opts,
        serializer_opts=serializer_opts,
        token2id=token2id,
        instream=None,
        pipeline=None,
    )

    stream: Iterator[checkpoint.CheckpointData] = task.protocol_stream()

    cps: List[checkpoint.CheckpointData] = [cp for cp in stream]

    assert cps and len(cps) == 3
