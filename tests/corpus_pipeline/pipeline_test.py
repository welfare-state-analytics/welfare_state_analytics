from typing import List

import pandas as pd
from penelope.pipeline import ContentType, CorpusConfig, CorpusPipeline, CorpusSerializeOpts, DocumentPayload


def test_load_corpus_config_returns_correctly_escaped_values():

    config: CorpusConfig = CorpusConfig.load(path='./tests/test_data/riksdagens-protokoll.yml')

    assert config.content_deserialize_opts.sep == '\t'
    assert config.text_reader_opts.filename_fields[0] == r'year:prot\_(\d{4}).*'
    assert config.text_reader_opts.filename_fields[1] == r'year2:prot_\d{4}(\d{2})__*'
    assert config.text_reader_opts.filename_fields[2] == r'number:prot_\d+[afk_]{0,4}__(\d+).*'
    assert config.text_reader_opts.sep == '\t'
    assert config.text_reader_opts.quoting == 3


def test_pipeline_can_load_pos_tagged_checkpoint():

    corpus_filename: str = './tests/test_data/riksdagens-protokoll.1920-2019.test.sparv4.csv.zip'
    config: CorpusConfig = CorpusConfig.load(path='./tests/test_data/riksdagens-protokoll.yml')
    options: CorpusSerializeOpts = config.serialize_opts.as_type(ContentType.TAGGEDFRAME)
    pipeline = CorpusPipeline(config=config).load_tagged_frame(
        corpus_filename, options, extra_reader_opts=config.text_reader_opts
    )

    payloads: List[DocumentPayload] = pipeline.to_list()

    assert len(payloads) == 158
    assert len(pipeline.payload.document_index) == 158
    assert isinstance(pipeline.payload.document_index, pd.DataFrame)
