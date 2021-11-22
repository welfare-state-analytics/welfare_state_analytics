from typing import List

import pandas as pd
from penelope.pipeline import CheckpointOpts, ContentType, CorpusConfig, CorpusPipeline, DocumentPayload

CONFIG_FILENAME = './tests/test_data/riksdagens_protokoll/kb_labb/riksdagens-protokoll.yml'


def test_load_corpus_config_returns_correctly_escaped_values():

    config: CorpusConfig = CorpusConfig.load(CONFIG_FILENAME)

    assert config.checkpoint_opts.sep == '\t'
    assert config.text_reader_opts.filename_fields[0] == r'year:prot\_(\d{4}).*'
    assert config.text_reader_opts.filename_fields[1] == r'year2:prot_\d{4}(\d{2})__*'
    assert config.text_reader_opts.filename_fields[2] == r'number:prot_\d+[afk_]{0,4}__(\d+).*'
    assert config.text_reader_opts.sep == '\t'
    assert config.text_reader_opts.quoting == 3


def test_pipeline_can_load_pos_tagged_checkpoint():

    corpus_filename: str = './tests/test_data/riksdagens_protokoll/kb_labb/riksdagens-protokoll.1920-2019.9files.sparv4.csv.zip'
    config: CorpusConfig = CorpusConfig.load(CONFIG_FILENAME)
    checkpoint_opts: CheckpointOpts = config.checkpoint_opts.as_type(ContentType.TAGGED_FRAME)
    pipeline = CorpusPipeline(config=config).load_tagged_frame(
        corpus_filename, checkpoint_opts=checkpoint_opts, extra_reader_opts=config.text_reader_opts
    )

    payloads: List[DocumentPayload] = pipeline.to_list()

    assert len(payloads) == 9
    assert len(pipeline.payload.document_index) == 9
    assert isinstance(pipeline.payload.document_index, pd.DataFrame)
