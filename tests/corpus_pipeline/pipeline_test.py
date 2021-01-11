from typing import List

import pandas as pd
from penelope.pipeline import ContentType, CorpusConfig, CorpusPipeline, CorpusSerializeOpts, DocumentPayload


def test_pipeline_can_load_pos_tagged_checkpoint():

    corpus_filename: str = './tests/test_data/riksdagens-protokoll.1920-2019.test.sparv4.csv.zip'
    config: CorpusConfig = CorpusConfig.load(path='./tests/test_data/riksdagens-protokoll.yml')
    options: CorpusSerializeOpts = config.serialize_opts.as_type(ContentType.TAGGEDFRAME)
    pipeline = CorpusPipeline(payload=config.pipeline_payload).load_tagged_frame(corpus_filename, options)

    payloads: List[DocumentPayload] = pipeline.to_list()

    assert len(payloads) == 158
    assert len(pipeline.payload.document_index) == 5
    assert isinstance(pipeline.payload.document_index, pd.DataFrame)


# def test_pipeline_load_text_tag_checkpoint_stores_checkpoint(config: CorpusConfig):

#     checkpoint_filename: str = os.path.join(OUTPUT_FOLDER, 'checkpoint_pos_tagged_test.zip')

#     transform_opts = TextTransformOpts()

#     os.makedirs(OUTPUT_FOLDER, exist_ok=True)
#     pathlib.Path(checkpoint_filename).unlink(missing_ok=True)

#     _ = (
#         SpacyPipeline(payload=config.pipeline_payload)
#         .set_spacy_model(config.pipeline_payload.memory_store['spacy_model'])
#         .load_text(reader_opts=config.text_reader_opts, transform_opts=transform_opts)
#         .text_to_spacy()
#         .tqdm()
#         .spacy_to_pos_tagged_frame()
#         .checkpoint(checkpoint_filename)
#     ).exhaust()

#     assert os.path.isfile(checkpoint_filename)
