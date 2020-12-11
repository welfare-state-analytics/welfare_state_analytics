from penelope.corpus.readers import TextReaderOpts
from penelope.pipeline import CorpusConfig, CorpusType, PipelinePayload


def ParliamentarySessions(*, corpus_folder: str):  # pylint: disable=unused-argument

    return CorpusConfig(
        corpus_name='ParliamentarySessions',
        corpus_type=CorpusType.SpacyCSV,
        corpus_pattern='*sparv4.csv.zip',
        language='swedish',
        text_reader_opts=TextReaderOpts(
            filename_fields=r"year:prot\_(\d{4}).*",
            index_field=None,  # Use filename as key
            filename_filter=None,
            filename_pattern="*.csv",
            as_binary=False,
        ),
        pipeline_payload=PipelinePayload(
            source=None,
            document_index_source=None,
            document_index_key=None,
            document_index_sep=None,
            pos_schema_name="SUC",
            memory_store={
                'tagger': 'Sparv4',
                'text_column': '????',
                'pos_column': '????',
                'lemma_column': '???',
            },
        ),
    )
