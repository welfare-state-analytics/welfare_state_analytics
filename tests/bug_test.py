import pytest
from penelope import workflows
from penelope.co_occurrence.interface import ContextOpts
from penelope.corpus import TextReaderOpts, TokensTransformOpts
from penelope.corpus.dtm.vectorizer import VectorizeOpts
from penelope.corpus.readers.interfaces import ExtractTaggedTokensOpts, TaggedTokensFilterOpts
from penelope.notebook.interface import ComputeOpts
from penelope.pipeline import CorpusConfig, CorpusType

RESOURCE_FOLDER = '/home/roger/source/welfare-state-analytics/welfare_state_analytics/resources'
CONFIG_FILENAME = 'riksdagens-protokoll'
DATA_FOLDER = '/home/roger/source/welfare-state-analytics/welfare_state_analytics/data'


@pytest.skip(reason="Long running")
def test_bug():

    corpus_config = CorpusConfig.find(CONFIG_FILENAME, RESOURCE_FOLDER).folders(DATA_FOLDER)

    compute_opts = ComputeOpts(
        corpus_type=CorpusType.SparvCSV,
        corpus_filename='/home/roger/source/welfare-state-analytics/welfare_state_analytics/data/riksdagens-protokoll.1920-2019.sparv4.csv.zip',
        target_folder='/home/roger/source/welfare-state-analytics/welfare_state_analytics/data/APA',
        corpus_tag='APA',
        tokens_transform_opts=TokensTransformOpts(
            only_alphabetic=False,
            only_any_alphanumeric=False,
            to_lower=True,
            to_upper=False,
            min_len=1,
            max_len=None,
            remove_accents=False,
            remove_stopwords=True,
            stopwords=None,
            extra_stopwords=['örn'],
            language='swedish',
            keep_numerals=True,
            keep_symbols=True,
        ),
        text_reader_opts=TextReaderOpts(
            filename_pattern='*.csv',
            filename_filter=None,
            filename_fields=[
                'year:prot\\_(\\d{4}).*',
                'year2:prot_\\d{4}(\\d{2})__*',
                'number:prot_\\d+[afk_]{0,4}__(\\d+).*',
            ],
            index_field=None,
            as_binary=False,
            sep='\t',
            quoting=3,
        ),
        extract_tagged_tokens_opts=ExtractTaggedTokensOpts(
            lemmatize=True,
            target_override=None,
            pos_includes='|NN|PM|UO|PC|VB|',
            pos_excludes='|MAD|MID|PAD|',
            passthrough_tokens=[],
            append_pos=False,
        ),
        tagged_tokens_filter_opts=TaggedTokensFilterOpts(),
        vectorize_opts=VectorizeOpts(
            already_tokenized=True, lowercase=False, stop_words=None, max_df=1.0, min_df=1, verbose=False
        ),
        count_threshold=10,
        create_subfolder=True,
        persist=True,
        context_opts=ContextOpts(context_width=2, concept={'information'}, ignore_concept=False),
        partition_keys=['year'],
    )

    corpus_config.pipeline_payload.files(
        source=compute_opts.corpus_filename,
        document_index_source=None,
    )
    bundle = workflows.co_occurrence.compute(
        args=compute_opts,
        corpus_config=corpus_config,
        checkpoint_file='./tests/output/test.zip',
    )

    assert bundle is not None
