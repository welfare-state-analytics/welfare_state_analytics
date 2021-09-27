# type: ignore

import uuid

import pytest
from penelope.co_occurrence import ContextOpts
from penelope.corpus import TextReaderOpts, TokensTransformOpts, VectorizeOpts
from penelope.corpus.readers import ExtractTaggedTokensOpts
from penelope.notebook.interface import ComputeOpts
from penelope.pipeline import CorpusConfig, CorpusType
from penelope.utility import PropertyValueMaskingOpts
from penelope.workflows import co_occurrence as workflow

RESOURCE_FOLDER = '/home/roger/source/welfare-state-analytics/welfare_state_analytics/resources'
CONFIG_FILENAME = 'riksdagens-protokoll'
DATA_FOLDER = '/home/roger/source/welfare-state-analytics/welfare_state_analytics/data'


@pytest.mark.long_running
@pytest.mark.skip(reason="Used when debugging bugs")
def test_bug():

    corpus_config = CorpusConfig.find(CONFIG_FILENAME, RESOURCE_FOLDER).folders(DATA_FOLDER)
    # corpus_filename = '/home/roger/source/welfare-state-analytics/welfare_state_analytics/data/riksdagens-protokoll.1920-2019.sparv4.csv.zip'
    corpus_filename = './tests/test_data/prot_1975__59.zip'
    compute_opts = ComputeOpts(
        corpus_type=CorpusType.SparvCSV,
        corpus_filename=corpus_filename,
        target_folder='./tests/output',
        corpus_tag='APA',
        transform_opts=TokensTransformOpts(
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
        extract_opts=ExtractTaggedTokensOpts(
            lemmatize=True,
            target_override=None,
            pos_includes='|VB|',
            pos_excludes='|MAD|MID|PAD|',
            pos_paddings='|JJ|',
            passthrough_tokens=[],
            block_tokens=[],
            append_pos=False,
            global_tf_threshold=1,
            global_tf_threshold_mask=False,
            **corpus_config.pipeline_payload.tagged_columns_names,
        ),
        tf_threshold=1,
        tf_threshold_mask=False,
        filter_opts=PropertyValueMaskingOpts(),
        vectorize_opts=VectorizeOpts(
            already_tokenized=True, lowercase=False, stop_words=None, max_df=1.0, min_df=1, verbose=False
        ),
        create_subfolder=True,
        persist=True,
        context_opts=ContextOpts(
            context_width=1,
            concept={'information'},
            ignore_concept=False,
        ),
        enable_checkpoint=True,
        force_checkpoint=False,
    )

    corpus_config.pipeline_payload.files(
        source=compute_opts.corpus_filename,
        document_index_source=None,
    )
    bundle = workflow.compute(
        args=compute_opts,
        corpus_config=corpus_config,
        tagged_frames_filename='./tests/output/test.zip',
    )

    assert bundle is not None


def test_checkpoint_feather():
    corpus_config = CorpusConfig.find(CONFIG_FILENAME, RESOURCE_FOLDER).folders(DATA_FOLDER)

    FEATHER_FOLDER: str = f'./output/{uuid.uuid1()}'
    compute_opts: ComputeOpts = ComputeOpts(
        corpus_type=CorpusType.SparvCSV,
        corpus_filename='./tests/test_data/riksdagens-protokoll.1920-2019.9files.sparv4.csv.zip',
        target_folder='./tests/output/PROPAGANDA',
        corpus_tag='PROPAGANDA',
        transform_opts=TokensTransformOpts(
            only_alphabetic=False,
            only_any_alphanumeric=True,
            to_lower=True,
            to_upper=False,
            min_len=2,
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
        extract_opts=ExtractTaggedTokensOpts(
            lemmatize=True,
            target_override=None,
            pos_includes='|NN|PM|VB|',
            pos_excludes='|MAD|MID|PAD|',
            pos_paddings=None,
            passthrough_tokens=[],
            block_tokens=[],
            append_pos=False,
            global_tf_threshold=1,
            global_tf_threshold_mask=False,
            **corpus_config.pipeline_payload.tagged_columns_names,
        ),
        filter_opts=PropertyValueMaskingOpts(),
        vectorize_opts=VectorizeOpts(
            already_tokenized=True,
            lowercase=False,
            stop_words=None,
            max_df=1.0,
            min_df=1,
            verbose=False,
        ),
        tf_threshold=1,
        tf_threshold_mask=False,
        create_subfolder=True,
        persist=True,
        enable_checkpoint=True,
        force_checkpoint=False,
        context_opts=ContextOpts(
            context_width=2,
            concept={'propaganda'},
            ignore_concept=False,
            partition_keys=['year'],
        ),
    )

    corpus_config.checkpoint_opts.feather_folder = FEATHER_FOLDER
    corpus_config.pipeline_payload.files(
        source=compute_opts.corpus_filename,
        document_index_source=None,
    )
    bundle = workflow.compute(
        args=compute_opts,
        corpus_config=corpus_config,
        tagged_frames_filename='./tests/output/test.zip',
    )

    assert bundle is not None


# def test_load_co_occurrence_bundle():

#     filename: str = '/data/westac/shared/information_w3_NNPM_lemma_no_stops_NEW/information_w3_NNPM_lemma_no_stops_NEW_co-occurrence.csv.zip'

#     bundle: co_occurrence.Bundle = co_occurrence.Bundle.load(filename)

#     assert bundle is not None

#     trends_data: BundleTrendsData = BundleTrendsData(bundle=bundle)
#     assert trends_data is not None

#     co_occurrence_gui.ExploreGUI(bundle).setup().display(trends_data=trends_data)
