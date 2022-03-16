# type: ignore

import os
import shutil
import uuid

import pytest
from penelope.co_occurrence import ContextOpts
from penelope.corpus import TextReaderOpts, TokensTransformOpts, VectorizeOpts
from penelope.corpus.readers import ExtractTaggedTokensOpts
from penelope.notebook.dtm import ComputeGUI
from penelope.pipeline import CheckpointOpts, CorpusConfig, CorpusType, PipelinePayload
from penelope.workflows import co_occurrence as workflow
from penelope.workflows.interface import ComputeOpts

KB_LABB_DATA_FOLDER = './tests/test_data/riksdagens_protokoll/kb_labb'

jj = os.path.join


@pytest.mark.skip("bug fixed")
def test_bug_load_word_trends():
    corpus_folder: str = '/home/roger/source/welfare-state-analytics/welfare_state_analytics/data'
    data_folder: str = '/home/roger/source/welfare-state-analytics/welfare_state_analytics/data'
    corpus_config: CorpusConfig = CorpusConfig(
        corpus_name='riksdagens-protokoll',
        corpus_type=CorpusType.SparvCSV,
        corpus_pattern='*sparv4.csv.zip',
        checkpoint_opts=CheckpointOpts(
            content_type_code=1,
            document_index_name=None,
            document_index_sep=None,
            sep='\t',
            quoting=3,
            custom_serializer_classname='penelope.pipeline.sparv.convert.SparvCsvSerializer',
            deserialize_processes=6,
            deserialize_chunksize=4,
            text_column='token',
            lemma_column='baseform',
            pos_column='pos',
            extra_columns=[],
            frequency_column=None,
            index_column=None,
            feather_folder='/data/westac/shared/checkpoints/riksdagens-protokoll.1920-2019.sparv4.csv_feather',
            lower_lemma=True,
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
            n_processes=1,
            n_chunksize=2,
        ),
        pipelines={'tagged_frame_pipeline': 'penelope.pipeline.sparv.pipelines.to_tagged_frame_pipeline'},
        pipeline_payload=PipelinePayload(
            source='/home/roger/source/welfare-state-analytics/welfare_state_analytics/data/riksdagens-protokoll.1920-2019.sparv4.csv.zip',
            document_index_source=None,
            document_index_sep='\t',
            memory_store={
                'lang': 'se',
                'tagger': 'Sparv',
                'sparv_version': 4,
                'text_column': 'token',
                'lemma_column': 'baseform',
                'pos_column': 'pos',
            },
            pos_schema_name='SUC',
            filenames=None,
            metadata=None,
            token2id=None,
            effective_document_index=None,
        ),
        language='swedish',
    )
    compute_callback = None
    done_callback = None

    gui = ComputeGUI(
        default_corpus_path=corpus_folder,
        default_corpus_filename=(corpus_config.pipeline_payload.source or ''),
        default_data_folder=data_folder,
    ).setup(
        config=corpus_config,
        compute_callback=compute_callback,
        done_callback=done_callback,
    )

    assert gui is not None


@pytest.mark.long_running
@pytest.mark.skip(reason="Used when debugging bugs")
def test_bug():
    config_filename = jj(KB_LABB_DATA_FOLDER, 'riksdagens-protokoll.yml')
    corpus_config = CorpusConfig.load(config_filename).folders(KB_LABB_DATA_FOLDER)
    corpus_source = jj(KB_LABB_DATA_FOLDER, 'prot_1975__59.zip')
    compute_opts = ComputeOpts(
        corpus_type=CorpusType.SparvCSV,
        corpus_source=corpus_source,
        target_folder='./tests/output',
        corpus_tag=f'{uuid.uuid1()}',
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
        vectorize_opts=VectorizeOpts(already_tokenized=True, lowercase=False, stop_words=None, max_df=1.0, min_df=1),
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
        source=compute_opts.corpus_source,
        document_index_source=None,
    )
    bundle = workflow.compute(
        args=compute_opts,
        corpus_config=corpus_config,
        tagged_corpus_source='./tests/output/test.zip',
    )

    assert bundle is not None


@pytest.mark.long_running
def test_checkpoint_feather():

    config_filename = jj(KB_LABB_DATA_FOLDER, 'riksdagens-protokoll.yml')
    corpus_config = CorpusConfig.load(config_filename).folders(KB_LABB_DATA_FOLDER)
    feather_folder: str = f'./tests/output/{uuid.uuid1()}'
    compute_opts: ComputeOpts = ComputeOpts(
        corpus_type=CorpusType.SparvCSV,
        corpus_source=jj(KB_LABB_DATA_FOLDER, 'riksdagens-protokoll.1920-2019.9files.sparv4.csv.zip'),
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
        vectorize_opts=VectorizeOpts(
            already_tokenized=True,
            lowercase=False,
            stop_words=None,
            max_df=1.0,
            min_df=1,
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

    corpus_config.checkpoint_opts.feather_folder = feather_folder
    corpus_config.pipeline_payload.files(
        source=compute_opts.corpus_source,
        document_index_source=None,
    )
    bundle = workflow.compute(
        args=compute_opts,
        corpus_config=corpus_config,
        tagged_corpus_source='./tests/output/test.zip',
    )

    assert bundle is not None

    shutil.rmtree(feather_folder, ignore_errors=True)
