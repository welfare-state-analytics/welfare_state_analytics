# type: ignore

from os.path import join as jj

import penelope.workflows.co_occurrence.compute as workflow
from penelope import corpus as corpora
from penelope import pipeline, utility
from penelope.co_occurrence import ContextOpts
from penelope.notebook.interface import ComputeOpts

DATA_FOLDER = "./tests/test_data/riksdagens_protokoll/kb_labb"
CONCEPT = set()  # {'information'}

SUC_SCHEMA: utility.PoS_Tag_Scheme = utility.PoS_Tag_Schemes.SUC
POS_TARGETS: str = 'NN|PM'
POS_PADDINGS: str = 'AB|DT|HA|HD|HP|HS|IE|IN|JJ|KN|PC|PL|PN|PP|PS|RG|RO|SN|UO|VB'
POS_EXLUDES: str = 'MAD|MID|PAD'

config_filename = jj(DATA_FOLDER, "riksdagens-protokoll.yml")
corpus_config = pipeline.CorpusConfig.load(config_filename).folders(DATA_FOLDER)

corpus_source = jj(DATA_FOLDER, 'riksdagens-protokoll.1920-2019.9files.sparv4.csv.zip')

compute_opts = ComputeOpts(
    corpus_type=pipeline.CorpusType.SparvCSV,
    corpus_source=corpus_source,
    target_folder='/home/roger/source/welfare-state-analytics/welfare_state_analytics/data/APA',
    corpus_tag='APA',
    transform_opts=corpora.TokensTransformOpts(
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
    text_reader_opts=corpora.TextReaderOpts(
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
    extract_opts=corpora.ExtractTaggedTokensOpts(
        lemmatize=True,
        target_override=None,
        pos_includes=POS_TARGETS,
        pos_excludes=POS_EXLUDES,
        pos_paddings=POS_PADDINGS,
        passthrough_tokens=[],
        block_tokens=[],
        append_pos=False,
        global_tf_threshold=1,
        global_tf_threshold_mask=False,
        **corpus_config.pipeline_payload.tagged_columns_names,
    ),
    filter_opts=utility.PropertyValueMaskingOpts(),
    vectorize_opts=corpora.VectorizeOpts(
        already_tokenized=True, lowercase=False, stop_words=None, max_df=1.0, min_df=1, verbose=False
    ),
    tf_threshold=1,
    tf_threshold_mask=False,
    create_subfolder=True,
    persist=True,
    context_opts=ContextOpts(
        context_width=1,
        concept=CONCEPT,
        ignore_concept=False,
        partition_keys=['year'],
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
    tagged_frames_filename='./tests/output/test.zip',
)

assert bundle is not None
