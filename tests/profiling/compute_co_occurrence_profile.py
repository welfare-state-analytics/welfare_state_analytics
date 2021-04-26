from penelope import corpus as corpora
from penelope import pipeline, utility, workflows
from penelope.co_occurrence import ContextOpts
from penelope.notebook.interface import ComputeOpts

RESOURCE_FOLDER = "./resources/"
CONFIG_FILENAME = "riksdagens-protokoll.yml"
DATA_FOLDER = "./tests/test_data"
CONCEPT = {}  # {'information'}

SUC_SCHEMA: utility.PoS_Tag_Scheme = utility.PoS_Tag_Schemes.SUC
POS_TARGETS: str = 'NN|PM'
POS_PADDINGS: str = 'AB|DT|HA|HD|HP|HS|IE|IN|JJ|KN|PC|PL|PN|PP|PS|RG|RO|SN|UO|VB'
POS_EXLUDES: str = 'MAD|MID|PAD'

corpus_config = pipeline.CorpusConfig.find(CONFIG_FILENAME, RESOURCE_FOLDER).folders(DATA_FOLDER)
corpus_filename = './tests/test_data/riksdagens-protokoll.1920-2019.9files.sparv4.csv.zip'
compute_opts = ComputeOpts(
    corpus_type=pipeline.CorpusType.SparvCSV,
    corpus_filename=corpus_filename,
    target_folder='/home/roger/source/welfare-state-analytics/welfare_state_analytics/data/APA',
    corpus_tag='APA',
    tokens_transform_opts=corpora.TokensTransformOpts(
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
    extract_tagged_tokens_opts=corpora.ExtractTaggedTokensOpts(
        lemmatize=True,
        target_override=None,
        pos_includes=POS_TARGETS,
        pos_excludes=POS_EXLUDES,
        pos_paddings=POS_PADDINGS,
        passthrough_tokens=[],
        append_pos=False,
    ),
    tagged_tokens_filter_opts=utility.PropertyValueMaskingOpts(),
    vectorize_opts=corpora.VectorizeOpts(
        already_tokenized=True, lowercase=False, stop_words=None, max_df=1.0, min_df=1, verbose=False
    ),
    count_threshold=1,
    create_subfolder=True,
    persist=True,
    context_opts=ContextOpts(context_width=1, concept=CONCEPT, ignore_concept=False),
    partition_keys=['year'],
    force=False,
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
