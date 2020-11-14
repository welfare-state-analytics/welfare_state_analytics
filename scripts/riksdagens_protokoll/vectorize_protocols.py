from penelope.corpus.tokens_transformer import TokensTransformOpts
from penelope.workflows import vectorize_corpus_workflow

tokens_transform_opts = TokensTransformOpts(
    keep_numerals=False,
    keep_symbols=False,
    max_len=None,
    min_len=2,
    only_alphabetic=True,
    only_any_alphanumeric=True,
    remove_accents=False,
    to_lower=True,
)

corpus_filename = './data/riksdagens_protokoll/riksdagens_protokoll_content_corpus_1945-1989.zip'
output_folder = './data/riksdagens_protokoll/'

vectorize_corpus_workflow(
    corpus_type="text",
    input_filename=corpus_filename,
    output_folder=output_folder,
    output_tag="abcdefg",
    create_subfolder=True,
    filename_field={'year': r"prot\_(\d{4}).*", 'year2': r"prot_\d{4}(\d{2})__*", 'number': r"prot_\d+__(\d+).*"},
    filename_pattern='*.txt',
    count_threshold=1,
    tokens_transform_opts=tokens_transform_opts,
)
