import os
import sys

root_folder = os.path.join(os.getcwd().split('welfare_state_analytics')[0], 'welfare_state_analytics')

#root_folder = os.path.abspath("..")
sys.path = list(set(sys.path + [ root_folder ]))

import westac.corpus.corpus_vectorizer as corpus_vectorizer

kwargs = dict(
    only_any_alphanumeric=True,
    to_lower=True,
    remove_accents=False,
    min_len=2,
    max_len=None,
    keep_numerals=False,
    keep_symbols=False,
    only_alphabetic=True,
    pattern='*.txt',
    filename_fields = {
        'year': r"prot\_(\d{4}).*",
        'year2': r"prot_\d{4}(\d{2})__*",
        'number': r"prot_\d+__(\d+).*"
    }
)


corpus_filename = os.path.join(root_folder, 'data/riksdagens_protokoll/riksdagens_protokoll_content_corpus_1945-1989.zip')
output_folder = os.path.join(root_folder, 'data/riksdagens_protokoll/')

corpus_vectorizer.generate_corpus(corpus_filename, output_folder=output_folder, **kwargs)
