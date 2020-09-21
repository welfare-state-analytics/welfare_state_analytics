import os
import sys

import click

import westac.corpus.iterators.sparv_xml_iterator as sparv_xml_iterator
from westac.corpus import utility
from westac.corpus.tokens_transformer import TokensTransformer

root_folder = os.path.join(os.getcwd().split('welfare_state_analytics')[0], 'welfare_state_analytics')
sys.path = list(set(sys.path + [ root_folder ]))


@click.command()
@click.argument('input') #, help='Model name.')
@click.argument('output') #, help='Model name.')
@click.option('--pos-includes', default='', help='List of POS tags to include e.g. "|NN|JJ|".')
@click.option('--pos-excludes',  default='|MAD|MID|PAD|', help='List of POS tags to exclude e.g. "|MAD|MID|PAD|".')
@click.option('--chunk-size', 'chunk_size', default=None, help='Document chunk size, defult one.')
@click.option('--lemmatize/--no-lemmatize', default=True, is_flag=True, help='')
@click.option('--lower/--no-lower', default=True, is_flag=True, help='')
@click.option('--remove-stopwords', default=None, type=click.Choice(['swedish', 'english']), help='Remove stopwords using given language')
@click.option('--min-word-length', default=None, type=click.IntRange(1,99), help='Remove stopwords using given language')
@click.option('--keep-symbols/--no-keep-symbols', default=True, is_flag=True, help='Keep symbols')
@click.option('--keep-numerals/--no-keep-numerals', default=True, is_flag=True, help='Keep numerals')
@click.option('--version', 'version', default=4, help='Sparv version i.e. 3 or 4', type=click.IntRange(3,4))
def prepare_train_corpus(input, output, pos_includes, pos_excludes, chunk_size, lemmatize, lower, remove_stopwords, min_word_length, keep_symbols, keep_numerals, version):
    """Prepares the a training corpus from Sparv XML archive
    """
    transformer = TokensTransformer(
        to_lower=lower,
        remove_stopwords=remove_stopwords is not None,
        language=remove_stopwords,
        min_len=min_word_length,
        max_len=None,
        keep_numerals=keep_numerals,
        keep_symbols=keep_symbols
    )

    opts = { **sparv_xml_iterator.DEFAULT_OPTS, **{
        'transforms': transformer.transforms,
        'pos_includes': pos_includes,
        'pos_excludes': pos_excludes,
        'chunk_size': chunk_size,
        'lemmatize': lemmatize,
        'version': version,
    }}

    sparv_xml_iterator.sparv_extract_and_store(input, output, **opts)

if __name__ == '__main__':
    prepare_train_corpus()
