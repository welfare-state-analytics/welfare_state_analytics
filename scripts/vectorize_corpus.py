import os
from os import close
import sys
import click
import re
import pandas as pd
from pprint import pprint as pp

root_folder = os.path.join(os.getcwd().split('welfare_state_analytics')[0], 'welfare_state_analytics')

sys.path = list(set(sys.path + [ root_folder ]))

import westac.common.utility as utility
import westac.corpus.corpus_vectorizer as corpus_vectorizer

def split_filename(filename, sep='_'):
    parts = filename.replace('.', sep).split(sep)
    return parts

def parse_extractor(data):

    if len(data) == 1:
        # regexp
        return data[0]

    if len(data) == 2:
        sep = data[0]
        position = int(data[1])
        return lambda f: f.replace('.', sep).split(sep)[position]

    raise Exception("to many parts in extract expression")

def parser_meta_fields(meta_field):

    try:
        meta_extract = {
            x[0]: parse_extractor(x[1:]) for x in [ y.split(':') for y in meta_field ]
        }

        return meta_extract
    except:
        print("parse error: meta-fields, must be in format 'name:regexp'")
        exit(-1)

@click.command()
@click.argument('filename')
@click.argument('output-folder')
@click.option('--to-lower/--no-to-lower', '-l', default=True, help='Transform text to lower case.')
@click.option('--remove-accents/--no-remove-accents', '-d', default=False, is_flag=True, help='Remove accents to lower case.')
@click.option('--min-length', default=2, help='Minimum length of words to keep', type=click.INT)
@click.option('--max-length', default=None, help='Maximum length of words to keep', type=click.INT)
@click.option('--doc-chunk-size', default=None, help='Split document in chunks of chunk-size words.', type=click.INT)
@click.option('--keep-numerals/--no-keep-numerals', default=True, is_flag=True, help='')
@click.option('--keep-symbols/--no-keep-symbols', default=False, is_flag=True, help='')
@click.option('--only-alphanumeric', default=False, is_flag=True, help='TBD.')
@click.option('--only-alphabetic', default=False, is_flag=True, help='')
@click.option('--file-pattern', default='*.txt', help='')
@click.option('--meta-field', '-f', help='RegExp fields to extract from document name', multiple=True)
def vectorize_text_corpus(
    filename=None,
    output_folder=None,
    to_lower=True,
    remove_accents=False,
    min_length=2,
    max_length=None,
    doc_chunk_size=None,
    keep_numerals=False,
    keep_symbols=False,
    only_alphanumeric=True,
    only_alphabetic=True,
    file_pattern='*.txt',
    meta_field=None
):

    kwargs = dict(
        isalnum=only_alphanumeric,
        to_lower=to_lower,
        remove_accents=remove_accents,
        min_len=min_length,
        max_len=max_length,
        doc_chunk_size=doc_chunk_size,
        keep_numerals=keep_numerals,
        keep_symbols=keep_symbols,
        only_alphabetic=only_alphabetic,
        pattern=file_pattern,
        meta_extract=parser_meta_fields(meta_field)
    )

    corpus_vectorizer.generate_corpus(filename, output_folder=output_folder, **kwargs)

if __name__ == "__main__":
    vectorize_text_corpus()