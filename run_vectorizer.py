import os
import click

from westac.corpus import corpus_vectorizer
from westac.corpus import text_corpus
from westac.corpus import vectorized_corpus
from westac.corpus import file_text_reader

def generate_corpus(filename, output_folder, **kwargs):

    if not os.path.isfile(filename):
        print('error: no such file: {}'.format(filename))
        return

    dump_tag = '{}_{}_{}_{}'.format(
        os.path.basename(filename).split('.')[0],
        'L{}'.format(kwargs.get('min_len', 0)),
        '-N' if kwargs.get('numerals', False) else '+N',
        '-S' if kwargs.get('symbols', False) else '+S',
    )

    if vectorized_corpus.VectorizedCorpus.dump_exists(dump_tag):
        print('notice: removing existing result files...')
        os.remove(os.path.join(output_folder, '{}_vector_data.npy'.format(dump_tag)))
        os.remove(os.path.join(output_folder, '{}_vectorizer_data.pickle'.format(dump_tag)))

    print('Creating new corpus...')
    reader = file_text_reader.FileTextReader(
        filename, meta_extract=kwargs.get("meta_extract"),
        compress_whitespaces=True,
        dehyphen=True,
        pattern=kwargs.get("pattern", "*.txt")
    )
    corpus = text_corpus.ProcessedCorpus(reader, **kwargs)

    print('Creating document-term matrix...')
    vectorizer = corpus_vectorizer.CorpusVectorizer()
    v_corpus = vectorizer.fit_transform(corpus)

    print('Saving data matrix...')
    v_corpus.dump(tag=dump_tag, folder=output_folder)

@click.command()
@click.argument('filename') #, help='Model name.')
@click.option('--output-folder', default='./output', help='Output folder.')
def _generate_corpus(filename, output_folder, **kwargs):

    kwargs = dict(
        isalnum=True,
        to_lower=True,
        deacc=False,
        min_len=2,
        max_len=None,
        numerals=False,
        symbols=False,
        only_alphabetic=True,
        pattern='*.txt',
        meta_extract = {
            'year': r"(\d{4})\_.*",
            'treaty_id': r"\d{4}\_(.{6}).*"
            # 'year': r"(\d{4})\_.*",
            # 'serial_no': r"\d{4}\_(\d+).*"
        }
    )


    generate_corpus(filename, output_folder=output_folder, **kwargs)

if __name__ == '__main__':
    # filename = './data/SOU_1945-1989_NN+VB+JJ_lemma.zip'
    _generate_corpus()