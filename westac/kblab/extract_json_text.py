import json
import logging
import os
import zipfile

import click

from . import utility

logger = logging.getLogger()

# pylint: disable=no-value-for-parameter

def extract_contents(source_filename):
    """Returns a stream of text extracted from downloaded content.json package file.

    Parameters
    ----------
    source_filename : str
        name of downloaded KB-LAB package files

    Yields
    -------
    Iterator[str, str, str, str]
        Stream of extracted (package-id, text, filename, date) tuples
    """
    with zipfile.ZipFile(source_filename, 'r') as zf:

        for package_id, filenames in utility.zip_folder_glob(zf, "*.json"):

            content_name = os.path.join(package_id, "content.json")
            meta_name = os.path.join(package_id, "meta.json")

            if content_name not in filenames:
                logger.warning("package {} has no content".format(package_id))
                continue

            if meta_name not in filenames:
                logger.warning("package {} has no meta".format(package_id))
                continue

            json_content = json.loads(zf.read(content_name).decode("utf-8"))
            json_meta = json.loads(zf.read(meta_name).decode("utf-8"))

            document = ''.join([
                block['content'] for block in json_content
            ])

            #parts = package_id.split('-')
            filename = "{}.txt".format(package_id.replace('-', '_')) # , json_meta['created'])
            print(filename)
            yield package_id, document, filename, json_meta['created']

@click.command()
@click.option('--source-filename', default='content.zip', help='Archive containing content.json files.')
@click.option('--target-filename', default='content_corpus.zip', help='Target text corpus filename.')
def extract_corpus(source_filename, target_filename):
    """Extracts text from downloaded content.json files and stores text in Zip-file.

    Parameters
    ----------
    source_filename : str
        name of archive that contains the downloaded KB-LAB package files
    target_filename : str
        Result (Zip) filename

    """
    texts = (
        (filename, text) for _, text, filename, _ in extract_contents(source_filename)
    )
    utility.store_to_zipfile(target_filename, texts)

if __name__ == '__main__':

#     source_filename = "/home/roger/tmp/riksdagens_protokoll_content.zip"
#     target_filename = "/home/roger/tmp/riksdagens_protokoll_content_corpus.zip"
#     extract_corpus(source_filename, target_filename)

    extract_corpus()
