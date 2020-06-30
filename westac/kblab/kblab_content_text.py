import os
import zipfile
import utility
import logging
import json

logger = logging.getLogger()

def read_json(zf, filename):
    pass

def extract_contents(source_filename):

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

            yield package_id, document, filename, json_meta['created']

def extract_corpus(source_filename, target_filename):

    texts = (
        (filename, text) for _, text, filename, _ in extract_contents(source_filename)
    )
    utility.store_to_zipfile(target_filename, texts)

source_filename = "/home/roger/tmp/riksdagens_protokoll_content.zip"
target_filename = "/home/roger/tmp/riksdagens_protokoll_content_corpus.zip"

extract_corpus(source_filename, target_filename)