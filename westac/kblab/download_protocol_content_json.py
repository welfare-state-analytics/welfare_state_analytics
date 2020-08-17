import os
import sys

root_folder = os.path.abspath("../..")
sys.path = list(set(sys.path + [ root_folder ]))

import download

def download_protocol_content_json():
    target_filename = "/home/roger/tmp/riksdagens_protokoll_content.zip"
    query = { "tags": "protokoll" }
    max_count = None
    excludes = [ "*.jpg", "*.jb2e", "*.xml", "coverage.*", "structure.json" ]
    includes = [ "content.json", "meta.json" ]
    download.download_query_to_zip(query, max_count, target_filename, includes=includes, excludes=excludes)

if __name__ == "__main__":

    download_protocol_content_json()
