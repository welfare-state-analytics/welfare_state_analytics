import os
import sys

root_folder = os.path.abspath("..")
sys.path = list(set(sys.path + [ root_folder ]))

from westac.kblab import kblab_download

def download_proto_content():
    target_filename = "/home/roger/tmp/riksdagens_protokoll_content.zip"
    query = { "tags": "protokoll" }
    max_count = None
    excludes = [ "*.jpg", "*.jb2e", "*.xml", "coverage.*", "structure.json" ]
    includes = [ "content.json", "meta.json" ]
    kblab_download.kblab_download_query_to_zip(query, max_count, target_filename, includes=includes, excludes=excludes)

if __name__ == "__main__":

    download_proto_content()
