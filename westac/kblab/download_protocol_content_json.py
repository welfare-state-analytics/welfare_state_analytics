import os
import sys

import dotenv

import westac.kblab.download as download

root_folder = os.path.join(os.getcwd().split('welfare_state_analytics')[0], 'welfare_state_analytics')

sys.path = list(set(sys.path + [ root_folder ]))


def download_protocol_content_json():

    dotenv.load_dotenv(dotenv_path=os.path.join(os.environ['HOME'], '.vault/.kblab.env'))

    target_filename = "/home/roger/tmp/riksdagens_protokoll_content.zip"
    query = { "tags": "protokoll" }
    max_count = None
    excludes = [ "*.jpg", "*.jb2e", "*.xml", "coverage.*", "structure.json" ]
    includes = [ "content.json", "meta.json" ]
    download.download_query_to_zip(query, max_count, target_filename, includes=includes, excludes=excludes)

if __name__ == "__main__":

    download_protocol_content_json()
