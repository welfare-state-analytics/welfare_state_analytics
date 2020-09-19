import os
import sys

import click
import dotenv

import westac.kblab.download as download

root_folder = os.path.join(os.getcwd().split('welfare_state_analytics')[0], 'welfare_state_analytics')

sys.path = list(set(sys.path + [ root_folder ]))

# pylint: disable=no-value-for-parameter

@click.command()
@click.argument('tag', )
@click.argument('target_filename', )
@click.option('--year', default=None, type=click.INT)
def download_tag_content_json(tag, target_filename, year=None):

    dotenv.load_dotenv(dotenv_path=os.path.join(os.environ['HOME'], '.vault/.kblab.env'))

    query = { "tags": tag }

    max_count = None

    excludes = [ "*.jpg", "*.jb2e", "*.xml", "coverage.*", "structure.json" ]
    includes = [ "content.json", "meta.json" ]

    if year is not None:

        #for year in range(year_range[0], year_range[1] + 1):

        query["meta.created"] = str(year)

        download.download_query_to_zip(query, max_count, target_filename, includes=includes, excludes=excludes, append=True)

    else:

        download.download_query_to_zip(query, max_count, target_filename, includes=includes, excludes=excludes, append=True)

if __name__ == "__main__":

    download_tag_content_json()
