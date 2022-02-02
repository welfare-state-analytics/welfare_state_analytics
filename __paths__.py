import sys

from penelope.utility.paths import find_data_folder, find_resources_folder, find_root_folder

project_name: str = 'welfare_state_analytics'
project_short_name: str = "westac"

corpus_folder: str = find_data_folder(project_name=project_name, project_short_name=project_short_name)
root_folder: str = find_root_folder(project_name=project_name)
resources_folder: str = find_resources_folder(project_name=project_name, project_short_name=project_short_name)

data_folder: str = corpus_folder

if root_folder not in sys.path:
    sys.path.insert(0, root_folder)
