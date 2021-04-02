import os
import sys


def find_root_folder(x: str) -> str:
    return os.path.join(os.getcwd().split(x)[0], x)


project_name: str = 'welfare_state_analytics'

if os.environ.get("JUPYTER_IMAGE_SPEC", "") != "":
    root_folder: str = f"/home/jovyan/work/{project_name}"
    corpus_folder: str = "/data/westac"
else:
    root_folder: str = find_root_folder(project_name)
    corpus_folder: str = os.path.join(root_folder, "data")

if root_folder not in sys.path:
    sys.path.insert(0, root_folder)

ROOT_FOLDER: str = root_folder

data_folder: str = os.path.join(root_folder, "data")
resources_folder: str = os.path.join(root_folder, "resources")
