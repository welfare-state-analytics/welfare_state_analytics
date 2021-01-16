import os
import sys


def find_root_folder(x):
    return os.path.join(os.getcwd().split(x)[0], x)


project_name = 'welfare_state_analytics'
root_folder = ''

if os.environ.get("JUPYTER_IMAGE_SPEC", "") != "":
    root_folder = f"/home/jovyan/work/{project_name}"
else:
    root_folder = find_root_folder(project_name)

if root_folder not in sys.path:
    sys.path.insert(0, root_folder)

ROOT_FOLDER = root_folder

data_folder = os.path.join(root_folder, "data")
work_folder = os.path.join(root_folder, "data")
output_folder = os.path.join(root_folder, "output")
resources_folder = os.path.join(root_folder, "resources")
