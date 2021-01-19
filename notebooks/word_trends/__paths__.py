import os
import sys


def find_root_folder(x):
    return os.path.join(os.getcwd().split(x)[0], x)


root_folder = ''

if os.environ.get("JUPYTER_IMAGE_SPEC", "") != "":
    root_folder = "/home/jovyan/work/text_analytics"
else:
    root_folder = find_root_folder("text_analytics")

if root_folder not in sys.path:
    sys.path.insert(0, root_folder)

ROOT_FOLDER = root_folder

data_folder = os.path.join(root_folder, "data")
work_folder = os.path.join(root_folder, "data")
resources_folder = os.path.join(root_folder, "resources")
