import os
import sys


def find_root_folder(x):
    return os.path.join(os.getcwd().split(x)[0], x)


root_folder = ''

if os.environ.get("JUPYTER_IMAGE_SPEC", "") == "westac_lab":
    root_folder = "/home/jovyan/work/welfare_state_analytics"
else:
    root_folder = find_root_folder("welfare_state_analytics")

if root_folder not in sys.path:
    sys.path.insert(0, root_folder)
