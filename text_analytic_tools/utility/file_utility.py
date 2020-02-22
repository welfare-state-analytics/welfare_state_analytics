
import os
import re
import shutil
import time
import zipfile
import typing
import types
import fnmatch
import pathlib
import pandas as pd
import gensim
from .utils import getLogger

logger = getLogger()

HYPHEN_REGEXP = re.compile(r'\b(\w+)-\s*\r?\n\s*(\w+)\b', re.UNICODE)

def find_parent_folder(name):
    path = pathlib.Path(os.getcwd())
    folder = os.path.join(*path.parts[:path.parts.index(name)+1])
    return folder

def dehyphen(text: str):
    result = re.sub(HYPHEN_REGEXP, r"\1\2\n", text)
    return result

def list_files(path_name, pattern):
    px = lambda x: pattern.match(x) if isinstance(pattern, typing.re.Pattern) else fnmatch.fnmatch(x, pattern)
    if os.path.isdir(path_name):
        files = [ f for f in os.listdir(path_name) if os.path.isfile(os.path.join(path_name, f)) ]
    else:
        with zipfile.ZipFile(path_name) as zf:
            files = zf.namelist()

    return [ name for name in files if px(name) ]

def compress_whitespaces(text):
    result = re.sub(r'\s+', ' ', text).strip()
    return result

def extract_metadata(filename, **kwargs):
    params = { x: None for x in kwargs}
    meta =  types.SimpleNamespace(filename=filename, **params)
    for k,r in kwargs.items():
        if r is None:
            continue
        if isinstance(r, str): # typing.re.Pattern):
            m = re.match(r, filename)
            if m is not None:
                v = m.groups()[0]
                meta.__setattr__(k, int(v) if v.isnumeric() else v)
    return meta

def read_file(path, filename):
    if os.path.isdir(path):
        with open(os.path.join(path, filename), 'r') as file:
            content = file.read()
    else:
        with zipfile.ZipFile(path) as zf:
            with zf.open(filename, 'r') as file:
                content = file.read()
    content = gensim.utils.to_unicode(content, 'utf8', errors='ignore')
    return content

class FileUtility:

    @staticmethod
    def create(directory, clear_target_dir=False):

        if os.path.exists(directory) and clear_target_dir:
            shutil.rmtree(directory)

        if not os.path.exists(directory):
            os.makedirs(directory)

    @staticmethod
    def read_excel(filename, sheet):
        if not os.path.isfile(filename):
            raise Exception("File {0} does not exist!".format(filename))
        with pd.ExcelFile(filename) as xls:
            return pd.read_excel(xls, sheet)

    @staticmethod
    def save_excel(data, filename):
        with pd.ExcelWriter(filename) as writer: # pylint: disable=abstract-class-instantiated
            for (df, name) in data:
                df.to_excel(writer, name, engine='xlsxwriter')
            writer.save()

    @staticmethod
    def data_path(directory, filename):
        return os.path.join(directory, filename)

    @staticmethod
    def ts_data_path(directory, filename):
        return os.path.join(directory, '{}_{}'.format(time.strftime("%Y%m%d%H%M"), filename))

    @staticmethod
    def data_path_ts(directory, path):
        basename, extension = os.path.splitext(path)
        return os.path.join(directory, '{}_{}{}'.format(basename, time.strftime("%Y%m%d%H%M"), extension))

def compress_file(path):
    if not os.path.exists(path):
        logger.error("ERROR: file not found (zip)")
        return
    folder, filename = os.path.split(path)
    basename, _ = os.path.splitext(filename)
    zip_name = os.path.join(folder, basename + '.zip')
    with zipfile.ZipFile(zip_name, mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(path)
    os.remove(path)
