import os
import zipfile
from pprint import pprint as pp

import pandas as pd


def read_file(path, filename):
    with zipfile.ZipFile(path) as zf:
        with zf.open(filename, 'r') as file:
            content = file.read()
    # content = gensim.utils.to_unicode(content, 'utf8', errors='ignore')
    return content


def read_file_index(index_name, year):
    df = pd.read_csv(index_name, header=None)[[1, 2, 3, 7]]
    df.columns = ['key_id', 'year', 'sou_id', 'part_id']
    df = df.fillna('d1')
    return df.loc[df['year'] == year]


# wget https://data.riksdagen.se/dataset/dokument/sou-1990-1999.csv.zip
# wget https://data.riksdagen.se/dataset/dokument/sou-1990-1999.text.zip

data_folder = '../../data'
index_name = os.path.join(data_folder, 'sou-1990-1999.csv')
archive_name = os.path.join(data_folder, 'sou-1990-1999.text.zip')

df = read_file_index(index_name, 1996)
for index, row in df.iterrows():
    original_filename = '{}.txt'.format(row['key_id'].lower())
    content = read_file(archive_name, original_filename)
    filename = '{}_{}_{}.txt'.format(row['year'], row['sou_id'], row['part_id'])
    print(original_filename, len(content), filename)
    with open(os.path.join(data_folder, filename), 'wb') as out:
        out.write(content)
