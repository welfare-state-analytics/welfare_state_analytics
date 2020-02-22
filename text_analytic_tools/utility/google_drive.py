import os
import logging
import requests
import pandas as pd

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)

        logger.info('Stored: {}'.format(destination))
        print('Stored: {}'.format(destination))

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)

def extract_sheets(path, sheets):

    folder, filename = os.path.split(path)
    basename, _ = os.path.splitext(filename)

    with pd.ExcelFile(path) as xls:
        data = pd.read_excel(xls, sheet_name=None)

    for sheet_name in data.keys():

        if not sheet_name in data.keys():
            continue

        df = data[sheet_name]

        if not hasattr(df, 'to_csv'):
            continue

        csv_name = os.path.join(folder, '{}_{}.csv'.format(basename, sheet_name))

        if os.path.exists(csv_name):
            os.remove(csv_name)

        df.to_csv(csv_name, sep='\t')

        logger.info('Extracted: {}'.format(csv_name))
        print('Extracted: {}'.format(csv_name))

def process_file(file, overwrite=False):

    print('Processing: {}'.format(file['file_id']))
    if overwrite and os.path.exists(file['destination']):
        os.remove(file['destination'])
        logger.info('Removed: {}'.format(file['destination']))
    else:
        print('Skipping. File exists in ./data!')

    #if not os.path.exists(file['destination']):
    print('Downloading: {}'.format(file['file_id']))
    download_file_from_google_drive(file['file_id'], file['destination'])

    if len(file['sheets'] or []) > 0:
        extract_sheets(file['destination'], file['sheets'])

def process_files(files_to_download):

    for file in files_to_download:
        process_file(file)



