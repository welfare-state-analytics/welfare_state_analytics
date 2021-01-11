import os

import pandas as pd
from penelope.co_occurrence import to_co_occurrence_matrix
from penelope.corpus.readers import PandasCorpusReader

# NOTE THIS FILE COMPUTES COUCCURRENCE FROM PREDEFINED WINDOWS READ FROM EXCEL FILE!


def load_text_windows(filename: str):
    """Reads excel file "filename" and returns content as a Pandas DataFrame.
    The file is written to tsv the first time read for faster subsequent reads.

    Parameters
    ----------
    filename : str
        Name of excel file that has two columns: year and txt

    Returns
    -------
    [DataFrame]
        Content of filename as a DataFrame

    Raises
    ------
    FileNotFoundError
    """
    filepath = os.path.abspath(filename)

    if not os.path.isdir(filepath):
        raise FileNotFoundError("Path {filepath} does not exist!")

    filebase = os.path.basename(filename).split('.')[0]
    textfile = os.path.join(filepath, filebase + '.txt')

    if not os.path.isfile(textfile):
        df = pd.read_excel(filename)
        df.to_csv(textfile, sep='\t')

    df = pd.read_csv(textfile, sep='\t')[['newspaper', 'year', 'txt']]

    return df


def compute_for_period_newpaper(df, period, newspaper, min_count, options):
    reader = PandasCorpusReader(df, year=period, newspaper=newspaper)
    df_y = to_co_occurrence_matrix(reader, min_count=min_count, **options)
    df_y['newspaper'] = newspaper
    df_y['period'] = str(period)
    return df_y


def compute_co_occurrence_for_periods(source_filename, newspapers, periods, target_filename, min_count=1, **options):

    columns = ['newspaper', 'period', 'w1', 'w2', 'value', 'value_n_d', 'value_n_t']

    df = pd.read_csv(source_filename, sep='\t')[['newspaper', 'year', 'txt']]
    df_r = pd.DataFrame(columns=columns)

    n_documents = 0
    for newspaper in newspapers:
        for period in periods:
            print("Processing: {} {}...".format(newspaper, period))
            df_y = compute_for_period_newpaper(df, period, newspaper, min_count, options)
            df_r = df_r.append(df_y[columns], ignore_index=True)
            n_documents += len(df_y)

    print("Done! Processed {} rows...".format(n_documents))

    # Scale a normalized data matrix to the [0, 1] range:
    df_r['value_n_t'] = df_r.value_n_t / df_r.value_n_t.max()
    df_r['value_n_d'] = df_r.value_n_d / df_r.value_n_d.max()

    extension = target_filename.split(".")[-1]
    if extension == ".xlsx":
        df_r.to_excel(target_filename, index=False)
    elif extension in ["zip", "gzip"]:
        df_r.to_csv(target_filename, sep='\t', compression=extension, index=False, header=True)
    else:
        df_r.to_csv(target_filename, sep='\t', index=False, header=True)
