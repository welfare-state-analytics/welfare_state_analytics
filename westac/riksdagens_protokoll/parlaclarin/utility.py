import os
from penelope.utility import replace_extension, strip_path_and_extension
import pandas as pd

jj = os.path.join


def to_feather(df: pd.DataFrame, target_filename: str) -> None:
    df.reset_index(drop=len(df) == 0).to_feather(target_filename, compression="lz4")


def to_zip(df: pd.DataFrame, target_filename: str) -> None:
    archive_name: str = f"{strip_path_and_extension(target_filename)}.csv"
    compression: dict = dict(method='zip', archive_name=archive_name)
    target_filename = replace_extension(target_filename, "zip")
    df.to_csv(target_filename, compression=compression, sep='\t', header=True, decimal=',')


def to_csv(df: pd.DataFrame, target_filename: str) -> None:
    df.to_csv(target_filename, sep='\t', header=True, decimal=',')


def to_xlsx(df: pd.DataFrame, target_filename: str) -> None:
    df.to_excel(target_filename)

