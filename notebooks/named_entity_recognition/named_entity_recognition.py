# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: 'Python 3.7.5 64-bit (''welfare_state_analytics'': pipenv)'
#     name: python37564bitwelfarestateanalyticspipenvb857bd21a5fc4575b483276067dc0241
# ---

# %%
# %load_ext autoreload
# %autoreload 2

import glob
import os
import sys
import zipfile

import numpy as np
import pandas as pd

root_folder = (lambda x: os.path.join(os.getcwd().split(x)[0], x))("welfare_state_analytics")
sys.path.append(root_folder)


# %% [markdown]
# ## Convert Excel data to a yearly document index
# This script creates merges the text lines into a single text file for each year and news-paper.

# %%

# Convert excel to a temporary tab seperated text file

data_folder = os.path.join(root_folder, "data")
source_excel_filename = os.path.join(data_folder, "year+text_window.xlsx")
target_text_filename = os.path.join(data_folder, "year+newspaper+text.txt")
target_zip_filename = os.path.join(data_folder, "year+newspaper+text_yearly_document.txt.zip")


def create_yearly_documents(source_filename, target_name):

    df = pd.read_csv(source_filename, sep="\t")
    document_index = df.fillna("").groupby(["year", "newspaper"])["txt"].apply(" ".join).reset_index()

    with zipfile.ZipFile(target_name, "w") as zf:
        for _, document in document_index.iterrows():
            store_filename = "{}_{}.txt".format(document["newspaper"], document["year"])
            zf.writestr(store_filename, document["txt"], zipfile.ZIP_DEFLATED)


if not os.path.exists(target_zip_filename):
    print("Creating yearly document index...")
    # excel_to_csv(source_excel_filename, target_text_filename)
    # create_yearly_documents(target_text_filename, target_zip_filename)

print("OK!")

# %% [markdown]
# ## Run STAGGER NER tagging
# Note that archive created above must first be unzipped into a seperate folder.

# %% language="bash"
# # nohup java -Xmx4G -jar ~/source/stagger/stagger.jar -modelfile ~/source/stagger/models/swedish.bin -lang sv -tag *.txt &
#

# %% [markdown]
# ## Compile result

# %%


def read_conll_ner_tag(filename, only_ner_tags=True):

    df = pd.read_csv(filename, sep="\t", header=None, index_col=0, skip_blank_lines=True, quoting=3)
    df.columns = [
        "token",
        "lemma",
        "pos",
        "F4",
        "pos2",
        "F6",
        "F7",
        "F8",
        "F9",
        "tag",
        "type",
        "id",
    ]
    df = df[["id", "token", "pos", "tag", "type"]]

    df["parts"] = df.id.str.split("_")
    df["paper"] = df.parts.apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else "??")
    df["year"] = df.parts.apply(lambda x: x[1].split(":")[0] if isinstance(x, list) and len(x) > 1 else "0").astype(
        np.int32
    )

    df = df[["paper", "year", "token", "tag", "type"]]

    if only_ner_tags:
        df = df.loc[df.type != "_"]

    return df


result_folder = os.path.join(data_folder, "year+newspaper+text_yearly_document")
result_files = glob.glob("{}/*.conll".format(result_folder))

df_all_tags = pd.concat([read_conll_ner_tag(filename) for filename in result_files])
df_all_tags.to_excel("year+newspaper+text_yearly_document_all_ner_tags.xlsx")


# %%
