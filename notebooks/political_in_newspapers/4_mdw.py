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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Most Discriminating Terms
# Note! The computation can take a very long time depending on corpus size and arguments. It might be preferable to use the CLI version instead.
#
# ### Example CLI call
# Compare AB 1949-1949 to DN 1989-1989. Only the 200 top words are included in the computation, and at most 100 words are returned per group. Words that occur in few documents ( less than 3%), and many documents (more than 95%) are filtered out.
#
# ```
# python mdw_runner.py --group AB 1945 1949 --group DN 1985 1989 --top-n-terms 200 --max-n-terms 100 --min-df 3.0 --max-df 95.0
# ```
#
# The result is stored in an excel file which is named based on the execution parameters.
#

# %%
# %load_ext autoreload
# %autoreload 2

import os
import sys

from .notebook_gui import mdw_gui


def get_root_folder():
    parts = os.getcwd().split(os.path.sep)
    parts = parts[: parts.index("welfare_state_analytics") + 1]
    return os.path.join("/", *parts)


if os.environ.get("JUPYTER_IMAGE_SPEC", "") == "westac_lab":
    root_folder = "/home/jovyan/welfare_state_analytics"
    corpus_folder = "/data/westac/textblock_politisk"
else:
    root_folder = get_root_folder()
    corpus_folder = os.path.join(root_folder, "data/textblock_politisk")

sys.path = [root_folder] + sys.path


v_corpus = mdw_gui.load_vectorized_corpus(corpus_folder, [1, 3])

gui = mdw_gui.display_gui(v_corpus, v_corpus.document_index)
