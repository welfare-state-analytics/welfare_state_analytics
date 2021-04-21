# -*- coding: utf-8 -*-
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
# ## Concept Context Co-Occurrences Analysis

# %% [markdown]
# ### Setup notebook
# %%
# %load_ext autoreload
# %autoreload 2
# pylint: disable=too-many-instance-attributes, unused-argument

import importlib
import warnings
from typing import Optional

import penelope.notebook.co_occurrence.load_co_occurrences_gui as load_gui
import penelope.notebook.co_occurrence.to_co_occurrence_gui as compute_gui
from bokeh.plotting import output_notebook
from IPython.display import display
from penelope import co_occurrence, pipeline, workflows
from penelope.notebook.interface import ComputeOpts

import __paths__  # pylint: disable=unused-import

from .loaded_callback import loaded_callback

warnings.filterwarnings("ignore", category=FutureWarning)

output_notebook()

# %% [markdown]
# ### Generate new concept context co-co_occurrences
# %%


def compute_co_occurrence_callback(
    corpus_config: pipeline.CorpusConfig,
    args: ComputeOpts,
    checkpoint_file: Optional[str] = None,
) -> co_occurrence.ComputeResult:
    compute_result = workflows.co_occurrence.compute(
        args=args,
        corpus_config=corpus_config,
        checkpoint_file=checkpoint_file,
    )
    return compute_result


importlib.reload(compute_gui)
gui: compute_gui.ComputeGUI = compute_gui.create_compute_gui(
    corpus_config="riksdagens-protokoll",
    corpus_folder=__paths__.corpus_folder,
    data_folder=__paths__.data_folder,
    done_callback=loaded_callback,
    compute_callback=compute_co_occurrence_callback,
)
display(gui.layout())
# %% [markdown]
# ### Load saved concept context co-occurrences
# %%
lgu: load_gui.LoadGUI = load_gui.create_load_gui(
    data_folder=__paths__.data_folder, filename_pattern="*.co_occurrence.csv.zip", loaded_callback=loaded_callback
)
display(lgu.layout())
