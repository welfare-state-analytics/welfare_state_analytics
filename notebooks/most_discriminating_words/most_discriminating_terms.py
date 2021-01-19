# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Most Discriminating Terms

# %%
# %load_ext autoreload
# %autoreload 2

# pylint: disable=wrong-import-position
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Most Discriminating Terms

# %%
# %load_ext autoreload
# %autoreload 2

# pylint: disable=wrong-import-position,unused-argument

from IPython.display import display
from ipywidgets import Output, VBox
from penelope.corpus import dtm
from penelope.notebook import ipyaggrid_utility
from penelope.notebook.dtm import load_dtm_gui
from penelope.notebook.mdw import create_mdw_gui

import __paths__

view_display, view_gui = Output(), Output()

@view_display.capture(clear_output=True)
def display_mdw(corpus: dtm.VectorizedCorpus, df_mdw):
    g = ipyaggrid_utility.display_grid(df_mdw)
    display(g)


@view_gui.capture(clear_output=True)
def loaded_callback(corpus: dtm.VectorizedCorpus, corpus_folder: str, corpus_tag: str):
    mdw_gui = create_mdw_gui(corpus, done_callback=display_mdw)
    display(mdw_gui.layout())


gui = load_dtm_gui.create_load_gui(corpus_folder=__paths__.data_folder, loaded_callback=loaded_callback)

display(gui.layout())
display(VBox([view_gui, view_display]))

# %%
