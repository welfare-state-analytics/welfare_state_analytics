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
#
# References:
#
#    King, Gary, Patrick Lam, and Margaret Roberts. "Computer-Assisted Keyword
#    and Document Set Discovery from Unstructured Text." (2014).
#    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.458.1445&rep=rep1&type=pdf
#

# %%

from IPython.display import display
from penelope.notebook.mdw import main_gui

import __paths__

gui = main_gui.create_main_gui(corpus_folder=__paths__.data_folder)

display(gui)
