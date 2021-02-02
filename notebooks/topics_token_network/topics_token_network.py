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

# %%

# %load_ext autoreload
# %autoreload 2

import __paths__
from IPython.display import display
import notebooks.topics_token_network.topics_token_network_gui as ttn_gui

data_folder = '/data/westac/sou_kb_labb'

gui = ttn_gui.create_gui(data_folder=data_folder)

display(gui.layout())

# %%
ttn_gui.TOPIC_TOKENS.to_csv(sep=';')
# %%
ttn_gui.SOURCE_NETWORK_DATA
# %%
