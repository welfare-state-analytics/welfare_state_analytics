import bokeh
import pandas as pd

import notebooks.word_trends.displayers.data_compilers as data_compilers

NAME = "Bar"

compile = data_compilers.compile_year_token_vector_data


def setup(container, **kwargs):
    pass


def plot(data, **kwargs):
    container = kwargs['container']
    container.data_source.data.update(data)
    bokeh.io.push_notebook(handle=container.handle)
