import bokeh

from . import data_compilers

NAME = "Bar"

compile = data_compilers.compile_year_token_vector_data  # pylint: disable=redefined-builtin


def setup(container, **kwargs):  # pylint: disable=unused-argument
    pass


def plot(data, **kwargs):
    container = kwargs['container']
    container.data_source.data.update(data)
    bokeh.io.push_notebook(handle=container.handle)
