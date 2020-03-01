import os
import types
import ipywidgets as widgets
import pandas as pd
import text_analytic_tools.text_analysis.topic_model as topic_model
import text_analytic_tools.text_analysis.derived_data_compiler as derived_data_compiler
import text_analytic_tools.text_analysis.utility as tmutility
import westac.common.utility as utility
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = utility.setup_logger(filename=None)

# from beakerx import *
# from beakerx.object import beakerx
# beakerx.pandas_display_table()

pd.set_option("max_rows", None)
pd.set_option("max_columns", None)
pd.set_option('colheader_justify', 'left')
pd.set_option('max_colwidth', 300)

from IPython.display import display

def load_model(corpus_folder, state, model_name, model_infos=None):

    model_infos = model_infos or tmutility.find_models(corpus_folder)
    model_info = next(x for x in model_infos if x['name'] == model_name)

    m_data = topic_model.load_model(*os.path.split(model_info['folder']))
    c_data = derived_data_compiler.CompiledData.load(corpus_folder, model_info['name'])

    state.set_data(m_data, c_data)

    topics = c_data.topic_token_overview
    topics.style.set_properties(**{'text-align': 'left'})\
        .set_table_styles([ dict(selector='td', props=[('text-align', 'left')] ) ])

    display(topics)

def display_gui(corpus_folder, state):

    model_infos = tmutility.find_models(corpus_folder)
    model_names = list(x['name'] for x in model_infos)

    gui = types.SimpleNamespace(
        model_name=widgets.Dropdown(description='Model', options=model_names, layout=widgets.Layout(width='40%')),
        load=widgets.Button(description='Load', button_style='Success', layout=widgets.Layout(width='80px')),
        output=widgets.Output()
    )

    def load_handler(*args):
        gui.output.clear_output()
        try:
            gui.load.disabled = True
            with gui.output:
                if gui.model_name.value is None:
                    print("Please specify which model to load.")
                    return
                load_model(corpus_folder, state, gui.model_name.value, model_infos)
        finally:
            gui.load.disabled = False

    gui.load.on_click(load_handler)

    display(widgets.VBox([
        widgets.HBox([gui.model_name, gui.load ]),
        widgets.VBox([gui.output])
    ]))
