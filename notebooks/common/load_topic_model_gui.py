import types
import warnings
from os.path import join as jj

import ipywidgets as widgets

import penelope.topic_modelling as topic_modelling
import penelope.utility as utility

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = utility.setup_logger(filename=None)

from IPython.display import display


def load_model(corpus_folder, state, model_name, model_infos=None):

    model_infos = model_infos or topic_modelling.find_models(corpus_folder)
    model_info = next(x for x in model_infos if x['name'] == model_name)

    m_data = topic_modelling.load_model(model_info['folder'])
    c_data = topic_modelling.CompiledData.load(jj(corpus_folder, model_info['name']))

    state.set_data(m_data, c_data)

    topics = c_data.topic_token_overview
    topics.style.set_properties(**{'text-align': 'left'}).set_table_styles(
        [dict(selector='td', props=[('text-align', 'left')])]
    )

    display(topics)


def display_gui(corpus_folder, state):

    model_infos = topic_modelling.find_models(corpus_folder)
    model_names = list(x['name'] for x in model_infos)

    gui = types.SimpleNamespace(
        model_name=widgets.Dropdown(description='Model', options=model_names, layout=widgets.Layout(width='40%')),
        load=widgets.Button(description='Load', button_style='Success', layout=widgets.Layout(width='80px')),
        output=widgets.Output(),
    )

    def load_handler(*_):  # pylint: disable=unused-argument
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

    display(widgets.VBox([widgets.HBox([gui.model_name, gui.load]), widgets.VBox([gui.output])]))
