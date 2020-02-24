import os
import types
import ipywidgets as widgets
import text_analytic_tools.text_analysis.topic_model as topic_model
import text_analytic_tools.text_analysis.topic_model_utility as topic_model_utility
import westac.notebooks.political_in_newspapers.corpus_data as corpus_data

from IPython.display import display

def load_model(corpus_folder, state, model_name, model_infos=None, documents=None):

    model_infos = model_infos or topic_model_utility.find_models(corpus_folder)
    model_info = next(x for x in model_infos if x['name'] == model_name)
    model_data = topic_model.load_model(*os.path.split(model_info['folder']))
    compiled_data = topic_model_utility.load_compiled_data(corpus_folder, model_info['name'])

    df = compiled_data.document_topic_weights
    if 'year' not in df.columns and documents is not None:
        compiled_data.document_topic_weights = corpus_data.extend_with_document_info(df, documents)

    state.set_data(model_data, compiled_data)

    # topics = topic_model_utility.get_lda_topics(state.topic_model, n_tokens=20)
    topics = compiled_data.topic_token_overview
    display(topics)

def display_gui(corpus_folder, state):

    model_infos = topic_model_utility.find_models(corpus_folder)
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
