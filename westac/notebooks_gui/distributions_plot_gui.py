import ipywidgets
from IPython.display import display
from bokeh.plotting import show
import westac.notebooks_gui.distributions_plot as plotter

def display_gui(x_corpus, tokens, n_columns=3):

    tokens  = sorted(list(tokens))
    tokens_map = {
        token: index for index, token in enumerate(tokens)
    }
    progress  = ipywidgets.IntProgress(description='', min=0, max=10, step=1, value=0, continuous_update=False, layout=ipywidgets.Layout(width='98%'))
    n_count   = ipywidgets.IntSlider(description='Count', min=0, max=100, step=1, value=3, continuous_update=False, layout=ipywidgets.Layout(width='300px'))
    forward   = ipywidgets.Button(description=">>", button_style='Success', layout=ipywidgets.Layout(width='40px', color='green'))
    back      = ipywidgets.Button(description="<<", button_style='Success', layout=ipywidgets.Layout(width='40px', color='green'))
    split     = ipywidgets.ToggleButton(description="Split", layout=ipywidgets.Layout(width='80px', color='green'))
    output    = ipywidgets.Output(layout=ipywidgets.Layout(width='99%'))
    wtokens   = ipywidgets.SelectMultiple(options=tokens, value=[], rows=30)

    def tick(x=None, p=progress, max=10): # pylint: disable=redefined-builtin
        if p.max != max: p.max = max
        p.value = x if x is not None else p.value + 1

    def update_plot(*args): # pylint: disable=unused-argument

        output.clear_output()

        selected_tokens = wtokens.value

        if len(selected_tokens) == 0:
            selected_tokens = tokens[:n_count.value]

        indices = [ x_corpus.token2id[token] for token in selected_tokens ]

        with output:
            x_columns = n_columns if split.value else None
            p = plotter.plot_distributions(x_corpus, indices, n_columns=x_columns, tick=tick)
            show(p)

    def stepper_clicked(b):

        current_token = wtokens.value[0] if len(wtokens.value) > 0 else tokens[0]
        current_index = tokens.index(current_token)

        if b.description == "<<":
            current_index = max(current_index - n_count.value, 0)

        if b.description == ">>":
            current_index = min(current_index + n_count.value, len(tokens) - n_count.value + 1)

        wtokens.value = tokens[current_index:current_index + n_count.value]

    def split_changed(*args): # pylint: disable=unused-argument)
        update_plot()

    def token_select_changed(*args): # pylint: disable=unused-argument)
        update_plot()

    n_count.observe(update_plot, 'value')
    split.observe(split_changed, 'value')
    wtokens.observe(token_select_changed, 'value')

    forward.on_click(stepper_clicked)
    back.on_click(stepper_clicked)

    display(ipywidgets.VBox([
        progress,
        ipywidgets.HBox([
            back, forward,
            n_count, split
        ]),
        ipywidgets.HBox([wtokens, output])
    ]))
    update_plot()
