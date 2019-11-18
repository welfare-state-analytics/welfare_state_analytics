import numpy as np
import ipywidgets
from IPython.display import display
import westac.notebooks_gui.distributions_plot as pld

def display_gui(x_corpus, most_deviating, metric, columns=None):

    progress = ipywidgets.IntProgress(description='', min=0, max=10, step=1, value=0, continuous_update=False, layout=ipywidgets.Layout(width='600px'))
    n_start  = ipywidgets.IntSlider(description='Jump', min=0, max=len(most_deviating)-4, step=1, continuous_update=False, layout=ipywidgets.Layout(width='600px'))
    n_count  = ipywidgets.IntSlider(description='Count', min=0, max=10, step=1, value=4, continuous_update=False, layout=ipywidgets.Layout(width='300px'))
    forward  = ipywidgets.Button(description=">>", layout=ipywidgets.Layout(width='40px', color='green'))
    back     = ipywidgets.Button(description="<<", layout=ipywidgets.Layout(width='40px', color='green'))
    split    = ipywidgets.ToggleButton(description="Split", layout=ipywidgets.Layout(width='80px', color='green'))
    output   = ipywidgets.Output(layout=ipywidgets.Layout(width='99%'))

    x_range  = x_corpus.year_range()
    indices  = [ x_corpus.token2id[token] for token in most_deviating[metric + '_token'] ]
    xs       = np.arange(x_range[0], x_range[1] + 1, 1)

    def update_plot(*args):

        output.clear_output()

        with output:
            pld.plot_distributions(x_corpus, xs, indices, n_count=n_count.value, start=n_start.value, columns=columns)

    def on_button_clicked(b):

        if b.description == "<<":
            n_start.value = max(n_start.value - n_count.value, 0)

        if b.description == ">>":
            n_start.value = min(n_start.value + n_count.value, n_start.max - n_count.value+1)

    n_start.observe(update_plot, 'value')
    n_count.observe(update_plot, 'value')
    split.observe(update_plot, 'value')

    forward.on_click(on_button_clicked)
    back.on_click(on_button_clicked)

    display(ipywidgets.VBox([
        progress,
        ipywidgets.HBox([
            back, forward, n_start, n_count, split
        ]),
        output
    ]))
    update_plot()
