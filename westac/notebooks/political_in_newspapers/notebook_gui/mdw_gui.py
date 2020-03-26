import types
import logging
import ipywidgets
import westac.notebooks.political_in_newspapers.corpus_data as corpus_data
import westac.common.utility as utility

from westac.common.textacy_most_discriminating_terms import compute_most_discriminating_terms
from IPython.display import display

logger = logging.getLogger(__name__)

def group_indicies(documents, period, pub_ids=None):

    assert 'year' in documents.columns

    docs = documents[documents.year.between(*period)]

    if isinstance(pub_ids, int):
        pub_ids = list(pub_ids)

    if len(pub_ids or []) > 0:
        docs = docs[(docs.publication_id.isin(pub_ids))]

    return docs.index

def display_gui(x_corpus, x_documents):

    publications = utility.extend(dict(corpus_data.PUBLICATION2ID), {'(ALLA)': None})

    lw = lambda w: ipywidgets.Layout(width=w)
    year_span = (x_documents.year.min(), x_documents.year.max())
    gui = types.SimpleNamespace(
        progress=ipywidgets.IntProgress(value=0, min=0, max=5, step=1, description='', layout=lw('90%')),
        top_n_terms=ipywidgets.IntSlider(description='#terms', min=10, max=1000, value=100, tooltip='The total number of most discriminating terms to return for each group'),
        max_n_terms=ipywidgets.IntSlider(description='#top', min=1, max=2000, value=2000, tooltip='Only consider terms whose document frequency is within the top # terms out of all terms'),
        period1=ipywidgets.IntRangeSlider(description='Period', min=year_span[0], max=year_span[1], value=(x_documents.year.min(), x_documents.year.min()+4), layout=lw('250px')),
        period2=ipywidgets.IntRangeSlider(description='Period', min=year_span[0], max=year_span[1], value=(x_documents.year.max()-4, x_documents.year.max()), layout=lw('250px')),
        publication_id1=ipywidgets.Dropdown(description='Publication', options=publications, value=None, layout=ipywidgets.Layout(width="200px")),
        publication_id2=ipywidgets.Dropdown(description='Publication', options=publications, value=None, layout=ipywidgets.Layout(width="200px")),
        compute=ipywidgets.Button(description='Compute', icon='', button_style='Success', layout=lw('120px')),
        output=ipywidgets.Output(layout={'border': '1px solid black'})
    )

    boxes = ipywidgets.VBox([
        ipywidgets.HBox([
            ipywidgets.VBox([ gui.period1, gui.publication_id1 ]),
            ipywidgets.VBox([ gui.period2, gui.publication_id2 ]),
            ipywidgets.VBox([
                gui.top_n_terms,
                gui.max_n_terms,
            ], layout=ipywidgets.Layout(align_items='flex-end')),
            ipywidgets.VBox([ gui.compute]),
        ]),
        gui.output
    ])

    display(boxes)

    def compute_callback_handler(*_args):
        gui.output.clear_output()
        with gui.output:
            try:
                gui.compute.disabled = True
                df = compute_most_discriminating_terms(
                    x_corpus,
                    top_n_terms=gui.top_n_terms.value,
                    max_n_terms=gui.max_n_terms.value,
                    group1_indices=group_indicies(x_documents, gui.period1.value, gui.publication_id1.value),
                    group2_indices=group_indicies(x_documents, gui.period2.value, gui.publication_id2.value)
                )
                if df is not None:
                    display(df)
                else:
                    logger.info('No data for selected groups or periods.')

            except Exception as ex:
                logger.error(ex)
            finally:
                gui.compute.disabled = False

    gui.compute.on_click(compute_callback_handler)
    return gui
