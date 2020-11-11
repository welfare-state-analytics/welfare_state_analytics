import pandas as pd
from penelope.notebook.utility import OutputsTabExt

from notebooks.word_trends.gui_callback import State, build_layout


def xtest_loaded_callback():
    pass


def test_build_layout():

    # FIXME: Use mocks/patches!
    state = State(
        compute_options={},
        corpus=None,
        corpus_folder="",
        corpus_tag="",
        goodness_of_fit=pd.DataFrame(
            data={
                k: []
                for k in [
                    "token",
                    "word_count",
                    "l2_norm",
                    "slope",
                    "chi2_stats",
                    "earth_mover",
                    "kld",
                    "skew",
                    "entropy",
                ]
            }
        ),
        most_deviating=pd.DataFrame(data={'l2_norm_token': [], 'l2_norm': [], 'abs_l2_norm': []}),
        most_deviating_overview=pd.DataFrame(data={'l2_norm_token': [], 'l2_norm': [], 'abs_l2_norm': []}),
    )

    w: OutputsTabExt = build_layout(state=state)

    assert w is not None and isinstance(w, OutputsTabExt)
    assert 2 == len(w.children)
    # assert 4 == len(w.children[1].children)
