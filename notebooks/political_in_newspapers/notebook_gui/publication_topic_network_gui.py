from typing import Any, Mapping, Sequence

import bokeh
import bokeh.plotting
import networkx as nx
import numpy as np
import pandas as pd
from IPython.display import display
from ipywidgets import HTML, Dropdown, FloatSlider, HBox, IntProgress, IntRangeSlider, Output, SelectMultiple, VBox
from penelope import topic_modelling, utility
from penelope.network import plot_utility
from penelope.network.bipartite_plot import plot_bipartite_network
from penelope.network.networkx import utility as network_utility
from penelope.notebook import widgets_utils
from penelope.notebook.topic_modelling import TopicModelContainer

from notebooks.political_in_newspapers import repository

TEXT_ID = 'nx_pub_topic'
LAYOUT_OPTIONS = ['Circular', 'Kamada-Kawai', 'Fruchterman-Reingold']


# pylint: disable=too-many-locals, too-many-arguments, too-many-instance-attributes


def display_document_topic_network(
    *,
    document_topic_weights: pd.DataFrame,
    topic_token_weights: pd.DataFrame,
    document_threshold: float = 0.0,
    mean_threshold: float = 0.10,
    period: Sequence[int] = None,
    ignores: Sequence[int] = None,
    scale: float = 1.0,
    aggregate: str = 'mean',
    output_format: str = 'network',
    layout_algorithm: str,
    tick=utility.noop,
):

    tick(1)

    weights = compute_weights(document_topic_weights, document_threshold, mean_threshold, period, ignores, aggregate)

    tick()

    if len(weights) == 0:

        print('No data! Please change selection.')
        return

    if output_format == 'network':

        titles: pd.DataFrame = topic_modelling.get_topic_titles(topic_token_weights=topic_token_weights)
        display_network(weights, layout_algorithm, scale, titles)

    else:

        display_tabular(weights, output_format)

    tick(0)


def display_tabular(weights: pd.DataFrame, output_format):

    weights = weights[['publication', 'topic_id', 'weight', 'mean', 'max']]
    weights.columns = ['Source', 'Target', 'weight', 'mean', 'max']

    if output_format == 'excel':
        filename: str = utility.timestamp("{}_publication_topic_network.xlsx")
        weights.to_excel(filename)

    if output_format == 'CSV':
        filename: str = utility.timestamp("{}_publication_topic_network.csv")
        weights.to_csv(filename, sep='\t')

    display(weights)


def display_network(weights: pd.DataFrame, layout_algorithm: str, scale: float, titles: pd.DataFrame) -> None:
    network: nx.Graph = network_utility.create_bipartite_network(
        weights[['publication', 'topic_id', 'weight']], 'publication', 'topic_id'
    )
    args: Mapping[str, Any] = plot_utility.layout_args(layout_algorithm, network, scale)
    layout: network_utility.NodesLayout = (plot_utility.layout_algorithms[layout_algorithm])(network, **args)
    p = plot_bipartite_network(network, layout, scale=scale, titles=titles, element_id=TEXT_ID)
    bokeh.plotting.show(p)


def compute_weights(
    document_topic_weights: pd.DataFrame,
    document_threshold: float = 0.0,
    mean_threshold: float = 0.10,
    period: Sequence[int] = None,
    ignores: Sequence[int] = None,
    aggregate: str = 'mean',
):
    period: Sequence[int] = period or []

    weights: pd.DataFrame = document_topic_weights

    if len(period or []) == 2:
        weights = weights[(weights.year >= period[0]) & (weights.year <= period[1])]

    if len(ignores or []) > 0:
        weights = weights[~weights.topic_id.isin(ignores)]

    weights = weights[(weights['weight'] >= document_threshold)]  # type: ignore

    weights = weights.groupby(['publication_id', 'topic_id']).agg([np.mean, np.max])['weight'].reset_index()
    weights.columns = ['publication_id', 'topic_id', 'mean', 'max']

    weights = weights[(weights[aggregate] > mean_threshold)].reset_index()

    if len(weights) == 0:
        return weights

    weights[aggregate] = utility.clamp_values(list(weights[aggregate]), (0.1, 1.0))  # type: ignore

    weights['publication'] = weights.publication_id.apply(lambda x: repository.ID2PUBLICATION[x])
    weights['weight'] = weights[aggregate]

    return weights


class PublicationTopicNetworkGUI:
    def __init__(self, state: TopicModelContainer):

        self.state = state

        year_min, year_max = state.inferred_topics.year_period

        n_topics = state.num_topics

        self.text: HTML = widgets_utils.text_widget(TEXT_ID)
        self.period = IntRangeSlider(
            description='Time', min=year_min, max=year_max, step=1, value=(year_min, year_max), continues_update=False
        )
        self.scale = FloatSlider(description='Scale', min=0.0, max=1.0, step=0.01, value=0.1, continues_update=False)
        self.document_threshold = FloatSlider(
            description='Threshold(D)', min=0.0, max=1.0, step=0.01, value=0.00, continues_update=False
        )
        self.mean_threshold = FloatSlider(
            description='Threshold(G)', min=0.0, max=1.0, step=0.01, value=0.10, continues_update=False
        )
        self.aggregate: Dropdown = Dropdown(
            description='Aggregate', options=['mean', 'max'], value='mean', layout=dict(width="200px")
        )
        self.output_format: Dropdown = Dropdown(
            description='Output',
            options={'Network': 'network', 'Table': 'table', 'Excel': 'excel', 'CSV': 'csv'},
            value='network',
            layout=dict(width='200px'),
        )
        self.layout_network: Dropdown = Dropdown(
            description='dict', options=LAYOUT_OPTIONS, value='Fruchterman-Reingold', layout=dict(width='250px')
        )
        self.progress = IntProgress(min=0, max=4, step=1, value=0, layout=dict(width="99%"))
        self.ignores = SelectMultiple(
            description='Ignore',
            options=[('', None)] + [(f'Topic #{i}', i) for i in range(0, n_topics)],  # type: ignore
            value=[],
            rows=8,
            layout=dict(width='240px'),
        )
        self.output: Output = Output()

    def setup(self) -> "PublicationTopicNetworkGUI":
        self.layout_network.observe(self.compute_handler, names='value')
        self.document_threshold.observe(self.compute_handler, names='value')
        self.period.observe(self.compute_handler, names='value')
        self.scale.observe(self.compute_handler, names='value')
        self.mean_threshold.observe(self.compute_handler, names='value')
        self.ignores.observe(self.compute_handler, names='value')
        self.aggregate.observe(self.compute_handler, names='value')
        self.output_format.observe(self.compute_handler, names='value')
        return self

    def compute_handler(self, *_):

        self.output.clear_output()
        self.tick(1)

        with self.output:

            display_document_topic_network(
                document_topic_weights=self.state.inferred_topics.document_topic_weights,
                topic_token_weights=self.state.inferred_topics.topic_token_weights,
                document_threshold=self.document_threshold.value,
                period=self.period.value,
                mean_threshold=self.mean_threshold.value,
                ignores=self.ignores.value,
                aggregate=self.aggregate.value,
                output_format=self.output_format.value,
                scale=self.scale.value,
                layout_algorithm=self.layout_network.value,
                tick=self.tick,
            )

        self.tick(0)

    def tick(self, x=None):
        self.progress.value = self.progress.value + 1 if x is None else x

    def layout(self) -> VBox:
        return VBox(
            [
                HBox(
                    [
                        VBox(
                            [self.layout_network, self.document_threshold, self.mean_threshold, self.scale, self.period]
                        ),
                        VBox([self.ignores]),
                        VBox([self.output_format, self.progress]),
                    ]
                ),
                self.output,
                self.text,
            ]
        )

    def compute_opts(self) -> dict:
        return dict(
            state=self.state,
            document_threshold=self.document_threshold.value,
            period=self.period.value,
            mean_threshold=self.mean_threshold.value,
            ignores=self.ignores.value,
            aggregate=self.aggregate.value,
            output_format=self.output_format.value,
            layout_algorithm=self.layout_network.value,
            scale=self.scale.value,
        )


def display_gui(state: TopicModelContainer):
    gui = PublicationTopicNetworkGUI(state).setup()
    display(gui.layout())
    gui.compute_handler()
