import glob
import os
from dataclasses import dataclass, field
from typing import Any, Callable, List

import ipycytoscape
import ipywidgets as widgets
import pandas as pd
from IPython.display import display
from penelope import topic_modelling
from penelope.notebook.ipyaggrid_utility import display_grid
from penelope.utility.filename_fields import FilenameFieldSpecs
from bokeh.palettes import Category20
from itertools import cycle

view = widgets.Output()

# TODO: #125 Add topic - token network feature

# class CustomNode(ipycytoscape.Node):
#    def __init__(self, name, classes=''):
#        super().__init__()
#        self.data['id'] = name
#        self.data['name'] = name
#        self.classes = classes
#
#    def __str__(self):
#        return self.name


# def df_to_nx_edges_list(
#    df: pd.DataFrame, source: str = 'source', target: str = 'target', **edge_attributes
# ) -> List[Tuple[Any, Any, Dict]]:
#    attr_fields = list(edge_attributes.values())
#    attr_names = {v: k for k, v in edge_attributes.items()}
#
#    edges = zip(
#        df[source].values,
#        df[target].values,
#        df[attr_fields].apply(lambda x: {attr_names[k]: v for k, v in x.to_dict().items()}, axis=1),
#    )
#    return list(edges)


# def create_nx_graph(df: pd.DataFrame, source: str = 'source', target: str = 'target', weight='weight') -> nx.Graph:
#    G = nx.Graph()
#    source_nodes = [CustomNode(name=str(i), classes='source_class') for i in set(df[source].values)]
#    target_nodes = [CustomNode(name=x, classes='target_class') for x in set(df[target].values)]
#    G.add_nodes_from(source_nodes, bipartite=0)
#    G.add_nodes_from(target_nodes, bipartite=1)
#    edges = df_to_nx_edges_list(df, source=source, target=target, weight=weight)
#    G.add_edges_from(edges)
#    return G


def css_styles(topic_ids: List[int]) -> dict:
    styles = [
        {
            'selector': 'node.target_class',
            'css': {
                'content': 'data(label)',
                'text-valign': 'center',
                "width": 1,
                "height": 1,
                "opacity": 0.8,
            },
        },
        {
            'selector': 'edge',
            'style': {
                'width': 4,
                #'target-arrow-shape': 'triangle',
                #'target-arrow-color': '#9dbaea',
                'curve-style': 'bezier',
                "opacity": 0.8,
            },
        },
    ]
    colors = cycle(
        [
            '#F05000',
            '#F000AD',
            '#00A371',
            '#0039F5',
            '#A38101',
            '#F01B01',
            '#94A301',
            '#AC00F0',
            '#00A321',
            '#008BF5',
        ]
        + list(Category20[20])
    )

    for topic_id in topic_ids:
        color = next(colors)
        styles.extend(
            [
                {
                    'selector': f'node.node_source_class_{topic_id}',
                    'css': {
                        'background-color': color,
                        'content': 'data(id)',
                        'text-valign': 'center',
                    },
                },
                {
                    "selector": f"edge.edge_class_{topic_id}",
                    "style": {
                        "width": 1,
                        "opacity": 1.0,
                        "line-color": color,
                    },
                },
            ]
        )
    return styles


def to_dict(topics_tokens: pd.DataFrame) -> dict:

    unique_topics = topics_tokens.groupby(['topic', 'topic_id']).size().reset_index()[['topic', 'topic_id']]

    source_network_data = {
        'nodes': [
            {"data": {"id": node['topic'], "label": node['topic']}, 'classes': f"node_source_class_{node['topic_id']}"}
            for node in unique_topics.to_dict('records')
        ]
        + [{"data": {"id": w, "label": w}, 'classes': 'target_class'} for w in topics_tokens['token'].unique()],
        'edges': [
            {
                "data": {
                    "id": f"{edge['topic_id']}_{edge['token']}",
                    "source": edge['topic'],
                    "target": edge['token'],
                    "weight": int(edge['weight'] * 100000),
                },
                "classes": f"edge_class_{edge['topic_id']}",
            }
            for edge in topics_tokens[['topic', 'token', 'topic_id', 'weight']].to_dict('records')
        ],
    }
    return source_network_data


def create_network(topics_tokens: pd.DataFrame) -> ipycytoscape.CytoscapeWidget:
    source_network_data = to_dict(topics_tokens=topics_tokens)
    w = ipycytoscape.CytoscapeWidget(layout={'height': '1000px', 'background-color': 'black'})
    w.graph.add_graph_from_json(source_network_data)
    w.set_layout(name='cola')
    w.set_style(css_styles(topics_tokens.topic_id.unique()))
    return w


@dataclass
class ViewModel:

    # FIXME: This must be bundled and stored in the InferredTopicsData data!!!
    filename_fields: FilenameFieldSpecs = None

    _top_count: int = field(init=False, default=50)
    _inferred_topics_data: topic_modelling.InferredTopicsData = field(init=False, default=None)
    _largest_topic_token_weights: pd.DataFrame = field(init=False, default=None)

    @property
    def inferred_topics_data(self) -> topic_modelling.InferredTopicsData:
        return self._inferred_topics_data

    @inferred_topics_data.setter
    def inferred_topics_data(self, value: topic_modelling.InferredTopicsData):
        self._inferred_topics_data = value
        self.update()

    @property
    def largest_topic_token_weights(self) -> pd.DataFrame:
        return self._largest_topic_token_weights

    @property
    def top_count(self) -> int:
        return self._top_count

    @top_count.setter
    def top_count(self, value: int):
        self._top_count = value
        self.update()

    def update(self) -> None:
        if self._inferred_topics_data is None:
            return
        self._largest_topic_token_weights = self._inferred_topics_data.n_largest_topic_token_weights(self.top_count)

    def get_topics_tokens(self, topic_ids: List[int]) -> pd.DataFrame:
        topics_tokens = self.largest_topic_token_weights[
            self.largest_topic_token_weights.index.isin(topic_ids)
        ].reset_index()
        topics_tokens['topic'] = topics_tokens.topic_id.apply(lambda x: f"Topic #{x}")
        return topics_tokens[['topic', 'token', 'weight', 'topic_id']]


def find_inferred_models(folder: str) -> List[str]:
    """Return YAML filenames in `folder`"""
    filenames = glob.glob(os.path.join(folder, "**/*document_topic_weights.zip"), recursive=True)
    folders = [os.path.split(filename)[0] for filename in filenames]
    return folders


@view.capture(clear_output=False)
def loader(folder: str, filename_fields: Any = None) -> topic_modelling.InferredTopicsData:
    data = topic_modelling.InferredTopicsData.load(folder=folder, filename_fields=filename_fields)
    return data


W = None


@view.capture(clear_output=True)
def displayer(opts: "GUI") -> None:
    global W
    if opts.model.largest_topic_token_weights is None:
        return

    opts.model.top_count = opts.top_tokens_count

    topics_tokens = opts.model.get_topics_tokens(opts.topics_ids)

    if opts.output_format == "network":
        network = create_network(topics_tokens)
        W = network
        display(network)
        return

    topics_tokens = topics_tokens[['topic', 'token', 'weight']]

    if opts.output_format == "table":
        g = display_grid(topics_tokens)
        display(g)

    if opts.output_format == "gephi":
        topics_tokens.columns = ['Source', 'Target', 'Weight']
        g = display_grid(topics_tokens)
        display(g)


@dataclass
class GUI:

    _source_folder: widgets.Dropdown = widgets.Dropdown(layout={'width': '200px'})
    _topic_ids: widgets.SelectMultiple = widgets.SelectMultiple(
        description="", options=[], value=[], rows=8, layout={'width': '100px'}
    )
    _top_tokens_count: widgets.IntSlider = widgets.IntSlider(
        description='', min=3, max=500, value=50, layout={'width': '200px'}
    )
    _label: widgets.HTML = widgets.HTML(value='&nbsp;', layout={'width': '200px'})
    _output_format = widgets.Dropdown(
        description='', options=['network', 'table', 'gephi'], value='network', layout={'width': '200px'}
    )
    _network_layout = widgets.Dropdown(
        description='', options=['cola', 'euler', 'klay', 'spread', ], value='cola', layout={'width': '200px'}
    )
    _button = widgets.Button(
        description="Display", button_style='Success', layout=widgets.Layout(width='115px', background_color='blue')
    )
    model: ViewModel = None

    loader: Callable[[str], topic_modelling.InferredTopicsData] = None
    displayer: Callable[["GUI"], None] = None

    @view.capture(clear_output=False)
    def _displayer(self, *_):
        if not self.displayer:
            return
        self.alert('Computing...')
        self.displayer(self)
        self.alert('')

    @view.capture(clear_output=False)
    def _load_handler(self, *_):

        if self.loader is None:
            return

        self.alert('Loading...')
        self.lock(True)
        self.model.inferred_topics_data = self.loader(
            self._source_folder.value, filename_fields=self.model.filename_fields
        )
        self._topic_ids.value = []
        self._topic_ids.options = [
            ("Topic #" + str(i), i) for i in range(0, self.model.inferred_topics_data.num_topics)
        ]

        # self._displayer()

        self.lock(False)
        self.alert('')

    def _update_top_count_handler(self, *_):
        self.model.top_count = self._top_tokens_count.value

    def lock(self, value: bool = True) -> None:
        self._source_folder.disabled = value
        self._topic_ids.disabled = value
        self._button.disabled = value
        if value:
            self._source_folder.unobserve(self._load_handler, names='value')
        else:
            self._source_folder.observe(self._load_handler, names='value')

    def alert(self, msg: str = '&nbsp;') -> None:
        self._label.value = msg or '&nbsp;'

    def setup(
        self,
        folders: str,
        loader: Callable[[str], topic_modelling.InferredTopicsData],
        displayer: Callable[["GUI"], None],
    ) -> "GUI":
        self.loader = loader
        self.displayer = displayer
        self._source_folder.options = {os.path.split(folder)[1]: folder for folder in folders}
        self._source_folder.value = None
        self._top_tokens_count.value = self.model.top_count
        self._top_tokens_count.observe(self._update_top_count_handler, names='value')
        self._source_folder.observe(self._load_handler, names='value')
        self._button.on_click(self._displayer)
        return self

    def layout(self):
        return widgets.VBox(
            [
                widgets.HBox(
                    [
                        widgets.VBox(
                            [
                                widgets.HTML("<b>Model</b>"),
                                self._source_folder,
                                widgets.HTML("<b>Output</b>"),
                                self._output_format,
                                widgets.HTML("<b>Top tokens</b>"),
                                self._top_tokens_count,
                            ]
                        ),
                        widgets.VBox(
                            [
                                widgets.HTML("<b>Topics</b>"),
                                self._topic_ids,
                            ]
                        ),
                        widgets.VBox(
                            [
                                self._label,
                                self._button,
                            ]
                        ),
                    ]
                ),
                view,
            ]
        )

    @property
    def source_folder(self) -> str:
        return self._source_folder.value

    @property
    def topics_ids(self) -> List[int]:
        return self._topic_ids.value

    @property
    def output_format(self) -> List[int]:
        return self._output_format.value

    @property
    def top_tokens_count(self) -> List[int]:
        return self._top_tokens_count.value


def create_gui(data_folder: str):
    gui = GUI(model=ViewModel(filename_fields=['year:_:1', 'sequence_id:_:2'])).setup(
        folders=find_inferred_models(data_folder),
        loader=loader,
        displayer=displayer,
    )
    return gui
