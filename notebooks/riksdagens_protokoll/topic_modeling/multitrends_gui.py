from __future__ import annotations

from collections import defaultdict

import ipywidgets as w
import pandas as pd
import penelope.notebook.mixins as mx
from IPython.display import display
from loguru import logger
from penelope import utility as pu
from penelope.notebook import topic_modelling as ntm
from penelope.notebook.grid_utility import table_widget
from penelope.plot import plot_multiple_value_series

import westac.riksprot.parlaclarin.codecs as md
import westac.riksprot.parlaclarin.speech_text as sr
import westac.riksprot.parlaclarin.utility as ru

from .mixins import RiksProtMetaDataMixIn

# pylint: disable=too-many-ancestors, no-member


class RiksprotTopicMultiTrendsGUI(RiksProtMetaDataMixIn, mx.MultiLinePivotKeysMixIn, ntm.TopicTrendsGUI):
    def __init__(
        self,
        person_codecs: md.PersonCodecs,
        speech_repository: sr.SpeechTextRepository,
        state: dict,
    ):
        super(RiksprotTopicMultiTrendsGUI, self).__init__(  # pylint: disable=super-with-arguments
            pivot_key_specs=person_codecs.property_values_specs,
            color_presets=ru.PARTY_COLOR_BY_ABBREV,
            person_codecs=person_codecs,
            speech_repository=speech_repository,
            state=state,
        )
        self._output_format.value = None
        self._output_format.options = ['Lines', 'Lines (smooth)', 'Bar', 'Table', 'xlsx', 'csv', 'clipboard', 'pandas']
        self._output_format.index = 1
        self._extra_placeholder = self.default_pivot_keys_layout(layout={'width': '180px'}, rows=8)

    def setup(self, **kwargs):
        super().setup(**kwargs)
        key_values_options: set[str] = self.pivot_keys.key_values_str(self.pivot_keys.text_names)
        self._filter_keys.options = key_values_options
        return self

    @property
    def lines_filter_opts(self) -> list[tuple[str, str, pu.PropertyValueMaskingOpts]]:
        """Returns lines' filter key/values as a name-to-values mapping."""
        lines_opts: list[tuple[str, str, pu.PropertyValueMaskingOpts]] = []
        for key, color, opts in self.lines:
            key_values = defaultdict(list)
            value_tuples: tuple[str, str] = [x.split(': ') for x in opts]
            for k, v in value_tuples:
                key_values[k].append(v)
            lines_opts.append((key, color, self.pivot_keys.create_filter_key_values_dict(key_values, decode=True)))
        return lines_opts

    def update(self) -> pd.DataFrame:
        years_range: tuple[int, int] = (self.years[0], self.years[1] + 1)
        ytw: pd.DataFrame = pd.DataFrame(data={'year': range(*years_range)}).set_index('year')
        topic_data: pd.DataFrame = (
            self.inferred_topics.calculator.reset()
            .filter_by_data_keys(topic_id=self.topic_id, year=list(range(*years_range)))
            .threshold(self.threshold)
        ).value
        # .overload(includes="gender_id,party_id,office_type_id,sub_office_type_id,person_id")
        for name, _, opts in self.lines_filter_opts:
            try:
                ytw_line: pd.DataFrame = (
                    (
                        self.inferred_topics.calculator.reset(topic_data)
                        .filter_by_keys(**opts.opts)
                        .yearly_topic_weights(
                            self.get_result_threshold(), n_top_relevance=None, topic_ids=self.topic_id
                        )
                    )
                    .value[['year', self.aggregate]]
                    .set_index('year')
                    .fillna(0)
                )
                ytw_line.columns = [name]
                ytw = ytw.merge(ytw_line[name], how='left', left_index=True, right_index=True)
                ytw[name].fillna(0, inplace=True)
            except pu.EmptyDataError as ex:
                logger.info(ex)
                ytw[name] = 0  # pylint: disable=unsupported-assignment-operation
        return ytw

    def display_handler(self, *_):
        self._output.clear_output()
        try:
            self._text.value = (
                f'ID {self.topic_id}: {self.inferred_topics.get_topic_title(self.topic_id, n_tokens=200)}'
            )
            with self._output:
                if self.yearly_topic_weights is None:
                    self.alert("😡 No data, please change filters..")
                elif self.output_format.lower() in ('xlsx', 'csv', 'clipboard'):
                    pu.ts_store(
                        data=self.yearly_topic_weights, extension=self.output_format, basename='heatmap_weights'
                    )
                elif self.output_format.lower() == "pandas":
                    with pd.option_context("display.max_rows", None, "display.max_columns", None):
                        display(self.yearly_topic_weights)
                elif self.output_format.lower() == "table":
                    g = table_widget(self.yearly_topic_weights, handler=self.click_handler)
                    display(g)
                else:
                    colors: list[str] = [color for _, color, _ in self.lines]
                    plot_multiple_value_series(
                        kind=self.output_format.lower(),
                        data=self.yearly_topic_weights.reset_index(),
                        category_name='year',
                        columns=None,
                        smooth=('smooth' in self.output_format.lower()),
                        fig_opts={'title': f'Topic {self.topic_label(self.topic_id)} prevalence'},
                        colors=colors,
                    )
            # self.alert("✅")
        except Exception as ex:
            self.warn(f"😡 {ex}")

    def observe(
        self, value: bool, **kwargs
    ) -> "RiksprotTopicMultiTrendsGUI":  # pylint: disable=unused-argument, arguments-differ
        """user must press update"""
        return self

    def topic_changed(self, *_):
        ...

    def layout(self):
        self._lines.layout.width = '135px'
        self._add_line.layout.width = '40px'
        self._del_line.layout.width = '40px'
        self._add_line.style.button_color = 'lightgreen'
        self._del_line.style.button_color = 'lightgreen'
        self._compute.style.button_color = 'lightgreen'
        self._compute.button_style = ''
        self._compute.layout.width = '80px'
        self._auto_compute.layout.width = '80px'
        self._output_format.layout.width = '180px'
        self._aggregate.layout.width = '180px'

        box_layout: w.Layout = w.Layout(display='flex', flex_flow='column', align_items='stretch', width='98%')

        return w.VBox(
            [
                w.HBox(
                    ([self._extra_placeholder] if self._extra_placeholder is not None else [])
                    + [
                        w.VBox(
                            [
                                self._year_range_label,
                                self._year_range,
                                self._threshold_label,
                                self._threshold,
                            ]
                        ),
                    ]
                    + [
                        w.VBox(
                            [
                                w.HTML("<b>Aggregate</b>"),
                                self._aggregate,
                                w.HTML("<b>Output</b>"),
                                self._output_format,
                            ]
                        ),
                    ]
                    + [
                        w.VBox(
                            [
                                w.HTML("<b>Topic</b>"),
                                w.HBox([self._next_prev_layout]),
                                w.HTML("<br>"),
                                w.HBox([self._compute, self._auto_compute]),
                                self._alert,
                            ]
                        ),
                    ]
                ),
                w.HBox([self._output], layout=box_layout),
                w.HBox([self._text] + ([self._content_placeholder] if self._content_placeholder is not None else [])),
            ],
            layout=box_layout,
        )
