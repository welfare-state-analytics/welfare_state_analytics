from __future__ import annotations

from dataclasses import dataclass
from typing import List

import ipywidgets as w
import pandas as pd
from ipywidgets import HTML
from penelope import corpus as pc
from penelope.common.keyness import KeynessMetric
from penelope.notebook import mixins as mx
from penelope.notebook import utility as nu
from penelope.notebook import word_trends as wt
from westac.riksprot.parlaclarin import metadata as md

view = w.Output(layout={'border': '2px solid green'})

CLEAR_OUTPUT = False

TEMPORAL_GROUP_BY = ['decade', 'lustrum', 'year']

# pylint: disable=no-member, too-many-instance-attributes, useless-super-delegation


@dataclass
class ComputeOpts(wt.TrendsComputeOpts):

    source_folder: str = None

    def invalidates_corpus(self, other: "ComputeOpts") -> bool:
        if super().invalidates_corpus(other):
            return True
        if self.source_folder != other.source_folder:
            return True
        return False

    @property
    def clone(self) -> "ComputeOpts":
        obj: ComputeOpts = super(ComputeOpts, self).clone  # pylint: disable=super-with-arguments
        obj.source_folder = self.source_folder
        return obj


class TrendsData(wt.TrendsData):
    def __init__(self, corpus: pc.VectorizedCorpus, riksproto_metadata: md.ProtoMetaData, n_top: int = 100000):
        super().__init__(corpus, n_top=n_top)
        self.riksproto_metadata: md.ProtoMetaData = riksproto_metadata
        self._compute_opts: ComputeOpts = ComputeOpts(
            normalize=False,
            keyness=KeynessMetric.TF,
            temporal_key='decade',
            top_count=None,
            words=None,
        )

    def _transform_corpus(self, opts: ComputeOpts) -> pc.VectorizedCorpus:
        corpus: pc.VectorizedCorpus = super()._transform_corpus(opts)
        di: pd.DataFrame = self.update_document_index(opts, corpus.document_index)
        corpus.replace_document_index(di)
        return corpus

    def update_document_index(self, opts: ComputeOpts, document_index: pd.DataFrame) -> pd.DataFrame:
        if not opts.pivot_keys_id_names:
            return document_index
        di: pd.DataFrame = self.riksproto_metadata.decode_members_data(document_index, drop=False)
        pivot_keys_text_names = self.riksproto_metadata.map_id2text_names(opts.pivot_keys_id_names)
        di['document_name'] = di[pivot_keys_text_names].apply(lambda x: '_'.join(x).lower(), axis=1)
        di['filename'] = di.document_name
        di['time_period'] = di[opts.temporal_key]
        return di


class RiksProtTrendsGUI(wt.TrendsGUI):
    def __init__(
        self,
        *,
        riksprot_metadata: md.ProtoMetaData,
        pivot_key_specs: mx.PivotKeySpecArg = None,
        default_folder: str,
        n_top_count: int = 1000,
        encoded: int = True,
        defaults: dict = None,
    ):
        if pivot_key_specs is None:
            pivot_key_specs = riksprot_metadata.member_property_specs if riksprot_metadata is not None else {}

        super().__init__(pivot_key_specs=riksprot_metadata.member_property_specs, n_top_count=n_top_count)

        self.riksprot_metadata: md.ProtoMetaData = riksprot_metadata
        self.data: pd.DataFrame = None

        self.default_folder: str = default_folder
        self.defaults: dict = defaults or {}
        self.encoded: bool = encoded
        # TODO: Used in compute of trends data, shuld be user settable
        self.n_top: int = 25000
        self._source_folder: nu.FileChooserExt2 = nu.FileChooserExt2(
            path=self.default_folder,
            title='<b>Corpus folder</b>',
            show_hidden=False,
            select_default=True,
            use_dir_icons=True,
            show_only_dirs=True,
        )

    def setup(self, **kwargs) -> RiksProtTrendsGUI:
        super().setup(displayers=kwargs.get('displayers', wt.DEFAULT_WORD_TREND_DISPLAYERS))
        self._source_folder.refresh()
        self._source_folder.register_callback(self._load)
        return self

    def plot(self, trends_data: TrendsData = None):
        return super().plot(trends_data)

    @view.capture(clear_output=CLEAR_OUTPUT)
    def load(self, compute: bool = True) -> RiksProtTrendsGUI:
        self.observe(False)
        try:
            self.alert("😐 Loading DTM...")
            corpus: pc.VectorizedCorpus = self.load_corpus(overload=True)
            self.trends_data: TrendsData = TrendsData(
                corpus=corpus, riksproto_metadata=self.riksprot_metadata, n_top=self.n_top
            )
            if compute:
                self.transform()
            self.alert("✅")
        except FileNotFoundError:
            self.alert("No (unique) DTM corpus found in folder")
        except Exception as ex:
            self.alert(str(ex))
            raise
        finally:
            self.observe(True)
        return self

    # def plot_current_displayer(self, data: List[pd.DataFrame] = None):
    #     self.current_displayer.clear()
    #     with self.current_displayer.output:
    #         for d in data:
    #             ipydisplay(d)

    def load_corpus(self, overload: bool = False) -> pc.VectorizedCorpus:

        folder: str = self.source_folder
        tags: List[str] = pc.VectorizedCorpus.find_tags(folder=folder)

        if len(tags) != 1:
            raise FileNotFoundError("No (unique) DTM corpus found in folder")

        corpus: pc.VectorizedCorpus = pc.VectorizedCorpus.load(folder=folder, tag=tags[0])
        if overload:
            if self.riksprot_metadata is None:
                raise ValueError("unable to overload, riksprot metadata not set!")
            corpus = self.assign_metadata(corpus, self.riksprot_metadata)
        return corpus

    def assign_metadata(self, corpus: pc.VectorizedCorpus, metadata: md.ProtoMetaData) -> pc.VectorizedCorpus:
        document_index: pd.DataFrame = metadata.overload_by_member_data(
            corpus.document_index, encoded=self.encoded, drop=True
        )
        document_index['lustrum'] = document_index.year - document_index.year % 5
        document_index['decade'] = document_index.year - document_index.year % 10
        document_index['tokens'] = document_index.n_raw_tokens
        corpus.replace_document_index(document_index)
        return corpus

    def _load(self, *_):
        self.load()

    @property
    def source_folder(self) -> str:
        return self._source_folder.selected_path

    @property
    def options(self) -> ComputeOpts:
        opts: ComputeOpts = ComputeOpts(
            **super().options.__dict__,
            **{
                'source_folder': self.source_folder,
            },
        )
        return opts

    def layout(self) -> w.HBox:
        self._filter_keys.layout = {'width': '180px'}
        self._pivot_keys_text_names.layout = {'width': '180px'}
        if self.pivot_keys.has_pivot_keys:

            # tab = Tab(children=[self._pivot_keys_text_names, self._filter_keys])
            # tab.titles = ["Pivot by", "Filter by"]
            self._picker.rows = 15
            self._sidebar_ctrls = (
                # [tab]
                [HTML("<b>Pivot by</b>"), self._pivot_keys_text_names]
                + [HTML("<b>Filter by</b>"), self._filter_keys]
                + self._sidebar_ctrls
            )
        self._header_placeholder.children = list(self._header_placeholder.children or []) + [self._source_folder]
        self._widgets_placeholder.children = (
            # [HTML("⇶")]
            list(self._widgets_placeholder.children or [])
            # + ([self._unstack_tabular] if len(self.pivot_keys.text_names) > 0 else [])
        )
        return super().layout()
