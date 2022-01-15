from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import List, Sequence

import ipywidgets as w
import pandas as pd
from ipywidgets import HTML
from penelope import corpus as pc
from penelope import utility as pu
from penelope.common.keyness import KeynessMetric
from penelope.notebook import utility as nu
from penelope.notebook import word_trends as wt
from penelope.notebook import mixins as mx
from westac.riksprot.parlaclarin import metadata as md

view = w.Output(layout={'border': '2px solid green'})

CLEAR_OUTPUT = False

TEMPORAL_GROUP_BY = ['decade', 'lustrum', 'year']

# pylint: disable=no-member, too-many-instance-attributes, useless-super-delegation


@dataclass
class ComputeOpts(wt.TrendsComputeOpts):
    source_folder: str = None
    temporal_key: str = None
    pivot_keys: List[str] = None
    unstack_tabular: bool = None
    proto_metadata: md.ProtoMetaData = None

    def invalidates_corpus(self, other: "ComputeOpts") -> bool:
        if super().invalidates_corpus(other):
            return True
        if self.source_folder != other.source_folder:
            return True
        if self.temporal_key != other.temporal_key:
            return True
        if self.pivot_keys != other.pivot_keys:
            return True
        if self.unstack_tabular != other.unstack_tabular:
            return True
        return False

    @property
    def clone(self) -> "ComputeOpts":
        return copy.deepcopy(self)


class TrendsData(wt.TrendsData):
    def __init__(self, corpus: pc.VectorizedCorpus, n_top: int = 100000):
        super().__init__(corpus, n_top=n_top)
        self._trends_opts: ComputeOpts = ComputeOpts(
            normalize=False,
            keyness=KeynessMetric.TF,
            time_period='decade',
            temporal_key='decade',
            top_count=None,
            words=None,
        )

    def transform(self, opts: ComputeOpts) -> "TrendsData":
        super().transform(opts)
        return self

    def _transform_corpus(self, opts: ComputeOpts) -> pc.VectorizedCorpus:

        transformed_corpus: pc.VectorizedCorpus = self.corpus

        """ Normal word trends """
        if opts.keyness == KeynessMetric.TF_IDF:
            transformed_corpus = transformed_corpus.tf_idf()
        elif opts.keyness == KeynessMetric.TF_normalized:
            transformed_corpus = transformed_corpus.normalize_by_raw_counts()

        transformed_corpus = transformed_corpus.group_by_pivot_keys(
            temporal_key=opts.temporal_key,
            pivot_keys=opts.pivot_keys,
            aggregate='sum',
            document_namer=None,
            target_column='tokens',
        )
        transformed_corpus.replace_document_index(self.update_document_index(opts, transformed_corpus.document_index))

        if opts.normalize:
            transformed_corpus = transformed_corpus.normalize_by_raw_counts()

        return transformed_corpus

    def update_document_index(self, opts: ComputeOpts, document_index: pd.DataFrame) -> pd.DataFrame:
        if not opts.pivot_keys:
            return document_index
        di: pd.DataFrame = opts.proto_metadata.decode_members_data(document_index, drop=False)
        fg = opts.proto_metadata.MEMBER_NAME2IDNAME_MAPPING.get
        di['document_name'] = di[opts.pivot_keys].apply(lambda x: ' '.join(fg(t, str(t)) for t in x))
        di['filename'] = di.document_name
        di['time_period'] = di[opts.temporal_key]
        return di


class BaseRiksProtTrendsGUI(wt.TrendsGUI):
    ...

    def __init__(
        self,
        *,
        riksprot_metadata: md.ProtoMetaData,
        default_folder: str,
        n_top_count: int = 1000,
        encoded: int = True,
        defaults: dict = None,
    ):
        super().__init__(n_top_count=n_top_count)

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

    def setup(self, *, displayers: Sequence[wt.ITrendDisplayer] = None) -> "BaseRiksProtTrendsGUI":
        super().setup(displayers=displayers or wt.DEFAULT_WORD_TREND_DISPLAYERS)
        self._source_folder.refresh()
        self._source_folder.register_callback(self._load)

        return self

    def plot(self):
        return super().plot()

    @view.capture(clear_output=CLEAR_OUTPUT)
    def load(self) -> "BaseRiksProtTrendsGUI":
        self.observe(False)
        try:
            self.alert("😐 Loading DTM...")
            corpus: pc.VectorizedCorpus = self.load_corpus()
            self.alert("😐 Assigning parliamentary member attributes...")
            corpus = self.assign_metadata(corpus, self.riksprot_metadata)
            self.trends_data: TrendsData = TrendsData(corpus=corpus, n_top=self.n_top)
            self._compute_keyness()
            self.alert("✅")

        except FileNotFoundError:
            self.alert("No (unique) DTM corpus found in folder")
        finally:
            self.observe(True)
        return self

    def load_corpus(self) -> pc.VectorizedCorpus:

        folder: str = self.source_folder
        tags: List[str] = pc.VectorizedCorpus.find_tags(folder=folder)

        if len(tags) != 1:
            raise FileNotFoundError("No (unique) DTM corpus found in folder")

        return pc.VectorizedCorpus.load(folder=folder, tag=tags[0])

    def assign_metadata(self, corpus: pc.VectorizedCorpus, md: md.ProtoMetaData) -> pc.VectorizedCorpus:
        document_index: pd.DataFrame = md.overload_by_member_data(
            corpus.document_index, encoded=self.encoded, drop=True
        )
        document_index['lustrum'] = document_index.year - document_index.year % 5
        document_index['decade'] = document_index.year - document_index.year % 10
        document_index['tokens'] = document_index.n_raw_tokens
        corpus.replace_document_index(document_index)
        return corpus

    def _load(self, *_):
        try:
            self.load()
            self.display(trends_data=self.trends_data)
        except Exception as ex:
            self.alert(ex)

    @property
    def source_folder(self) -> str:
        return self._source_folder.selected_path

    @property
    def options(self) -> ComputeOpts:
        opts: ComputeOpts = ComputeOpts(
            **super().options.__dict__,
            temporal_key=self.time_period,
            source_folder=self.source_folder,
            proto_metadata=self.riksprot_metadata,
        )
        opts.time_period = self.time_period
        return opts


class RiksProtTrendsGUI(mx.PivotKeysMixIn, BaseRiksProtTrendsGUI):
    def __init__(self, riksprot_metadata: md.ProtoMetaData, pivot_key_specs: mx.PivotKeySpecArg = None, **kwargs):
        if pivot_key_specs is None:
            pivot_key_specs = riksprot_metadata.member_property_specs if riksprot_metadata is not None else {}
        super().__init__(pivot_key_specs=pivot_key_specs, riksprot_metadata=riksprot_metadata, **kwargs)

    @property
    def options(self) -> ComputeOpts:
        opts = ComputeOpts(
            **{
                **super().options.__dict__,
                **{'pivot_keys': self.pivot_keys_id_names, 'unstack_tabular': self.unstack_tabular},
            }
        )
        return opts

    def buzy(self, value: bool) -> None:
        super().buzy(value)
        self._unstack_tabular.disables = value
        self._pivot_keys_text_names.disabled = value
        self._filter_keys.disabled = value

    def layout(self) -> w.HBox:
        self._filter_keys.layout = {'width': '180px'}
        self._pivot_keys_text_names.layout = {'width': '180px'}
        if self.pivot_keys.has_pivot_keys:
            self._picker.rows = 15
            self._sidebar_ctrls = (
                [HTML("<b>Pivot by</b>"), self._pivot_keys_text_names]
                + [HTML("<b>Filter by</b>"), self._filter_keys]
                + self._sidebar_ctrls
            )
        self._header_placeholder.children = list(self._header_placeholder.children or []) + [self._source_folder]
        self._widgets_placeholder.children = [HTML("⇶")] + list(self._widgets_placeholder.children or []) + (
            [self._unstack_tabular] if len(self.pivot_keys.text_names) > 0 else []
        )
        return super().layout()

    def unstack_pivot_keys(self, data: pd.DataFrame) -> pd.DataFrame:
        return pu.unstack_data(data, [self.temporal_key] + self.pivot_keys_text_names)

    def _display(self, *_):
        print("Do something smart")