import glob
import logging
import os
import types
import uuid

import ipywidgets as widgets
import pandas as pd
import penelope.topic_modelling as topic_modelling
import penelope.utility as utility
import penelope.vendor.gensim as gensim_utility
import penelope.vendor.textacy as textacy_utility
from IPython.display import display
from penelope.notebook.topic_modelling import TopicModelContainer

# from . topic_model_compute import compute_topic_model

logger = utility.getLogger('corpus_text_analysis')

gensim_logger = logging.getLogger('gensim')
gensim_logger.setLevel(logging.INFO)

ENGINE_OPTIONS = [
    ('MALLET LDA', 'gensim_mallet-lda'),
    ('gensim LDA', 'gensim_lda'),
    ('gensim LDA multicore', 'gensim_lda-multicore'),
    ('gensim LSI', 'gensim_lsi'),
    ('gensim HDP', 'gensim_hdp'),
    ('gensim DTM', 'gensim_dtm'),
    ('scikit LDA', 'sklearn_lda'),
    ('scikit NMF', 'sklearn_nmf'),
    ('scikit LSA', 'sklearn_lsa'),
    ('STTM   LDA', 'gensim_sttm-lda'),
    ('STTM   BTM', 'gensim_sttm-btm'),
    ('STTM   PTM', 'gensim_sttm-ptm'),
    ('STTM  SATM', 'gensim_sttm-satm'),
    ('STTM   DMM', 'gensim_sttm-dmm'),
    ('STTM  WATM', 'gensim_sttm-watm'),
]


def get_pos_options(tag_set):
    options = [
        x
        for x in tag_set.POS.unique()
        if x not in ['PUNCT', '', 'DET', 'X', 'SPACE', 'PART', 'CONJ', 'SYM', 'INTJ', 'PRON']
    ]
    return options


def get_spinner_widget(filename="images/spinner-02.gif", width=40, height=40):
    with open(filename, "rb") as image_file:
        image = image_file.read()
    return widgets.Image(value=image, format='gif', width=width, height=height, layout={'visibility': 'hidden'})


class ComputeTopicModelUserInterface:
    def __init__(self, data_folder: str, state: TopicModelContainer, document_index: pd.DataFrame, **opts):
        self.terms = []
        self.data_folder = data_folder
        self.state = state
        self.document_index = document_index
        self.opts = opts
        self.model_widgets, self.widget_boxes = self.prepare_widgets()

    def prepare_widgets(self):

        gui = types.SimpleNamespace(
            apply_idf=widgets.ToggleButton(
                value=False,
                description='TF-IDF',
                tooltip='Apply IDF (skikit-learn) or TF-IDF (gensim)',
                icon='check',
                layout=widgets.Layout(width='115px'),
            ),
            method=widgets.Dropdown(
                description='Engine',
                options=ENGINE_OPTIONS,
                value='gensim_lda',
                layout=widgets.Layout(width='200px'),
            ),
            n_topics=widgets.IntSlider(
                description='Topics', min=2, max=100, value=20, step=1, layout=widgets.Layout(width='240px')
            ),
            max_iter=widgets.IntSlider(
                description='Iterations',
                min=100,
                max=6000,
                value=2000,
                step=10,
                layout=widgets.Layout(width='240px'),
            ),
            show_trace=widgets.ToggleButton(
                value=False,
                description='Show trace',
                disabled=False,
                icon='check',
                layout=widgets.Layout(width='115px'),
            ),
            compute=widgets.Button(
                description='Compute',
                button_style='Success',
                layout=widgets.Layout(width='115px', background_color='blue'),
            ),
            output=widgets.Output(layout={'border': '1px solid black'}),
            spinner=get_spinner_widget(),
        )

        boxes = [
            widgets.VBox(
                [
                    gui.method,
                    gui.n_topics,
                    gui.max_iter,
                ],
                layout=widgets.Layout(margin='0px 0px 0px 0px'),
            ),
            widgets.VBox(
                [
                    gui.apply_idf,
                    gui.show_trace,
                    gui.compute,
                    gui.spinner,
                ],
                layout=widgets.Layout(align_items='flex-start'),
            ),
        ]

        return gui, boxes

    def get_corpus_terms(self, corpus):
        # assert isinstance(corpus, collections.Isiterable), 'Must be a iterable!'
        return corpus

    def display(self, corpus=None):
        def buzy(is_buzy):
            self.model_widgets.compute.disabled = is_buzy
            self.model_widgets.spinner.layout.visibility = 'visible' if is_buzy else 'hidden'

        def compute_topic_model_handler(*_):

            self.model_widgets.output.clear_output()

            buzy(True)

            gensim_logger.setLevel(logging.INFO if self.model_widgets.show_trace.value else logging.WARNING)

            with self.model_widgets.output:

                try:

                    # FIXME: Generate folder name based on corpus filename and options
                    name = uuid.uuid1()

                    target_folder = os.path.join(self.data_folder, name)

                    vectorizer_args = dict(apply_idf=self.model_widgets.apply_idf.value)

                    topic_modeller_args = dict(
                        n_topics=self.model_widgets.n_topics.value,
                        max_iter=self.model_widgets.max_iter.value,
                        learning_method='online',
                        n_jobs=1,
                    )

                    method = self.model_widgets.method.value

                    train_corpus = topic_modelling.TrainingCorpus(
                        terms=list(self.get_corpus_terms(corpus)),
                        document_index=self.document_index,
                        vectorizer_args=vectorizer_args,
                    )

                    inferred_model = topic_modelling.infer_model(
                        train_corpus=train_corpus, method=method, engine_args=topic_modeller_args
                    )

                    inferred_topics = topic_modelling.compile_inferred_topics_data(
                        inferred_model.topic_model,
                        train_corpus.corpus,
                        train_corpus.id2word,
                        train_corpus.document_index,
                    )

                    inferred_model.topic_model.save(os.path.join(target_folder, 'gensim.model'))
                    topic_modelling.store_model(inferred_model, target_folder)
                    inferred_topics.store(target_folder)

                    topics = topic_modelling.get_topics_unstacked(
                        self.state.topic_model,
                        n_tokens=100,
                        id2term=self.state.id2term,
                        topic_ids=self.state.relevant_topics,
                    )

                    self.state.inferred_model = inferred_model
                    self.state.inferred_topics = inferred_model
                    self.state.inferred_model = inferred_model

                    display(topics)

                except Exception as ex:
                    logger.error(ex)
                    self.state.data = None
                    raise
                finally:
                    buzy(False)

        self.model_widgets.compute.on_click(compute_topic_model_handler)

        def method_change_handler(*_):
            with self.model_widgets.output:

                self.model_widgets.compute.disabled = True
                method = self.model_widgets.method.value

                self.model_widgets.apply_idf.disabled = False
                self.model_widgets.apply_idf.description = (
                    'Apply TF-IDF' if method.startswith('gensim') else 'Apply IDF'
                )

                if 'MALLET' in method:
                    self.model_widgets.apply_idf.description = 'TF-IDF N/A'
                    self.model_widgets.apply_idf.disabled = True

                self.model_widgets.n_topics.disabled = False
                if 'HDP' in method:
                    self.model_widgets.n_topics.value = self.model_widgets.n_topics.max
                    self.model_widgets.n_topics.disabled = True

                self.model_widgets.compute.disabled = False

        self.model_widgets.method.observe(method_change_handler, 'value')

        method_change_handler()

        display(widgets.VBox([widgets.HBox(self.widget_boxes), self.model_widgets.output]))


class TextacyCorpusUserInterface(ComputeTopicModelUserInterface):
    def __init__(self, data_folder: str, state: TopicModelContainer, document_index: pd.DataFrame, **opts):

        super().__init__(data_folder, state, document_index, **opts)

        self.substitution_filename = self.opts.get('substitution_filename', None)
        self.tagset = self.opts.get('tagset', None)

        self.corpus_widgets, self.corpus_widgets_boxes = self.prepare_textacy_widgets()
        self.widget_boxes = self.corpus_widgets_boxes + self.widget_boxes

    def display(self, corpus=None):

        # assert hasattr(corpus, 'spacy_lang), 'Must be a textaCy corpus!'

        def pos_change_handler(*_):
            with self.model_widgets.output:
                self.model_widgets.compute.disabled = True
                selected = set(self.corpus_widgets.stop_words.value)
                frequent_words = [
                    x[0]
                    for x in textacy_utility.get_most_frequent_words(
                        corpus,
                        100,
                        normalize=self.corpus_widgets.normalize.value,
                        include_pos=self.corpus_widgets.include_pos.value,
                    )
                ]
                self.corpus_widgets.stop_words.options = frequent_words
                selected = selected & set(self.corpus_widgets.stop_words.options)
                self.corpus_widgets.stop_words.value = list(selected)
                self.model_widgets.compute.disabled = False

        self.corpus_widgets.include_pos.observe(pos_change_handler, 'value')
        pos_change_handler()

        def corpus_method_change_handler(*_):
            self.corpus_widgets.ngrams.disabled = False
            if 'MALLET' in self.model_widgets.method.value:
                self.corpus_widgets.ngrams.value = [1]
                self.corpus_widgets.ngrams.disabled = True

        self.model_widgets.method.observe(corpus_method_change_handler, 'value')

        ComputeTopicModelUserInterface.display(self, corpus)

    def get_corpus_terms(self, corpus):
        pipeline = self._create_extract_pipeline(corpus=corpus)
        terms = [list(doc) for doc in pipeline.process()]
        return terms

    def _create_extract_pipeline(self, corpus):

        gui = self.corpus_widgets

        pipeline = (
            textacy_utility.ExtractPipeline.build(corpus, target=gui.normalize.value)
            .ingest(
                ngrams=gui.ngrams.value,
                as_strings=True,
                include_pos=gui.include_pos.value,
                filter_stops=gui.filter_stops.value,
                filter_punct=True,
            )
            .frequent_word_filter(max_doc_freq=gui.max_doc_freq.value)
            .infrequent_word_filter(min_freq=gui.min_freq.value)
            .remove_stopwords(extra_stopwords=set(gui.stop_words.value))
        )

        if gui.substitute_terms.value is True:
            pipeline = pipeline.substitute(subst_map=None, filename=self.substitution_filename)

        return pipeline

    def prepare_textacy_widgets(self):

        item_layout = dict(
            display='flex',
            flex_flow='row',
            justify_content='space-between',
        )

        pos_options = get_pos_options(self.tagset)

        normalize_options = {'None': False, 'Lemma': 'lemma', 'Lower': 'lower'}
        ngrams_options = {'1': [1], '1, 2': [1, 2], '1,2,3': [1, 2, 3]}
        default_include_pos = ['NOUN', 'PROPN']
        frequent_words = ['_mask_']

        gui = types.SimpleNamespace(
            min_freq=widgets.Dropdown(
                description='Min word freq',
                options=list(range(0, 11)),
                value=2,
                layout=widgets.Layout(width='200px', **item_layout),
            ),
            max_doc_freq=widgets.Dropdown(
                description='Min doc %',
                options=list(range(75, 101)),
                value=100,
                layout=widgets.Layout(width='200px', **item_layout),
            ),
            ngrams=widgets.Dropdown(
                description='n-grams', options=ngrams_options, value=[1], layout=widgets.Layout(width='200px')
            ),
            normalize=widgets.Dropdown(
                description='Normalize',
                options=normalize_options,
                value='lemma',
                layout=widgets.Layout(width='200px'),
            ),
            filter_stops=widgets.ToggleButton(
                value=True, description='Remove stopword', tooltip='Filter out stopwords', icon='check'
            ),
            substitute_terms=widgets.ToggleButton(
                value=False, description='Map words', tooltip='Substitute words', icon='check'
            ),
            include_pos=widgets.SelectMultiple(
                options=pos_options,
                value=default_include_pos,
                rows=7,
                layout=widgets.Layout(width='60px', **item_layout),
            ),
            stop_words=widgets.SelectMultiple(
                options=frequent_words, value=list([]), rows=7, layout=widgets.Layout(width='120px', **item_layout)
            ),
        )
        boxes = [
            widgets.VBox(
                [
                    gui.min_freq,
                    gui.max_doc_freq,
                    gui.normalize,
                    gui.ngrams,
                ]
            ),
            widgets.VBox(
                [gui.filter_stops, gui.substitute_terms],
                layout=widgets.Layout(margin='0px 0px 0px 10px'),
            ),
            widgets.HBox(
                [widgets.Label(value='POS', layout=widgets.Layout(width='40px')), gui.include_pos],
                layout=widgets.Layout(margin='0px 0px 0px 10px'),
            ),
            widgets.HBox(
                [widgets.Label(value='STOP'), gui.stop_words], layout=widgets.Layout(margin='0px 0px 0px 10px')
            ),
        ]
        return gui, boxes


class PreparedCorpusUserInterface(ComputeTopicModelUserInterface):
    def __init__(self, data_folder: str, state: TopicModelContainer, fn_doc_index, **opts):

        super().__init__(data_folder, state, document_index=None, **opts)

        self.corpus_widgets, self.corpus_widgets_boxes = self.prepare_source_widgets()
        self.widget_boxes = self.corpus_widgets_boxes + self.widget_boxes
        self.corpus = None
        self.fn_doc_index = fn_doc_index

    def prepare_source_widgets(self):
        corpus_files = sorted(glob.glob(os.path.join(self.data_folder, '*.tokenized.zip')))
        gui = types.SimpleNamespace(
            filepath=widgets.Dropdown(
                description='Corpus', options=corpus_files, value=None, layout=widgets.Layout(width='500px')
            )
        )

        return gui, [gui.filepath]

    def get_corpus_terms(self, _):
        filepath = self.corpus_widgets.filepath.value
        self.corpus = gensim_utility.SimpleExtTextCorpus(filepath)
        doc_terms = [list(terms) for terms in self.corpus.get_texts()]
        self.document_index = self.fn_doc_index(self.corpus)
        return doc_terms

    def display(self, _=None):  # pylint: disable=arguments-differ, unused-argument

        ComputeTopicModelUserInterface.display(self, None)


BUTTON_STYLE = dict(description_width='initial', button_color='lightgreen')
