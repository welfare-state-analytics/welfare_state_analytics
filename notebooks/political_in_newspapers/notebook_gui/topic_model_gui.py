import os
import types
import glob
import ipywidgets
import logging
import text_analytic_tools.utility as utility
import text_analytic_tools.text_analysis.derived_data_compiler as derived_data_compiler
import text_analytic_tools.text_analysis.topic_model as topic_model
import text_analytic_tools.common.text_corpus as text_corpus
import text_analytic_tools.common.textacy_utility as textacy_utility
#from . topic_model_compute import compute_topic_model

logger = utility.getLogger('corpus_text_analysis')

gensim_logger = logging.getLogger('gensim')
gensim_logger.setLevel(logging.INFO)

from IPython.display import display

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
    options = [ x for x in tag_set.POS.unique() if x not in ['PUNCT', '', 'DET', 'X', 'SPACE', 'PART', 'CONJ', 'SYM', 'INTJ', 'PRON']]
    return options

def get_spinner_widget(filename="images/spinner-02.gif", width=40, height=40):
    with open(filename, "rb") as image_file:
        image = image_file.read()
    return ipywidgets.Image(value=image, format='gif', width=width, height=height, layout={'visibility': 'hidden'})


class ComputeTopicModelUserInterface():

    def __init__(self, data_folder, state, document_index, **opts):
        self.terms = []
        self.data_folder = data_folder
        self.state = state
        self.document_index = document_index
        self.opts = opts
        self.model_widgets, self.widget_boxes = self.prepare_widgets()

    def prepare_widgets(self):

        gui = types.SimpleNamespace(
            apply_idf=ipywidgets.ToggleButton(value=False, description='TF-IDF',  tooltip='Apply IDF (skikit-learn) or TF-IDF (gensim)', icon='check', layout=ipywidgets.Layout(width='115px')),
            method=ipywidgets.Dropdown(description='Engine', options=ENGINE_OPTIONS, value='gensim_lda', layout=ipywidgets.Layout(width='200px')),
            n_topics=ipywidgets.IntSlider(description='Topics', min=2, max=100, value=20, step=1, layout=ipywidgets.Layout(width='240px')),
            max_iter=ipywidgets.IntSlider(description='Iterations', min=100, max=6000, value=2000, step=10, layout=ipywidgets.Layout(width='240px')),
            show_trace=ipywidgets.ToggleButton(value=False, description='Show trace', disabled=False, icon='check', layout=ipywidgets.Layout(width='115px')),
            compute=ipywidgets.Button(description='Compute', button_style='Success', layout=ipywidgets.Layout(width='115px',background_color='blue')),
            output=ipywidgets.Output(layout={'border': '1px solid black'}),
            spinner=get_spinner_widget()
        )

        boxes = [
            ipywidgets.VBox([
                gui.method,
                gui.n_topics,
                gui.max_iter,
            ], layout=ipywidgets.Layout(margin='0px 0px 0px 0px')),
            ipywidgets.VBox([
                gui.apply_idf,
                gui.show_trace,
                gui.compute,
                gui.spinner,
            ], layout=ipywidgets.Layout(align_items='flex-start'))
        ]

        return gui, boxes

    def get_corpus_terms(self, corpus):
        # assert isinstance(corpus, collections.Isiterable), 'Must be a iterable!'
        return corpus

    def display(self, corpus=None):

        def buzy(is_buzy):
            self.model_widgets.compute.disabled = is_buzy
            self.model_widgets.spinner.layout.visibility = 'visible' if is_buzy else 'hidden'

        def compute_topic_model_handler(*args):

            self.model_widgets.output.clear_output()

            buzy(True)

            gensim_logger.setLevel(logging.INFO if self.model_widgets.show_trace.value else logging.WARNING)

            with self.model_widgets.output:

                try:

                    vectorizer_args = dict(apply_idf=self.model_widgets.apply_idf.value)

                    topic_modeller_args = dict(n_topics=self.model_widgets.n_topics.value, max_iter=self.model_widgets.max_iter.value, learning_method='online', n_jobs=1)

                    method = self.model_widgets.method.value

                    terms = list(self.get_corpus_terms(corpus))

                    # FIXME API change, use named args
                    self.state.data = topic_model.compute(self.data_folder, method, terms, self.document_index, vectorizer_args, topic_modeller_args)

                    topics = derived_data_compiler.get_topics_unstacked(self.state.topic_model, n_tokens=100, id2term=self.state.id2term, topic_ids=self.state.relevant_topics)

                    display(topics)

                except Exception as ex:
                    logger.error(ex)
                    self.state.data = None
                    raise
                finally:
                    buzy(False)

        self.model_widgets.compute.on_click(compute_topic_model_handler)

        def method_change_handler(*args):
            with self.model_widgets.output:

                self.model_widgets.compute.disabled = True
                method = self.model_widgets.method.value

                self.model_widgets.apply_idf.disabled = False
                self.model_widgets.apply_idf.description = 'Apply TF-IDF' if method.startswith('gensim') else 'Apply IDF'

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

        display(ipywidgets.VBox([ ipywidgets.HBox(self.widget_boxes), self.model_widgets.output ]))

class TextacyCorpusUserInterface(ComputeTopicModelUserInterface):

    def __init__(self, data_folder, state, document_index, **opts):

        ComputeTopicModelUserInterface.__init__(self, data_folder, state, document_index, **opts)

        self.substitution_filename = self.opts.get('substitution_filename', None)
        self.tagset = self.opts.get('tagset', None)

        self.corpus_widgets, self.corpus_widgets_boxes = self.prepare_textacy_widgets()
        self.widget_boxes = self.corpus_widgets_boxes + self.widget_boxes

    def display(self, corpus=None):

        # assert hasattr(corpus, 'spacy_lang), 'Must be a textaCy corpus!'
        self.corpus_widgets.named_entities.disabled = len(corpus) == 0 or len(corpus[0].ents) == 0

        def pos_change_handler(*args):
            with self.model_widgets.output:
                self.model_widgets.compute.disabled = True
                selected = set(self.corpus_widgets.stop_words.value)
                frequent_words = [
                    x[0] for x in textacy_utility.get_most_frequent_words(
                        corpus, 100, normalize=self.corpus_widgets.normalize.value, include_pos=self.corpus_widgets.include_pos.value) ]
                self.corpus_widgets.stop_words.options = frequent_words
                selected = selected & set(self.corpus_widgets.stop_words.options)
                self.corpus_widgets.stop_words.value = list(selected)
                self.model_widgets.compute.disabled = False

        self.corpus_widgets.include_pos.observe(pos_change_handler, 'value')
        pos_change_handler()

        def corpus_method_change_handler(*args):
            self.corpus_widgets.ngrams.disabled = False
            if 'MALLET' in self.model_widgets.method.value:
                self.corpus_widgets.ngrams.value = [1]
                self.corpus_widgets.ngrams.disabled = True

        self.model_widgets.method.observe(corpus_method_change_handler, 'value')

        ComputeTopicModelUserInterface.display(self, corpus)

    def get_corpus_terms(self, corpus):

        tokenizer_args = self.compile_tokenizer_args(vocab=corpus.spacy_lang.vocab)
        terms = [ list(doc) for doc in textacy_utility.extract_corpus_terms(corpus, tokenizer_args) ]
        return terms

    def compile_tokenizer_args(self, vocab=None):

        term_substitutions = {}

        gui = self.corpus_widgets

        if gui.substitute_terms.value is True:
            assert self.substitution_filename is not None
            term_substitutions = textacy_utility.load_term_substitutions(self.substitution_filename, default_term='_mask_', delim=';', vocab=vocab)

        args = dict(
            args=dict(
                ngrams=gui.ngrams.value,
                named_entities=gui.named_entities.value,
                normalize=gui.normalize.value,
                as_strings=True
            ),
            kwargs=dict(
                min_freq=gui.min_freq.value,
                include_pos=gui.include_pos.value,
                filter_stops=gui.filter_stops.value,
                filter_punct=True
            ),
            extra_stop_words=set(gui.stop_words.value),
            substitutions=term_substitutions,
            min_freq=gui.min_freq.value,
            max_doc_freq=gui.max_doc_freq.value
        )

        return args

    def prepare_textacy_widgets(self):

        item_layout = dict(
            display='flex',
            flex_flow='row',
            justify_content='space-between',
        )

        pos_options = get_pos_options(self.tagset)

        normalize_options = { 'None': False, 'Lemma': 'lemma', 'Lower': 'lower'}
        ngrams_options = { '1': [1], '1, 2': [1, 2], '1,2,3': [1, 2, 3] }
        default_include_pos = [ 'NOUN', 'PROPN' ]
        frequent_words = [ '_mask_' ]
        # ipywidgets.Label(
        gui = types.SimpleNamespace(
            #min_freq=ipywidgets.IntSlider(description='Min word freq',min=0, max=10, value=2, step=1, layout=ipywidgets.Layout(width='240px', **item_layout)),
            #max_doc_freq=ipywidgets.IntSlider(description='Min doc %', min=75, max=100, value=100, step=1, layout=ipywidgets.Layout(width='240px', **item_layout)),
            min_freq=ipywidgets.Dropdown(description='Min word freq', options=list(range(0,11)), value=2, layout=ipywidgets.Layout(width='200px', **item_layout)),
            max_doc_freq=ipywidgets.Dropdown(description='Min doc %', options=list(range(75,101)), value=100, layout=ipywidgets.Layout(width='200px', **item_layout)),
            ngrams=ipywidgets.Dropdown(description='n-grams', options=ngrams_options, value=[1], layout=ipywidgets.Layout(width='200px')),
            normalize=ipywidgets.Dropdown(description='Normalize', options=normalize_options, value='lemma', layout=ipywidgets.Layout(width='200px')),
            filter_stops=ipywidgets.ToggleButton(value=True, description='Remove stopword',  tooltip='Filter out stopwords', icon='check'),
            named_entities=ipywidgets.ToggleButton(value=False, description='Merge entities',  tooltip='Merge entities', icon='check', disabled=False),
            substitute_terms=ipywidgets.ToggleButton(value=False, description='Map words',  tooltip='Substitute words', icon='check'),
            include_pos=ipywidgets.SelectMultiple(options=pos_options, value=default_include_pos, rows=7, layout=ipywidgets.Layout(width='60px', **item_layout)),
            stop_words=ipywidgets.SelectMultiple(options=frequent_words, value=list([]), rows=7, layout=ipywidgets.Layout(width='120px', **item_layout)),
        )
        boxes = [
            ipywidgets.VBox([gui.min_freq, gui.max_doc_freq, gui.normalize, gui.ngrams,]),
            ipywidgets.VBox([gui.filter_stops, gui.named_entities, gui.substitute_terms], layout=ipywidgets.Layout(margin='0px 0px 0px 10px')),
            ipywidgets.HBox([
                ipywidgets.Label(value='POS', layout=ipywidgets.Layout(width='40px')),
                gui.include_pos
            ], layout=ipywidgets.Layout(margin='0px 0px 0px 10px')),
            ipywidgets.HBox([ipywidgets.Label(value='STOP'), gui.stop_words ], layout=ipywidgets.Layout(margin='0px 0px 0px 10px'))
        ]
        return gui,boxes

class PreparedCorpusUserInterface(ComputeTopicModelUserInterface):

    def __init__(self, data_folder, state, fn_doc_index, **opts):

        ComputeTopicModelUserInterface.__init__(self, data_folder, state, document_index=None, **opts)

        self.corpus_widgets, self.corpus_widgets_boxes = self.prepare_source_widgets()
        self.widget_boxes = self.corpus_widgets_boxes + self.widget_boxes
        self.corpus = None
        self.fn_doc_index = fn_doc_index

    def prepare_source_widgets(self):
        corpus_files = sorted(glob.glob(os.path.join(self.data_folder, '*.tokenized.zip')))
        gui = types.SimpleNamespace(
            filepath=utility.widgets.dropdown(description='Corpus', options=corpus_files, value=None, layout=ipywidgets.Layout(width='500px'))
        )

        return gui, [ gui.filepath ]

    def get_corpus_terms(self, corpus):
        filepath = self.corpus_widgets.filepath.value
        self.corpus = text_corpus.SimplePreparedTextCorpus(filepath)
        doc_terms = [ list(terms) for terms in self.corpus.get_texts() ]
        self.document_index = self.fn_doc_index(self.corpus)
        return doc_terms

    def display(self, default_source=None):

        ComputeTopicModelUserInterface.display(self, None)

BUTTON_STYLE = dict(description_width='initial', button_color='lightgreen')
