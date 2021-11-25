import os
import shutil
import uuid
from typing import Any, Mapping

import gensim
import pandas as pd
import penelope.topic_modelling as tm
import pytest
from penelope.corpus.text_lines_corpus import SimpleTextLinesCorpus
from penelope.scripts.topic_model_legacy import main
from penelope.vendor.gensim.wrappers import LdaMallet

jj = os.path.join

OUTPUT_FOLDER = "./tests/output/"

TOPIC_MODELING_OPTS = {
    "gensim_lda": {
        'method': "gensim_lda",
        'skip': False,
        'class': gensim.models.ldamodel.LdaModel,
        'run_opts': {
            'n_topics': 4,
            'passes': 1,
            'random_seed': 42,
            'alpha': 'auto',
            'workers': 1,
            'max_iter': 100,
            'work_folder': '',
        },
    },
    "gensim_mallet-lda": {
        'method': "gensim_mallet-lda",
        'skip': os.environ.get('MALLET_HOME', None) is not None,
        'class': LdaMallet,
        'run_opts': {
            'default_mallet_home': os.environ.get('MALLET_HOME', None),
            'n_topics': 4,  # note: mallet num_topics
            'max_iter': 3000,  # note: mallet iterations
            'work_folder': OUTPUT_FOLDER,
            'workers': 1,
            'optimize_interval': 10,
            'topic_threshold': 0.0,
            'random_seed': 0,
        },
    },
}


class TranströmerCorpus(SimpleTextLinesCorpus):
    def __init__(self):
        super().__init__(
            filename='./tests/test_data/tranströmer/tranströmer.txt',
            fields={'filename': 0, 'title': 1, 'text': 2},
            filename_fields=["year:_:1", "year_serial_id:_:2"],  # type: ignore
        )


def compute_inferred_model(engine: str, opts: Mapping[str, Any]) -> tm.InferredModel:

    corpus: TranströmerCorpus = TranströmerCorpus()
    train_corpus: tm.TrainingCorpus = tm.TrainingCorpus(
        terms=corpus.terms,
        document_index=corpus.document_index,
    )

    trained_model: tm.InferredModel = tm.train_model(
        train_corpus=train_corpus,
        method=engine,
        engine_args=opts,
    )
    return trained_model


def test_tranströmers_corpus():

    corpus: TranströmerCorpus = TranströmerCorpus()
    for filename, tokens in corpus:
        assert len(filename) > 0
        assert len(tokens) > 0


@pytest.mark.parametrize("opts", list(TOPIC_MODELING_OPTS.values()))
def test_infer_model(opts):

    inferred_model: tm.InferredModel = compute_inferred_model(opts['method'], opts['run_opts'])

    assert inferred_model is not None
    assert inferred_model.method == opts['method']
    assert isinstance(inferred_model.topic_model, opts['class'])
    assert isinstance(inferred_model.train_corpus.document_index, pd.DataFrame)
    assert len(inferred_model.train_corpus.corpus) == len(inferred_model.train_corpus.document_index)  # type: ignore
    assert len(inferred_model.train_corpus.document_index) == 5
    assert len(inferred_model.train_corpus.document_index.columns) == 7
    assert 'n_terms' in inferred_model.train_corpus.document_index.columns
    assert inferred_model.train_corpus.corpus is not None

    # dtm = corpus
    # data = np.empty(n_nonzero, dtype=np.intc)     # all non-zero term frequencies at data[k]
    # rows = np.empty(n_nonzero, dtype=np.intc)     # row index for kth data item (kth term freq.)
    # cols = np.empty(n_nonzero, dtype=np.intc)


@pytest.mark.parametrize("opts", list(TOPIC_MODELING_OPTS.values()))
def test_store_inferred_model(opts):

    # Arrange
    name: str = f"{uuid.uuid1()}"
    target_folder: str = os.path.join(OUTPUT_FOLDER, name)
    inferred_model: tm.InferredModel = compute_inferred_model(opts['method'], opts['run_opts'])

    # Act
    inferred_model.store(target_folder)

    # Assert
    assert os.path.isfile(os.path.join(target_folder, "topic_model.pickle.pbz2"))

    shutil.rmtree(target_folder)


@pytest.mark.parametrize("opts", list(TOPIC_MODELING_OPTS.values()))
def test_store_gensim_model(opts):

    # Arrange
    folder = f"{uuid.uuid1()}"
    target_name = os.path.join(OUTPUT_FOLDER, folder, "test_corpus.gensim")
    inferred_model = compute_inferred_model(opts['method'], opts['run_opts'])
    os.makedirs(os.path.join(OUTPUT_FOLDER, folder), exist_ok=True)
    # Act
    inferred_model.topic_model.save(target_name)
    inferred_model.topic_model.save(target_name + '.gz')

    assert os.path.isfile(target_name)


@pytest.mark.parametrize("opts", list(TOPIC_MODELING_OPTS.values()))
def test_load_inferred_model_when_stored_corpus_is_true_has_same_loaded_trained_corpus(opts):

    name = f"{uuid.uuid1()}"
    target_folder = os.path.join(OUTPUT_FOLDER, name)
    test_inferred_model: tm.InferredModel = compute_inferred_model(opts['method'], opts['run_opts'])
    test_inferred_model.store(target_folder, store_corpus=True)

    # Act

    inferred_model: tm.InferredModel = tm.InferredModel.load(target_folder)

    # Assert
    assert inferred_model is not None
    assert inferred_model.method == opts['method']
    assert isinstance(inferred_model.topic_model, opts['class'])
    assert isinstance(inferred_model.train_corpus.document_index, pd.DataFrame)
    assert len(inferred_model.train_corpus.corpus) == len(inferred_model.train_corpus.document_index)
    assert len(inferred_model.train_corpus.document_index) == 5
    assert len(inferred_model.train_corpus.document_index.columns) == 7
    assert 'n_terms' in inferred_model.train_corpus.document_index.columns
    assert inferred_model.train_corpus.corpus is not None

    shutil.rmtree(target_folder)


@pytest.mark.parametrize("opts", list(TOPIC_MODELING_OPTS.values()))
def test_load_inferred_model_when_stored_corpus_is_false_has_no_trained_corpus(opts):

    # Arrange
    name: str = f"{uuid.uuid1()}"
    target_folder: str = os.path.join(OUTPUT_FOLDER, name)
    test_inferred_model: tm.InferredModel = compute_inferred_model(opts['method'], opts['run_opts'])
    test_inferred_model.store(target_folder, store_corpus=False)

    # Actopts['method'],

    inferred_model: tm.InferredModel = tm.InferredModel.load(target_folder)

    # Assert
    assert inferred_model is not None
    assert inferred_model.method == opts['method']
    assert isinstance(inferred_model.topic_model, opts['class'])
    assert inferred_model.train_corpus is None

    shutil.rmtree(target_folder)


@pytest.mark.parametrize("opts", list(TOPIC_MODELING_OPTS.values()))
def test_infer_topics_data(opts):

    inferred_model = compute_inferred_model(opts['method'], opts['run_opts'])

    inferred_topics_data: tm.InferredTopicsData = tm.predict_topics(
        topic_model=inferred_model.topic_model,
        corpus=inferred_model.train_corpus.corpus,
        id2token=inferred_model.train_corpus.id2token,
        document_index=inferred_model.train_corpus.document_index,
        n_tokens=5,
    )

    assert inferred_topics_data is not None
    assert isinstance(inferred_topics_data.document_index, pd.DataFrame)
    assert isinstance(inferred_topics_data.dictionary, pd.DataFrame)
    assert isinstance(inferred_topics_data.topic_token_weights, pd.DataFrame)
    assert isinstance(inferred_topics_data.topic_token_overview, pd.DataFrame)
    assert isinstance(inferred_topics_data.document_topic_weights, pd.DataFrame)


@pytest.mark.parametrize("opts", list(TOPIC_MODELING_OPTS.values()))
def test_store_inferred_topics_data(opts):

    inferred_model = compute_inferred_model(opts['method'], opts['run_opts'])

    inferred_topics_data: tm.InferredTopicsData = tm.predict_topics(
        topic_model=inferred_model.topic_model,
        corpus=inferred_model.train_corpus.corpus,
        id2token=inferred_model.train_corpus.id2token,
        document_index=inferred_model.train_corpus.document_index,
        n_tokens=5,
    )
    target_folder = jj(OUTPUT_FOLDER, f"{uuid.uuid1()}")

    inferred_topics_data.store(target_folder)

    assert os.path.isfile(jj(target_folder, "dictionary.zip"))
    assert os.path.isfile(jj(target_folder, "document_topic_weights.zip"))
    assert os.path.isfile(jj(target_folder, "documents.zip"))
    assert os.path.isfile(jj(target_folder, "topic_token_overview.zip"))
    assert os.path.isfile(jj(target_folder, "topic_token_weights.zip"))

    shutil.rmtree(target_folder)


def test_run_cli():

    kwargs = {
        'target_name': f"{uuid.uuid1()}",
        'corpus_folder': './tests/output',
        'corpus_source': './tests/test_data/tranströmer/test_corpus.zip',
        'engine': 'gensim_lda-multicore',
        'engine_args': {
            # 'passes': None,
            # 'random_seed': None,
            'n_topics': 5,
            'alpha': 'asymmetric',
            # 'workers': None,
            # 'max_iter': None,
        },
        'filename_field': ('year:_:1', 'sequence_id:_:2'),
    }

    main(**kwargs)

    target_folder = jj(kwargs['corpus_folder'], kwargs['target_name'])

    assert os.path.isdir(target_folder)
    assert os.path.isfile(jj(target_folder, 'topic_model.pickle.pbz2'))
    assert os.path.isfile(jj(target_folder, 'model_options.json'))

    shutil.rmtree(target_folder)
