import os
from glob import glob
from typing import Iterable, List, Set
from unittest.mock import Mock

import pandas as pd
import pytest
import tqdm
from penelope import corpus
from penelope import pipeline as pp
from penelope import utility
from penelope.pipeline import checkpoint, interfaces
from westac.riksdagens_protokoll.parlaclarin import tasks

# pylint: disable=redefined-outer-name
CONFIG_FILENAME = './tests/test_data/riksdagens_protokoll/parlaclarin/riksdagens-protokoll.yml'
TAGGED_DATA_FOLDER = './tests/test_data/parlaclarin/tagged-1'
jj = os.path.join


@pytest.fixture
def config() -> pp.CorpusConfig:
    config: pp.CorpusConfig = pp.CorpusConfig.load(CONFIG_FILENAME)
    config.pipeline_payload.files(
        source='./tests/test_data/annotated',
        document_index_source=None,
    )
    return config


@pytest.fixture
def checkpoint_opts() -> checkpoint.CheckpointOpts:
    opts = checkpoint.CheckpointOpts(
        content_type_code=pp.ContentType.TAGGED_FRAME,
        sep="\t",
        quoting=3,
        document_index_name="document_index.csv",
        document_index_sep="\t",
        text_column="text",
        lemma_column="lemma",
        pos_column="pos",
        custom_serializer_classname=None,
        index_column=None,
    )
    return opts


@pytest.mark.skip("Not implemented")
def test_load_checkpoints_when_stanza_csv_files_succeeds(checkpoint_opts: checkpoint.CheckpointOpts):

    file_pattern: str = "*.zip"
    source_folder: str = TAGGED_DATA_FOLDER

    stream: Iterable[checkpoint.CheckpointData] = tasks.load_checkpoints(source_folder, file_pattern, checkpoint_opts)

    cps: List[checkpoint.CheckpointData] = [cp for cp in stream]

    assert cps and len(cps) == 5
    assert all(isinstance(x, checkpoint.CheckpointData) for x in cps)

    source_names = {os.path.basename(x.source_name) for x in cps}
    assert source_names == {
        'prot-197879--18.zip',
        'prot-1960--fk--3.zip',
        'prot-202021--2.zip',
        'prot-198990--15.zip',
        'prot-1932--fk--1.zip',
    }

    document_indexes: List[corpus.DocumentIndex] = [x.document_index for x in cps]
    merged_index: corpus.DocumentIndex = pd.concat(document_indexes, ignore_index=False, sort=False)

    assert len(merged_index) == 15

    unique_document_names = set(merged_index.document_name.apply(lambda x: x.split('@')[0]).unique())
    assert unique_document_names == {
        'prot-197879--18',
        'prot-1960--fk--3',
        'prot-202021--2',
        'prot-198990--15',
        'prot-1932--fk--1',
    }


@pytest.mark.skip("Not implemented")
def test_load_checkpoints_with_predicate_filter(checkpoint_opts: checkpoint.CheckpointOpts):

    file_pattern: str = "*.zip"
    source_folder: str = TAGGED_DATA_FOLDER
    filenames_to_load = {
        'prot-1960--fk--3.zip',
        'prot-202021--2.zip',
        'prot-198990--15.zip',
    }

    def predicate_filter(path: str):
        nonlocal filenames_to_load
        return os.path.basename(path) in filenames_to_load

    stream: Iterable[checkpoint.CheckpointData] = tasks.load_checkpoints(
        source_folder, file_pattern, checkpoint_opts, predicate_filter
    )

    loaded_filenames: Set[str] = {os.path.basename(cp.source_name) for cp in stream}

    assert loaded_filenames == filenames_to_load


@pytest.mark.skip("Not implemented")
def test_to_tagged_frame_when_loading_checkpoints_succeeds(checkpoint_opts: checkpoint.CheckpointOpts):

    corpus_source: str = TAGGED_DATA_FOLDER
    reader_opts: corpus.TextReaderOpts = corpus.TextReaderOpts(filename_pattern="*.csv")

    pipeline: pp.CorpusPipeline = Mock(spec=pp.CorpusPipeline)

    task: interfaces.ITask = tasks.LoadTaggedFrame(
        corpus_source=corpus_source,
        reader_opts=reader_opts,
        checkpoint_opts=checkpoint_opts,
        pipeline=pipeline,
    )

    payloads: List[interfaces.DocumentPayload] = [p for p in task.outstream()]

    assert payloads is not None
    assert len(payloads) == 15

    payload = next((x for x in payloads if x.filename == 'prot-202021--2@1.csv'), None)
    assert payload is not None

    assert (
        ' '.join(payload.content.head(13).text)
        == 'Eders Majestäter ! Herr talman ! Ärade ledamöter av Sveriges riksdag ! Sverige'
    )

    assert (
        ' '.join(payload.content.head(13).lemma)
        == 'eder majestat ! herr talman ! ärad ledamot av sverige riksdag ! sverige'
    )

    assert ' '.join(payload.content.head(13).pos) == 'PM NN MID NN NN MID PC NN PP PM NN MID PM'


def test_load_tagged_frame_groups():

    source_folder: str = '/data/riksdagen_corpus_data/tagged-speech-corpus.numeric.feather'
    document_index: pd.DataFrame = corpus.DocumentIndexHelper.load(
        jj(source_folder, 'document_index.feather')
    ).document_index

    assert document_index is not None

    filenames = sorted(glob(jj(source_folder, '**/prot-*.feather'), recursive=True))

    assert len(filenames) == len(document_index.document_name.apply(lambda x: x.split('_')[0]).unique())

    n_tokens = 0
    for filename in tqdm.tqdm(filenames, total=len(filenames)):
        document_name: str = utility.strip_path_and_extension(filename)
        tagged_frame: pd.DataFrame = pd.read_feather(filename)
        n_tokens += len(tagged_frame)
