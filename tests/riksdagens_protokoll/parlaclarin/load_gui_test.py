from unittest.mock import MagicMock, patch

import pytest
from penelope.topic_modelling import ModelFolder

from notebooks.riksdagens_protokoll.topic_modeling.utility import RiksprotLoadGUI, TopicModelContainer, load_metadata
from notebooks.riksdagens_protokoll.topic_modeling.utility.metadata import (
    _probe_corpus_version,
    _probe_document_index,
    _probe_metadata_filename,
    _probe_paths,
    _probe_vrt_folder,
)


def test_probe_filenames_existing_file(tmp_path):
    file = tmp_path / 'test_file.txt'
    file.touch()

    candidates = ['non_existent_file.txt', str(file), 'another_non_existent_file.txt']
    result = _probe_paths(candidates)
    assert result == str(file)


def test_probe_filenames_no_existing_file():
    candidates = ['non_existent_file.txt', 'another_non_existent_file.txt']
    with pytest.raises(FileNotFoundError):
        _probe_paths(candidates)


def test_probe_filenames_empty_list():
    candidates = []
    with pytest.raises(FileNotFoundError):
        _probe_paths(candidates)


def test_existing_version_file(tmp_path):
    expected_version = 'v1.2.3'
    (tmp_path / "corpus_version").write_text(expected_version)

    corpus_version: str = _probe_corpus_version(tmp_path)

    assert corpus_version == expected_version


def test_existing_config_file(tmp_path):
    expected_version = 'v1.2.3'
    (tmp_path / 'corpus.yml').touch()
    with patch('penelope.pipeline.config.CorpusConfig.load') as mock_load:
        mock_load.return_value = MagicMock(corpus_version=expected_version)
        corpus_version: str = _probe_corpus_version(str(tmp_path))

    assert corpus_version == expected_version


def test_folder_name_with_version():
    folder = '/path/v1.2.3/xyz_v1.2.3_folder'
    expected_version = 'v1.2.3'

    corpus_version = _probe_corpus_version(folder)

    assert corpus_version == expected_version


def test_folder_name_with_version_in_config(tmp_path):
    folder = tmp_path
    (tmp_path / 'corpus.yml').touch()

    expected_version = 'v1.2.3'

    with patch('penelope.pipeline.config.CorpusConfig.load') as mock_load:
        mock_load.return_value = MagicMock(corpus_version=expected_version)
        corpus_version = _probe_corpus_version(str(folder))

    assert corpus_version == expected_version

    with patch('penelope.pipeline.config.CorpusConfig.load') as mock_load:
        mock_load.return_value = MagicMock(
            corpus_version=None, pipeline_payload=MagicMock(source=f'xx_{expected_version}_xyz')
        )
        corpus_version = _probe_corpus_version(str(folder))

    assert corpus_version == expected_version


def test_folder_name_with_version_in_folder_name():
    expected_version = 'v1.2.3'

    assert expected_version == _probe_corpus_version(f'{expected_version}/apa')
    assert expected_version == _probe_corpus_version(f'apa/xyz_{expected_version}')
    assert expected_version == _probe_corpus_version(f'apa/xyz_{expected_version}_xyz')


def test_invalid_folder():
    folder = '/path/to/nonexistent_folder'

    corpus_version = _probe_corpus_version(folder)

    assert corpus_version is None


def test_load_metadata_invalid_folder():
    with pytest.raises(FileNotFoundError):
        load_metadata('invalid_root_folder', 'invalid_folder')


@patch(
    'notebooks.riksdagens_protokoll.topic_modeling.utility.metadata._probe_corpus_version',
    MagicMock(return_value='v1.2.3'),
)
@patch(
    'notebooks.riksdagens_protokoll.topic_modeling.utility.metadata._probe_metadata_filename',
    MagicMock(return_value='metadata_filename'),
)
@patch(
    'notebooks.riksdagens_protokoll.topic_modeling.utility.metadata._probe_document_index',
    MagicMock(return_value='document_index'),
)
@patch(
    'notebooks.riksdagens_protokoll.topic_modeling.utility.metadata._probe_vrt_folder',
    MagicMock(return_value='vrt_folder'),
)
@patch('westac.riksprot.parlaclarin.codecs.PersonCodecs.load', MagicMock(return_value='person_codecs'))
@patch('westac.riksprot.parlaclarin.speech_text.SpeechTextRepository', MagicMock(return_value='speech_repository'))
def test_load_metadata_valid_folder():
    result = load_metadata('root_folder', 'folder')
    assert result == {
        'corpus_version': 'v1.2.3',
        'data_folder': 'root_folder',
        'person_codecs': 'person_codecs',
        'speech_repository': 'speech_repository',
        'speech_index': 'document_index',
    }


def test_probe_vrt_folder_existing_folder(tmp_path):
    # Create a directory in the temporary path
    corpus_version = 'v1.2.3'
    folder = tmp_path / corpus_version / 'tagged_frames'
    folder.mkdir(parents=True)

    result = _probe_vrt_folder(str(tmp_path), corpus_version)
    assert result == str(folder)


def test_probe_vrt_folder_no_existing_folder(tmp_path):
    corpus_version = 'v1'

    with pytest.raises(FileNotFoundError):
        _probe_vrt_folder(str(tmp_path), corpus_version)


def test_probe_vrt_folder_empty_list():
    with pytest.raises(FileNotFoundError):
        _probe_vrt_folder('', '')


def test_probe_document_index_no_existing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        _probe_document_index(str(tmp_path), 'v1')


def test_probe_metadata_filename_existing_file(tmp_path):
    # Create a file in the temporary path
    corpus_version = 'v1'
    filename = tmp_path / 'metadata' / f'riksprot_metadata.{corpus_version}.db'
    filename.parent.mkdir(parents=True)
    filename.touch()

    result = _probe_metadata_filename(str(tmp_path), corpus_version)
    assert result == str(filename)


def test_probe_metadata_filename_no_existing_file(tmp_path):
    corpus_version = 'v1'

    with pytest.raises(FileNotFoundError):
        _probe_metadata_filename(str(tmp_path), corpus_version)


def test_riksprot_load_gui_load(mocker):
    mocker.patch('penelope.notebook.topic_modelling.LoadGUI.load', MagicMock())
    mocker.patch('penelope.notebook.topic_modelling.LoadGUI.model_info', return_value=ModelFolder('folder', 'name', {}))
    mocker.patch(
        'notebooks.riksdagens_protokoll.topic_modeling.utility.metadata.load_metadata',
        MagicMock(return_value={'corpus_version': 'v1', 'data_folder': 'data_folder'}),
    )

    state: TopicModelContainer = TopicModelContainer()
    gui: RiksprotLoadGUI = RiksprotLoadGUI('data_folder', state)
    gui.load()

    assert gui.state.get('corpus_version') == 'v1'
    assert gui.state.get('data_folder') == 'data_folder'


def test_riksprot_load_gui_version():
    state: TopicModelContainer = TopicModelContainer()
    gui: RiksprotLoadGUI = RiksprotLoadGUI(data_folder='data_folder', state=state)

    state.store(corpus_version='v1.2.3')
    assert gui.corpus_version == 'v1.2.3'
