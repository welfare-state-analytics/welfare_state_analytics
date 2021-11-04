import pytest
from penelope.pipeline import CorpusPipeline
from penelope.pipeline.config import CorpusConfig
from tqdm.auto import tqdm
from westac.riksdagens_protokoll.members import GITHUB_DATA_URL, ParliamentaryData
from westac.riksdagens_protokoll.pipelines import to_tagged_frame_pipeline


@pytest.mark.skip("Not implemented")
def test_load_members():

    parliament_data = ParliamentaryData.load(GITHUB_DATA_URL)

    assert parliament_data is not None

    assert parliament_data.genders == [None, 'man', 'woman']


@pytest.mark.skip("long running")
@pytest.mark.long_running
@pytest.mark.parametrize(
    'corpus_path',
    [
        '/data/riksdagen_corpus_data/annotated',
    ],
)
def test_run_through_entire_corpus(corpus_path: str):

    corpus_config: CorpusConfig = CorpusConfig.load('./tests/test_data/riksdagens-protokoll.yml')

    _ = ParliamentaryData.load(GITHUB_DATA_URL)

    pipeline: CorpusPipeline = to_tagged_frame_pipeline(
        source_folder=corpus_path,
        corpus_config=corpus_config,
        checkpoint_filter=None,
        filename_filter=None,
        filename_pattern=None,
    )

    for _ in tqdm(pipeline.resolve(), total=11100):
        pass
