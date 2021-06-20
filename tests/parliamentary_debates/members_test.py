import pytest
from tqdm.auto import tqdm
from penelope.pipeline import CorpusPipeline
from penelope.pipeline.config import CorpusConfig
from westac.parliamentary_debates.members import GITHUB_DATA_URL, ParliamentaryMembers
from westac.parliamentary_debates.pipelines import to_tagged_frame_pipeline


def test_load_members():

    parliament_data = ParliamentaryMembers.load(GITHUB_DATA_URL)

    assert parliament_data is not None


@pytest.mark.long_running
def test_run_through_entire_corpus():

    corpus_config: CorpusConfig = CorpusConfig.load('./tests/test_data/parliamentary-debates.yml')

    _ = ParliamentaryMembers.load(GITHUB_DATA_URL)

    corpus_path = '/data/riksdagen_corpus_data/annotated'

    pipeline: CorpusPipeline = to_tagged_frame_pipeline(
        source_folder=corpus_path,
        corpus_config=corpus_config,
        checkpoint_filter=None,
        filename_filter=None,
        filename_pattern=None,
    )

    for _ in tqdm(pipeline.resolve(), total=11100):
        pass
